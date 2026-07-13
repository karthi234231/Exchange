"""
features/engine.py — FeatureComputer ZeroMQ SUB/PUB process. [P2I-2.1]

Middle tier of the 3-process ZMQ pipeline:

    [DataIngestor PUB:5555] → [FeatureComputer SUB:5555 / PUB:5556] → [MLInference SUB:5556]

Responsibilities:
  1. Subscribe to ZMQ PUB:5555 (raw ticks from websocket_feed.py).
  2. Maintain a RingBuffer per symbol (in-process memory, no QuestDB reads on hot path).
  3. Maintain a VolumeBarAggregator per symbol (from data/candle_store.py).
  4. On each new completed volume bar:
       a. Extract last N ticks from RingBuffer.
       b. Run all 23 features (TechnicalFeatures, StatisticalFeatures, CustomFeatures).
       c. Validate no NaNs in feature vector; skip bar if insufficient history.
       d. Serialize feature vector as np.float32 bytes.
       e. Publish to ZMQ PUB:5556 with symbol as topic.

ZMQ socket settings [SURV-7]:
  - sub.setsockopt(zmq.RCVHWM, 1000)
  - sub.setsockopt(zmq.RCVTIMEO, 5000)   # 5 s timeout — detect dead publisher
  - pub.setsockopt(zmq.SNDHWM, 1000)
  - pub.setsockopt(zmq.LINGER, 0)

SequenceGuard integration [SURV-7]:
  - Every incoming tick carries a ``seq`` field.
  - Per-symbol expected sequence number is tracked.
  - If gap > 3 consecutive: halt feature computation for that symbol + send alert.
  - Every gap is logged via structlog.

Feature schema hash:
  - sha256 of the feature_names tuple is computed once at module import.
  - Hash is attached as metadata to every published feature vector.
  - Downstream inference engine validates this hash matches FrozenModelSpec.

Stage guard:
  - features/microstructure.py (Hawkes/TFI) is Stage 3 only.
  - This file must NOT import microstructure.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import msgpack  # type: ignore[import]
    _HAS_MSGPACK = True
except ImportError:
    msgpack = None  # type: ignore[assignment]
    _HAS_MSGPACK = False
import numpy as np
import structlog
import zmq

from algo_trading.data.ring_buffer import RingBuffer

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature names — defines the feature vector layout.
# 25 names for 23 conceptual features because tod uses 2 values (sin + cos).
# Any change here requires a new FrozenModelSpec version [SURV-5].
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    "rsi_14",
    "macd_hist",
    "bb_pct_b",
    "bb_width",
    "atr_14",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "obv_normalized",
    "supertrend_dir",
    "ema_cross_ratio",
    "ichimoku_position",
    "vwap_dev_bps",
    "kalman_price",
    "hmm_regime",
    "garch_vol",
    "obi",
    "price_vol_divergence",
    "ha_direction",
    "pivot_dist_bps",
    "autocorr_lag1",
    "relative_strength",
    "tod_sin",
    "tod_cos",
    "spread_percentile",
]

N_FEATURES: int = len(FEATURE_NAMES)  # 25

# ---------------------------------------------------------------------------
# ZMQ configuration [SURV-7]
# Read from settings with fallbacks to the hardcoded defaults.
# ---------------------------------------------------------------------------
try:
    from algo_trading.config import settings as _settings  # type: ignore[import]
except Exception:
    _settings = None  # type: ignore[assignment]

_ZMQ_RCVHWM: int = getattr(_settings, "zmq_rcvhwm", 1_000)
_ZMQ_RCVTIMEO_MS: int = 5_000   # 5 s — detect dead publisher (not configurable via settings)
_ZMQ_SNDHWM: int = getattr(_settings, "zmq_sndhwm", 1_000)
_ZMQ_LINGER: int = 0

# SequenceGuard: halt feature computation after this many consecutive drops
_MAX_ALLOWED_GAP: int = 3

# Minimum number of ticks in the ring buffer before features are attempted.
# Read from settings with fallback to 500.
_MIN_TICKS_FOR_FEATURES: int = getattr(_settings, "min_ticks_for_features", 500)

# Ring buffer capacity per symbol.
# Read from settings with fallback to 10,000.
_RING_BUFFER_CAPACITY: int = getattr(_settings, "ring_buffer_capacity", 10_000)


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def compute_feature_schema_hash(feature_names: list[str]) -> str:
    """Compute a stable sha256 fingerprint for the ordered feature name list.

    Returns the first 8 hex characters of the SHA-256 digest — identical to
    ``FrozenModelSpec.compute_feature_hash()`` so that the feature_hash field
    published in ZMQ messages can be directly compared by the inference engine
    without truncation [SURV-5].

    Args:
        feature_names: Ordered list of feature name strings.

    Returns:
        First 8 hex characters of the SHA-256 digest (same format as
        ``FrozenModelSpec.compute_feature_hash``).

    Example:
        >>> h = compute_feature_schema_hash(FEATURE_NAMES)
        >>> len(h)
        8
    """
    canonical = str(tuple(feature_names)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:8]


# Pre-computed at module import. Downstream MLInference validates this value.
FEATURE_SCHEMA_HASH: str = compute_feature_schema_hash(FEATURE_NAMES)

try:
    from algo_trading.core.calendar import IST as _IST  # type: ignore[import]
except Exception:
    from zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore[assignment]
    _IST = _ZoneInfo("Asia/Kolkata")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Feature schema versioning (SURV-5)
# ---------------------------------------------------------------------------

class FeatureSchemaMismatchError(RuntimeError):
    """Raised when the saved feature schema hash does not match the current hash.

    Attributes:
        saved_hash: The hash stored in the schema JSON artifact.
        current_hash: The hash freshly computed from FEATURE_NAMES at import time.
    """

    def __init__(self, saved: str, current: str) -> None:
        super().__init__(
            f"Feature schema changed: saved={saved}, current={current}. Retrain model."
        )
        self.saved_hash: str = saved
        self.current_hash: str = current


def write_feature_schema(output_path: Path) -> str:
    """Write the current feature schema to a JSON artifact file.

    The artifact records the ordered feature names, the feature count, the
    schema hash, and the timestamp at which it was written.  Downstream
    model loading code should call :func:`load_and_validate_feature_schema`
    with this file to detect silent schema drift (SURV-5).

    Args:
        output_path: Destination file path.  Parent directories are created
            if they do not exist.

    Returns:
        The 8-character schema hash that was written into the artifact.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema: dict = {
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "schema_hash": FEATURE_SCHEMA_HASH,
        "written_at": datetime.now(tz=_IST).isoformat(),
    }
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    log.info(
        "feature_schema.written",
        path=str(output_path),
        n_features=N_FEATURES,
        schema_hash=FEATURE_SCHEMA_HASH,
    )
    return FEATURE_SCHEMA_HASH


def load_and_validate_feature_schema(schema_path: Path) -> None:
    """Load a saved feature schema artifact and validate it against the current schema.

    Reads the JSON written by :func:`write_feature_schema` and recomputes the
    schema hash from the current :data:`FEATURE_NAMES`.  If the hashes differ
    the model was trained on a different feature set and inference must be
    blocked (SURV-5).

    Args:
        schema_path: Path to the JSON artifact previously written by
            :func:`write_feature_schema`.

    Raises:
        FileNotFoundError: If ``schema_path`` does not exist.
        FeatureSchemaMismatchError: If the saved hash does not match the
            current hash computed from :data:`FEATURE_NAMES`.
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema artifact not found: {schema_path}. "
            "Run write_feature_schema() after training to persist the schema."
        )

    data: dict = json.loads(schema_path.read_text(encoding="utf-8"))
    saved_hash: str = data.get("schema_hash", "")
    current_hash: str = FEATURE_SCHEMA_HASH

    if saved_hash != current_hash:
        log.error(
            "feature_schema.mismatch",
            schema_path=str(schema_path),
            saved_hash=saved_hash,
            current_hash=current_hash,
            saved_features=data.get("feature_names"),
            current_features=FEATURE_NAMES,
        )
        raise FeatureSchemaMismatchError(saved=saved_hash, current=current_hash)

    log.info(
        "feature_schema.validated",
        schema_path=str(schema_path),
        schema_hash=current_hash,
        n_features=N_FEATURES,
        written_at=data.get("written_at"),
    )


# ---------------------------------------------------------------------------
# VolumeBarAggregator — defined here as a lightweight stub that mirrors the
# spec from candle_store.py [P2I-2.3].  When candle_store.py gains its own
# VolumeBarAggregator, swap the import:
#   from algo_trading.data.candle_store import VolumeBarAggregator
# ---------------------------------------------------------------------------

class _VolumeBar:
    """Minimal OHLCV + metadata for a completed volume bar."""

    __slots__ = (
        "symbol", "open", "high", "low", "close",
        "volume", "turnover", "tick_count",
        "start_ts", "end_ts", "bar_type", "vwap",
    )

    def __init__(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        turnover: float,
        tick_count: int,
        start_ts: int,
        end_ts: int,
        bar_type: str,
    ) -> None:
        """Initialise a completed volume bar.

        Args:
            symbol: Instrument identifier.
            open_: Opening price.
            high: Highest price in bar.
            low: Lowest price in bar.
            close: Closing price.
            volume: Total volume traded.
            turnover: Total turnover (price × volume sum).
            tick_count: Number of ticks that contributed.
            start_ts: Bar open timestamp in nanoseconds (UTC epoch).
            end_ts: Bar close timestamp in nanoseconds (UTC epoch).
            bar_type: One of "volume", "tick", "dollar".
        """
        self.symbol = symbol
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.turnover = turnover
        self.tick_count = tick_count
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.bar_type = bar_type
        self.vwap = turnover / volume if volume > 0 else close


class VolumeBarAggregator:
    """Accumulates ticks into information-uniform volume bars.

    Fires a completed bar when accumulated volume, tick count, or turnover
    exceeds the configured threshold. The ``on_bar`` callback is invoked
    synchronously from :meth:`push_tick`.

    Attributes:
        symbol: Instrument identifier.
        volume_threshold: Shares required to close a volume bar.
        tick_threshold: Trade count required to close a tick bar.
        dollar_threshold: INR turnover required to close a dollar bar.
        on_bar: Optional callback invoked with a completed :class:`_VolumeBar`.
    """

    def __init__(
        self,
        symbol: str,
        volume_threshold: int = 50_000,
        tick_threshold: int = 500,
        dollar_threshold: float = 2_00_00_000.0,
        on_bar: Optional[Any] = None,
    ) -> None:
        """Initialise the aggregator with threshold parameters.

        Args:
            symbol: Instrument identifier.
            volume_threshold: Volume (shares) required to close a bar.
            tick_threshold: Tick count required to close a bar.
            dollar_threshold: Turnover (INR) required to close a bar.
            on_bar: Callable invoked with the completed _VolumeBar. May be None.
        """
        self.symbol = symbol
        self.volume_threshold = volume_threshold
        self.tick_threshold = tick_threshold
        self.dollar_threshold = dollar_threshold
        self.on_bar = on_bar
        self._bar_seq: int = 0
        self._reset()

    def _reset(self) -> None:
        """Reset accumulator state for the next bar."""
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = float("inf")
        self._close: float = 0.0
        self._volume: int = 0
        self._turnover: float = 0.0
        self._ticks: int = 0
        self._start_ts: int = 0
        self._initialized: bool = False

    def push_tick(
        self,
        ltp: float,
        volume: int,
        timestamp_ns: int,
    ) -> Optional[_VolumeBar]:
        """Feed one tick. Returns a completed bar if a threshold is crossed.

        Args:
            ltp: Last traded price.
            volume: Incremental volume for this tick.
            timestamp_ns: Tick timestamp in nanoseconds (UTC epoch).

        Returns:
            A completed :class:`_VolumeBar` if a threshold was crossed,
            otherwise None.
        """
        if not self._initialized:
            self._open = ltp
            self._high = ltp
            self._low = ltp
            self._start_ts = timestamp_ns
            self._initialized = True

        self._high = max(self._high, ltp)
        self._low = min(self._low, ltp)
        self._close = ltp
        self._volume += volume
        self._turnover += ltp * volume
        self._ticks += 1

        threshold_hit = (
            self._volume >= self.volume_threshold
            or self._ticks >= self.tick_threshold
            or self._turnover >= self.dollar_threshold
        )

        if not threshold_hit:
            return None

        bar_type = (
            "volume" if self._volume >= self.volume_threshold
            else "tick" if self._ticks >= self.tick_threshold
            else "dollar"
        )
        bar = _VolumeBar(
            symbol=self.symbol,
            open_=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            turnover=self._turnover,
            tick_count=self._ticks,
            start_ts=self._start_ts,
            end_ts=timestamp_ns,
            bar_type=bar_type,
        )
        self._bar_seq += 1
        self._reset()

        if self.on_bar is not None:
            self.on_bar(bar)
        return bar

    def calibrate_from_history(
        self,
        daily_volumes: list[int],
        target_bars: int = 200,
    ) -> None:
        """Calibrate volume_threshold from historical daily volumes.

        Sets ``volume_threshold = mean(daily_volumes) / target_bars``.
        Call once after market open using recent ADV data [P2I-2.3].

        Args:
            daily_volumes: List of total daily volumes (shares) for recent trading days.
            target_bars: Desired number of bars per session. Defaults to 200.
        """
        if not daily_volumes:
            log.warning("volume_bar_aggregator.calibrate.empty_history", symbol=self.symbol)
            return
        self.volume_threshold = max(1, int(np.mean(daily_volumes)) // target_bars)
        log.info(
            "volume_bar_aggregator.calibrated",
            symbol=self.symbol,
            volume_threshold=self.volume_threshold,
            target_bars=target_bars,
        )

    @property
    def bar_seq(self) -> int:
        """Monotonically increasing count of completed bars for this aggregator.

        Returns:
            Integer count of bars emitted since construction.
        """
        return self._bar_seq


# ---------------------------------------------------------------------------
# Feature stubs — TechnicalFeatures / StatisticalFeatures / CustomFeatures
# ---------------------------------------------------------------------------
# These are forward-declared stubs.  When the real feature modules are built
# (features/technical.py, features/statistical.py, features/custom.py),
# replace the try/except import block below with direct imports.
# The engine itself only cares about the contract:
#   feature_module.compute(ticks_df, bar) -> dict[str, float]
# ---------------------------------------------------------------------------

try:
    from algo_trading.features.technical import TechnicalFeatures  # type: ignore[import]
    _HAS_TECHNICAL = True
except ImportError:
    _HAS_TECHNICAL = False
    log.debug("features.technical not yet available — using placeholder zeros")

try:
    from algo_trading.features.statistical import StatisticalFeatures  # type: ignore[import]
    _HAS_STATISTICAL = True
except ImportError:
    _HAS_STATISTICAL = False
    log.debug("features.statistical not yet available — using placeholder zeros")

try:
    from algo_trading.features.custom import CustomFeatures  # type: ignore[import]
    _HAS_CUSTOM = True
except ImportError:
    _HAS_CUSTOM = False
    log.debug("features.custom not yet available — using placeholder zeros")

# ---------------------------------------------------------------------------
# SequenceGuard [SURV-7]
# ---------------------------------------------------------------------------

class SequenceGuard:
    """Detects ZMQ message drops via monotonic per-symbol sequence numbers.

    Tracks the expected next sequence number for every symbol.  If a gap
    greater than ``MAX_ALLOWED_GAP`` consecutive drops is detected, the
    symbol is added to the halt set and a Telegram alert is dispatched.

    Attributes:
        MAX_ALLOWED_GAP: Class-level constant; gaps larger than this halt
            feature computation for the affected symbol.
    """

    MAX_ALLOWED_GAP: int = _MAX_ALLOWED_GAP

    def __init__(self) -> None:
        """Initialise with empty tracking state."""
        self._expected: dict[str, int] = {}   # symbol → next expected seq
        self._halted: set[str] = set()         # symbols with unrecoverable gaps

    def check(self, symbol: str, seq: int) -> bool:
        """Validate an incoming sequence number and update tracking state.

        Logs every gap to structlog.  If the gap exceeds MAX_ALLOWED_GAP the
        symbol is added to the halt set and the method returns False.

        Args:
            symbol: Instrument identifier.
            seq: Sequence number from the incoming tick.

        Returns:
            True if the tick should be processed, False if it should be dropped
            (old/duplicate message or halt condition triggered).
        """
        expected = self._expected.get(symbol, seq)  # bootstrap on first tick

        if seq < expected:
            log.warning(
                "sequence_guard.old_message",
                symbol=symbol,
                received_seq=seq,
                expected_seq=expected,
            )
            return False  # Stale / duplicate — drop silently

        if seq > expected:
            gap = seq - expected
            log.error(
                "sequence_guard.gap_detected",
                symbol=symbol,
                gap=gap,
                expected_seq=expected,
                received_seq=seq,
            )
            if gap > self.MAX_ALLOWED_GAP:
                self._halted.add(symbol)
                log.critical(
                    "sequence_guard.symbol_halted",
                    symbol=symbol,
                    gap=gap,
                )
                # Fire-and-forget alert via alerts module (best effort — never block).
                try:
                    from algo_trading.alerts.telegram_bot import send_alert  # type: ignore[import]
                    send_alert(
                        f"[FEATURE ENGINE] ZMQ data gap: {gap} ticks dropped on {symbol}. "
                        "Feature computation halted for this symbol.",
                        level="CRITICAL",
                    )
                except Exception:
                    pass
                return False

        self._expected[symbol] = seq + 1
        return True

    def is_halted(self, symbol: str) -> bool:
        """Return True if feature computation is halted for this symbol.

        Args:
            symbol: Instrument identifier.

        Returns:
            True if the symbol has been halted due to excessive sequence gaps.
        """
        return symbol in self._halted

    def reset_symbol(self, symbol: str) -> None:
        """Clear halt state and sequence tracking for a symbol.

        Call this after a confirmed reconnect/resync to resume feature
        computation for the symbol.

        Args:
            symbol: Instrument identifier to reset.
        """
        self._halted.discard(symbol)
        self._expected.pop(symbol, None)
        log.info("sequence_guard.symbol_reset", symbol=symbol)


# ---------------------------------------------------------------------------
# FeatureComputer
# ---------------------------------------------------------------------------

class FeatureComputer:
    """ZMQ-based feature computation process.  Runs as a separate process.

    Subscribes to a tick PUB socket, accumulates ticks into ring buffers and
    volume bar aggregators, and publishes feature vectors to downstream
    inference processes.

    The published ZMQ multipart message format is:
        frame 0: symbol bytes  (e.g. b"RELIANCE")
        frame 1: msgpack-packed dict:
            {
                "seq": int,           # per-symbol bar sequence number
                "timestamp_ns": int,  # bar close timestamp (UTC nanoseconds)
                "features": bytes,    # float32 array, len = N_FEATURES
                "feature_hash": str,  # sha256 of FEATURE_NAMES tuple
                "bar_type": str,      # "volume" | "tick" | "dollar"
                "bar_seq": int,       # monotonic bar counter for this symbol
            }

    Attributes:
        symbols: Ordered list of instrument identifiers to subscribe to.
        sub_port: ZMQ PUB port to subscribe to (upstream DataIngestor).
        pub_port: ZMQ PUB port to bind and publish features on (downstream).
        min_bars_for_features: Minimum completed bars required before features
            are computed for a symbol (warm-up guard).
    """

    def __init__(
        self,
        symbols: list[str],
        sub_port: int = 5555,
        pub_port: int = 5556,
        min_bars_for_features: int = 60,
    ) -> None:
        """Initialise the FeatureComputer.

        Args:
            symbols: List of NSE instrument identifiers to subscribe to.
                     At least one symbol must be provided.
            sub_port: TCP port of the upstream ZMQ PUB socket (DataIngestor).
                      Defaults to 5555.
            pub_port: TCP port to bind this process's ZMQ PUB socket on.
                      Defaults to 5556.
            min_bars_for_features: Minimum completed volume bars required
                      before feature computation begins for a symbol.
                      Prevents feature computation during warm-up.
                      Defaults to 60.

        Raises:
            ValueError: If ``symbols`` is empty.
        """
        if not symbols:
            raise ValueError("FeatureComputer requires at least one symbol")
        self.symbols: list[str] = list(symbols)
        self.sub_port: int = sub_port
        self.pub_port: int = pub_port
        self.min_bars_for_features: int = min_bars_for_features

        # Per-symbol state
        self._ring_buffers: dict[str, RingBuffer] = {
            sym: RingBuffer(capacity=_RING_BUFFER_CAPACITY) for sym in symbols
        }
        self._bar_aggregators: dict[str, VolumeBarAggregator] = {}
        self._bar_counts: dict[str, int] = {sym: 0 for sym in symbols}
        self._pub_seq: dict[str, int] = {sym: 0 for sym in symbols}
        self._last_bar: dict[str, Optional[_VolumeBar]] = {sym: None for sym in symbols}
        self._bar_history: dict[str, deque] = {sym: deque(maxlen=200) for sym in symbols}

        # Sequence guard for ZMQ drop detection [SURV-7]
        self._seq_guard: SequenceGuard = SequenceGuard()

        # ZMQ context and sockets (created in start())
        self._ctx: Optional[zmq.Context] = None  # type: ignore[type-arg]
        self._sub: Optional[zmq.Socket] = None   # type: ignore[type-arg]
        self._pub: Optional[zmq.Socket] = None   # type: ignore[type-arg]

        # Shutdown flag (set by SIGTERM / shutdown())
        self._running: bool = False

        # Optional feature computation helpers (lazy import pattern)
        self._tech: Optional[Any] = TechnicalFeatures() if _HAS_TECHNICAL else None
        self._stat: Optional[Any] = StatisticalFeatures() if _HAS_STATISTICAL else None
        self._cust: Optional[Any] = CustomFeatures() if _HAS_CUSTOM else None

        # Load pre-fitted HMM / GARCH from training artefacts so that
        # statistical features are non-NaN from the very first bar.
        if self._stat is not None:
            _stat_model_path = Path(
                getattr(_settings, "model_output_dir", "models/trained")
            ) / "statistical_models.pkl"
            if _stat_model_path.exists():
                try:
                    import pickle  # noqa: PLC0415
                    with _stat_model_path.open("rb") as _fh:
                        _loaded_stat = pickle.load(_fh)  # noqa: S301
                    self._stat = _loaded_stat
                    log.info(
                        "feature_computer.statistical_models_loaded",
                        path=str(_stat_model_path),
                        hmm_fitted=getattr(getattr(_loaded_stat, "hmm", None), "_is_fitted", False),
                    )
                except Exception as _exc:  # noqa: BLE001
                    log.warning(
                        "feature_computer.statistical_models_load_failed",
                        path=str(_stat_model_path),
                        error=str(_exc),
                        note="HMM/GARCH features will be NaN until models are trained",
                    )
            else:
                log.warning(
                    "feature_computer.statistical_models_missing",
                    path=str(_stat_model_path),
                    note="Run train_models.py first to fit HMM/GARCH. Features will be NaN.",
                )

        log.info(
            "feature_computer.init",
            symbols=symbols,
            sub_port=sub_port,
            pub_port=pub_port,
            min_bars_for_features=min_bars_for_features,
            n_features=N_FEATURES,
            feature_hash=FEATURE_SCHEMA_HASH,
        )

    # ------------------------------------------------------------------
    # Socket setup helpers
    # ------------------------------------------------------------------

    def _setup_sockets(self) -> None:
        """Create and configure ZMQ sockets.  Called from :meth:`start`."""
        self._ctx = zmq.Context()

        # Subscriber — connects to DataIngestor PUB:5555
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, _ZMQ_RCVHWM)
        self._sub.setsockopt(zmq.RCVTIMEO, _ZMQ_RCVTIMEO_MS)
        self._sub.setsockopt(zmq.LINGER, _ZMQ_LINGER)
        # Subscribe to each symbol topic; also subscribe to empty string
        # for backwards compat with non-multipart publishers.
        for sym in self.symbols:
            self._sub.setsockopt(zmq.SUBSCRIBE, sym.encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")  # catch-all for single-frame pubs
        self._sub.connect(f"tcp://127.0.0.1:{self.sub_port}")

        # Publisher — binds and broadcasts feature vectors to MLInference PUB:5556
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, _ZMQ_SNDHWM)
        self._pub.setsockopt(zmq.LINGER, _ZMQ_LINGER)
        self._pub.bind(f"tcp://127.0.0.1:{self.pub_port}")

        log.info(
            "feature_computer.sockets_ready",
            sub_port=self.sub_port,
            pub_port=self.pub_port,
        )

    def _setup_bar_aggregators(self) -> None:
        """Create one VolumeBarAggregator per symbol with default thresholds."""
        for sym in self.symbols:
            # Capture sym in closure via default-argument binding
            def _make_callback(s: str) -> Any:
                def _on_bar(bar: _VolumeBar) -> None:
                    self._on_bar_complete(bar)
                return _on_bar

            self._bar_aggregators[sym] = VolumeBarAggregator(
                symbol=sym,
                on_bar=_make_callback(sym),
            )
        log.debug("feature_computer.bar_aggregators_ready", count=len(self._bar_aggregators))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Main event loop.  Blocks until shutdown signal received.

        Sets up ZMQ sockets, registers SIGTERM handler, then enters a
        tight receive loop.  On RCVTIMEO expiry the loop continues to
        allow signal checking.  On SIGTERM / :meth:`shutdown` call the
        loop exits cleanly and sockets are closed.
        """
        self._setup_sockets()
        self._setup_bar_aggregators()
        self._running = True

        # Reset SequenceGuard for all symbols at session open so that any
        # halt state from a previous run or reconnect does not persist [SURV-7].
        for sym in self.symbols:
            self._seq_guard.reset_symbol(sym)
        log.info("feature_computer.sequence_guard_reset", symbols=self.symbols)

        # Allow clean shutdown via SIGTERM (systemd / multiprocessing)
        def _sig_handler(signum: int, frame: Any) -> None:
            log.info("feature_computer.shutdown_signal", signum=signum)
            self._running = False

        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)

        log.info(
            "feature_computer.started",
            pid=os.getpid(),
            symbols=self.symbols,
        )

        while self._running:
            try:
                # recv_multipart blocks up to RCVTIMEO ms, then raises zmq.Again
                parts = self._sub.recv_multipart()  # type: ignore[union-attr]
            except zmq.Again:
                # Timeout — no tick available; loop back and check _running
                continue
            except zmq.ZMQError as exc:
                if self._running:
                    log.error("feature_computer.recv_error", error=str(exc))
                break

            try:
                self._handle_raw_message(parts)
            except Exception as exc:
                log.error("feature_computer.dispatch_error", error=str(exc), exc_info=True)

        self._teardown()

    def shutdown(self) -> None:
        """Request a graceful shutdown of the event loop.

        Safe to call from any thread.  The event loop will exit on the
        next timeout cycle (≤ RCVTIMEO ms after this call).
        """
        self._running = False
        log.info("feature_computer.shutdown_requested")

    # ------------------------------------------------------------------
    # Internal message handling
    # ------------------------------------------------------------------

    def _handle_raw_message(self, parts: list[bytes]) -> None:
        """Decode a raw ZMQ multipart message and dispatch to _process_tick.

        Supports both:
          - 2-frame format: [symbol_bytes, msgpack_payload]
          - 1-frame format: [msgpack_payload] (symbol inside the dict)

        Args:
            parts: Raw frames from recv_multipart().
        """
        if len(parts) == 2:
            symbol_bytes, payload_bytes = parts
            try:
                tick_data: dict = msgpack.unpackb(payload_bytes, raw=False)
            except Exception as exc:
                log.warning("feature_computer.decode_error", error=str(exc))
                return
            # Prefer the explicit topic frame over any "symbol" key in payload
            symbol: str = symbol_bytes.decode("utf-8", errors="replace").strip("\x00")
            if not symbol:
                symbol = tick_data.get("symbol", "")
        elif len(parts) == 1:
            try:
                tick_data = msgpack.unpackb(parts[0], raw=False)
            except Exception as exc:
                log.warning("feature_computer.decode_error", error=str(exc))
                return
            symbol = tick_data.get("symbol", "")
        else:
            log.warning("feature_computer.unexpected_frame_count", count=len(parts))
            return

        if symbol not in self._ring_buffers:
            # Unknown symbol — could be a subscription filter race; ignore
            return

        self._process_tick(symbol, tick_data)

    def _process_tick(self, symbol: str, tick_data: dict) -> None:
        """Push tick to ring buffer and volume bar aggregator.

        Validates the sequence number via :class:`SequenceGuard` before
        accepting the tick.  Halted symbols are silently skipped.

        Args:
            symbol: Instrument identifier.
            tick_data: Decoded tick dict with keys:
                ``seq``          — monotonic sequence number (required),
                ``ts_ns``        — timestamp in nanoseconds (required),
                ``ltp``          — last traded price (required),
                ``volume``       — tick volume delta (required),
                ``bid``          — best bid price (optional, default 0),
                ``ask``          — best ask price (optional, default 0).
        """
        seq: int = int(tick_data.get("seq", 0))

        # Sequence integrity check [SURV-7]
        if not self._seq_guard.check(symbol, seq):
            return  # Gap > MAX_ALLOWED_GAP or old message — drop

        if self._seq_guard.is_halted(symbol):
            return

        ltp: float = float(tick_data.get("ltp", 0.0))
        volume: int = int(tick_data.get("volume", 0))
        timestamp_ns: int = int(tick_data.get("ts_ns", time.time_ns()))
        bid: float = float(tick_data.get("bid", 0.0))
        ask: float = float(tick_data.get("ask", 0.0))

        # Push to ring buffer (in-process, no QuestDB reads on hot path)
        self._ring_buffers[symbol].push(
            ltp=ltp,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp_ns=timestamp_ns,
        )

        # Push to volume bar aggregator (fires _on_bar_complete when bar closes)
        # Use incremental volume: the aggregator tracks cumulative per-bar volume.
        agg = self._bar_aggregators[symbol]
        agg.push_tick(ltp=ltp, volume=max(volume, 0), timestamp_ns=timestamp_ns)

    def _on_bar_complete(self, bar: _VolumeBar) -> None:
        """Called by VolumeBarAggregator when a bar closes.

        Triggers feature computation on the most recent ring buffer ticks and,
        if features are valid (no NaNs), publishes to the downstream PUB socket.

        Args:
            bar: Completed volume bar with OHLCV + metadata.
        """
        symbol: str = bar.symbol
        self._bar_counts[symbol] += 1
        self._last_bar[symbol] = bar
        self._bar_history[symbol].append(bar)

        if self._bar_counts[symbol] < self.min_bars_for_features:
            log.debug(
                "feature_computer.warmup",
                symbol=symbol,
                bars_seen=self._bar_counts[symbol],
                min_bars=self.min_bars_for_features,
            )
            return

        features = self._compute_features(symbol, bar)
        if features is None:
            return  # Insufficient history or computation error — skip bar

        agg = self._bar_aggregators[symbol]
        self._publish_features(
            symbol=symbol,
            features=features,
            timestamp_ns=bar.end_ts,
            bar_seq=agg.bar_seq,
        )

    def _compute_features(
        self,
        symbol: str,
        bar: _VolumeBar,
    ) -> Optional[np.ndarray]:
        """Compute the 25-element float32 feature vector for a completed bar.

        Extracts the last ``_MIN_TICKS_FOR_FEATURES`` ticks from the ring
        buffer and delegates to TechnicalFeatures, StatisticalFeatures, and
        CustomFeatures.  Returns None if the ring buffer does not have
        sufficient history or if any NaN is present in the result.

        Args:
            symbol: Instrument identifier.
            bar: The completed volume bar that triggered this computation.

        Returns:
            float32 numpy array of shape ``(N_FEATURES,)`` or None if the
            vector cannot be safely computed.
        """
        rbuf = self._ring_buffers[symbol]
        if not rbuf.is_ready(_MIN_TICKS_FOR_FEATURES):
            log.debug(
                "feature_computer.insufficient_ticks",
                symbol=symbol,
                ticks_available=rbuf.count,
                ticks_required=_MIN_TICKS_FOR_FEATURES,
            )
            return None

        # Build feature dict by delegating to each feature module.
        # When real modules are not available, return placeholder zeros so the
        # pipeline topology can be exercised before all feature code is ready.
        feat: dict[str, float] = {}

        # Build OHLCV history from completed bars
        bar_history = self._bar_history[symbol]
        if len(bar_history) < 60:
            log.debug(
                "feature_computer.insufficient_bar_history",
                symbol=symbol,
                bars=len(bar_history),
                required=60,
            )
            return None

        import polars as pl
        bar_df = pl.DataFrame({
            "open":   [b.open   for b in bar_history],
            "high":   [b.high   for b in bar_history],
            "low":    [b.low    for b in bar_history],
            "close":  [b.close  for b in bar_history],
            "volume": [b.volume for b in bar_history],
        })

        if _HAS_TECHNICAL and self._tech is not None:
            try:
                tech_feats = self._tech.compute_single_bar(bar_df)
                feat.update(tech_feats)
            except Exception as exc:
                log.warning(
                    "feature_computer.technical_compute_error",
                    symbol=symbol,
                    error=str(exc),
                )

        if _HAS_STATISTICAL and self._stat is not None:
            try:
                stat_feats = self._stat.compute_single_bar(bar_df, current_ofi=0.0)
                feat.update(stat_feats)
            except Exception as exc:
                log.warning(
                    "feature_computer.statistical_compute_error",
                    symbol=symbol,
                    error=str(exc),
                )

        if _HAS_CUSTOM and self._cust is not None:
            try:
                cust_feats = self._cust.compute_single_bar(bar_df)
                feat.update(cust_feats)
            except Exception as exc:
                log.warning(
                    "feature_computer.custom_compute_error",
                    symbol=symbol,
                    error=str(exc),
                )

        # Assemble vector in exact FEATURE_NAMES order.
        # Missing keys default to 0.0 (placeholder until all modules are built).
        vector = np.array(
            [feat.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float32,
        )

        # NaN guard — skip bar if any feature is NaN/inf [spec requirement]
        if not np.isfinite(vector).all():
            nan_features = [
                FEATURE_NAMES[i]
                for i in range(len(vector))
                if not np.isfinite(vector[i])
            ]
            log.warning(
                "feature_computer.nan_in_vector",
                symbol=symbol,
                nan_features=nan_features,
            )
            return None

        return vector

    def _publish_features(
        self,
        symbol: str,
        features: np.ndarray,
        timestamp_ns: int,
        bar_seq: int,
    ) -> None:
        """Serialize and publish a feature vector to the ZMQ PUB socket.

        Published ZMQ multipart message:
          - frame 0: symbol bytes (e.g. ``b"RELIANCE"``)
          - frame 1: msgpack-packed dict with keys:
              ``seq``          — per-symbol monotonic publish counter,
              ``timestamp_ns`` — bar close timestamp (UTC nanoseconds),
              ``features``     — raw bytes of float32 array,
              ``feature_hash`` — sha256 hex string of FEATURE_NAMES tuple,
              ``bar_type``     — "volume" | "tick" | "dollar",
              ``bar_seq``      — bar sequence number from VolumeBarAggregator.

        Args:
            symbol: Instrument identifier (used as ZMQ topic).
            features: float32 numpy array of shape ``(N_FEATURES,)``.
            timestamp_ns: Bar close timestamp in nanoseconds (UTC epoch).
            bar_seq: Monotonic bar counter from the VolumeBarAggregator.
        """
        if self._pub is None:
            return

        seq = self._pub_seq[symbol]
        self._pub_seq[symbol] = seq + 1

        last_bar = self._last_bar.get(symbol)
        bar_type: str = last_bar.bar_type if last_bar is not None else "volume"

        payload = msgpack.packb(
            {
                "seq": seq,
                "timestamp_ns": timestamp_ns,
                "features": features.tobytes(),
                "feature_hash": FEATURE_SCHEMA_HASH,
                "bar_type": bar_type,
                "bar_seq": bar_seq,
            },
            use_bin_type=True,
        )

        try:
            self._pub.send_multipart(
                [symbol.encode("utf-8"), payload],
                flags=zmq.NOBLOCK,
            )
            log.debug(
                "feature_computer.published",
                symbol=symbol,
                seq=seq,
                bar_seq=bar_seq,
                timestamp_ns=timestamp_ns,
            )
        except zmq.Again:
            log.warning(
                "feature_computer.pub_hwm_hit",
                symbol=symbol,
                seq=seq,
            )
        except zmq.ZMQError as exc:
            log.error(
                "feature_computer.pub_send_error",
                symbol=symbol,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Close ZMQ sockets and release context."""
        log.info("feature_computer.teardown_start")
        if self._sub is not None:
            try:
                self._sub.close()
            except Exception:
                pass
            self._sub = None
        if self._pub is not None:
            try:
                self._pub.close()
            except Exception:
                pass
            self._pub = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        log.info("feature_computer.teardown_complete")


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run_feature_computer(
    symbols: list[str],
    sub_port: int = 5555,
    pub_port: int = 5556,
) -> None:
    """Entry point for :class:`multiprocessing.Process` target.

    Constructs a :class:`FeatureComputer`, configures logging for the child
    process, then blocks in the event loop until shutdown.

    Args:
        symbols: List of NSE instrument identifiers to compute features for.
        sub_port: ZMQ PUB port of the upstream DataIngestor. Defaults to 5555.
        pub_port: ZMQ PUB port this process binds to publish features.
                  Defaults to 5556.

    Example:
        >>> import multiprocessing as mp
        >>> p = mp.Process(
        ...     target=run_feature_computer,
        ...     args=(["RELIANCE", "TCS"], 5555, 5556),
        ...     daemon=True,
        ... )
        >>> p.start()
    """
    # Configure structlog for the child process (the parent's handlers are not
    # inherited reliably across multiprocessing.Process on all platforms).
    try:
        from algo_trading.core.logging import configure_logging
        configure_logging()
    except Exception:
        pass  # Non-fatal — continue without structured logging

    child_log = structlog.get_logger(__name__)
    child_log.info(
        "feature_computer.process_start",
        pid=os.getpid(),
        symbols=symbols,
        sub_port=sub_port,
        pub_port=pub_port,
        n_features=N_FEATURES,
        feature_hash=FEATURE_SCHEMA_HASH,
    )

    computer = FeatureComputer(
        symbols=symbols,
        sub_port=sub_port,
        pub_port=pub_port,
    )
    computer.start()

    child_log.info("feature_computer.process_exit", pid=os.getpid())
