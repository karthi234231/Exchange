"""Custom engineered features (features 16-17, 21-23).

Provides:
- OBI (Order Book Imbalance)          — numba-JIT, exponentially-weighted (feature 16)
- Price-volume divergence              — |return| vs volume correlation (feature 17)
- Relative strength vs Nifty          — z-score of excess return (feature 21)
- Time-of-day encoding (Volume Clock) — sin/cos of VolumeTime [P2I-2.4] (feature 22)
- Bid-ask spread rolling percentile   — [0, 1] rank over last 20 bars (feature 23)
- warmup_numba()                       — pre-JIT-compile all numba kernels at startup

OBI is only valid for volume/tick bars.  Calling compute_obi when bar_type
is "time" raises a clear ValueError to prevent silent misuse.

All outputs are float32-compatible Python floats.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import polars as pl
import structlog
from numba import njit  # type: ignore[import]

from algo_trading.core.calendar import SESSION_START as _MKT_START, SESSION_END as _MKT_END  # [P0-FIX-7]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# OHLCV integrity check helper
# ---------------------------------------------------------------------------

def _validate_ohlcv(df: pl.DataFrame) -> None:
    """Raise ValueError if OHLCV columns are missing or contain invalid values.

    Checks performed (on every row):
      - high >= low
      - close in [low, high]
      - open > 0
      - volume >= 0
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing}")

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    volume = df["volume"].to_numpy()

    bad_hl = np.where(high < low)[0]
    if len(bad_hl) > 0:
        i = int(bad_hl[0])
        raise ValueError(
            f"OHLCV integrity: high={high[i]} < low={low[i]} at row {i}"
        )

    bad_close = np.where((close < low) | (close > high))[0]
    if len(bad_close) > 0:
        i = int(bad_close[0])
        raise ValueError(
            f"OHLCV integrity: close={close[i]} not in [low={low[i]}, high={high[i]}] at row {i}"
        )

    bad_open = np.where(open_ <= 0)[0]
    if len(bad_open) > 0:
        i = int(bad_open[0])
        raise ValueError(
            f"OHLCV integrity: open={open_[i]} must be > 0 at row {i}"
        )

    bad_vol = np.where(volume < 0)[0]
    if len(bad_vol) > 0:
        i = int(bad_vol[0])
        raise ValueError(
            f"OHLCV integrity: volume={volume[i]} must be >= 0 at row {i}"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OBI_ALPHA: float = 0.01        # exponential decay parameter for OBI
_DIVERGENCE_WINDOW: int = 20    # bars for price-volume divergence correlation
_RS_WINDOW: int = 20            # bars for relative-strength z-score
_SPREAD_WINDOW: int = 20        # bars for rolling spread percentile
# P0-FIX-7: Derive session length from market_calendar constants (not hardcoded)
_NSE_MINUTES_PER_DAY: int = (
    (_MKT_END.hour * 60 + _MKT_END.minute)
    - (_MKT_START.hour * 60 + _MKT_START.minute)
)  # 375 for standard 09:15-15:30 session
_VOL_PROFILE_PATH: str = os.path.join(
    os.path.dirname(__file__), "nse_volume_profile.json"
)


# ---------------------------------------------------------------------------
# Numba kernel — OBI computation
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _compute_obi_numba(
    bid_prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_prices: np.ndarray,
    ask_volumes: np.ndarray,
    mid_price: float,
    alpha: float,
) -> float:
    """Compute exponentially-weighted Order Book Imbalance (OBI).

    Each level's weight decays with its distance from the mid-price:
        weight = volume × exp(-alpha × |price - mid| / mid × 10000)
    OBI = (bid_weight - ask_weight) / (bid_weight + ask_weight)

    This is a numba-JIT function (cache=True). Call ``warmup_numba()`` at
    startup to avoid a first-call latency spike.

    Args:
        bid_prices:  1-D array of bid price levels (descending order preferred).
        bid_volumes: 1-D array of bid volumes at each price level.
        ask_prices:  1-D array of ask price levels (ascending order preferred).
        ask_volumes: 1-D array of ask volumes at each price level.
        mid_price:   Current mid-price ((best_bid + best_ask) / 2).
        alpha:       Decay coefficient (default 0.01).

    Returns:
        OBI value in [-1.0, +1.0]. Returns 0.0 when total weight is zero.
    """
    bid_weight = 0.0
    for i in range(len(bid_prices)):
        dist_bps = abs(bid_prices[i] - mid_price) / mid_price * 10_000.0
        bid_weight += bid_volumes[i] * math.exp(-alpha * dist_bps)

    ask_weight = 0.0
    for i in range(len(ask_prices)):
        dist_bps = abs(ask_prices[i] - mid_price) / mid_price * 10_000.0
        ask_weight += ask_volumes[i] * math.exp(-alpha * dist_bps)

    total = bid_weight + ask_weight
    if total < 1e-12:
        return 0.0
    return (bid_weight - ask_weight) / total


# ---------------------------------------------------------------------------
# Volume profile loader (for feature 22)
# ---------------------------------------------------------------------------

def _load_volume_profile(path: str) -> Optional[np.ndarray]:
    """Load cumulative intraday volume profile from JSON.

    The JSON file is expected to be a list of 375 floats (one per minute of
    the NSE session) representing the cumulative fraction of daily volume
    up to that minute.  Example: ``[0.002, 0.005, ..., 1.0]``.

    Args:
        path: Absolute path to ``nse_volume_profile.json``.

    Returns:
        Numpy array of shape (375,) or None if file is missing / malformed.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        profile = np.asarray(data, dtype="float64")
        if profile.shape != (_NSE_MINUTES_PER_DAY,):
            logger.warning(
                "nse_volume_profile.json has unexpected shape; using fallback",
                shape=profile.shape,
            )
            return None
        return profile
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load nse_volume_profile.json; using fallback", exc=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CustomFeatures:
    """Compute custom engineered features 16-17, 21-23.

    Args:
        bar_type: One of ``"volume"``, ``"tick"``, or ``"time"``.  OBI
                  computation is disabled when ``bar_type == "time"`` (see
                  OBI method docstring for rationale).
    """

    def __init__(self, bar_type: str = "volume") -> None:
        self._bar_type = bar_type.lower()
        self._volume_profile: Optional[np.ndarray] = _load_volume_profile(_VOL_PROFILE_PATH)
        if self._volume_profile is not None:
            logger.info("NSE volume profile loaded from disk")
        else:
            logger.info("NSE volume profile not found; using chronological sin/cos fallback")

    # ------------------------------------------------------------------
    # Feature 16 — Order Book Imbalance
    # ------------------------------------------------------------------

    def compute_obi(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid: float,
    ) -> float:
        """Compute exponentially-weighted OBI via the numba kernel.

        OBI is only meaningful for bars that aggregate order-flow at a
        representative bid/ask snapshot (volume bars, tick bars). It must
        NOT be called for time-based (15-minute OHLCV) bars because the
        bid/ask snapshot would not be representative of the bar's order flow.

        Args:
            bid_prices:  Bid price levels as 1-D float64 array.
            bid_volumes: Bid volumes at each level, same length as bid_prices.
            ask_prices:  Ask price levels as 1-D float64 array.
            ask_volumes: Ask volumes at each level, same length as ask_prices.
            mid:         Mid-price ((best_bid + best_ask) / 2).

        Returns:
            OBI in [-1.0, +1.0]. Returns 0.0 when bid/ask arrays are empty.

        Raises:
            ValueError: When ``bar_type`` was set to ``"time"`` at construction.
        """
        if self._bar_type == "time":
            raise ValueError(
                "OBI must NOT be computed on time bars (bar_type='time'). "
                "OBI is only valid for volume or tick bars where each bar has "
                "a representative bid/ask snapshot."
            )

        if len(bid_prices) == 0 or len(ask_prices) == 0:
            logger.debug("OBI: empty bid/ask arrays; returning 0.0")
            return 0.0

        bp = np.ascontiguousarray(bid_prices, dtype="float64")
        bv = np.ascontiguousarray(bid_volumes, dtype="float64")
        ap = np.ascontiguousarray(ask_prices, dtype="float64")
        av = np.ascontiguousarray(ask_volumes, dtype="float64")
        return float(_compute_obi_numba(bp, bv, ap, av, float(mid), _OBI_ALPHA))

    # ------------------------------------------------------------------
    # Feature 17 — Price-volume divergence
    # ------------------------------------------------------------------

    def compute_price_vol_divergence(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """Compute correlation of |close_change| with volume over last 20 bars.

        A negative divergence (price moving on low volume) produces a value
        in [-1, 0]; positive (price moving on high volume) produces [0, +1].

        Args:
            closes:  1-D array of close prices, length >= 2. Uses last 20 bars.
            volumes: 1-D array of bar volumes, same length as closes.

        Returns:
            Pearson correlation coefficient in [-1.0, +1.0], or NaN if
            insufficient data or zero variance.
        """
        window = min(_DIVERGENCE_WINDOW, len(closes))
        if window < 2:
            return float("nan")

        close_window = closes[-window:]
        vol_window = volumes[-window:]

        abs_changes = np.abs(np.diff(close_window.astype("float64")))
        vol_lagged = vol_window[1:]  # align with price changes

        if np.std(abs_changes) < 1e-12 or np.std(vol_lagged) < 1e-12:
            return 0.0

        corr = float(np.corrcoef(abs_changes, vol_lagged)[0, 1])
        return corr

    # ------------------------------------------------------------------
    # Feature 21 — Relative strength vs Nifty (z-score)
    # ------------------------------------------------------------------

    def compute_relative_strength(
        self,
        symbol_closes: np.ndarray,
        nifty_closes: np.ndarray,
    ) -> float:
        """Compute z-score of excess return (symbol minus Nifty) over 20 bars.

        The z-score is computed over the rolling 20-bar excess return series:
            excess_return_t = log(symbol_t / symbol_{t-1}) - log(nifty_t / nifty_{t-1})
        z-score = (current_excess - mean(excess[-20:])) / std(excess[-20:])

        Args:
            symbol_closes: 1-D array of symbol close prices. Uses last 21
                           values (20 returns + 1 for diff).
            nifty_closes:  1-D array of Nifty 50 close prices, same length.

        Returns:
            Z-score float. Returns NaN when arrays are too short or variance
            is zero.

        Raises:
            ValueError: If arrays have different lengths.
        """
        if len(symbol_closes) != len(nifty_closes):
            raise ValueError(
                f"symbol_closes and nifty_closes must have the same length; "
                f"got {len(symbol_closes)} vs {len(nifty_closes)}"
            )

        n_needed = _RS_WINDOW + 1
        if len(symbol_closes) < n_needed:
            return float("nan")

        sym = symbol_closes[-n_needed:].astype("float64")
        nft = nifty_closes[-n_needed:].astype("float64")

        sym_ret = np.diff(np.log(np.where(sym > 0, sym, np.nan)))
        nft_ret = np.diff(np.log(np.where(nft > 0, nft, np.nan)))
        excess = sym_ret - nft_ret

        valid = excess[~np.isnan(excess)]
        if len(valid) < 2:
            return float("nan")

        std = float(np.std(valid, ddof=1))
        if std < 1e-12:
            return 0.0

        return float((valid[-1] - float(np.mean(valid))) / std)

    # ------------------------------------------------------------------
    # Feature 22 — Time-of-day encoding (Volume Clock)
    # ------------------------------------------------------------------

    def encode_time_of_day(
        self,
        hour: int,
        minute: int,
        cumulative_volume: Optional[float] = None,
        total_day_volume: Optional[float] = None,
    ) -> tuple[float, float]:
        """Encode time of day as (sin, cos) of VolumeTime or chronological time.

        Volume Clock [P2I-2.4]:
            VolumeTime(t) = cumulative_volume_up_to_t / total_day_volume
            Encoding = (sin(2π × VolumeTime), cos(2π × VolumeTime))

        Fallback (chronological):
            minute_of_day = (hour - 9) * 60 + (minute - 15)   [0..374]
            Encoding = (sin(2π × minute_of_day / 375),
                        cos(2π × minute_of_day / 375))

        The volume profile JSON (if loaded) provides a pre-fitted mapping from
        minute-of-day to cumulative fraction, used when live cumulative volumes
        are unavailable.

        Args:
            hour:              Current hour (IST, 24-h).
            minute:            Current minute (IST).
            cumulative_volume: Volume accumulated so far in the session.
                               If provided together with total_day_volume,
                               these take precedence over the JSON profile.
            total_day_volume:  Expected total day volume. Must be > 0.

        Returns:
            (sin_value, cos_value) both in [-1, +1].
        """
        # Attempt live volume clock
        if cumulative_volume is not None and total_day_volume is not None and total_day_volume > 0:
            volume_time = float(cumulative_volume) / float(total_day_volume)
            angle = 2.0 * math.pi * volume_time
            return (math.sin(angle), math.cos(angle))

        # Attempt profile-based volume clock
        minute_of_day = (hour - 9) * 60 + (minute - 15)
        minute_of_day = max(0, min(minute_of_day, _NSE_MINUTES_PER_DAY - 1))

        if self._volume_profile is not None:
            volume_time = float(self._volume_profile[minute_of_day])
            angle = 2.0 * math.pi * volume_time
            return (math.sin(angle), math.cos(angle))

        # Fallback: purely chronological encoding
        angle = 2.0 * math.pi * minute_of_day / _NSE_MINUTES_PER_DAY
        return (math.sin(angle), math.cos(angle))

    # ------------------------------------------------------------------
    # Batch live-bar interface (mirrors TechnicalFeatures / StatisticalFeatures)
    # ------------------------------------------------------------------

    def compute_single_bar(self, recent_bars: pl.DataFrame) -> dict[str, float]:
        """Compute custom features for the most recent (last) bar.

        Returns canonical FEATURE_NAMES keys for the custom feature subset:
        ``obi``, ``price_vol_divergence``, ``relative_strength``,
        ``tod_sin``, ``tod_cos``, ``spread_percentile``.

        Features that require external data not present in an OHLCV DataFrame
        (OBI requires live order-book, relative_strength requires Nifty prices,
        spread_percentile requires bid-ask spread history) are returned as 0.0
        or 0.5 (spread) so the pipeline can continue without halting.

        Args:
            recent_bars: Polars DataFrame of recent OHLCV bars. Must have
                ``close`` and ``volume`` columns. Optional ``timestamp`` column
                (datetime) is used for time-of-day encoding.

        Returns:
            Dict with keys matching FEATURE_NAMES: ``obi``,
            ``price_vol_divergence``, ``relative_strength``, ``tod_sin``,
            ``tod_cos``, ``spread_percentile``.

        Raises:
            ValueError: If OHLCV columns are missing or contain invalid values
                (e.g. high < low, close outside [low, high], open <= 0, volume < 0).
        """
        _validate_ohlcv(recent_bars)
        closes = recent_bars["close"].to_numpy().astype("float64")
        volumes = recent_bars["volume"].to_numpy().astype("float64")

        # Feature 16 — OBI: requires live order-book data → 0.0 placeholder
        obi_val: float = 0.0

        # Feature 17 — Price-volume divergence (computable from OHLCV)
        pvd_val: float = self.compute_price_vol_divergence(closes, volumes)
        if pvd_val != pvd_val:  # NaN guard
            pvd_val = 0.0

        # Feature 21 — Relative strength: requires Nifty closes → 0.0 placeholder
        rs_val: float = 0.0

        # Feature 22 — Time-of-day encoding
        tod_sin_val: float = 0.0
        tod_cos_val: float = 0.0
        if "timestamp" in recent_bars.columns and len(recent_bars) > 0:
            try:
                # Use Polars .dt accessor to extract hour/minute from the last row
                h = int(recent_bars["timestamp"].dt.hour[-1])
                m = int(recent_bars["timestamp"].dt.minute[-1])
                tod_sin_val, tod_cos_val = self.encode_time_of_day(h, m)
            except Exception:
                tod_sin_val, tod_cos_val = 0.0, 1.0  # fallback: session-start angle

        # Feature 23 — Spread percentile: requires bid-ask history → 0.5 neutral
        spread_pct_val: float = 0.5

        return {
            "obi":                 obi_val,
            "price_vol_divergence": pvd_val,
            "relative_strength":   rs_val,
            "tod_sin":             tod_sin_val,
            "tod_cos":             tod_cos_val,
            "spread_percentile":   spread_pct_val,
        }

    # ------------------------------------------------------------------
    # Feature 23 — Bid-ask spread rolling percentile
    # ------------------------------------------------------------------

    def compute_spread_percentile(
        self,
        current_spread: float,
        recent_spreads: np.ndarray,
    ) -> float:
        """Compute rolling 20-bar percentile rank of the current spread.

        The percentile rank tells how wide the current spread is relative
        to the most recent ``len(recent_spreads)`` historical spreads.
        Returns 1.0 when current spread is the widest in the window,
        0.0 when it is the tightest.

        Args:
            current_spread: Absolute or relative bid-ask spread for the
                            current bar.
            recent_spreads: 1-D array of historical spreads (last 20 bars,
                            not including current bar). May be shorter if
                            insufficient history.

        Returns:
            Percentile rank in [0.0, 1.0]. Returns 0.5 when the window
            is empty (no history available).
        """
        if len(recent_spreads) == 0:
            return 0.5

        window = recent_spreads[-_SPREAD_WINDOW:]
        n = len(window)
        n_below = float(np.sum(window < current_spread))
        n_equal = float(np.sum(window == current_spread))
        # Mid-rank formula to handle ties: (n_below + 0.5 * n_equal) / n
        percentile = (n_below + 0.5 * n_equal) / n
        return float(np.clip(percentile, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------

def warmup_numba() -> None:
    """Pre-compile numba functions at startup.

    Call this once (e.g. in the main entry point or orchestrator __init__)
    before market open to avoid a JIT compilation latency spike on the first
    live order-book event.

    The function runs the numba kernel on tiny synthetic arrays to trigger
    compilation without side effects.
    """
    logger.info("warming up numba JIT kernels")
    _dummy_bid_prices = np.array([100.0, 99.9, 99.8], dtype="float64")
    _dummy_bid_vols = np.array([100.0, 200.0, 150.0], dtype="float64")
    _dummy_ask_prices = np.array([100.1, 100.2, 100.3], dtype="float64")
    _dummy_ask_vols = np.array([120.0, 180.0, 90.0], dtype="float64")
    _mid = 100.05

    result = _compute_obi_numba(
        _dummy_bid_prices,
        _dummy_bid_vols,
        _dummy_ask_prices,
        _dummy_ask_vols,
        _mid,
        _OBI_ALPHA,
    )
    logger.info("numba warmup complete", obi_result=float(result))
