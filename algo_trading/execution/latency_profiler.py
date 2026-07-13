"""
latency_profiler.py — Five-component latency chain profiler for NSE AlgoTrader.

Measures and tracks the full tick-to-ack latency chain defined in spec
[SURV-3]:

    1. tick_received_ns     → WebSocket tick arrived
    2. feature_computed_ns  → all 23 features computed
    3. signal_generated_ns  → ML inference complete
    4. order_placed_ns      → order sent to broker API
    5. exchange_ack_ns      → broker acknowledgement received

Rolling P50/P95/P99 are maintained per component in a fixed-size deque
(``window`` most-recent samples).  Any component whose latest sample exceeds
``alert_threshold_multiplier × P95`` triggers a structlog WARNING.

QuestDB ILP writes are delegated to the caller-supplied ``sender`` object so
this module has zero network dependencies of its own.

Usage::

    from algo_trading.execution.latency_profiler import LatencyProfiler, LatencySnapshot

    profiler = LatencyProfiler(window=100, alert_threshold_multiplier=2.0)

    snap = LatencySnapshot(
        symbol="RELIANCE",
        bar_seq=42,
        tick_received_ns=...,
        feature_computed_ns=...,
        signal_generated_ns=...,
        order_placed_ns=...,
        exchange_ack_ns=...,
    )
    profiler.record(snap)
    spiked = profiler.check_spike(snap)
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

try:
    from questdb.ingress import TimestampNanos as _TimestampNanos
except ImportError:
    _TimestampNanos = None  # type: ignore[assignment,misc]

from algo_trading.config import settings  # noqa: F401

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Component names — used as dict keys everywhere
# ---------------------------------------------------------------------------
_COMPONENTS: tuple[str, ...] = (
    "tick_to_feature",
    "feature_to_signal",
    "signal_to_order",
    "order_to_ack",
    "total",
)

_NS_TO_MS: float = 1e-6  # nanoseconds → milliseconds
_STATS_PATH = Path("state/latency_stats.json")
_PERSIST_INTERVAL_SEC: float = 15 * 60  # 15 minutes


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class LatencySnapshot:
    """Single bar's full latency measurements (all timestamps in nanoseconds).

    Attributes:
        symbol: NSE symbol, e.g. ``"RELIANCE"``.
        bar_seq: Monotonically increasing bar sequence number.
        tick_received_ns: ``time.time_ns()`` when the WebSocket tick arrived.
        feature_computed_ns: Timestamp after all 23 features are computed.
        signal_generated_ns: Timestamp after ML inference completes.
        order_placed_ns: Timestamp when the order is sent to the broker.
        exchange_ack_ns: Timestamp when the broker acknowledgement arrives.
    """

    symbol: str
    bar_seq: int
    tick_received_ns: int
    feature_computed_ns: int
    signal_generated_ns: int
    order_placed_ns: int
    exchange_ack_ns: int

    # ------------------------------------------------------------------
    # Derived properties (milliseconds)
    # ------------------------------------------------------------------

    @property
    def tick_to_feature_ms(self) -> float:
        """Milliseconds from tick received to features computed.

        Returns:
            Latency in milliseconds as a float.
        """
        return (self.feature_computed_ns - self.tick_received_ns) * _NS_TO_MS

    @property
    def feature_to_signal_ms(self) -> float:
        """Milliseconds from features computed to ML signal generated.

        Returns:
            Latency in milliseconds as a float.
        """
        return (self.signal_generated_ns - self.feature_computed_ns) * _NS_TO_MS

    @property
    def signal_to_order_ms(self) -> float:
        """Milliseconds from signal generated to order placed.

        Returns:
            Latency in milliseconds as a float.
        """
        return (self.order_placed_ns - self.signal_generated_ns) * _NS_TO_MS

    @property
    def order_to_ack_ms(self) -> float:
        """Milliseconds from order placed to exchange acknowledgement.

        Returns:
            Latency in milliseconds as a float.
        """
        return (self.exchange_ack_ns - self.order_placed_ns) * _NS_TO_MS

    @property
    def total_ms(self) -> float:
        """End-to-end milliseconds from tick received to exchange ack.

        Returns:
            Total latency in milliseconds as a float.
        """
        return (self.exchange_ack_ns - self.tick_received_ns) * _NS_TO_MS

    def component_values(self) -> dict[str, float]:
        """Return all five component latencies as a dict keyed by name.

        Returns:
            Mapping of component name to latency in milliseconds.
        """
        return {
            "tick_to_feature": self.tick_to_feature_ms,
            "feature_to_signal": self.feature_to_signal_ms,
            "signal_to_order": self.signal_to_order_ms,
            "order_to_ack": self.order_to_ack_ms,
            "total": self.total_ms,
        }


# ---------------------------------------------------------------------------
# Rolling profiler
# ---------------------------------------------------------------------------


class LatencyProfiler:
    """Rolling percentile tracker with spike alerts per latency component.

    Maintains a fixed-length deque of samples for each of the five latency
    components and computes P50/P95/P99 on demand using numpy.  If any
    component in a new snapshot exceeds ``alert_threshold_multiplier × P95``
    of its rolling window, a structlog WARNING is emitted.

    Attributes:
        window: Number of most-recent samples to retain per component.
        alert_threshold_multiplier: Spike threshold expressed as a multiple
            of the rolling P95.  Default is ``2.0`` (2× P95).
    """

    def __init__(
        self,
        window: int = 100,
        alert_threshold_multiplier: float = 2.0,
    ) -> None:
        """Initialise the profiler with rolling-window size and alert threshold.

        Loads persisted latency percentiles from ``state/latency_stats.json``
        on startup if the file exists.  Starts a background thread that
        persists current stats every 15 minutes.

        Args:
            window: Rolling-window size (number of samples per component).
                Must be >= 2.
            alert_threshold_multiplier: A component is flagged as a spike when
                its latest value exceeds this multiple of the window P95.

        Raises:
            ValueError: If ``window`` < 2 or ``alert_threshold_multiplier`` <= 0.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2; got {window}")
        if alert_threshold_multiplier <= 0:
            raise ValueError(
                f"alert_threshold_multiplier must be > 0; got {alert_threshold_multiplier}"
            )
        self.window = window
        self.alert_threshold_multiplier = alert_threshold_multiplier
        self._buffers: dict[str, deque[float]] = {
            c: deque(maxlen=window) for c in _COMPONENTS
        }
        self._rate_reduce_active: bool = False
        self._consecutive_spikes: int = 0

        # Load persisted stats from previous session if available
        self._load_persisted_stats()

        # Background thread: persist stats every 15 minutes
        self._persist_timer: threading.Timer | None = None
        self._schedule_persist()

    def _load_persisted_stats(self) -> None:
        """Load latency percentiles from disk into rolling buffers on startup."""
        if not _STATS_PATH.exists():
            return
        try:
            with _STATS_PATH.open(encoding="utf-8") as fh:
                saved: dict = json.load(fh)
            # Seed each buffer with the saved p50 value so history is not lost
            for component, stats in saved.items():
                if component in self._buffers and "p50" in stats:
                    self._buffers[component].append(float(stats["p50"]))
            log.info("latency_profiler.stats_loaded", path=str(_STATS_PATH))
        except Exception as exc:
            log.warning("latency_profiler.load_failed", error=str(exc))

    def _persist_stats(self) -> None:
        """Write current percentile stats to disk and reschedule."""
        try:
            _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            stats = self.get_stats()
            with _STATS_PATH.open("w", encoding="utf-8") as fh:
                json.dump(stats, fh)
            log.debug("latency_profiler.stats_persisted", path=str(_STATS_PATH))
        except Exception as exc:
            log.error("latency_profiler.persist_failed", error=str(exc))
        finally:
            self._schedule_persist()

    def _schedule_persist(self) -> None:
        """Schedule the next stats persist in 15 minutes (daemon thread)."""
        self._persist_timer = threading.Timer(
            _PERSIST_INTERVAL_SEC, self._persist_stats
        )
        self._persist_timer.daemon = True
        self._persist_timer.start()

    def is_rate_reduced(self) -> bool:
        """Return True when consecutive spike threshold has been breached.

        Callers should reduce order submission rate while this returns True.
        The flag is set when any component exceeds ``alert_threshold_multiplier``
        × P95 for 5 consecutive bars, and is cleared when a non-spike bar is
        seen.

        Returns:
            bool: True if rate reduction is active.
        """
        return self._rate_reduce_active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, snap: LatencySnapshot) -> None:
        """Add a new snapshot to the rolling buffers and check for spikes.

        Appends each component latency to its deque (oldest value is
        automatically evicted once ``window`` is reached).  Spike detection
        runs automatically via :meth:`check_spike`.

        Args:
            snap: The :class:`LatencySnapshot` to record.
        """
        values = snap.component_values()
        for component, ms in values.items():
            self._buffers[component].append(ms)

        spiked = self.check_spike(snap)

        log.debug(
            "latency_recorded",
            symbol=snap.symbol,
            bar_seq=snap.bar_seq,
            **{f"{k}_ms": round(v, 3) for k, v in values.items()},
            spiked_components=spiked or None,
        )

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Compute rolling P50/P95/P99 for every component.

        Only components with at least two samples contribute meaningful
        percentiles; components with a single sample return that value for
        all three percentiles.

        Returns:
            Nested dict keyed ``component → {p50, p95, p99}`` with values in
            milliseconds rounded to three decimal places.  Returns an empty
            inner dict for components with no data.
        """
        stats: dict[str, dict[str, float]] = {}
        for component, buf in self._buffers.items():
            if not buf:
                stats[component] = {}
                continue
            arr = np.array(buf, dtype=np.float64)
            p50, p95, p99 = float(np.percentile(arr, 50)), float(
                np.percentile(arr, 95)
            ), float(np.percentile(arr, 99))
            stats[component] = {
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "n": len(buf),
            }
        return stats

    def check_spike(self, snap: LatencySnapshot) -> list[str]:
        """Identify components in ``snap`` that exceed the spike threshold.

        A component is classified as a spike when:
            ``latest_value > alert_threshold_multiplier × rolling_P95``

        Insufficient data (< 2 samples in buffer after recording) suppresses
        the check for that component to avoid false positives on warm-up.

        A structlog WARNING is emitted for each spiked component.

        Args:
            snap: The :class:`LatencySnapshot` to evaluate (must already be
                recorded via :meth:`record` or the buffers may not yet contain
                this sample).

        Returns:
            List of component names that spiked.  Empty list when none spike.
        """
        spiked: list[str] = []
        values = snap.component_values()

        for component, latest_ms in values.items():
            buf = self._buffers[component]
            if len(buf) < 2:
                continue  # Not enough history for a meaningful P95

            arr = np.array(buf, dtype=np.float64)
            p95 = float(np.percentile(arr, 95))
            threshold = p95 * self.alert_threshold_multiplier

            if latest_ms > threshold:
                spiked.append(component)
                log.warning(
                    "latency_spike_detected",
                    symbol=snap.symbol,
                    bar_seq=snap.bar_seq,
                    component=component,
                    latest_ms=round(latest_ms, 3),
                    p95_ms=round(p95, 3),
                    threshold_ms=round(threshold, 3),
                    multiplier=self.alert_threshold_multiplier,
                )

        # Track consecutive bars with at least one spike; activate rate reduction
        # after 5 consecutive spiked bars.
        if spiked:
            self._consecutive_spikes += 1
            if self._consecutive_spikes >= 5 and not self._rate_reduce_active:
                self._rate_reduce_active = True
                log.warning(
                    "latency_rate_reduce_activated",
                    consecutive_spikes=self._consecutive_spikes,
                    symbol=snap.symbol,
                    bar_seq=snap.bar_seq,
                )
        else:
            if self._consecutive_spikes > 0:
                self._consecutive_spikes = 0
            if self._rate_reduce_active:
                self._rate_reduce_active = False
                log.info(
                    "latency_rate_reduce_cleared",
                    symbol=snap.symbol,
                    bar_seq=snap.bar_seq,
                )

        return spiked

    def log_to_questdb(self, snap: LatencySnapshot, sender: Any) -> None:
        """Write a latency snapshot to the QuestDB ``latency_log`` table via ILP.

        The caller must supply an active ``questdb.ingress.Sender`` (or any
        object with a compatible ``.row()`` method).  This module never
        owns or manages a QuestDB connection — that responsibility belongs to
        the caller.

        ILP line written::

            latency_log,symbol=<sym>,bar_seq=<n> \\
                tick_to_feature_ms=<f>, \\
                feature_to_signal_ms=<f>, \\
                signal_to_order_ms=<f>, \\
                order_to_ack_ms=<f>, \\
                total_ms=<f> \\
                <exchange_ack_ns>

        Args:
            snap: The :class:`LatencySnapshot` to write.
            sender: An active QuestDB ILP sender object.  Must expose a
                ``.row(table, symbols, columns, at)`` interface compatible
                with the ``questdb-connect`` / ``questdb.ingress`` client.

        Raises:
            Exception: Any exception from the underlying sender is propagated
                to the caller (not swallowed here).
        """
        sender.row(
            "latency_log",
            symbols={
                "symbol": snap.symbol,
            },
            columns={
                "bar_seq": snap.bar_seq,
                "tick_to_feature_ms": snap.tick_to_feature_ms,
                "feature_to_signal_ms": snap.feature_to_signal_ms,
                "signal_to_order_ms": snap.signal_to_order_ms,
                "order_to_ack_ms": snap.order_to_ack_ms,
                "total_ms": snap.total_ms,
            },
            at=_TimestampNanos(snap.exchange_ack_ns) if _TimestampNanos is not None else snap.exchange_ack_ns,
        )
        log.debug(
            "latency_written_to_questdb",
            symbol=snap.symbol,
            bar_seq=snap.bar_seq,
            total_ms=round(snap.total_ms, 3),
        )
