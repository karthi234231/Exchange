"""
algo_trading/market_data/staleness_detector.py — Live quote freshness monitoring.

Separate failure mode from normalisation
-----------------------------------------
``ChainNormalizer`` handles format and content issues (crossed quotes, negative IV,
missing strikes).  ``StalenessDetector`` handles timing/freshness issues only.
Keeping them separate means a feed that is corrupt-but-fresh and a feed that is
valid-but-stale produce different alert paths.

Replay clock semantics (design Section 9.2)
-------------------------------------------
Staleness is computed using ``quote_timestamp`` from each ``OptionQuote``, which
carries the exchange timestamp.  During live trading this approximates wall-clock
age accurately.  During replay the detector MUST NOT be used, because replayed
events will always appear "stale" relative to the replay wall clock.  Pass
``now=event_timestamp`` explicitly when calling ``check_snapshot`` in simulation.

Design constraints
-------------------
- Does NOT modify snapshot data — only monitors and fires callbacks.
- All timestamps handled as timezone-aware UTC datetimes.
- ``severity`` levels:
    - ``"WARNING"`` : quote older than ``max_quote_age_seconds``
    - ``"CRITICAL"`` : quote older than ``halt_trading_after_seconds``
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import structlog

from algo_trading.market_data.chain_snapshot import ChainSnapshot

log = structlog.get_logger(__name__)


@dataclass
class StalenessConfig:
    """Thresholds governing the staleness detector.

    Attributes:
        max_quote_age_seconds: Fire a WARNING alert if any quote is older
            than this many seconds.  Default: 30.0.
        max_snapshot_age_seconds: Fire a WARNING alert if the snapshot
            ``snapshot_timestamp`` itself is this old (feed dropout).
            Default: 120.0.
        halt_trading_after_seconds: Escalate to CRITICAL and signal
            ``should_halt_trading`` after this many seconds of stale data.
            Must be >= ``max_quote_age_seconds``.  Default: 60.0.
        check_interval_seconds: How often the background monitor thread
            calls ``check_snapshot`` on the last-seen snapshot.  Default: 5.0.
    """

    max_quote_age_seconds: float = 30.0
    max_snapshot_age_seconds: float = 120.0
    halt_trading_after_seconds: float = 60.0
    check_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.halt_trading_after_seconds < self.max_quote_age_seconds:
            raise ValueError(
                "halt_trading_after_seconds must be >= max_quote_age_seconds"
            )


@dataclass
class StalenessAlert:
    """Describes a single staleness violation detected on a quote.

    Attributes:
        symbol: Underlying symbol of the affected snapshot.
        quote_timestamp: Exchange timestamp on the stale quote.
        detected_at: Wall-clock time when the staleness was detected.
        age_seconds: How old the quote was at detection time.
        severity: ``"WARNING"`` or ``"CRITICAL"``.
        description: Human-readable explanation.
    """

    symbol: str
    quote_timestamp: datetime
    detected_at: datetime
    age_seconds: float
    severity: str   # "WARNING" | "CRITICAL"
    description: str = ""


class StalenessDetector:
    """Monitors quote age and fires callbacks when quotes exceed age thresholds.

    This detector does NOT modify data.  It observes snapshots, computes
    quote ages, and calls the registered ``alert_callback`` for every
    ``StalenessAlert`` it creates.

    Two failure modes are distinguished:
        - WARNING  : quote age > ``config.max_quote_age_seconds``
        - CRITICAL : quote age > ``config.halt_trading_after_seconds``

    Usage (live trading)::

        config = StalenessConfig(max_quote_age_seconds=30.0)
        detector = StalenessDetector(config, alert_callback=my_handler)

        # On every new chain snapshot:
        detector.update(snapshot)

        # Before placing an order:
        if detector.should_halt_trading(snapshot):
            raise StaleDataError("...")

    Usage (replay — pass explicit now)::

        # During backtesting, pass the event_timestamp as 'now' to avoid
        # false CRITICAL alerts from wall-clock skew.
        alerts = detector.check_snapshot(snapshot, now=snapshot.event_timestamp)

    Parameters
    ----------
    config:
        ``StalenessConfig`` instance with threshold values.
    alert_callback:
        Optional callable ``(StalenessAlert) -> None``.  Called from
        ``check_snapshot`` for every alert detected.  Must be thread-safe
        if ``start_background_monitoring`` is used.
    """

    def __init__(
        self,
        config: StalenessConfig | None = None,
        alert_callback: Callable[[StalenessAlert], None] | None = None,
    ) -> None:
        self._config = config if config is not None else StalenessConfig()
        self._alert_callback = alert_callback
        self._last_snapshot: ChainSnapshot | None = None
        self._last_snapshot_lock = threading.Lock()

        # Background monitor state.
        self._bg_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        log.info(
            "staleness_detector.init",
            max_quote_age_seconds=self._config.max_quote_age_seconds,
            halt_trading_after_seconds=self._config.halt_trading_after_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: ChainSnapshot) -> None:
        """Register the most recent snapshot for background monitoring.

        Call this on every new snapshot received from the feed.  The
        background monitor thread (if running) will use the stored snapshot
        for periodic checks.

        Does NOT trigger an alert on its own — call ``check_snapshot``
        explicitly if you want immediate feedback.

        Args:
            snapshot: The newly received ``ChainSnapshot``.
        """
        with self._last_snapshot_lock:
            self._last_snapshot = snapshot

    def check_snapshot(
        self,
        snapshot: ChainSnapshot,
        now: datetime | None = None,
    ) -> list[StalenessAlert]:
        """Inspect all quotes in *snapshot* and return any staleness alerts.

        For each ``OptionQuote``, the age is computed as::

            age = (now - quote.quote_timestamp).total_seconds()

        Replay note (design Section 9.2): During backtesting pass
        ``now=snapshot.event_timestamp`` (the exchange timestamp) rather
        than leaving it as ``None`` (wall clock).  Using wall-clock time
        during replay creates false CRITICAL alerts because replayed quotes
        will always appear very old relative to the current wall clock.

        Args:
            snapshot: Chain snapshot to inspect.
            now: Reference time for age computation.  Defaults to
                ``datetime.now(timezone.utc)`` (wall clock).

        Returns:
            List of ``StalenessAlert`` objects, one per stale quote detected.
            Empty list means all quotes are within thresholds.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        # Ensure both sides are timezone-aware for subtraction.
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        alerts: list[StalenessAlert] = []

        # Check snapshot-level freshness (feed dropout).
        snap_ts = snapshot.snapshot_timestamp
        if snap_ts.tzinfo is None:
            snap_ts = snap_ts.replace(tzinfo=timezone.utc)
        snap_age = (now - snap_ts).total_seconds()

        if snap_age > self._config.max_snapshot_age_seconds:
            severity = (
                "CRITICAL"
                if snap_age > self._config.halt_trading_after_seconds
                else "WARNING"
            )
            alert = StalenessAlert(
                symbol=snapshot.symbol,
                quote_timestamp=snapshot.snapshot_timestamp,
                detected_at=now,
                age_seconds=snap_age,
                severity=severity,
                description=(
                    f"Snapshot itself is {snap_age:.1f}s old "
                    f"(threshold {self._config.max_snapshot_age_seconds}s)"
                ),
            )
            alerts.append(alert)
            self._fire(alert)
            log.warning(
                "staleness_detector.stale_snapshot",
                symbol=snapshot.symbol,
                age_seconds=snap_age,
                severity=severity,
            )

        # Check individual quote freshness.
        for (expiry, strike, opt_type), quote in snapshot.quotes.items():
            qt = quote.quote_timestamp
            if qt is None:
                continue
            if qt.tzinfo is None:
                qt = qt.replace(tzinfo=timezone.utc)

            age = (now - qt).total_seconds()
            if age <= self._config.max_quote_age_seconds:
                continue  # fresh — no alert

            severity = (
                "CRITICAL"
                if age > self._config.halt_trading_after_seconds
                else "WARNING"
            )
            alert = StalenessAlert(
                symbol=snapshot.symbol,
                quote_timestamp=qt,
                detected_at=now,
                age_seconds=age,
                severity=severity,
                description=(
                    f"{snapshot.symbol} {expiry} {strike} {opt_type} "
                    f"quote is {age:.1f}s old "
                    f"(threshold {self._config.max_quote_age_seconds}s)"
                ),
            )
            alerts.append(alert)
            self._fire(alert)
            log.warning(
                "staleness_detector.stale_quote",
                symbol=snapshot.symbol,
                expiry=str(expiry),
                strike=strike,
                option_type=opt_type,
                age_seconds=age,
                severity=severity,
            )

        return alerts

    def should_halt_trading(
        self,
        snapshot: ChainSnapshot,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` if data staleness is severe enough to stop trading.

        A snapshot triggers a halt when:
            - The snapshot_timestamp is older than ``halt_trading_after_seconds``, OR
            - Any individual quote is older than ``halt_trading_after_seconds``.

        Replay note (design Section 9.2): Pass ``now=snapshot.event_timestamp``
        during backtesting to avoid false halts from wall-clock skew.

        Args:
            snapshot: Chain snapshot to inspect.
            now: Reference time.  Defaults to ``datetime.now(timezone.utc)``.

        Returns:
            ``True`` if trading should be halted; ``False`` otherwise.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        snap_ts = snapshot.snapshot_timestamp
        if snap_ts.tzinfo is None:
            snap_ts = snap_ts.replace(tzinfo=timezone.utc)
        if (now - snap_ts).total_seconds() > self._config.halt_trading_after_seconds:
            return True

        for quote in snapshot.quotes.values():
            qt = quote.quote_timestamp
            if qt is None:
                continue
            if qt.tzinfo is None:
                qt = qt.replace(tzinfo=timezone.utc)
            if (now - qt).total_seconds() > self._config.halt_trading_after_seconds:
                return True

        return False

    def start_background_monitoring(self) -> None:
        """Start a background thread that periodically checks the last snapshot.

        The thread calls ``check_snapshot(last_snapshot)`` every
        ``config.check_interval_seconds``.  Alerts are fired via the
        callback registered at construction time.

        Raises:
            RuntimeError: If monitoring is already running.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            raise RuntimeError("Background monitoring already running.  Call stop() first.")

        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="StalenessDetector-bg",
        )
        self._bg_thread.start()
        log.info("staleness_detector.background_started")

    def stop(self) -> None:
        """Stop the background monitoring thread gracefully.

        Waits up to 10 seconds for the thread to exit.  Safe to call
        even if no background thread is running.
        """
        self._stop_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=10.0)
        self._bg_thread = None
        log.info("staleness_detector.stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, alert: StalenessAlert) -> None:
        """Invoke the alert callback if one is registered.

        Args:
            alert: Alert to dispatch.
        """
        if self._alert_callback is not None:
            try:
                self._alert_callback(alert)
            except Exception:
                log.exception("staleness_detector.callback_error")

    def _background_loop(self) -> None:
        """Periodic staleness check loop run in the background thread."""
        log.info("staleness_detector.background_loop_started")
        while not self._stop_event.is_set():
            with self._last_snapshot_lock:
                snapshot = self._last_snapshot

            if snapshot is not None:
                try:
                    self.check_snapshot(snapshot)
                except Exception:
                    log.exception("staleness_detector.background_loop_error")

            self._stop_event.wait(timeout=self._config.check_interval_seconds)

        log.info("staleness_detector.background_loop_stopped")
