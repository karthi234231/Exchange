"""
algo_trading/research/chain_replay_engine.py — Deterministic replay of historical
NSE options chain snapshots for a date range.

Design principles (FO_SYSTEM_DESIGN.md Section 9.3):
  - Deterministic: two identical runs produce identical outputs, always.
  - Uses event_time (exchange timestamp on each quote), NOT ingestion time.
    Ingestion latency creates false causality in backtesting.
  - Sequence numbers for ordering within the same timestamp.
  - Exposes a simple iterator interface to strategy code.

DETERMINISM CONTRACT: replay() must NOT call datetime.now() / time.time(),
use random(), make network calls, or depend on OS state.  All state comes
from the data_provider and the ReplayConfig.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import structlog

log = structlog.get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplayConfig:
    """
    Configuration for a single replay run.

    Stored in the audit log alongside every result — reproducibility requires
    knowing exactly what config produced the output.

    Parameters
    ----------
    symbol : str
        Underlying symbol, e.g. "NIFTY" or "BANKNIFTY".
    start_date : date
        First calendar date in the replay window (inclusive).
    end_date : date
        Last calendar date in the replay window (inclusive).
    snapshot_interval_seconds : int
        Archival interval of the source chain snapshots (default 60 s).
    expiry_filter : list[date] | None
        If provided, only snapshots whose nearest expiry is in this list
        are yielded.  None = all expiries.
    data_quality_threshold : str
        Minimum acceptable quality level.  Quality order (ascending):
        UNUSABLE < PARTIAL < COMPLETE.  Snapshots below this threshold
        are skipped with a warning.
    strategy_params : dict
        Arbitrary strategy parameters stored for reproducibility.  The
        engine does not interpret these; they exist purely so that the
        audit log captures everything needed to re-run the experiment.
    """
    symbol: str
    start_date: date
    end_date: date
    snapshot_interval_seconds: int = 60
    expiry_filter: list[date] | None = None
    data_quality_threshold: str = "PARTIAL"   # COMPLETE > PARTIAL > UNUSABLE
    strategy_params: dict = field(default_factory=dict)

    def config_hash(self) -> str:
        """
        Return a 16-hex-char SHA-256 fingerprint of this config.

        Used to detect accidental config drift across runs.  Store alongside
        every result; two results with identical hashes are guaranteed to have
        been produced with identical configs.
        """
        payload = json.dumps(
            {
                "symbol": self.symbol,
                "start_date": str(self.start_date),
                "end_date": str(self.end_date),
                "snapshot_interval_seconds": self.snapshot_interval_seconds,
                "expiry_filter": [str(e) for e in (self.expiry_filter or [])],
                "data_quality_threshold": self.data_quality_threshold,
                "strategy_params": self.strategy_params,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


# ── Replay event ───────────────────────────────────────────────────────────────

@dataclass
class ReplayEvent:
    """
    A single event in the replay stream.

    Strategy code receives these from the engine's ``replay()`` iterator.
    All time references must be taken from ``event_time`` — never from
    wall-clock or OS time.

    Attributes
    ----------
    event_time : datetime
        Exchange-stamped timestamp of the snapshot.  USE THIS for all
        strategy logic.
    sequence_number : int
        Monotonically increasing integer.  Guarantees a total ordering
        even when two events share the same ``event_time``.
    snapshot : Any
        The raw chain snapshot object (ChainSnapshot or dict depending on
        the data provider).  Typed as Any to avoid a circular import with
        algo_trading.market_data.chain_snapshot.
    session_day : date
        The trading day this event belongs to.
    bars_elapsed_today : int
        Number of snapshots yielded for ``session_day`` before this one
        (0-indexed, so the first bar of the day has bars_elapsed_today=0).
    is_expiry_bar : bool
        True when ``session_day`` is an F&O expiry day for the configured
        symbol.
    minutes_to_session_close : float
        Minutes from ``event_time`` until 15:30 IST (NSE session close).
        Floored at 0.0; never negative.
    """
    event_time: datetime
    sequence_number: int
    snapshot: Any
    session_day: date
    bars_elapsed_today: int
    is_expiry_bar: bool
    minutes_to_session_close: float

    @property
    def is_near_close(self) -> bool:
        """True when fewer than 30 minutes remain in the trading session."""
        return self.minutes_to_session_close <= 30.0


# ── Replay engine ─────────────────────────────────────────────────────────────

class ChainReplayEngine:
    """
    Deterministic replay of historical NSE options chain snapshots.

    Core contract: same input → same output, always.  No randomness, no
    wall-clock dependency, no network calls during replay.

    Replay uses ``event_time`` (the exchange timestamp embedded in each
    archived quote), NOT ingestion time.  Using ingestion time introduces
    false causality: a snapshot ingested late would appear to arrive before
    events that actually preceded it in market time.

    Parameters
    ----------
    data_provider : SnapshotStore
        Source of historical chain data.  Must implement
        ``algo_trading.fo.interfaces.SnapshotStore`` (or any object with
        a compatible ``read_range(symbol, start, end)`` method).
    config : ReplayConfig
        Defines what to replay and how.

    Usage
    -----
    ::

        engine = ChainReplayEngine(provider, config)
        for event in engine.replay():
            signal = my_strategy.on_snapshot(event)
            if signal:
                fill = fill_simulator.simulate_fill(..., event.event_time)

    Notes
    -----
    The ``data_provider.read_range`` interface is expected to accept
    ``(symbol, start_datetime, end_datetime)`` and return an iterable of
    snapshot objects or dicts.  Snapshots must carry at least one of the
    keys ``event_timestamp`` or ``snapshot_timestamp`` (datetime) and
    optionally ``sequence_number`` (int) and ``data_quality`` (str).
    """

    # Quality level ordering — higher = better
    _QUALITY_ORDER: dict[str, int] = {"UNUSABLE": 0, "PARTIAL": 1, "COMPLETE": 2}

    def __init__(self, data_provider: Any, config: ReplayConfig) -> None:
        self._provider = data_provider
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def replay(self) -> Iterator[ReplayEvent]:
        """
        Main replay iterator.  Yields :class:`ReplayEvent` in event_time order.

        DETERMINISM GUARANTEE: this method does NOT call ``datetime.now()``,
        ``time.time()``, ``random()``, make network calls, or depend on any
        OS state.  All state comes from the data_provider and the config.

        Yields
        ------
        ReplayEvent
            One event per accepted snapshot, in chronological order.

        Notes
        -----
        Snapshots are sorted by ``(event_time, sequence_number)`` within
        each day before being yielded.  Low-quality snapshots are skipped
        with a structlog WARNING and do not increment the sequence counter.
        """
        from algo_trading.core.calendar import is_trading_day
        from algo_trading.fo.expiry import (
            nifty_expiry_calendar,
            banknifty_expiry_calendar,
        )

        expiry_cal = (
            nifty_expiry_calendar
            if self._config.symbol.upper() == "NIFTY"
            else banknifty_expiry_calendar
        )

        current_date = self._config.start_date
        end_date = self._config.end_date
        sequence = 0

        while current_date <= end_date:
            if not is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            # NSE session window for this trading day
            day_start = datetime(
                current_date.year, current_date.month, current_date.day, 9, 15
            )
            day_end = datetime(
                current_date.year, current_date.month, current_date.day, 15, 30
            )

            # Fetch all snapshots for the day from the provider.
            # The provider's read_range signature:
            #   read_range(symbol: str, start: datetime, end: datetime) -> list[ChainSnapshot]
            raw_snapshots = list(
                self._provider.read_range(self._config.symbol, day_start, day_end)
            )

            if not raw_snapshots:
                log.debug(
                    "replay.no_snapshots",
                    date=str(current_date),
                    symbol=self._config.symbol,
                )
                current_date += timedelta(days=1)
                continue

            # CRITICAL: sort by event_time, then by sequence_number.
            # Do NOT use ingestion time or wall-clock order.
            raw_snapshots.sort(
                key=lambda s: (
                    self._extract_event_time(s) or day_start,
                    self._extract_sequence(s),
                )
            )

            is_expiry = expiry_cal.is_expiry_day(current_date)
            bars_today = 0

            for snap in raw_snapshots:
                # Skip snapshots that fail the data quality threshold
                quality = self._extract_quality(snap)
                if not self._quality_ok(quality):
                    log.warning(
                        "replay.skip_low_quality",
                        date=str(current_date),
                        quality=quality,
                        threshold=self._config.data_quality_threshold,
                    )
                    continue

                event_ts = self._extract_event_time(snap)
                if event_ts is None:
                    log.warning(
                        "replay.skip_missing_timestamp",
                        date=str(current_date),
                    )
                    continue

                # Apply expiry filter if configured
                if self._config.expiry_filter is not None:
                    snap_expiry = self._extract_expiry(snap)
                    if snap_expiry not in self._config.expiry_filter:
                        continue

                mins_to_close = (day_end - event_ts).total_seconds() / 60.0

                yield ReplayEvent(
                    event_time=event_ts,
                    sequence_number=sequence,
                    snapshot=snap,
                    session_day=current_date,
                    bars_elapsed_today=bars_today,
                    is_expiry_bar=is_expiry,
                    minutes_to_session_close=max(0.0, mins_to_close),
                )
                sequence += 1
                bars_today += 1

            current_date += timedelta(days=1)

    def get_config_hash(self) -> str:
        """Return the SHA-256 fingerprint of the current :class:`ReplayConfig`."""
        return self._config.config_hash()

    def estimate_event_count(self) -> int:
        """
        Rough estimate of total replay events (trading days × bars_per_day).

        Useful for progress reporting.  Does NOT guarantee the exact count
        because actual snapshot availability varies.
        """
        from algo_trading.core.calendar import is_trading_day

        session_minutes = 375  # 09:15 – 15:30
        bars_per_day = max(
            1,
            session_minutes // max(1, self._config.snapshot_interval_seconds // 60),
        )

        count = 0
        d = self._config.start_date
        while d <= self._config.end_date:
            if is_trading_day(d):
                count += bars_per_day
            d += timedelta(days=1)
        return count

    # ── Private helpers ───────────────────────────────────────────────────────

    def _quality_ok(self, quality: str) -> bool:
        """Return True if *quality* meets or exceeds the configured threshold."""
        threshold = self._QUALITY_ORDER.get(
            self._config.data_quality_threshold, 1
        )
        return self._QUALITY_ORDER.get(quality, 0) >= threshold

    @staticmethod
    def _extract_event_time(snap: Any) -> datetime | None:
        """
        Extract the exchange-stamped event time from a snapshot.

        Supports both ChainSnapshot objects (attribute access) and raw dicts
        (key access).  Prefers ``event_timestamp`` over ``snapshot_timestamp``.
        """
        for attr in ("event_timestamp", "snapshot_timestamp"):
            # Attribute access (ChainSnapshot dataclass)
            val = getattr(snap, attr, None)
            if val is not None:
                return val
            # Dict access (raw provider output)
            if isinstance(snap, dict):
                val = snap.get(attr)
                if val is not None:
                    return val
        return None

    @staticmethod
    def _extract_sequence(snap: Any) -> int:
        """Extract the sequence_number from a snapshot; default 0."""
        val = getattr(snap, "sequence_number", None)
        if val is None and isinstance(snap, dict):
            val = snap.get("sequence_number", 0)
        return int(val) if val is not None else 0

    @staticmethod
    def _extract_quality(snap: Any) -> str:
        """Extract the data_quality string from a snapshot; default 'UNUSABLE'."""
        val = getattr(snap, "data_quality", None)
        if val is None and isinstance(snap, dict):
            val = snap.get("data_quality", "UNUSABLE")
        return str(val) if val is not None else "UNUSABLE"

    @staticmethod
    def _extract_expiry(snap: Any) -> Any:
        """Extract the expiry date from a snapshot for expiry-filter checks."""
        val = getattr(snap, "expiry", None)
        if val is None and isinstance(snap, dict):
            val = snap.get("expiry")
        return val
