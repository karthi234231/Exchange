"""
algo_trading/market_data/snapshot_archiver.py — Writes ChainSnapshot objects to a
SnapshotStore at configurable frequency.

Key rules (design Section 9.2)
-------------------------------
- Every snapshot tagged with monotonic timestamp + sequence number + schema version
  before writing.
- Normal market hours: snapshot every 60 seconds (configurable).
- Last 30 min of expiry day: snapshot every 30 seconds (configurable).
- All corrections made by ChainNormalizer are already logged before archiving.

Replay clock semantics (design Section 9.2)
-------------------------------------------
- ``snapshot_timestamp`` carries wall-clock time of ingestion.
- ``event_timestamp`` carries the exchange timestamp.
- All replay must use event time, not ingestion time.
  Ingestion latency creates false causality if wall-clock time is used.

Backend decoupling
------------------
The archiver writes to any ``SnapshotStore`` ABC from
``algo_trading.fo.interfaces``.  The store's ``write_snapshot`` method
receives:
    - ``timestamp``      : the snapshot's wall-clock timestamp (UTC)
    - ``chain_data``     : the full serialised snapshot dict
    - ``schema_version`` : ``CURRENT_SCHEMA_VERSION``

A concrete in-memory store is provided here for testing without QuestDB.
The real QuestDB store is a separate implementation satisfying the same ABC.
"""
from __future__ import annotations

import dataclasses
import itertools
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import structlog

from algo_trading.fo.interfaces import SnapshotStore
from algo_trading.market_data.chain_snapshot import (
    CURRENT_SCHEMA_VERSION,
    ChainSnapshot,
)

log = structlog.get_logger(__name__)

# NSE F&O session close times (IST) used to identify the expiry-day final window.
# ASSUMED: market closes at 15:30 IST (standard NSE equity derivatives close).
# Verify against live NSE circular before use in production.
_NSE_FO_CLOSE_HOUR_IST: int = 15
_NSE_FO_CLOSE_MINUTE_IST: int = 30
_EXPIRY_FINAL_WINDOW_MINUTES: int = 30  # last N minutes before close on expiry day


def _snapshot_to_dict(snapshot: ChainSnapshot) -> dict:
    """Serialise a ``ChainSnapshot`` to a plain dict for the store.

    ``OptionQuote`` objects are serialised with ``dataclasses.asdict``.
    Tuple keys in ``quotes`` are stringified as ``"expiry|strike|opt_type"``
    so they survive JSON / msgpack round-trips.

    Args:
        snapshot: Snapshot to serialise.

    Returns:
        Plain dict with no ChainSnapshot-specific types.
    """
    raw = dataclasses.asdict(snapshot)
    # quotes dict has tuple keys — convert to string keys for storage portability.
    raw["quotes"] = {
        f"{expiry}|{strike}|{opt_type}": quote_dict
        for (expiry, strike, opt_type), quote_dict in (
            (k, v) for k, v in zip(snapshot.quotes.keys(), raw["quotes"].values())
        )
    }
    return raw


class InMemorySnapshotStore(SnapshotStore):
    """Simple in-memory store for testing.

    Thread-safe.  Not suitable for production: data is lost on process exit.
    Keeps ALL snapshots in memory; no eviction policy.

    Implements the full ``SnapshotStore`` ABC from ``algo_trading.fo.interfaces``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # List of (snapshot_id, timestamp, schema_version, chain_data_dict)
        self._store: list[tuple[str, datetime, int, dict]] = []

    # ------------------------------------------------------------------
    # SnapshotStore ABC
    # ------------------------------------------------------------------

    def write_snapshot(
        self,
        timestamp: datetime,
        chain_data: dict,
        schema_version: int,
    ) -> str:
        """Persist a snapshot in memory and return its ID.

        Args:
            timestamp: Wall-clock UTC timestamp of the snapshot.
            chain_data: Serialised snapshot dict.
            schema_version: Schema version tag for forward-compat checks.

        Returns:
            UUID string snapshot ID.
        """
        snapshot_id = str(uuid.uuid4())
        with self._lock:
            self._store.append((snapshot_id, timestamp, schema_version, chain_data))
        log.debug(
            "in_memory_store.write_snapshot",
            snapshot_id=snapshot_id,
            symbol=chain_data.get("symbol"),
        )
        return snapshot_id

    def read_snapshot(self, timestamp: datetime) -> dict:
        """Return the stored snapshot closest to and not after *timestamp*.

        Args:
            timestamp: Target timestamp.

        Returns:
            Snapshot dict enriched with ``"_snapshot_meta"``.

        Raises:
            KeyError: If no snapshot exists at or before *timestamp*.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        with self._lock:
            candidates = [
                (sid, ts, sv, data)
                for sid, ts, sv, data in self._store
                if (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)) <= timestamp
            ]
        if not candidates:
            raise KeyError(f"No snapshot at or before {timestamp}")
        sid, ts, sv, data = max(candidates, key=lambda x: x[1])
        return {**data, "_snapshot_meta": {"timestamp": ts, "schema_version": sv, "snapshot_id": sid}}

    def read_range(
        self,
        start_ts: datetime,
        end_ts: datetime,
    ) -> Iterator[dict]:
        """Yield snapshots in [start_ts, end_ts) in chronological order.

        Args:
            start_ts: Inclusive start.
            end_ts: Exclusive end.

        Yields:
            Snapshot dicts enriched with ``"_snapshot_meta"``.

        Raises:
            ValueError: If start_ts >= end_ts.
        """
        if start_ts >= end_ts:
            raise ValueError(f"start_ts {start_ts} must be < end_ts {end_ts}")

        def _aware(dt: datetime) -> datetime:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        start_ts = _aware(start_ts)
        end_ts = _aware(end_ts)

        with self._lock:
            matches = [
                (sid, ts, sv, data)
                for sid, ts, sv, data in self._store
                if start_ts <= _aware(ts) < end_ts
            ]
        for sid, ts, sv, data in sorted(matches, key=lambda x: x[1]):
            yield {**data, "_snapshot_meta": {"timestamp": ts, "schema_version": sv, "snapshot_id": sid}}

    def get_schema_version(self, snapshot_id: str) -> int:
        """Return the schema version of the snapshot with *snapshot_id*.

        Args:
            snapshot_id: Opaque ID returned by ``write_snapshot``.

        Returns:
            Integer schema version.

        Raises:
            KeyError: If *snapshot_id* is not found.
        """
        with self._lock:
            for sid, _, sv, _ in self._store:
                if sid == snapshot_id:
                    return sv
        raise KeyError(f"snapshot_id {snapshot_id!r} not found")

    def list_gaps(
        self,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Return time gaps with no stored snapshots in [start_ts, end_ts].

        The in-memory store does not enforce a density requirement, so this
        method always returns an empty list (no gap detection without a
        configured expected interval).

        Args:
            start_ts: Range start (inclusive).
            end_ts: Range end (inclusive).

        Returns:
            Empty list — gap detection requires a configurable interval.

        Raises:
            ValueError: If start_ts >= end_ts.
        """
        if start_ts >= end_ts:
            raise ValueError(f"start_ts {start_ts} must be < end_ts {end_ts}")
        return []


class SnapshotArchiver:
    """Writes ``ChainSnapshot`` objects to a ``SnapshotStore`` at configurable frequency.

    The archiver normalises each snapshot via ``ChainNormalizer`` before writing,
    then tags it with a wall-clock timestamp, a monotonic sequence number, and
    the current schema version.

    Replay clock semantics
    ----------------------
    The archiver sets ``snapshot_timestamp`` to wall-clock time and preserves
    ``event_timestamp`` from the incoming snapshot unchanged.  Callers doing
    historical replay MUST use ``event_timestamp``, not ``snapshot_timestamp``.
    Ingestion latency creates false causality when wall-clock time is used.

    Parameters
    ----------
    store:
        Backend storage implementation satisfying ``SnapshotStore`` ABC.
        Defaults to ``InMemorySnapshotStore`` when ``None`` (useful for unit
        tests without a live database).
    normalizer:
        ``ChainNormalizer`` instance.  When ``None`` the snapshot is written
        as-is (no normalisation).  Pass an explicit normalizer in production.
    snapshot_interval_seconds:
        Normal-session archiving interval in seconds.  Default: 60.
    expiry_day_interval_seconds:
        Archiving interval during the last ``_EXPIRY_FINAL_WINDOW_MINUTES``
        minutes on an expiry day.  Default: 30.
    """

    def __init__(
        self,
        store: SnapshotStore | None = None,
        normalizer: Any | None = None,   # ChainNormalizer — typed Any to avoid circular import
        snapshot_interval_seconds: int = 60,
        expiry_day_interval_seconds: int = 30,
    ) -> None:
        self._store: SnapshotStore = store if store is not None else InMemorySnapshotStore()
        self._normalizer = normalizer
        self._snapshot_interval = snapshot_interval_seconds
        self._expiry_interval = expiry_day_interval_seconds

        # Monotonic sequence counter shared across all archives in this process.
        self._sequence_counter = itertools.count(start=1)
        self._counter_lock = threading.Lock()

        # Background thread state.
        self._bg_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        log.info(
            "snapshot_archiver.init",
            store_type=type(self._store).__name__,
            snapshot_interval_seconds=snapshot_interval_seconds,
            expiry_day_interval_seconds=expiry_day_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def archive(self, snapshot: ChainSnapshot) -> str:
        """Normalise, tag, and write a snapshot.

        Tags applied before writing:
            - ``snapshot_timestamp``: current wall-clock UTC time.
            - ``sequence_number``: next value from monotonic counter.
            - ``schema_version``: ``CURRENT_SCHEMA_VERSION``.

        The ``event_timestamp`` from the incoming snapshot is preserved
        unchanged for replay correctness (design Section 9.2).

        Args:
            snapshot: Raw or pre-normalised ``ChainSnapshot`` from the feed.

        Returns:
            Unique ``snapshot_id`` string assigned by the store.
        """
        # Normalise if a normalizer is configured.
        if self._normalizer is not None:
            snapshot = self._normalizer.normalize(snapshot)

        # Assign monotonic sequence number.
        with self._counter_lock:
            seq = next(self._sequence_counter)

        wall_clock_now = datetime.now(tz=timezone.utc)

        # Build tagged snapshot (replace mutable fields; keep event_timestamp intact).
        tagged = ChainSnapshot(
            symbol=snapshot.symbol,
            snapshot_timestamp=wall_clock_now,          # wall clock — do NOT use for replay
            event_timestamp=snapshot.event_timestamp,   # exchange time — USE for replay
            underlying_price=snapshot.underlying_price,
            underlying_bid=snapshot.underlying_bid,
            underlying_ask=snapshot.underlying_ask,
            schema_version=CURRENT_SCHEMA_VERSION,
            sequence_number=seq,
            data_quality=snapshot.data_quality,
            quotes=snapshot.quotes,
            metadata={
                **snapshot.metadata,
                "archiver_schema_version": CURRENT_SCHEMA_VERSION,
                "archiver_sequence": seq,
            },
        )

        # Serialise and write via the SnapshotStore ABC.
        chain_data = _snapshot_to_dict(tagged)
        snapshot_id = self._store.write_snapshot(
            timestamp=wall_clock_now,
            chain_data=chain_data,
            schema_version=CURRENT_SCHEMA_VERSION,
        )

        log.debug(
            "snapshot_archiver.archived",
            snapshot_id=snapshot_id,
            symbol=tagged.symbol,
            sequence_number=seq,
            data_quality=tagged.data_quality,
            quote_count=len(tagged.quotes),
        )
        return snapshot_id

    def start_background_archiving(self, chain_feed: Any) -> None:
        """Start a background thread that polls ``chain_feed`` and archives snapshots.

        The interval used depends on whether the current time falls within the
        expiry-day final window (see ``_is_expiry_day_final_window``).

        ``chain_feed`` must expose a ``get_snapshot() -> ChainSnapshot | None``
        method.  The archiver calls this method on every tick.

        Args:
            chain_feed: Live chain feed object.  Must implement
                ``get_snapshot() -> ChainSnapshot | None``.

        Raises:
            RuntimeError: If a background thread is already running.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            raise RuntimeError("Background archiving already running.  Call stop() first.")

        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._background_loop,
            args=(chain_feed,),
            daemon=True,
            name="SnapshotArchiver-bg",
        )
        self._bg_thread.start()
        log.info("snapshot_archiver.background_started")

    def stop(self) -> None:
        """Gracefully stop the background archiving thread.

        Signals the background thread to exit and waits for it to finish
        (up to 10 seconds).  Safe to call even if no background thread is running.
        """
        self._stop_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=10.0)
            if self._bg_thread.is_alive():
                log.warning("snapshot_archiver.stop_timeout", thread=self._bg_thread.name)
            else:
                log.info("snapshot_archiver.stopped")
        self._bg_thread = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _background_loop(self, chain_feed: Any) -> None:
        """Main body of the background archiving thread.

        Sleeps for the configured interval (expiry-day or normal) between
        each archive call.  Exits when ``_stop_event`` is set.

        Args:
            chain_feed: Feed object with ``get_snapshot() -> ChainSnapshot | None``.
        """
        log.info("snapshot_archiver.background_loop_started")
        while not self._stop_event.is_set():
            now = datetime.now(tz=timezone.utc)
            interval = (
                self._expiry_interval
                if self._is_expiry_day_final_window(now)
                else self._snapshot_interval
            )

            try:
                snapshot: ChainSnapshot | None = chain_feed.get_snapshot()
                if snapshot is not None:
                    self.archive(snapshot)
            except Exception:
                log.exception("snapshot_archiver.background_loop_error")

            # Wait for the interval, but wake early if stop is requested.
            self._stop_event.wait(timeout=float(interval))

        log.info("snapshot_archiver.background_loop_stopped")

    def _is_expiry_day_final_window(self, ts: datetime) -> bool:
        """Return ``True`` if *ts* falls within the last 30 min of an expiry day.

        ASSUMED: NSE F&O session closes at 15:30 IST on expiry days.
        The 30-minute window therefore runs from 15:00 to 15:30 IST.

        This implementation converts UTC to IST (+05:30) for the check.
        It does NOT know which dates are expiry dates — it only checks
        whether the time-of-day falls in the 15:00–15:30 IST window.
        A more sophisticated caller can subclass and override this method
        to also gate on whether today is actually an expiry date.

        Args:
            ts: Timestamp to check (timezone-aware or naive UTC).

        Returns:
            ``True`` if the time portion (IST) falls in the expiry final window.
        """
        from datetime import timedelta

        IST_OFFSET = timedelta(hours=5, minutes=30)
        if ts.tzinfo is not None:
            ist_ts = ts.astimezone(timezone.utc).replace(tzinfo=None) + IST_OFFSET
        else:
            # Assume naive ts is UTC.
            ist_ts = ts + IST_OFFSET

        close_minute = _NSE_FO_CLOSE_HOUR_IST * 60 + _NSE_FO_CLOSE_MINUTE_IST
        current_minute = ist_ts.hour * 60 + ist_ts.minute
        window_start = close_minute - _EXPIRY_FINAL_WINDOW_MINUTES

        return window_start <= current_minute < close_minute
