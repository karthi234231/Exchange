"""
algo_trading/infra/event_store.py — Immutable append-only event log.

Implements an event-sourced architecture where ALL system state changes are
recorded as immutable events. This enables:
  1. Deterministic replay: reconstruct any system state from events alone
  2. Disaster recovery: given event log + broker order book = full state recovery
  3. Audit trail: SEBI requires 5-year trade audit trail
  4. Debug: replay any trading day to reproduce exact sequence of decisions

Event log is append-only JSONL (one JSON object per line). NEVER modify existing lines.

CQRS separation:
  Commands: actions the system takes (place_order, cancel_order, update_config)
  Events: things that happened (order_filled, order_rejected, position_opened)
  Queries: read-only (get_position, get_daily_pnl) — never written to event log

Disaster recovery procedure:
  1. At startup: read last N events from event_store
  2. Call StartupReconciler.reconcile(events, broker_positions)
  3. Resolve any mismatches before trading
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class EventType(str, Enum):
    """All event types emitted by the trading system."""

    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_TIMEOUT = "ORDER_TIMEOUT"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    DAILY_LIMIT_HIT = "DAILY_LIMIT_HIT"
    DRAWDOWN_LIMIT_HIT = "DRAWDOWN_LIMIT_HIT"
    DATA_QUALITY_HALT = "DATA_QUALITY_HALT"
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"
    BROKER_RECONNECT = "BROKER_RECONNECT"
    MODEL_PROMOTED = "MODEL_PROMOTED"
    MODEL_ROLLED_BACK = "MODEL_ROLLED_BACK"
    RISK_VETO = "RISK_VETO"
    ALERT_SENT = "ALERT_SENT"


@dataclass(frozen=True)
class TradingEvent:
    """Immutable record of a system state change.

    Attributes:
        event_id: UUID4 string; globally unique.
        event_type: Category of event.
        timestamp_ns: Wall-clock time in nanoseconds since Unix epoch (UTC).
        symbol: NSE instrument symbol (e.g., "RELIANCE", "NIFTY25JUNFUT").
        strategy_id: Identifier of the strategy that generated this event.
        payload: Arbitrary event-specific data (order details, fill prices, etc.).
        sequence_number: Monotonically increasing integer within this EventStore.
    """

    event_id: str
    event_type: EventType
    timestamp_ns: int
    symbol: str
    strategy_id: str
    payload: dict
    sequence_number: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradingEvent:
        """Deserialise from a JSON-decoded dict.

        Args:
            data: Dict produced by ``to_dict`` / read from JSONL file.

        Returns:
            Reconstructed TradingEvent.
        """
        data = dict(data)
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


class EventStore:
    """Append-only JSONL event log with rotation and replay support.

    Thread-safe: all writes are serialised through an internal lock.

    The on-disk layout is::

        <store_path>/events_<YYYYMMDD>_<seq>.jsonl   (rotated files)
        <store_path>/events_current.jsonl             (active file)

    Args:
        store_path: Directory where JSONL files are written.
        max_file_size_mb: Rotate to a new file when the active file
            exceeds this size. ASSUMED: 100 MB.
    """

    _CURRENT_FILE_NAME = "events_current.jsonl"

    def __init__(self, store_path: Path, max_file_size_mb: float = 100.0) -> None:
        self._dir = Path(store_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_file_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._sequence = self._load_last_sequence()
        self._current_path = self._dir / self._CURRENT_FILE_NAME
        log.info(
            "event_store.init",
            path=str(self._dir),
            last_sequence=self._sequence,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def append(
        self,
        event_type: EventType,
        symbol: str,
        strategy_id: str,
        payload: dict[str, Any],
        timestamp_ns: Optional[int] = None,
    ) -> TradingEvent:
        """Append a new event to the log.

        Assigns the next monotonically increasing sequence_number and writes
        one JSON line to the active JSONL file.  Rotates the file first if
        the active file has reached ``max_file_size_mb``.

        Args:
            event_type: Category of event.
            symbol: NSE instrument symbol.
            strategy_id: Identifier of the emitting strategy.
            payload: Arbitrary event-specific data.
            timestamp_ns: Override timestamp (nanoseconds UTC). Defaults to now.

        Returns:
            The persisted TradingEvent (immutable).
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        with self._lock:
            self._sequence += 1
            event = TradingEvent(
                event_id=str(uuid4()),
                event_type=event_type,
                timestamp_ns=timestamp_ns,
                symbol=symbol,
                strategy_id=strategy_id,
                payload=payload,
                sequence_number=self._sequence,
            )
            self._maybe_rotate()
            self._write_line(event)

        log.debug(
            "event_store.appended",
            seq=event.sequence_number,
            event_type=event_type.value,
            symbol=symbol,
        )
        return event

    def replay(self, from_sequence: int = 0) -> Iterator[TradingEvent]:
        """Yield all events with sequence_number >= from_sequence.

        Reads archived files in chronological order, then the current file.

        Args:
            from_sequence: First sequence number to include (0 = all events).

        Yields:
            TradingEvent in ascending sequence order.
        """
        for path in self._all_files_ordered():
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        event = TradingEvent.from_dict(data)
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        log.warning(
                            "event_store.corrupt_line",
                            path=str(path),
                            error=str(exc),
                        )
                        continue
                    if event.sequence_number >= from_sequence:
                        yield event

    def get_last_n_events(self, n: int) -> list[TradingEvent]:
        """Return the last n events from the log (most-recent last).

        Args:
            n: Maximum number of events to return.

        Returns:
            List of TradingEvent, at most n items, oldest first.
        """
        # Collect all events into a buffer; replace with deque to keep memory bounded
        from collections import deque
        buf: deque[TradingEvent] = deque(maxlen=n)
        for event in self.replay():
            buf.append(event)
        return list(buf)

    def reconstruct_positions(self, from_sequence: int = 0) -> dict[str, int]:
        """Replay fills and closes to reconstruct net qty per symbol.

        Processes ORDER_FILLED events (payload must contain ``qty`` signed:
        positive = BUY, negative = SELL) and POSITION_CLOSED events (sets qty
        to 0 for the symbol).

        Args:
            from_sequence: Start replay at this sequence number.

        Returns:
            Dict mapping symbol → net_qty (positive = long, negative = short).
        """
        positions: dict[str, int] = {}
        for event in self.replay(from_sequence=from_sequence):
            if event.event_type == EventType.ORDER_FILLED:
                qty = event.payload.get("qty", 0)
                positions[event.symbol] = positions.get(event.symbol, 0) + int(qty)
            elif event.event_type == EventType.POSITION_CLOSED:
                positions[event.symbol] = 0
        log.info(
            "event_store.positions_reconstructed",
            from_sequence=from_sequence,
            symbols=list(positions.keys()),
        )
        return positions

    def disaster_recovery_summary(self) -> dict[str, Any]:
        """Build a summary for the startup reconciliation process.

        Returns:
            Dict with keys:
              - ``last_event_time``: ISO-8601 UTC string of latest event, or None.
              - ``total_events``: Total number of events replayed.
              - ``reconstructed_positions``: {symbol: net_qty}.
              - ``open_orders``: {order_id: payload} for placed but not
                filled/cancelled/rejected orders.
        """
        total = 0
        last_ts_ns: Optional[int] = None
        open_orders: dict[str, dict] = {}

        for event in self.replay():
            total += 1
            if last_ts_ns is None or event.timestamp_ns > last_ts_ns:
                last_ts_ns = event.timestamp_ns

            oid = event.payload.get("order_id")
            if event.event_type == EventType.ORDER_PLACED and oid:
                open_orders[oid] = event.payload
            elif event.event_type in (
                EventType.ORDER_FILLED,
                EventType.ORDER_CANCELLED,
                EventType.ORDER_REJECTED,
                EventType.ORDER_TIMEOUT,
            ):
                if oid:
                    open_orders.pop(oid, None)

        last_event_time: Optional[str] = None
        if last_ts_ns is not None:
            last_event_time = datetime.fromtimestamp(
                last_ts_ns / 1e9, tz=timezone.utc
            ).isoformat()

        positions = self.reconstruct_positions()
        return {
            "last_event_time": last_event_time,
            "total_events": total,
            "reconstructed_positions": positions,
            "open_orders": open_orders,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_last_sequence(self) -> int:
        """Scan all JSONL files to find the highest sequence_number."""
        last_seq = 0
        for path in self._all_files_ordered():
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        seq = int(data.get("sequence_number", 0))
                        if seq > last_seq:
                            last_seq = seq
                    except (json.JSONDecodeError, ValueError):
                        continue
        return last_seq

    def _all_files_ordered(self) -> list[Path]:
        """Return all JSONL paths in chronological write order.

        Archived files sort lexicographically by their date+seq suffix;
        the current file is always appended last.
        """
        archived = sorted(
            p for p in self._dir.glob("events_*.jsonl")
            if p.name != self._CURRENT_FILE_NAME
        )
        current = self._dir / self._CURRENT_FILE_NAME
        return archived + [current]

    def _maybe_rotate(self) -> None:
        """Rotate current file if it has exceeded max_file_size_mb.

        Must be called while holding ``self._lock``.
        """
        if not self._current_path.exists():
            return
        if self._current_path.stat().st_size < self._max_bytes:
            return

        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"events_{date_str}_{self._sequence}.jsonl"
        archive_path = self._dir / archive_name
        self._current_path.rename(archive_path)
        log.info(
            "event_store.rotated",
            archive=archive_name,
            sequence=self._sequence,
        )

    def _write_line(self, event: TradingEvent) -> None:
        """Append a single JSON line to the current file (atomic open+write+close).

        Must be called while holding ``self._lock``.
        """
        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with open(self._current_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
