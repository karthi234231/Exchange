"""
execution/order_fsm.py — Order lifecycle state machine.

Manages the state of every order from creation to terminal state.
Persists every transition to QuestDB (orders table).

States (from ACCEPTANCE_CRITERIA.md):
  PENDING       → Created locally, not yet sent to broker
  SENT          → Submitted to broker REST API, awaiting ACK
  ACKED         → Broker acknowledged, awaiting fill or cancel
  PARTIAL       → Partially filled, order still open at broker
  FILLED        → Fully filled (terminal)
  CANCEL_PENDING→ Cancel request sent to broker
  CANCELLED     → Cancel confirmed (terminal)
  REJECTED      → Broker rejected the order (terminal)
  HALTED        → Unknown state after timeout — requires manual intervention (terminal)

Key invariants:
  - Duplicate ACK for same order_id is idempotent (no state change, no error)
  - place_order() is idempotent: same client_order_id cannot create two orders
  - Late fill after CANCELLED → logged, position reconciled, NOT silently dropped
  - Partial fill then cancel → remaining qty tracked correctly
  - Every transition persisted to QuestDB before the transition completes
  - Broker disconnect mid-order → state preserved in DB, reconciled on reconnect [SURV-6]
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import structlog

from algo_trading.data.broker_base import Fill, Order, OrderUpdate
from algo_trading.core.calendar import IST, now_ist

log = structlog.get_logger(__name__)

_WAL_PATH = Path("state/orders_wal.jsonl")
_PERSIST_FAILURE_THRESHOLD = 5


class PersistenceHaltedError(Exception):
    """Raised when persistence has failed too many times and trading is halted."""


class OrderState(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    ACKED = "ACKED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    HALTED = "HALTED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.HALTED,
        )

    @property
    def is_active(self) -> bool:
        return not self.is_terminal


class OrderEvent(str, Enum):
    SEND = "SEND"               # Caller: submitting to broker
    ACK = "ACK"                 # Broker: order accepted
    REJECT = "REJECT"           # Broker: order rejected
    FILL = "FILL"               # Broker: full fill received
    PARTIAL_FILL = "PARTIAL_FILL"  # Broker: partial fill received
    CANCEL_REQUEST = "CANCEL_REQUEST"  # Caller: requesting cancellation
    CANCEL_CONFIRM = "CANCEL_CONFIRM"  # Broker: cancellation confirmed
    TIMEOUT = "TIMEOUT"         # Caller: order exceeded stale timeout → HALTED


# Valid transitions: (current_state, event) → new_state
# fmt: off
_TRANSITIONS: dict[tuple[OrderState, OrderEvent], OrderState] = {
    (OrderState.PENDING,        OrderEvent.SEND):            OrderState.SENT,
    (OrderState.PENDING,        OrderEvent.REJECT):          OrderState.REJECTED,   # rate-limit / pre-send block
    (OrderState.SENT,           OrderEvent.ACK):             OrderState.ACKED,
    (OrderState.SENT,           OrderEvent.REJECT):          OrderState.REJECTED,
    (OrderState.SENT,           OrderEvent.TIMEOUT):         OrderState.HALTED,
    # P2-FIX-99: Some brokers send FILL/PARTIAL_FILL before ACK — treat as implicit ACK
    (OrderState.SENT,           OrderEvent.FILL):            OrderState.FILLED,
    (OrderState.SENT,           OrderEvent.PARTIAL_FILL):    OrderState.PARTIAL,
    (OrderState.ACKED,          OrderEvent.FILL):            OrderState.FILLED,
    (OrderState.ACKED,          OrderEvent.PARTIAL_FILL):    OrderState.PARTIAL,
    (OrderState.ACKED,          OrderEvent.CANCEL_REQUEST):  OrderState.CANCEL_PENDING,
    (OrderState.ACKED,          OrderEvent.TIMEOUT):         OrderState.HALTED,
    (OrderState.PARTIAL,        OrderEvent.FILL):            OrderState.FILLED,
    (OrderState.PARTIAL,        OrderEvent.PARTIAL_FILL):    OrderState.PARTIAL,  # more partial fills
    (OrderState.PARTIAL,        OrderEvent.CANCEL_REQUEST):  OrderState.CANCEL_PENDING,
    (OrderState.PARTIAL,        OrderEvent.TIMEOUT):         OrderState.HALTED,
    (OrderState.CANCEL_PENDING, OrderEvent.CANCEL_CONFIRM):  OrderState.CANCELLED,
    (OrderState.CANCEL_PENDING, OrderEvent.FILL):            OrderState.FILLED,   # race: fill beats cancel
    (OrderState.CANCEL_PENDING, OrderEvent.PARTIAL_FILL):    OrderState.PARTIAL,  # race: partial beats cancel
    (OrderState.CANCEL_PENDING, OrderEvent.TIMEOUT):         OrderState.HALTED,
}

# Idempotent events: same event in same state → no-op (return current state)
_IDEMPOTENT: set[tuple[OrderState, OrderEvent]] = {
    (OrderState.ACKED,     OrderEvent.ACK),              # Duplicate ACK
    (OrderState.CANCELLED, OrderEvent.CANCEL_CONFIRM),   # Duplicate cancel confirm
    (OrderState.FILLED,    OrderEvent.FILL),             # Duplicate fill (network retry)
    (OrderState.FILLED,    OrderEvent.PARTIAL_FILL),     # P2-FIX-99: delayed partial after full fill
}
# fmt: on


@dataclass
class OrderRecord:
    """Full lifecycle record for one order."""
    order: Order
    state: OrderState = OrderState.PENDING
    exchange_order_id: str = ""
    filled_quantity: int = 0
    fills: list[Fill] = field(default_factory=list)
    reject_reason: str = ""
    created_at: datetime = field(default_factory=now_ist)
    updated_at: datetime = field(default_factory=now_ist)
    sent_at: Optional[datetime] = None
    fill_price_avg: Decimal = Decimal(0)  # VWAP of all fills

    @property
    def remaining_quantity(self) -> int:
        return self.order.quantity - self.filled_quantity

    def add_fill(self, fill: Fill) -> None:
        """Record a fill and update average fill price."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        # VWAP of fills
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.fill_price_avg = total_value / self.filled_quantity if self.filled_quantity else Decimal(0)


class FillCallback:
    """Called when a fill occurs (for position + P&L update)."""
    def on_fill(self, record: OrderRecord, fill: Fill) -> None: ...


class OrderFSM:
    """
    Thread-safe order state machine.

    All transitions are validated before being applied.
    Every transition is persisted to QuestDB before the lock is released.
    Duplicate ACKs are silently accepted (idempotent).

    Usage:
        fsm = OrderFSM(store=questdb_store)
        record = fsm.create(order)
        fsm.transition(record.order.client_order_id, OrderEvent.SEND, exchange_order_id="X123")
        # ... later, from WebSocket callback:
        fsm.on_broker_update(order_update)
    """

    def __init__(self, store: Optional[object] = None) -> None:  # store: QuestDBStore
        self._orders: dict[str, OrderRecord] = {}  # client_order_id → record
        self._lock = threading.Lock()
        self._store = store  # QuestDBStore — injected to avoid circular import
        self._fill_callbacks: list[FillCallback] = []
        self._persist_failure_count: int = 0
        self._persistence_halted: bool = False

    def register_fill_callback(self, cb: FillCallback) -> None:
        self._fill_callbacks.append(cb)

    def replay_wal(self) -> int:
        """Re-submit WAL entries to QuestDB after reconnect.

        Reads ``state/orders_wal.jsonl``, attempts to write each entry via
        ``self._store.write_order_event()``, and truncates the file on full
        success.  Partial success leaves the file intact so the next call
        retries all entries (idempotent inserts on the QuestDB side).

        Returns the number of entries successfully replayed.
        """
        if not _WAL_PATH.exists():
            return 0
        if self._store is None:
            log.warning("order_fsm.wal_replay_skipped", reason="no store injected")
            return 0

        try:
            lines = _WAL_PATH.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.error("order_fsm.wal_read_failed", error=str(exc))
            return 0

        replayed = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                self._store.write_order_event(  # type: ignore[union-attr]
                    ts=entry.get("timestamp", ""),
                    client_order_id=entry.get("order_id", ""),
                    exchange_order_id=entry.get("exchange_order_id", ""),
                    symbol=entry.get("symbol", ""),
                    side="",
                    quantity=0,
                    price=0.0,
                    order_type="",
                    state=entry.get("state", ""),
                    strategy_id="",
                    reject_reason=entry.get("error", ""),
                )
                replayed += 1
            except Exception as exc:
                log.error("order_fsm.wal_replay_entry_failed", error=str(exc), entry=line[:120])
                return replayed  # stop on first failure; retry next time

        # All entries replayed successfully — clear the WAL
        try:
            _WAL_PATH.unlink()
            log.info("order_fsm.wal_replayed", entries=replayed)
        except OSError as exc:
            log.error("order_fsm.wal_clear_failed", error=str(exc))

        # Reset circuit-breaker if we successfully replayed
        self._persist_failure_count = 0
        self._persistence_halted = False
        return replayed

    def create(self, order: Order) -> OrderRecord:
        """
        Register a new order. Idempotent: returns existing record if ID already exists.
        Raises ValueError if a different order with the same client_order_id exists.
        """
        with self._lock:
            existing = self._orders.get(order.client_order_id)
            if existing is not None:
                if existing.order.symbol != order.symbol or existing.order.quantity != order.quantity:
                    raise ValueError(
                        f"client_order_id {order.client_order_id!r} already exists "
                        f"for a different order. place_order() is not idempotent."
                    )
                return existing

            record = OrderRecord(order=order)
            self._orders[order.client_order_id] = record
            self._persist(record)
            log.info(
                "order_fsm.created",
                client_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.quantity,
                price=str(order.price),
                type=order.order_type,
                decision_id=order.decision_id,    # P2-FIX-89: correlation ID
                model_version=order.model_version,  # P2-FIX-57
            )
            return record

    def transition(
        self,
        client_order_id: str,
        event: OrderEvent,
        exchange_order_id: str = "",
        filled_quantity: int = 0,
        fill_price: Decimal = Decimal(0),
        reject_reason: str = "",
    ) -> Optional[OrderRecord]:
        """
        Apply a state transition. Returns the updated record.
        Returns None if client_order_id is not found.
        Raises TransitionError on invalid transition.
        """
        # P2-FIX-99: initialized before lock so callback notification works after with-block
        fill_to_notify: Optional[Fill] = None
        record_to_notify: Optional[OrderRecord] = None

        with self._lock:
            # Circuit-breaker: block all transitions when persistence is halted
            if self._persistence_halted:
                log.critical(
                    "order_fsm.persistence_halted.transition_blocked",
                    client_id=client_order_id,
                    fsm_event=event.value,
                )
                raise PersistenceHaltedError(
                    f"Persistence halted after {_PERSIST_FAILURE_THRESHOLD} consecutive failures — "
                    f"manual intervention required"
                )

            record = self._orders.get(client_order_id)
            if record is None:
                log.warning("order_fsm.unknown_order", client_id=client_order_id, fsm_event=event.value)
                return None

            current = record.state
            key = (current, event)

            # P2-FIX-99: Late fill after CANCELLED — broker filled the order after we sent cancel.
            # State stays CANCELLED for audit integrity, but fill callbacks MUST fire so the
            # position tracker can reconcile the unexpected open position.
            if current == OrderState.CANCELLED and event in (OrderEvent.FILL, OrderEvent.PARTIAL_FILL):
                if fill_price > 0 and filled_quantity > 0:
                    late_fill = Fill(
                        order_id=client_order_id,
                        exchange_order_id=record.exchange_order_id,
                        symbol=record.order.symbol,
                        side=record.order.side,
                        quantity=filled_quantity,
                        price=fill_price,
                        timestamp=now_ist(),
                    )
                    record.add_fill(late_fill)
                    fill_to_notify = late_fill
                    record_to_notify = record
                log.critical(
                    "order_fsm.late_fill_after_cancel",
                    client_id=client_order_id,
                    exchange_id=record.exchange_order_id,
                    symbol=record.order.symbol,
                    filled_qty=filled_quantity,
                    fill_price=str(fill_price),
                    decision_id=record.order.decision_id,
                    note="POSITION MISMATCH: fill received after cancel confirm — reconcile positions",
                )
                # Fall through to notify fill callbacks outside the lock

            # Idempotent events — no-op
            elif key in _IDEMPOTENT:
                log.debug(
                    "order_fsm.idempotent_event",
                    client_id=client_order_id,
                    state=current.value,
                    fsm_event=event.value,
                )
                return record

            # Terminal state guard (all other events in terminal states)
            elif current.is_terminal:
                log.warning(
                    "order_fsm.event_in_terminal_state",
                    client_id=client_order_id,
                    state=current.value,
                    fsm_event=event.value,
                )
                return record

            else:
                new_state = _TRANSITIONS.get(key)
                if new_state is None:
                    raise TransitionError(
                        f"Invalid transition: {current.value} + {event.value} "
                        f"for order {client_order_id}"
                    )

                # Apply transition
                old_state = record.state
                record.state = new_state
                record.updated_at = now_ist()

                if exchange_order_id:
                    record.exchange_order_id = exchange_order_id

                if event == OrderEvent.SEND:
                    record.sent_at = now_ist()

                if event in (OrderEvent.FILL, OrderEvent.PARTIAL_FILL) and fill_price > 0:
                    fill = Fill(
                        order_id=client_order_id,
                        exchange_order_id=record.exchange_order_id,
                        symbol=record.order.symbol,
                        side=record.order.side,
                        quantity=filled_quantity,
                        price=fill_price,
                        timestamp=now_ist(),
                    )
                    record.add_fill(fill)
                    fill_to_notify = fill
                    record_to_notify = record

                if reject_reason:
                    record.reject_reason = reject_reason

                self._persist(record)

                log.info(
                    "order_fsm.transition",
                    client_id=client_order_id,
                    exchange_id=record.exchange_order_id,
                    symbol=record.order.symbol,
                    from_state=old_state.value,
                    to_state=new_state.value,
                    fsm_event=event.value,
                    filled_qty=filled_quantity,
                    fill_price=str(fill_price),
                    reject_reason=reject_reason,
                    decision_id=record.order.decision_id,  # P2-FIX-89: correlation ID
                )

        # Notify fill callbacks outside the lock
        if fill_to_notify and record_to_notify:
            for cb in self._fill_callbacks:
                try:
                    cb.on_fill(record_to_notify, fill_to_notify)
                except Exception as exc:
                    log.error("order_fsm.fill_callback_error", error=str(exc))

        return record

    def on_broker_update(self, update: OrderUpdate) -> Optional[OrderRecord]:
        """
        Handle a broker WebSocket order update.
        Routes the update to the appropriate transition.
        """
        event_map: dict[str, OrderEvent] = {
            "ACK": OrderEvent.ACK,
            "REJECT": OrderEvent.REJECT,
            "FILL": OrderEvent.FILL,
            "PARTIAL_FILL": OrderEvent.PARTIAL_FILL,
            "CANCEL_CONFIRM": OrderEvent.CANCEL_CONFIRM,
        }
        event = event_map.get(update.event)
        if event is None:
            log.warning("order_fsm.unknown_broker_event", event=update.event)
            return None

        # Map exchange_order_id back to client_order_id if needed
        client_id = update.client_order_id
        if not client_id and update.exchange_order_id:
            with self._lock:
                client_id = next(
                    (cid for cid, r in self._orders.items()
                     if r.exchange_order_id == update.exchange_order_id),
                    "",
                )
        if not client_id:
            log.warning("order_fsm.cannot_map_exchange_id", exchange_id=update.exchange_order_id)
            return None

        return self.transition(
            client_id,
            event,
            exchange_order_id=update.exchange_order_id,
            filled_quantity=update.filled_quantity,
            fill_price=update.fill_price,
            reject_reason=update.reject_reason,
        )

    def get_record(self, client_order_id: str) -> Optional[OrderRecord]:
        with self._lock:
            return self._orders.get(client_order_id)

    def record_exchange_id(self, client_order_id: str, exchange_order_id: str) -> None:
        """
        Store the exchange_order_id returned by the broker REST API immediately
        after order submission, before the WebSocket ACK arrives.

        This ensures cancel_order() can find the exchange id even if it is called
        in the window between REST submission and the first WS update (e.g., during
        a stale-order timeout that fires very quickly). [P0-FIX-3]
        """
        if not exchange_order_id:
            return
        with self._lock:
            record = self._orders.get(client_order_id)
            if record is not None and not record.exchange_order_id:
                record.exchange_order_id = exchange_order_id
                log.debug(
                    "order_fsm.exchange_id_recorded",
                    client_id=client_order_id,
                    exchange_id=exchange_order_id,
                )

    def get_active_orders(self) -> list[OrderRecord]:
        with self._lock:
            return [r for r in self._orders.values() if r.state.is_active]

    def get_all_records(self) -> list[OrderRecord]:
        with self._lock:
            return list(self._orders.values())

    def _persist(self, record: OrderRecord) -> None:
        """Persist the current order state to QuestDB. Called with lock held."""
        if self._store is None:
            return
        try:
            self._store.write_order_event(  # type: ignore[union-attr]
                ts=record.updated_at,
                client_order_id=record.order.client_order_id,
                exchange_order_id=record.exchange_order_id,
                symbol=record.order.symbol,
                side=record.order.side,
                quantity=record.order.quantity,
                price=record.order.price,
                order_type=record.order.order_type,
                state=record.state.value,
                strategy_id=record.order.strategy_id,
                reject_reason=record.reject_reason,
            )
            # Successful persist — reset failure counter
            self._persist_failure_count = 0
        except Exception as exc:
            # Fix A: Log at ERROR and write to WAL file for durability
            log.error(
                "order_fsm.persist_failed",
                error=str(exc),
                client_id=record.order.client_order_id,
            )
            try:
                _WAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                wal_entry = json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "order_id": record.order.client_order_id,
                    "exchange_order_id": record.exchange_order_id,
                    "symbol": record.order.symbol,
                    "state": record.state.value,
                    "event": "persist_failure",
                    "error": str(exc),
                })
                with _WAL_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(wal_entry + "\n")
            except Exception as wal_exc:
                log.error("order_fsm.wal_write_failed", error=str(wal_exc))

            # Fix B: Circuit-breaker on consecutive persist failures
            self._persist_failure_count += 1
            if self._persist_failure_count >= _PERSIST_FAILURE_THRESHOLD:
                self._persistence_halted = True
                log.critical(
                    "order_fsm.persistence_halted",
                    failure_count=self._persist_failure_count,
                    message="Trading halted after too many consecutive persistence failures",
                )


class TransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
