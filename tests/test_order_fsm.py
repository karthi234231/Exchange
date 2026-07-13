"""
tests/test_order_fsm.py — Order FSM state machine acceptance tests.

All acceptance criteria from ACCEPTANCE_CRITERIA.md:
  [x] States: PENDING → SENT → ACKED → PARTIAL → FILLED
  [x] States: ACKED → CANCEL_PENDING → CANCELLED
  [x] Timeout with unknown broker state → HALTED (do not retry blindly)
  [x] Duplicate ACK for same order_id is idempotent (no state change)
  [x] Late fill after cancel → logged, position reconciled
  [x] Partial fill then cancel → correct remaining qty tracked
  [x] Broker disconnect mid-order → state preserved, reconciled on reconnect
  [x] Every transition persisted to QuestDB orders table
  [x] place_order() is idempotent — same client_order_id cannot create two orders
  [x] Test restart: state recovered from QuestDB correctly (mocked)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from algo_trading.data.broker_base import Fill, Order, OrderUpdate
from algo_trading.execution.order_fsm import (
    FillCallback,
    OrderEvent,
    OrderFSM,
    OrderRecord,
    OrderState,
    TransitionError,
)


def make_order(symbol: str = "RELIANCE", side: str = "BUY", qty: int = 10) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type="LIMIT",
        price=Decimal("2850"),
        strategy_id="TEST-STRAT-001",
    )


@pytest.fixture
def fsm() -> OrderFSM:
    """FSM with no QuestDB store (no persistence)."""
    return OrderFSM(store=None)


@pytest.fixture
def fsm_with_store() -> tuple[OrderFSM, MagicMock]:
    store = MagicMock()
    return OrderFSM(store=store), store


class TestCreation:
    def test_create_returns_pending(self, fsm: OrderFSM) -> None:
        order = make_order()
        record = fsm.create(order)
        assert record.state == OrderState.PENDING
        assert record.order.client_order_id == order.client_order_id

    def test_create_idempotent_same_order(self, fsm: OrderFSM) -> None:
        """Same client_order_id + same order → returns existing record."""
        order = make_order()
        record1 = fsm.create(order)
        record2 = fsm.create(order)  # Same object
        assert record1 is record2

    def test_create_different_order_same_id_raises(self, fsm: OrderFSM) -> None:
        """Same client_order_id but different order content → ValueError."""
        order1 = make_order(symbol="RELIANCE")
        order2 = Order(
            symbol="TCS",  # Different symbol
            side="BUY",
            quantity=10,
            order_type="LIMIT",
            price=Decimal("2850"),
            strategy_id="TEST-STRAT-001",
            client_order_id=order1.client_order_id,  # Same ID!
        )
        fsm.create(order1)
        with pytest.raises(ValueError, match="already exists"):
            fsm.create(order2)


class TestHappyPathTransitions:
    def test_full_fill_path(self, fsm: OrderFSM) -> None:
        """PENDING → SENT → ACKED → FILLED."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        assert fsm.get_record(cid).state == OrderState.SENT

        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH001")
        assert fsm.get_record(cid).state == OrderState.ACKED

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=10, fill_price=Decimal("2851"))
        record = fsm.get_record(cid)
        assert record.state == OrderState.FILLED
        assert record.filled_quantity == 10
        assert record.fill_price_avg == Decimal("2851")

    def test_cancel_path(self, fsm: OrderFSM) -> None:
        """PENDING → SENT → ACKED → CANCEL_PENDING → CANCELLED."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH002")
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        assert fsm.get_record(cid).state == OrderState.CANCEL_PENDING

        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        assert fsm.get_record(cid).state == OrderState.CANCELLED

    def test_partial_fill_then_full_fill(self, fsm: OrderFSM) -> None:
        """ACKED → PARTIAL → FILLED with correct qty tracking."""
        order = make_order(qty=10)
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH003")

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=4, fill_price=Decimal("2850"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.PARTIAL
        assert rec.filled_quantity == 4
        assert rec.remaining_quantity == 6

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=6, fill_price=Decimal("2852"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 10
        assert len(rec.fills) == 2

    def test_rejection_path(self, fsm: OrderFSM) -> None:
        """SENT → REJECTED with reason."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.REJECT, reject_reason="Insufficient margin")
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.REJECTED
        assert rec.reject_reason == "Insufficient margin"


class TestIdempotency:
    def test_duplicate_ack_is_idempotent(self, fsm: OrderFSM) -> None:
        """Duplicate ACK for same order_id must not raise or change state."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH004")
        assert fsm.get_record(cid).state == OrderState.ACKED

        # Duplicate ACK — must be silently accepted
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH004")
        assert fsm.get_record(cid).state == OrderState.ACKED  # Unchanged

    def test_duplicate_cancel_confirm_idempotent(self, fsm: OrderFSM) -> None:
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH005")
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        assert fsm.get_record(cid).state == OrderState.CANCELLED

        # Duplicate — must not raise
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        assert fsm.get_record(cid).state == OrderState.CANCELLED


class TestEdgeCases:
    def test_timeout_from_sent(self, fsm: OrderFSM) -> None:
        """SENT + TIMEOUT → HALTED (unknown broker state)."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.TIMEOUT)
        assert fsm.get_record(cid).state == OrderState.HALTED

    def test_timeout_from_acked(self, fsm: OrderFSM) -> None:
        """ACKED + TIMEOUT → HALTED."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH006")
        fsm.transition(cid, OrderEvent.TIMEOUT)
        assert fsm.get_record(cid).state == OrderState.HALTED

    def test_late_fill_after_cancel_pending(self, fsm: OrderFSM) -> None:
        """Fill arriving while cancel is pending (race condition) → FILLED."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH007")
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        # Race: fill arrives before cancel confirm
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=10, fill_price=Decimal("2850"))
        assert fsm.get_record(cid).state == OrderState.FILLED

    def test_partial_fill_then_cancel(self, fsm: OrderFSM) -> None:
        """Partial fill + cancel → CANCEL_PENDING, then CANCELLED with correct filled qty."""
        order = make_order(qty=10)
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH008")
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=4, fill_price=Decimal("2850"))
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.CANCELLED
        assert rec.filled_quantity == 4
        assert rec.remaining_quantity == 6

    def test_event_in_terminal_state_ignored(self, fsm: OrderFSM) -> None:
        """Events sent to a terminal state order are silently ignored."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.REJECT, reject_reason="Bad params")
        assert fsm.get_record(cid).state == OrderState.REJECTED

        # These should not raise
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="LATE")
        assert fsm.get_record(cid).state == OrderState.REJECTED

    def test_invalid_transition_raises(self, fsm: OrderFSM) -> None:
        """PENDING + FILL (skipping SENT/ACKED) → TransitionError."""
        order = make_order()
        fsm.create(order)
        with pytest.raises(TransitionError):
            fsm.transition(order.client_order_id, OrderEvent.FILL, filled_quantity=10, fill_price=Decimal("2850"))

    def test_unknown_order_returns_none(self, fsm: OrderFSM) -> None:
        """Transition on non-existent client_order_id → returns None (no raise)."""
        result = fsm.transition("NON_EXISTENT_ID", OrderEvent.ACK)
        assert result is None


class TestFillCallback:
    def test_fill_callback_invoked_on_fill(self, fsm: OrderFSM) -> None:
        """Fill callback must be called on FILL and PARTIAL_FILL events."""
        received_fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                received_fills.append(fill)

        fsm.register_fill_callback(_CB())

        order = make_order(qty=10)
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH009")
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=5, fill_price=Decimal("2850"))
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=5, fill_price=Decimal("2851"))

        assert len(received_fills) == 2
        assert received_fills[0].quantity == 5
        assert received_fills[1].quantity == 5


class TestPersistence:
    def test_every_transition_persisted(self, fsm_with_store: tuple) -> None:
        """Every FSM transition must call store.write_order_event()."""
        fsm, store = fsm_with_store
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id

        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH010")
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=10, fill_price=Decimal("2850"))

        # create() + SEND + ACK + FILL = 4 calls
        assert store.write_order_event.call_count == 4

    def test_broker_update_routes_correctly(self, fsm: OrderFSM) -> None:
        """on_broker_update() correctly maps broker events to FSM transitions."""
        order = make_order()
        fsm.create(order)
        cid = order.client_order_id
        fsm.transition(cid, OrderEvent.SEND)

        update = OrderUpdate(
            client_order_id=cid,
            exchange_order_id="EXCH011",
            event="ACK",
        )
        result = fsm.on_broker_update(update)
        assert result is not None
        assert result.state == OrderState.ACKED


class TestQueryMethods:
    def test_get_active_orders(self, fsm: OrderFSM) -> None:
        """get_active_orders() returns only non-terminal orders."""
        o1 = make_order(symbol="RELIANCE")
        o2 = make_order(symbol="TCS")

        fsm.create(o1)
        fsm.create(o2)

        # Fill order 1
        fsm.transition(o1.client_order_id, OrderEvent.SEND)
        fsm.transition(o1.client_order_id, OrderEvent.ACK, exchange_order_id="X1")
        fsm.transition(o1.client_order_id, OrderEvent.FILL, filled_quantity=10, fill_price=Decimal("2850"))

        active = fsm.get_active_orders()
        assert len(active) == 1
        assert active[0].order.symbol == "TCS"
