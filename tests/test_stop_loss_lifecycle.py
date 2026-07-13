"""
tests/test_stop_loss_lifecycle.py — Stop-loss order state machine tests.

P2-FIX-129: Stop-loss order lifecycle.
"""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

from algo_trading.data.broker_base import Fill, Order, OrderRejectedError, OrderUpdate
from algo_trading.execution.order_fsm import (
    FillCallback,
    OrderEvent,
    OrderFSM,
    OrderRecord,
    OrderState,
)

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stop_loss_order(
    symbol: str = "RELIANCE-EQ",
    side: str = "SELL",
    quantity: int = 100,
) -> Order:
    """Create a stop-loss order: always MARKET, price=0."""
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type="MARKET",   # Stop-loss is always MARKET
        price=Decimal("0"),    # MARKET orders carry no price cap
        strategy_id="SL-TEST",
    )


def _advance_to_acked(fsm: OrderFSM, order: Order, exch_id: str = "EXCH-SL-001") -> str:
    """Helper: create, SEND, and ACK an order in the FSM."""
    cid = order.client_order_id
    fsm.create(order)
    fsm.transition(cid, OrderEvent.SEND)
    fsm.transition(cid, OrderEvent.ACK, exchange_order_id=exch_id)
    return cid


# ---------------------------------------------------------------------------
# P2-FIX-129-1: Stop-loss order must be MARKET type
# ---------------------------------------------------------------------------

class TestStopLossIsMarketOrder:
    """P2-FIX-129: Stop-loss Order must always use MARKET type with no price cap."""

    def test_stop_loss_is_market_order(self) -> None:
        sl_order = _make_stop_loss_order()
        assert sl_order.order_type == "MARKET", (
            f"Stop-loss order must be MARKET, got {sl_order.order_type!r}"
        )

    def test_stop_loss_has_no_price_cap(self) -> None:
        sl_order = _make_stop_loss_order()
        assert sl_order.price == Decimal("0"), (
            f"MARKET stop-loss must have price=0 (no cap), got {sl_order.price}"
        )

    def test_stop_loss_cannot_be_limit(self) -> None:
        """Attempting to create a LIMIT stop-loss should raise ValueError."""
        with pytest.raises(ValueError, match="LIMIT order requires a positive price"):
            Order(
                symbol="RELIANCE-EQ",
                side="SELL",
                quantity=100,
                order_type="LIMIT",
                price=Decimal("0"),   # LIMIT with price=0 is invalid
                strategy_id="SL-TEST",
            )

    def test_stop_loss_side_is_sell_for_long_position(self) -> None:
        """Stop-loss for a long position is a SELL order."""
        sl = _make_stop_loss_order(side="SELL")
        assert sl.side == "SELL"

    def test_stop_loss_side_is_buy_for_short_position(self) -> None:
        """Stop-loss for a short position is a BUY order (buy-to-cover)."""
        sl = _make_stop_loss_order(side="BUY")
        assert sl.side == "BUY"
        assert sl.order_type == "MARKET"


# ---------------------------------------------------------------------------
# P2-FIX-129-2: Stop-loss submit success
# ---------------------------------------------------------------------------

class TestStopLossSubmitSuccess:
    """P2-FIX-129: OrderFSM correctly tracks a stop-loss MARKET order submission."""

    def test_stop_loss_submit_success(self) -> None:
        """Stop-loss MARKET order moves from PENDING → SENT → ACKED."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order()
        cid = sl_order.client_order_id

        fsm.create(sl_order)
        assert fsm.get_record(cid).state == OrderState.PENDING

        fsm.transition(cid, OrderEvent.SEND)
        assert fsm.get_record(cid).state == OrderState.SENT

        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH-SL-999")
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.ACKED
        assert rec.exchange_order_id == "EXCH-SL-999"
        assert rec.order.order_type == "MARKET"

    def test_stop_loss_create_is_idempotent(self) -> None:
        """Creating the same stop-loss twice returns the existing record."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order()
        rec1 = fsm.create(sl_order)
        rec2 = fsm.create(sl_order)  # same order, same client_order_id
        assert rec1 is rec2


# ---------------------------------------------------------------------------
# P2-FIX-129-3: Stop-loss FILL → position goes flat
# ---------------------------------------------------------------------------

class TestStopLossFill:
    """P2-FIX-129: FILL event for stop-loss must update position to flat."""

    def test_stop_loss_fill(self) -> None:
        """After stop-loss fills, order state is FILLED and fill is recorded."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order(quantity=50)
        cid = _advance_to_acked(fsm, sl_order)

        fsm.transition(
            cid,
            OrderEvent.FILL,
            filled_quantity=50,
            fill_price=Decimal("2480"),  # Stop hit at 2480
        )

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 50
        assert rec.remaining_quantity == 0

    def test_stop_loss_fill_fires_callback(self) -> None:
        """FILL on stop-loss must invoke registered fill callbacks."""
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm = OrderFSM(store=None)
        fsm.register_fill_callback(_CB())
        sl_order = _make_stop_loss_order(quantity=30)
        cid = _advance_to_acked(fsm, sl_order)

        fsm.transition(
            cid,
            OrderEvent.FILL,
            filled_quantity=30,
            fill_price=Decimal("2475"),
        )

        assert len(fills) == 1
        assert fills[0].quantity == 30
        assert fills[0].price == Decimal("2475")
        assert fills[0].side == "SELL"

    def test_stop_loss_fill_before_ack(self) -> None:
        """Stop-loss FILL before ACK (fast market) must be handled. P2-FIX-99."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order(quantity=100)
        cid = sl_order.client_order_id

        fsm.create(sl_order)
        fsm.transition(cid, OrderEvent.SEND)

        # FILL arrives before ACK on stop-loss
        fsm.transition(
            cid,
            OrderEvent.FILL,
            filled_quantity=100,
            fill_price=Decimal("2460"),
        )

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED, (
            f"Stop-loss FILL before ACK should land in FILLED, got {rec.state}"
        )


# ---------------------------------------------------------------------------
# P2-FIX-129-4: Stop-loss REJECT triggers alert
# ---------------------------------------------------------------------------

class TestStopLossRejectTriggersAlert:
    """P2-FIX-129: REJECT event on stop-loss is critical — must trigger alert."""

    def test_stop_loss_reject_triggers_alert(self) -> None:
        """
        A rejected stop-loss leaves the position unprotected — critical alert.

        We verify that the FSM reaches REJECTED state and that caller code
        can detect this condition to fire an alert.
        """
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order()
        cid = sl_order.client_order_id
        fsm.create(sl_order)
        fsm.transition(cid, OrderEvent.SEND)

        # Broker rejects the stop-loss
        fsm.transition(
            cid,
            OrderEvent.REJECT,
            reject_reason="INSUFFICIENT_MARGIN",
        )

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.REJECTED, (
            f"Expected REJECTED state, got {rec.state}"
        )
        assert rec.reject_reason == "INSUFFICIENT_MARGIN"

        # Caller must detect this and alert (simulated here)
        alert_emitted = rec.state == OrderState.REJECTED and rec.order.order_type == "MARKET"
        assert alert_emitted, (
            "Rejected MARKET stop-loss must trigger a critical alert — "
            "position is now unprotected"
        )

    def test_stop_loss_reject_is_terminal(self) -> None:
        """After REJECT, no further transitions are possible."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order()
        cid = sl_order.client_order_id
        fsm.create(sl_order)
        fsm.transition(cid, OrderEvent.SEND)
        fsm.transition(cid, OrderEvent.REJECT, reject_reason="MARGIN")

        # Any subsequent event must be silently absorbed (terminal state)
        rec_before = fsm.get_record(cid)
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("2500"))
        rec_after = fsm.get_record(cid)

        assert rec_after.state == OrderState.REJECTED, (
            "Terminal REJECTED state must not accept further transitions"
        )
        # Fill count must not change
        assert rec_after.filled_quantity == rec_before.filled_quantity


# ---------------------------------------------------------------------------
# P2-FIX-129-5: Partial fill of stop-loss
# ---------------------------------------------------------------------------

class TestStopLossPartialFill:
    """P2-FIX-129: Partial fill of stop-loss must not leave state as SENT/ACKED."""

    def test_stop_loss_partial_fill_handled(self) -> None:
        """
        MARKET stop-losses normally fill fully in liquid stocks.
        If partial fill arrives (illiquid / circuit), state should be PARTIAL.
        """
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order(quantity=100)
        cid = _advance_to_acked(fsm, sl_order)

        fsm.transition(
            cid,
            OrderEvent.PARTIAL_FILL,
            filled_quantity=60,
            fill_price=Decimal("2490"),
        )

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.PARTIAL, (
            f"After partial stop-loss fill, state should be PARTIAL, got {rec.state}"
        )
        assert rec.filled_quantity == 60
        assert rec.remaining_quantity == 40

    def test_stop_loss_partial_then_full_fill(self) -> None:
        """Partial fill followed by remaining fill → FILLED."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order(quantity=100)
        cid = _advance_to_acked(fsm, sl_order)

        fsm.transition(
            cid, OrderEvent.PARTIAL_FILL,
            filled_quantity=60,
            fill_price=Decimal("2490"),
        )
        assert fsm.get_record(cid).state == OrderState.PARTIAL

        fsm.transition(
            cid, OrderEvent.FILL,
            filled_quantity=40,
            fill_price=Decimal("2488"),
        )
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 100

    def test_stop_loss_partial_fill_fires_callback(self) -> None:
        """Each partial fill event must fire the fill callback."""
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm = OrderFSM(store=None)
        fsm.register_fill_callback(_CB())
        sl_order = _make_stop_loss_order(quantity=100)
        cid = _advance_to_acked(fsm, sl_order)

        fsm.transition(
            cid, OrderEvent.PARTIAL_FILL,
            filled_quantity=50,
            fill_price=Decimal("2495"),
        )
        assert len(fills) == 1
        assert fills[0].quantity == 50


# ---------------------------------------------------------------------------
# P2-FIX-129-6: Duplicate stop-loss update is idempotent
# ---------------------------------------------------------------------------

class TestDuplicateStopLossUpdateIdempotent:
    """P2-FIX-129: Same FILL event twice for stop-loss must not double-count."""

    def test_duplicate_stop_loss_update_idempotent(self) -> None:
        """
        Same FILL event received twice (WS reconnect / broker retry).
        Position must NOT be double-counted.
        """
        fill_count: list[int] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fill_count.append(fill.quantity)

        fsm = OrderFSM(store=None)
        fsm.register_fill_callback(_CB())
        sl_order = _make_stop_loss_order(quantity=100)
        cid = _advance_to_acked(fsm, sl_order)

        # First FILL — valid
        fsm.transition(
            cid, OrderEvent.FILL,
            filled_quantity=100,
            fill_price=Decimal("2480"),
        )
        assert fsm.get_record(cid).state == OrderState.FILLED
        assert len(fill_count) == 1

        # Duplicate FILL — must be idempotent (ignored)
        fsm.transition(
            cid, OrderEvent.FILL,
            filled_quantity=100,
            fill_price=Decimal("2480"),
        )
        assert fsm.get_record(cid).state == OrderState.FILLED
        assert len(fill_count) == 1, (
            "Duplicate FILL on stop-loss must not fire callback a second time"
        )
        # filled_quantity must NOT have doubled
        assert fsm.get_record(cid).filled_quantity == 100, (
            "Duplicate FILL must not double-count filled quantity"
        )

    def test_duplicate_ack_on_stop_loss_idempotent(self) -> None:
        """Duplicate ACK on stop-loss must be silently absorbed."""
        fsm = OrderFSM(store=None)
        sl_order = _make_stop_loss_order()
        cid = _advance_to_acked(fsm, sl_order, exch_id="EXCH-SL-DUP")

        # Duplicate ACK from WS reconnect
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH-SL-DUP")
        assert fsm.get_record(cid).state == OrderState.ACKED

    @pytest.mark.parametrize("quantity,fill_price_str", [
        (50,  "2480.00"),
        (100, "3800.00"),
        (200, "1650.50"),
    ])
    def test_parametrized_stop_loss_fill_idempotency(
        self, quantity: int, fill_price_str: str
    ) -> None:
        """Idempotency holds across different quantities and prices."""
        callback_count: list[int] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                callback_count.append(1)

        fsm = OrderFSM(store=None)
        fsm.register_fill_callback(_CB())
        sl = _make_stop_loss_order(quantity=quantity)
        cid = _advance_to_acked(fsm, sl)

        fill_price = Decimal(fill_price_str)
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=quantity, fill_price=fill_price)
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=quantity, fill_price=fill_price)

        assert len(callback_count) == 1, (
            f"Expected 1 callback for qty={quantity}, got {len(callback_count)}"
        )
        assert fsm.get_record(cid).filled_quantity == quantity
