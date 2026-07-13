"""
tests/test_order_lifecycle_replay.py — P2-FIX-60: Replay tests for order lifecycle.

Exercises event sequences that are race-prone or hard to reproduce manually:
  1. ACK arrives while broker REST call is in-flight (order in SENT)
  2. FILL arrives before ACK (broker sends fill without explicit ACK first)
  3. REJECT arrives after REST returned success (conflicting confirmations)
  4. Cancel and fill race (CANCEL_PENDING + FILL race → FILLED)
  5. Duplicate FILL event (network retry / broker resend)
  6. Partial fill then cancel (with correct remaining-qty tracking)
  7. Late fill after CANCEL_CONFIRM (CANCELLED + FILL → position mismatch alert)
  8. WebSocket reconnect with missed updates (order still in SENT after reconnect)
  9. Multiple partial fills then full fill (VWAP tracking)
 10. Duplicate PARTIAL_FILL from WS reconnect
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock

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


def make_order(symbol: str = "NIFTY", side: str = "BUY", qty: int = 100) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type="LIMIT",
        price=Decimal("22500"),
        strategy_id="REPLAY-TEST",
    )


@pytest.fixture
def fsm() -> OrderFSM:
    return OrderFSM(store=None)


# ---------------------------------------------------------------------------
# Helper: advance order to SENT state
# ---------------------------------------------------------------------------
def _to_sent(fsm: OrderFSM, order: Order) -> str:
    fsm.create(order)
    cid = order.client_order_id
    fsm.transition(cid, OrderEvent.SEND)
    assert fsm.get_record(cid).state == OrderState.SENT
    return cid


def _to_acked(fsm: OrderFSM, order: Order, exch_id: str = "EXCHXXX") -> str:
    cid = _to_sent(fsm, order)
    fsm.transition(cid, OrderEvent.ACK, exchange_order_id=exch_id)
    assert fsm.get_record(cid).state == OrderState.ACKED
    return cid


# ---------------------------------------------------------------------------
# Scenario 1: ACK arrives while REST call is still in-flight
# ---------------------------------------------------------------------------
class TestAckWhileRESTInFlight:
    """
    Timeline:
      t=0ms  fsm.create() + SEND transition → SENT
      t=80ms WS ACK fires (broker is fast) → SENT + ACK → ACKED
      t=120ms REST returns exchange_order_id → record_exchange_id() no-op

    The key invariant: exchange_order_id set by WS ACK must survive the
    subsequent record_exchange_id() call from the REST response.
    """

    def test_ack_from_ws_transitions_to_acked(self, fsm: OrderFSM) -> None:
        order = make_order()
        cid = _to_sent(fsm, order)

        # WS ACK arrives during REST call
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH_WS_001")
        assert fsm.get_record(cid).state == OrderState.ACKED
        assert fsm.get_record(cid).exchange_order_id == "EXCH_WS_001"

    def test_rest_record_exchange_id_is_noop_after_ws_ack(self, fsm: OrderFSM) -> None:
        order = make_order()
        cid = _to_sent(fsm, order)

        # WS ACK sets exchange id first
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH_WS_001")
        # REST returns (possibly same or different id — record_exchange_id is no-op if already set)
        fsm.record_exchange_id(cid, "EXCH_REST_001")

        rec = fsm.get_record(cid)
        # WS id should be preserved (record_exchange_id skips if already set)
        assert rec.exchange_order_id == "EXCH_WS_001"

    def test_duplicate_ack_from_ws_is_idempotent(self, fsm: OrderFSM) -> None:
        order = make_order()
        cid = _to_acked(fsm, order, "EXCH_ACK_DUP")

        # Second ACK from a reconnected WS must be silently accepted
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH_ACK_DUP")
        assert fsm.get_record(cid).state == OrderState.ACKED


# ---------------------------------------------------------------------------
# Scenario 2: FILL arrives before ACK (P2-FIX-99)
# ---------------------------------------------------------------------------
class TestFillBeforeAck:
    """
    Some brokers (e.g., Shoonya on fast markets) send FILL before ACK.
    The FSM must accept SENT + FILL → FILLED without error.
    """

    def test_fill_before_ack_transitions_to_filled(self, fsm: OrderFSM) -> None:
        order = make_order(qty=50)
        cid = _to_sent(fsm, order)

        # FILL arrives before ACK
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("22510"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 50

    def test_partial_fill_before_ack(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_sent(fsm, order)

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=40, fill_price=Decimal("22500"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.PARTIAL
        assert rec.filled_quantity == 40
        assert rec.remaining_quantity == 60

    def test_fill_before_ack_fires_callback(self, fsm: OrderFSM) -> None:
        fills_received: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills_received.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=50)
        cid = _to_sent(fsm, order)

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("22510"))
        assert len(fills_received) == 1
        assert fills_received[0].quantity == 50


# ---------------------------------------------------------------------------
# Scenario 3: REJECT after REST success
# ---------------------------------------------------------------------------
class TestRejectAfterRestSuccess:
    """
    REST returns OK (order accepted) but broker later sends REJECT via WS.
    This can happen if validation passes at REST layer but fails at exchange.
    """

    def test_reject_from_sent_state(self, fsm: OrderFSM) -> None:
        order = make_order()
        cid = _to_sent(fsm, order)

        fsm.transition(cid, OrderEvent.REJECT, reject_reason="Exchange margin reject")
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.REJECTED
        assert "margin" in rec.reject_reason

    def test_reject_from_acked_is_invalid_transition(self, fsm: OrderFSM) -> None:
        """Once ACKED, broker cannot send REJECT — that is a protocol violation."""
        order = make_order()
        cid = _to_acked(fsm, order)

        with pytest.raises(TransitionError):
            fsm.transition(cid, OrderEvent.REJECT, reject_reason="Late reject")


# ---------------------------------------------------------------------------
# Scenario 4: Cancel and fill race
# ---------------------------------------------------------------------------
class TestCancelFillRace:
    """
    We send CANCEL_REQUEST, but the fill beats the cancel to the exchange.
    The FSM must accept the fill and move to FILLED despite CANCEL_PENDING state.
    """

    def test_fill_beats_cancel_confirm(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        assert fsm.get_record(cid).state == OrderState.CANCEL_PENDING

        # Fill arrives before cancel confirm — fill wins
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22505"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 100

    def test_partial_fill_in_cancel_pending_stays_partial(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        # Partial fill arrives in CANCEL_PENDING — stays PARTIAL (race, not fully filled)
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=60, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.PARTIAL
        assert fsm.get_record(cid).filled_quantity == 60

    def test_cancel_confirm_after_partial_in_cancel_pending(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=40, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.PARTIAL

        # Now cancel is confirmed — PARTIAL should still accept CANCEL_REQUEST then CANCEL_CONFIRM
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.CANCELLED
        assert rec.filled_quantity == 40
        assert rec.remaining_quantity == 60


# ---------------------------------------------------------------------------
# Scenario 5: Duplicate FILL event
# ---------------------------------------------------------------------------
class TestDuplicateFill:
    """
    Broker sends the same FILL event twice (WS reconnect / retry).
    Must be idempotent — no double-counting of position or P&L.
    """

    def test_duplicate_fill_is_idempotent(self, fsm: OrderFSM) -> None:
        fills_received: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills_received.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.FILLED
        assert len(fills_received) == 1

        # Duplicate fill — must be ignored, callback must NOT fire again
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.FILLED
        assert len(fills_received) == 1  # Still 1 — no duplicate callback

    def test_duplicate_partial_fill_after_full_fill(self, fsm: OrderFSM) -> None:
        """PARTIAL_FILL arriving after FILLED (WS replay) must be silently discarded."""
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.FILLED

        # Delayed PARTIAL_FILL from reconnect — must not change state
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=50, fill_price=Decimal("22495"))
        assert fsm.get_record(cid).state == OrderState.FILLED


# ---------------------------------------------------------------------------
# Scenario 6: Partial fill then cancel
# ---------------------------------------------------------------------------
class TestPartialFillThenCancel:
    """
    Order partially fills, then operator cancels remainder.
    filled_quantity must track correctly throughout.
    """

    def test_partial_fill_qty_tracking_through_cancel(self, fsm: OrderFSM) -> None:
        order = make_order(qty=200)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=80, fill_price=Decimal("22490"))
        assert fsm.get_record(cid).filled_quantity == 80
        assert fsm.get_record(cid).remaining_quantity == 120

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=50, fill_price=Decimal("22495"))
        assert fsm.get_record(cid).filled_quantity == 130
        assert fsm.get_record(cid).remaining_quantity == 70

        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.CANCELLED
        assert rec.filled_quantity == 130
        assert rec.remaining_quantity == 70

    def test_fill_callback_fires_on_each_partial(self, fsm: OrderFSM) -> None:
        fills: list[int] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill.quantity)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=30, fill_price=Decimal("22500"))
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=70, fill_price=Decimal("22502"))
        assert fills == [30, 70]

    def test_vwap_of_partial_fills(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=50, fill_price=Decimal("100"))
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("200"))

        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        # VWAP = (50*100 + 50*200) / 100 = 150
        assert rec.fill_price_avg == Decimal("150")


# ---------------------------------------------------------------------------
# Scenario 7: Late fill after CANCEL_CONFIRM (P2-FIX-99)
# ---------------------------------------------------------------------------
class TestLateFillAfterCancel:
    """
    Broker confirms cancellation, but a fill sneaks through (exchange race).
    The FSM must:
    - NOT change state from CANCELLED (audit integrity)
    - Call fill callbacks so position tracker can reconcile the open position
    - Log at CRITICAL to alert operator
    """

    def test_late_fill_triggers_callback(self, fsm: OrderFSM) -> None:
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        assert fsm.get_record(cid).state == OrderState.CANCELLED

        # Late fill arrives after cancel confirm
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22500"))

        # State stays CANCELLED for audit integrity
        assert fsm.get_record(cid).state == OrderState.CANCELLED

        # BUT fill callback must fire so position tracking can reconcile
        assert len(fills) == 1
        assert fills[0].quantity == 100

    def test_late_fill_updates_filled_quantity(self, fsm: OrderFSM) -> None:
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("22500"))
        rec = fsm.get_record(cid)
        assert rec.filled_quantity == 100  # tracked for reconciliation
        assert rec.state == OrderState.CANCELLED  # state unchanged

    def test_late_partial_fill_after_cancel(self, fsm: OrderFSM) -> None:
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)

        # Late partial fill
        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=30, fill_price=Decimal("22500"))
        assert len(fills) == 1
        assert fsm.get_record(cid).state == OrderState.CANCELLED
        assert fsm.get_record(cid).filled_quantity == 30

    def test_zero_price_late_fill_is_ignored(self, fsm: OrderFSM) -> None:
        """Late fill with price=0 must not create a fill record (data quality guard)."""
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=100)
        cid = _to_acked(fsm, order)
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)

        # fill_price=0 means bad data — should not create a fill record
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("0"))
        assert len(fills) == 0
        assert fsm.get_record(cid).filled_quantity == 0


# ---------------------------------------------------------------------------
# Scenario 8: WebSocket reconnect with missed updates
# ---------------------------------------------------------------------------
class TestWebSocketReconnect:
    """
    System reconnects to broker WS after a disconnect.
    During disconnect, broker may have ACKed/filled the order.
    Startup reconciler would normally re-apply missed events.
    The FSM must handle re-applied events idempotently.
    """

    def test_reconcile_ack_on_sent_order(self, fsm: OrderFSM) -> None:
        """After reconnect, order was ACKed during disconnect — apply ACK."""
        order = make_order()
        cid = _to_sent(fsm, order)
        # Reconnect: broker reports this order as ACKed
        fsm.transition(cid, OrderEvent.ACK, exchange_order_id="EXCH_RECON_001")
        assert fsm.get_record(cid).state == OrderState.ACKED

    def test_reconcile_fill_on_acked_order(self, fsm: OrderFSM) -> None:
        """After reconnect, order was filled during disconnect."""
        order = make_order(qty=50)
        cid = _to_acked(fsm, order)
        # Reconnect: broker reports this order as FILLED
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.FILLED

    def test_reconcile_replay_of_already_completed_order(self, fsm: OrderFSM) -> None:
        """
        Broker replays FILL event for an order already in FILLED state.
        Must be idempotent — no callback fired, no state change.
        """
        fills: list[Fill] = []

        class _CB(FillCallback):
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                fills.append(fill)

        fsm.register_fill_callback(_CB())
        order = make_order(qty=50)
        cid = _to_acked(fsm, order)
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("22500"))
        assert len(fills) == 1

        # Reconnect replay of same fill — idempotent
        fsm.transition(cid, OrderEvent.FILL, filled_quantity=50, fill_price=Decimal("22500"))
        assert fsm.get_record(cid).state == OrderState.FILLED
        assert len(fills) == 1  # No second callback

    def test_reconcile_cancelled_order_gets_duplicate_cancel(self, fsm: OrderFSM) -> None:
        order = make_order()
        cid = _to_acked(fsm, order)
        fsm.transition(cid, OrderEvent.CANCEL_REQUEST)
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        # Reconnect replay of CANCEL_CONFIRM — idempotent
        fsm.transition(cid, OrderEvent.CANCEL_CONFIRM)
        assert fsm.get_record(cid).state == OrderState.CANCELLED


# ---------------------------------------------------------------------------
# Scenario 9: Multiple partial fills with VWAP tracking
# ---------------------------------------------------------------------------
class TestMultiplePartialFills:
    """
    Three partial fills arrive sequentially. VWAP must be computed correctly
    and filled_quantity must accumulate correctly.
    """

    def test_three_partial_fills_accumulate(self, fsm: OrderFSM) -> None:
        order = make_order(qty=300)
        cid = _to_acked(fsm, order)

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=100, fill_price=Decimal("100"))
        assert fsm.get_record(cid).filled_quantity == 100

        fsm.transition(cid, OrderEvent.PARTIAL_FILL, filled_quantity=100, fill_price=Decimal("110"))
        assert fsm.get_record(cid).filled_quantity == 200

        fsm.transition(cid, OrderEvent.FILL, filled_quantity=100, fill_price=Decimal("90"))
        rec = fsm.get_record(cid)
        assert rec.state == OrderState.FILLED
        assert rec.filled_quantity == 300
        # VWAP = (100*100 + 100*110 + 100*90) / 300 = 300/3 = 100
        assert rec.fill_price_avg == Decimal("100")


# ---------------------------------------------------------------------------
# Scenario 10: on_broker_update exchange-id lookup
# ---------------------------------------------------------------------------
class TestBrokerUpdateRouting:
    """
    on_broker_update() must map exchange_order_id back to client_order_id when
    the client_order_id is not in the WS payload (which Shoonya does).
    """

    def test_fill_via_exchange_id_lookup(self, fsm: OrderFSM) -> None:
        order = make_order(qty=50)
        cid = _to_acked(fsm, order, exch_id="EXCH_LOOKUP_001")

        update = OrderUpdate(
            client_order_id="",           # Shoonya does not always return this
            exchange_order_id="EXCH_LOOKUP_001",
            event="FILL",
            filled_quantity=50,
            fill_price=Decimal("22500"),
        )
        result = fsm.on_broker_update(update)
        assert result is not None
        assert result.state == OrderState.FILLED

    def test_unknown_exchange_id_returns_none(self, fsm: OrderFSM) -> None:
        update = OrderUpdate(
            client_order_id="",
            exchange_order_id="NONEXISTENT_99999",
            event="ACK",
        )
        result = fsm.on_broker_update(update)
        assert result is None
