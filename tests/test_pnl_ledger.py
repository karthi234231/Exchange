"""
tests/test_pnl_ledger.py — P2-FIX-61: Property tests for the P&L ledger.

Tests the `LiveSystemState.update_position_from_fill()` P&L accounting logic
for correctness invariants using both unit tests and Hypothesis property tests.

Invariants verified:
  1. Flat after BUY then SELL same qty → realized PnL = (sell_price - buy_price)*qty - costs
  2. Flat after SHORT then BUY same qty → realized PnL = (short_price - cover_price)*qty - costs
  3. Total net_quantity equals algebraic sum of all fill quantities
  4. average_price only applies when position is open (non-zero net_qty)
  5. Reversing position (SELL more than held LONG) closes old side, opens short remainder
  6. Adding to position updates VWAP correctly
  7. daily_pnl monotonically decreases on open costs, then adjusts on close
  8. peak_pnl is always >= daily_pnl at any point in time (after any profitable close)
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pytest

try:
    from hypothesis import given, settings as h_settings, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from algo_trading.data.system_state import LiveSystemState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state() -> LiveSystemState:
    """Fresh state with persistence disabled (no state file writes in tests)."""
    s = LiveSystemState()
    # Prevent file I/O during tests by redirecting save() to no-op
    s.save = lambda: None  # type: ignore[method-assign]
    return s


def _fill(
    state: LiveSystemState,
    symbol: str,
    side: str,
    qty: int,
    price: str,
    cost: str = "0",
) -> None:
    state.update_position_from_fill(
        symbol=symbol,
        side=side,
        quantity=qty,
        fill_price=Decimal(price),
        trade_cost=Decimal(cost),
    )


# ---------------------------------------------------------------------------
# Invariant 1: Round-trip LONG position — flat after BUY+SELL
# ---------------------------------------------------------------------------

class TestLongRoundTrip:
    def test_buy_then_sell_same_qty_returns_to_flat(self) -> None:
        s = _state()
        _fill(s, "RELIANCE", "BUY", 100, "2500")
        _fill(s, "RELIANCE", "SELL", 100, "2600")
        pos = s.get_position("RELIANCE")
        assert pos is not None
        assert pos.net_quantity == 0
        assert pos.is_flat

    def test_realized_pnl_long_round_trip_no_costs(self) -> None:
        """PnL = (exit - entry) * qty when costs = 0."""
        s = _state()
        _fill(s, "NIFTY", "BUY", 50, "22000", cost="0")
        _fill(s, "NIFTY", "SELL", 50, "22500", cost="0")
        # Expected: (22500 - 22000) * 50 = 25000
        assert s.daily_pnl == Decimal("25000")

    def test_realized_pnl_long_round_trip_with_costs(self) -> None:
        """PnL = (exit - entry) * qty - entry_cost - exit_cost."""
        s = _state()
        _fill(s, "TCS", "BUY", 10, "3000", cost="30")   # open cost deducted
        _fill(s, "TCS", "SELL", 10, "3100", cost="31")   # close cost deducted
        # gross = (3100 - 3000) * 10 = 1000; net = 1000 - 30 - 31 = 939
        assert s.daily_pnl == Decimal("939")

    def test_loss_trade_reduces_pnl(self) -> None:
        s = _state()
        _fill(s, "INFY", "BUY", 100, "1500", cost="0")
        _fill(s, "INFY", "SELL", 100, "1450", cost="0")
        # PnL = (1450 - 1500) * 100 = -5000
        assert s.daily_pnl == Decimal("-5000")


# ---------------------------------------------------------------------------
# Invariant 2: Round-trip SHORT position
# ---------------------------------------------------------------------------

class TestShortRoundTrip:
    def test_short_then_cover_returns_to_flat(self) -> None:
        s = _state()
        _fill(s, "HDFC", "SELL", 50, "1600")
        _fill(s, "HDFC", "BUY", 50, "1550")
        pos = s.get_position("HDFC")
        assert pos is not None
        assert pos.net_quantity == 0

    def test_short_pnl_no_costs(self) -> None:
        """Short P&L = (short_price - cover_price) * qty."""
        s = _state()
        _fill(s, "HDFC", "SELL", 100, "1600", cost="0")
        _fill(s, "HDFC", "BUY", 100, "1550", cost="0")
        # Expected: (1600 - 1550) * 100 = 5000
        assert s.daily_pnl == Decimal("5000")

    def test_short_loss_trade(self) -> None:
        s = _state()
        _fill(s, "HDFC", "SELL", 100, "1600", cost="0")
        _fill(s, "HDFC", "BUY", 100, "1700", cost="0")
        # PnL = (1600 - 1700) * 100 = -10000
        assert s.daily_pnl == Decimal("-10000")


# ---------------------------------------------------------------------------
# Invariant 3: net_quantity equals algebraic sum of fills
# ---------------------------------------------------------------------------

class TestNetQuantity:
    def test_single_buy(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100")
        assert s.get_position("X").net_quantity == 100

    def test_multiple_buys_accumulate(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100")
        _fill(s, "X", "BUY", 50, "101")
        assert s.get_position("X").net_quantity == 150

    def test_partial_sell_reduces_net_qty(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100")
        _fill(s, "X", "SELL", 40, "110")
        assert s.get_position("X").net_quantity == 60

    def test_signed_sum_invariant(self) -> None:
        """net_quantity must equal sum of +qty for BUY and -qty for SELL."""
        s = _state()
        fills = [
            ("BUY", 100), ("BUY", 50), ("SELL", 30), ("BUY", 20), ("SELL", 80),
        ]
        expected = sum(q if side == "BUY" else -q for side, q in fills)
        for side, qty in fills:
            _fill(s, "X", side, qty, "1000")
        assert s.get_position("X").net_quantity == expected

    def test_flat_position_is_zero_qty(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100")
        _fill(s, "X", "SELL", 100, "100")
        assert s.get_position("X").net_quantity == 0


# ---------------------------------------------------------------------------
# Invariant 4: VWAP (average_price) correct for multi-fill opens
# ---------------------------------------------------------------------------

class TestVWAP:
    def test_single_fill_avg_price(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "2500")
        assert s.get_position("X").average_price == Decimal("2500")

    def test_two_fills_equal_qty_avg(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "2500")
        _fill(s, "X", "BUY", 100, "2600")
        assert s.get_position("X").average_price == Decimal("2550")

    def test_two_fills_unequal_qty_vwap(self) -> None:
        """VWAP = (100*2500 + 200*2600) / 300 = (250000 + 520000) / 300 = 770000/300."""
        s = _state()
        _fill(s, "X", "BUY", 100, "2500")
        _fill(s, "X", "BUY", 200, "2600")
        expected = (Decimal("100") * Decimal("2500") + Decimal("200") * Decimal("2600")) / Decimal("300")
        assert s.get_position("X").average_price == expected

    def test_avg_price_zero_when_flat(self) -> None:
        """Once position is flat, average_price should be 0 (sentinel for no open position)."""
        s = _state()
        _fill(s, "X", "BUY", 100, "2500")
        _fill(s, "X", "SELL", 100, "2600")
        assert s.get_position("X").average_price == Decimal("0")


# ---------------------------------------------------------------------------
# Invariant 5: Reversal (sell more than long, or buy more than short)
# ---------------------------------------------------------------------------

class TestReversal:
    def test_long_to_short_reversal(self) -> None:
        """
        BUY 100 @ 100, then SELL 150 @ 110 — closes long 100, opens short 50.
        Realized PnL on 100 closed = (110 - 100) * 100 = 1000.
        Net qty = -50.
        """
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="0")
        _fill(s, "X", "SELL", 150, "110", cost="0")
        pos = s.get_position("X")
        assert pos.net_quantity == -50
        # Realized PnL from closing the long
        assert s.daily_pnl == Decimal("1000")

    def test_short_to_long_reversal(self) -> None:
        """
        SELL 100 @ 100, then BUY 120 @ 90 — closes short 100, opens long 20.
        Realized PnL on 100 closed = (100 - 90) * 100 = 1000.
        Net qty = +20.
        """
        s = _state()
        _fill(s, "X", "SELL", 100, "100", cost="0")
        _fill(s, "X", "BUY", 120, "90", cost="0")
        pos = s.get_position("X")
        assert pos.net_quantity == 20
        assert s.daily_pnl == Decimal("1000")

    def test_reversal_new_position_avg_price(self) -> None:
        """After reversal, average_price of new position = fill price of reversing trade."""
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="0")
        _fill(s, "X", "SELL", 150, "110", cost="0")   # opens short 50 @ 110
        pos = s.get_position("X")
        assert pos.net_quantity == -50
        assert pos.average_price == Decimal("110")


# ---------------------------------------------------------------------------
# Invariant 6: Partial close — partial P&L realization
# ---------------------------------------------------------------------------

class TestPartialClose:
    def test_partial_sell_realises_partial_pnl(self) -> None:
        """Sell half the position — realize half the gross P&L."""
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="0")
        _fill(s, "X", "SELL", 50, "120", cost="0")
        # Realized on 50 shares: (120 - 100) * 50 = 1000
        assert s.daily_pnl == Decimal("1000")
        assert s.get_position("X").net_quantity == 50

    def test_partial_close_preserves_avg_price(self) -> None:
        """Average price of remaining long should not change on partial close."""
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="0")
        _fill(s, "X", "SELL", 40, "120", cost="0")
        pos = s.get_position("X")
        # avg_price stays 100 for the remaining 60
        assert pos.average_price == Decimal("100")
        assert pos.net_quantity == 60


# ---------------------------------------------------------------------------
# Invariant 7: peak_pnl tracks maximum
# ---------------------------------------------------------------------------

class TestPeakPnl:
    def test_peak_pnl_tracks_max(self) -> None:
        s = _state()
        # Trade 1: profit 5000
        _fill(s, "A", "BUY", 100, "100", cost="0")
        _fill(s, "A", "SELL", 100, "150", cost="0")
        assert s.peak_pnl == Decimal("5000")

        # Trade 2: loss 1000
        _fill(s, "B", "BUY", 100, "200", cost="0")
        _fill(s, "B", "SELL", 100, "190", cost="0")
        # daily_pnl = 5000 - 1000 = 4000, peak stays at 5000
        assert s.peak_pnl == Decimal("5000")
        assert s.daily_pnl == Decimal("4000")

    def test_peak_pnl_updates_on_new_high(self) -> None:
        s = _state()
        _fill(s, "A", "BUY", 100, "100", cost="0")
        _fill(s, "A", "SELL", 100, "150", cost="0")
        assert s.peak_pnl == Decimal("5000")

        _fill(s, "B", "BUY", 100, "200", cost="0")
        _fill(s, "B", "SELL", 100, "300", cost="0")
        # daily_pnl = 5000 + 10000 = 15000; peak updates
        assert s.peak_pnl == Decimal("15000")


# ---------------------------------------------------------------------------
# Invariant 8: Multiple symbols are independent
# ---------------------------------------------------------------------------

class TestMultiSymbol:
    def test_two_symbols_independent_pnl(self) -> None:
        s = _state()
        _fill(s, "RELIANCE", "BUY", 100, "2500", cost="0")
        _fill(s, "INFY", "BUY", 50, "1500", cost="0")

        _fill(s, "RELIANCE", "SELL", 100, "2600", cost="0")
        _fill(s, "INFY", "SELL", 50, "1400", cost="0")

        reliance_pnl = (Decimal("2600") - Decimal("2500")) * 100
        infy_pnl = (Decimal("1400") - Decimal("1500")) * 50
        assert s.daily_pnl == reliance_pnl + infy_pnl

    def test_position_of_one_symbol_doesnt_affect_other(self) -> None:
        s = _state()
        _fill(s, "A", "BUY", 100, "100")
        _fill(s, "B", "BUY", 50, "200")
        assert s.get_position("A").net_quantity == 100
        assert s.get_position("B").net_quantity == 50

    def test_get_all_positions_excludes_flat(self) -> None:
        s = _state()
        _fill(s, "A", "BUY", 100, "100")
        _fill(s, "A", "SELL", 100, "110")  # A now flat
        _fill(s, "B", "BUY", 50, "200")    # B open

        active = s.get_all_positions()
        symbols = {p.symbol for p in active}
        assert "B" in symbols
        assert "A" not in symbols


# ---------------------------------------------------------------------------
# Invariant 9: Cost accounting — costs always reduce PnL
# ---------------------------------------------------------------------------

class TestCostAccounting:
    def test_open_cost_immediately_reduces_pnl(self) -> None:
        """Opening cost is deducted immediately on entry."""
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="50")
        assert s.daily_pnl == Decimal("-50")

    def test_close_cost_deducted_on_exit(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="50")
        _fill(s, "X", "SELL", 100, "100", cost="50")
        # gross pnl = 0; net = 0 - 50 (open) - 50 (close) = -100
        assert s.daily_pnl == Decimal("-100")

    def test_zero_cost_is_neutral(self) -> None:
        s = _state()
        _fill(s, "X", "BUY", 100, "100", cost="0")
        assert s.daily_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# Hypothesis property tests (if installed)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    _price = st.decimals(
        min_value=Decimal("1"), max_value=Decimal("10000"),
        allow_nan=False, allow_infinity=False, places=2
    )
    _qty = st.integers(min_value=1, max_value=1000)

    @given(buy_price=_price, sell_price=_price, qty=_qty)
    @h_settings(max_examples=200, deadline=1000)
    def test_property_long_round_trip_pnl(
        buy_price: Decimal, sell_price: Decimal, qty: int
    ) -> None:
        """
        Property: For any BUY then SELL same qty with zero costs,
        realized PnL = (sell - buy) * qty and position is flat.
        """
        s = _state()
        _fill(s, "SYM", "BUY", qty, str(buy_price), cost="0")
        _fill(s, "SYM", "SELL", qty, str(sell_price), cost="0")

        pos = s.get_position("SYM")
        assert pos is not None
        assert pos.net_quantity == 0  # Always flat after round trip

        expected = (sell_price - buy_price) * qty
        # Allow tiny floating point tolerance via Decimal arithmetic
        assert abs(s.daily_pnl - expected) < Decimal("0.01"), (
            f"PnL mismatch: got {s.daily_pnl}, expected {expected}"
        )

    @given(price=_price, qty1=_qty, qty2=_qty)
    @h_settings(max_examples=200, deadline=1000)
    def test_property_net_quantity_equals_signed_fill_sum(
        price: Decimal, qty1: int, qty2: int
    ) -> None:
        """
        Property: net_quantity after any sequence of fills equals
        the algebraic sum of +qty (BUY) and -qty (SELL).
        """
        s = _state()
        _fill(s, "SYM", "BUY", qty1, str(price))
        _fill(s, "SYM", "SELL", qty2, str(price))

        expected_net = qty1 - qty2
        pos = s.get_position("SYM")
        assert pos is not None
        assert pos.net_quantity == expected_net

    @given(price1=_price, price2=_price, qty=_qty)
    @h_settings(max_examples=200, deadline=1000)
    def test_property_vwap_two_equal_qty_fills(
        price1: Decimal, price2: Decimal, qty: int
    ) -> None:
        """Property: VWAP of two equal-size BUY fills = (p1 + p2) / 2."""
        s = _state()
        _fill(s, "SYM", "BUY", qty, str(price1))
        _fill(s, "SYM", "BUY", qty, str(price2))

        pos = s.get_position("SYM")
        assert pos is not None
        expected_avg = (price1 + price2) / 2
        assert abs(pos.average_price - expected_avg) < Decimal("0.01")

    @given(cost=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000"),
                            allow_nan=False, allow_infinity=False, places=2),
           qty=_qty, price=_price)
    @h_settings(max_examples=100, deadline=1000)
    def test_property_cost_always_reduces_pnl(
        cost: Decimal, qty: int, price: Decimal
    ) -> None:
        """Property: opening cost ≥ 0 always reduces or maintains daily_pnl."""
        s = _state()
        before = s.daily_pnl
        _fill(s, "SYM", "BUY", qty, str(price), cost=str(cost))
        # Cost deducted immediately on open; pnl must not increase
        assert s.daily_pnl <= before
