"""
tests/test_fo_cost_model.py — Unit tests for algo_trading/fo/cost_model.py

Tests verify that the F&O cost model produces results consistent with
known NSE charge examples. These tests are a Tier 1 blocker — must pass
before any backtest can be trusted.

Test design philosophy:
  - All expected values are computed manually from verified NSE charge rates
  - Tests use Decimal arithmetic to match the production code
  - Each test documents the exact calculation it verifies
  - No mocking of cost components — test the real math

NSE F&O cost rates (effective April 1 2026, Union Budget 2026):
  STT (Futures sell):     0.05% of turnover
  STT (Options sell):     0.15% of premium
  STT (Options exercise): 0.15% of intrinsic value
  Exchange (Futures):     0.0019% of turnover
  Exchange (Options):     0.0500% of premium
  SEBI fee:               ₹10 per crore
  GST:                    18% of (exchange + SEBI)
  Stamp duty (buy only):  0.002% of premium/turnover
  Brokerage:              ₹0 (Shoonya)
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from algo_trading.fo.cost_model import FOCostModel, FOTradeCost, FOInstrumentType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model() -> FOCostModel:
    return FOCostModel(slippage_bps_futures=0.0, slippage_bps_options=0.0)


# ── Helper: quantize to paise ─────────────────────────────────────────────────

def q(x: float | Decimal) -> Decimal:
    """Round to 2 decimal places (paise)."""
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ── Option Tests ──────────────────────────────────────────────────────────────

class TestOptionCosts:
    """
    Verify option leg cost computation.

    Reference calculation (SELL 1 lot Nifty CE, ltp=150, underlying=24000):
      qty = 1 × 65 = 65 units
      premium = 65 × 150 = ₹9,750
      turnover = 65 × 24000 = ₹15,60,000

      STT (sell side) = 0.15% × 9750 = ₹14.625 → ₹14.63
      Exchange        = 0.05% × 9750 = ₹4.875  → ₹4.88
      SEBI fee        = (15,60,000 / 1,00,00,000) × 10 = ₹1.56
      GST             = 18% × (4.875 + 1.56) = 18% × 6.435 = ₹1.1583 → ₹1.16
      Stamp duty      = 0 (sell side)
      Total           = 14.63 + 4.88 + 1.56 + 1.16 = ₹22.23
    """

    def test_sell_option_stt(self, model):
        """STT = 0.15% of premium for option sell."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        # premium = 65 × 150 = 9750
        # STT = 0.15% × 9750 = 14.625 → 14.63
        assert cost.stt == Decimal("14.63"), f"Expected 14.63, got {cost.stt}"

    def test_buy_option_no_stt(self, model):
        """STT = 0 for option BUY side."""
        cost = model.compute_option(
            side="BUY", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        assert cost.stt == Decimal("0.00"), f"Expected 0.00, got {cost.stt}"

    def test_option_exchange_charge(self, model):
        """Exchange charge = 0.05% of premium for options."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        # 0.05% × 9750 = 4.875 → 4.88
        assert cost.exchange == Decimal("4.88"), f"Expected 4.88, got {cost.exchange}"

    def test_option_sebi_fee(self, model):
        """SEBI fee = ₹10 per crore of turnover (qty × underlying for index options)."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        # turnover = 65 × 24000 = 15,60,000
        # SEBI = (15,60,000 / 1,00,00,000) × 10 = 1.56
        assert cost.sebi_fee == Decimal("1.56"), f"Expected 1.56, got {cost.sebi_fee}"

    def test_option_gst(self, model):
        """GST = 18% of (exchange + SEBI fee). NOT on STT or stamp duty."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        # GST = 18% × (4.875 + 1.56) = 18% × 6.435 = 1.1583 → 1.16
        assert cost.gst == Decimal("1.16"), f"Expected 1.16, got {cost.gst}"

    def test_option_buy_stamp_duty(self, model):
        """Stamp duty applies to BUY side only."""
        cost = model.compute_option(
            side="BUY", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        # Stamp = 0.002% × 9750 = 0.195 → 0.20
        assert cost.stamp_duty == Decimal("0.20"), f"Expected 0.20, got {cost.stamp_duty}"

    def test_option_sell_no_stamp_duty(self, model):
        """Stamp duty is zero for SELL side."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        assert cost.stamp_duty == Decimal("0.00")

    def test_option_premium_field(self, model):
        """premium field = qty × ltp."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        assert cost.premium == Decimal("9750.00")

    def test_option_turnover_field(self, model):
        """turnover field = qty × underlying."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        assert cost.turnover == Decimal("1560000.00")

    def test_option_brokerage_zero(self, model):
        """Brokerage = ₹0 for Shoonya zero-brokerage platform."""
        cost = model.compute_option(
            side="BUY", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
        )
        assert cost.brokerage == Decimal("0.00")

    def test_multi_lot_scales_linearly(self, model):
        """2 lots should cost exactly 2× the 1-lot cost."""
        cost_1 = model.compute_option("SELL", 1, 65, 150.0, 24000.0)
        cost_2 = model.compute_option("SELL", 2, 65, 150.0, 24000.0)
        # Allow ±0.02 for rounding across lots
        assert abs(float(cost_2.total) - 2 * float(cost_1.total)) < 0.02, (
            f"2-lot total {cost_2.total} should be ~2× {cost_1.total}"
        )

    def test_exercise_stt_uses_intrinsic_value(self, model):
        """
        Exercise STT = 0.15% of intrinsic value (not premium).

        Example: strike=24000 CE, underlying settled at 24100, intrinsic=100/unit.
        intrinsic_total = 65 × 100 = ₹6,500
        STT = 0.15% × 6500 = ₹9.75
        """
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=65,
            ltp=50.0, underlying=24000.0,
            is_exercise=True, intrinsic_value=100.0
        )
        # 0.15% × (65 × 100) = 0.0015 × 6500 = 9.75
        assert cost.stt == Decimal("9.75"), f"Expected 9.75, got {cost.stt}"

    def test_instrument_type_is_options(self, model):
        cost = model.compute_option("SELL", 1, 65, 150.0, 24000.0)
        assert cost.instrument_type == FOInstrumentType.OPTIONS

    def test_bps_of_premium_calculation(self, model):
        """as_bps_of_premium() should return total / premium × 10000."""
        cost = model.compute_option("SELL", 1, 65, 150.0, 24000.0,)
        expected_bps = cost.total / cost.premium * 10000
        assert cost.as_bps_of_premium() == expected_bps.quantize(Decimal("0.01"))

    def test_banknifty_lot_size_30(self, model):
        """BankNifty lot size = 30 (effective Jan 2026 per FAOP70616)."""
        cost = model.compute_option(
            side="SELL", lots=1, lot_size=30, ltp=200.0, underlying=52000.0
        )
        # qty = 30; premium = 30 × 200 = 6000
        expected_premium = Decimal("6000.00")
        assert cost.premium == expected_premium, (
            f"BankNifty premium should be {expected_premium}, got {cost.premium}"
        )


# ── Futures Tests ─────────────────────────────────────────────────────────────

class TestFuturesCosts:
    """
    Verify futures leg cost computation.

    Reference calculation (SELL 1 lot Nifty futures at 24000):
      qty = 65 units
      turnover = 65 × 24000 = ₹15,60,000

      STT (sell side) = 0.05% × 15,60,000 = ₹780.00
      Exchange        = 0.0019% × 15,60,000 = ₹29.64
      SEBI fee        = (15,60,000 / 1,00,00,000) × 10 = ₹1.56
      GST             = 18% × (29.64 + 1.56) = 18% × 31.20 = ₹5.616 → ₹5.62
      Stamp duty      = 0 (sell side)
      Total           = 780.00 + 29.64 + 1.56 + 5.62 = ₹816.82
    """

    def test_futures_sell_stt(self, model):
        """STT = 0.05% of turnover for futures SELL."""
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        # 0.05% × 15,60,000 = 780.00
        assert cost.stt == Decimal("780.00"), f"Expected 780.00, got {cost.stt}"

    def test_futures_buy_no_stt(self, model):
        """STT = 0 for futures BUY side."""
        cost = model.compute_futures(
            side="BUY", lots=1, lot_size=65, price=24000.0
        )
        assert cost.stt == Decimal("0.00")

    def test_futures_exchange_charge(self, model):
        """Exchange charge = 0.0019% of turnover for futures."""
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        # 0.0019% × 1560000 = 29.64
        assert cost.exchange == Decimal("29.64"), f"Expected 29.64, got {cost.exchange}"

    def test_futures_sebi_fee(self, model):
        """SEBI fee = ₹10 per crore."""
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        # (1560000 / 10000000) × 10 = 1.56
        assert cost.sebi_fee == Decimal("1.56")

    def test_futures_gst(self, model):
        """GST = 18% of (exchange + SEBI fee)."""
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        # 18% × (29.64 + 1.56) = 18% × 31.20 = 5.616 → 5.62
        assert cost.gst == Decimal("5.62"), f"Expected 5.62, got {cost.gst}"

    def test_futures_buy_stamp_duty(self, model):
        """Stamp duty on BUY side = 0.002% of turnover."""
        cost = model.compute_futures(
            side="BUY", lots=1, lot_size=65, price=24000.0
        )
        # 0.002% × 1560000 = 31.20
        assert cost.stamp_duty == Decimal("31.20"), f"Expected 31.20, got {cost.stamp_duty}"

    def test_futures_sell_no_stamp_duty(self, model):
        """Stamp duty = 0 on SELL side."""
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        assert cost.stamp_duty == Decimal("0.00")

    def test_futures_total_sell(self, model):
        """
        Total SELL futures cost = STT + exchange + SEBI + GST (no stamp, no brokerage).
        780.00 + 29.64 + 1.56 + 5.62 = 816.82
        """
        cost = model.compute_futures(
            side="SELL", lots=1, lot_size=65, price=24000.0
        )
        assert cost.total == Decimal("816.82"), f"Expected 816.82, got {cost.total}"

    def test_futures_brokerage_zero(self, model):
        cost = model.compute_futures("BUY", 1, 65, 24000.0)
        assert cost.brokerage == Decimal("0.00")

    def test_futures_instrument_type(self, model):
        cost = model.compute_futures("BUY", 1, 65, 24000.0)
        assert cost.instrument_type == FOInstrumentType.FUTURES


# ── Round-trip Tests ──────────────────────────────────────────────────────────

class TestRoundTrip:
    """Test round-trip cost helpers."""

    def test_option_round_trip_returns_two_costs(self, model):
        entry, exit_ = model.option_round_trip(
            lots=1, lot_size=65,
            entry_ltp=150.0, exit_ltp=100.0,
            underlying=24000.0,
            initial_side="SELL",
        )
        assert isinstance(entry, FOTradeCost)
        assert isinstance(exit_, FOTradeCost)
        # Entry = SELL, exit = BUY
        assert entry.side == "SELL"
        assert exit_.side == "BUY"

    def test_strategy_break_even_bps_positive(self, model):
        """Break-even bps should be > 0 (there are always costs)."""
        bps = model.strategy_break_even_bps(
            lots=1, lot_size=65, ltp=100.0, underlying=24000.0
        )
        assert bps > 0, "Break-even bps should be positive"

    def test_strategy_break_even_zero_premium(self, model):
        """Break-even with zero premium should return 0."""
        bps = model.strategy_break_even_bps(
            lots=1, lot_size=65, ltp=0.0, underlying=24000.0
        )
        assert bps == 0.0


# ── Slippage Tests ────────────────────────────────────────────────────────────

class TestSlippage:
    """Verify that slippage is additive and proportional."""

    def test_slippage_adds_to_total(self):
        """With non-zero slippage, total > cost without slippage."""
        model_no_slip = FOCostModel(slippage_bps_options=0.0)
        model_with_slip = FOCostModel(slippage_bps_options=25.0)
        c0 = model_no_slip.compute_option("SELL", 1, 65, 150.0, 24000.0)
        c1 = model_with_slip.compute_option("SELL", 1, 65, 150.0, 24000.0)
        assert c1.total > c0.total
        # Slippage = 25 bps of 9750 = 24.375 → 24.38
        assert c1.slippage == Decimal("24.38")

    def test_futures_slippage_proportional(self):
        """Futures slippage = slippage_bps × turnover / 10000."""
        model_slip = FOCostModel(slippage_bps_futures=2.0)
        cost = model_slip.compute_futures("BUY", 1, 65, 24000.0)
        # 2 bps × 1,560,000 / 10000 = 312.00
        assert cost.slippage == Decimal("312.00"), f"Expected 312.00, got {cost.slippage}"


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions."""

    def test_zero_premium_does_not_crash(self, model):
        """Near-zero premium option should not crash."""
        cost = model.compute_option("SELL", 1, 65, 0.05, 24000.0)
        assert cost.total >= Decimal("0")

    def test_bps_zero_premium(self, model):
        """as_bps_of_premium returns 0 if premium is 0."""
        cost = model.compute_option("SELL", 1, 65, 0.0, 24000.0)
        assert cost.as_bps_of_premium() == Decimal("0")

    def test_round_trip_estimate(self, model):
        """round_trip_estimate = 2× this cost."""
        cost = model.compute_option("SELL", 1, 65, 150.0, 24000.0)
        expected = (cost.total * 2).quantize(Decimal("0.01"))
        assert cost.round_trip_estimate() == expected

    def test_multiple_lots_correct_scaling(self, model):
        """5 lots should have higher absolute cost than 1 lot."""
        c1 = model.compute_option("SELL", 1, 65, 150.0, 24000.0)
        c5 = model.compute_option("SELL", 5, 65, 150.0, 24000.0)
        assert c5.total > c1.total
        assert c5.premium == c1.premium * 5

    def test_futures_turnover_field(self, model):
        """turnover = qty × price for futures."""
        cost = model.compute_futures("BUY", 2, 65, 24000.0)
        # 2 × 65 × 24000 = 3,120,000
        assert cost.turnover == Decimal("3120000.00")
