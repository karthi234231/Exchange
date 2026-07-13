"""
tests/test_fo_strategies.py — Tests for fo/strategies.py payoff math.

Covers all 7 strategy templates and free-function helpers.
All monetary assertions use exact integer INR arithmetic (lot-size scaled).
"""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import patch

import pytest

from algo_trading.fo.strategies import (
    StrategyLeg,
    StrategyBuilder,
    payoff_at_expiry,
    payoff_curve,
    validate_strategy_legs,
)
from algo_trading.fo.cost_model import FOCostModel

# ── Test constants (NIFTY, lot size per FAOP70616) ───────────────────────────
_SYM      = "NIFTY"
_LS       = 65           # lot size
_S        = 24_000.0     # spot
_EXPIRY   = date(2026, 6, 26)

# Patch target: suppress session-time guard so tests run at any wall-clock time
_PATCH_TIME = "algo_trading.fo.strategies._check_entry_time"


@pytest.fixture(scope="module")
def cost_model() -> FOCostModel:
    return FOCostModel()


@pytest.fixture(scope="module")
def builder(cost_model: FOCostModel) -> StrategyBuilder:
    return StrategyBuilder(cost_model=cost_model, symbol=_SYM)


# ── Leg factories (bypass StrategyBuilder for payoff-function tests) ─────────

def _ce(strike: float, prem: float, direction: int = -1, lots: int = 1) -> StrategyLeg:
    return StrategyLeg(
        symbol=_SYM, option_type="CE", strike=strike, expiry=_EXPIRY,
        direction=direction, lots=lots, lot_size=_LS, entry_premium=prem,
    )

def _pe(strike: float, prem: float, direction: int = -1, lots: int = 1) -> StrategyLeg:
    return StrategyLeg(
        symbol=_SYM, option_type="PE", strike=strike, expiry=_EXPIRY,
        direction=direction, lots=lots, lot_size=_LS, entry_premium=prem,
    )

def _fut(entry: float, direction: int = 1, lots: int = 1) -> StrategyLeg:
    return StrategyLeg(
        symbol=_SYM, option_type="FUT", strike=0.0, expiry=_EXPIRY,
        direction=direction, lots=lots, lot_size=_LS, entry_premium=entry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# payoff_at_expiry — fundamental formula tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPayoffAtExpiry:
    """CE/PE/FUT per-unit and scaled payoff correctness."""

    def test_ce_long_itm(self):
        # Long CE at 24000 with premium 100, S_T = 24300 (ITM)
        # pnl = +1 × (300 − 100) × 1 × 65 = 13 000
        leg = _ce(strike=24_000, prem=100.0, direction=+1)
        assert payoff_at_expiry([leg], S_T=24_300.0) == pytest.approx(200 * _LS)

    def test_ce_long_otm(self):
        # Long CE, S_T below strike → worthless, lose premium
        leg = _ce(strike=24_200, prem=80.0, direction=+1)
        assert payoff_at_expiry([leg], S_T=24_000.0) == pytest.approx(-80 * _LS)

    def test_pe_long_itm(self):
        # Long PE at 23800 with premium 70, S_T = 23500 (ITM)
        # pnl = +1 × (300 − 70) × 65 = 14 950
        leg = _pe(strike=23_800, prem=70.0, direction=+1)
        assert payoff_at_expiry([leg], S_T=23_500.0) == pytest.approx(230 * _LS)

    def test_pe_long_otm(self):
        leg = _pe(strike=23_800, prem=70.0, direction=+1)
        assert payoff_at_expiry([leg], S_T=24_000.0) == pytest.approx(-70 * _LS)

    def test_fut_long_profit(self):
        # Long futures at 24000, S_T = 24200 → profit = 200 per unit × 65
        leg = _fut(entry=24_000.0)
        assert payoff_at_expiry([leg], S_T=24_200.0) == pytest.approx(200 * _LS)

    def test_fut_long_loss(self):
        leg = _fut(entry=24_000.0)
        assert payoff_at_expiry([leg], S_T=23_800.0) == pytest.approx(-200 * _LS)

    def test_multi_leg_additivity(self):
        # Long CE + Long PE at same strike = long straddle
        # S_T = 24400 → CE profit = (400-150)×65, PE expires OTM → -140×65
        ce = _ce(strike=24_000, prem=150.0, direction=+1)
        pe = _pe(strike=24_000, prem=140.0, direction=+1)
        expected = (250 * _LS) + (-140 * _LS)
        assert payoff_at_expiry([ce, pe], S_T=24_400.0) == pytest.approx(expected)

    def test_invalid_option_type_raises(self):
        leg = StrategyLeg(
            symbol=_SYM, option_type="INVALID", strike=24_000.0, expiry=_EXPIRY,
            direction=+1, lots=1, lot_size=_LS, entry_premium=100.0,
        )
        with pytest.raises(ValueError, match="Unknown option_type"):
            payoff_at_expiry([leg], S_T=24_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Short Strangle (Strategy 1)
# sell CE(24200)@80 + sell PE(23800)@70, lots=1
# net_credit_per_unit = 150,  net_credit_inr = 150 × 65 = 9 750
# BE_upper = 24 350,  BE_lower = 23 650,  max_loss = ∞
# ─────────────────────────────────────────────────────────────────────────────

class TestShortStrangle:

    def test_max_profit_in_tent(self, builder):
        # S_T = 24000 → both legs expire OTM → total P&L = net credit
        legs = [_ce(24_200, 80.0), _pe(23_800, 70.0)]
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(9_750.0)

    def test_upper_loss_above_be(self, builder):
        # S_T = 24500 (above BE_upper 24350)
        # CE: intrinsic=300, pnl=-1×(300-80)×65=-14300
        # PE: intrinsic=0,   pnl=-1×(0-70)×65=4550
        legs = [_ce(24_200, 80.0), _pe(23_800, 70.0)]
        assert payoff_at_expiry(legs, S_T=24_500.0) == pytest.approx(-14_300 + 4_550)

    def test_lower_loss_below_be(self, builder):
        # S_T = 23500 (below BE_lower 23650)
        legs = [_ce(24_200, 80.0), _pe(23_800, 70.0)]
        # CE: 0, PE: intrinsic=300 → pnl=-1×(300-70)×65=-14950
        assert payoff_at_expiry(legs, S_T=23_500.0) == pytest.approx(5_200 - 14_950)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.short_strangle(
                S=_S, call_strike=24_200, put_strike=23_800,
                call_premium=80.0, put_premium=70.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Short Strangle"
        assert res.net_premium == pytest.approx(9_750.0)
        assert res.max_profit == pytest.approx(9_750.0)
        assert math.isinf(res.max_loss)
        assert res.break_even_upper == pytest.approx(24_350.0)
        assert res.break_even_lower == pytest.approx(23_650.0)
        assert len(res.legs) == 2

    def test_call_not_otm_raises(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            with pytest.raises(ValueError, match="must be > spot"):
                builder.short_strangle(
                    S=24_000, call_strike=23_900,  # call below spot → not OTM
                    put_strike=23_800, call_premium=80, put_premium=70,
                    expiry=_EXPIRY,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Short Straddle (Strategy 2)
# sell CE(24000)@150 + sell PE(24000)@140
# net_credit_per_unit = 290,  net_credit_inr = 290 × 65 = 18 850
# BE_upper = 24 290,  BE_lower = 23 710
# ─────────────────────────────────────────────────────────────────────────────

class TestShortStraddle:

    def test_max_profit_at_strike(self):
        # S_T = 24000 → both expire exactly at strike (no intrinsic)
        legs = [_ce(24_000, 150.0), _pe(24_000, 140.0)]
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(18_850.0)

    def test_loss_above_upper_be(self):
        legs = [_ce(24_000, 150.0), _pe(24_000, 140.0)]
        # CE: intrinsic=400, pnl=-1×(400-150)×65=-16250
        # PE: pnl=-1×(0-140)×65=9100
        assert payoff_at_expiry(legs, S_T=24_400.0) == pytest.approx(-16_250 + 9_100)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.short_straddle(
                S=_S, strike=24_000, call_premium=150.0, put_premium=140.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.net_premium == pytest.approx(18_850.0)
        assert res.max_profit == pytest.approx(18_850.0)
        assert math.isinf(res.max_loss)
        assert res.break_even_upper == pytest.approx(24_290.0)
        assert res.break_even_lower == pytest.approx(23_710.0)


# ─────────────────────────────────────────────────────────────────────────────
# Iron Condor (Strategy 3)
# buy PE(23700)@30, sell PE(23900)@60, sell CE(24100)@55, buy CE(24300)@25
# net_credit_per_unit = (60+55)-(30+25) = 60
# net_credit_inr = 60×65 = 3900
# put_wing = call_wing = 200
# max_loss_per_unit = 200-60 = 140  →  max_loss = 9100
# BE_lower = 23840,  BE_upper = 24160
# ─────────────────────────────────────────────────────────────────────────────

class TestIronCondor:

    _LEGS = lambda self: [
        _pe(23_700, 30.0, direction=+1),   # buy PE K1
        _pe(23_900, 60.0, direction=-1),   # sell PE K2
        _ce(24_100, 55.0, direction=-1),   # sell CE K3
        _ce(24_300, 25.0, direction=+1),   # buy CE K4
    ]

    def test_max_profit_in_tent(self):
        # S_T = 24000 → all expire OTM → net credit
        legs = self._LEGS()
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(3_900.0)

    def test_max_loss_below_k1(self):
        # S_T = 23600 (below K1=23700) → max loss on put side
        legs = self._LEGS()
        # PE buy (23700): intrinsic=100, pnl=+1×(100-30)×65=4550
        # PE sell (23900): intrinsic=300, pnl=-1×(300-60)×65=-15600
        # CE sell (24100): intrinsic=0,   pnl=-1×(0-55)×65=3575
        # CE buy (24300): intrinsic=0,    pnl=+1×(0-25)×65=-1625
        # Total = 4550-15600+3575-1625 = -9100
        assert payoff_at_expiry(legs, S_T=23_600.0) == pytest.approx(-9_100.0)

    def test_max_loss_above_k4(self):
        # S_T = 24500 (above K4=24300) → max loss on call side
        legs = self._LEGS()
        # PE buy: 0, PE sell: 0 → +3900 net on put side
        # CE sell (24100): intrinsic=400, pnl=-1×(400-55)×65=-22425
        # CE buy (24300): intrinsic=200,  pnl=+1×(200-25)×65=11375
        # Total = 1950+3575-22425+11375 = ... let me calculate
        # PE_buy pnl: 0-30)×65 = -1950, PE_sell: -(0-60)×65=3900
        # CE_sell: -(400-55)×65=-22425, CE_buy: (200-25)×65=11375
        # Total = -1950+3900-22425+11375 = -9100
        assert payoff_at_expiry(legs, S_T=24_500.0) == pytest.approx(-9_100.0)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.iron_condor(
                S=_S,
                put_buy_strike=23_700, put_sell_strike=23_900,
                call_sell_strike=24_100, call_buy_strike=24_300,
                put_buy_prem=30.0, put_sell_prem=60.0,
                call_sell_prem=55.0, call_buy_prem=25.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Iron Condor"
        assert res.net_premium == pytest.approx(3_900.0)
        assert res.max_profit == pytest.approx(3_900.0)
        assert res.max_loss == pytest.approx(9_100.0)
        assert res.break_even_lower == pytest.approx(23_840.0)
        assert res.break_even_upper == pytest.approx(24_160.0)

    def test_invalid_strike_order_raises(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            with pytest.raises(ValueError, match="Iron Condor"):
                builder.iron_condor(
                    S=_S,
                    put_buy_strike=23_900, put_sell_strike=23_700,  # K1 > K2 — wrong
                    call_sell_strike=24_100, call_buy_strike=24_300,
                    put_buy_prem=30, put_sell_prem=60,
                    call_sell_prem=55, call_buy_prem=25,
                    expiry=_EXPIRY,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Bull Put Spread (Strategy 4)
# sell PE(23900)@60, buy PE(23700)@30
# net_credit_per_unit = 30, spread = 200
# max_profit = 30×65 = 1950,  max_loss = (200-30)×65 = 11050
# BE = 23900-30 = 23870
# ─────────────────────────────────────────────────────────────────────────────

class TestBullPutSpread:

    def test_max_profit_above_sell_strike(self):
        legs = [_pe(23_900, 60.0, direction=-1), _pe(23_700, 30.0, direction=+1)]
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(1_950.0)

    def test_max_loss_below_buy_strike(self):
        legs = [_pe(23_900, 60.0, direction=-1), _pe(23_700, 30.0, direction=+1)]
        # PE sell (23900): intrinsic=300, pnl=-1×(300-60)×65=-15600
        # PE buy (23700): intrinsic=100,  pnl=+1×(100-30)×65=4550
        assert payoff_at_expiry(legs, S_T=23_600.0) == pytest.approx(-11_050.0)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.bull_put_spread(
                S=_S, sell_strike=23_900, buy_strike=23_700,
                sell_premium=60.0, buy_premium=30.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Bull Put Spread"
        assert res.net_premium == pytest.approx(1_950.0)
        assert res.max_profit == pytest.approx(1_950.0)
        assert res.max_loss == pytest.approx(11_050.0)
        assert res.break_even_lower == pytest.approx(23_870.0)
        assert res.break_even_upper is None  # single break-even strategy


# ─────────────────────────────────────────────────────────────────────────────
# Bear Call Spread (Strategy 5)
# sell CE(24100)@55, buy CE(24300)@25
# net_credit_per_unit = 30, spread = 200
# max_profit = 1950,  max_loss = 11050,  BE = 24100+30 = 24130
# ─────────────────────────────────────────────────────────────────────────────

class TestBearCallSpread:

    def test_max_profit_below_sell_strike(self):
        legs = [_ce(24_100, 55.0, direction=-1), _ce(24_300, 25.0, direction=+1)]
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(1_950.0)

    def test_max_loss_above_buy_strike(self):
        legs = [_ce(24_100, 55.0, direction=-1), _ce(24_300, 25.0, direction=+1)]
        # CE sell (24100): intrinsic=400, pnl=-1×(400-55)×65=-22425
        # CE buy  (24300): intrinsic=200, pnl=+1×(200-25)×65=11375
        assert payoff_at_expiry(legs, S_T=24_500.0) == pytest.approx(-11_050.0)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.bear_call_spread(
                S=_S, sell_strike=24_100, buy_strike=24_300,
                sell_premium=55.0, buy_premium=25.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Bear Call Spread"
        assert res.max_profit == pytest.approx(1_950.0)
        assert res.max_loss == pytest.approx(11_050.0)
        assert res.break_even_upper == pytest.approx(24_130.0)
        assert res.break_even_lower is None


# ─────────────────────────────────────────────────────────────────────────────
# Long Straddle (Strategy 6)
# buy CE(24000)@150 + buy PE(24000)@140
# net_debit_per_unit = 290, max_loss = 18850, max_profit = ∞
# BE_upper = 24290,  BE_lower = 23710
# ─────────────────────────────────────────────────────────────────────────────

class TestLongStraddle:

    def test_max_loss_at_strike(self):
        # S_T = 24000 → both expire OTM → lose entire debit
        legs = [_ce(24_000, 150.0, direction=+1), _pe(24_000, 140.0, direction=+1)]
        assert payoff_at_expiry(legs, S_T=24_000.0) == pytest.approx(-18_850.0)

    def test_profit_above_upper_be(self):
        legs = [_ce(24_000, 150.0, direction=+1), _pe(24_000, 140.0, direction=+1)]
        # S_T=24400: CE intrinsic=400→(400-150)×65=16250, PE OTM→-140×65=-9100
        assert payoff_at_expiry(legs, S_T=24_400.0) == pytest.approx(16_250 - 9_100)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.long_straddle(
                S=_S, strike=24_000, call_premium=150.0, put_premium=140.0,
                expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Long Straddle"
        assert res.net_premium == pytest.approx(-18_850.0)   # debit → negative
        assert res.max_loss == pytest.approx(18_850.0)
        assert math.isinf(res.max_profit)
        assert res.break_even_upper == pytest.approx(24_290.0)
        assert res.break_even_lower == pytest.approx(23_710.0)


# ─────────────────────────────────────────────────────────────────────────────
# Covered Future + Short Call (Strategy 7)
# long FUT@24000, short CE(24200)@80
# max_profit = (24200-24000+80)×65 = 280×65 = 18200
# BE_lower = 24000-80 = 23920, max_loss = ∞
# ─────────────────────────────────────────────────────────────────────────────

class TestCoveredFutureShortCall:

    def test_capped_profit_above_call_strike(self):
        # S_T = 24500: FUT pnl=500×65=32500, CE pnl=-1×(300-80)×65=-14300
        legs = [_fut(24_000.0, direction=+1), _ce(24_200, 80.0, direction=-1)]
        assert payoff_at_expiry(legs, S_T=24_500.0) == pytest.approx(18_200.0)

    def test_breakeven_at_futures_entry_minus_premium(self):
        # S_T = 23920 (= be_lower): FUT pnl=-80×65=-5200, CE pnl=+80×65=5200
        legs = [_fut(24_000.0, direction=+1), _ce(24_200, 80.0, direction=-1)]
        assert payoff_at_expiry(legs, S_T=23_920.0) == pytest.approx(0.0, abs=1e-6)

    def test_loss_below_breakeven(self):
        legs = [_fut(24_000.0, direction=+1), _ce(24_200, 80.0, direction=-1)]
        # S_T = 23800: FUT=-200×65=-13000, CE=+80×65=5200 → total -7800
        assert payoff_at_expiry(legs, S_T=23_800.0) == pytest.approx(-7_800.0)

    def test_strategy_result_metrics(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.covered_future_short_call(
                S=_S, futures_entry=24_000.0, call_strike=24_200,
                call_premium=80.0, expiry=_EXPIRY, lots=1,
            )
        assert res.name == "Covered Future + Short Call"
        assert res.max_profit == pytest.approx(18_200.0)
        assert math.isinf(res.max_loss)
        assert res.break_even_lower == pytest.approx(23_920.0)
        assert res.break_even_upper is None

    def test_call_not_otm_raises(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            with pytest.raises(ValueError, match="call_strike"):
                builder.covered_future_short_call(
                    S=_S, futures_entry=24_200.0, call_strike=24_000,  # ITM
                    call_premium=80.0, expiry=_EXPIRY,
                )


# ─────────────────────────────────────────────────────────────────────────────
# payoff_curve helper
# ─────────────────────────────────────────────────────────────────────────────

class TestPayoffCurve:

    def test_returns_correct_length(self):
        legs = [_ce(24_200, 80.0)]
        spots, pnls = payoff_curve(legs, S_range=(23_000, 25_000), n_points=50)
        assert len(spots) == 50
        assert len(pnls) == 50

    def test_first_and_last_spot(self):
        legs = [_ce(24_200, 80.0)]
        spots, _ = payoff_curve(legs, S_range=(23_000.0, 25_000.0), n_points=5)
        assert spots[0] == pytest.approx(23_000.0)
        assert spots[-1] == pytest.approx(25_000.0)

    def test_invalid_range_raises(self):
        legs = [_ce(24_200, 80.0)]
        with pytest.raises(ValueError, match="S_min < S_max"):
            payoff_curve(legs, S_range=(25_000, 23_000))


# ─────────────────────────────────────────────────────────────────────────────
# validate_strategy_legs
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateStrategyLegs:

    def test_valid_ce_leg_no_errors(self):
        leg = _ce(strike=24_200, prem=80.0, direction=-1)
        errors = validate_strategy_legs([leg])
        assert errors == []

    def test_empty_legs_returns_error(self):
        errors = validate_strategy_legs([])
        assert any("at least one leg" in e for e in errors)

    def test_bad_direction_returns_error(self):
        leg = StrategyLeg(
            symbol=_SYM, option_type="CE", strike=24_200, expiry=_EXPIRY,
            direction=0, lots=1, lot_size=_LS, entry_premium=80.0,  # 0 is invalid
        )
        errors = validate_strategy_legs([leg])
        assert any("direction" in e for e in errors)

    def test_zero_lots_returns_error(self):
        leg = StrategyLeg(
            symbol=_SYM, option_type="CE", strike=24_200, expiry=_EXPIRY,
            direction=-1, lots=0, lot_size=_LS, entry_premium=80.0,
        )
        errors = validate_strategy_legs([leg])
        assert any("lots" in e for e in errors)

    def test_unknown_symbol_returns_error(self):
        leg = StrategyLeg(
            symbol="FAKESYMBOL", option_type="CE", strike=24_200, expiry=_EXPIRY,
            direction=-1, lots=1, lot_size=_LS, entry_premium=80.0,
        )
        errors = validate_strategy_legs([leg])
        assert any("not found" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# StrategyResult.summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyResultSummary:

    def test_summary_contains_strategy_name(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.short_straddle(
                S=_S, strike=24_000, call_premium=150.0, put_premium=140.0,
                expiry=_EXPIRY,
            )
        summary = res.summary()
        assert "Short Straddle" in summary
        assert "max_profit" in summary

    def test_summary_shows_inf_for_unlimited_loss(self, builder):
        with patch(_PATCH_TIME, return_value=None):
            res = builder.short_strangle(
                S=_S, call_strike=24_200, put_strike=23_800,
                call_premium=80.0, put_premium=70.0, expiry=_EXPIRY,
            )
        assert "∞" in res.summary()
