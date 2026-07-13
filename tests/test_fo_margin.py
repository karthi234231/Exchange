"""
tests/test_fo_margin.py — Unit tests for algo_trading/fo/margin.py

Verifies:
  - SPANMarginEngine.compute_span() produces 16 scenarios
  - Margin is positive for short options and zero for empty portfolio
  - Short Option Minimum Charge (SOMC): applies to short options only
  - Long options: SOMC = 0
  - Futures leg: linear P&L, SOMC = 0
  - total_initial_margin = span_margin + exposure_margin
  - WARNING field always contains APPROX disclaimer
  - Down scenarios hurt short calls, up scenarios hurt short puts
  - Worst-case scenario is the scenario with minimum P&L
"""
from __future__ import annotations

import math
import pytest
from datetime import date

from algo_trading.fo.margin import SPANMarginEngine, SPANResult, StrategyLeg
from algo_trading.fo.pricing import BSMEngine


# ── Helpers ──────────────────────────────────────────────────────────────────

BSM = BSMEngine(r=0.065, q=0.015)
ENGINE = SPANMarginEngine(bsm_engine=BSM)

EXPIRY   = date(2026, 6, 26)
UNDERLY  = 24000.0
VOL      = 0.15
T_30D    = 30.0 / 365.0
LOT_SIZE = 65


def short_ce(strike: float = 24500.0) -> StrategyLeg:
    return StrategyLeg(
        option_type="CE", strike=strike, expiry=EXPIRY,
        direction=-1, lots=1, lot_size=LOT_SIZE,
        entry_premium=BSM.call(S=UNDERLY, K=strike, T=T_30D, sigma=VOL),
    )


def short_pe(strike: float = 23500.0) -> StrategyLeg:
    return StrategyLeg(
        option_type="PE", strike=strike, expiry=EXPIRY,
        direction=-1, lots=1, lot_size=LOT_SIZE,
        entry_premium=BSM.put(S=UNDERLY, K=strike, T=T_30D, sigma=VOL),
    )


def long_ce(strike: float = 26000.0) -> StrategyLeg:
    return StrategyLeg(
        option_type="CE", strike=strike, expiry=EXPIRY,
        direction=+1, lots=1, lot_size=LOT_SIZE,
        entry_premium=BSM.call(S=UNDERLY, K=strike, T=T_30D, sigma=VOL),
    )


def long_future(entry_px: float = 24050.0) -> StrategyLeg:
    return StrategyLeg(
        option_type="FUT", strike=0.0, expiry=EXPIRY,
        direction=+1, lots=1, lot_size=LOT_SIZE,
        entry_premium=entry_px,
    )


# ── Scenario count ────────────────────────────────────────────────────────────

class TestSPANScenarios:
    def test_16_scenarios_returned(self):
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert len(result.all_scenarios) == 16

    def test_scenario_ids_are_1_to_16(self):
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        ids = {s.scenario_id for s in result.all_scenarios}
        assert ids == set(range(1, 17))


# ── Margin properties ─────────────────────────────────────────────────────────

class TestMarginProperties:
    def test_short_option_margin_positive(self):
        """Short option must require positive margin."""
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert result.total_initial_margin > 0

    def test_total_margin_equals_span_plus_exposure(self):
        """total_initial_margin = span_margin + exposure_margin."""
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert abs(result.total_initial_margin
                   - (result.span_margin + result.exposure_margin)) < 0.01

    def test_span_margin_ge_somc_for_short_option(self):
        """SPAN margin must be ≥ SOMC (SOMC is the minimum charge floor)."""
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert result.span_margin >= result.somc - 0.01

    def test_long_option_zero_somc(self):
        """Long options do not incur SOMC."""
        result = ENGINE.compute_span([long_ce()], UNDERLY, VOL, T_30D)
        assert result.somc == 0.0

    def test_futures_zero_somc(self):
        """Futures legs do not incur SOMC."""
        result = ENGINE.compute_span([long_future()], UNDERLY, VOL, T_30D)
        assert result.somc == 0.0

    def test_warning_contains_approx(self):
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert "APPROX" in result.WARNING

    def test_worst_scenario_has_minimum_pnl(self):
        """worst_scenario must be the scenario with the minimum P&L."""
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        min_pnl = min(s.pnl for s in result.all_scenarios)
        assert result.worst_scenario.pnl == pytest.approx(min_pnl, abs=0.01)


# ── Directional intuition ─────────────────────────────────────────────────────

class TestDirectionalIntuition:
    def test_short_call_hurt_by_up_moves(self):
        """A short call loses money when the underlying moves up significantly."""
        result = ENGINE.compute_span([short_ce(24500.0)], UNDERLY, VOL, T_30D)
        # Scenario 1: +1σ vol up = large up move → worst for short call
        up_scenarios = [s for s in result.all_scenarios
                        if s.price_move_sigma > 0.5]
        down_scenarios = [s for s in result.all_scenarios
                          if s.price_move_sigma < -0.5]
        if up_scenarios and down_scenarios:
            avg_up_pnl = sum(s.pnl for s in up_scenarios) / len(up_scenarios)
            avg_down_pnl = sum(s.pnl for s in down_scenarios) / len(down_scenarios)
            # For an OTM short call: up moves hurt, down moves help
            assert avg_up_pnl < avg_down_pnl

    def test_short_strangle_worst_case_is_large_move(self):
        """For a short strangle, worst case is a large directional move."""
        result = ENGINE.compute_span(
            [short_ce(24500.0), short_pe(23500.0)],
            UNDERLY, VOL, T_30D,
        )
        # Extreme scenarios (15, 16) should be among the worst
        extreme_pnls = [s.pnl for s in result.all_scenarios
                        if s.scenario_id in (15, 16)]
        all_pnls = [s.pnl for s in result.all_scenarios]
        # At least one extreme is in the worst 3 scenarios
        worst_3_pnls = sorted(all_pnls)[:3]
        assert any(p in worst_3_pnls for p in extreme_pnls)


# ── Strategy combinations ─────────────────────────────────────────────────────

class TestStrategyCombinations:
    def test_long_option_reduces_margin_vs_naked_short(self):
        """Buying protection (long leg) reduces margin for short option."""
        naked_result  = ENGINE.compute_span([short_ce(24500.0)], UNDERLY, VOL, T_30D)
        spread_result = ENGINE.compute_span(
            [short_ce(24500.0), long_ce(25000.0)],
            UNDERLY, VOL, T_30D,
        )
        assert spread_result.span_margin <= naked_result.span_margin

    def test_empty_legs_returns_zero_margin(self):
        """No legs → zero margin."""
        result = ENGINE.compute_span([], UNDERLY, VOL, T_30D)
        assert result.total_initial_margin == pytest.approx(0.0, abs=1.0)
        assert len(result.all_scenarios) == 16


# ── SPAN result attributes ────────────────────────────────────────────────────

class TestSPANResultAttributes:
    def test_span_result_has_all_attributes(self):
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        assert isinstance(result, SPANResult)
        assert isinstance(result.span_margin, float)
        assert isinstance(result.somc, float)
        assert isinstance(result.exposure_margin, float)
        assert isinstance(result.total_initial_margin, float)
        assert result.worst_scenario is not None
        assert len(result.all_scenarios) == 16

    def test_scenario_pnl_fields(self):
        result = ENGINE.compute_span([short_ce()], UNDERLY, VOL, T_30D)
        for s in result.all_scenarios:
            assert isinstance(s.scenario_id, int)
            assert isinstance(s.pnl, float)
            assert isinstance(s.price_move_inr, float)
            assert isinstance(s.vol_multiplier, float)
