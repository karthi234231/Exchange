"""
tests/test_stress_engine.py — Comprehensive tests for StressEngine.

Covers P&L math, Greek contributions, VIX action ladder, boundary conditions,
scenario count, and the WARNING / worst-case helpers.
"""
from __future__ import annotations

import pytest

from algo_trading.risk.stress_engine import (
    SCENARIOS,
    ScenarioPnL,
    StressEngine,
    StressReport,
    StressResult,
    StressScenario,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_UNDERLYING = 22_000.0
_IV = 0.15  # 15% annualised ATM IV


def _pos(
    delta: float = 0.0,
    gamma: float = 0.0,
    vega: float = 0.0,
    lots: int = 1,
    lot_size: int = 50,
    direction: int = 1,
    pid: str = "P1",
) -> dict:
    return dict(
        position_id=pid,
        lots=lots,
        lot_size=lot_size,
        direction=direction,
        delta=delta,
        gamma=gamma,
        vega=vega,
    )


def _engine(*custom_scenarios: StressScenario) -> StressEngine:
    """Return engine with custom scenario list, or default if none given."""
    return StressEngine(list(custom_scenarios) if custom_scenarios else None)


def _scenario(name: str, shock: float, vol_mult: float = 1.0) -> StressScenario:
    return StressScenario(name, shock, vol_mult, "test scenario")


def _run(engine: StressEngine, positions: list[dict]) -> StressReport:
    return engine.run(positions, _UNDERLYING, _IV)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEmptyPositions:
    """Test 1: Empty portfolio → all P&L components are zero in every scenario."""

    def test_empty_positions_all_zero(self):
        report = _run(_engine(), [])
        for result in report.scenarios:
            assert result.portfolio_pnl == pytest.approx(0.0), result.scenario.name
            assert result.delta_contribution == pytest.approx(0.0)
            assert result.gamma_contribution == pytest.approx(0.0)
            assert result.vega_contribution == pytest.approx(0.0)
            assert result.position_pnls == []


class TestDeltaPnLProportional:
    """Test 2: Pure delta position — P&L scales linearly with shock."""

    def test_delta_pnl_proportional_up5pct(self):
        sc = _scenario("UP_5PCT", +0.05, 1.0)
        engine = _engine(sc)
        lots, lot_size, direction, delta = 2, 50, 1, 0.5
        report = engine.run([_pos(delta=delta, lots=lots, lot_size=lot_size, direction=direction)],
                            _UNDERLYING, _IV)
        result = report.scenarios[0]
        delta_s = _UNDERLYING * 0.05
        units = lots * lot_size * direction
        expected_delta_pnl = delta * units * delta_s
        assert result.delta_contribution == pytest.approx(expected_delta_pnl, rel=1e-9)
        assert result.gamma_contribution == pytest.approx(0.0, abs=1e-9)
        assert result.vega_contribution == pytest.approx(0.0, abs=1e-9)
        assert result.portfolio_pnl == pytest.approx(expected_delta_pnl, rel=1e-9)


class TestGammaPnLSecondOrder:
    """Test 3: Pure gamma position — gamma P&L is always positive for a down move."""

    def test_gamma_pnl_positive_down5pct(self):
        sc = _scenario("DOWN_5PCT", -0.05, 1.0)
        engine = _engine(sc)
        lots, lot_size, direction, gamma = 1, 50, 1, 0.001
        report = engine.run(
            [_pos(gamma=gamma, lots=lots, lot_size=lot_size, direction=direction)],
            _UNDERLYING, _IV,
        )
        result = report.scenarios[0]
        delta_s = _UNDERLYING * (-0.05)
        units = lots * lot_size * direction
        expected_gamma_pnl = 0.5 * gamma * units * (delta_s ** 2)
        assert result.gamma_contribution == pytest.approx(expected_gamma_pnl, rel=1e-9)
        # Gamma P&L must be positive (long option, convex payoff)
        assert result.gamma_contribution > 0.0
        assert result.delta_contribution == pytest.approx(0.0, abs=1e-9)


class TestVegaPnLDirection:
    """Test 4: Short-vega position suffers when vol expands.

    Model formula: vp = vega * lots * direction * (delta_sigma * 100)
    Short-vega representation: direction=+1, vega=-50 (negative vega per unit,
    long direction). This gives vp < 0 when vol rises.
    """

    def test_short_vega_loses_on_vix_2x(self):
        sc = _scenario("VIX_2X", 0.0, 2.0)
        engine = _engine(sc)
        # direction=+1, vega=-50 (negative vega per unit → short vega exposure)
        lots, lot_size, direction, vega = 2, 50, 1, -50.0
        report = engine.run(
            [_pos(vega=vega, lots=lots, lot_size=lot_size, direction=direction)],
            _UNDERLYING, _IV,
        )
        result = report.scenarios[0]
        # vol doubles → delta_sigma = IV * (2.0 - 1.0) = IV
        delta_sigma = _IV * (2.0 - 1.0)
        expected_vega_pnl = vega * lots * direction * (delta_sigma * 100)
        assert result.vega_contribution == pytest.approx(expected_vega_pnl, rel=1e-9)
        assert result.vega_contribution < 0.0, "Short vega must lose when vol spikes"


class TestWorstCaseScenario:
    """Test 5: DOWN_10PCT_VIX_3X should be worst for a long-delta/short-vega position.

    Position: direction=+1, delta=+0.5, vega=-200 (large short-vega, long-delta).
    Worst case = big down move (delta loss) combined with vol spike (vega loss).
    """

    def test_worst_case_among_most_negative_for_short_vega(self):
        engine = _engine()
        # long direction, positive delta (loses on down), negative vega (loses on vol spike)
        pos = _pos(delta=0.5, gamma=0.0, vega=-200.0, lots=5, lot_size=50, direction=1)
        report = _run(engine, [pos])
        worst = report.worst_case()
        assert worst is not None
        # Worst P&L: large down + large vol spike hurts both delta and vega
        adverse_names = {"DOWN_10PCT_VIX_3X", "DOWN_15PCT_VIX_4X", "DOWN_5PCT_VIX_2X"}
        assert worst.scenario.name in adverse_names, (
            f"Expected adverse combined scenario, got {worst.scenario.name}"
        )


class TestWorstPnlInr:
    """Test 6: worst_pnl_inr() returns the minimum scenario P&L."""

    def test_worst_pnl_inr_matches_worst_case(self):
        engine = _engine()
        pos = _pos(delta=0.5, gamma=0.001, vega=50.0, lots=3, lot_size=50, direction=1)
        report = _run(engine, [pos])
        worst = report.worst_case()
        assert report.worst_pnl_inr() == pytest.approx(worst.portfolio_pnl, rel=1e-9)


class TestPnlAsPctOfCapital:
    """Test 7: pnl_as_pct_of_capital arithmetic."""

    def test_pnl_as_pct_of_capital_correct(self):
        sc = _scenario("SC", +0.10, 1.0)
        engine = _engine(sc)
        pos = _pos(delta=0.5, lots=1, lot_size=50, direction=1)
        report = _run(engine, [pos])
        result = report.scenarios[0]
        capital = 100_000.0
        expected_pct = result.portfolio_pnl / capital * 100.0
        assert result.pnl_as_pct_of_capital(capital) == pytest.approx(expected_pct, rel=1e-9)

    def test_pnl_as_pct_zero_capital_returns_zero(self):
        sc = _scenario("SC2", -0.05, 1.0)
        result = StressResult(
            scenario=sc,
            portfolio_pnl=5000.0,
            delta_contribution=5000.0,
            gamma_contribution=0.0,
            vega_contribution=0.0,
        )
        assert result.pnl_as_pct_of_capital(0.0) == 0.0


class TestVixActionNormal:
    """Test 8: VIX ≤ 22 → NORMAL."""

    def test_vix_action_normal(self):
        action, _ = _engine().vix_action_required(15.0)
        assert action == "NORMAL"


class TestVixActionReduce50Pct:
    """Test 9: VIX > 22 and ≤ 25 → REDUCE_SIZE_50PCT."""

    def test_vix_action_reduce_50pct(self):
        action, desc = _engine().vix_action_required(23.0)
        assert action == "REDUCE_SIZE_50PCT"
        assert "23.0" in desc or "23" in desc


class TestVixActionNoNewPositions:
    """Test 10: VIX > 25 and ≤ 30 → NO_NEW_POSITIONS."""

    def test_vix_action_no_new_positions(self):
        action, _ = _engine().vix_action_required(27.0)
        assert action == "NO_NEW_POSITIONS"


class TestVixActionEmergencyFlatten:
    """Test 11: VIX > 30 → EMERGENCY_FLATTEN."""

    def test_vix_action_emergency_flatten(self):
        action, _ = _engine().vix_action_required(31.0)
        assert action == "EMERGENCY_FLATTEN"


class TestVixBoundary30:
    """Test 12: VIX exactly 30.0 is NOT > 30 → NO_NEW_POSITIONS (not EMERGENCY)."""

    def test_vix_boundary_exactly_30(self):
        action, _ = _engine().vix_action_required(30.0)
        assert action == "NO_NEW_POSITIONS"


class TestVixBoundary22:
    """Test 13: VIX exactly 22.0 is NOT > 22 → NORMAL (not REDUCE)."""

    def test_vix_boundary_exactly_22(self):
        action, _ = _engine().vix_action_required(22.0)
        assert action == "NORMAL"


class TestStandardScenariosCount:
    """Test 14: Standard scenario list has at least 10 entries."""

    def test_standard_scenarios_count_gte_10(self):
        assert len(SCENARIOS) >= 10


class TestStressResultWarning:
    """Test 15: WARNING field on StressResult contains 'APPROX'."""

    def test_stress_result_has_approx_in_warning(self):
        sc = _scenario("T", 0.0, 1.0)
        result = StressResult(
            scenario=sc,
            portfolio_pnl=0.0,
            delta_contribution=0.0,
            gamma_contribution=0.0,
            vega_contribution=0.0,
        )
        assert "APPROX" in result.WARNING

    def test_engine_results_all_carry_warning(self):
        report = _run(_engine(), [_pos(delta=0.5)])
        for r in report.scenarios:
            assert "APPROX" in r.WARNING


class TestWorstCaseEmptyReport:
    """Test 16: worst_case() on a report with no scenarios returns None."""

    def test_worst_case_empty_report_returns_none(self):
        report = StressReport(underlying_price=_UNDERLYING, current_iv=_IV, scenarios=[])
        assert report.worst_case() is None
        assert report.worst_pnl_inr() == 0.0
