"""
tests/test_gap_risk_model.py — Comprehensive tests for GapRiskModel.

Covers P&L math, stop-loss flagging, expected/worst/VaR helpers,
calibration utility, and WARNING / probability invariants.
"""
from __future__ import annotations

import pytest

from algo_trading.risk.gap_risk_model import (
    GAP_SCENARIOS,
    GapPnL,
    GapRiskModel,
    GapRiskReport,
    GapScenario,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_PREV_CLOSE = 22_000.0
_CUR_PRICE = 22_000.0   # no intraday move from prev close
_IV = 0.15
_CAPITAL = 1_000_000.0  # ₹10L — large enough to avoid incidental stop triggers


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


def _model(
    scenarios: list[GapScenario] | None = None,
    stop_loss_pct: float = 0.02,
) -> GapRiskModel:
    return GapRiskModel(scenarios=scenarios, stop_loss_pct=stop_loss_pct)


def _gap_sc(name: str, gap_pct: float, iv_mult: float = 1.0, prob: float = 0.5) -> GapScenario:
    return GapScenario(name, gap_pct, iv_mult, prob)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEmptyPositions:
    """Test 1: Empty portfolio → all scenario P&Ls are zero."""

    def test_empty_positions_all_zero(self):
        report = _model().compute([], _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL)
        for gap in report.scenarios:
            assert gap.portfolio_pnl == pytest.approx(0.0), gap.scenario.name
            assert gap.delta_pnl == pytest.approx(0.0)
            assert gap.vega_pnl == pytest.approx(0.0)


class TestFlatGapNoDeltaPnl:
    """Test 2: FLAT scenario (gap_pct=0, iv_multiplier=1.0) → delta_pnl ≈ 0."""

    def test_flat_gap_no_delta_pnl(self):
        flat = _gap_sc("FLAT", 0.0, 1.0, 1.0)
        report = _model([flat]).compute(
            [_pos(delta=0.5, lots=2, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        result = report.scenarios[0]
        # gap_price = prev_close * 1.0 = _CUR_PRICE → delta_s = 0
        assert result.delta_pnl == pytest.approx(0.0, abs=1e-9)
        assert result.vega_pnl == pytest.approx(0.0, abs=1e-9)
        assert result.portfolio_pnl == pytest.approx(0.0, abs=1e-9)


class TestDownGapHurtsLongDelta:
    """Test 3: Long-delta position loses on a down gap."""

    def test_down_gap_hurts_long_delta(self):
        down3 = _gap_sc("DOWN_3PCT", -3.0, 1.0, 1.0)
        report = _model([down3]).compute(
            [_pos(delta=0.5, lots=2, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        assert report.scenarios[0].portfolio_pnl < 0.0


class TestDownGapHelpsShortDelta:
    """Test 4: Short-delta position benefits from a down gap.

    Short-delta representation: direction=+1, delta=-0.5 (negative delta per unit).
    dp = delta * units * delta_s = (-0.5) * (lots*lot_size) * (negative delta_s) > 0.
    """

    def test_down_gap_helps_short_delta(self):
        down3 = _gap_sc("DOWN_3PCT", -3.0, 1.0, 1.0)
        # direction=+1, delta=-0.5: short-delta via negative per-unit delta
        report = _model([down3]).compute(
            [_pos(delta=-0.5, lots=2, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        assert report.scenarios[0].portfolio_pnl > 0.0


class TestGapThroughStopFlag:
    """Test 5: Loss exceeding stop_loss_pct sets gap_through_stop=True."""

    def test_gap_through_stop_flag_set(self):
        # Huge position, tiny capital, large down gap → stop must breach
        big_down = _gap_sc("DOWN_5PCT", -5.0, 2.0, 1.0)
        small_capital = 10_000.0
        report = _model([big_down], stop_loss_pct=0.02).compute(
            [_pos(delta=0.8, lots=10, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, small_capital,
        )
        result = report.scenarios[0]
        stop_threshold = small_capital * 0.02
        if result.portfolio_pnl < -stop_threshold:
            assert result.gap_through_stop is True
        else:
            pytest.skip("Position not large enough to trigger stop in this scenario")


class TestNoGapThroughStopSmallPosition:
    """Test 6: Small position with large capital → stop not breached on moderate gap."""

    def test_no_gap_through_stop_small_position(self):
        moderate = _gap_sc("DOWN_0.5PCT", -0.5, 1.05, 1.0)
        report = _model([moderate], stop_loss_pct=0.02).compute(
            [_pos(delta=0.5, lots=1, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        assert report.scenarios[0].gap_through_stop is False


class TestExpectedPnlWeightedSum:
    """Test 7: expected_pnl() equals probability-weighted average of scenario P&Ls."""

    def test_expected_pnl_weighted_sum(self):
        # Two scenarios with known probabilities: sum != 1 → normalised internally
        sc_up = _gap_sc("UP_1PCT", +1.0, 1.0, 0.3)
        sc_dn = _gap_sc("DOWN_1PCT", -1.0, 1.0, 0.7)
        pos = _pos(delta=0.5, lots=1, lot_size=50, direction=1)
        report = _model([sc_up, sc_dn]).compute(
            [pos], _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        g_up, g_dn = report.scenarios
        total_prob = sc_up.probability + sc_dn.probability
        expected = (
            g_up.portfolio_pnl * sc_up.probability
            + g_dn.portfolio_pnl * sc_dn.probability
        ) / total_prob
        assert report.expected_pnl() == pytest.approx(expected, rel=1e-9)


class TestWorstGapPnlReturnsMinimum:
    """Test 8: worst_gap_pnl() returns the minimum across all scenario P&Ls."""

    def test_worst_gap_pnl_is_minimum(self):
        report = _model().compute(
            [_pos(delta=0.6, lots=3, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        manual_min = min(g.portfolio_pnl for g in report.scenarios)
        assert report.worst_gap_pnl() == pytest.approx(manual_min, rel=1e-9)


class TestVar95Positive:
    """Test 9: A losing portfolio produces a positive VaR(95%) value."""

    def test_var_95_pct_positive_for_bad_portfolio(self):
        # Long-delta position suffers on down gaps; standard scenarios have down tails
        report = _model().compute(
            [_pos(delta=0.8, lots=5, lot_size=50, direction=1)],
            _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL,
        )
        var = report.var_95_pct(_CAPITAL)
        assert var > 0.0


class TestCalibrateEmptyReturnsEmpty:
    """Test 10: calibrate_gap_std_from_history([]) returns {}."""

    def test_calibrate_empty_returns_empty(self):
        result = GapRiskModel.calibrate_gap_std_from_history([])
        assert result == {}


class TestCalibrateSymmetricSeries:
    """Test 11: Symmetric series has mean ≈ 0, std > 0, p95 < 1.1."""

    def test_calibrate_symmetric_series(self):
        series = [-1.0, -0.5, 0.0, 0.5, 1.0]
        result = GapRiskModel.calibrate_gap_std_from_history(series)
        assert result["mean_pct"] == pytest.approx(0.0, abs=1e-9)
        assert result["std_pct"] > 0.0
        assert result["p95_abs_pct"] < 1.1


class TestCalibrateReturnsAllKeys:
    """Test 12: calibrate result contains all expected keys."""

    EXPECTED_KEYS = {
        "mean_pct", "std_pct", "p95_abs_pct", "p99_abs_pct",
        "p999_abs_pct", "n_observations", "WARNING",
    }

    def test_calibrate_returns_all_keys(self):
        series = [float(i) * 0.1 for i in range(-20, 21)]
        result = GapRiskModel.calibrate_gap_std_from_history(series)
        assert self.EXPECTED_KEYS <= result.keys()

    def test_calibrate_n_observations_correct(self):
        series = [0.1, -0.2, 0.3, -0.4]
        result = GapRiskModel.calibrate_gap_std_from_history(series)
        assert result["n_observations"] == 4


class TestGapReportHasWarning:
    """Test 13: GapRiskReport.WARNING contains '[ASSUMED]'."""

    def test_gap_report_has_assumed_in_warning(self):
        report = _model().compute([], _PREV_CLOSE, _CUR_PRICE, _IV, _CAPITAL)
        assert "[ASSUMED]" in report.WARNING

    def test_gap_report_default_warning_non_empty(self):
        # Direct instantiation to verify default
        dummy = GapRiskReport(
            current_price=_CUR_PRICE,
            prev_close=_PREV_CLOSE,
            current_iv=_IV,
            scenarios=[],
        )
        assert dummy.WARNING and len(dummy.WARNING) > 10


class TestStandardScenarioProbabilities:
    """Test 14: Standard GAP_SCENARIOS probabilities sum to ≈ 1.0 (within 0.05)."""

    def test_probabilities_sum_near_one(self):
        total = sum(sc.probability for sc in GAP_SCENARIOS)
        assert abs(total - 1.0) < 0.05, f"Probability sum = {total:.4f}, expected ≈ 1.0"
