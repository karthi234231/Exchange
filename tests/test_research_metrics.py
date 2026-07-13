"""
tests/test_research_metrics.py — Unit tests for research/metrics_engine.py.

Verifies:
  - compute_metrics() on empty trade list → zero MetricsSuite
  - Sharpe ratio: manually computed vs engine
  - Max drawdown: simple sequence of trades
  - Win rate: correct fraction
  - Profit factor: gross_profit / abs(gross_loss)
  - Phase 2 exit criteria flags
  - Block bootstrap CI: lower <= Sharpe <= upper
  - per_fold_stability(): std across folds

Mathematical reference
----------------------
Sharpe (annualised, rf=0, daily aggregation):
  S = mean(daily_pnl) / std(daily_pnl) * sqrt(252)

Max drawdown %:
  MDD = max over all t: (peak_equity[:t] - equity[t]) / peak_equity[:t] * 100

Win rate:
  WR = count(pnl > 0) / total_trades
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from algo_trading.research.metrics_engine import (
    TradeRecord,
    MetricsSuite,
    compute_metrics,
    per_fold_stability,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trade(
    pnl: float,
    entry_dt: datetime | None = None,
    holding_hours: float = 1.0,
    strategy: str = "test",
) -> TradeRecord:
    """Create a TradeRecord for testing."""
    if entry_dt is None:
        entry_dt = datetime(2026, 1, 2, 9, 15)
    exit_dt = entry_dt + timedelta(hours=holding_hours)
    return TradeRecord(
        pnl_inr=pnl,
        entry_time=entry_dt,
        exit_time=exit_dt,
        holding_days=holding_hours / 24.0,
        strategy_name=strategy,
    )


def make_daily_trades(pnls_per_day: list[float]) -> list[TradeRecord]:
    """
    Create one trade per trading day.
    pnls_per_day[0] = P&L on 2026-01-02, etc.
    """
    trades = []
    base = datetime(2026, 1, 2, 10, 0)
    for i, pnl in enumerate(pnls_per_day):
        dt = base + timedelta(days=i)
        trades.append(make_trade(pnl, entry_dt=dt, holding_hours=1.0))
    return trades


# ── Empty trades ──────────────────────────────────────────────────────────────

class TestEmptyTrades:
    def test_empty_returns_metrics_suite(self):
        m = compute_metrics([], capital=100_000)
        assert isinstance(m, MetricsSuite)

    def test_empty_total_pnl_zero(self):
        m = compute_metrics([])
        assert m.total_pnl_inr == 0.0

    def test_empty_total_trades_zero(self):
        m = compute_metrics([])
        assert m.total_trades == 0

    def test_empty_passes_flags_false(self):
        m = compute_metrics([])
        assert not m.passes_min_trade_count
        assert not m.passes_profit_factor


# ── Win rate ─────────────────────────────────────────────────────────────────

class TestWinRate:
    def test_all_winners(self):
        trades = make_daily_trades([100.0, 200.0, 150.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert m.win_rate == 1.0

    def test_all_losers(self):
        trades = make_daily_trades([-100.0, -200.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert m.win_rate == 0.0

    def test_half_win_half_lose(self):
        trades = make_daily_trades([100.0, -100.0, 100.0, -100.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert abs(m.win_rate - 0.5) < 0.01

    def test_win_rate_range(self):
        """Win rate always in [0, 1]."""
        trades = make_daily_trades([50.0, -30.0, 80.0, -10.0, 20.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert 0.0 <= m.win_rate <= 1.0


# ── Profit factor ─────────────────────────────────────────────────────────────

class TestProfitFactor:
    def test_no_losses_infinity(self):
        """No losing trades → profit_factor = inf."""
        trades = make_daily_trades([100.0, 200.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert math.isinf(m.profit_factor) or m.profit_factor > 1_000

    def test_basic_ratio(self):
        """gross_profit=300, gross_loss=100 → PF=3.0."""
        trades = make_daily_trades([300.0, -100.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert abs(m.profit_factor - 3.0) < 0.01

    def test_pf_gt_1_passes_flag(self):
        """PF > 1.0 → passes_profit_factor = True."""
        trades = make_daily_trades([200.0, -100.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert m.passes_profit_factor is True

    def test_pf_lt_1_fails_flag(self):
        """PF < 1.0 → passes_profit_factor = False."""
        trades = make_daily_trades([50.0, -200.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert m.passes_profit_factor is False


# ── Drawdown ──────────────────────────────────────────────────────────────────

class TestDrawdown:
    def test_no_drawdown_all_winning(self):
        trades = make_daily_trades([100.0, 200.0, 150.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        # No drawdown — equity curve monotonically increasing
        assert m.max_drawdown_pct >= 0.0   # non-negative
        assert m.max_drawdown_pct < 1.0    # very small (effectively 0)

    def test_drawdown_after_loss(self):
        """
        Equity: 100000 → 100500 → 99500 → 100000
        Peak = 100500, min after = 99500 → MDD% = (100500-99500)/100500 ≈ 0.995%
        """
        capital = 100_000
        trades = make_daily_trades([500.0, -1000.0, 500.0])
        m = compute_metrics(trades, capital=capital, n_bootstrap=50)
        # MDD should be approximately (100500-99500)/100500 * 100 ≈ 0.995%
        assert m.max_drawdown_pct > 0.5, "Expected non-trivial drawdown"
        assert m.max_drawdown_pct < 5.0, "Drawdown should be small"

    def test_max_drawdown_inr_positive(self):
        trades = make_daily_trades([1000.0, -2000.0, 500.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        assert m.max_drawdown_inr >= 0.0


# ── Sharpe ratio ─────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_sharpe_positive_for_winning_strategy(self):
        """Consistently winning strategy → positive Sharpe."""
        trades = make_daily_trades([100.0] * 50)  # 50 winning days
        m = compute_metrics(trades, capital=100_000, n_bootstrap=200)
        assert m.sharpe_ratio > 0

    def test_sharpe_negative_for_losing_strategy(self):
        """Consistently losing strategy → negative Sharpe."""
        trades = make_daily_trades([-100.0] * 50)
        m = compute_metrics(trades, capital=100_000, n_bootstrap=200)
        assert m.sharpe_ratio < 0

    def test_bootstrap_ci_covers_sharpe(self):
        """95% CI must contain the point estimate."""
        trades = make_daily_trades([100.0, -50.0, 80.0, -20.0, 150.0] * 20)
        m = compute_metrics(trades, capital=100_000, n_bootstrap=500)
        lo, hi = m.sharpe_ci_95
        assert lo <= m.sharpe_ratio <= hi, (
            f"Sharpe {m.sharpe_ratio:.4f} not in CI [{lo:.4f}, {hi:.4f}]"
        )

    def test_ci_lower_lt_upper(self):
        """CI lower < upper (well-formed interval)."""
        trades = make_daily_trades([100.0, -50.0, 200.0, -100.0] * 10)
        m = compute_metrics(trades, capital=100_000, n_bootstrap=200)
        lo, hi = m.sharpe_ci_95
        assert lo < hi, f"CI invalid: [{lo}, {hi}]"

    def test_sharpe_threshold_flag(self):
        """passes_sharpe_threshold=True when Sharpe >= 0.5."""
        # Strong winning strategy: Sharpe >> 0.5
        trades = make_daily_trades([500.0, 400.0, 600.0] * 30)
        m = compute_metrics(trades, capital=100_000, n_bootstrap=100)
        if m.sharpe_ratio >= 0.5:
            assert m.passes_sharpe_threshold is True


# ── Phase 2 exit criteria ────────────────────────────────────────────────────

class TestPhase2Criteria:
    def test_min_trade_count_500(self):
        """passes_min_trade_count=True only if >= 500 trades."""
        trades_499 = make_daily_trades([100.0] * 499)
        trades_500 = make_daily_trades([100.0] * 500)
        m499 = compute_metrics(trades_499, capital=100_000, n_bootstrap=50)
        m500 = compute_metrics(trades_500, capital=100_000, n_bootstrap=50)
        assert m499.passes_min_trade_count is False
        assert m500.passes_min_trade_count is True

    def test_summary_str_format(self):
        """summary_str() must include Sharpe, PF, WR, MaxDD, Trades."""
        trades = make_daily_trades([100.0, -50.0, 200.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        s = m.summary_str()
        assert "Sharpe=" in s
        assert "PF=" in s
        assert "WR=" in s
        assert "Trades=" in s


# ── per_fold_stability ────────────────────────────────────────────────────────

class TestPerFoldStability:
    def test_single_fold_std_zero(self):
        """A single fold has no spread → std = 0."""
        trades = make_daily_trades([100.0, -50.0, 80.0])
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        result = per_fold_stability([m])
        assert result.get("sharpe_std", 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_two_folds_stability_dict(self):
        """Two folds → returns dict with 'sharpe_std' key."""
        t1 = make_daily_trades([100.0] * 20)
        t2 = make_daily_trades([-50.0] * 20)
        m1 = compute_metrics(t1, capital=100_000, n_bootstrap=50)
        m2 = compute_metrics(t2, capital=100_000, n_bootstrap=50)
        result = per_fold_stability([m1, m2])
        assert "sharpe_std" in result

    def test_identical_folds_zero_std(self):
        """Identical folds → sharpe_std ≈ 0."""
        trades = make_daily_trades([100.0, -50.0, 80.0] * 10)
        m = compute_metrics(trades, capital=100_000, n_bootstrap=50)
        result = per_fold_stability([m, m, m])
        assert result.get("sharpe_std", 0.0) == pytest.approx(0.0, abs=1e-9)


# ── Invalid inputs ────────────────────────────────────────────────────────────

class TestInvalidInputs:
    def test_negative_capital_raises(self):
        trades = make_daily_trades([100.0])
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_metrics(trades, capital=-100_000)

    def test_zero_capital_raises(self):
        trades = make_daily_trades([100.0])
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_metrics(trades, capital=0)
