"""
tests/test_metrics_improvements.py — Tests for new metrics: Sortino, VaR, CVaR,
max consecutive losses, and BSMEngine.chain_iv().

Added in the GitHub-research improvement sprint:
  - sortino_ratio (Sortino & van der Meer 1991)
  - var_95_inr / cvar_95_inr (empirical, 95% confidence)
  - max_consecutive_losses
  - BSMEngine.chain_iv() (batch IV computation)
"""
from __future__ import annotations

import math
import pytest
from datetime import date, datetime

from algo_trading.research.metrics_engine import (
    TradeRecord,
    MetricsSuite,
    compute_metrics,
    _sortino_ratio,
    _var_cvar_95,
    _max_consecutive_losses,
)
from algo_trading.fo.pricing import BSMEngine, IVSolverResult
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trade(pnl: float, day: int = 1) -> TradeRecord:
    t = datetime(2026, 1, day % 28 + 1)
    return TradeRecord(pnl_inr=pnl, entry_time=t, exit_time=t, holding_days=0)


def make_trades(pnls: list[float]) -> list[TradeRecord]:
    return [make_trade(p, i + 1) for i, p in enumerate(pnls)]


# ── Sortino ratio ─────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_all_positive_returns_inf(self):
        """No downside → Sortino = inf (or very large)."""
        pnls = np.array([100.0, 200.0, 150.0])
        s = _sortino_ratio(pnls, capital=100_000.0)
        assert math.isinf(s) or s > 100

    def test_all_negative_returns_negative(self):
        """All losses → negative Sortino."""
        pnls = np.array([-100.0, -200.0, -150.0])
        s = _sortino_ratio(pnls, capital=100_000.0)
        assert s < 0

    def test_mixed_returns_finite(self):
        """Mixed returns → finite Sortino."""
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
        s = _sortino_ratio(pnls, capital=100_000.0)
        assert math.isfinite(s)

    def test_insufficient_data_returns_zero(self):
        pnls = np.array([100.0])
        s = _sortino_ratio(pnls, capital=100_000.0)
        assert s == 0.0

    def test_sortino_higher_than_sharpe_for_upside_skew(self):
        """
        For positively-skewed returns (large wins, small losses), Sortino > Sharpe.
        """
        from algo_trading.research.metrics_engine import _annualized_sharpe
        # Large positive outliers, small negatives
        pnls = np.array([1000.0, 500.0, 800.0, -50.0, -30.0, 700.0])
        capital = 100_000.0
        sortino = _sortino_ratio(pnls, capital)
        sharpe = _annualized_sharpe(pnls, capital)
        # With upside skew: Sortino should be >= Sharpe
        assert sortino >= sharpe

    def test_full_compute_metrics_has_sortino(self):
        """compute_metrics() populates sortino_ratio."""
        trades = make_trades([100.0, -50.0, 200.0, -30.0, 150.0] * 5)
        m = compute_metrics(trades, capital=100_000.0, n_bootstrap=50)
        assert hasattr(m, "sortino_ratio")
        assert math.isfinite(m.sortino_ratio)

    def test_empty_trades_sortino_zero(self):
        m = compute_metrics([], capital=100_000.0)
        assert m.sortino_ratio == 0.0


# ── VaR and CVaR ─────────────────────────────────────────────────────────────

class TestVaRCVaR:
    def test_insufficient_data_returns_zero_zero(self):
        """Fewer than 20 observations → (0.0, 0.0)."""
        pnls = np.array([100.0, -50.0, 200.0])
        var, cvar = _var_cvar_95(pnls)
        assert var == 0.0 and cvar == 0.0

    def test_var_positive_for_mixed_distribution(self):
        """VaR should be positive (a loss) for distributions with losses."""
        rng = np.random.default_rng(42)
        pnls = rng.normal(-100.0, 500.0, 100)  # negative mean → VaR > 0
        var, cvar = _var_cvar_95(pnls)
        assert var > 0

    def test_cvar_gte_var(self):
        """CVaR >= VaR by definition (CVaR is the tail average below VaR)."""
        rng = np.random.default_rng(42)
        pnls = rng.normal(0.0, 1000.0, 200)
        var, cvar = _var_cvar_95(pnls)
        assert cvar >= var - 1e-6  # allow tiny numerical tolerance

    def test_all_positive_returns_negative_var(self):
        """All positive P&Ls → VaR is negative (5th percentile is positive profit)."""
        pnls = np.linspace(100.0, 1000.0, 100)  # all positive
        var, cvar = _var_cvar_95(pnls)
        # var = -percentile(pnls, 5) = -110... so var is negative
        assert var < 0  # VaR is negative when even worst 5% are profits

    def test_full_compute_metrics_has_var_cvar(self):
        """compute_metrics() populates var_95_inr and cvar_95_inr."""
        pnls = np.random.default_rng(99).normal(50.0, 500.0, 30)
        trades = [make_trade(float(p), i + 1) for i, p in enumerate(pnls)]
        m = compute_metrics(trades, capital=100_000.0, n_bootstrap=50)
        assert hasattr(m, "var_95_inr")
        assert hasattr(m, "cvar_95_inr")


# ── Max consecutive losses ─────────────────────────────────────────────────────

class TestMaxConsecutiveLosses:
    def test_empty_returns_zero(self):
        assert _max_consecutive_losses([]) == 0

    def test_all_winning(self):
        trades = make_trades([100.0, 200.0, 150.0])
        assert _max_consecutive_losses(trades) == 0

    def test_all_losing(self):
        trades = make_trades([-100.0, -200.0, -150.0])
        assert _max_consecutive_losses(trades) == 3

    def test_streak_of_three(self):
        trades = make_trades([100.0, -50.0, -60.0, -70.0, 100.0])
        assert _max_consecutive_losses(trades) == 3

    def test_two_separate_streaks(self):
        """Longest streak wins."""
        trades = make_trades([-10.0, -20.0, 100.0, -30.0, -40.0, -50.0, 200.0])
        assert _max_consecutive_losses(trades) == 3

    def test_breakeven_resets_streak(self):
        """Zero P&L resets the consecutive loss streak."""
        trades = make_trades([-100.0, -200.0, 0.0, -50.0])
        assert _max_consecutive_losses(trades) == 2

    def test_full_compute_metrics_has_max_consec_losses(self):
        trades = make_trades([100.0, -50.0, -60.0, 200.0, -30.0])
        m = compute_metrics(trades, capital=100_000.0, n_bootstrap=50)
        assert hasattr(m, "max_consecutive_losses")
        assert m.max_consecutive_losses == 2


# ── BSMEngine.chain_iv() ──────────────────────────────────────────────────────

class TestChainIV:
    """Tests for BSMEngine.chain_iv() — batch implied vol for option chains."""

    S = 24000.0
    T = 7.0 / 365.0
    engine = BSMEngine(r=0.065, q=0.015)

    def test_returns_list_of_same_length(self):
        strikes = [23000.0, 23500.0, 24000.0, 24500.0, 25000.0]
        prices = [250.0, 180.0, 150.0, 130.0, 90.0]
        option_types = ["PE"] * 2 + ["CE"] * 3
        results = self.engine.chain_iv(
            prices=prices, S=self.S, strikes=strikes,
            T=self.T, option_types=option_types,
        )
        assert len(results) == len(strikes)

    def test_returns_iv_solver_results(self):
        """Each element must be an IVSolverResult instance."""
        results = self.engine.chain_iv(
            prices=[150.0, 120.0], S=self.S,
            strikes=[24000.0, 24500.0], T=self.T,
            option_types=["CE", "CE"],
        )
        for r in results:
            assert isinstance(r, IVSolverResult)

    def test_atm_call_converges(self):
        """ATM call with a reasonable premium → solver converges."""
        price = self.engine.call(S=self.S, K=self.S, T=self.T, sigma=0.15)
        results = self.engine.chain_iv(
            prices=[price], S=self.S,
            strikes=[self.S], T=self.T,
            option_types=["CE"],
        )
        assert results[0].converged
        assert results[0].iv is not None
        assert abs(results[0].iv - 0.15) < 0.001

    def test_roundtrip_iv_matches_input_sigma(self):
        """For each strike: compute price from sigma, then invert → sigma_back ≈ sigma."""
        sigma_in = 0.18
        strikes = [22000.0, 23000.0, 24000.0, 25000.0, 26000.0]
        prices = [
            self.engine.put(S=self.S, K=K, T=self.T, sigma=sigma_in)
            if K < self.S
            else self.engine.call(S=self.S, K=K, T=self.T, sigma=sigma_in)
            for K in strikes
        ]
        option_types = ["PE" if K < self.S else "CE" for K in strikes]
        results = self.engine.chain_iv(
            prices=prices, S=self.S, strikes=strikes,
            T=self.T, option_types=option_types,
        )
        for K, r, sigma_expected in zip(strikes, results, [sigma_in] * len(strikes)):
            assert r.converged, f"K={K}: solver did not converge"
            assert abs(r.iv - sigma_expected) < 0.001, (
                f"K={K}: expected IV={sigma_expected:.4f}, got {r.iv:.4f}"
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            self.engine.chain_iv(
                prices=[100.0, 200.0], S=self.S,
                strikes=[24000.0],  # length mismatch
                T=self.T, option_types=["CE", "CE"],
            )

    def test_empty_chain_returns_empty_list(self):
        results = self.engine.chain_iv(
            prices=[], S=self.S, strikes=[], T=self.T, option_types=[],
        )
        assert results == []
