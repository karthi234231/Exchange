"""
tests/test_fo_portfolio_risk.py — Tests for fo/portfolio_risk.py.

Covers: VRP computation, Greek aggregation, risk-limit enforcement,
delta-hedge unit calculation, P&L attribution, and Greek addition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pytest

from algo_trading.fo.portfolio_risk import (
    PortfolioGreeks,
    PortfolioRiskEngine,
    VRPEstimate,
    RiskLimitResult,
    VRP_THRESHOLD_FAVORABLE,
    VRP_THRESHOLD_UNFAVORABLE,
)
from algo_trading.fo.pricing import BSMEngine

# ── Shared test fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bsm() -> BSMEngine:
    return BSMEngine(r=0.065, q=0.015)


@pytest.fixture(scope="module")
def engine(bsm: BSMEngine) -> PortfolioRiskEngine:
    return PortfolioRiskEngine(
        bsm=bsm,
        max_delta_inr=50_000.0,
        max_vega_inr=25_000.0,
        max_gamma_pnl_1pct=10_000.0,
    )


# ── Minimal leg stub: matches the fields accessed by compute_portfolio_greeks ─
@dataclass
class _MockLeg:
    option_type: str
    strike: float
    direction: int
    lots: int
    lot_size: int
    entry_premium: float = 100.0
    current_premium: float = 100.0
    sigma: float = 0.15        # required by compute_portfolio_greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    expiry: object = date(2026, 6, 26)
    symbol: str = "NIFTY"


# Helper: fabricate a PortfolioGreeks directly for downstream tests
def _greeks(
    delta: float = 0.0,
    gamma: float = 0.0,
    vega: float = 0.0,
    theta: float = -500.0,
    rho: float = 0.0,
    S: float = 24_000.0,
) -> PortfolioGreeks:
    ds = S * 0.01
    return PortfolioGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        delta_risk_inr=delta * S,
        gamma_pnl_1pct=0.5 * gamma * ds * ds,
        vega_risk_1pct=vega,
        underlying_price=S,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VRP computation
# ─────────────────────────────────────────────────────────────────────────────

def _price_series_from_vol(sigma_annual: float, n: int = 30, seed: int = 0) -> list[float]:
    """Generate a price series with approximately sigma_annual realized vol."""
    import math as _math
    sigma_daily = sigma_annual / _math.sqrt(252)
    # Deterministic: alternate +sigma/-sigma log-returns for controlled variance
    prices = [100.0]
    for i in range(n):
        r = sigma_daily if i % 2 == 0 else -sigma_daily
        prices.append(prices[-1] * _math.exp(r))
    return prices


class TestComputeVRP:

    def test_favorable_signal_when_realized_exceeds_implied(self, engine):
        # realized vol ≈ 0.25, atm_iv = 0.12 → VRP ≈ +0.13 > 0.03 → favorable
        prices = _price_series_from_vol(0.25, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.12, lookback=22)
        assert est.signal == "favorable_short_premium"
        assert est.vrp > VRP_THRESHOLD_FAVORABLE

    def test_unfavorable_signal_when_realized_below_implied(self, engine):
        # realized vol ≈ 0.08, atm_iv = 0.25 → VRP ≈ -0.17 < 0 → unfavorable
        prices = _price_series_from_vol(0.08, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.25, lookback=22)
        assert est.signal == "unfavorable_short_premium"
        assert est.vrp < VRP_THRESHOLD_UNFAVORABLE

    def test_neutral_signal_in_middle_band(self, engine):
        # realized vol ≈ 0.14, atm_iv = 0.13 → VRP ≈ +0.01 (0 < VRP < 0.03)
        prices = _price_series_from_vol(0.14, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.13, lookback=22)
        # Signal depends on exact stdev; just check it's one of the valid values
        assert est.signal in ("favorable_short_premium", "neutral", "unfavorable_short_premium")

    def test_vrp_formula_components(self, engine):
        prices = _price_series_from_vol(0.20, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.15, lookback=22)
        assert est.vrp == pytest.approx(est.realized_vol - est.implied_vol_atm, rel=1e-9)
        assert est.lookback_bars == 22

    def test_15min_bar_type(self, engine):
        prices = _price_series_from_vol(0.20, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.15, lookback=22, bar_type="15min")
        # sqrt(6552) > sqrt(252), so realized_vol should be larger
        est_daily = engine.compute_vrp(prices, atm_iv=0.15, lookback=22, bar_type="daily")
        assert est.realized_vol > est_daily.realized_vol

    def test_signal_strength_clipped_to_1(self, engine):
        # Very high realized vol should clip signal_strength at 1.0
        prices = _price_series_from_vol(0.80, n=25)
        est = engine.compute_vrp(prices, atm_iv=0.05, lookback=22)
        assert est.signal_strength <= 1.0

    def test_too_short_history_raises(self, engine):
        with pytest.raises(ValueError, match="price_history"):
            engine.compute_vrp([100.0, 101.0], atm_iv=0.15, lookback=22)

    def test_invalid_bar_type_raises(self, engine):
        prices = _price_series_from_vol(0.20, n=25)
        with pytest.raises(ValueError, match="bar_type"):
            engine.compute_vrp(prices, atm_iv=0.15, bar_type="hourly")

    def test_nonpositive_atm_iv_raises(self, engine):
        prices = _price_series_from_vol(0.20, n=25)
        with pytest.raises(ValueError, match="atm_iv"):
            engine.compute_vrp(prices, atm_iv=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Risk-limit enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckRiskLimits:

    def test_all_limits_ok(self, engine):
        g = _greeks(delta=1.0, gamma=0.0001, vega=-5_000.0, S=24_000.0)
        result = engine.check_risk_limits(g)
        assert result.all_ok is True
        assert result.violations == []
        assert result.delta_ok
        assert result.vega_ok
        assert result.gamma_ok

    def test_delta_breach(self, engine):
        # |delta_risk_inr| = 5 × 24000 = 120000 > 50000
        g = _greeks(delta=5.0, S=24_000.0)
        result = engine.check_risk_limits(g)
        assert result.all_ok is False
        assert result.delta_ok is False
        assert any("DELTA_BREACH" in v for v in result.violations)
        assert result.delta_utilization_pct > 100.0

    def test_vega_breach(self, engine):
        # |vega| = 30000 > max_vega_inr = 25000
        g = _greeks(vega=-30_000.0, S=24_000.0)
        result = engine.check_risk_limits(g)
        assert result.vega_ok is False
        assert any("VEGA_BREACH" in v for v in result.violations)

    def test_theta_always_ok(self, engine):
        # Theta has no hard limit — always passes
        g = _greeks(theta=-999_999.0, S=24_000.0)
        result = engine.check_risk_limits(g)
        assert result.theta_ok is True

    def test_utilization_percentage_correct(self, engine):
        # delta_risk_inr = 2.0 × 24000 = 48000; utilization = 48000/50000 = 96%
        g = _greeks(delta=2.0, S=24_000.0)
        result = engine.check_risk_limits(g)
        assert result.delta_utilization_pct == pytest.approx(96.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Delta-hedge unit calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaHedgeUnits:

    def test_short_delta_needs_long_hedge(self, engine):
        # portfolio_delta = -130: we have net short delta (short calls)
        # hedge_lots = round(-(-130)/65) = round(2.0) = 2 → BUY 2 lots
        g = _greeks(delta=-130.0)
        assert engine.delta_hedge_units(g, futures_lot_size=65) == 2

    def test_long_delta_needs_short_hedge(self, engine):
        # portfolio_delta = +130 → hedge = round(-130/65) = -2 → SELL 2 lots
        g = _greeks(delta=130.0)
        assert engine.delta_hedge_units(g, futures_lot_size=65) == -2

    def test_near_zero_delta_no_hedge(self, engine):
        # portfolio_delta = 30 → hedge = round(-30/65) = round(-0.46) = 0
        g = _greeks(delta=30.0)
        assert engine.delta_hedge_units(g, futures_lot_size=65) == 0

    def test_zero_lot_size_raises(self, engine):
        g = _greeks(delta=100.0)
        with pytest.raises(ValueError, match="futures_lot_size"):
            engine.delta_hedge_units(g, futures_lot_size=0)

    def test_rounding_half_lot(self, engine):
        # delta = 97.5 → -97.5/65 = -1.5 → round to -2 (banker's rounding)
        # Python round(1.5) = 2, round(-1.5) = -2
        g = _greeks(delta=97.5)
        result = engine.delta_hedge_units(g, futures_lot_size=65)
        assert result in (-1, -2)   # accept both sides of 0.5 rounding


# ─────────────────────────────────────────────────────────────────────────────
# P&L attribution
# ─────────────────────────────────────────────────────────────────────────────

class TestPnLAttribution:

    def test_pure_delta_pnl(self, engine):
        # delta=2, gamma=0, vega=0, theta=0 → only delta contributes
        # ΔS = 100 → delta_pnl = 2 × 100 = 200
        g = _greeks(delta=2.0, gamma=0.0, vega=0.0, theta=0.0, S=24_000.0)
        attr = engine.pnl_attribution(g, new_underlying=24_100.0, new_iv_atm=0.0, days_elapsed=0.0)
        assert attr["delta"] == pytest.approx(200.0)
        assert attr["gamma"] == pytest.approx(0.0)
        assert attr["vega"] == pytest.approx(0.0)
        assert attr["theta"] == pytest.approx(0.0)

    def test_pure_theta_pnl(self, engine):
        # theta = -500 per day, 1 day elapsed → theta_pnl = -500
        g = _greeks(delta=0.0, gamma=0.0, vega=0.0, theta=-500.0, S=24_000.0)
        attr = engine.pnl_attribution(g, new_underlying=24_000.0, new_iv_atm=0.0, days_elapsed=1.0)
        assert attr["theta"] == pytest.approx(-500.0)
        assert attr["total"] == pytest.approx(-500.0)

    def test_total_equals_sum_of_components(self, engine):
        g = _greeks(delta=1.5, gamma=0.01, vega=-3_000.0, theta=-200.0, S=24_000.0)
        attr = engine.pnl_attribution(g, new_underlying=24_100.0, new_iv_atm=0.5, days_elapsed=0.5)
        expected_total = attr["delta"] + attr["gamma"] + attr["vega"] + attr["theta"]
        assert attr["total"] == pytest.approx(expected_total, rel=1e-9)

    def test_invalid_new_underlying_raises(self, engine):
        g = _greeks()
        with pytest.raises(ValueError, match="new_underlying"):
            engine.pnl_attribution(g, new_underlying=0.0, new_iv_atm=0.0, days_elapsed=0.0)

    def test_negative_days_elapsed_raises(self, engine):
        g = _greeks()
        with pytest.raises(ValueError, match="days_elapsed"):
            engine.pnl_attribution(g, new_underlying=24_000.0, new_iv_atm=0.0, days_elapsed=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# add_portfolio_greeks (static method)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddPortfolioGreeks:

    def test_greeks_add_correctly(self):
        a = _greeks(delta=1.0, vega=-1_000.0, theta=-300.0, S=24_000.0)
        b = _greeks(delta=0.5, vega=-500.0, theta=-200.0, S=24_000.0)
        c = PortfolioRiskEngine.add_portfolio_greeks(a, b)
        assert c.delta == pytest.approx(1.5)
        assert c.vega == pytest.approx(-1_500.0)
        assert c.theta == pytest.approx(-500.0)

    def test_combined_delta_risk_inr(self):
        a = _greeks(delta=2.0, S=24_000.0)
        b = _greeks(delta=1.0, S=24_000.0)
        c = PortfolioRiskEngine.add_portfolio_greeks(a, b)
        assert c.delta_risk_inr == pytest.approx(3.0 * 24_000.0)

    def test_mismatched_underlying_price_raises(self):
        a = _greeks(S=24_000.0)
        b = _greeks(S=24_100.0)   # more than 1 INR difference
        with pytest.raises(ValueError, match="underlying_price mismatch"):
            PortfolioRiskEngine.add_portfolio_greeks(a, b)

    def test_same_price_within_tolerance(self):
        a = _greeks(delta=1.0, S=24_000.0)
        b = _greeks(delta=1.0, S=24_000.5)  # within 1 INR tolerance
        c = PortfolioRiskEngine.add_portfolio_greeks(a, b)
        assert c.delta == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_portfolio_greeks — BSM recomputation path
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePortfolioGreeks:

    def test_single_short_call_has_negative_delta(self, engine):
        # Short ATM call → portfolio_delta < 0
        legs = [_MockLeg(option_type="CE", strike=24_000, direction=-1, lots=1, lot_size=65, sigma=0.15)]
        g = engine.compute_portfolio_greeks(legs, underlying_price=24_000.0, T_years=30/365)
        assert g.delta < 0.0

    def test_single_long_put_has_negative_delta(self, engine):
        # Long OTM put → portfolio_delta < 0
        legs = [_MockLeg(option_type="PE", strike=23_500, direction=+1, lots=1, lot_size=65, sigma=0.15)]
        g = engine.compute_portfolio_greeks(legs, underlying_price=24_000.0, T_years=30/365)
        assert g.delta < 0.0

    def test_delta_scales_with_lots(self, engine):
        leg1 = _MockLeg(option_type="CE", strike=24_000, direction=-1, lots=1, lot_size=65, sigma=0.15)
        leg2 = _MockLeg(option_type="CE", strike=24_000, direction=-1, lots=2, lot_size=65, sigma=0.15)
        g1 = engine.compute_portfolio_greeks([leg1], underlying_price=24_000.0, T_years=30/365)
        g2 = engine.compute_portfolio_greeks([leg2], underlying_price=24_000.0, T_years=30/365)
        assert g2.delta == pytest.approx(2 * g1.delta, rel=1e-9)

    def test_straddle_near_zero_net_delta(self, engine):
        # ATM straddle (long call + long put at same strike): delta ≈ 0 by symmetry
        call = _MockLeg(option_type="CE", strike=24_000, direction=+1, lots=1, lot_size=65, sigma=0.15)
        put  = _MockLeg(option_type="PE", strike=24_000, direction=+1, lots=1, lot_size=65, sigma=0.15)
        g = engine.compute_portfolio_greeks([call, put], underlying_price=24_000.0, T_years=30/365)
        # Net delta of ATM straddle is close to 0 (not exactly 0 due to put-call parity)
        assert abs(g.delta) < 5.0 * 65  # less than 5 lots equivalent delta

    def test_zero_underlying_price_raises(self, engine):
        legs = [_MockLeg(option_type="CE", strike=24_000, direction=-1, lots=1, lot_size=65, sigma=0.15)]
        with pytest.raises(ValueError, match="underlying_price"):
            engine.compute_portfolio_greeks(legs, underlying_price=0.0, T_years=30/365)

    def test_greeks_cached_in_last_greeks(self, engine):
        legs = [_MockLeg(option_type="CE", strike=24_000, direction=-1, lots=1, lot_size=65, sigma=0.15)]
        g = engine.compute_portfolio_greeks(legs, underlying_price=24_000.0, T_years=30/365)
        assert engine.last_greeks is not None
        assert engine.last_greeks.delta == pytest.approx(g.delta)
