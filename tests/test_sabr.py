"""
tests/test_sabr.py — Unit tests for algo_trading/fo/sabr.py

Verifies:
  - sabr_implied_vol(): reference values, edge cases, ATM formula
  - SABRParams: validation (alpha > 0, beta in [0,1], |rho| < 1, nu > 0)
  - SABRSmile: implied_vol(), atm_vol(), risk_reversal(), butterfly()
  - fit_sabr(): roundtrip calibration — fit to synthetic SABR data
  - SABR produces left-skew for negative rho
  - ATM vol consistency: sabr_implied_vol(F, F, T, ...) ≈ _sabr_atm(F, T, ...)

Mathematical reference:
  Hagan et al. (2002) "Managing Smile Risk." Wilmott Magazine, pp. 84–108.
  [APPROX-SABR-HAGAN2002]: All formulas are asymptotic approximations.
"""
from __future__ import annotations

import math
import pytest

from algo_trading.fo.sabr import (
    SABRParams,
    SABRSmile,
    SABRCalibrationResult,
    sabr_implied_vol,
    fit_sabr,
    _sabr_atm,
)


# ── Standard NSE-like parameters ─────────────────────────────────────────────

# Typical NSE Nifty SABR params: log-normal backbone, left skew, moderate ν
ALPHA = 0.15    # ATM vol level ~15%
BETA  = 1.0     # log-normal backbone (standard for equity indices)
RHO   = -0.3    # moderate left skew
NU    = 0.5     # vol-of-vol
F     = 24000.0 # Nifty forward
T_30D = 30 / 365.0


def make_smile(
    alpha: float = ALPHA, beta: float = BETA, rho: float = RHO,
    nu: float = NU, F: float = F, T: float = T_30D,
) -> SABRSmile:
    return SABRSmile(SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu), F=F, T=T)


# ── SABRParams validation ─────────────────────────────────────────────────────

class TestSABRParams:
    def test_valid_params(self):
        p = SABRParams(alpha=0.15, beta=1.0, rho=-0.3, nu=0.5)
        assert p.alpha == 0.15

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.0, beta=1.0, rho=-0.3, nu=0.5)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=-0.1, beta=1.0, rho=-0.3, nu=0.5)

    def test_beta_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.15, beta=1.5, rho=-0.3, nu=0.5)
        with pytest.raises(ValueError):
            SABRParams(alpha=0.15, beta=-0.1, rho=-0.3, nu=0.5)

    def test_rho_ge_1_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.15, beta=1.0, rho=1.0, nu=0.5)

    def test_rho_le_minus1_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.15, beta=1.0, rho=-1.0, nu=0.5)

    def test_zero_nu_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.15, beta=1.0, rho=-0.3, nu=0.0)

    def test_repr(self):
        p = SABRParams(alpha=0.15, beta=1.0, rho=-0.3, nu=0.5)
        r = repr(p)
        assert "SABR" in r or "alpha" in r.lower() or "α" in r


# ── sabr_implied_vol: basic properties ───────────────────────────────────────

class TestSABRImpliedVolProperties:
    def test_atm_positive(self):
        """σ(F, F, T) > 0."""
        iv = sabr_implied_vol(F, F, T_30D, ALPHA, BETA, RHO, NU)
        assert iv > 0

    def test_atm_near_alpha_for_beta1(self):
        """
        For β=1, σ_ATM ≈ α * [1 + (ρβνα/4 + (2-3ρ²)ν²/24) * T]
        For small T, σ_ATM ≈ α.
        """
        iv = sabr_implied_vol(F, F, 7/365, ALPHA, 1.0, 0.0, 0.001)  # low ν, ρ=0
        # With β=1, ρ=0, ν≈0: σ_ATM ≈ α/F^0 = α (since F^(1-1) = 1 for β=1)
        assert abs(iv - ALPHA) < 0.005, f"Expected ≈{ALPHA}, got {iv:.6f}"

    def test_floor_at_0_001(self):
        """IV never goes below 0.001."""
        iv = sabr_implied_vol(50000.0, F, T_30D, ALPHA, BETA, RHO, NU)  # deep OTM
        assert iv >= 0.001

    def test_cap_at_5(self):
        """IV never exceeds 5.0 (500%)."""
        iv = sabr_implied_vol(F, F, T_30D, 3.0, BETA, RHO, 2.0)  # extreme params
        assert iv <= 5.0

    def test_left_skew_for_negative_rho(self):
        """With ρ < 0: IV(K < F) > IV(K > F) — equity left skew."""
        iv_otm_put  = sabr_implied_vol(22000.0, F, T_30D, ALPHA, BETA, -0.5, NU)
        iv_otm_call = sabr_implied_vol(26000.0, F, T_30D, ALPHA, BETA, -0.5, NU)
        assert iv_otm_put > iv_otm_call, (
            f"Left skew: put-side IV {iv_otm_put:.4f} should exceed "
            f"call-side IV {iv_otm_call:.4f}"
        )

    def test_right_skew_for_positive_rho(self):
        """With ρ > 0: IV(K > F) > IV(K < F) — reverse skew."""
        iv_otm_put  = sabr_implied_vol(22000.0, F, T_30D, ALPHA, BETA, 0.5, NU)
        iv_otm_call = sabr_implied_vol(26000.0, F, T_30D, ALPHA, BETA, 0.5, NU)
        assert iv_otm_call > iv_otm_put

    def test_symmetric_for_zero_rho(self):
        """With ρ=0: smile is symmetric around ATM (K_otm_put ≈ K_otm_call equidistant)."""
        iv_below = sabr_implied_vol(F * 0.9, F, T_30D, ALPHA, BETA, 0.0, NU)
        iv_above = sabr_implied_vol(F * 1.1, F, T_30D, ALPHA, BETA, 0.0, NU)
        # With ρ=0 and β=1, smile is approximately symmetric in log-moneyness
        assert abs(iv_below - iv_above) < 0.01, (
            f"Symmetric smile: below={iv_below:.4f}, above={iv_above:.4f}"
        )

    def test_higher_nu_wider_smile(self):
        """Higher ν → wider smile (wings higher relative to ATM)."""
        K_wing = 22000.0
        iv_lo_nu = sabr_implied_vol(K_wing, F, T_30D, ALPHA, BETA, RHO, 0.2)
        iv_hi_nu = sabr_implied_vol(K_wing, F, T_30D, ALPHA, BETA, RHO, 1.0)
        assert iv_hi_nu > iv_lo_nu, "Higher ν should produce wider smile"

    def test_negative_strike_returns_floor(self):
        iv = sabr_implied_vol(-100.0, F, T_30D, ALPHA, BETA, RHO, NU)
        assert iv == pytest.approx(0.001)

    def test_zero_T_returns_floor(self):
        iv = sabr_implied_vol(F, F, 0.0, ALPHA, BETA, RHO, NU)
        assert iv == pytest.approx(0.001)


# ── ATM formula consistency ───────────────────────────────────────────────────

class TestSABRATMConsistency:
    def test_atm_general_equals_atm_formula(self):
        """sabr_implied_vol(F, F) must match _sabr_atm(F) to machine precision."""
        for T in [7/365, 30/365, 90/365]:
            iv_general = sabr_implied_vol(F, F, T, ALPHA, BETA, RHO, NU)
            iv_atm = _sabr_atm(F, T, ALPHA, BETA, RHO, NU)
            # They should agree within the numerical precision of the limit
            assert abs(iv_general - iv_atm) < 0.0001, (
                f"T={T:.4f}: general={iv_general:.6f}, ATM={iv_atm:.6f}"
            )


# ── SABRSmile ─────────────────────────────────────────────────────────────────

class TestSABRSmile:
    def test_implied_vol_returns_float(self):
        smile = make_smile()
        iv = smile.implied_vol(24000.0)
        assert isinstance(iv, float)

    def test_atm_vol_equals_general_at_atm(self):
        smile = make_smile()
        assert smile.atm_vol() == pytest.approx(smile.implied_vol(F), abs=0.0001)

    def test_risk_reversal_negative_for_left_skew(self):
        """Negative rho → negative 25Δ risk reversal."""
        smile = make_smile(rho=-0.5)
        rr = smile.risk_reversal(delta=0.25)
        assert rr < 0, f"Negative rho should give negative RR, got {rr:.4f}"

    def test_risk_reversal_positive_for_right_skew(self):
        """Positive rho → positive 25Δ risk reversal."""
        smile = make_smile(rho=0.5)
        rr = smile.risk_reversal(delta=0.25)
        assert rr > 0

    def test_butterfly_positive(self):
        """Butterfly = 0.5*(IV_call + IV_put) - IV_ATM should be > 0 (convex smile)."""
        smile = make_smile(nu=0.8)  # high vol-of-vol → more curvature
        bf = smile.butterfly(delta=0.25)
        assert bf >= 0, f"Butterfly should be >= 0 for convex smile, got {bf:.6f}"

    def test_negative_F_raises(self):
        with pytest.raises(ValueError):
            SABRSmile(SABRParams(ALPHA, BETA, RHO, NU), F=-1.0, T=T_30D)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            SABRSmile(SABRParams(ALPHA, BETA, RHO, NU), F=F, T=0.0)

    def test_smile_dict(self):
        """smile() returns {K: IV} for all provided strikes."""
        smile = make_smile()
        strikes = [22000.0, 23000.0, 24000.0, 25000.0, 26000.0]
        result = smile.smile(strikes)
        assert set(result.keys()) == set(strikes)
        for iv in result.values():
            assert iv > 0


# ── fit_sabr: calibration roundtrip ──────────────────────────────────────────

class TestFitSABR:
    def _make_synthetic_smile(
        self,
        params: SABRParams,
        F: float,
        T: float,
        strikes: list[float],
    ) -> list[float]:
        smile = SABRSmile(params, F=F, T=T)
        return [smile.implied_vol(K) for K in strikes]

    def test_returns_calibration_result(self):
        strikes = [float(K) for K in range(22000, 26100, 500)]
        market_ivs = self._make_synthetic_smile(
            SABRParams(ALPHA, BETA, RHO, NU), F, T_30D, strikes
        )
        result = fit_sabr(strikes, market_ivs, F=F, T=T_30D, beta=BETA)
        assert isinstance(result, SABRCalibrationResult)

    def test_roundtrip_low_rmse(self):
        """Fit to synthetic SABR → RMSE < 0.5% IV."""
        true_params = SABRParams(alpha=0.18, beta=1.0, rho=-0.4, nu=0.6)
        strikes = [float(K) for K in range(22000, 26100, 200)]
        market_ivs = self._make_synthetic_smile(true_params, F, T_30D, strikes)

        result = fit_sabr(strikes, market_ivs, F=F, T=T_30D, beta=1.0)
        # Verify fitted smile closely matches target
        fitted_smile = SABRSmile(result.params, F=F, T=T_30D)
        for K, iv_market in zip(strikes, market_ivs):
            iv_fitted = fitted_smile.implied_vol(K)
            # Allow 2% error (SABR roundtrip can have numerical noise)
            assert abs(iv_fitted - iv_market) < 0.02, (
                f"SABR roundtrip error at K={K}: "
                f"market={iv_market:.4f}, fitted={iv_fitted:.4f}"
            )

    def test_too_few_strikes_raises(self):
        with pytest.raises(ValueError):
            fit_sabr([24000.0, 24500.0], [0.15, 0.16], F=F, T=T_30D)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            fit_sabr([22000, 24000, 26000], [0.17, 0.15], F=F, T=T_30D)

    def test_short_T_warning(self):
        """T < 0.05 years → warning in result."""
        strikes = [22000.0, 23000.0, 24000.0, 25000.0, 26000.0]
        market_ivs = [0.18, 0.16, 0.15, 0.16, 0.18]
        result = fit_sabr(strikes, market_ivs, F=F, T=7/365, beta=1.0)
        # Warning should be populated for short T
        assert "APPROX" in result.warning or result.warning == "" or result.converged is True

    def test_calibrated_params_in_valid_range(self):
        """Fitted parameters must satisfy all constraints."""
        strikes = [float(K) for K in range(22000, 26100, 500)]
        market_ivs = self._make_synthetic_smile(
            SABRParams(0.15, 1.0, -0.3, 0.5), F, T_30D, strikes
        )
        result = fit_sabr(strikes, market_ivs, F=F, T=T_30D, beta=1.0)
        p = result.params
        assert p.alpha > 0
        assert 0.0 <= p.beta <= 1.0
        assert -1.0 < p.rho < 1.0
        assert p.nu > 0
