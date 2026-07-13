"""
tests/test_vol_surface_router.py — Unit tests for VolSurfaceRouter.

Verifies:
  - Short-dated (≤ 21 DTE) → routes to SABR
  - Longer-dated (> 21 DTE) → routes to SVI
  - Fallback: if preferred model not registered, uses the other
  - get_iv() returns float from routed model
  - interpolated_iv() linearly interpolates in vol space between two expiries
  - smile_dict() returns {K: IV} mapping
  - available_expiries() returns sorted combined list
  - coverage_summary() has expected keys
"""
from __future__ import annotations

import math
import pytest

from algo_trading.fo.vol_surface import SVIParams, SVISmile, VolSurfaceRouter
from algo_trading.fo.sabr import SABRParams, SABRSmile


# ── Standard test parameters ────────────────────────────────────────────────

F_NIFTY   = 24000.0
T_7D      = 7.0 / 365.0
T_30D     = 30.0 / 365.0
T_90D     = 90.0 / 365.0
T_SHORT   = VolSurfaceRouter._SHORT_DTE_THRESHOLD  # 21/365


def make_svi_smile(T: float = T_30D, forward: float = F_NIFTY) -> SVISmile:
    """Minimal valid SVI smile for testing."""
    params = SVIParams(a=0.03, b=0.10, rho=-0.3, m=0.0, sigma=0.20)
    return SVISmile(params=params, T=T, forward=forward)


def make_sabr_smile(T: float = T_7D, F: float = F_NIFTY) -> SABRSmile:
    """Minimal valid SABR smile for testing."""
    params = SABRParams(alpha=0.15, beta=1.0, rho=-0.3, nu=0.5)
    return SABRSmile(params=params, F=F, T=T)


# ── Construction ─────────────────────────────────────────────────────────────

class TestVolSurfaceRouterConstruction:
    def test_empty_router(self):
        router = VolSurfaceRouter()
        assert router.available_expiries() == []

    def test_with_initial_svi(self):
        smile = make_svi_smile(T_30D)
        router = VolSurfaceRouter(svi_smiles={T_30D: smile})
        assert T_30D in router.available_expiries()

    def test_with_initial_sabr(self):
        smile = make_sabr_smile(T_7D)
        router = VolSurfaceRouter(sabr_smiles={T_7D: smile})
        assert T_7D in router.available_expiries()

    def test_add_svi_wrong_type_raises(self):
        router = VolSurfaceRouter()
        with pytest.raises(TypeError):
            router.add_svi_smile(T_30D, "not a smile")

    def test_add_sabr_wrong_type_raises(self):
        router = VolSurfaceRouter()
        with pytest.raises(TypeError):
            router.add_sabr_smile(T_7D, "not a smile")


# ── Model routing ─────────────────────────────────────────────────────────────

class TestModelRouting:
    def test_short_dated_routes_to_svi(self):
        """T ≤ 21 DTE → SVI preferred (Hagan SABR less accurate for short T)."""
        router = VolSurfaceRouter()
        router.add_svi_smile(T_7D, make_svi_smile(T_7D))
        assert router.model_for_expiry(T_7D) == "SVI"

    def test_long_dated_routes_to_sabr(self):
        """T > 21 DTE → SABR preferred (better NSE monthly skew capture)."""
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_30D, make_sabr_smile(T_30D))
        assert router.model_for_expiry(T_30D) == "SABR"

    def test_short_dated_fallback_to_sabr(self):
        """T ≤ 21 DTE but no SVI registered → falls back to SABR."""
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        assert router.model_for_expiry(T_7D) == "SABR"

    def test_long_dated_fallback_to_svi(self):
        """T > 21 DTE but no SABR registered → falls back to SVI."""
        router = VolSurfaceRouter()
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        assert router.model_for_expiry(T_30D) == "SVI"

    def test_unregistered_expiry_returns_none(self):
        """No smile registered for T → model_for_expiry returns 'NONE'."""
        router = VolSurfaceRouter()
        assert router.model_for_expiry(T_30D) == "NONE"

    def test_threshold_boundary_short(self):
        """T exactly at threshold → short-dated (SVI preferred)."""
        router = VolSurfaceRouter()
        router.add_svi_smile(T_SHORT, make_svi_smile(T_SHORT))
        # T <= threshold → "SVI"
        assert router.model_for_expiry(T_SHORT) == "SVI"


# ── get_iv ────────────────────────────────────────────────────────────────────

class TestGetIV:
    def test_get_iv_svi_short_dated(self):
        """get_iv() for short-dated T returns SVI smile value."""
        router = VolSurfaceRouter()
        smile = make_svi_smile(T_7D)
        router.add_svi_smile(T_7D, smile)
        K = 24500.0
        iv_router = router.get_iv(T_7D, K)
        iv_direct = smile.implied_vol(K)
        assert iv_router == pytest.approx(iv_direct, abs=1e-10)

    def test_get_iv_sabr_long_dated(self):
        """get_iv() for long-dated T returns SABR smile value."""
        router = VolSurfaceRouter()
        smile = make_sabr_smile(T_30D)
        router.add_sabr_smile(T_30D, smile)
        K = 23500.0
        iv_router = router.get_iv(T_30D, K)
        iv_direct = smile.implied_vol(K)
        assert iv_router == pytest.approx(iv_direct, abs=1e-10)

    def test_get_iv_returns_none_for_missing_expiry(self):
        router = VolSurfaceRouter()
        assert router.get_iv(T_7D, 24000.0) is None

    def test_get_iv_positive(self):
        """All routed IVs must be positive."""
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        for T, K in [(T_7D, 23500.0), (T_30D, 24500.0)]:
            iv = router.get_iv(T, K)
            assert iv is not None and iv > 0


# ── interpolated_iv ───────────────────────────────────────────────────────────

class TestInterpolatedIV:
    def _make_router(self) -> VolSurfaceRouter:
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        router.add_svi_smile(T_90D, make_svi_smile(T_90D))
        return router

    def test_empty_router_returns_none(self):
        router = VolSurfaceRouter()
        assert router.interpolated_iv(T_30D, F_NIFTY) is None

    def test_exact_expiry_matches_get_iv(self):
        router = self._make_router()
        K = 24000.0
        assert (router.interpolated_iv(T_7D, K)
                == pytest.approx(router.get_iv(T_7D, K), abs=1e-10))

    def test_interpolated_between_expiries_is_between_endpoints(self):
        router = self._make_router()
        K = 24000.0
        T_mid = (T_7D + T_90D) / 2

        iv1 = router.get_iv(T_7D, K)
        iv2 = router.get_iv(T_90D, K)
        iv_mid = router.interpolated_iv(T_mid, K)

        assert iv_mid is not None
        low, high = min(iv1, iv2), max(iv1, iv2)
        assert low <= iv_mid <= high, (
            f"Interpolated IV {iv_mid:.4f} not between {low:.4f} and {high:.4f}"
        )

    def test_left_extrapolation_returns_nearest(self):
        """T before first expiry → flat extrapolation at T_7D."""
        router = self._make_router()
        K = 24000.0
        iv_before = router.interpolated_iv(T_7D * 0.5, K)
        iv_at_7d = router.get_iv(T_7D, K)
        assert iv_before == pytest.approx(iv_at_7d, abs=1e-10)

    def test_right_extrapolation_returns_nearest(self):
        router = self._make_router()
        K = 24000.0
        iv_after = router.interpolated_iv(T_90D * 1.5, K)
        iv_at_90d = router.get_iv(T_90D, K)
        assert iv_after == pytest.approx(iv_at_90d, abs=1e-10)


# ── smile_dict ────────────────────────────────────────────────────────────────

class TestSmileDict:
    def test_smile_dict_returns_all_strikes(self):
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        strikes = [23000.0, 23500.0, 24000.0, 24500.0, 25000.0]
        result = router.smile_dict(T_7D, strikes)
        assert set(result.keys()) == set(strikes)

    def test_smile_dict_all_positive(self):
        router = VolSurfaceRouter()
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        strikes = [22000.0, 24000.0, 26000.0]
        result = router.smile_dict(T_30D, strikes)
        assert all(v > 0 for v in result.values())

    def test_smile_dict_missing_expiry_returns_empty(self):
        router = VolSurfaceRouter()
        result = router.smile_dict(T_7D, [24000.0])
        assert result == {}


# ── available_expiries / coverage_summary ─────────────────────────────────────

class TestCoverage:
    def test_available_expiries_sorted(self):
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        router.add_svi_smile(T_90D, make_svi_smile(T_90D))
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        expiries = router.available_expiries()
        assert expiries == sorted(expiries)

    def test_coverage_summary_has_model_key(self):
        router = VolSurfaceRouter()
        router.add_sabr_smile(T_7D, make_sabr_smile(T_7D))
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        summary = router.coverage_summary()
        for T, info in summary.items():
            assert "model" in info
            assert "dte_days" in info
            assert info["model"] in ("SABR", "SVI", "NONE")

    def test_repr_includes_n_expiries(self):
        router = VolSurfaceRouter()
        router.add_svi_smile(T_30D, make_svi_smile(T_30D))
        assert "n_expiries=1" in repr(router)
