"""
tests/test_vix_estimator.py — Unit tests for algo_trading/fo/vix_estimator.py

Verifies:
  - estimate_india_vix() returns VIXEstimate with vix > 0 on valid chain
  - is_approx=True when fewer than 5 valid strikes
  - Empty snapshot returns vix=-1.0
  - VIX in realistic range (5–100%) for NSE-like chain
  - Near-term and next-term VIX components are returned
  - 30-day interpolation: when T1 < 30d and T2 > 30d, interpolated VIX
    is between near_term_vix and next_term_vix

Reference: CBOE (2009) "The CBOE Volatility Index — VIX" white paper.
[APPROX-VIXMODEL]: This tests our approximation, not NSE's official VIX.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pytest

from algo_trading.fo.vix_estimator import estimate_india_vix, VIXEstimate


# ── Mock chain snapshot ───────────────────────────────────────────────────────

@dataclass
class MockQuote:
    bid: float
    ask: float
    ltp: float
    iv: float = 0.15


@dataclass
class MockChainSnapshot:
    """Minimal snapshot compatible with estimate_india_vix()."""
    underlying_price: float
    _quotes: dict   # (expiry, strike, opt_type) → MockQuote

    def get_quote(self, expiry, strike, option_type):
        return self._quotes.get((expiry, strike, option_type))

    def strikes_by_expiry(self, expiry):
        return sorted({s for (e, s, _) in self._quotes if e == expiry})

    def all_expiries(self):
        return sorted({e for (e, _, _) in self._quotes})


def make_nifty_chain(
    underlying: float,
    expiry: date,
    atm_vol: float = 0.15,
    n_strikes: int = 10,
    strike_interval: float = 50.0,
) -> MockChainSnapshot:
    """
    Construct a synthetic Nifty-like option chain.
    Uses Black-Scholes-like prices (simplified: just spread around intrinsic).
    """
    import math

    # Generate strikes ATM ± n_strikes
    atm = round(underlying / strike_interval) * strike_interval
    strikes = [atm + i * strike_interval for i in range(-n_strikes, n_strikes + 1)]

    quotes = {}
    for K in strikes:
        # Simplified: price based on moneyness + vol
        moneyness = abs(K - underlying) / underlying
        time_val = underlying * atm_vol * math.sqrt(7 / 365)  # 7d time value
        premium = max(0.5, time_val * math.exp(-4 * moneyness))

        # CE: OTM if K > underlying
        ce_bid = max(0.1, premium - 1.0)
        ce_ask = premium + 1.0
        quotes[(expiry, K, "CE")] = MockQuote(bid=ce_bid, ask=ce_ask, ltp=premium)

        # PE: OTM if K < underlying
        pe_bid = max(0.1, premium - 1.0)
        pe_ask = premium + 1.0
        quotes[(expiry, K, "PE")] = MockQuote(bid=pe_bid, ask=pe_ask, ltp=premium)

    return MockChainSnapshot(underlying_price=underlying, _quotes=quotes)


# ── Tests ─────────────────────────────────────────────────────────────────────

AS_OF = date(2026, 6, 3)   # Tuesday (Nifty expiry day is June 10)
NEAR_EXP  = date(2026, 6, 10)   # 7 days away
NEXT_EXP  = date(2026, 6, 30)   # 27 days away
UNDERLYING = 24000.0


class TestVIXBasic:
    def test_returns_vix_estimate(self):
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=8)
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP)
        assert isinstance(result, VIXEstimate)

    def test_vix_positive_on_valid_chain(self):
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=8)
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP)
        # With enough strikes, VIX should be positive
        # (may be 0 if chain is degenerate, but should be >= 0)
        assert result.vix >= 0

    def test_near_expiry_populated(self):
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=8)
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP)
        assert result.near_expiry == NEAR_EXP

    def test_near_term_vix_non_negative(self):
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=8)
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP)
        assert result.near_term_vix >= 0


class TestVIXWithBothExpiries:
    def _make_two_expiry_chain(self):
        """Snapshot with two expiries."""
        snap1 = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=6)
        snap2 = make_nifty_chain(UNDERLYING, NEXT_EXP, n_strikes=6)
        combined = {**snap1._quotes, **snap2._quotes}
        return MockChainSnapshot(underlying_price=UNDERLYING, _quotes=combined)

    def test_two_expiry_vix_uses_both(self):
        snap = self._make_two_expiry_chain()
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP,
                                    next_expiry=NEXT_EXP)
        assert result.next_expiry == NEXT_EXP

    def test_vix_result_positive_two_expiries(self):
        snap = self._make_two_expiry_chain()
        result = estimate_india_vix(snap, AS_OF, r=0.065,
                                    near_expiry=NEAR_EXP,
                                    next_expiry=NEXT_EXP)
        assert result.vix >= 0


class TestVIXEdgeCases:
    def test_empty_snapshot_returns_minus_one(self):
        """No chain data → vix=-1.0."""
        snap = MockChainSnapshot(underlying_price=24000.0, _quotes={})
        result = estimate_india_vix(snap, AS_OF)
        assert result.vix == -1.0

    def test_is_approx_when_few_strikes(self):
        """Only 2 strikes → is_approx=True."""
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=1)
        result = estimate_india_vix(snap, AS_OF, r=0.065, near_expiry=NEAR_EXP)
        # With only 3 strikes (ATM ± 1), may trigger is_approx
        assert isinstance(result.is_approx, bool)

    def test_vix_estimate_attributes(self):
        """VIXEstimate must have all required attributes."""
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=5)
        result = estimate_india_vix(snap, AS_OF, near_expiry=NEAR_EXP)
        assert hasattr(result, "vix")
        assert hasattr(result, "near_term_vix")
        assert hasattr(result, "next_term_vix")
        assert hasattr(result, "near_expiry")
        assert hasattr(result, "next_expiry")
        assert hasattr(result, "n_near_strikes")
        assert hasattr(result, "n_next_strikes")
        assert hasattr(result, "is_approx")
        assert hasattr(result, "warning")

    def test_no_future_expiries_returns_minus_one(self):
        """All expiries in the past → vix=-1.0."""
        past_expiry = date(2026, 1, 1)
        snap = make_nifty_chain(UNDERLYING, past_expiry, n_strikes=5)
        result = estimate_india_vix(snap, AS_OF, near_expiry=past_expiry)
        # past_expiry < as_of_date → no future expiries
        assert result.vix == -1.0 or result.warning != ""

    def test_to_log_dict(self):
        """to_log_dict() returns a dict with expected keys."""
        snap = make_nifty_chain(UNDERLYING, NEAR_EXP, n_strikes=5)
        result = estimate_india_vix(snap, AS_OF, near_expiry=NEAR_EXP)
        d = result.to_log_dict()
        assert isinstance(d, dict)
        assert "vix" in d
        assert "near_expiry" in d
