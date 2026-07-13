"""
tests/test_fo_chain.py

Unit tests for BlackScholesGreeks and F&O chain loading from
algo_trading/data/fo_chain.py.

Tests cover:
  - Call and put pricing (positive prices)
  - Put-call parity
  - Delta bounds for calls and puts
  - Deep ITM / OTM delta limits
  - Gamma and vega positivity
  - IV round-trip accuracy (call and put)
  - IV returns None for below-intrinsic prices
  - compute_all_greeks populates an OptionContract
  - Synthetic chain structure (CE + PE contracts)
  - ATM strike selection
  - chain.calls and chain.puts properties
"""
from __future__ import annotations

import math
import pytest
import numpy as np

try:
    from algo_trading.data.fo_chain import (
        BlackScholesGreeks,
        FOChainLoader,
        OptionChain,
        OptionContract,
    )
except ImportError:
    pytest.skip("fo_chain not available", allow_module_level=True)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

S = 2800.0       # underlying price
K = 2800.0       # ATM strike
T = 0.1          # time to expiry in years (~36 calendar days)
R = 0.065        # risk-free rate (NSE convention)
SIGMA = 0.25     # implied volatility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract(option_type: str, ltp: float, strike: float = K) -> OptionContract:
    from datetime import date, timedelta
    return OptionContract(
        symbol="NIFTY",
        expiry=date.today() + timedelta(days=36),
        strike=strike,
        option_type=option_type,
        oi=1000,
        change_in_oi=0,
        volume=500,
        iv=0.0,
        ltp=ltp,
    )


# ---------------------------------------------------------------------------
# Pricing — call and put must be positive for ATM options
# ---------------------------------------------------------------------------

class TestPricing:
    def test_call_price_positive(self):
        price = BlackScholesGreeks.call_price(S, K, T, R, SIGMA)
        assert price > 0.0, f"Call price must be positive, got {price}"

    def test_put_price_positive(self):
        price = BlackScholesGreeks.put_price(S, K, T, R, SIGMA)
        assert price > 0.0, f"Put price must be positive, got {price}"

    def test_put_call_parity(self):
        """C - P = S - K * exp(-r * T) within 0.01."""
        C = BlackScholesGreeks.call_price(S, K, T, R, SIGMA)
        P = BlackScholesGreeks.put_price(S, K, T, R, SIGMA)
        lhs = C - P
        rhs = S - K * math.exp(-R * T)
        assert abs(lhs - rhs) < 0.01, (
            f"Put-call parity violated: C-P={lhs:.4f}, S-Ke^(-rT)={rhs:.4f}"
        )


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_call_delta_range(self):
        d = BlackScholesGreeks.delta(S, K, T, R, SIGMA, "CE")
        assert 0.0 < d < 1.0, f"Call delta must be in (0, 1), got {d}"

    def test_put_delta_range(self):
        d = BlackScholesGreeks.delta(S, K, T, R, SIGMA, "PE")
        assert -1.0 < d < 0.0, f"Put delta must be in (-1, 0), got {d}"

    def test_deep_itm_call_delta_near_1(self):
        """Very deep ITM call: S >> K → delta ≈ 1.0."""
        d = BlackScholesGreeks.delta(5000.0, 1000.0, T, R, SIGMA, "CE")
        assert d > 0.99, f"Deep ITM call delta should be near 1.0, got {d}"

    def test_deep_otm_call_delta_near_0(self):
        """Very OTM call: S << K → delta ≈ 0.0."""
        d = BlackScholesGreeks.delta(1000.0, 5000.0, T, R, SIGMA, "CE")
        assert d < 0.01, f"Deep OTM call delta should be near 0.0, got {d}"


# ---------------------------------------------------------------------------
# Gamma and vega — always positive
# ---------------------------------------------------------------------------

class TestGammaVega:
    def test_gamma_positive(self):
        g = BlackScholesGreeks.gamma(S, K, T, R, SIGMA)
        assert g > 0.0, f"Gamma must always be positive, got {g}"

    def test_vega_positive(self):
        v = BlackScholesGreeks.vega(S, K, T, R, SIGMA)
        assert v > 0.0, f"Vega must always be positive, got {v}"


# ---------------------------------------------------------------------------
# Implied volatility — round-trip accuracy
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    def test_iv_roundtrip_call(self):
        """IV recovered from a BS call price should reproduce the original sigma."""
        market_price = BlackScholesGreeks.call_price(S, K, T, R, SIGMA)
        iv = BlackScholesGreeks.implied_volatility(market_price, S, K, T, R, "CE")
        assert iv is not None, "IV solver should converge for a valid call price"
        assert abs(iv - SIGMA) < 0.001, (
            f"IV round-trip failed: expected {SIGMA}, got {iv}"
        )

    def test_iv_roundtrip_put(self):
        """IV recovered from a BS put price should reproduce the original sigma."""
        market_price = BlackScholesGreeks.put_price(S, K, T, R, SIGMA)
        iv = BlackScholesGreeks.implied_volatility(market_price, S, K, T, R, "PE")
        assert iv is not None, "IV solver should converge for a valid put price"
        assert abs(iv - SIGMA) < 0.001, (
            f"IV round-trip failed: expected {SIGMA}, got {iv}"
        )

    def test_iv_returns_none_for_below_intrinsic(self):
        """Market price well below intrinsic value → IV solver returns None."""
        # Deep ITM call: intrinsic ≈ S - K*exp(-rT) ≈ 100 for S=3000, K=2800
        intrinsic_approx = max(0.0, 3000.0 - 2800.0 * math.exp(-R * T))
        below_intrinsic = intrinsic_approx - 50.0  # clearly below intrinsic
        iv = BlackScholesGreeks.implied_volatility(
            below_intrinsic, 3000.0, 2800.0, T, R, "CE"
        )
        assert iv is None, (
            f"IV should be None for sub-intrinsic price, got {iv}"
        )


# ---------------------------------------------------------------------------
# compute_all_greeks — populates an OptionContract
# ---------------------------------------------------------------------------

class TestComputeAllGreeks:
    def test_compute_all_greeks_updates_contract(self):
        """compute_all_greeks fills iv, delta, gamma on the returned contract."""
        ltp = BlackScholesGreeks.call_price(S, K, T, R, SIGMA)
        contract = _make_contract("CE", ltp=ltp)
        assert contract.iv == 0.0, "iv should start at 0"
        assert contract.delta == 0.0, "delta should start at 0"
        assert contract.gamma == 0.0, "gamma should start at 0"

        updated = BlackScholesGreeks.compute_all_greeks(contract, S, T, R)

        assert updated.iv > 0.0, f"iv should be > 0 after compute, got {updated.iv}"
        assert 0.0 < updated.delta < 1.0, (
            f"call delta should be in (0,1), got {updated.delta}"
        )
        assert updated.gamma > 0.0, f"gamma should be > 0, got {updated.gamma}"


# ---------------------------------------------------------------------------
# Synthetic chain structure
# ---------------------------------------------------------------------------

class TestSyntheticChain:
    """Tests on a FOChainLoader.load_synthetic() chain."""

    @pytest.fixture(scope="class")
    def chain(self) -> OptionChain:
        loader = FOChainLoader()
        return loader.load_synthetic(
            symbol="NIFTY",
            underlying_price=S,
            n_strikes=6,
            expiry_days=36,
        )

    def test_synthetic_chain_has_both_types(self, chain: OptionChain):
        """Synthetic chain must contain both CE and PE contracts."""
        types = {c.option_type for c in chain.contracts}
        assert "CE" in types, "Synthetic chain must include CE contracts"
        assert "PE" in types, "Synthetic chain must include PE contracts"

    def test_atm_strike_closest_to_underlying(self, chain: OptionChain):
        """ATM strike is the strike closest to the underlying price."""
        atm = chain.atm_strike
        strikes = [c.strike for c in chain.contracts]
        closest = min(set(strikes), key=lambda k: abs(k - chain.underlying_price))
        assert atm == closest, (
            f"atm_strike={atm} should equal closest strike {closest} "
            f"to underlying_price={chain.underlying_price}"
        )

    def test_chain_calls_property(self, chain: OptionChain):
        """chain.calls returns only CE contracts."""
        calls = chain.calls
        assert len(calls) > 0, "chain.calls should not be empty"
        for c in calls:
            assert c.option_type == "CE", (
                f"All items in chain.calls must be CE, found {c.option_type}"
            )

    def test_chain_puts_property(self, chain: OptionChain):
        """chain.puts returns only PE contracts."""
        puts = chain.puts
        assert len(puts) > 0, "chain.puts should not be empty"
        for p in puts:
            assert p.option_type == "PE", (
                f"All items in chain.puts must be PE, found {p.option_type}"
            )
