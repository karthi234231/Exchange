"""
Stage 4: Tests for F&O chain analysis — fo_chain.py and fo_features.py.

All tests use synthetic data (no file I/O).
"""
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
    from algo_trading.features.fo_features import FOFeatureEngine, FOFeaturesResult
except ImportError as exc:
    pytest.skip(f"algo_trading package not importable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_synthetic_chain(underlying: float = 2800.0, n_strikes: int = 10) -> OptionChain:
    from datetime import date, timedelta

    expiry = date.today() + timedelta(days=7)
    return FOChainLoader().load_synthetic("RELIANCE", underlying, n_strikes=n_strikes, expiry_days=7)


S, K, T, r, sigma = 2800.0, 2800.0, 0.1, 0.065, 0.25


# ---------------------------------------------------------------------------
# Black-Scholes price tests
# ---------------------------------------------------------------------------

def test_bs_call_price_positive():
    """BS call price must be strictly positive for ATM options."""
    price = BlackScholesGreeks.call_price(S, K, T, r, sigma)
    assert price > 0, f"Expected positive call price, got {price}"


def test_bs_put_call_parity():
    """
    Put-Call parity: C - P = S - K * exp(-rT).
    Tolerance: ±0.01 rupees.
    """
    call = BlackScholesGreeks.call_price(S, K, T, r, sigma)
    put = BlackScholesGreeks.put_price(S, K, T, r, sigma)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01, (
        f"Put-call parity violated: C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}"
    )


def test_bs_delta_call_range():
    """Call delta must lie in [0, 1]."""
    d = BlackScholesGreeks.delta(S, K, T, r, sigma, "CE")
    assert 0.0 <= d <= 1.0, f"Call delta out of range: {d}"


def test_bs_delta_put_range():
    """Put delta must lie in [-1, 0]."""
    d = BlackScholesGreeks.delta(S, K, T, r, sigma, "PE")
    assert -1.0 <= d <= 0.0, f"Put delta out of range: {d}"


def test_bs_iv_roundtrip():
    """
    IV round-trip: implied_volatility(call_price(S,K,T,r,σ)) ≈ σ.
    Tolerance: ±0.001 (0.1%).
    """
    target_sigma = 0.25
    market_price = BlackScholesGreeks.call_price(S, K, T, r, target_sigma)
    recovered = BlackScholesGreeks.implied_volatility(market_price, S, K, T, r, "CE")
    assert recovered is not None, "IV solver returned None for a valid call price"
    assert abs(recovered - target_sigma) < 0.001, (
        f"IV round-trip failed: expected {target_sigma}, got {recovered}"
    )


# ---------------------------------------------------------------------------
# Synthetic chain structure tests
# ---------------------------------------------------------------------------

def test_synthetic_chain_has_contracts():
    """Synthetic chain must contain at least one contract."""
    chain = make_synthetic_chain()
    assert len(chain.contracts) > 0


def test_synthetic_chain_calls_and_puts():
    """Synthetic chain must contain both CE and PE contracts."""
    chain = make_synthetic_chain()
    assert len(chain.calls) > 0, "No call contracts in synthetic chain"
    assert len(chain.puts) > 0, "No put contracts in synthetic chain"


# ---------------------------------------------------------------------------
# FOFeatureEngine tests
# ---------------------------------------------------------------------------

def test_pcr_positive():
    """PCR by OI and by volume must always be > 0 (denominator +1 guard)."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)
    assert result.pcr_oi > 0, f"pcr_oi is not positive: {result.pcr_oi}"
    assert result.pcr_volume > 0, f"pcr_volume is not positive: {result.pcr_volume}"


def test_max_pain_in_strike_range():
    """Max pain must fall within the min and max strikes of the chain."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)
    strikes = [c.strike for c in chain.contracts]
    assert min(strikes) <= result.max_pain <= max(strikes), (
        f"max_pain {result.max_pain} outside strike range "
        f"[{min(strikes)}, {max(strikes)}]"
    )


def test_iv_rank_bounded():
    """IV rank must be in [0, 1]."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)
    assert 0.0 <= result.iv_rank <= 1.0, f"iv_rank out of bounds: {result.iv_rank}"


def test_iv_skew_finite():
    """IV skew must be a finite float (not NaN, not Inf)."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)
    assert math.isfinite(result.iv_skew), f"iv_skew is not finite: {result.iv_skew}"


def test_oi_buildup_signal_valid():
    """OI buildup signal must be one of the four valid string values."""
    valid = {"long_buildup", "short_buildup", "long_unwinding", "short_unwinding"}
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    # Test all four branches by varying price and OI changes
    for price_chg, oi_chg in [(0.5, 2.0), (-0.5, 2.0), (-0.5, -2.0), (0.5, -2.0)]:
        result = engine.compute(chain, price_change_pct=price_chg, oi_change_pct=oi_chg)
        assert result.oi_buildup_signal in valid, (
            f"Unexpected oi_buildup_signal: '{result.oi_buildup_signal}'"
        )


def test_walls_in_strike_range():
    """Call wall and put wall must be actual strikes present in the chain."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)
    call_strikes = {c.strike for c in chain.calls}
    put_strikes = {p.strike for p in chain.puts}
    assert result.call_wall in call_strikes, (
        f"call_wall {result.call_wall} not in call strikes {sorted(call_strikes)}"
    )
    assert result.put_wall in put_strikes, (
        f"put_wall {result.put_wall} not in put strikes {sorted(put_strikes)}"
    )


def test_pcr_signal_classification():
    """
    Manually construct chains with controlled OI to verify PCR → signal mapping.
    PCR < 0.8 → "bullish", PCR > 1.2 → "bearish".
    """
    from datetime import date, timedelta, datetime as dt

    expiry = date.today() + timedelta(days=7)
    base_ts = dt.now()

    def _make_chain(put_oi: int, call_oi: int) -> OptionChain:
        contracts = [
            OptionContract(
                symbol="TEST", expiry=expiry, strike=2800.0, option_type="CE",
                oi=call_oi, change_in_oi=0, volume=call_oi,
                iv=0.25, ltp=50.0,
            ),
            OptionContract(
                symbol="TEST", expiry=expiry, strike=2800.0, option_type="PE",
                oi=put_oi, change_in_oi=0, volume=put_oi,
                iv=0.25, ltp=50.0,
            ),
        ]
        return OptionChain(
            symbol="TEST",
            underlying_price=2800.0,
            timestamp=base_ts,
            expiry=expiry,
            contracts=contracts,
        )

    engine = FOFeatureEngine()

    # Bullish: low put OI relative to call OI → PCR < 0.8
    chain_bull = _make_chain(put_oi=700, call_oi=10_000)
    res_bull = engine.compute(chain_bull)
    assert res_bull.pcr_signal == "bullish", (
        f"Expected 'bullish', got '{res_bull.pcr_signal}' (pcr_oi={res_bull.pcr_oi:.4f})"
    )

    # Bearish: high put OI relative to call OI → PCR > 1.2
    chain_bear = _make_chain(put_oi=15_000, call_oi=10_000)
    res_bear = engine.compute(chain_bear)
    assert res_bear.pcr_signal == "bearish", (
        f"Expected 'bearish', got '{res_bear.pcr_signal}' (pcr_oi={res_bear.pcr_oi:.4f})"
    )


def test_fo_features_result_fields():
    """FOFeaturesResult must expose all required fields with correct types."""
    chain = make_synthetic_chain()
    engine = FOFeatureEngine()
    result = engine.compute(chain)

    # Float fields
    for field_name in (
        "pcr_oi", "pcr_volume", "max_pain", "iv_rank", "iv_percentile",
        "iv_skew", "atm_iv", "call_wall", "put_wall", "price_vs_max_pain_pct",
    ):
        val = getattr(result, field_name)
        assert isinstance(val, float), (
            f"Field '{field_name}' expected float, got {type(val)}"
        )

    # String fields
    for field_name in ("oi_buildup_signal", "pcr_signal"):
        val = getattr(result, field_name)
        assert isinstance(val, str), (
            f"Field '{field_name}' expected str, got {type(val)}"
        )

    # Timestamp
    from datetime import datetime as dt
    assert isinstance(result.timestamp, dt), (
        f"'timestamp' expected datetime, got {type(result.timestamp)}"
    )

    # Symbol
    assert isinstance(result.symbol, str)
