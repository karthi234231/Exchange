"""
tests/test_fo_signal_generator.py — Tests for fo/signal_generator.py.
"""
from __future__ import annotations

import math

import pytest

from algo_trading.fo.signal_generator import (
    FOSignal,
    FOSignalGenerator,
    STRATEGY_NO_TRADE,
    STRATEGY_SHORT_STRANGLE,
    STRATEGY_IRON_CONDOR,
    STRATEGY_BULL_PUT_SPREAD,
    STRATEGY_LONG_STRADDLE,
    DIRECTION_SHORT_PREMIUM,
    DIRECTION_LONG_PREMIUM,
    DIRECTION_NEUTRAL,
)
from algo_trading.fo.portfolio_risk import VRPEstimate


# ── VRPEstimate factory ────────────────────────────────────────────────────────

def _vrp(signal: str = "favorable_short_premium", vrp: float = 0.05, strength: float = 0.25) -> VRPEstimate:
    return VRPEstimate(
        realized_vol=0.15 + vrp,
        implied_vol_atm=0.15,
        vrp=vrp,
        lookback_bars=22,
        signal=signal,
        signal_strength=strength,
    )


@pytest.fixture(scope="module")
def gen() -> FOSignalGenerator:
    return FOSignalGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# DTE gate
# ─────────────────────────────────────────────────────────────────────────────

class TestDTEGate:

    def test_dte_too_low_no_trade(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=3)
        assert sig.strategy_name == STRATEGY_NO_TRADE

    def test_dte_too_high_no_trade(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=60)
        assert sig.strategy_name == STRATEGY_NO_TRADE

    def test_dte_at_min_boundary_passes(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=7)
        # DTE=7 is inclusive; signal may be actionable
        assert sig.strategy_name != STRATEGY_NO_TRADE or "DTE" not in sig.reasoning[0]

    def test_dte_reasoning_mentions_dte(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=2)
        assert any("DTE" in r for r in sig.reasoning)


# ─────────────────────────────────────────────────────────────────────────────
# Short-premium strategy selection
# ─────────────────────────────────────────────────────────────────────────────

class TestShortPremiumSignals:

    def test_high_iv_balanced_pcr_iron_condor(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.1, days_to_expiry=21)
        assert sig.strategy_name == STRATEGY_IRON_CONDOR
        assert sig.trade_direction == DIRECTION_SHORT_PREMIUM

    def test_high_iv_bearish_pcr_short_strangle(self, gen):
        sig = gen.generate(iv_rank=70.0, vrp_estimate=_vrp(), pcr=1.7, days_to_expiry=21)
        assert sig.strategy_name == STRATEGY_SHORT_STRANGLE

    def test_high_iv_bullish_pcr_bull_put_spread(self, gen):
        sig = gen.generate(iv_rank=65.0, vrp_estimate=_vrp(), pcr=0.6, days_to_expiry=21)
        assert sig.strategy_name == STRATEGY_BULL_PUT_SPREAD

    def test_confidence_is_positive(self, gen):
        sig = gen.generate(iv_rank=80.0, vrp_estimate=_vrp(strength=0.7), pcr=1.5, days_to_expiry=21)
        assert sig.confidence > 0.0

    def test_low_confidence_returns_no_trade(self):
        # Set min_confidence very high to force no_trade
        strict = FOSignalGenerator(min_confidence=0.99)
        sig = strict.generate(
            iv_rank=62.0,
            vrp_estimate=_vrp(signal="neutral", vrp=0.01, strength=0.05),
            pcr=1.0,
            days_to_expiry=21,
        )
        assert sig.strategy_name == STRATEGY_NO_TRADE

    def test_reasoning_is_non_empty(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=21)
        assert len(sig.reasoning) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Long-premium signal
# ─────────────────────────────────────────────────────────────────────────────

class TestLongPremiumSignal:

    def test_low_iv_unfavorable_vrp_long_straddle(self, gen):
        sig = gen.generate(
            iv_rank=20.0,
            vrp_estimate=_vrp(signal="unfavorable_short_premium", vrp=-0.10, strength=0.50),
            pcr=1.0,
            days_to_expiry=21,
        )
        assert sig.strategy_name == STRATEGY_LONG_STRADDLE
        assert sig.trade_direction == DIRECTION_LONG_PREMIUM

    def test_low_iv_but_favorable_vrp_no_long_straddle(self, gen):
        # VRP favorable → don't buy vol even if IV is low
        sig = gen.generate(
            iv_rank=20.0,
            vrp_estimate=_vrp(signal="favorable_short_premium", vrp=0.05, strength=0.25),
            pcr=1.0,
            days_to_expiry=21,
        )
        # Should NOT trigger long straddle (VRP not unfavorable)
        assert sig.strategy_name != STRATEGY_LONG_STRADDLE


# ─────────────────────────────────────────────────────────────────────────────
# FOSignal properties
# ─────────────────────────────────────────────────────────────────────────────

class TestFOSignalProperties:

    def test_is_actionable_true_for_real_strategy(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=21)
        assert sig.is_actionable is True

    def test_is_actionable_false_for_no_trade(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=2)
        assert sig.is_actionable is False

    def test_summary_contains_strategy_name(self, gen):
        sig = gen.generate(iv_rank=75.0, vrp_estimate=_vrp(), pcr=1.2, days_to_expiry=21)
        assert sig.strategy_name.upper() in sig.summary()

    def test_confidence_bounded_0_1(self, gen):
        for iv_rank in [0, 30, 60, 80, 100]:
            sig = gen.generate(iv_rank=float(iv_rank), vrp_estimate=_vrp(), pcr=1.0, days_to_expiry=21)
            assert 0.0 <= sig.confidence <= 1.0
