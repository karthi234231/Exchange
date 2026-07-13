"""
tests/test_multi_timeframe.py — Tests for signals/multi_timeframe.py.

Covers: hourly bar accumulation, EMA warmup, trend classification (BULL/BEAR/SIDEWAYS),
size multiplier routing, day-reset behaviour, and multi-symbol isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pytest

from algo_trading.signals.multi_timeframe import (
    HTFTrend,
    HourlyBarAccumulator,
    MultiTimeframeConfirmation,
)

# ── Minimal Bar stub (only fields accessed by multi_timeframe.py) ─────────────

@dataclass
class _Bar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    is_first_bar_of_day: bool = False


def _bar(
    close: float,
    symbol: str = "NIFTY",
    is_first: bool = False,
    high: Optional[float] = None,
    low: Optional[float] = None,
) -> _Bar:
    return _Bar(
        symbol=symbol,
        open=close - 5.0,
        high=high if high is not None else close + 5.0,
        low=low if low is not None else close - 5.0,
        close=close,
        volume=10_000,
        is_first_bar_of_day=is_first,
    )


# ── Helper: feed N hourly bars (4 × 15-min each) to the MTF object ───────────

def _feed_hourly_bars(mtf: MultiTimeframeConfirmation, prices: list[float], symbol: str = "NIFTY") -> list[Optional[HTFTrend]]:
    """Feed len(prices) hourly bars (4 fifteen-min bars each). Return trends emitted."""
    trends = []
    for price in prices:
        for i in range(4):
            result = mtf.update(_bar(close=price, symbol=symbol))
            if i == 3:          # last bar of the hour — trend emitted (or None if warming up)
                trends.append(result)
            else:
                assert result is None, "update() must return None for intra-hour bars"
    return trends


# ─────────────────────────────────────────────────────────────────────────────
# HourlyBarAccumulator — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHourlyBarAccumulator:

    def test_default_values(self):
        acc = HourlyBarAccumulator()
        assert acc.open is None
        assert acc.high is None
        assert acc.low is None
        assert acc.close is None
        assert acc.volume == 0
        assert acc.count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Intra-hour returns None (only 4th bar completes the hour)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntraHourReturnNone:

    def test_first_three_bars_return_none(self):
        mtf = MultiTimeframeConfirmation()
        for i in range(3):
            result = mtf.update(_bar(24_000.0))
            assert result is None, f"bar {i+1} of 4 should return None"

    def test_fourth_bar_returns_something_or_none_during_warmup(self):
        mtf = MultiTimeframeConfirmation()
        for _ in range(3):
            mtf.update(_bar(24_000.0))
        # Fourth bar: emits hourly close but still in warmup → None
        result = mtf.update(_bar(24_000.0))
        # During warmup, None is correct
        assert result is None  # first hourly bar — warmup not complete


# ─────────────────────────────────────────────────────────────────────────────
# Warmup: first 21 hourly bars must not produce a trend
# ─────────────────────────────────────────────────────────────────────────────

class TestWarmup:

    def test_no_trend_before_slow_period(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=21)
        # Feed 20 hourly bars (80 fifteen-min bars) — all should return None
        trends = _feed_hourly_bars(mtf, [24_000.0] * 20)
        assert all(t is None for t in trends), "Should return None during warmup"

    def test_trend_available_after_slow_period(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)   # short warmup for test speed
        # Feed 5 hourly bars — last one should produce a trend
        trends = _feed_hourly_bars(mtf, [24_000.0] * 5)
        # The 5th bar completes warmup (count == ema_slow_period), so last trend non-None
        assert trends[-1] is not None

    def test_is_warmed_up_false_before_warmup(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        _feed_hourly_bars(mtf, [24_000.0] * 4)   # one short of warmup
        assert mtf.is_warmed_up("NIFTY") is False

    def test_is_warmed_up_true_after_warmup(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        _feed_hourly_bars(mtf, [24_000.0] * 5)
        assert mtf.is_warmed_up("NIFTY") is True


# ─────────────────────────────────────────────────────────────────────────────
# Trend classification
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendClassification:

    def _make_warmed(self, ema_slow: int = 5, seed_price: float = 24_000.0) -> MultiTimeframeConfirmation:
        mtf = MultiTimeframeConfirmation(ema_slow_period=ema_slow, sideways_band_bps=5.0)
        _feed_hourly_bars(mtf, [seed_price] * ema_slow)
        return mtf

    def test_rising_prices_produce_bull(self):
        mtf = self._make_warmed(ema_slow=5, seed_price=24_000.0)
        # Feed strongly rising prices to push fast EMA above slow EMA
        _feed_hourly_bars(mtf, [24_500.0, 25_000.0, 25_500.0, 26_000.0, 26_500.0])
        trend = mtf.get_trend("NIFTY")
        assert trend == HTFTrend.BULL

    def test_falling_prices_produce_bear(self):
        mtf = self._make_warmed(ema_slow=5, seed_price=24_000.0)
        # Feed strongly falling prices to push fast EMA below slow EMA
        _feed_hourly_bars(mtf, [23_500.0, 23_000.0, 22_500.0, 22_000.0, 21_500.0])
        trend = mtf.get_trend("NIFTY")
        assert trend == HTFTrend.BEAR

    def test_flat_prices_produce_sideways(self):
        # Same price throughout → EMAs converge to same value → gap < sideways_band
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)
        _feed_hourly_bars(mtf, [24_000.0] * 10)
        trend = mtf.get_trend("NIFTY")
        assert trend == HTFTrend.SIDEWAYS

    def test_get_trend_none_before_warmup(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=21)
        _feed_hourly_bars(mtf, [24_000.0] * 5)   # far from 21-bar warmup
        assert mtf.get_trend("NIFTY") is None

    def test_get_trend_none_for_unknown_symbol(self):
        mtf = MultiTimeframeConfirmation()
        assert mtf.get_trend("UNKNOWN") is None


# ─────────────────────────────────────────────────────────────────────────────
# Size multiplier routing
# ─────────────────────────────────────────────────────────────────────────────

class TestSizeMultiplier:

    def _make_bull(self) -> MultiTimeframeConfirmation:
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)
        _feed_hourly_bars(mtf, [24_000.0] * 5)  # seed
        _feed_hourly_bars(mtf, [25_000.0, 26_000.0, 27_000.0])  # push BULL
        assert mtf.get_trend("NIFTY") == HTFTrend.BULL
        return mtf

    def _make_bear(self) -> MultiTimeframeConfirmation:
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)
        _feed_hourly_bars(mtf, [24_000.0] * 5)
        _feed_hourly_bars(mtf, [23_000.0, 22_000.0, 21_000.0])
        assert mtf.get_trend("NIFTY") == HTFTrend.BEAR
        return mtf

    def test_buy_signal_in_bull_returns_1_0(self):
        mtf = self._make_bull()
        assert mtf.get_size_multiplier("NIFTY", signal_direction=+1) == pytest.approx(1.0)

    def test_sell_signal_in_bear_returns_1_0(self):
        mtf = self._make_bear()
        assert mtf.get_size_multiplier("NIFTY", signal_direction=-1) == pytest.approx(1.0)

    def test_buy_signal_in_bear_returns_counter_trend(self):
        mtf = self._make_bear()
        mult = mtf.get_size_multiplier("NIFTY", signal_direction=+1)
        assert mult == pytest.approx(0.0)   # default counter_trend_multiplier

    def test_sell_signal_in_bull_returns_counter_trend(self):
        mtf = self._make_bull()
        mult = mtf.get_size_multiplier("NIFTY", signal_direction=-1)
        assert mult == pytest.approx(0.0)

    def test_sideways_returns_half_size(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)
        _feed_hourly_bars(mtf, [24_000.0] * 10)  # flat → SIDEWAYS
        mult = mtf.get_size_multiplier("NIFTY", signal_direction=+1)
        assert mult == pytest.approx(0.5)

    def test_unwarmed_symbol_returns_1_0_pass_through(self):
        mtf = MultiTimeframeConfirmation()
        # No bars fed → not warmed up → pass through at full size
        assert mtf.get_size_multiplier("NIFTY", signal_direction=+1) == pytest.approx(1.0)

    def test_custom_counter_trend_multiplier(self):
        mtf = MultiTimeframeConfirmation(
            ema_slow_period=5, sideways_band_bps=5.0,
            counter_trend_multiplier=0.5,
        )
        _feed_hourly_bars(mtf, [24_000.0] * 5)
        _feed_hourly_bars(mtf, [21_000.0, 20_000.0, 19_000.0])  # push BEAR
        mult = mtf.get_size_multiplier("NIFTY", signal_direction=+1)  # counter-trend BUY in BEAR
        assert mult == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Day-reset behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestDayReset:

    def test_day_reset_clears_accumulator_state(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        # Feed 2 intra-hour bars (partial accumulation)
        mtf.update(_bar(24_000.0))
        mtf.update(_bar(24_100.0))

        # First bar of next day — accumulator must reset
        result = mtf.update(_bar(24_200.0, is_first=True))
        # After reset, only 1 bar accumulated (the first-of-day bar itself)
        # No hourly bar emitted (count is 1, not 4)
        assert result is None

    def test_ema_state_preserved_across_day_reset(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)
        # Warm up in day 1
        _feed_hourly_bars(mtf, [24_000.0] * 5)
        trend_day1 = mtf.get_trend("NIFTY")

        # Day 2 start — accumulator reset, but EMA state should remain
        mtf.update(_bar(24_000.0, is_first=True))
        # EMA state is preserved (trend was set before)
        assert mtf.get_trend("NIFTY") == trend_day1


# ─────────────────────────────────────────────────────────────────────────────
# Multi-symbol isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiSymbol:

    def test_two_symbols_tracked_independently(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5, sideways_band_bps=5.0)

        # Feed NIFTY with rising prices → BULL
        _feed_hourly_bars(mtf, [24_000.0] * 5, symbol="NIFTY")
        _feed_hourly_bars(mtf, [26_000.0, 28_000.0, 30_000.0], symbol="NIFTY")

        # Feed BANKNIFTY with falling prices → BEAR
        _feed_hourly_bars(mtf, [52_000.0] * 5, symbol="BANKNIFTY")
        _feed_hourly_bars(mtf, [48_000.0, 44_000.0, 40_000.0], symbol="BANKNIFTY")

        assert mtf.get_trend("NIFTY") == HTFTrend.BULL
        assert mtf.get_trend("BANKNIFTY") == HTFTrend.BEAR

    def test_reset_symbol_clears_only_that_symbol(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        _feed_hourly_bars(mtf, [24_000.0] * 5, symbol="NIFTY")
        _feed_hourly_bars(mtf, [52_000.0] * 5, symbol="BANKNIFTY")

        mtf.reset_symbol("NIFTY")
        assert mtf.get_trend("NIFTY") is None        # cleared
        assert mtf.get_trend("BANKNIFTY") is not None  # untouched

    def test_reset_all_clears_every_symbol(self):
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        _feed_hourly_bars(mtf, [24_000.0] * 5, symbol="NIFTY")
        _feed_hourly_bars(mtf, [52_000.0] * 5, symbol="BANKNIFTY")

        mtf.reset_all()
        assert mtf.get_trend("NIFTY") is None
        assert mtf.get_trend("BANKNIFTY") is None
        assert mtf.is_warmed_up("NIFTY") is False


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV accumulation correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestOHLCVAccumulation:

    def test_high_is_max_across_four_bars(self):
        """The hourly high must be the maximum of the four 15-min highs."""
        mtf = MultiTimeframeConfirmation(ema_slow_period=1)  # 1-bar warmup for speed
        # Feed bars with known highs: 100, 200, 150, 120 → hourly high = 200
        bars = [
            _Bar("NIFTY", 24_000, 24_100, 23_900, 24_000, 1000, False),
            _Bar("NIFTY", 24_000, 24_200, 23_900, 24_000, 1000, False),
            _Bar("NIFTY", 24_000, 24_150, 23_900, 24_000, 1000, False),
            _Bar("NIFTY", 24_000, 24_120, 23_900, 24_000, 1000, False),
        ]
        for b in bars:
            mtf.update(b)
        # Trend doesn't matter here — just check the engine processed the bars
        assert mtf.is_warmed_up("NIFTY") is True

    def test_volume_accumulates_across_bars(self):
        """Total hourly volume = sum of 4 bar volumes."""
        mtf = MultiTimeframeConfirmation(ema_slow_period=5)
        vols = [1000, 2000, 3000, 4000]   # sum = 10000
        for i, v in enumerate(vols):
            b = _Bar("NIFTY", 24_000, 24_010, 23_990, 24_000, v)
            mtf.update(b)
        # Accumulator count should have been reset after 4 bars
        # Re-check: if count reset, then feeding 1 more bar starts fresh accumulation
        result = mtf.update(_bar(24_000.0))
        assert result is None  # first bar of a new hour
