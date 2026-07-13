"""
tests/test_tick_aggregator.py

Unit tests for TickAggregator and TickBar from algo_trading/data/tick_aggregator.py.

Tests cover:
  - Bar completion lifecycle (interval boundary detection)
  - OHLCV correctness
  - VWAP computation
  - Buy/sell volume tracking
  - Tick validation (monotonicity, zero price, zero quantity)
  - Reset behaviour
  - Multiple sequential bars
  - Incomplete bar peeking
  - Trade count (n_trades)
"""
from __future__ import annotations

import pytest
import numpy as np
from algo_trading.data.tick_aggregator import TickAggregator, TickBar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NS = 1_000_000_000          # nanoseconds per second
INTERVAL = 5                # seconds per bar (used in most tests)
INTERVAL_NS = INTERVAL * NS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tick(timestamp_ns: int, price: float, qty: int, side: str = "BUY"):
    """Return a tick tuple (timestamp_ns, price, qty, side)."""
    return (timestamp_ns, price, qty, side)


def add_ticks(agg: TickAggregator, ticks):
    """Feed a sequence of ticks to *agg*; collect and return completed bars."""
    bars = []
    for ts, price, qty, side in ticks:
        bar = agg.add_tick(ts, price, qty, side)
        if bar is not None:
            bars.append(bar)
    return bars


# ---------------------------------------------------------------------------
# Basic bar lifecycle
# ---------------------------------------------------------------------------

class TestNoBarBeforeInterval:
    """No completed bar when all ticks fall within the same 5-second window."""

    def test_no_bar_before_interval(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        base = 0 * NS  # epoch 0 ns — floor is bar window [0, 5s)
        ticks = [
            make_tick(base + 1 * NS, 100.0, 10),   # t = 1 s
            make_tick(base + 2 * NS, 101.0, 20),   # t = 2 s
            make_tick(base + 3 * NS, 99.0,  30),   # t = 3 s
        ]
        bars = add_ticks(agg, ticks)
        assert bars == [], "Expected no completed bar within the 5-second window"


class TestBarCompletedAfterInterval:
    """A completed bar is returned when a tick crosses the interval boundary."""

    def test_bar_completed_after_interval(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        # Ticks in window [0s, 5s)
        ticks = [
            make_tick(1 * NS, 100.0, 10),
            make_tick(3 * NS, 102.0, 5),
        ]
        add_ticks(agg, ticks)

        # This tick is in the next window [5s, 10s), should close previous bar
        bars = []
        bar = agg.add_tick(6 * NS, 103.0, 8, "BUY")
        if bar is not None:
            bars.append(bar)

        assert len(bars) == 1, "Expected one completed bar"
        assert isinstance(bars[0], TickBar)


# ---------------------------------------------------------------------------
# OHLC correctness
# ---------------------------------------------------------------------------

class TestOHLCCorrect:
    """First tick is open, last is close, max is high, min is low."""

    def test_ohlc_correct(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        prices = [100.0, 105.0, 98.0, 103.0]
        for i, price in enumerate(prices):
            agg.add_tick((i + 1) * NS, price, 10, "BUY")

        # Force bar completion with a tick in the next interval
        bar = agg.add_tick(7 * NS, 104.0, 5, "BUY")

        assert bar is not None
        assert bar.open == prices[0],  f"open should be {prices[0]}, got {bar.open}"
        assert bar.close == prices[-1], f"close should be {prices[-1]}, got {bar.close}"
        assert bar.high == max(prices),  f"high should be {max(prices)}, got {bar.high}"
        assert bar.low == min(prices),   f"low should be {min(prices)}, got {bar.low}"


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

class TestVolumeTracking:
    """Total volume equals the sum of all tick quantities in the bar."""

    def test_volume_summed(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        qtys = [10, 20, 30, 5]
        for i, qty in enumerate(qtys):
            agg.add_tick((i + 1) * NS, 100.0, qty, "BUY")

        bar = agg.add_tick(7 * NS, 100.0, 1, "BUY")
        assert bar is not None
        assert bar.volume == sum(qtys), f"Expected volume {sum(qtys)}, got {bar.volume}"


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class TestVWAPCorrect:
    """VWAP = Σ(price × qty) / Σqty."""

    def test_vwap_correct(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        ticks = [
            (100.0, 10),
            (200.0, 20),
            (150.0, 30),
        ]
        for i, (price, qty) in enumerate(ticks):
            agg.add_tick((i + 1) * NS, price, qty, "BUY")

        bar = agg.add_tick(7 * NS, 0.01, 1, "BUY")  # trigger close
        # VWAP of the ticks above (not including the trigger tick)
        assert bar is not None
        expected_vwap = sum(p * q for p, q in ticks) / sum(q for _, q in ticks)
        assert abs(bar.vwap - expected_vwap) < 1e-6, (
            f"VWAP mismatch: expected {expected_vwap:.6f}, got {bar.vwap:.6f}"
        )


# ---------------------------------------------------------------------------
# Buy / Sell volume
# ---------------------------------------------------------------------------

class TestBuySellVolume:
    """buy_volume tracks BUY-side; sell_volume tracks SELL-side quantities."""

    def test_buy_volume_tracked(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        agg.add_tick(1 * NS, 100.0, 40, "BUY")
        agg.add_tick(2 * NS, 101.0, 10, "SELL")
        agg.add_tick(3 * NS, 102.0, 25, "BUY")

        bar = agg.add_tick(7 * NS, 103.0, 1, "BUY")
        assert bar is not None
        assert bar.buy_volume == 65, f"Expected buy_volume=65, got {bar.buy_volume}"

    def test_sell_volume_tracked(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        agg.add_tick(1 * NS, 100.0, 40, "BUY")
        agg.add_tick(2 * NS, 101.0, 10, "SELL")
        agg.add_tick(3 * NS, 102.0, 15, "SELL")

        bar = agg.add_tick(7 * NS, 103.0, 1, "BUY")
        assert bar is not None
        assert bar.sell_volume == 25, f"Expected sell_volume=25, got {bar.sell_volume}"


# ---------------------------------------------------------------------------
# Validation — invalid ticks are silently skipped
# ---------------------------------------------------------------------------

class TestTickValidation:
    """Invalid ticks (non-monotonic, zero price, zero qty) are rejected silently."""

    def test_monotonicity_violation_skipped(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        agg.add_tick(3 * NS, 100.0, 10, "BUY")

        # Earlier timestamp — should be skipped, no crash
        result = agg.add_tick(2 * NS, 101.0, 5, "BUY")
        assert result is None, "Backward-timestamp tick must return None"

        # Bar should still contain only the first tick
        bar_peek = agg.get_current_incomplete_bar()
        assert bar_peek is not None
        assert bar_peek.n_trades == 1, (
            f"Only 1 valid tick expected, got n_trades={bar_peek.n_trades}"
        )

    def test_zero_price_rejected(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        result = agg.add_tick(1 * NS, 0.0, 10, "BUY")
        assert result is None
        bar_peek = agg.get_current_incomplete_bar()
        assert bar_peek is None, "No bar state should exist after a zero-price tick"

    def test_zero_quantity_rejected(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        result = agg.add_tick(1 * NS, 100.0, 0, "BUY")
        assert result is None
        bar_peek = agg.get_current_incomplete_bar()
        assert bar_peek is None, "No bar state should exist after a zero-quantity tick"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    """After reset(), the aggregator behaves as if freshly constructed."""

    def test_reset_clears_state(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        agg.add_tick(1 * NS, 100.0, 10, "BUY")
        agg.add_tick(2 * NS, 101.0, 20, "BUY")

        agg.reset()

        # After reset, there should be no in-progress bar
        assert agg.get_current_incomplete_bar() is None

        # New ticks start a fresh bar — no completed bar yet
        result = agg.add_tick(1 * NS, 200.0, 5, "SELL")
        assert result is None, "No completed bar expected immediately after reset"

        peek = agg.get_current_incomplete_bar()
        assert peek is not None
        assert peek.open == 200.0, f"Open should be 200.0 after reset, got {peek.open}"


# ---------------------------------------------------------------------------
# Multiple bars in sequence
# ---------------------------------------------------------------------------

class TestMultipleBars:
    """30 seconds of ticks at 1-tick-per-second → 6 completed 5-second bars."""

    def test_multiple_bars_in_sequence(self):
        agg = TickAggregator(interval_seconds=INTERVAL)
        bars = []

        # Feed ticks at t=0s, 1s, 2s, ..., 29s and then one at t=30s to seal last bar
        for sec in range(31):
            bar = agg.add_tick(sec * NS, 100.0 + sec, 10, "BUY")
            if bar is not None:
                bars.append(bar)

        # We expect 6 completed bars: [0-4s], [5-9s], [10-14s], [15-19s], [20-24s], [25-29s]
        assert len(bars) == 6, f"Expected 6 completed bars, got {len(bars)}"

        # Each completed bar should have 5 ticks
        for i, bar in enumerate(bars):
            assert bar.n_trades == 5, (
                f"Bar {i} should have 5 trades, got {bar.n_trades}"
            )


# ---------------------------------------------------------------------------
# Incomplete bar peeking
# ---------------------------------------------------------------------------

class TestIncompletBarPeek:
    """get_current_incomplete_bar() exposes the in-progress bar without completing it."""

    def test_incomplete_bar_peek(self):
        agg = TickAggregator(interval_seconds=INTERVAL)

        agg.add_tick(1 * NS, 100.0, 10, "BUY")
        agg.add_tick(2 * NS, 102.0, 20, "SELL")

        bar = agg.get_current_incomplete_bar()

        assert bar is not None, "Should return an in-progress bar"
        assert isinstance(bar, TickBar)
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 100.0
        assert bar.close == 102.0
        assert bar.volume == 30
        assert bar.n_trades == 2


# ---------------------------------------------------------------------------
# Trade count
# ---------------------------------------------------------------------------

class TestNTradesCounted:
    """n_trades equals the number of ticks that contributed to the completed bar."""

    def test_n_trades_counted(self):
        # Use interval_seconds=10 so all 7 ticks at t=1..7s fall in the same bar [0s, 10s)
        agg = TickAggregator(interval_seconds=10)
        n = 7
        for i in range(n):
            agg.add_tick((i + 1) * NS, 100.0 + i, 5, "BUY")

        # Trigger at t=10s closes bar [0s, 10s) which contains all 7 ticks
        bar = agg.add_tick(10 * NS, 200.0, 1, "SELL")
        assert bar is not None
        assert bar.n_trades == n, f"Expected n_trades={n}, got {bar.n_trades}"
