"""
signals/multi_timeframe.py — Multi-timeframe trend confirmation.

15-min signals are confirmed against 1-hour trend direction.
The 1-hour trend is constructed by resampling 4 consecutive 15-min bars
(OHLCV aggregation) and computing EMA(9) vs EMA(21) on the resampled data.

Rules:
  - 15-min BUY signal  + 1-hr trend BULL → full size (multiplier = 1.0)
  - 15-min BUY signal  + 1-hr trend BEAR → skip or half size (multiplier = 0.0 or 0.5)
  - 15-min SELL signal + 1-hr trend BEAR → full size (multiplier = 1.0)
  - 15-min SELL signal + 1-hr trend BULL → skip or half size (multiplier = 0.0 or 0.5)
  - 1-hr trend SIDEWAYS (EMAs within _SIDEWAYS_BAND_BPS of each other) → half size

Historical effect (estimated, not validated):
  - Trade count: -30% (filters counter-trend entries)
  - Win rate: +8-12 percentage points
  - Sharpe improvement: ~0.3-0.5 (estimated)

This module provides a multiplier (0.0, 0.5, or 1.0) — the caller applies it
to the position size computed by PositionSizer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

import structlog

from algo_trading.data.broker_base import Bar

log = structlog.get_logger(__name__)

# 1-hour bar = 4 × 15-min bars
_BARS_PER_HOUR: int = 4

# EMA periods on hourly bars
_EMA_FAST_PERIOD: int = 9
_EMA_SLOW_PERIOD: int = 21

# Band within which EMAs are considered "sideways" (in basis points of price)
_SIDEWAYS_BAND_BPS: float = 5.0

# Size multiplier for counter-trend signals
# 0.0 = skip entirely; 0.5 = half size; 1.0 = full size
_COUNTER_TREND_MULTIPLIER: float = 0.0
_SIDEWAYS_MULTIPLIER: float = 0.5


class HTFTrend(Enum):
    """Higher-timeframe trend direction computed from 1-hour EMA cross."""
    BULL = "BULL"        # EMA9 > EMA21 on hourly bars
    BEAR = "BEAR"        # EMA9 < EMA21 on hourly bars
    SIDEWAYS = "SIDEWAYS"  # EMAs within _SIDEWAYS_BAND_BPS


@dataclass
class HourlyBarAccumulator:
    """Accumulates 15-min bars into 1-hour OHLCV bars.

    Once _BARS_PER_HOUR (4) 15-min bars are collected, emits one hourly bar
    and resets. Partial (incomplete) hourly bars are never emitted.
    """
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: int = 0
    count: int = 0


@dataclass
class HTFState:
    """Per-symbol multi-timeframe state."""
    # 15-min bar accumulator → hourly bar
    accumulator: HourlyBarAccumulator = field(default_factory=HourlyBarAccumulator)

    # Hourly EMA values
    ema_fast: Optional[float] = None    # EMA9 on hourly bars
    ema_slow: Optional[float] = None    # EMA21 on hourly bars

    # Number of hourly bars seen (for warmup)
    hourly_bar_count: int = 0

    # Current trend
    trend: Optional[HTFTrend] = None


class MultiTimeframeConfirmation:
    """
    Multi-timeframe signal filter / size multiplier.

    Takes a 15-min base signal (+1 BUY, -1 SELL) and returns a position-size
    multiplier based on whether the 1-hour trend aligns.

    Usage (in the main bar loop):
        mtf = MultiTimeframeConfirmation()

        # Feed every 15-min bar to update hourly state:
        mtf.update(bar)

        # When a 15-min signal fires, get the size multiplier:
        multiplier = mtf.get_size_multiplier(symbol="RELIANCE", signal_direction=+1)
        final_size = base_size * multiplier

    Thread-safety: NOT internally thread-safe.
    Use from a single-threaded bar consumer.
    """

    def __init__(
        self,
        bars_per_hour: int = _BARS_PER_HOUR,
        ema_fast_period: int = _EMA_FAST_PERIOD,
        ema_slow_period: int = _EMA_SLOW_PERIOD,
        sideways_band_bps: float = _SIDEWAYS_BAND_BPS,
        counter_trend_multiplier: float = _COUNTER_TREND_MULTIPLIER,
        sideways_multiplier: float = _SIDEWAYS_MULTIPLIER,
    ) -> None:
        self._bars_per_hour = bars_per_hour
        self._slow_period = ema_slow_period
        # Ensure fast_period < slow_period (fast EMA must react faster than slow EMA)
        self._fast_period = min(ema_fast_period, max(1, ema_slow_period - 1))
        self._sideways_band_bps = sideways_band_bps
        self._counter_trend_mult = counter_trend_multiplier
        self._sideways_mult = sideways_multiplier

        self._alpha_fast = 2.0 / (self._fast_period + 1)
        self._alpha_slow = 2.0 / (ema_slow_period + 1)

        self._min_hourly_bars = ema_slow_period  # warmup period

        self._states: dict[str, HTFState] = {}

    def update(self, bar: Bar) -> Optional[HTFTrend]:
        """
        Feed one 15-min bar. Updates internal hourly state.

        Returns the current hourly trend (or None during warmup) each time
        a complete hourly bar is formed. Returns None for the other 3 bars
        in the accumulation window.

        Args:
            bar: A completed 15-min OHLCV bar.

        Returns:
            HTFTrend if a new hourly bar was completed this call, else None.
        """
        symbol = bar.symbol

        if bar.is_first_bar_of_day:
            # Reset accumulator at day open — don't carry partial hourly bars
            # across days. Keep EMA state (cross-day EMAs are fine for HTF).
            state = self._states.get(symbol)
            if state is not None:
                state.accumulator = HourlyBarAccumulator()
            else:
                state = HTFState()
                self._states[symbol] = state
        else:
            state = self._states.setdefault(symbol, HTFState())

        acc = state.accumulator
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)

        # ── Accumulate into hourly bar ────────────────────────────────────────
        if acc.open is None:
            acc.open = float(bar.open)
            acc.high = high
            acc.low = low
        else:
            acc.high = max(acc.high, high)  # type: ignore[type-var]
            acc.low = min(acc.low, low)     # type: ignore[type-var]

        acc.close = close
        acc.volume += bar.volume
        acc.count += 1

        if acc.count < self._bars_per_hour:
            return None

        # ── Hourly bar complete — update EMAs ────────────────────────────────
        hourly_close = acc.close
        state.hourly_bar_count += 1

        if state.ema_fast is None:
            state.ema_fast = hourly_close
            state.ema_slow = hourly_close
        else:
            state.ema_fast = (
                self._alpha_fast * hourly_close
                + (1 - self._alpha_fast) * state.ema_fast
            )
            state.ema_slow = (
                self._alpha_slow * hourly_close
                + (1 - self._alpha_slow) * state.ema_slow
            )

        # Reset accumulator for next hour
        state.accumulator = HourlyBarAccumulator()

        # ── Determine trend ───────────────────────────────────────────────────
        if state.hourly_bar_count < self._min_hourly_bars:
            # Not warmed up yet
            state.trend = None
            return None

        ema_fast = state.ema_fast
        ema_slow = state.ema_slow

        if ema_slow > 0:
            gap_bps = abs(ema_fast - ema_slow) / ema_slow * 10_000
        else:
            gap_bps = 0.0

        if gap_bps < self._sideways_band_bps:
            state.trend = HTFTrend.SIDEWAYS
        elif ema_fast > ema_slow:
            state.trend = HTFTrend.BULL
        else:
            state.trend = HTFTrend.BEAR

        log.debug(
            "mtf.hourly_bar",
            symbol=symbol,
            ema_fast=round(ema_fast, 2),
            ema_slow=round(ema_slow, 2),
            gap_bps=round(gap_bps, 1),
            trend=state.trend.value if state.trend else None,
            hourly_bar_count=state.hourly_bar_count,
        )
        return state.trend

    def get_size_multiplier(self, symbol: str, signal_direction: int) -> float:
        """
        Return position size multiplier for a 15-min signal given the HTF trend.

        Args:
            symbol:           NSE symbol.
            signal_direction: +1 (BUY) or -1 (SELL) from the 15-min signal.

        Returns:
            - 1.0  if signal aligns with HTF trend (or not warmed up — pass through)
            - 0.5  if HTF trend is SIDEWAYS
            - 0.0  (default) or counter_trend_multiplier if signal is counter-trend

        Note:
            During warmup (first slow_period hourly bars), the filter is bypassed
            and 1.0 is returned. This avoids blocking all signals on Day 1.
        """
        state = self._states.get(symbol)
        if state is None or state.trend is None:
            # Not warmed up — pass all signals through at full size
            return 1.0

        trend = state.trend

        if trend == HTFTrend.SIDEWAYS:
            log.info(
                "mtf.sideways",
                symbol=symbol,
                direction=signal_direction,
                multiplier=self._sideways_mult,
            )
            return self._sideways_mult

        # Aligned: BUY in BULL or SELL in BEAR
        aligned = (
            (signal_direction > 0 and trend == HTFTrend.BULL)
            or (signal_direction < 0 and trend == HTFTrend.BEAR)
        )

        if aligned:
            log.info(
                "mtf.aligned",
                symbol=symbol,
                direction=signal_direction,
                trend=trend.value,
                multiplier=1.0,
            )
            return 1.0

        # Counter-trend
        log.info(
            "mtf.counter_trend",
            symbol=symbol,
            direction=signal_direction,
            trend=trend.value,
            multiplier=self._counter_trend_mult,
        )
        return self._counter_trend_mult

    def get_trend(self, symbol: str) -> Optional[HTFTrend]:
        """Return current hourly trend for a symbol (None during warmup)."""
        state = self._states.get(symbol)
        return state.trend if state is not None else None

    def reset_symbol(self, symbol: str) -> None:
        """Reset all state for a symbol."""
        self._states.pop(symbol, None)

    def reset_all(self) -> None:
        """Reset all symbol states."""
        self._states.clear()

    def is_warmed_up(self, symbol: str) -> bool:
        """True once the hourly EMA has the required warmup bars."""
        state = self._states.get(symbol)
        return state is not None and state.trend is not None
