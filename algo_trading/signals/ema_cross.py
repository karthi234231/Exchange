"""
signals/ema_cross.py — EMA(9/21) cross strategy for Stage 1 paper trading.

Strategy: EMA(9) crosses above EMA(21) → LONG signal.
          EMA(9) crosses below EMA(21) → SHORT/EXIT signal.

Design:
  - Incremental EMA: O(1) per bar, no history buffer needed for EMA itself.
  - ATR(14): needs 14-bar rolling buffer of high/low/close.
  - Minimum bars before signal: ema_slow_period (21) — no signals before that.
  - Returns None during warmup or if no crossover occurred this bar.
  - No microstructure features, no ML — pure price action for Stage 1.

Signal types returned (TradeSignal.direction):
  +1 → BUY (EMA9 crossed above EMA21)
  -1 → SELL (EMA9 crossed below EMA21)

Stop-loss: entry_price ± ATR × atr_stop_multiplier (from settings).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

from algo_trading.config import settings
from algo_trading.data.broker_base import Bar
from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import IST, now_ist

log = structlog.get_logger(__name__)


@dataclass
class EMACrossState:
    """Per-symbol EMA cross computation state."""
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    prev_ema_fast: Optional[float] = None
    prev_ema_slow: Optional[float] = None
    bar_count: int = 0

    # ATR ring buffer — need (atr_period + 1) bars: one for previous close
    closes: deque = field(default_factory=lambda: deque(maxlen=settings.atr_period + 2))
    highs: deque = field(default_factory=lambda: deque(maxlen=settings.atr_period + 2))
    lows: deque = field(default_factory=lambda: deque(maxlen=settings.atr_period + 2))
    atr: Optional[float] = None

    # Current open position direction (+1 long, -1 short, 0 flat)
    position: int = 0


class EMACrossStrategy:
    """
    EMA(9/21) cross signal generator.

    Thread-safety: NOT internally thread-safe.
    Use from a single-threaded bar consumer (BarBuilder loop or ZMQ subscriber).

    Usage:
        strategy = EMACrossStrategy()
        # Feed bars one at a time, oldest first:
        for bar in historical_bars:
            strategy.feed_bar(bar)  # Warms up state, returns None
        # In live loop:
        signal = strategy.feed_bar(live_bar)
        if signal:
            order_executor.submit_signal(signal)
    """

    def __init__(self) -> None:
        self._fast = settings.ema_fast_period
        self._slow = settings.ema_slow_period
        self._atr_period = settings.atr_period
        self._min_bars = settings.min_bars_before_signal

        self._alpha_fast = 2.0 / (self._fast + 1)
        self._alpha_slow = 2.0 / (self._slow + 1)

        self._states: dict[str, EMACrossState] = {}

    def feed_bar(self, bar: Bar) -> Optional[TradeSignal]:
        """
        Feed one completed bar. Returns a TradeSignal on crossover, else None.

        Guards enforced here:
          - No signal during first bar of the day (opening bar skip [SURV-8])
          - No signal before min_bars warmup
          - No signal if ATR is unavailable
        """
        if bar.is_first_bar_of_day and settings.skip_first_bar:
            log.debug("ema_cross.skip_opening_bar", symbol=bar.symbol)
            return None

        state = self._states.setdefault(bar.symbol, EMACrossState(
            closes=deque(maxlen=self._atr_period + 2),
            highs=deque(maxlen=self._atr_period + 2),
            lows=deque(maxlen=self._atr_period + 2),
        ))

        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)

        # ── Update EMA ───────────────────────────────────────────────────────
        state.prev_ema_fast = state.ema_fast
        state.prev_ema_slow = state.ema_slow

        if state.ema_fast is None:
            state.ema_fast = close
            state.ema_slow = close
        else:
            state.ema_fast = self._alpha_fast * close + (1 - self._alpha_fast) * state.ema_fast
            state.ema_slow = self._alpha_slow * close + (1 - self._alpha_slow) * state.ema_slow

        # ── Update ATR buffer ─────────────────────────────────────────────────
        state.closes.append(close)
        state.highs.append(high)
        state.lows.append(low)
        state.bar_count += 1

        # ── Recompute ATR using Wilder's smoothing ────────────────────────────
        if len(state.closes) >= 2:
            state.atr = self._compute_atr(state)

        # ── Warmup check ──────────────────────────────────────────────────────
        if state.bar_count < self._min_bars:
            return None
        if state.prev_ema_fast is None or state.prev_ema_slow is None:
            return None
        if state.atr is None or state.atr <= 0:
            return None

        # ── Crossover detection ───────────────────────────────────────────────
        fast = state.ema_fast
        slow = state.ema_slow
        prev_fast = state.prev_ema_fast
        prev_slow = state.prev_ema_slow

        # Use a tiny epsilon to handle floating-point convergence drift.
        # When price is flat for many bars, EMAs converge to the price, but
        # non-representable alpha values (e.g. 2/22) can leave ema_slow slightly
        # below the true flat price.  Without epsilon, prev_fast <= prev_slow may
        # evaluate False even when both values are effectively equal, blocking the
        # first cross after a flat-then-rising sequence.
        _EPS = 1e-9
        bullish_cross = (prev_fast <= prev_slow + _EPS) and fast > slow  # EMA9 crossed above EMA21
        bearish_cross = (prev_fast >= prev_slow - _EPS) and fast < slow  # EMA9 crossed below EMA21

        signal_direction: Optional[int] = None
        if bullish_cross and state.position <= 0:
            signal_direction = +1
        elif bearish_cross and state.position >= 0:
            signal_direction = -1

        if signal_direction is None:
            return None

        # ── Construct TradeSignal ─────────────────────────────────────────────
        entry_price = bar.close
        atr_dec = Decimal(str(round(state.atr, 2)))
        stop_mult = Decimal(str(settings.atr_stop_multiplier))

        if signal_direction > 0:
            stop_loss = entry_price - atr_dec * stop_mult
        else:
            stop_loss = entry_price + atr_dec * stop_mult

        state.position = signal_direction

        cross_type = "BULLISH" if signal_direction > 0 else "BEARISH"
        log.info(
            "ema_cross.signal",
            symbol=bar.symbol,
            cross=cross_type,
            ema_fast=f"{fast:.4f}",
            ema_slow=f"{slow:.4f}",
            close=str(bar.close),
            atr=str(atr_dec),
            stop=str(stop_loss),
            bars_since_open=state.bar_count,
        )

        return TradeSignal(
            symbol=bar.symbol,
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            atr=atr_dec,
            confidence=self._compute_confidence(fast, slow, state.atr),
            timestamp=bar.timestamp,
        )

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a symbol (call at start of each trading day)."""
        self._states.pop(symbol, None)

    def reset_all(self) -> None:
        """Reset all symbol states (call at market open)."""
        self._states.clear()

    def get_state(self, symbol: str) -> Optional[EMACrossState]:
        return self._states.get(symbol)

    def is_warmed_up(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        if state is None:
            return False
        return state.bar_count >= self._min_bars and state.atr is not None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_atr(self, state: EMACrossState) -> float:
        """
        ATR via Wilder's smoothing.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        closes = list(state.closes)
        highs = list(state.highs)
        lows = list(state.lows)

        if len(closes) < 2:
            return 0.0

        true_ranges: list[float] = []
        for i in range(1, len(closes)):
            prev_c = closes[i - 1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_c),
                abs(lows[i] - prev_c),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return 0.0

        # Simple mean for initial ATR, then Wilder's smoothing for subsequent
        if state.atr is None or state.atr <= 0:
            return sum(true_ranges) / len(true_ranges)

        # Wilder's smoothing: ATR = (prev_ATR × (n-1) + current_TR) / n
        n = self._atr_period
        return (state.atr * (n - 1) + true_ranges[-1]) / n

    @staticmethod
    def _compute_confidence(ema_fast: float, ema_slow: float, atr: float) -> float:
        """
        Signal confidence [0, 1].
        Higher when the gap between EMAs is larger relative to ATR.
        Normalised to 1.0 when gap >= 0.5 * ATR.
        """
        if atr <= 0:
            return 0.5
        gap = abs(ema_fast - ema_slow)
        confidence = min(1.0, gap / (0.5 * atr))
        return round(confidence, 3)
