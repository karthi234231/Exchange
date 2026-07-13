"""
signals/opening_range_breakout.py — Opening Range Breakout (ORB) strategy.

Opening range = high and low of the first 2 bars (09:15–09:44 IST).
Breakout signal fires when price closes above opening high (BUY) or below
opening low (SELL) during 09:45–12:00 IST, with volume confirmation.

Why ORB works on NSE:
  - Overnight news and pre-open orders drive a genuine directional impulse
    in the first 30 minutes. The ORB captures this impulse.
  - Volume confirmation filters gap fills and noise breakouts.
  - Time exit before 12:00 PM avoids the midday dead zone.

Signal rules:
  1. Record high/low of first 2 completed bars (09:15 and 09:30).
  2. After 09:44, watch for a CLOSE above opening high + volume > 1.5× 20-bar avg.
  3. Signal direction: +1 if above opening high, -1 if below opening low.
  4. Stop-loss: below opening low (for BUY), above opening high (for SELL).
  5. No new ORB signals after 12:00 PM IST.
  6. Only one ORB signal per symbol per day.
  7. Minimum opening range size = 0.2% of price (avoid flat opens).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, time
from decimal import Decimal
from typing import Optional

import structlog

from algo_trading.config import settings
from algo_trading.data.broker_base import Bar
from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import IST

log = structlog.get_logger(__name__)

# Opening range = first 2 bars (09:15 and 09:30 IST)
_ORB_BARS: int = 2
# After this bar time opening range is fully built (09:44 end of 2nd bar)
_ORB_BUILD_COMPLETE_TIME: time = time(9, 44)
# No ORB signals after this time (midday dead zone)
_ORB_CUTOFF_TIME: time = time(12, 0)
# No new trades after this time [SURV-8]
_NO_NEW_TRADES_TIME: time = time(15, 15)
# Volume multiplier — bar volume must exceed this × 20-bar average
_VOLUME_MULTIPLIER: float = 1.5
# Minimum range size as fraction of price (0.002 = 0.2%)
_MIN_RANGE_FRACTION: float = 0.002


@dataclass
class ORBState:
    """
    Per-symbol opening range breakout state.
    Reset every trading day via is_first_bar_of_day flag.
    """
    # Opening range bounds (populated during first _ORB_BARS bars)
    opening_high: Optional[float] = None
    opening_low: Optional[float] = None

    # Number of bars seen today
    bars_in_opening: int = 0

    # Whether opening range has been fully collected
    range_ready: bool = False

    # Whether a signal has been fired today (one per day per symbol)
    signal_fired: bool = False

    # P1-FIX-16: Whether the opening range can be built this session.
    # Set to False if the first bar received is already past the ORB window
    # (e.g. process started late, or bars were missing at open).
    # A False value means ORB is ineligible for this symbol for the rest of the day.
    range_eligible: bool = True

    # 20-bar rolling volume deque for volume confirmation
    volume_history: deque = field(
        default_factory=lambda: deque(maxlen=20)
    )

    # ATR rolling buffers (14-bar Wilder)
    closes: deque = field(default_factory=lambda: deque(maxlen=16))
    highs: deque = field(default_factory=lambda: deque(maxlen=16))
    lows: deque = field(default_factory=lambda: deque(maxlen=16))
    atr: Optional[float] = None

    # Total bars seen (for debug / warm-up tracking)
    total_bars: int = 0


class OpeningRangeBreakoutStrategy:
    """
    Opening Range Breakout strategy.

    Opening range = high and low of the first 2 completed 15-min bars (09:15 and 09:30).
    Breakout signal fires on a close above/below that range AFTER the range is complete,
    provided volume exceeds 1.5× the 20-bar rolling average.

    One ORB signal maximum per symbol per trading day. Signals cut off at 12:00 PM IST.

    Interface identical to EMACrossStrategy: feed_bar() → Optional[TradeSignal].

    Usage:
        strategy = OpeningRangeBreakoutStrategy()
        # Feed bars one at a time (oldest first during warmup):
        for bar in historical_bars:
            strategy.feed_bar(bar)
        # In live loop:
        signal = strategy.feed_bar(live_bar)
        if signal:
            order_executor.submit_signal(signal)
    """

    def __init__(
        self,
        orb_bars: int = _ORB_BARS,
        cutoff_time: time = _ORB_CUTOFF_TIME,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        volume_multiplier: float = _VOLUME_MULTIPLIER,
        min_range_fraction: float = _MIN_RANGE_FRACTION,
    ) -> None:
        self._orb_bars = orb_bars
        self._cutoff_time = cutoff_time
        self._atr_period = atr_period
        self._atr_stop_multiplier = Decimal(str(atr_stop_multiplier))
        self._volume_multiplier = volume_multiplier
        self._min_range_fraction = min_range_fraction

        self._states: dict[str, ORBState] = {}
        # Explicit per-symbol date tracking for day rollover detection
        self._current_dates: dict[str, Optional[date]] = {}

    def feed_bar(self, bar: Bar) -> Optional[TradeSignal]:
        """
        Feed one completed OHLCV bar.

        Returns a TradeSignal on a valid confirmed breakout, else None.

        Guards enforced:
          - Daily state reset on is_first_bar_of_day
          - Opening range only built from first _orb_bars bars
          - No signal before range is complete
          - Minimum range size (0.2% of price)
          - Volume > 1.5× 20-bar average required
          - No signal after cutoff_time (12:00 PM IST)
          - No signal after no-new-trades time (15:15 IST)
          - One signal maximum per symbol per day
        """
        symbol = bar.symbol
        bar_time = bar.timestamp.astimezone(IST).time()

        # ── Day reset (explicit date comparison — timestamp-safe) ─────────────
        bar_date = bar.timestamp.date()
        prev_date = self._current_dates.get(symbol)
        if prev_date is None or bar_date != prev_date:
            self._current_dates[symbol] = bar_date
            # P1-FIX-16: Determine eligibility based on the timestamp of the first
            # bar seen for this day. If it's already past the ORB build window
            # (09:44 IST — when the 2nd opening bar closes), the opening range
            # cannot be faithfully reconstructed; mark ineligible for the day.
            is_eligible = bar_time <= _ORB_BUILD_COMPLETE_TIME
            if not is_eligible:
                log.warning(
                    "orb.ineligible_late_start",
                    symbol=symbol,
                    bar_time=str(bar_time),
                    cutoff=str(_ORB_BUILD_COMPLETE_TIME),
                    reason="First bar received after ORB window — skipping ORB for today",
                )
            self._states[symbol] = ORBState(
                range_eligible=is_eligible,
                volume_history=deque(maxlen=20),
                closes=deque(maxlen=self._atr_period + 2),
                highs=deque(maxlen=self._atr_period + 2),
                lows=deque(maxlen=self._atr_period + 2),
            )
            log.debug("orb.day_reset", symbol=symbol, bar_date=str(bar_date),
                      range_eligible=is_eligible)

        state = self._states.setdefault(symbol, ORBState(
            volume_history=deque(maxlen=20),
            closes=deque(maxlen=self._atr_period + 2),
            highs=deque(maxlen=self._atr_period + 2),
            lows=deque(maxlen=self._atr_period + 2),
        ))

        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        volume = bar.volume

        state.total_bars += 1

        # ── Update ATR buffers (always, across days for multi-day context) ────
        state.closes.append(close)
        state.highs.append(high)
        state.lows.append(low)
        if len(state.closes) >= 2:
            state.atr = self._compute_atr(state)

        # ── Volume history (always — builds the 20-bar rolling average) ───────
        state.volume_history.append(volume)

        # ── Build opening range ───────────────────────────────────────────────
        # P1-FIX-16: Skip ORB if the day started too late to capture the opening window
        if not state.range_eligible:
            return None

        if not state.range_ready:
            state.bars_in_opening += 1

            if state.opening_high is None:
                state.opening_high = high
                state.opening_low = low
            else:
                state.opening_high = max(state.opening_high, high)
                state.opening_low = min(state.opening_low, low)

            if state.bars_in_opening >= self._orb_bars:
                state.range_ready = True
                log.debug(
                    "orb.range_built",
                    symbol=symbol,
                    orb_high=round(state.opening_high, 2),
                    orb_low=round(state.opening_low, 2),
                    bars=state.bars_in_opening,
                )
            return None

        # ── Already fired a signal today ──────────────────────────────────────
        if state.signal_fired:
            return None

        # ── Time gates ────────────────────────────────────────────────────────
        if bar_time >= self._cutoff_time:
            return None
        if bar_time >= _NO_NEW_TRADES_TIME:
            return None

        # ── Minimum range size guard (avoid flat open noise) ─────────────────
        orb_high = state.opening_high
        orb_low = state.opening_low
        range_size = orb_high - orb_low
        mid_price = (orb_high + orb_low) / 2.0
        if mid_price > 0 and (range_size / mid_price) < self._min_range_fraction:
            log.debug(
                "orb.range_too_small",
                symbol=symbol,
                range_size=round(range_size, 2),
                mid_price=round(mid_price, 2),
                min_fraction=self._min_range_fraction,
            )
            return None

        # ── ATR must be available ─────────────────────────────────────────────
        if state.atr is None or state.atr <= 0:
            return None

        # ── Volume confirmation ───────────────────────────────────────────────
        vol_history = list(state.volume_history)
        # Use the last 20 bars (excluding current bar for the average)
        # The current bar is already appended above, so the average is over all entries.
        # If we have fewer than 2 bars there is no average yet — skip check.
        if len(vol_history) >= 2:
            # Compute average over all but the most recent (current) bar
            avg_volume = sum(vol_history[:-1]) / len(vol_history[:-1])
            volume_ok = volume >= avg_volume * self._volume_multiplier
        else:
            # Not enough history for confirmation — let bar through
            volume_ok = True
            avg_volume = float(volume)

        # ── Breakout detection ────────────────────────────────────────────────
        signal_direction: Optional[int] = None

        if close > orb_high:
            signal_direction = +1
        elif close < orb_low:
            signal_direction = -1

        if signal_direction is None:
            return None

        # ── Volume gate ───────────────────────────────────────────────────────
        if not volume_ok:
            log.debug(
                "orb.volume_filter",
                symbol=symbol,
                direction=signal_direction,
                volume=volume,
                avg_volume=round(avg_volume, 0),
                required=round(avg_volume * self._volume_multiplier, 0),
            )
            return None

        # ── Mark signal fired ─────────────────────────────────────────────────
        state.signal_fired = True

        # ── Construct TradeSignal ─────────────────────────────────────────────
        entry_price = bar.close
        atr_dec = Decimal(str(round(state.atr, 2)))

        # Stop-loss: below opening low for BUY, above opening high for SELL
        if signal_direction > 0:
            stop_loss = Decimal(str(round(orb_low, 2)))
        else:
            stop_loss = Decimal(str(round(orb_high, 2)))

        # Confidence: (close - opening_high) / ATR for BUY, normalised to [0, 1]
        if signal_direction > 0:
            raw_confidence = (close - orb_high) / state.atr
        else:
            raw_confidence = (orb_low - close) / state.atr
        confidence = float(max(0.0, min(1.0, raw_confidence)))

        direction_label = "BULL_BREAKOUT" if signal_direction > 0 else "BEAR_BREAKOUT"
        log.info(
            "orb.signal",
            symbol=symbol,
            direction=direction_label,
            close=str(bar.close),
            orb_high=round(orb_high, 2),
            orb_low=round(orb_low, 2),
            atr=str(atr_dec),
            stop_loss=str(stop_loss),
            confidence=round(confidence, 3),
            volume=volume,
            avg_volume=round(avg_volume, 0),
        )

        return TradeSignal(
            symbol=symbol,
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            atr=atr_dec,
            confidence=round(confidence, 3),
            timestamp=bar.timestamp,
        )

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a symbol. Call at market open for forced reset."""
        self._states.pop(symbol, None)
        self._current_dates.pop(symbol, None)

    def reset_all(self) -> None:
        """Reset all symbol states."""
        self._states.clear()
        self._current_dates.clear()

    def get_state(self, symbol: str) -> Optional[ORBState]:
        return self._states.get(symbol)

    def is_range_ready(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        return state is not None and state.range_ready

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_atr(self, state: ORBState) -> float:
        """
        ATR via Wilder's smoothing (identical algorithm to EMACrossStrategy).
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

        # Seed with simple mean, then apply Wilder's smoothing
        if state.atr is None or state.atr <= 0:
            return sum(true_ranges) / len(true_ranges)

        n = self._atr_period
        return (state.atr * (n - 1) + true_ranges[-1]) / n
