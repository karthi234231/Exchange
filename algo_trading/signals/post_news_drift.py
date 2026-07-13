"""
signals/post_news_drift.py — Post-earnings/news drift for NSE equities.

After a major positive announcement (high sentiment score), stocks tend to
drift UP for 3-5 days as institutional buyers accumulate. The signal exploits
this under-reaction by entering when price confirms the drift (breaks above
prior day high) on above-average volume.

Entry conditions (ALL must be true):
  1. news_sentiment > sentiment_threshold  (injected from news pipeline)
  2. close > yesterday's high              (price confirms upward drift)
  3. volume > volume_multiplier × 20-bar rolling mean

Exit:
  - Hold for up to hold_bars bars (3 trading days × 26 bars/day = 78 bars)
  - OR ATR-based stop triggers first

Design intent
-------------
NSE stocks often gap up on announcement day but the full drift takes 3-5 days
as mutual funds and FIIs build positions. The volume confirmation filter
reduces false positives from low-liquidity continuation moves.

CRITICAL ASSUMPTIONS (calibrate on 12+ months NSE 15-min data before live):
  - sentiment_threshold  = 0.7  (ASSUMED — source dependent on news provider)
  - volume_multiplier    = 1.5  (ASSUMED)
  - hold_bars            = 78   (ASSUMED — 3 days × 26 bars per trading day)
  - atr_period           = 14   (standard Wilder ATR)
  - atr_multiplier       = 2.0  (ASSUMED — wider stop for multi-day hold)
  - volume_lookback      = 20   (ASSUMED)

References
----------
Chan, E. (2013) "Algorithmic Trading." Wiley.
Bernard, V. & Thomas, J. (1989) "Post-Earnings-Announcement Drift." JAE.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import now_ist

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE 15-min data) ─────────────────────────
_SENTIMENT_THRESHOLD:  float = 0.7    # ASSUMED
_VOLUME_MULTIPLIER:    float = 1.5    # ASSUMED
_HOLD_BARS:            int   = 78     # ASSUMED: 3 days × 26 bars
_ATR_PERIOD:           int   = 14     # standard
_ATR_MULTIPLIER:       float = 2.0    # ASSUMED: wider stop for multi-day
_VOLUME_LOOKBACK:      int   = 20     # ASSUMED


@dataclass
class PostNewsDriftSignal:
    """Output from the post-news drift strategy for one bar.

    Attributes:
        signal: +1 = BUY (enter drift long), 0 = hold/flat.
        stop_loss: Absolute price level for the hard stop.
        confidence: Normalised [0.0, 1.0]; driven by sentiment score strength.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal: int
    stop_loss: Decimal
    confidence: float
    metadata: dict


@dataclass
class _SymbolState:
    """Mutable per-symbol state (not exposed externally)."""

    volumes: deque = field(default_factory=lambda: deque(maxlen=_VOLUME_LOOKBACK))
    highs: deque = field(default_factory=lambda: deque(maxlen=2))   # current + prev bar
    # ATR (Wilder smoothing)
    prev_close: Optional[float] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    smooth_atr: Optional[float] = None
    atr_seed_trs: list = field(default_factory=list)
    atr_seeded: bool = False
    # Drift state
    bars_held: int = 0
    in_position: bool = False


class PostNewsDriftStrategy:
    """Post-earnings/news drift strategy for NSE equities.

    Feed bars one at a time.  Before calling ``on_bar`` for a bar, set the
    ``news_sentiment`` attribute to the current sentiment score from your
    news pipeline (or leave as 0.0 if no news event detected).

    Usage::

        strategy = PostNewsDriftStrategy()
        strategy.news_sentiment["RELIANCE"] = 0.85   # injected from news pipeline
        signal = strategy.on_bar("RELIANCE", o, h, l, c, v)

    Parameters
    ----------
    sentiment_threshold : float
        Minimum news sentiment score to consider. ASSUMED: 0.7.
    volume_multiplier : float
        Volume must exceed this multiple of 20-bar mean. ASSUMED: 1.5.
    hold_bars : int
        Maximum bars to hold a drift position. ASSUMED: 78 (3 days).
    atr_multiplier : float
        Stop distance in ATR units. ASSUMED: 2.0.
    """

    def __init__(
        self,
        sentiment_threshold: float = _SENTIMENT_THRESHOLD,
        volume_multiplier: float = _VOLUME_MULTIPLIER,
        hold_bars: int = _HOLD_BARS,
        atr_multiplier: float = _ATR_MULTIPLIER,
    ) -> None:
        self._sentiment_threshold = sentiment_threshold
        self._volume_multiplier = volume_multiplier
        self._hold_bars = hold_bars
        self._atr_multiplier = atr_multiplier

        # External injection point: set before calling on_bar
        self.news_sentiment: dict[str, float] = {}

        self._states: dict[str, _SymbolState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def on_bar(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> PostNewsDriftSignal:
        """Process one completed bar and return a signal.

        Args:
            symbol: NSE instrument symbol.
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume in shares.

        Returns:
            PostNewsDriftSignal with signal, stop_loss, confidence, metadata.
        """
        state = self._ensure_state(symbol)
        self._update_atr(state, high, low, close)
        state.volumes.append(volume)
        # Track rolling high for prior-day comparison (deque(maxlen=2) keeps prev+current)
        state.highs.append(high)

        atr = state.smooth_atr or 1.0
        sentiment = self.news_sentiment.get(symbol, 0.0)

        # Decrement hold counter if in position
        if state.in_position:
            state.bars_held += 1
            if state.bars_held >= self._hold_bars:
                state.in_position = False
                state.bars_held = 0
                log.info(
                    "post_news_drift.hold_expired",
                    symbol=symbol,
                    hold_bars=self._hold_bars,
                )
            # No new entry while already in position
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return PostNewsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={})

        # Not enough history yet
        if (
            len(state.volumes) < _VOLUME_LOOKBACK
            or len(state.highs) < 2
            or state.smooth_atr is None
        ):
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return PostNewsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "warming_up"
            })

        avg_volume = sum(state.volumes) / len(state.volumes)
        prev_high = state.highs[-2]

        # ── Gate 1: News sentiment ─────────────────────────────────────────────
        if sentiment <= self._sentiment_threshold:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return PostNewsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "low_sentiment",
                "sentiment": sentiment,
            })

        # ── Gate 2: Price breaks above yesterday's high ────────────────────────
        if close <= prev_high:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return PostNewsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "no_price_breakout",
                "close": close,
                "prev_high": prev_high,
            })

        # ── Gate 3: Volume confirmation ────────────────────────────────────────
        if volume < self._volume_multiplier * avg_volume:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return PostNewsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "low_volume",
                "volume": volume,
                "avg_volume": round(avg_volume, 0),
                "threshold": round(self._volume_multiplier * avg_volume, 0),
            })

        # ── All gates passed — generate BUY signal ─────────────────────────────
        state.in_position = True
        state.bars_held = 0

        stop_price = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
        # Confidence: scaled by how much sentiment exceeds threshold
        confidence = round(min((sentiment - self._sentiment_threshold) / (1.0 - self._sentiment_threshold), 1.0), 3)

        log.info(
            "post_news_drift.signal",
            symbol=symbol,
            direction="BUY",
            sentiment=round(sentiment, 3),
            close=close,
            prev_high=round(prev_high, 2),
            volume_ratio=round(volume / avg_volume, 2),
            atr=round(atr, 2),
            stop=str(stop_price),
            confidence=confidence,
        )
        return PostNewsDriftSignal(
            signal=+1,
            stop_loss=stop_price,
            confidence=confidence,
            metadata={
                "sentiment": round(sentiment, 3),
                "prev_high": round(prev_high, 2),
                "volume_ratio": round(volume / avg_volume, 2),
                "atr": round(atr, 2),
            },
        )

    def reset_symbol(self, symbol: str) -> None:
        """Reset per-symbol state (call at start of each trading day).

        Args:
            symbol: NSE instrument symbol to reset.
        """
        self._states.pop(symbol, None)
        self.news_sentiment.pop(symbol, None)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    @staticmethod
    def _update_atr(state: _SymbolState, high: float, low: float, close: float) -> None:
        """Update Wilder ATR in-place."""
        if state.prev_close is not None:
            tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
        else:
            tr = high - low

        state.prev_close = close
        state.prev_high = high
        state.prev_low = low

        if not state.atr_seeded:
            state.atr_seed_trs.append(tr)
            if len(state.atr_seed_trs) == _ATR_PERIOD:
                state.smooth_atr = sum(state.atr_seed_trs) / _ATR_PERIOD
                state.atr_seeded = True
        else:
            assert state.smooth_atr is not None
            state.smooth_atr = (state.smooth_atr * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD
