"""
signals/earnings_drift.py — Earnings surprise drift for NSE equities.

Stocks with an earnings beat (actual EPS > consensus estimate) tend to drift UP
for 5-10 days post-announcement as the market slowly re-prices the better
fundamentals.  The signal enters once price clears the pre-announcement high on
volume, confirming the drift rather than chasing the gap.

Entry conditions (ALL must be true):
  1. earnings_surprise_pct > min_surprise_pct  (injected externally)
  2. close > pre_announcement_high             (price confirms drift)
  3. volume > 1.0× 20-bar average             (some volume participation)

Exit:
  - Hold for up to hold_bars bars (5 trading days × 26 bars = 130 bars)
  - OR ATR-based stop triggers first

Design intent
-------------
Bernard & Thomas (1989) documented post-earnings-announcement drift (PEAD) on
US markets.  The same phenomenon is observed on NSE, especially for mid-cap
stocks where institutional coverage is thinner and information diffuses more
slowly.  The volume-on-breakout filter avoids entering on low-float gap days.

CRITICAL ASSUMPTIONS (calibrate on 12+ months NSE 15-min data before live):
  - min_surprise_pct   = 5.0   (ASSUMED — positive earnings beat percentage)
  - hold_bars          = 130   (ASSUMED — 5 days × 26 bars)
  - atr_period         = 14    (standard)
  - atr_multiplier     = 2.5   (ASSUMED — multi-day hold needs wider stop)
  - volume_lookback    = 20    (ASSUMED)

References
----------
Bernard, V. & Thomas, J. (1989) "Post-Earnings-Announcement Drift." JAE.
Chan, L., Jegadeesh, N. & Lakonishok, J. (1996) "Momentum Strategies." JF.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import now_ist

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE data) ────────────────────────────────
_MIN_SURPRISE_PCT:  float = 5.0    # ASSUMED: 5% earnings beat
_HOLD_BARS:         int   = 130    # ASSUMED: 5 days × 26 bars
_ATR_PERIOD:        int   = 14     # standard Wilder ATR
_ATR_MULTIPLIER:    float = 2.5    # ASSUMED: wider stop for 5-day hold
_VOLUME_LOOKBACK:   int   = 20     # ASSUMED


@dataclass
class EarningsDriftSignal:
    """Output from the earnings drift strategy for one bar.

    Attributes:
        signal: +1 = BUY (enter drift long), 0 = hold/flat.
        stop_loss: Absolute price for hard stop.
        confidence: Normalised [0.0, 1.0]; driven by surprise magnitude.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal: int
    stop_loss: Decimal
    confidence: float
    metadata: dict


@dataclass
class _SymbolState:
    """Mutable per-symbol state."""

    volumes: deque = field(default_factory=lambda: deque(maxlen=_VOLUME_LOOKBACK))
    prev_close: Optional[float] = None
    smooth_atr: Optional[float] = None
    atr_seed_trs: list = field(default_factory=list)
    atr_seeded: bool = False
    bars_held: int = 0
    in_position: bool = False


class EarningsDriftStrategy:
    """Earnings surprise drift strategy for NSE equities.

    Inject ``earnings_surprise_pct`` and ``pre_announcement_high`` before
    calling ``on_bar`` for each earnings-event symbol.

    Usage::

        strategy = EarningsDriftStrategy()
        strategy.earnings_surprise_pct["INFY"] = 8.5      # injected externally
        strategy.pre_announcement_high["INFY"] = 1425.50  # last close before result
        signal = strategy.on_bar("INFY", o, h, l, c, v)

    Parameters
    ----------
    min_surprise_pct : float
        Minimum positive earnings beat (%). ASSUMED: 5.0.
    hold_bars : int
        Maximum bars to hold the position. ASSUMED: 130 (5 days).
    atr_multiplier : float
        Stop distance in ATR units. ASSUMED: 2.5.
    """

    def __init__(
        self,
        min_surprise_pct: float = _MIN_SURPRISE_PCT,
        hold_bars: int = _HOLD_BARS,
        atr_multiplier: float = _ATR_MULTIPLIER,
    ) -> None:
        self._min_surprise_pct = min_surprise_pct
        self._hold_bars = hold_bars
        self._atr_multiplier = atr_multiplier

        # External injection points
        self.earnings_surprise_pct: dict[str, float] = {}
        self.pre_announcement_high: dict[str, float] = {}

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
    ) -> EarningsDriftSignal:
        """Process one completed bar and return a signal.

        Args:
            symbol: NSE instrument symbol.
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume in shares.

        Returns:
            EarningsDriftSignal with signal, stop_loss, confidence, metadata.
        """
        state = self._ensure_state(symbol)
        self._update_atr(state, high, low, close)
        state.volumes.append(volume)

        atr = state.smooth_atr or 1.0
        surprise = self.earnings_surprise_pct.get(symbol, 0.0)
        ann_high = self.pre_announcement_high.get(symbol, None)

        # Decrement hold counter
        if state.in_position:
            state.bars_held += 1
            if state.bars_held >= self._hold_bars:
                state.in_position = False
                state.bars_held = 0
                log.info("earnings_drift.hold_expired", symbol=symbol)
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return EarningsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={})

        if len(state.volumes) < _VOLUME_LOOKBACK or state.smooth_atr is None:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return EarningsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "warming_up"
            })

        avg_volume = sum(state.volumes) / len(state.volumes)

        # ── Gate 1: Earnings surprise ──────────────────────────────────────────
        if surprise <= self._min_surprise_pct:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return EarningsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "insufficient_surprise",
                "surprise_pct": surprise,
            })

        # ── Gate 2: Price above pre-announcement high ──────────────────────────
        if ann_high is None or close <= ann_high:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return EarningsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "below_pre_announcement_high",
                "close": close,
                "pre_ann_high": ann_high,
            })

        # ── Gate 3: Volume participation ───────────────────────────────────────
        if volume < avg_volume:
            stop = Decimal(str(round(close - atr * self._atr_multiplier, 2)))
            return EarningsDriftSignal(signal=0, stop_loss=stop, confidence=0.0, metadata={
                "reason": "low_volume",
                "volume_ratio": round(volume / avg_volume, 2),
            })

        # ── All gates passed — generate BUY signal ─────────────────────────────
        state.in_position = True
        state.bars_held = 0
        stop_price = Decimal(str(round(close - atr * self._atr_multiplier, 2)))

        # Confidence: scales with size of earnings beat beyond threshold
        confidence = round(min((surprise - self._min_surprise_pct) / 20.0, 1.0), 3)

        log.info(
            "earnings_drift.signal",
            symbol=symbol,
            direction="BUY",
            surprise_pct=round(surprise, 2),
            close=close,
            pre_ann_high=round(ann_high, 2),
            volume_ratio=round(volume / avg_volume, 2),
            stop=str(stop_price),
            confidence=confidence,
        )
        return EarningsDriftSignal(
            signal=+1,
            stop_loss=stop_price,
            confidence=confidence,
            metadata={
                "surprise_pct": round(surprise, 2),
                "pre_announcement_high": round(ann_high, 2),
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

        if not state.atr_seeded:
            state.atr_seed_trs.append(tr)
            if len(state.atr_seed_trs) == _ATR_PERIOD:
                state.smooth_atr = sum(state.atr_seed_trs) / _ATR_PERIOD
                state.atr_seeded = True
        else:
            assert state.smooth_atr is not None
            state.smooth_atr = (state.smooth_atr * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD
