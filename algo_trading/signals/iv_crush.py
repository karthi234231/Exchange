"""
signals/iv_crush.py — Implied volatility crush signal for NSE F&O.

After a scheduled event (earnings, RBI policy, budget), implied volatility
collapses as the uncertainty is resolved.  Selling a straddle/strangle just
before the event (while IV is high) and buying it back after (when IV has
collapsed) captures the IV crush premium.

Signal logic:
  - iv_rank > min_iv_rank  AND  days_to_event <= max_days_to_event
    → signal = +1 (sell straddle / short vol)
  - iv_rank < 30  (IV has already crushed post-event)
    → signal = -1 (buy straddle if already short, else flat)
  - Otherwise
    → signal = 0 (hold)

iv_rank = (current_iv − 52wk_low_iv) / (52wk_high_iv − 52wk_low_iv) × 100

Inputs injected externally:
  - iv: current implied volatility (annualised %, e.g. 35.0 = 35%)
  - days_to_event: calendar days until next known event (0 = event today)

Design intent
-------------
NSE index options (NIFTY, BANKNIFTY) have well-defined monthly and weekly
expiry events.  IV typically spikes 3-5 days before expiry/earnings and then
collapses sharply on the announcement day.  This signal is only appropriate
for traders with F&O capability and sufficient margin.

CRITICAL ASSUMPTIONS (calibrate on 12+ months NSE options data before live):
  - min_iv_rank          = 70    (ASSUMED — top 30% of IV range)
  - max_days_to_event    = 3     (ASSUMED — enter up to 3 days before event)
  - hold_bars            = 52    (ASSUMED — 2 days × 26 bars; hold until crush)
  - iv_history_period    = 252   (ASSUMED — 52wk in trading days for IV rank)

References
----------
Natenberg, S. (1994) "Option Volatility & Pricing." McGraw-Hill.
Sinclair, E. (2010) "Option Trading." Wiley.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE options data) ───────────────────────
_MIN_IV_RANK:        float = 70.0   # ASSUMED
_MAX_DAYS_TO_EVENT:  int   = 3      # ASSUMED
_HOLD_BARS:          int   = 52     # ASSUMED: 2 days × 26 bars
_IV_HISTORY_PERIOD:  int   = 252    # ASSUMED: 52-week history


@dataclass
class IVCrushSignal:
    """Output from the IV crush strategy for one bar.

    Attributes:
        signal: +1 = sell straddle (enter short-vol), -1 = buy straddle
            (exit/reverse), 0 = hold.
        iv_rank: Current IV rank [0, 100].
        confidence: Normalised [0.0, 1.0]; driven by how elevated IV rank is.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal: int
    iv_rank: float
    confidence: float
    metadata: dict


@dataclass
class _SymbolState:
    """Mutable per-symbol state."""

    iv_history: deque = field(default_factory=lambda: deque(maxlen=_IV_HISTORY_PERIOD))
    bars_held: int = 0
    in_position: bool = False


class IVCrushStrategy:
    """IV crush strategy for NSE F&O options.

    Before calling ``on_bar`` for each bar, set the ``iv`` and
    ``days_to_event`` attributes for the relevant symbol.

    Usage::

        strategy = IVCrushStrategy()
        strategy.iv["NIFTY25JUNFUT"] = 18.5        # injected from options chain
        strategy.days_to_event["NIFTY25JUNFUT"] = 2  # days to RBI policy
        signal = strategy.on_bar("NIFTY25JUNFUT", o, h, l, c, v)

    Parameters
    ----------
    min_iv_rank : float
        IV rank threshold above which to enter short-vol. ASSUMED: 70.
    max_days_to_event : int
        Only enter when event is within this many days. ASSUMED: 3.
    hold_bars : int
        Maximum bars to hold short-vol position. ASSUMED: 52 (2 days).
    """

    def __init__(
        self,
        min_iv_rank: float = _MIN_IV_RANK,
        max_days_to_event: int = _MAX_DAYS_TO_EVENT,
        hold_bars: int = _HOLD_BARS,
    ) -> None:
        self._min_iv_rank = min_iv_rank
        self._max_days_to_event = max_days_to_event
        self._hold_bars = hold_bars

        # External injection points
        self.iv: dict[str, float] = {}               # current IV per symbol
        self.days_to_event: dict[str, int] = {}      # days to next event

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
    ) -> IVCrushSignal:
        """Process one completed bar and return an IV crush signal.

        Args:
            symbol: NSE instrument symbol (typically an index or stock option root).
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume.

        Returns:
            IVCrushSignal with signal, iv_rank, confidence, metadata.
        """
        state = self._ensure_state(symbol)
        current_iv = self.iv.get(symbol, 0.0)
        days_to_event = self.days_to_event.get(symbol, 999)

        # Update IV history
        if current_iv > 0:
            state.iv_history.append(current_iv)

        iv_rank = self._compute_iv_rank(state, current_iv)

        # Manage existing position
        if state.in_position:
            state.bars_held += 1
            if state.bars_held >= self._hold_bars:
                state.in_position = False
                state.bars_held = 0
                log.info("iv_crush.hold_expired", symbol=symbol)
                return IVCrushSignal(signal=-1, iv_rank=iv_rank, confidence=0.0, metadata={
                    "reason": "hold_expired_exit"
                })
            return IVCrushSignal(signal=0, iv_rank=iv_rank, confidence=0.0, metadata={})

        # Need IV history to compute rank
        if len(state.iv_history) < 20 or current_iv <= 0:
            return IVCrushSignal(signal=0, iv_rank=0.0, confidence=0.0, metadata={
                "reason": "warming_up"
            })

        # ── Gate 1: High IV rank ───────────────────────────────────────────────
        if iv_rank < self._min_iv_rank:
            return IVCrushSignal(signal=0, iv_rank=iv_rank, confidence=0.0, metadata={
                "reason": "low_iv_rank",
                "iv_rank": round(iv_rank, 1),
            })

        # ── Gate 2: Event is imminent ──────────────────────────────────────────
        if days_to_event > self._max_days_to_event:
            return IVCrushSignal(signal=0, iv_rank=iv_rank, confidence=0.0, metadata={
                "reason": "event_too_far",
                "days_to_event": days_to_event,
            })

        # ── All gates passed — sell straddle ───────────────────────────────────
        state.in_position = True
        state.bars_held = 0

        confidence = round(min((iv_rank - self._min_iv_rank) / (100.0 - self._min_iv_rank), 1.0), 3)

        log.info(
            "iv_crush.signal",
            symbol=symbol,
            direction="SELL_STRADDLE",
            iv_rank=round(iv_rank, 1),
            current_iv=round(current_iv, 2),
            days_to_event=days_to_event,
            confidence=confidence,
        )
        return IVCrushSignal(
            signal=+1,
            iv_rank=iv_rank,
            confidence=confidence,
            metadata={
                "iv_rank": round(iv_rank, 1),
                "current_iv": round(current_iv, 2),
                "days_to_event": days_to_event,
            },
        )

    def reset_symbol(self, symbol: str) -> None:
        """Reset per-symbol state.

        Args:
            symbol: NSE instrument symbol to reset.
        """
        self._states.pop(symbol, None)
        self.iv.pop(symbol, None)
        self.days_to_event.pop(symbol, None)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    @staticmethod
    def _compute_iv_rank(state: _SymbolState, current_iv: float) -> float:
        """Compute IV rank = (current − 52wk_low) / (52wk_high − 52wk_low) × 100.

        Args:
            state: Symbol-level state holding IV history.
            current_iv: Current bar's implied volatility.

        Returns:
            IV rank in [0, 100], or 0.0 if history is insufficient.
        """
        if len(state.iv_history) < 2:
            return 0.0
        low_iv = min(state.iv_history)
        high_iv = max(state.iv_history)
        if high_iv - low_iv < 1e-8:
            return 50.0  # flat IV history → neutral
        return (current_iv - low_iv) / (high_iv - low_iv) * 100.0
