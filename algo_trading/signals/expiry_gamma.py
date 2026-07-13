"""
signals/expiry_gamma.py — Expiry-day gamma pin / breakout signal for NSE F&O.

On NSE F&O expiry day (last Thursday of the month), market makers and dealers
carry large gamma exposure near the max-pain strike.  Gamma creates a magnetic
effect: the underlying index/stock is often "pinned" near max-pain as dealers
continuously delta-hedge, suppressing directional moves.

However, if the price breaks away from max-pain by more than breakout_pct,
the dealers must aggressively re-hedge in the same direction, amplifying the
move.  This creates a strong directional signal.

Signal logic:
  - is_expiry_day AND |close − max_pain| / max_pain < pin_zone_pct
    → signal = 0  (pin zone: no new entries, suppress existing signals)
  - is_expiry_day AND (close − max_pain) / max_pain > +breakout_pct
    → signal = +1 (bullish breakout from max-pain)
  - is_expiry_day AND (close − max_pain) / max_pain < −breakout_pct
    → signal = -1 (bearish breakout from max-pain)
  - Not expiry day
    → signal = 0  (this strategy only acts on expiry day)

Inputs injected externally:
  - is_expiry_day: bool — True when today is F&O expiry (last Thursday)
  - max_pain_strike: float — strike with maximum option open-interest decay

Design intent
-------------
Max-pain theory (Gujral, 2001) states that expiry tends to gravitate toward
the strike at which aggregate option holder losses are maximised.  The NSE
F&O market is large enough (₹2T+ daily notional) that pin risk is a real,
observable phenomenon in NIFTY and BANKNIFTY.

CRITICAL ASSUMPTIONS (calibrate on NSE weekly and monthly expiry data):
  - pin_zone_pct     = 0.5   (ASSUMED — ±0.5% of max-pain = pin zone)
  - breakout_pct     = 1.0   (ASSUMED — >1.0% move = directional breakout)

References
----------
Gujral, A. (2001) "Option Pain Theory."
Cohen, G. (2005) "The Bible of Options Strategies." FT Press.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE expiry data) ─────────────────────────
_PIN_ZONE_PCT:   float = 0.5    # ASSUMED: ±0.5% of max-pain = magnetic zone
_BREAKOUT_PCT:   float = 1.0    # ASSUMED: >1% away from max-pain = breakout


@dataclass
class ExpiryGammaSignal:
    """Output from the expiry gamma strategy for one bar.

    Attributes:
        signal: +1 = bullish breakout, -1 = bearish breakout, 0 = pin/no-trade.
        in_pin_zone: True when price is within pin_zone_pct of max_pain.
        distance_pct: Signed % distance of close from max_pain_strike.
        confidence: Normalised [0.0, 1.0]; driven by distance beyond breakout_pct.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal: int
    in_pin_zone: bool
    distance_pct: float
    confidence: float
    metadata: dict


class ExpiryGammaStrategy:
    """Expiry-day gamma pin / breakout strategy for NSE F&O.

    Inject ``is_expiry_day`` and ``max_pain_strike`` before calling ``on_bar``.
    On non-expiry days the strategy returns signal=0 immediately.

    Usage::

        strategy = ExpiryGammaStrategy()
        strategy.is_expiry_day["NIFTY25JUNFUT"] = True   # Thursday expiry
        strategy.max_pain_strike["NIFTY25JUNFUT"] = 22500.0
        signal = strategy.on_bar("NIFTY25JUNFUT", o, h, l, c, v)

    Parameters
    ----------
    pin_zone_pct : float
        Half-width of the no-trade pin zone as % of max_pain_strike. ASSUMED: 0.5.
    breakout_pct : float
        Minimum move beyond max-pain to trigger a directional signal. ASSUMED: 1.0.
    """

    def __init__(
        self,
        pin_zone_pct: float = _PIN_ZONE_PCT,
        breakout_pct: float = _BREAKOUT_PCT,
    ) -> None:
        self._pin_zone_pct = pin_zone_pct
        self._breakout_pct = breakout_pct

        # External injection points
        self.is_expiry_day: dict[str, bool] = {}
        self.max_pain_strike: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def on_bar(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> ExpiryGammaSignal:
        """Process one completed bar and return an expiry-gamma signal.

        Args:
            symbol: NSE instrument symbol (index future or equity future).
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume.

        Returns:
            ExpiryGammaSignal with signal, in_pin_zone, distance_pct, confidence, metadata.
        """
        is_expiry = self.is_expiry_day.get(symbol, False)

        # Not expiry day — strategy is inactive
        if not is_expiry:
            return ExpiryGammaSignal(
                signal=0,
                in_pin_zone=False,
                distance_pct=0.0,
                confidence=0.0,
                metadata={"reason": "not_expiry_day"},
            )

        max_pain = self.max_pain_strike.get(symbol)
        if max_pain is None or max_pain <= 0:
            log.warning(
                "expiry_gamma.no_max_pain",
                symbol=symbol,
            )
            return ExpiryGammaSignal(
                signal=0,
                in_pin_zone=False,
                distance_pct=0.0,
                confidence=0.0,
                metadata={"reason": "no_max_pain_strike"},
            )

        distance_pct = (close - max_pain) / max_pain * 100.0
        abs_distance_pct = abs(distance_pct)

        # ── Pin zone: suppress all entries ────────────────────────────────────
        if abs_distance_pct < self._pin_zone_pct:
            log.debug(
                "expiry_gamma.pin_zone",
                symbol=symbol,
                close=close,
                max_pain=max_pain,
                distance_pct=round(distance_pct, 3),
            )
            return ExpiryGammaSignal(
                signal=0,
                in_pin_zone=True,
                distance_pct=round(distance_pct, 3),
                confidence=0.0,
                metadata={
                    "max_pain": max_pain,
                    "distance_pct": round(distance_pct, 3),
                    "reason": "pin_zone",
                },
            )

        # ── Breakout zone: directional signal ─────────────────────────────────
        if abs_distance_pct >= self._breakout_pct:
            direction = +1 if distance_pct > 0 else -1
            # Confidence: how far beyond the breakout threshold (capped at 1.0)
            excess = abs_distance_pct - self._breakout_pct
            confidence = round(min(excess / self._breakout_pct, 1.0), 3)

            log.info(
                "expiry_gamma.signal",
                symbol=symbol,
                direction="BUY" if direction > 0 else "SELL",
                close=close,
                max_pain=max_pain,
                distance_pct=round(distance_pct, 3),
                confidence=confidence,
            )
            return ExpiryGammaSignal(
                signal=direction,
                in_pin_zone=False,
                distance_pct=round(distance_pct, 3),
                confidence=confidence,
                metadata={
                    "max_pain": max_pain,
                    "distance_pct": round(distance_pct, 3),
                    "breakout_pct": self._breakout_pct,
                },
            )

        # Between pin zone and breakout zone — wait
        return ExpiryGammaSignal(
            signal=0,
            in_pin_zone=False,
            distance_pct=round(distance_pct, 3),
            confidence=0.0,
            metadata={
                "reason": "transition_zone",
                "max_pain": max_pain,
                "distance_pct": round(distance_pct, 3),
            },
        )

    def reset_symbol(self, symbol: str) -> None:
        """Clear external state for a symbol.

        Args:
            symbol: NSE instrument symbol to reset.
        """
        self.is_expiry_day.pop(symbol, None)
        self.max_pain_strike.pop(symbol, None)
