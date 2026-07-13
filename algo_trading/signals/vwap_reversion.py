"""
signals/vwap_reversion.py — VWAP reversion after volume shock for NSE intraday.

After a sudden volume spike (> volume_spike_multiplier × 20-bar average), price
is frequently extended relative to VWAP.  Institutional activity often reverts
price back to VWAP within 2-4 bars as the imbalance is absorbed.

Signal logic:
  - volume > volume_spike_multiplier × 20-bar avg volume  (volume shock detected)
  - |close − VWAP| > atr_distance × ATR(14)              (price extended from VWAP)
  - close > VWAP  → signal = -1  (sell: fade the spike up)
  - close < VWAP  → signal = +1  (buy:  fade the spike down)

VWAP resets at session open (09:15 IST) each day.  It is computed as:
  VWAP(t) = Σ(typical_price × volume) / Σ(volume)
  typical_price = (high + low + close) / 3

Stop-loss is placed at 1.5 × ATR from entry (ASSUMED: tighter fade stop).

Design intent
-------------
NSE 15-min data frequently shows transient volume shocks during institutional
block trades, index rebalancing, and ETF creation/redemption.  These shocks
push price away from the session VWAP but rarely sustain.  The strategy fades
the extended move and exits at VWAP reversion or after hold_bars.

CRITICAL ASSUMPTIONS (calibrate on 12+ months NSE 15-min data before live):
  - volume_spike_multiplier = 3.0   (ASSUMED — >3× average = spike)
  - atr_distance            = 1.0   (ASSUMED — price extended > 1 ATR from VWAP)
  - hold_bars               = 4     (ASSUMED — expect reversion within 4 bars)
  - atr_period              = 14    (standard)
  - volume_lookback         = 20    (ASSUMED)
  - stop_atr_multiplier     = 1.5   (ASSUMED)

References
----------
Katz, J. & McCormick, D. (2000) "The Encyclopedia of Trading Strategies." McGraw-Hill.
Lerman, G. (2008) "VWAP Trading Strategies." Institutional Investor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE 15-min data) ─────────────────────────
_VOLUME_SPIKE_MULTIPLIER: float = 3.0   # ASSUMED
_ATR_DISTANCE:            float = 1.0   # ASSUMED: >1 ATR from VWAP
_HOLD_BARS:               int   = 4     # ASSUMED: expect reversion within 4 bars
_ATR_PERIOD:              int   = 14    # standard
_VOLUME_LOOKBACK:         int   = 20    # ASSUMED
_STOP_ATR_MULTIPLIER:     float = 1.5   # ASSUMED


@dataclass
class VWAPReversionSignal:
    """Output from the VWAP reversion strategy for one bar.

    Attributes:
        signal: +1 = BUY (fade down-spike), -1 = SELL (fade up-spike), 0 = hold.
        stop_loss: Absolute stop price.
        vwap: Current session VWAP at time of signal.
        confidence: Normalised [0.0, 1.0]; driven by ATR distance from VWAP.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal: int
    stop_loss: Decimal
    vwap: float
    confidence: float
    metadata: dict


@dataclass
class _SymbolState:
    """Mutable per-symbol session state (reset each day)."""

    # VWAP accumulators (reset at session open)
    cum_tp_vol: float = 0.0   # Σ(typical_price × volume)
    cum_vol: float = 0.0      # Σ(volume)

    # ATR (Wilder smoothing)
    prev_close: Optional[float] = None
    smooth_atr: Optional[float] = None
    atr_seed_trs: list = field(default_factory=list)
    atr_seeded: bool = False

    # Volume history
    volumes: deque = field(default_factory=lambda: deque(maxlen=_VOLUME_LOOKBACK))

    # Position management
    bars_held: int = 0
    in_position: bool = False
    entry_direction: int = 0


class VWAPReversionStrategy:
    """VWAP reversion after volume shock strategy for NSE 15-min intraday.

    Call ``reset_session(symbol)`` at 09:15 IST each day to reset VWAP
    accumulators.

    Usage::

        strategy = VWAPReversionStrategy()
        strategy.reset_session("RELIANCE")   # called at session open
        signal = strategy.on_bar("RELIANCE", o, h, l, c, v)

    Parameters
    ----------
    volume_spike_multiplier : float
        Volume threshold relative to 20-bar average. ASSUMED: 3.0.
    atr_distance : float
        Minimum ATR units price must be from VWAP to trigger signal. ASSUMED: 1.0.
    hold_bars : int
        Maximum bars before position is exited. ASSUMED: 4.
    stop_atr_multiplier : float
        Stop-loss placement in ATR units. ASSUMED: 1.5.
    """

    def __init__(
        self,
        volume_spike_multiplier: float = _VOLUME_SPIKE_MULTIPLIER,
        atr_distance: float = _ATR_DISTANCE,
        hold_bars: int = _HOLD_BARS,
        stop_atr_multiplier: float = _STOP_ATR_MULTIPLIER,
    ) -> None:
        self._volume_spike_multiplier = volume_spike_multiplier
        self._atr_distance = atr_distance
        self._hold_bars = hold_bars
        self._stop_atr_multiplier = stop_atr_multiplier

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
    ) -> VWAPReversionSignal:
        """Process one completed bar and return a VWAP reversion signal.

        Args:
            symbol: NSE instrument symbol.
            open_: Bar open price.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.
            volume: Bar volume in shares.

        Returns:
            VWAPReversionSignal with signal, stop_loss, vwap, confidence, metadata.
        """
        state = self._ensure_state(symbol)
        self._update_atr(state, high, low, close)

        # Update VWAP accumulators
        typical_price = (high + low + close) / 3.0
        state.cum_tp_vol += typical_price * volume
        state.cum_vol += volume
        vwap = state.cum_tp_vol / state.cum_vol if state.cum_vol > 0 else close

        state.volumes.append(volume)

        atr = state.smooth_atr or 1.0

        # Manage existing position
        if state.in_position:
            state.bars_held += 1
            # Exit at VWAP touch or max hold
            at_vwap = abs(close - vwap) < 0.1 * atr
            if at_vwap or state.bars_held >= self._hold_bars:
                state.in_position = False
                state.bars_held = 0
                log.info(
                    "vwap_reversion.exit",
                    symbol=symbol,
                    reason="vwap_touch" if at_vwap else "hold_expired",
                )
            stop = _make_stop(state.entry_direction, close, atr, self._stop_atr_multiplier)
            return VWAPReversionSignal(signal=0, stop_loss=stop, vwap=round(vwap, 2), confidence=0.0, metadata={})

        if len(state.volumes) < _VOLUME_LOOKBACK or state.smooth_atr is None:
            stop = Decimal(str(round(close - atr, 2)))
            return VWAPReversionSignal(signal=0, stop_loss=stop, vwap=round(vwap, 2), confidence=0.0, metadata={
                "reason": "warming_up"
            })

        avg_volume = sum(state.volumes) / len(state.volumes)

        # ── Gate 1: Volume spike ───────────────────────────────────────────────
        if volume < self._volume_spike_multiplier * avg_volume:
            stop = Decimal(str(round(close - atr, 2)))
            return VWAPReversionSignal(signal=0, stop_loss=stop, vwap=round(vwap, 2), confidence=0.0, metadata={
                "reason": "no_volume_spike",
                "volume_ratio": round(volume / avg_volume, 2),
            })

        # ── Gate 2: Price extended from VWAP ──────────────────────────────────
        price_diff = close - vwap
        atr_units_from_vwap = abs(price_diff) / atr if atr > 0 else 0.0

        if atr_units_from_vwap < self._atr_distance:
            stop = Decimal(str(round(close - atr, 2)))
            return VWAPReversionSignal(signal=0, stop_loss=stop, vwap=round(vwap, 2), confidence=0.0, metadata={
                "reason": "price_near_vwap",
                "atr_units_from_vwap": round(atr_units_from_vwap, 2),
            })

        # ── Signal: fade the spike ─────────────────────────────────────────────
        direction = -1 if price_diff > 0 else +1

        state.in_position = True
        state.bars_held = 0
        state.entry_direction = direction

        stop_price = _make_stop(direction, close, atr, self._stop_atr_multiplier)
        confidence = round(min((atr_units_from_vwap - self._atr_distance) / self._atr_distance, 1.0), 3)

        log.info(
            "vwap_reversion.signal",
            symbol=symbol,
            direction="BUY" if direction > 0 else "SELL",
            close=close,
            vwap=round(vwap, 2),
            atr_units_from_vwap=round(atr_units_from_vwap, 2),
            volume_ratio=round(volume / avg_volume, 2),
            stop=str(stop_price),
            confidence=confidence,
        )
        return VWAPReversionSignal(
            signal=direction,
            stop_loss=stop_price,
            vwap=round(vwap, 2),
            confidence=confidence,
            metadata={
                "atr_units_from_vwap": round(atr_units_from_vwap, 2),
                "volume_ratio": round(volume / avg_volume, 2),
                "atr": round(atr, 2),
            },
        )

    def reset_session(self, symbol: str) -> None:
        """Reset VWAP accumulators for a new trading session (call at 09:15 IST).

        Args:
            symbol: NSE instrument symbol.
        """
        state = self._ensure_state(symbol)
        state.cum_tp_vol = 0.0
        state.cum_vol = 0.0
        state.bars_held = 0
        state.in_position = False
        state.entry_direction = 0
        log.debug("vwap_reversion.session_reset", symbol=symbol)

    def reset_symbol(self, symbol: str) -> None:
        """Fully clear per-symbol state.

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


def _make_stop(direction: int, close: float, atr: float, multiplier: float) -> Decimal:
    """Compute stop-loss price.

    Args:
        direction: +1 = long (stop below entry), -1 = short (stop above entry).
        close: Entry price.
        atr: Current ATR value.
        multiplier: ATR multiplier for stop distance.

    Returns:
        Stop-loss price as Decimal.
    """
    if direction > 0:
        return Decimal(str(round(close - atr * multiplier, 2)))
    return Decimal(str(round(close + atr * multiplier, 2)))
