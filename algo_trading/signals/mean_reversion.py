"""
signals/mean_reversion.py — RSI mean-reversion signal for sideways NSE markets.

Fires when the market is in a sideways/choppy regime (ADX < threshold) and
price has stretched to an RSI extreme.  The signal fades the move:

  RSI > rsi_overbought AND ADX < adx_max → SELL (fade the overbought push)
  RSI < rsi_oversold  AND ADX < adx_max → BUY  (fade the oversold dip)

Design intent
-------------
The EMA cross strategy (ema_cross_filtered.py) is blocked when ADX < 25 —
which describes roughly 60% of NSE intraday time.  During those sideways hours
price tends to oscillate within a range.  Mean reversion exploits that
oscillation by fading RSI extremes.

These two strategies are complementary:
  - ADX ≥ 25 → EMA cross (trend-following)
  - ADX <  20 → Mean reversion (RSI fade)
  - 20 ≤ ADX < 25 → dead zone, neither strategy trades

ADX computation is reused from ema_cross_filtered.ADXComputer (same Wilder
smoothing, same period 14 — avoiding duplicated state).

RSI computation
---------------
RSI(14) using Wilder's smoothing (same approach as standard RSI):
  avg_gain(t) = (avg_gain(t-1) × 13 + gain(t)) / 14
  avg_loss(t) = (avg_loss(t-1) × 13 + loss(t)) / 14
  RS = avg_gain / avg_loss
  RSI = 100 − 100 / (1 + RS)

Stop-loss
---------
Mean-reversion trades use a tighter ATR-based stop than trend trades:
  stop = entry ± atr_multiplier × ATR(14)
The ATR is taken directly from the ADXComputer's smoothed_atr state.

CRITICAL ASSUMPTIONS (must calibrate on real NSE data before live trading):
  - rsi_overbought = 72  (ASSUMED)
  - rsi_oversold   = 28  (ASSUMED)
  - adx_max        = 20  (ASSUMED — dead zone starts above this)
  - atr_multiplier = 1.0 (ASSUMED — tighter than trend's 1.5)
  - RSI period     = 14  (standard)
  - ADX period     = 14  (matches ema_cross_filtered for shared state)

References
----------
Connors, L. & Alvarez, C. (2009) "Short Term Trading Strategies That Work."
Wilder, J.W. (1978) "New Concepts in Technical Trading Systems."
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

from algo_trading.data.broker_base import Bar
from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import now_ist
from algo_trading.signals.ema_cross_filtered import ADXComputer, ADXState

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on 12+ months NSE 15-min data) ──────────────
_RSI_PERIOD:         int   = 14
_ADX_PERIOD:         int   = 14
_RSI_OVERBOUGHT:     float = 72.0   # ASSUMED: sell when RSI > 72
_RSI_OVERSOLD:       float = 28.0   # ASSUMED: buy  when RSI < 28
_ADX_MAX:            float = 20.0   # ASSUMED: only trade when ADX < 20 (sideways)
_ATR_MULTIPLIER:     float = 1.0    # ASSUMED: stop = entry ± 1.0 × ATR
_WARMUP_BARS:        int   = 28     # 14 (RSI seed) + 14 (ADX seed)


@dataclass
class RSIState:
    """Wilder-smoothed RSI state per symbol."""
    bar_count:   int   = 0
    prev_close:  Optional[float] = None

    # Seed phase: accumulate raw gains/losses for first 14 bars
    seed_gains:  list  = field(default_factory=list)
    seed_losses: list  = field(default_factory=list)
    seeded:      bool  = False

    # Wilder-smoothed average gain and loss
    avg_gain:    Optional[float] = None
    avg_loss:    Optional[float] = None

    # Current RSI value (None until warmed up)
    rsi:         Optional[float] = None


class RSIComputer:
    """
    Incremental RSI(14) using Wilder's smoothing.
    O(1) per bar after seed. No pandas dependency.
    """

    def __init__(self, period: int = _RSI_PERIOD) -> None:
        self._period = period

    def update(self, bar: Bar, state: RSIState) -> None:
        """Feed one bar and update RSI state in-place."""
        close = float(bar.close)
        state.bar_count += 1

        if state.prev_close is None:
            state.prev_close = close
            return

        change = close - state.prev_close
        gain   = max(change,  0.0)
        loss   = max(-change, 0.0)
        state.prev_close = close

        n = self._period

        if not state.seeded:
            state.seed_gains.append(gain)
            state.seed_losses.append(loss)

            if len(state.seed_gains) == n:
                state.avg_gain = sum(state.seed_gains) / n
                state.avg_loss = sum(state.seed_losses) / n
                state.seeded = True
                self._compute_rsi(state)
        else:
            # Wilder smoothing: avg = (prev × (n-1) + current) / n
            state.avg_gain = (state.avg_gain * (n - 1) + gain) / n
            state.avg_loss = (state.avg_loss * (n - 1) + loss) / n
            self._compute_rsi(state)

    @staticmethod
    def _compute_rsi(state: RSIState) -> None:
        if state.avg_loss is None or state.avg_gain is None:
            return
        # Both near zero → flat prices, RSI is neutral (50)
        if state.avg_gain < 1e-10 and state.avg_loss < 1e-10:
            state.rsi = 50.0
            return
        if state.avg_loss < 1e-10:
            state.rsi = 100.0
            return
        rs = state.avg_gain / state.avg_loss
        state.rsi = 100.0 - (100.0 / (1.0 + rs))


@dataclass
class MRSignalStats:
    """Per-symbol signal statistics for calibration."""
    total_checks:   int = 0
    adx_blocked:    int = 0   # ADX too high (trending)
    rsi_no_extreme: int = 0   # RSI in neutral zone
    signals_emitted: int = 0


class MeanReversionStrategy:
    """
    RSI mean-reversion strategy for sideways NSE markets.

    Use when ADX < adx_max (choppy conditions).
    Complements FilteredEMACrossStrategy which requires ADX ≥ 25.

    Usage
    -----
        strategy = MeanReversionStrategy()
        for bar in bars:
            signal = strategy.feed_bar(bar)
            if signal:
                executor.submit_signal(signal)

    Parameters
    ----------
    rsi_overbought : float
        RSI above which a SELL signal is generated. ASSUMED default: 72.
    rsi_oversold : float
        RSI below which a BUY signal is generated. ASSUMED default: 28.
    adx_max : float
        Only trade when ADX is BELOW this value (sideways regime).
        ASSUMED default: 20.
    atr_multiplier : float
        Stop-loss distance in ATR units. ASSUMED default: 1.0.
    """

    def __init__(
        self,
        rsi_overbought:  float = _RSI_OVERBOUGHT,
        rsi_oversold:    float = _RSI_OVERSOLD,
        adx_max:         float = _ADX_MAX,
        atr_multiplier:  float = _ATR_MULTIPLIER,
    ) -> None:
        self._rsi_overbought = rsi_overbought
        self._rsi_oversold   = rsi_oversold
        self._adx_max        = adx_max
        self._atr_multiplier = atr_multiplier

        # Per-symbol state
        self._rsi_computers: dict[str, RSIComputer] = {}
        self._rsi_states:    dict[str, RSIState]    = {}
        self._adx_computers: dict[str, ADXComputer] = {}
        self._adx_states:    dict[str, ADXState]    = {}
        self._stats:         dict[str, MRSignalStats] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def feed_bar(self, bar: Bar) -> Optional[TradeSignal]:
        """
        Feed one completed 15-min bar.

        Returns a TradeSignal when RSI extreme detected in sideways regime,
        else None.

        Parameters
        ----------
        bar : Bar
            Completed OHLCV bar for the traded symbol.

        Returns
        -------
        TradeSignal or None
            direction=+1 (buy oversold) or -1 (sell overbought), or None.
        """
        symbol = bar.symbol
        self._ensure_state(symbol)

        # Update ADX
        self._adx_computers[symbol].update(bar, self._adx_states[symbol])
        # Update RSI
        self._rsi_computers[symbol].update(bar, self._rsi_states[symbol])

        adx_state = self._adx_states[symbol]
        rsi_state = self._rsi_states[symbol]

        # Not yet warmed up
        if adx_state.adx is None or rsi_state.rsi is None:
            return None

        stats = self._stats[symbol]
        stats.total_checks += 1

        adx = adx_state.adx
        rsi = rsi_state.rsi

        # ── Gate 1: ADX must indicate sideways market ─────────────────────────
        if adx >= self._adx_max:
            stats.adx_blocked += 1
            log.debug(
                "mean_reversion.adx_blocked",
                symbol=symbol,
                adx=round(adx, 2),
                threshold=self._adx_max,
            )
            return None

        # ── Gate 2: RSI must be at an extreme ─────────────────────────────────
        if rsi > self._rsi_overbought:
            direction = -1   # sell the overbought
        elif rsi < self._rsi_oversold:
            direction = +1   # buy the oversold
        else:
            stats.rsi_no_extreme += 1
            return None

        # ── Build TradeSignal ─────────────────────────────────────────────────
        # ATR from the ADX computer's smoothed state (already computed)
        atr = self._get_atr(adx_state)
        entry_price = Decimal(str(round(float(bar.close), 2)))

        if direction == -1:
            stop_price = entry_price + Decimal(str(round(atr * self._atr_multiplier, 2)))
        else:
            stop_price = entry_price - Decimal(str(round(atr * self._atr_multiplier, 2)))

        # Confidence: inverse of RSI deviation normalised [0.5, 1.0]
        # Further from neutral (50) = more extreme = higher confidence
        rsi_deviation = abs(rsi - 50.0) / 50.0   # 0.44 at RSI=72, 1.0 at RSI=0 or 100
        confidence = round(0.5 + 0.5 * min(rsi_deviation, 1.0), 3)

        stats.signals_emitted += 1

        log.info(
            "mean_reversion.signal",
            symbol=symbol,
            direction="BUY" if direction > 0 else "SELL",
            rsi=round(rsi, 2),
            adx=round(adx, 2),
            entry=str(entry_price),
            stop=str(stop_price),
            confidence=confidence,
        )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            atr=Decimal(str(round(atr, 4))),
            confidence=confidence,
            timestamp=now_ist(),
        )

    def is_warmed_up(self, symbol: str) -> bool:
        """True when both RSI and ADX are warmed up for this symbol."""
        adx_state = self._adx_states.get(symbol)
        rsi_state = self._rsi_states.get(symbol)
        return (
            adx_state is not None and adx_state.adx is not None
            and rsi_state is not None and rsi_state.rsi is not None
        )

    def get_rsi(self, symbol: str) -> Optional[float]:
        """Return current RSI value, or None if not warmed up."""
        state = self._rsi_states.get(symbol)
        return state.rsi if state else None

    def get_adx(self, symbol: str) -> Optional[float]:
        """Return current ADX value, or None if not warmed up."""
        state = self._adx_states.get(symbol)
        return state.adx if state else None

    def get_stats(self) -> dict[str, dict]:
        """Return per-symbol signal statistics for calibration."""
        result: dict[str, dict] = {}
        for symbol, stats in self._stats.items():
            total = stats.total_checks
            result[symbol] = {
                "total_checks":    total,
                "adx_blocked":     stats.adx_blocked,
                "rsi_no_extreme":  stats.rsi_no_extreme,
                "signals_emitted": stats.signals_emitted,
                "adx_block_rate":  round(stats.adx_blocked / total, 4) if total else 0.0,
                "signal_rate":     round(stats.signals_emitted / total, 4) if total else 0.0,
            }
        return result

    def reset_symbol(self, symbol: str) -> None:
        """Reset all state for a symbol (call at start of each day)."""
        self._rsi_states.pop(symbol, None)
        self._rsi_computers.pop(symbol, None)
        self._adx_states.pop(symbol, None)
        self._adx_computers.pop(symbol, None)
        self._stats.pop(symbol, None)

    def reset_all(self) -> None:
        """Reset all symbol state."""
        self._rsi_states.clear()
        self._rsi_computers.clear()
        self._adx_states.clear()
        self._adx_computers.clear()
        self._stats.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_state(self, symbol: str) -> None:
        if symbol not in self._rsi_computers:
            self._rsi_computers[symbol] = RSIComputer(period=_RSI_PERIOD)
            self._rsi_states[symbol]    = RSIState()
        if symbol not in self._adx_computers:
            self._adx_computers[symbol] = ADXComputer(period=_ADX_PERIOD)
            self._adx_states[symbol]    = ADXState()
        if symbol not in self._stats:
            self._stats[symbol] = MRSignalStats()

    @staticmethod
    def _get_atr(adx_state: ADXState) -> float:
        """Extract ATR from the ADX state's Wilder-smoothed value.

        Falls back to a small non-zero value if ATR is not yet computed
        (during warmup) to avoid ZeroDivisionError in stop calculation.
        """
        if adx_state.smooth_atr is not None and adx_state.smooth_atr > 0:
            return adx_state.smooth_atr
        return 1.0   # fallback: 1 point stop — should never reach live trading
