"""
signals/stat_arb.py — Statistical arbitrage for cointegrated NSE stock pairs.

Two cointegrated NSE stocks (e.g., HDFCBANK/ICICIBANK, RELIANCE/IOC) maintain
a long-run spread relationship.  When the spread deviates by more than Z standard
deviations from its mean, it is expected to revert, generating a pairs trade.

Spread = log(price_A) − β × log(price_B)

The hedge ratio β is estimated dynamically using a Kalman filter, which allows
it to evolve slowly as the cointegration relationship shifts over time.

Signal logic:
  spread_zscore > +entry_zscore → SELL A / BUY B  (spread too wide)
  spread_zscore < −entry_zscore → BUY A / SELL B  (spread too narrow)
  |spread_zscore| < exit_zscore → EXIT position   (spread reverted to mean)

Kalman filter model:
  State:  β (hedge ratio, 1D)
  Obs:    log(price_A) = β × log(price_B) + ε
  R:      observation noise variance  (ASSUMED: 0.01)
  Q:      state transition variance   (ASSUMED: 1e-5)

Fallback: if filterpy is not installed, rolling OLS over the last 60 bars is
used as a simpler (non-adaptive) hedge ratio estimator.

Design intent
-------------
Pairs trading exploits temporary mispricing between fundamentally linked
stocks while remaining market-neutral (delta ≈ 0 at entry).  NSE bank pairs
(HDFCBANK / ICICIBANK) are commonly cited examples with high cointegration.

CRITICAL ASSUMPTIONS (calibrate on 12+ months NSE daily + 15-min data):
  - entry_zscore    = 2.0    (ASSUMED — enter at 2σ)
  - exit_zscore     = 0.5    (ASSUMED — exit when spread reverts to 0.5σ)
  - kalman_R        = 0.01   (ASSUMED — observation noise)
  - kalman_Q        = 1e-5   (ASSUMED — β drift rate)
  - spread_lookback = 60     (ASSUMED — rolling window for spread mean/std)

References
----------
Vidyamurthy, G. (2004) "Pairs Trading." Wiley.
Pole, A. (2007) "Statistical Arbitrage." Wiley.
Elliott, R., Van der Hoek, J. & Malcolm, W. (2005) "Pairs Trading." QF.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — calibrate on NSE data) ────────────────────────────────
_ENTRY_ZSCORE:    float = 2.0    # ASSUMED
_EXIT_ZSCORE:     float = 0.5    # ASSUMED
_KALMAN_R:        float = 0.01   # ASSUMED: observation noise
_KALMAN_Q:        float = 1e-5   # ASSUMED: beta drift rate
_SPREAD_LOOKBACK: int   = 60     # ASSUMED: rolling window for spread stats


@dataclass
class StatArbSignal:
    """Output from the stat-arb strategy for one bar.

    Attributes:
        signal_a: +1 = BUY A, -1 = SELL A, 0 = hold.
        signal_b: +1 = BUY B, -1 = SELL B, 0 = hold.
        spread_zscore: Current spread standardised score.
        beta: Current estimated hedge ratio.
        confidence: Normalised [0.0, 1.0]; driven by |zscore| beyond entry threshold.
        metadata: Diagnostic values for logging and post-trade analysis.
    """

    signal_a: int
    signal_b: int
    spread_zscore: float
    beta: float
    confidence: float
    metadata: dict


@dataclass
class _KalmanBeta:
    """Scalar Kalman filter state for dynamic hedge ratio estimation.

    State: β  (scalar, 1-dimensional)
    """

    beta: float = 1.0         # Prior estimate of β
    p: float = 1.0            # Prior error variance (P)
    r: float = _KALMAN_R
    q: float = _KALMAN_Q

    def update(self, log_a: float, log_b: float) -> float:
        """Update Kalman state with one observation pair.

        Uses log_b as the design vector: log_a = β × log_b + ε

        Args:
            log_a: log(price_A)
            log_b: log(price_B)

        Returns:
            Updated β estimate.
        """
        if abs(log_b) < 1e-12:
            return self.beta

        # Predict step
        p_pred = self.p + self.q

        # Kalman gain
        k = p_pred * log_b / (log_b ** 2 * p_pred + self.r)

        # Update
        innovation = log_a - self.beta * log_b
        self.beta += k * innovation
        self.p = (1.0 - k * log_b) * p_pred

        return self.beta


@dataclass
class _OLSBeta:
    """Rolling OLS fallback for hedge ratio estimation (no filterpy needed).

    Estimates β via:  β = Cov(log_a, log_b) / Var(log_b)
    """

    log_a_history: deque = field(default_factory=lambda: deque(maxlen=_SPREAD_LOOKBACK))
    log_b_history: deque = field(default_factory=lambda: deque(maxlen=_SPREAD_LOOKBACK))

    def update(self, log_a: float, log_b: float) -> float:
        """Update OLS state and return current β estimate.

        Args:
            log_a: log(price_A)
            log_b: log(price_B)

        Returns:
            β estimate (1.0 until enough history).
        """
        self.log_a_history.append(log_a)
        self.log_b_history.append(log_b)

        n = len(self.log_a_history)
        if n < 5:
            return 1.0

        mean_a = sum(self.log_a_history) / n
        mean_b = sum(self.log_b_history) / n

        cov = sum(
            (a - mean_a) * (b - mean_b)
            for a, b in zip(self.log_a_history, self.log_b_history)
        )
        var_b = sum((b - mean_b) ** 2 for b in self.log_b_history)

        if var_b < 1e-12:
            return 1.0

        return cov / var_b


@dataclass
class _SymbolPairState:
    """Mutable state for one (symbol_a, symbol_b) pair."""

    kalman: _KalmanBeta = field(default_factory=_KalmanBeta)
    ols: _OLSBeta = field(default_factory=_OLSBeta)
    spread_history: deque = field(default_factory=lambda: deque(maxlen=_SPREAD_LOOKBACK))
    in_position: bool = False
    position_direction: int = 0   # +1 = long A short B, -1 = short A long B
    use_kalman: bool = True


def _try_import_filterpy() -> bool:
    """Check if filterpy is available.

    Returns:
        True if filterpy can be imported, False otherwise.
    """
    try:
        import filterpy  # noqa: F401
        return True
    except ImportError:
        return False


class StatArbStrategy:
    """Statistical arbitrage strategy for cointegrated NSE stock pairs.

    This strategy is multi-symbol: each call to ``on_bar_pair`` processes
    a simultaneous observation of both legs.

    Usage::

        strategy = StatArbStrategy(symbol_a="HDFCBANK", symbol_b="ICICIBANK")
        signal = strategy.on_bar_pair(
            price_a=1650.0, price_b=960.0,
            volume_a=50000, volume_b=80000,
        )
        if signal.signal_a != 0:
            executor.submit(symbol_a, signal.signal_a)
            executor.submit(symbol_b, signal.signal_b)

    Parameters
    ----------
    symbol_a : str
        First leg of the pair.
    symbol_b : str
        Second leg of the pair.
    entry_zscore : float
        Z-score threshold to enter. ASSUMED: 2.0.
    exit_zscore : float
        Z-score threshold to exit (reversion). ASSUMED: 0.5.
    kalman_r : float
        Kalman observation noise. ASSUMED: 0.01.
    kalman_q : float
        Kalman state noise (β drift). ASSUMED: 1e-5.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        entry_zscore: float = _ENTRY_ZSCORE,
        exit_zscore: float = _EXIT_ZSCORE,
        kalman_r: float = _KALMAN_R,
        kalman_q: float = _KALMAN_Q,
    ) -> None:
        self._symbol_a = symbol_a
        self._symbol_b = symbol_b
        self._entry_zscore = entry_zscore
        self._exit_zscore = exit_zscore

        _use_kalman = _try_import_filterpy()
        if not _use_kalman:
            log.warning(
                "stat_arb.filterpy_not_available",
                fallback="rolling_ols",
            )

        self._state = _SymbolPairState(
            kalman=_KalmanBeta(r=kalman_r, q=kalman_q),
            use_kalman=_use_kalman,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def on_bar_pair(
        self,
        price_a: float,
        price_b: float,
        volume_a: float = 0.0,
        volume_b: float = 0.0,
    ) -> StatArbSignal:
        """Process a simultaneous observation of both pair legs.

        Args:
            price_a: Current price of symbol_a.
            price_b: Current price of symbol_b.
            volume_a: Bar volume of symbol_a (informational).
            volume_b: Bar volume of symbol_b (informational).

        Returns:
            StatArbSignal with signal_a, signal_b, spread_zscore, beta, confidence.
        """
        if price_a <= 0 or price_b <= 0:
            return StatArbSignal(
                signal_a=0, signal_b=0, spread_zscore=0.0, beta=1.0,
                confidence=0.0, metadata={"reason": "invalid_prices"}
            )

        log_a = math.log(price_a)
        log_b = math.log(price_b)

        state = self._state

        # Update hedge ratio estimate
        if state.use_kalman:
            beta = state.kalman.update(log_a, log_b)
        else:
            beta = state.ols.update(log_a, log_b)

        # Compute spread
        spread = log_a - beta * log_b
        state.spread_history.append(spread)

        # Need enough history for spread statistics
        if len(state.spread_history) < 10:
            return StatArbSignal(
                signal_a=0, signal_b=0, spread_zscore=0.0, beta=round(beta, 4),
                confidence=0.0, metadata={"reason": "warming_up"}
            )

        spread_mean = sum(state.spread_history) / len(state.spread_history)
        spread_std = _std(state.spread_history, spread_mean)

        if spread_std < 1e-10:
            return StatArbSignal(
                signal_a=0, signal_b=0, spread_zscore=0.0, beta=round(beta, 4),
                confidence=0.0, metadata={"reason": "zero_spread_std"}
            )

        zscore = (spread - spread_mean) / spread_std

        # ── Manage existing position ───────────────────────────────────────────
        if state.in_position:
            if abs(zscore) < self._exit_zscore:
                state.in_position = False
                state.position_direction = 0
                log.info(
                    "stat_arb.exit",
                    symbol_a=self._symbol_a,
                    symbol_b=self._symbol_b,
                    zscore=round(zscore, 3),
                )
                return StatArbSignal(
                    signal_a=-state.position_direction,
                    signal_b=+state.position_direction,
                    spread_zscore=round(zscore, 4),
                    beta=round(beta, 4),
                    confidence=0.0,
                    metadata={"reason": "spread_reverted"},
                )
            return StatArbSignal(
                signal_a=0, signal_b=0, spread_zscore=round(zscore, 4),
                beta=round(beta, 4), confidence=0.0, metadata={}
            )

        # ── Entry logic ───────────────────────────────────────────────────────
        if zscore > self._entry_zscore:
            # Spread too wide: SELL A / BUY B
            direction = -1
        elif zscore < -self._entry_zscore:
            # Spread too narrow: BUY A / SELL B
            direction = +1
        else:
            return StatArbSignal(
                signal_a=0, signal_b=0, spread_zscore=round(zscore, 4),
                beta=round(beta, 4), confidence=0.0,
                metadata={"reason": "zscore_insufficient"}
            )

        state.in_position = True
        state.position_direction = direction
        confidence = round(min((abs(zscore) - self._entry_zscore) / self._entry_zscore, 1.0), 3)

        log.info(
            "stat_arb.signal",
            symbol_a=self._symbol_a,
            symbol_b=self._symbol_b,
            direction="LONG_A" if direction > 0 else "SHORT_A",
            zscore=round(zscore, 3),
            beta=round(beta, 4),
            confidence=confidence,
        )
        return StatArbSignal(
            signal_a=direction,
            signal_b=-direction,
            spread_zscore=round(zscore, 4),
            beta=round(beta, 4),
            confidence=confidence,
            metadata={
                "spread": round(spread, 6),
                "spread_mean": round(spread_mean, 6),
                "spread_std": round(spread_std, 6),
                "zscore": round(zscore, 3),
                "beta_estimator": "kalman" if state.use_kalman else "ols",
            },
        )

    def reset(self) -> None:
        """Reset all state for this pair (call at start of each day)."""
        use_k = self._state.use_kalman
        r = self._state.kalman.r
        q = self._state.kalman.q
        self._state = _SymbolPairState(
            kalman=_KalmanBeta(r=r, q=q),
            use_kalman=use_k,
        )

    @property
    def current_beta(self) -> float:
        """Current hedge ratio estimate."""
        return self._state.kalman.beta if self._state.use_kalman else (
            self._state.ols.update.__self__.log_a_history  # type: ignore[attr-defined]
            if False else 1.0
        )


def _std(values: deque, mean: float) -> float:
    """Compute population standard deviation of a deque.

    Args:
        values: Numeric deque.
        mean: Pre-computed mean.

    Returns:
        Standard deviation (0.0 if fewer than 2 values).
    """
    n = len(values)
    if n < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)
