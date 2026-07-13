"""
hawkes.py — Hawkes Self-Exciting Point Process for NSE order flow burst detection.

Stage 3+ only: requires sub-minute tick data (not 15-min bars).
At 15-min resolution, Hawkes inter-arrival times are meaningless.
Only import this module when DATA_FEED produces sub-second ticks.

Theory (Hawkes 1971, Bacry et al. 2015):
    Intensity:  λ(t) = μ + Σ_{tᵢ < t} α · exp(-β · (t - tᵢ))
    Stationarity condition (branching ratio):  n = α/β < 1
    Typical NSE values:  μ ≈ 2-10 events/sec,  n ≈ 0.3-0.7

Application:
    If λ(t) / μ > 3.0  → burst / toxic order flow → reduce size or skip entry.
    If λ(t) / μ > 1.5  → elevated activity.
    Otherwise           → normal baseline.

Log-likelihood (for MLE fitting):
    ℓ = Σᵢ log(λ(tᵢ)) − ∫₀ᵀ λ(t) dt
    Integral: μ·T + (α/β)·Σᵢ (1 − exp(−β·(T − tᵢ)))

The running-sum trick keeps log-likelihood computation O(n):
    R₀ = 0
    Rᵢ = exp(−β·(tᵢ − tᵢ₋₁)) · (Rᵢ₋₁ + 1)
    λ(tᵢ) = μ + α·Rᵢ₋₁
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage gate
# ---------------------------------------------------------------------------
STAGE = "3+"
MIN_TIMEFRAME_SECONDS = 60  # Hawkes is invalid above this bar interval

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

import numpy as np
from scipy import optimize
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# HawkesParams
# ---------------------------------------------------------------------------

@dataclass
class HawkesParams:
    """
    Parameters of a univariate Hawkes process.

    Attributes
    ----------
    mu    : float — baseline intensity (events / second), must be > 0
    alpha : float — excitation jump size per event, must be > 0
    beta  : float — decay rate (1/s), must be > alpha for stationarity
    """

    mu: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")

    @property
    def branching_ratio(self) -> float:
        """n = α/β.  Must be < 1 for the process to be stationary."""
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        """True when the branching ratio α/β < 1."""
        return self.branching_ratio < 1.0


# ---------------------------------------------------------------------------
# HawkesProcess
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = HawkesParams(mu=2.0, alpha=0.5, beta=2.0)


class HawkesProcess:
    """
    Univariate Hawkes self-exciting point process with MLE fitting.

    Usage
    -----
    >>> hp = HawkesProcess()
    >>> params = hp.fit(timestamps)          # sorted float array (seconds)
    >>> ratio  = hp.intensity_ratio(timestamps, eval_time=timestamps[-1] + 0.1)
    """

    def __init__(self, params: Optional[HawkesParams] = None) -> None:
        self._params: HawkesParams = params if params is not None else _DEFAULT_PARAMS
        self._fitted: bool = params is not None  # treat supplied params as pre-fitted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, timestamps: np.ndarray) -> HawkesParams:
        """
        Fit Hawkes parameters to `timestamps` via MLE (L-BFGS-B).

        Parameters
        ----------
        timestamps : np.ndarray (float64)
            Sorted event arrival times in *seconds*.

        Returns
        -------
        HawkesParams
            Fitted parameters (always stationary due to bounds).
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("timestamps must be a 1-D array with ≥ 2 elements")

        mu0, alpha0, beta0 = 2.0, 0.5, 2.0

        def neg_ll(theta: np.ndarray) -> float:
            mu, alpha, beta = theta
            if mu <= 0 or alpha <= 0 or beta <= alpha:
                return 1e18
            p = HawkesParams(mu=mu, alpha=alpha, beta=beta)
            return self.log_likelihood(timestamps, p)

        x0 = np.array([mu0, alpha0, beta0])
        # Bounds enforce stationarity: beta > alpha  (handled by lower-bound on beta).
        # We set beta_lb = alpha0 + 0.001 as an approximation; the objective also
        # returns a large value if beta ≤ alpha at any candidate point.
        bounds = [
            (0.001, 50.0),          # mu
            (0.001, 10.0),          # alpha
            (alpha0 + 0.001, 100.0),  # beta  (> initial alpha → stationary start)
        ]

        result = optimize.minimize(
            neg_ll,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        mu_fit, alpha_fit, beta_fit = result.x
        # Clamp to valid domain (optimizer can land exactly on boundary).
        mu_fit = max(mu_fit, 1e-6)
        alpha_fit = max(alpha_fit, 1e-6)
        beta_fit = max(beta_fit, alpha_fit + 1e-6)  # guarantee stationarity

        fitted = HawkesParams(mu=mu_fit, alpha=alpha_fit, beta=beta_fit)
        self._params = fitted
        self._fitted = True

        log.info(
            "hawkes_fitted",
            mu=round(fitted.mu, 4),
            alpha=round(fitted.alpha, 4),
            beta=round(fitted.beta, 4),
            branching_ratio=round(fitted.branching_ratio, 4),
            n_events=len(timestamps),
            converged=result.success,
        )
        return fitted

    # ------------------------------------------------------------------
    # Log-likelihood (O(n) via running sum)
    # ------------------------------------------------------------------

    def log_likelihood(
        self, timestamps: np.ndarray, params: HawkesParams
    ) -> float:
        """
        Compute the *negative* Hawkes log-likelihood (for use with minimisers).

        Uses the O(n) running-sum recursion:
            R₀ = 0
            Rᵢ = exp(−β·(tᵢ − tᵢ₋₁)) · (Rᵢ₋₁ + 1)
            λ(tᵢ) = μ + α·Rᵢ₋₁

        Parameters
        ----------
        timestamps : np.ndarray (float64) — sorted event times in seconds.
        params     : HawkesParams

        Returns
        -------
        float — *negative* log-likelihood (so minimising gives MLE).
        """
        ts = np.asarray(timestamps, dtype=np.float64)
        n = len(ts)
        if n == 0:
            return 0.0

        mu = params.mu
        alpha = params.alpha
        beta = params.beta
        T = ts[-1]

        # --- Intensity sum term: Σ log(λ(tᵢ)) ---
        R = 0.0
        log_lam_sum = 0.0
        for i in range(n):
            if i > 0:
                dt = ts[i] - ts[i - 1]
                R = np.exp(-beta * dt) * (R + 1.0)
            lam_i = mu + alpha * R
            if lam_i <= 0.0:
                # Degenerate — penalise heavily.
                return 1e18
            log_lam_sum += np.log(lam_i)

        # --- Integral term: μ·T + (α/β)·Σᵢ (1 − exp(−β·(T − tᵢ))) ---
        decay_terms = np.exp(-beta * (T - ts))
        integral = mu * T + (alpha / beta) * np.sum(1.0 - decay_terms)

        ll = log_lam_sum - integral
        return -ll  # negative for minimisation

    # ------------------------------------------------------------------
    # Intensity evaluation
    # ------------------------------------------------------------------

    def current_intensity(
        self, timestamps: np.ndarray, eval_time: float
    ) -> float:
        """
        Compute λ(eval_time) = μ + α·Σᵢ exp(−β·(eval_time − tᵢ))
        using only events strictly before eval_time.

        Requires the process to have been fitted (or constructed with params).

        Parameters
        ----------
        timestamps : np.ndarray — sorted event times in seconds.
        eval_time  : float      — time at which to evaluate intensity.

        Returns
        -------
        float ≥ μ
        """
        if not self._fitted:
            raise RuntimeError(
                "HawkesProcess must be fitted before calling current_intensity. "
                "Call hp.fit(timestamps) first."
            )
        ts = np.asarray(timestamps, dtype=np.float64)
        past = ts[ts < eval_time]
        if len(past) == 0:
            return self._params.mu

        excitation = np.sum(
            np.exp(-self._params.beta * (eval_time - past))
        )
        return self._params.mu + self._params.alpha * excitation

    def intensity_ratio(
        self, timestamps: np.ndarray, eval_time: float
    ) -> float:
        """
        Return λ(eval_time) / μ.

        Interpretation:
            < 1.5  → normal baseline
            1.5–3  → elevated order flow
            ≥ 3.0  → burst (potentially toxic)
        """
        lam = self.current_intensity(timestamps, eval_time)
        return lam / self._params.mu


# ---------------------------------------------------------------------------
# HawkesSignal enum
# ---------------------------------------------------------------------------

class HawkesSignal(str, Enum):
    """Order-flow burst classification based on Hawkes intensity ratio."""

    NORMAL = "normal"      # ratio < 1.5
    ELEVATED = "elevated"  # 1.5 ≤ ratio < 3.0
    BURST = "burst"        # ratio ≥ 3.0


def get_hawkes_signal(intensity_ratio: float) -> HawkesSignal:
    """
    Classify the current Hawkes intensity ratio into a trading signal.

    Parameters
    ----------
    intensity_ratio : float — output of HawkesProcess.intensity_ratio()

    Returns
    -------
    HawkesSignal
    """
    if intensity_ratio >= 3.0:
        return HawkesSignal.BURST
    if intensity_ratio >= 1.5:
        return HawkesSignal.ELEVATED
    return HawkesSignal.NORMAL


# ---------------------------------------------------------------------------
# HawkesFeatureEngine — rolling, auto-refit wrapper
# ---------------------------------------------------------------------------

class HawkesFeatureEngine:
    """
    Stateful feature engine that maintains a rolling window of tick timestamps,
    periodically refits a Hawkes process, and returns a feature dict on each
    `update()` call.

    Parameters
    ----------
    window_seconds      : int — how many seconds of history to keep (default 300).
    refit_every_n_events: int — refit after this many new ticks (default 500).
    """

    _MIN_EVENTS_TO_FIT = 50  # need enough data for stable MLE

    def __init__(
        self,
        window_seconds: int = 300,
        refit_every_n_events: int = 500,
    ) -> None:
        self._window_ns: int = window_seconds * 1_000_000_000
        self._refit_every: int = refit_every_n_events
        self._ts_deque: Deque[int] = deque()   # nanosecond timestamps
        self._hawkes: HawkesProcess = HawkesProcess()
        self._events_since_refit: int = 0
        self._total_events: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, timestamp_ns: int, price: float, qty: int) -> dict:
        """
        Ingest a new tick and return the current Hawkes feature vector.

        Parameters
        ----------
        timestamp_ns : int   — epoch nanoseconds of the tick.
        price        : float — last traded price (not used for Hawkes, reserved).
        qty          : int   — tick size (not used for Hawkes, reserved).

        Returns
        -------
        dict with keys:
            hawkes_intensity  : float — λ(now) in events/sec
            hawkes_ratio      : float — λ(now) / μ
            hawkes_signal     : str   — "normal" | "elevated" | "burst"
            branching_ratio   : float — α/β
        """
        self._ts_deque.append(timestamp_ns)
        self._events_since_refit += 1
        self._total_events += 1

        # Trim old events outside the rolling window.
        cutoff_ns = timestamp_ns - self._window_ns
        while self._ts_deque and self._ts_deque[0] < cutoff_ns:
            self._ts_deque.popleft()

        # Not enough data yet — return safe defaults.
        if self._total_events < self._MIN_EVENTS_TO_FIT:
            return self._default_features()

        # Refit if due.
        if (
            not self._hawkes._fitted
            or self._events_since_refit >= self._refit_every
        ):
            self._refit()

        # Evaluate intensity at "now".
        ts_sec = np.array(self._ts_deque, dtype=np.float64) / 1e9
        eval_time_sec = timestamp_ns / 1e9

        try:
            ratio = self._hawkes.intensity_ratio(ts_sec, eval_time_sec)
            intensity = self._hawkes.current_intensity(ts_sec, eval_time_sec)
            br = self._hawkes._params.branching_ratio
        except RuntimeError:
            return self._default_features()

        signal = get_hawkes_signal(ratio)

        return {
            "hawkes_intensity": float(intensity),
            "hawkes_ratio": float(ratio),
            "hawkes_signal": signal.value,
            "branching_ratio": float(br),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        ts_sec = np.array(self._ts_deque, dtype=np.float64) / 1e9
        if len(ts_sec) < self._MIN_EVENTS_TO_FIT:
            return
        try:
            self._hawkes.fit(ts_sec)
            self._events_since_refit = 0
        except Exception as exc:
            log.warning("hawkes_refit_failed", error=str(exc))

    @staticmethod
    def _default_features() -> dict:
        return {
            "hawkes_intensity": 0.0,
            "hawkes_ratio": 1.0,
            "hawkes_signal": HawkesSignal.NORMAL.value,
            "branching_ratio": 0.0,
        }
