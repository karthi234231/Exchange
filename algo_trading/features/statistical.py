"""Statistical and ML-based features (features 13-15, 20).

Provides:
- OFIKalmanFilter     : Kalman-smoothed price with OFI in the observation model (feature 13)
- HMMRegimeDetector   : 3-state Gaussian HMM market regime (feature 14)
- GARCHVolForecaster  : GARCH(1,1) one-step-ahead volatility forecast (feature 15)
- StatisticalFeatures : Orchestrator that adds all four features to a Polars DataFrame

No lookahead bias — all features are computable from data available at bar close.
All outputs are float32 except where exact precision is required (internal model state).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import polars as pl
import structlog
try:
    from filterpy.kalman import KalmanFilter  # type: ignore[import]
except ImportError:  # pragma: no cover
    KalmanFilter = None  # type: ignore[assignment,misc]

try:
    from hmmlearn import hmm  # type: ignore[import]
except ImportError:  # pragma: no cover
    hmm = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


class HMMFitError(Exception):
    """Raised when HMMRegimeDetector.fit() cannot proceed due to insufficient data."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_AUTOCORR_LAGS: int = 20          # lag-1 autocorrelation window (feature 20)
_HMM_N_STATES: int = 3
_HMM_MIN_DURATION: int = 5        # bars before regime is allowed to change [CDR-3B-C16]
_HMM_N_ITER: int = 200
_GARCH_ANNUALISE_FACTOR: float = 252.0 * 25.0  # 252 trading days × 25 15-min bars/day
# For 1-min bars: use 252 * 375; for daily: use 252.
# Caller is responsible for passing the appropriate returns series.


# ---------------------------------------------------------------------------
# Feature 13 — OFI-augmented Kalman Filter
# ---------------------------------------------------------------------------

class OFIKalmanFilter:
    """Kalman filter with Order Flow Imbalance in the observation model.

    State vector: [price, velocity]
    Observation:  price + beta * ofi   (OFI shifts the effective observation)

    Uses a 2-state constant-velocity model (standard for price tracking).
    Process noise Q and observation noise R are constructor parameters; they
    can be tuned via expectation-maximisation offline.

    Args:
        process_noise: Diagonal value for process noise covariance Q (default 0.01).
        obs_noise:     Observation noise variance R (default 1.0).
        ofi_beta:      Coupling coefficient for OFI in the observation equation.
            Defaults to 0.05 (derived from NSE historical data; recalibrate by
            regressing mid-price changes on OFI over a representative in-sample
            period and taking the median beta across liquid large-cap symbols).
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        obs_noise: float = 1.0,
        ofi_beta: float = 0.05,
    ) -> None:
        self._ofi_beta = ofi_beta
        self._kf = _build_kalman_filter(process_noise, obs_noise)
        self._initialized = False

    # ------------------------------------------------------------------
    def update(self, price: float, ofi: float) -> float:
        """Update filter with a new price observation and OFI value.

        The effective observation fed to the filter is:
            z = price - ofi_beta * ofi
        so that OFI shifts the observed price toward its fair value.

        Args:
            price: Mid-price or last traded price for the current bar.
            ofi:   Order-flow imbalance value for the current bar.
                   Use 0.0 when L2 data is unavailable.

        Returns:
            Kalman-smoothed (filtered) price estimate as a float.
        """
        if not self._initialized:
            self._kf.x[0] = price
            self._kf.x[1] = 0.0
            self._initialized = True

        z = np.array([[price - self._ofi_beta * ofi]])
        self._kf.predict()
        self._kf.update(z)
        return float(self._kf.x[0])

    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[float, float]:
        """Return current (price_estimate, velocity_estimate)."""
        return float(self._kf.x[0]), float(self._kf.x[1])


def _build_kalman_filter(q: float, r: float) -> "KalmanFilter":
    """Instantiate a 2-state constant-velocity KalmanFilter.

    Args:
        q: Scalar process noise; broadcast to 2×2 diagonal.
        r: Scalar observation noise variance.

    Returns:
        Configured ``filterpy.kalman.KalmanFilter`` instance.
    """
    if KalmanFilter is None:
        raise ImportError(
            "filterpy is required for Kalman filter features. "
            "Install with: pip install filterpy"
        )
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # State transition: price_t = price_{t-1} + velocity_{t-1}; velocity unchanged
    kf.F = np.array([[1.0, 1.0],
                     [0.0, 1.0]])
    # Observation: we only observe price
    kf.H = np.array([[1.0, 0.0]])
    # Process noise covariance
    kf.Q = np.eye(2) * q
    # Observation noise variance
    kf.R = np.array([[r]])
    # Initial state covariance
    kf.P = np.eye(2) * 1.0
    # Initial state
    kf.x = np.zeros((2, 1))
    return kf


# ---------------------------------------------------------------------------
# Feature 14 — HMM Regime Detection
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """3-state Gaussian HMM for market regime detection.

    States are labelled bull / bear / sideways based on the mean return of
    each fitted state (highest mean → bull, lowest → bear, middle → sideways).

    A minimum-duration post-processing step prevents chattering: a regime
    change is confirmed only when the model has predicted the new state for
    at least ``min_duration_bars`` consecutive bars [CDR-3B-C16].

    Args:
        n_states:           Number of HMM hidden states (default 3).
        min_duration_bars:  Minimum consecutive bars in new state before a
                            regime transition is accepted (default 5).
    """

    def __init__(
        self,
        n_states: int = _HMM_N_STATES,
        min_duration_bars: int = _HMM_MIN_DURATION,
    ) -> None:
        if hmm is None:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install with: pip install hmmlearn"
            )
        self._n_states = n_states
        self._min_duration = min_duration_bars
        self._model: Optional[hmm.GaussianHMM] = None
        self._is_fitted = False
        # Sticky-prior state for min-duration filter
        self._confirmed_regime: int = 0
        self._candidate_regime: int = 0
        self._candidate_count: int = 0

    # ------------------------------------------------------------------
    def fit(self, returns: np.ndarray) -> None:
        """Fit the HMM on a sequence of log returns.

        Should be called weekly (or whenever a full history refit is needed).
        Fitting is idempotent — subsequent calls replace the previous model.

        Args:
            returns: 1-D array of log returns, shape (T,). Must have at least
                     10 × n_states observations for meaningful estimation.
        """
        log = logger.bind(n_obs=len(returns), n_states=self._n_states)
        if len(returns) < self._n_states * 10:
            raise HMMFitError(
                f"Insufficient data: need {self._n_states * 10} observations, "
                f"got {len(returns)}"
            )

        obs = returns.reshape(-1, 1)
        model = hmm.GaussianHMM(
            n_components=self._n_states,
            covariance_type="diag",
            n_iter=_HMM_N_ITER,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(obs)

        self._model = model
        self._is_fitted = True
        # Reset sticky-prior state
        self._confirmed_regime = 0
        self._candidate_regime = 0
        self._candidate_count = 0
        log.info("HMM fitted", converged=model.monitor_.converged)

    # ------------------------------------------------------------------
    def predict_regime(
        self, recent_returns: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Predict the current regime with minimum-duration smoothing.

        Decodes the most likely state sequence using Viterbi, reads the last
        predicted state, then applies the sticky-prior filter [CDR-3B-C16]:
        the regime index is only updated when the raw prediction has been
        stable for ``min_duration_bars`` consecutive calls.

        Args:
            recent_returns: 1-D array of recent log returns. Should include
                            enough history for stable decoding (≥ 30 bars).

        Returns:
            A ``(regime_index, state_probabilities)`` tuple where
            ``regime_index`` is the smoothed regime (0/1/2) and
            ``state_probabilities`` is a 1-D array of length ``n_states``
            giving the posterior probability for the last step.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("HMMRegimeDetector.fit() must be called before predict_regime()")

        obs = recent_returns.reshape(-1, 1)
        raw_seq = self._model.predict(obs)
        raw_regime: int = int(raw_seq[-1])

        # Posterior probabilities for last bar.
        # hmmlearn.GaussianHMM.score_samples returns (log_likelihood_array, posteriors_2d)
        _, posteriors_2d = self._model.score_samples(obs)
        posteriors: np.ndarray = posteriors_2d[-1]

        # Min-duration sticky filter
        if raw_regime == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = raw_regime
            self._candidate_count = 1

        if self._candidate_count >= self._min_duration:
            self._confirmed_regime = self._candidate_regime

        return self._confirmed_regime, posteriors.astype("float32")

    # ------------------------------------------------------------------
    def regime_to_label(self, regime_idx: int, means: np.ndarray) -> str:
        """Map a state index to a human-readable label.

        Labelling is based on ranked mean returns: highest mean → bull,
        lowest → bear, middle → sideways.

        Args:
            regime_idx: Integer state index (0 to n_states-1).
            means:      1-D array of mean returns per state, length n_states.
                        Typically ``self._model.means_.flatten()``.

        Returns:
            One of ``"bull"``, ``"bear"``, or ``"sideways"``.
        """
        ranked = np.argsort(means)  # ascending: [bear_idx, sideways_idx, bull_idx]
        if regime_idx == ranked[-1]:
            return "bull"
        if regime_idx == ranked[0]:
            return "bear"
        return "sideways"

    # ------------------------------------------------------------------
    @property
    def state_means(self) -> Optional[np.ndarray]:
        """Return mean returns per state, or None if not fitted."""
        if self._model is None:
            return None
        return self._model.means_.flatten()


# ---------------------------------------------------------------------------
# Feature 15 — GARCH(1,1) volatility forecast
# ---------------------------------------------------------------------------

class GARCHVolForecaster:
    """GARCH(1,1) one-step-ahead conditional volatility forecast.

    Fits on log returns (daily gap-adjusted or intraday depending on bar
    type). Returns the annualised 1-step-ahead conditional standard deviation
    (not variance).

    Re-estimation is triggered externally by calling ``refit_at_open`` at
    9:10 AM IST each trading day [IRF-Q5].
    """

    def __init__(self) -> None:
        self._result: Any = None  # arch ModelResult
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, log_returns: np.ndarray) -> None:
        """Fit GARCH(1,1) on log returns.

        Args:
            log_returns: 1-D array of log returns. Must have at least 50
                         non-NaN observations for stable estimation.
        """
        from arch import arch_model  # type: ignore[import]  # lazy import

        clean = log_returns[~np.isnan(log_returns)]
        if len(clean) < 50:
            logger.warning("GARCH fit: fewer than 50 observations; skipping", n=len(clean))
            return

        # Scale by 100 for numerical stability (arch recommendation)
        model = arch_model(clean * 100.0, vol="Garch", p=1, q=1, dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp="off", show_warning=False)

        self._is_fitted = True
        logger.info("GARCH fitted", n_obs=len(clean))

    # ------------------------------------------------------------------
    def forecast_one_step(self) -> float:
        """Return annualised 1-step-ahead conditional volatility forecast.

        The GARCH model was fitted on returns scaled by 100, so the forecast
        variance is divided by 10000 before annualising.

        Returns:
            Annualised volatility as a float. Returns ``float('nan')`` if the
            model has not been fitted.

        Raises:
            RuntimeError: If ``fit`` has not been called (guarded with NaN
                return rather than hard raise to avoid crashing live pipeline).
        """
        if not self._is_fitted or self._result is None:
            logger.warning("GARCH not fitted; returning NaN")
            return float("nan")

        forecast = self._result.forecast(horizon=1, reindex=False)
        # forecast.variance is a DataFrame; last row, first column is h.1
        variance_scaled = float(forecast.variance.iloc[-1, 0])
        # Unscale from the ×100 scaling applied at fit time
        variance = variance_scaled / 10_000.0
        # Annualise: sqrt(variance * bars_per_year)
        annualised_vol = math.sqrt(variance * _GARCH_ANNUALISE_FACTOR)
        return annualised_vol

    # ------------------------------------------------------------------
    def refit_at_open(self, gap_adjusted_returns: np.ndarray) -> None:
        """Refit GARCH on gap-adjusted overnight returns at 9:10 AM IST.

        Called once per trading day before market open. The gap-adjusted
        returns should account for the overnight price gap to avoid
        artificially inflated return estimates [IRF-Q5].

        Args:
            gap_adjusted_returns: 1-D array of gap-adjusted log returns.
        """
        logger.info("GARCH refit at open", n_obs=len(gap_adjusted_returns))
        self.fit(gap_adjusted_returns)


# ---------------------------------------------------------------------------
# Feature 20 — Returns autocorrelation (lag-1 over last 20 bars)
# ---------------------------------------------------------------------------

def _lag1_autocorr(log_returns: np.ndarray, window: int = _AUTOCORR_LAGS) -> np.ndarray:
    """Compute rolling lag-1 autocorrelation of log returns.

    Args:
        log_returns: 1-D array of log returns of length T.
        window:      Rolling window size (default 20).

    Returns:
        1-D array of length T. First ``window`` elements are NaN because
        the lookback period is not met.
    """
    n = len(log_returns)
    result = np.full(n, np.nan, dtype="float32")
    for i in range(window, n):
        chunk = log_returns[i - window: i]
        # lag-1 autocorrelation: corr(x[1:], x[:-1])
        x = chunk[1:]
        y = chunk[:-1]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            result[i] = 0.0
        else:
            result[i] = float(np.corrcoef(x, y)[0, 1])
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class StatisticalFeatures:
    """Add statistical features 13, 14, 15, 20 to a Polars DataFrame.

    Maintains stateful instances of each sub-model so they can be shared
    across batch and single-bar calls.
    """

    def __init__(self) -> None:
        self.kalman = OFIKalmanFilter()
        self.hmm = HMMRegimeDetector()
        self.garch = GARCHVolForecaster()

    # ------------------------------------------------------------------
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add statistical feature columns to a Polars DataFrame.

        Computes:
        - feat_kalman_price  (feature 13): Kalman-smoothed price (float32)
        - feat_hmm_regime    (feature 14): regime index 0/1/2 (float32)
        - feat_hmm_prob      (feature 14): probability of current regime (float32)
        - feat_garch_vol     (feature 15): annualised GARCH vol forecast (float32)
        - feat_autocorr_lag1 (feature 20): lag-1 return autocorrelation (float32)

        The GARCH and HMM models must be pre-fitted via ``garch.fit`` /
        ``hmm.fit`` before calling this method; otherwise the respective
        columns will be all-NaN.

        Args:
            df: Polars DataFrame with at minimum a ``close`` column.
                Optional ``ofi`` column is used by the Kalman filter.

        Returns:
            Input DataFrame with five additional float32 columns.
        """
        _validate_close(df)
        closes = df["close"].to_numpy().astype("float64")
        ofi_vals = df["ofi"].to_numpy().astype("float64") if "ofi" in df.columns else np.zeros(len(df))

        # --- Feature 13: Kalman-smoothed price ---
        kalman_prices = np.empty(len(df), dtype="float32")
        local_kf = OFIKalmanFilter()  # fresh instance for batch (no state leak)
        for i, (p, o) in enumerate(zip(closes, ofi_vals)):
            kalman_prices[i] = local_kf.update(float(p), float(o))

        # --- Feature 15: GARCH vol (single value broadcast; re-fit externally) ---
        garch_vol = self.garch.forecast_one_step()
        garch_arr = np.full(len(df), garch_vol, dtype="float32")

        # --- Feature 20: lag-1 autocorrelation ---
        log_returns = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
        log_returns = np.concatenate([[np.nan], log_returns])
        autocorr_arr = _lag1_autocorr(log_returns, _AUTOCORR_LAGS)

        # --- Feature 14: HMM regime (applied to log returns) ---
        hmm_regime_arr = np.full(len(df), np.nan, dtype="float32")
        hmm_prob_arr = np.full(len(df), np.nan, dtype="float32")
        if self.hmm._is_fitted:
            for i in range(_HMM_MIN_DURATION + 1, len(df)):
                window_returns = log_returns[max(0, i - 60): i]
                valid = window_returns[~np.isnan(window_returns)]
                if len(valid) >= _HMM_N_STATES * 5:
                    regime, probs = self.hmm.predict_regime(valid)
                    hmm_regime_arr[i] = float(regime)
                    hmm_prob_arr[i] = float(probs[regime])

        feat_df = pl.DataFrame({
            "feat_kalman_price": kalman_prices,
            "feat_hmm_regime": hmm_regime_arr,
            "feat_hmm_prob": hmm_prob_arr,
            "feat_garch_vol": garch_arr,
            "feat_autocorr_lag1": autocorr_arr,
        })
        return pl.concat([df, feat_df], how="horizontal")

    # ------------------------------------------------------------------
    def compute_single_bar(
        self,
        recent_bars: pl.DataFrame,
        current_ofi: float,
    ) -> dict[str, float]:
        """Compute statistical features for the most recent bar.

        Args:
            recent_bars: Polars DataFrame with recent OHLCV bars. Should
                         include at least 60 bars for HMM stability.
            current_ofi: Order-flow imbalance for the current (latest) bar.

        Returns:
            Dict with canonical FEATURE_NAMES keys: ``kalman_price``,
            ``hmm_regime``, ``garch_vol``, ``autocorr_lag1``.
        """
        closes = recent_bars["close"].to_numpy().astype("float64")
        if len(closes) == 0:
            return {
                "kalman_price":   float("nan"),
                "hmm_regime":     float("nan"),
                "garch_vol":      float("nan"),
                "autocorr_lag1":  float("nan"),
            }

        # Kalman: stream through history to build up state, then update with latest
        kf = OFIKalmanFilter()
        for p in closes[:-1]:
            kf.update(float(p), 0.0)
        kalman_price = kf.update(float(closes[-1]), current_ofi)

        # GARCH: broadcast latest forecast
        garch_vol = self.garch.forecast_one_step()

        # Autocorrelation
        log_returns = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
        if len(log_returns) >= _AUTOCORR_LAGS:
            chunk = log_returns[-_AUTOCORR_LAGS:]
            x, y = chunk[1:], chunk[:-1]
            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                autocorr = float(np.corrcoef(x, y)[0, 1])
            else:
                autocorr = 0.0
        else:
            autocorr = float("nan")

        # HMM regime
        hmm_regime: float = float("nan")
        hmm_prob: float = float("nan")
        if self.hmm._is_fitted and len(log_returns) >= _HMM_N_STATES * 5:
            valid = log_returns[~np.isnan(log_returns)]
            if len(valid) >= _HMM_N_STATES * 5:
                regime, probs = self.hmm.predict_regime(valid)
                hmm_regime = float(regime)
                hmm_prob = float(probs[regime])

        # Map to canonical FEATURE_NAMES from engine.py (no feat_ prefix, no hmm_prob)
        return {
            "kalman_price":  float(kalman_price),
            "hmm_regime":    hmm_regime,
            "garch_vol":     float(garch_vol),
            "autocorr_lag1": float(autocorr),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_close(df: pl.DataFrame) -> None:
    """Raise ValueError if ``close`` column is absent."""
    if "close" not in df.columns:
        raise ValueError("DataFrame missing required column: close")


