"""
test_hawkes.py — pytest tests for algo_trading.features.hawkes

Tests use synthetic tick timestamps only (no file I/O).

Helpers
-------
make_poisson_timestamps(n, rate, seed)
    Homogeneous Poisson process — inter-arrivals ~ Exp(1/rate).

make_hawkes_timestamps(n, mu, alpha, beta, seed)
    Hawkes process via Ogata's thinning algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from algo_trading.features.hawkes import (
    HawkesFeatureEngine,
    HawkesParams,
    HawkesProcess,
    HawkesSignal,
    get_hawkes_signal,
)


# ---------------------------------------------------------------------------
# Synthetic timestamp generators
# ---------------------------------------------------------------------------

def make_poisson_timestamps(
    n: int = 200, rate: float = 5.0, seed: int = 42
) -> np.ndarray:
    """
    Generate `n` sorted event times from a homogeneous Poisson process
    with the given `rate` (events/second).

    Inter-arrival times are i.i.d. Exponential(1/rate).
    """
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / rate, size=n)
    timestamps = np.cumsum(inter_arrivals)
    return timestamps.astype(np.float64)


def make_hawkes_timestamps(
    n: int = 200,
    mu: float = 2.0,
    alpha: float = 0.5,
    beta: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate `n` events from a univariate Hawkes process using Ogata's
    thinning algorithm.

    Algorithm:
    1.  t = 0, events = []
    2.  M = λ(t) (upper bound on intensity — starts at μ)
    3.  Draw s ~ Exp(1/M), set t = t + s
    4.  Recompute λ(t); accept with probability λ(t)/M
    5.  If accepted: record t, update M = λ(t) [with new event included]
        If rejected: update M = λ(t) [no new event; M can only decrease]
    6.  Repeat until n events collected.
    """
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0

    def intensity(t_eval: float, history: list[float]) -> float:
        if not history:
            return mu
        diffs = np.array(history)
        return mu + alpha * np.sum(np.exp(-beta * (t_eval - diffs)))

    lam_current = mu  # upper bound starts at baseline

    while len(events) < n:
        # Draw candidate inter-arrival time.
        s = rng.exponential(scale=1.0 / lam_current)
        t = t + s

        # Compute exact intensity at the candidate time.
        lam_t = intensity(t, events)

        # Accept / reject (thinning step).
        if rng.random() <= lam_t / lam_current:
            events.append(t)
            # After acceptance the intensity jumps by alpha; update upper bound.
            lam_current = lam_t + alpha  # jump then decay bound
        else:
            # Intensity can only have decreased — update bound.
            lam_current = lam_t

        # Guard: if upper bound somehow collapses, reset to at least mu.
        lam_current = max(lam_current, mu)

    return np.array(events, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests — HawkesParams
# ---------------------------------------------------------------------------

def test_params_stationarity():
    """α/β = 0.25 < 1  →  stationary."""
    p = HawkesParams(mu=2.0, alpha=0.5, beta=2.0)
    assert p.is_stationary is True


def test_params_non_stationary():
    """α/β = 1.5 ≥ 1  →  non-stationary."""
    p = HawkesParams(mu=2.0, alpha=3.0, beta=2.0)
    assert p.is_stationary is False


def test_branching_ratio():
    """α=0.6, β=2.0  →  n = 0.3."""
    p = HawkesParams(mu=1.0, alpha=0.6, beta=2.0)
    assert p.branching_ratio == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Tests — HawkesProcess.fit
# ---------------------------------------------------------------------------

def test_fit_returns_stationary_params():
    """
    Fitting on Poisson data must return stationary HawkesParams.
    The MLE optimiser uses bounds that enforce α/β < 1 (approximately).
    We also clamp post-optimisation, so the result must always be stationary.
    """
    ts = make_poisson_timestamps(n=300, rate=5.0, seed=0)
    hp = HawkesProcess()
    params = hp.fit(ts)
    assert isinstance(params, HawkesParams)
    assert params.is_stationary, (
        f"Fitted params not stationary: α={params.alpha:.4f}, β={params.beta:.4f}, "
        f"n={params.branching_ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests — HawkesProcess.log_likelihood
# ---------------------------------------------------------------------------

def test_log_likelihood_negative():
    """
    log_likelihood() returns the *negative* log-likelihood (a positive float
    when the process has reasonable parameters on real data).
    The return type must be float.
    """
    ts = make_poisson_timestamps(n=100, rate=5.0, seed=7)
    hp = HawkesProcess()
    p = HawkesParams(mu=2.0, alpha=0.5, beta=2.0)
    nll = hp.log_likelihood(ts, p)
    assert isinstance(nll, float)
    # The negative LL should be a finite number (not inf or nan).
    assert np.isfinite(nll), f"log_likelihood returned non-finite value: {nll}"


# ---------------------------------------------------------------------------
# Tests — HawkesProcess.current_intensity
# ---------------------------------------------------------------------------

def test_current_intensity_ge_mu():
    """
    Hawkes intensity λ(t) ≥ μ everywhere because excitation only adds.
    Test at multiple evaluation times.
    """
    ts = make_hawkes_timestamps(n=200, mu=2.0, alpha=0.5, beta=2.0, seed=1)
    hp = HawkesProcess()
    hp.fit(ts)
    mu = hp._params.mu

    eval_times = np.linspace(ts[10], ts[-1] + 1.0, 20)
    for t in eval_times:
        lam = hp.current_intensity(ts, t)
        assert lam >= mu - 1e-9, (
            f"intensity {lam:.6f} < mu {mu:.6f} at t={t:.3f}"
        )


def test_intensity_ratio_one_for_empty_history():
    """
    When no events precede eval_time, intensity = μ  →  ratio = 1.0.
    """
    # Fit on some data first so the process is marked fitted.
    ts = make_poisson_timestamps(n=150, rate=5.0, seed=3)
    hp = HawkesProcess()
    hp.fit(ts)

    # Evaluate well before the first event — no history.
    ratio = hp.intensity_ratio(ts, eval_time=ts[0] - 10.0)
    assert ratio == pytest.approx(1.0, abs=1e-9), (
        f"Expected ratio ≈ 1.0 for empty history, got {ratio}"
    )


# ---------------------------------------------------------------------------
# Tests — HawkesSignal / get_hawkes_signal
# ---------------------------------------------------------------------------

def test_hawkes_signal_thresholds():
    """get_hawkes_signal covers all three cases at the boundary values."""
    assert get_hawkes_signal(0.9) == HawkesSignal.NORMAL
    assert get_hawkes_signal(1.0) == HawkesSignal.NORMAL
    assert get_hawkes_signal(1.49) == HawkesSignal.NORMAL
    assert get_hawkes_signal(1.5) == HawkesSignal.ELEVATED
    assert get_hawkes_signal(2.9) == HawkesSignal.ELEVATED
    assert get_hawkes_signal(3.0) == HawkesSignal.BURST
    assert get_hawkes_signal(10.0) == HawkesSignal.BURST


# ---------------------------------------------------------------------------
# Tests — HawkesFeatureEngine
# ---------------------------------------------------------------------------

_REQUIRED_FEATURE_KEYS = {
    "hawkes_intensity",
    "hawkes_ratio",
    "hawkes_signal",
    "branching_ratio",
}


def test_feature_engine_returns_dict():
    """
    After feeding enough ticks, update() must return a dict with all
    required keys and sensible types.
    """
    engine = HawkesFeatureEngine(window_seconds=300, refit_every_n_events=100)
    ts_sec = make_hawkes_timestamps(n=200, mu=3.0, alpha=0.6, beta=2.5, seed=5)

    result: dict | None = None
    base_ns = int(1_700_000_000 * 1e9)  # arbitrary epoch anchor
    for i, t in enumerate(ts_sec):
        ts_ns = base_ns + int(t * 1e9)
        result = engine.update(timestamp_ns=ts_ns, price=500.0, qty=100)

    assert result is not None
    assert _REQUIRED_FEATURE_KEYS.issubset(result.keys()), (
        f"Missing keys: {_REQUIRED_FEATURE_KEYS - result.keys()}"
    )
    assert isinstance(result["hawkes_intensity"], float)
    assert isinstance(result["hawkes_ratio"], float)
    assert isinstance(result["hawkes_signal"], str)
    assert isinstance(result["branching_ratio"], float)
    assert result["hawkes_signal"] in {s.value for s in HawkesSignal}


def test_feature_engine_not_fitted_returns_defaults():
    """
    Before 50 events have been ingested, update() must return the safe
    defaults: ratio=1.0, signal="normal".
    """
    engine = HawkesFeatureEngine()

    base_ns = int(1_700_000_000 * 1e9)
    interval_ns = int(0.1 * 1e9)  # 100 ms between ticks

    # Feed only 10 ticks — well below the 50-event threshold.
    result = {}
    for i in range(10):
        ts_ns = base_ns + i * interval_ns
        result = engine.update(timestamp_ns=ts_ns, price=450.0, qty=50)

    assert result["hawkes_ratio"] == pytest.approx(1.0)
    assert result["hawkes_signal"] == HawkesSignal.NORMAL.value
    assert result["hawkes_intensity"] == pytest.approx(0.0)
    assert result["branching_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — comparative log-likelihood
# ---------------------------------------------------------------------------

def test_fit_hawkes_data_better_than_poisson():
    """
    Fitting a Hawkes process on Hawkes-generated data should achieve a strictly
    better (lower negative) log-likelihood than the maximum-likelihood
    Poisson/exponential baseline.

    Poisson ML estimate: μ_hat = n / T (the trivial constant-intensity model).
    Its log-likelihood under the Hawkes parameterisation is evaluated with
    alpha→0 (essentially a pure Poisson process): α=1e-6, β=1.0.

    We check that the Hawkes MLE NLL < Poisson NLL, i.e., Hawkes fits better.
    """
    ts = make_hawkes_timestamps(n=300, mu=2.0, alpha=0.6, beta=2.5, seed=99)

    hp = HawkesProcess()
    hawkes_params = hp.fit(ts)
    hawkes_nll = hp.log_likelihood(ts, hawkes_params)

    # Poisson baseline: rate = n/T, modelled as Hawkes with near-zero alpha.
    T = ts[-1]
    mu_poisson = len(ts) / T
    poisson_params = HawkesParams(mu=mu_poisson, alpha=1e-6, beta=1.0)
    poisson_nll = hp.log_likelihood(ts, poisson_params)

    assert hawkes_nll < poisson_nll, (
        f"Hawkes NLL ({hawkes_nll:.2f}) should be lower than Poisson NLL "
        f"({poisson_nll:.2f}) on Hawkes-generated data"
    )
