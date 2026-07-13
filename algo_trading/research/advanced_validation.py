"""algo_trading/research/advanced_validation.py — Advanced backtest validation.

Three complementary techniques guard against the most common failure modes of
quantitative strategy research on NSE data:

**Deflated Sharpe Ratio (DSR)**
    NSE F&O strategy research almost always involves iterating over many
    parameter configurations (strike selection, holding period, entry filter
    thresholds).  Each IS iteration that is discarded increases the probability
    that the one retained configuration looks good purely by chance.  DSR
    (Bailey & López de Prado 2014) penalises the observed Sharpe for (a) the
    number of IS trials conducted, (b) fat tails and negative skewness in
    returns, and (c) short sample length.  A strategy that only survives DSR
    has a signal that is robust to these three inflation factors.

**Probability of Backtest Overfitting (PBO)**
    PBO (Bailey et al. 2014) answers: "If we had selected our best IS
    configuration on a different historical sub-period, what fraction of the
    time would it underperform the median OOS?"  Values above 0.1 (10%) are a
    strong warning that the strategy would not survive a regime change — which
    is especially relevant for NSE F&O given structural market shifts (margin
    reforms, new instruments, liquidity cycles).

**Combinatorial Purged Cross-Validation (CPCV)**
    Walk-forward gives a single equity curve and therefore a single Sharpe
    estimate.  CPCV (López de Prado 2018, AFML ch. 12) generates
    C(n_splits, n_test_splits) combinatorially distinct equity curves.  The
    *distribution* of Sharpes — and in particular the minimum — provides a
    far more conservative and honest estimate of live-trading performance than
    any single walk-forward result.  The purge+embargo logic mirrors that of
    :class:`~algo_trading.labeling.walk_forward.WalkForwardSplitter`.

All three methods operate on ``np.ndarray`` inputs so they compose cleanly
with the rest of the research pipeline without introducing sklearn or polars
dependencies.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import structlog
from scipy.stats import norm

log = structlog.get_logger(__name__)

# Euler–Mascheroni constant used in the expected-maximum-SR formula.
_EULER_GAMMA: float = 0.5772156649015328


# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


@dataclass
class DeflatedSharpeResult:
    """Output of :func:`deflated_sharpe_ratio`.

    Attributes:
        observed_sharpe: Annualised Sharpe ratio as reported (not deflated).
        deflated_sharpe: SR after adjustment for trials, non-normality, and
            sample length.  The strategy passes if this is >= 0.
        pbo_probability: Approximate probability that the observed SR is a
            false discovery due to multiple testing, derived from the
            selection-bias correction term.  Range [0, 1].
        skewness: Third standardised moment of the daily returns series.
        kurtosis: Fourth standardised moment (excess kurtosis relative to
            normal = kurtosis - 3).
        n_observations: Number of daily return observations used.
        n_trials: Number of IS parameter configurations tried before the
            current one was selected.
        passes_threshold: ``True`` if ``deflated_sharpe >= 0.0``.
    """

    observed_sharpe: float
    deflated_sharpe: float
    pbo_probability: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    passes_threshold: bool


def deflated_sharpe_ratio(
    returns: np.ndarray,
    sharpe_observed: float,
    n_trials: int = 1,
    annual_factor: float = 252.0,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio (DSR).

    Corrects the observed Sharpe ratio for three biases simultaneously:

    1. **Multiple testing** — more IS trials raise the expected maximum SR
       under the null, so the same observed SR is less impressive.
    2. **Non-normality** — fat tails (positive excess kurtosis) and negative
       skewness both reduce the probability of the true SR being positive.
    3. **Short sample** — SR estimates have higher variance when N is small,
       making large observed values less trustworthy.

    The implementation follows the closed-form approximation in Bailey &
    López de Prado (2014), equations (2)–(5):

    * Variance of the SR estimator::

          Var[SR_hat] ≈ (1/T) * (1 - skew*SR + (kurtosis-1)/4 * SR²)

    * Expected maximum SR across *n_trials* independent trials::

          E[max SR] ≈ (1 - γ) * Φ⁻¹(1 - 1/n) + γ * Φ⁻¹(1 - 1/(n·e))

      where γ is the Euler–Mascheroni constant and Φ⁻¹ is the standard normal
      quantile function.

    * DSR::

          DSR = Φ((SR_obs - E[max SR]) / sqrt(Var[SR_hat]))

    A DSR >= 0.5 means the observed SR exceeds the expected maximum SR under
    the null, indicating the result is statistically meaningful.  We use the
    more conservative threshold DSR > 0 (i.e., the observed SR exceeds the
    expected noise level in the direction of the alternative).

    Args:
        returns: 1-D array of daily returns (not annualised; dimensionless
            fractions or P&L / capital).
        sharpe_observed: Annualised Sharpe ratio already computed from *returns*.
            Must be consistent with ``annual_factor``.
        n_trials: Total number of IS configurations evaluated (including the
            one being reported).  Must be >= 1.  Higher values increase the
            multiple-testing penalty.
        annual_factor: Trading days per year.  252 for NSE equity/F&O.

    Returns:
        :class:`DeflatedSharpeResult` with all diagnostic fields populated.

    Raises:
        ValueError: If *returns* has fewer than 5 observations or *n_trials* < 1.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 5:
        raise ValueError(
            f"deflated_sharpe_ratio requires at least 5 return observations, got {n}."
        )
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

    # De-annualise SR back to per-period for the variance formula.
    sr_daily = sharpe_observed / math.sqrt(annual_factor)

    # Third and fourth standardised moments.
    skew = float(np.mean(((returns - returns.mean()) / (returns.std(ddof=1) + 1e-15)) ** 3))
    kurt = float(np.mean(((returns - returns.mean()) / (returns.std(ddof=1) + 1e-15)) ** 4))
    excess_kurtosis = kurt - 3.0

    # Variance of the SR estimator (Bailey & López de Prado 2014, eq. 2).
    # Uses excess kurtosis directly: (kurtosis - 1)/4 = (excess_kurtosis + 2)/4.
    sr_variance = (1.0 / n) * (
        1.0
        - skew * sr_daily
        + ((excess_kurtosis + 3.0 - 1.0) / 4.0) * sr_daily ** 2
    )
    sr_std = math.sqrt(max(sr_variance, 1e-15))

    # Expected maximum SR under the null for n_trials independent strategies
    # (Bailey & López de Prado 2014, eq. 4).
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max_sr = (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2
        # expected_max_sr is in daily units; scale to match sr_daily.
        # (The quantile formula already operates in standardised SR space.)

    # DSR: probability that observed SR exceeds the expected maximum under null.
    # We normalise against sr_std so the result is a z-score transformed to [0,1].
    dsr_z = (sr_daily - expected_max_sr) / sr_std
    dsr_prob = float(norm.cdf(dsr_z))  # = Φ(DSR z-score)

    # For a scalar "deflated Sharpe" value (annualised), scale back.
    deflated_sr = dsr_prob * sharpe_observed  # convention: prob × observed SR

    # pbo_probability: 1 - DSR (fraction explained by chance).
    pbo_prob = 1.0 - dsr_prob

    log.debug(
        "deflated_sharpe_ratio",
        observed_sharpe=round(sharpe_observed, 4),
        deflated_sharpe=round(deflated_sr, 4),
        dsr_z=round(dsr_z, 4),
        pbo_probability=round(pbo_prob, 4),
        n_observations=n,
        n_trials=n_trials,
        skewness=round(skew, 4),
        excess_kurtosis=round(excess_kurtosis, 4),
    )

    return DeflatedSharpeResult(
        observed_sharpe=float(sharpe_observed),
        deflated_sharpe=deflated_sr,
        pbo_probability=pbo_prob,
        skewness=skew,
        kurtosis=kurt,
        n_observations=n,
        n_trials=n_trials,
        passes_threshold=deflated_sr >= 0.0,
    )


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting (PBO)
# ---------------------------------------------------------------------------


@dataclass
class PBOResult:
    """Output of :func:`probability_of_backtest_overfitting`.

    Attributes:
        pbo: Estimated probability of backtest overfitting.  Range [0, 1].
            Values below 0.10 indicate the best IS configuration is likely to
            outperform the OOS median in live trading.
        logit_values: Per-split logit of the relative OOS rank of the best IS
            configuration.  Negative values indicate underperformance.
        median_logit: Median of *logit_values*.  Negative implies the typical
            best-IS config ranks below the median OOS.
        passes_threshold: ``True`` if ``pbo < 0.10``.
    """

    pbo: float
    logit_values: list[float]
    median_logit: float
    passes_threshold: bool


def probability_of_backtest_overfitting(
    is_performance_matrix: np.ndarray,
    oos_performance_matrix: np.ndarray,
    metric_is_higher_better: bool = True,
) -> PBOResult:
    """Estimate the Probability of Backtest Overfitting (PBO).

    Uses combinatorial cross-validation to answer: "Across all possible ways
    of splitting the sub-periods into IS and OOS halves, what fraction of the
    time does the best IS configuration underperform the median OOS config?"

    **Algorithm (Bailey et al. 2014, Section 3)**:

    1. Let *S* be the number of sub-periods (columns of the matrices).
    2. For each combination of *S//2* columns as IS (and the remainder as OOS):
       a. Compute mean IS performance for each configuration (row).
       b. Identify the configuration *c** with the best IS mean.
       c. Compute mean OOS performance for each configuration.
       d. Rank *c** among all configurations by OOS mean.
       e. Record ``λ = logit(rank / n_configs)`` where a negative λ means
          *c** is below-median OOS.
    3. PBO = fraction of combinations where λ < 0 (i.e., rank < n_configs/2).

    For large matrices (more than C(20,10) = 184,756 combinations) the
    function samples 1 000 random splits instead of exhaustive enumeration.

    Args:
        is_performance_matrix: Shape ``(n_configs, n_subperiods)``.  Each row
            is a strategy configuration; each column is a sub-period.  Values
            are any performance metric (Sharpe, return, etc.).
        oos_performance_matrix: Same shape as *is_performance_matrix*.  Must
            correspond to the same configurations and sub-periods.
        metric_is_higher_better: ``True`` if higher metric values are better
            (e.g. Sharpe), ``False`` for metrics like drawdown.

    Returns:
        :class:`PBOResult` with PBO and logit distribution.

    Raises:
        ValueError: If the two matrices have different shapes, or if there are
            fewer than 2 configurations or fewer than 4 sub-periods.
    """
    is_mat = np.asarray(is_performance_matrix, dtype=float)
    oos_mat = np.asarray(oos_performance_matrix, dtype=float)

    if is_mat.shape != oos_mat.shape:
        raise ValueError(
            f"is_performance_matrix shape {is_mat.shape} != "
            f"oos_performance_matrix shape {oos_mat.shape}."
        )

    n_configs, n_subperiods = is_mat.shape

    if n_configs < 2:
        raise ValueError(
            f"PBO requires at least 2 strategy configurations, got {n_configs}."
        )
    if n_subperiods < 4:
        raise ValueError(
            f"PBO requires at least 4 sub-periods for meaningful splits, got {n_subperiods}."
        )

    n_is = n_subperiods // 2
    all_indices = list(range(n_subperiods))

    # Decide enumeration vs sampling.
    n_combos = math.comb(n_subperiods, n_is)
    _MAX_EXHAUSTIVE = 1_000
    _N_SAMPLES = 1_000

    rng = np.random.default_rng(42)

    if n_combos <= _MAX_EXHAUSTIVE:
        split_generator = itertools.combinations(all_indices, n_is)
        n_evaluated = n_combos
    else:
        # Random sampling of IS column subsets.
        def _random_splits():
            for _ in range(_N_SAMPLES):
                chosen = sorted(
                    rng.choice(n_subperiods, size=n_is, replace=False).tolist()
                )
                yield tuple(chosen)

        split_generator = _random_splits()  # type: ignore[assignment]
        n_evaluated = _N_SAMPLES

    logit_values: list[float] = []
    overfitted_count = 0

    for is_cols in split_generator:
        oos_cols = [c for c in all_indices if c not in is_cols]

        is_means = is_mat[:, list(is_cols)].mean(axis=1)
        oos_means = oos_mat[:, oos_cols].mean(axis=1)

        # Best IS config index.
        best_is_idx = int(
            np.argmax(is_means) if metric_is_higher_better else np.argmin(is_means)
        )

        # Rank of the best IS config in OOS (1 = best OOS, n_configs = worst OOS).
        if metric_is_higher_better:
            oos_ranks = n_configs - np.argsort(np.argsort(oos_means))
        else:
            oos_ranks = np.argsort(np.argsort(oos_means)) + 1

        oos_rank = int(oos_ranks[best_is_idx])  # 1-indexed, 1 = best OOS

        # Relative rank in (0, 1): 1 = best OOS, approaches 0 = worst OOS.
        relative_rank = oos_rank / n_configs

        # logit(relative_rank): negative means below-median OOS.
        # Clamp away from 0/1 to avoid ±inf.
        relative_rank_clamped = float(np.clip(relative_rank, 1e-6, 1.0 - 1e-6))
        lam = math.log(relative_rank_clamped / (1.0 - relative_rank_clamped))
        logit_values.append(lam)

        if lam < 0.0:
            overfitted_count += 1

    pbo = overfitted_count / max(len(logit_values), 1)
    median_logit = float(np.median(logit_values)) if logit_values else 0.0

    log.debug(
        "probability_of_backtest_overfitting",
        pbo=round(pbo, 4),
        median_logit=round(median_logit, 4),
        n_splits_evaluated=n_evaluated,
        n_configs=n_configs,
        n_subperiods=n_subperiods,
    )

    return PBOResult(
        pbo=pbo,
        logit_values=logit_values,
        median_logit=median_logit,
        passes_threshold=pbo < 0.10,
    )


# ---------------------------------------------------------------------------
# 3. Combinatorial Purged Cross-Validation (CPCV)
# ---------------------------------------------------------------------------


@dataclass
class CPCVFold:
    """A single train/test split produced by CPCV.

    Attributes:
        train_indices: 1-D integer array of row indices for training.
        test_indices: 1-D integer array of row indices for testing.
        fold_id: Zero-based sequential identifier for this fold.
        embargo_size: Number of rows embargoed immediately after each
            training block to prevent autocorrelation leakage.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    embargo_size: int


@dataclass
class CPCVResult:
    """Aggregated output of :func:`run_cpcv_backtest`.

    Attributes:
        n_folds: Total number of CPCV folds generated (= C(n_splits, n_test_splits)).
        n_test_paths: Number of combinatorially distinct OOS equity paths.
            Equal to n_folds for independent path counting; in practice equals
            C(n_splits, n_test_splits).
        per_path_sharpe: Annualised Sharpe ratio evaluated on each OOS path.
        mean_sharpe: Mean of *per_path_sharpe*.
        std_sharpe: Standard deviation of *per_path_sharpe* (path-dependence
            indicator; high std means the strategy is fragile to period selection).
        min_sharpe: Worst-case OOS Sharpe across all paths — the pessimistic
            estimate used for go/no-go decisions.
        passes_threshold: ``True`` if ``min_sharpe >= 0.0`` (even the worst
            combinatorial path is non-negative).
    """

    n_folds: int
    n_test_paths: int
    per_path_sharpe: list[float]
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    passes_threshold: bool


def combinatorial_purged_cv(
    n_samples: int,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_size: int = 5,
) -> list[CPCVFold]:
    """Generate all CPCV train/test index arrays.

    Divides ``n_samples`` rows into ``n_splits`` equal-width groups.  For each
    combination of ``n_test_splits`` groups chosen as test, the remaining
    ``n_splits - n_test_splits`` groups form the training set — with an embargo
    gap of ``embargo_size`` rows removed from the **start** of each training
    block that immediately follows a test block.

    **Why CPCV over standard walk-forward**:

    * Walk-forward produces one equity curve and one Sharpe estimate.
    * CPCV with default params produces C(6, 2) = 15 equity curves.
    * The *distribution* of those 15 Sharpes reveals path-dependence and
      gives a far more realistic picture of live-trading variance.
    * The minimum Sharpe is the conservative estimate used in go/no-go gates.

    The purge/embargo design mirrors
    :class:`~algo_trading.labeling.walk_forward.WalkForwardSplitter`:
    ``embargo_size`` rows immediately after each training block are excluded
    from training to absorb autocorrelation and label-lookahead leakage.

    Args:
        n_samples: Total number of observations (bars) in the dataset.
        n_splits: Total number of equal-width groups.  C(n_splits, n_test_splits)
            folds are generated.  Default 6 gives 15 folds with n_test_splits=2.
        n_test_splits: Number of groups held out for testing per combination.
            Must satisfy 1 <= n_test_splits < n_splits.
        embargo_size: Rows to drop from the start of each training block that
            immediately follows a test block.  5 bars at 15-min resolution
            corresponds to 75 minutes of embargo.

    Returns:
        List of :class:`CPCVFold` objects, one per C(n_splits, n_test_splits)
        combination, ordered by combination enumeration.

    Raises:
        ValueError: If parameter constraints are violated or ``n_samples`` is
            too small to form meaningful folds.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if not (1 <= n_test_splits < n_splits):
        raise ValueError(
            f"n_test_splits must satisfy 1 <= n_test_splits < n_splits, "
            f"got n_test_splits={n_test_splits}, n_splits={n_splits}."
        )
    if embargo_size < 0:
        raise ValueError(f"embargo_size must be >= 0, got {embargo_size}.")

    group_size = n_samples // n_splits
    if group_size < embargo_size + 1:
        raise ValueError(
            f"group_size={group_size} is too small for embargo_size={embargo_size}.  "
            "Use more data or reduce n_splits / embargo_size."
        )

    # Build group boundary indices [start, end) for each of the n_splits groups.
    group_bounds: list[tuple[int, int]] = []
    for g in range(n_splits):
        start = g * group_size
        # Last group absorbs any remainder rows.
        end = (g + 1) * group_size if g < n_splits - 1 else n_samples
        group_bounds.append((start, end))

    folds: list[CPCVFold] = []
    all_group_ids = list(range(n_splits))

    for fold_id, test_group_combo in enumerate(
        itertools.combinations(all_group_ids, n_test_splits)
    ):
        test_group_set = set(test_group_combo)
        train_group_ids = [g for g in all_group_ids if g not in test_group_set]

        # Build test indices: all rows in the selected test groups.
        test_idx_list: list[int] = []
        for g in sorted(test_group_combo):
            s, e = group_bounds[g]
            test_idx_list.extend(range(s, e))

        # Build training indices with embargo applied.
        # Embargo: if a training group immediately follows a test group, skip
        # the first `embargo_size` rows of that training group.
        train_idx_list: list[int] = []
        for g in train_group_ids:
            s, e = group_bounds[g]
            # Check if the preceding group (g-1) is a test group.
            preceding_is_test = (g - 1) in test_group_set
            effective_start = s + embargo_size if preceding_is_test else s
            if effective_start < e:
                train_idx_list.extend(range(effective_start, e))

        folds.append(
            CPCVFold(
                train_indices=np.array(train_idx_list, dtype=np.intp),
                test_indices=np.array(test_idx_list, dtype=np.intp),
                fold_id=fold_id,
                embargo_size=embargo_size,
            )
        )

    log.debug(
        "combinatorial_purged_cv",
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        n_folds=len(folds),
        embargo_size=embargo_size,
        n_samples=n_samples,
    )
    return folds


def run_cpcv_backtest(
    features: np.ndarray,
    labels: np.ndarray,
    model_factory,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_size: int = 5,
) -> CPCVResult:
    """Train and evaluate a model on all CPCV folds, returning a Sharpe distribution.

    For each fold generated by :func:`combinatorial_purged_cv`:

    1. Fit ``model_factory()`` on the training rows.
    2. Predict on the test rows.
    3. Compute a daily return proxy: ``sign(prediction) * actual_return``.
    4. Compute the annualised Sharpe ratio for that OOS path.

    The resulting distribution of Sharpes across all C(n_splits, n_test_splits)
    paths reveals whether the strategy is path-dependent.  The ``min_sharpe``
    field is the conservative go/no-go gate.

    Args:
        features: 2-D float array, shape ``(n_samples, n_features)``.  Must be
            chronologically ordered.
        labels: 1-D float array of target values, shape ``(n_samples,)``.
            These are treated as forward returns (or directional labels as
            +1/-1).  Non-finite values are handled by masking.
        model_factory: Zero-argument callable that returns a fresh sklearn-
            compatible estimator (with ``.fit(X, y)`` and ``.predict(X)``).
            Called once per fold to prevent any state leakage between folds.
        n_splits: Passed directly to :func:`combinatorial_purged_cv`.
        n_test_splits: Passed directly to :func:`combinatorial_purged_cv`.
        embargo_size: Passed directly to :func:`combinatorial_purged_cv`.

    Returns:
        :class:`CPCVResult` with the full distribution of per-path Sharpe ratios.

    Raises:
        ValueError: If *features* and *labels* have incompatible lengths, or if
            any fold produces fewer than 2 finite test observations.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    if len(features) != len(labels):
        raise ValueError(
            f"features length {len(features)} != labels length {len(labels)}."
        )

    folds = combinatorial_purged_cv(
        n_samples=len(features),
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_size=embargo_size,
    )

    per_path_sharpe: list[float] = []

    for fold in folds:
        tr_idx = fold.train_indices
        te_idx = fold.test_indices

        X_train, y_train = features[tr_idx], labels[tr_idx]
        X_test, y_test = features[te_idx], labels[te_idx]

        # Filter non-finite labels from training set.
        valid_train = np.isfinite(y_train)
        if valid_train.sum() < 2:
            log.warning(
                "cpcv_fold_skipped_insufficient_train",
                fold_id=fold.fold_id,
                valid_train=int(valid_train.sum()),
            )
            per_path_sharpe.append(float("nan"))
            continue

        model = model_factory()
        model.fit(X_train[valid_train], y_train[valid_train])
        preds = model.predict(X_test)

        # Daily return proxy: direction of prediction × realised label.
        signs = np.sign(preds)
        oos_returns = signs * y_test

        valid_oos = np.isfinite(oos_returns)
        if valid_oos.sum() < 2:
            log.warning(
                "cpcv_fold_skipped_insufficient_oos",
                fold_id=fold.fold_id,
                valid_oos=int(valid_oos.sum()),
            )
            per_path_sharpe.append(float("nan"))
            continue

        r = oos_returns[valid_oos]
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        if std_r < 1e-10:
            sharpe = 0.0
        else:
            sharpe = float(math.sqrt(252.0) * mean_r / std_r)

        per_path_sharpe.append(sharpe)

        log.debug(
            "cpcv_fold_complete",
            fold_id=fold.fold_id,
            n_train=int(valid_train.sum()),
            n_test=int(valid_oos.sum()),
            sharpe=round(sharpe, 4),
        )

    # Summary statistics over finite paths only.
    finite_sharpes = [s for s in per_path_sharpe if math.isfinite(s)]
    n_finite = len(finite_sharpes)

    if n_finite == 0:
        mean_s = std_s = min_s = 0.0
    elif n_finite == 1:
        mean_s = finite_sharpes[0]
        std_s = 0.0
        min_s = finite_sharpes[0]
    else:
        mean_s = float(np.mean(finite_sharpes))
        std_s = float(np.std(finite_sharpes, ddof=1))
        min_s = float(np.min(finite_sharpes))

    log.info(
        "run_cpcv_backtest_complete",
        n_folds=len(folds),
        n_finite_paths=n_finite,
        mean_sharpe=round(mean_s, 4),
        std_sharpe=round(std_s, 4),
        min_sharpe=round(min_s, 4),
        passes_threshold=min_s >= 0.0,
    )

    return CPCVResult(
        n_folds=len(folds),
        n_test_paths=n_finite,
        per_path_sharpe=per_path_sharpe,
        mean_sharpe=mean_s,
        std_sharpe=std_s,
        min_sharpe=min_s,
        passes_threshold=min_s >= 0.0,
    )
