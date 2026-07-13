"""Stage 2 feature engine tests — shape, type, and no-crash tests.

Uses synthetic Polars DataFrames with Geometric Brownian Motion price paths.
Numerical accuracy is NOT tested here; only output shapes, types, and
absence of crashes / inf values are verified.

Column name expectations follow the actual output of each feature class:
  TechnicalFeatures.compute()  → feat_rsi, feat_macd_hist, feat_bb_pct_b, ...
  StatisticalFeatures.compute() → feat_kalman_price, feat_hmm_regime, feat_garch_vol, ...
  CustomFeatures               → methods encode_time_of_day(), compute_spread_percentile()
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from algo_trading.features.technical import TechnicalFeatures
from algo_trading.features.engine import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Synthetic OHLCV data helper
# ---------------------------------------------------------------------------

def make_sample_ohlcv(
    n: int = 200,
    base_price: float = 2800.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic OHLCV Polars DataFrame using Geometric Brownian Motion.

    Each bar represents a 15-minute interval starting from 2023-01-02 09:15 IST.

    Args:
        n: Number of bars to generate (default 200).
        base_price: Starting close price (default 2800.0).
        seed: Random seed for reproducibility.

    Returns:
        Polars DataFrame with columns: timestamp (Datetime), open, high, low,
        close (Float64), volume (Int64), symbol (Utf8).
    """
    rng = np.random.default_rng(seed)

    # GBM log returns → close prices
    log_returns = rng.normal(0.0001, 0.01, n)
    close = base_price * np.cumprod(1.0 + log_returns)

    # Open = previous close (shift by 1; first bar open = base_price)
    open_ = np.empty(n)
    open_[0] = base_price
    open_[1:] = close[:-1]

    # High = max(open, close) * uniform(1.001, 1.005)
    high_mult = rng.uniform(1.001, 1.005, n)
    high = np.maximum(open_, close) * high_mult

    # Low = min(open, close) * uniform(0.995, 0.999)
    low_mult = rng.uniform(0.995, 0.999, n)
    low = np.minimum(open_, close) * low_mult

    # Volume: uniform integers in [100_000, 2_000_000)
    volume = rng.integers(100_000, 2_000_000, n)

    # Timestamps: 15-minute bars from 2023-01-02 09:15 IST
    # Use naive datetime; TechnicalFeatures handles both aware and naive.
    import datetime
    start = datetime.datetime(2023, 1, 2, 9, 15, 0)
    timestamps = [start + datetime.timedelta(minutes=15 * i) for i in range(n)]

    return pl.DataFrame({
        "timestamp": pl.Series(timestamps).cast(pl.Datetime),
        "open": pl.Series(open_.tolist(), dtype=pl.Float64),
        "high": pl.Series(high.tolist(), dtype=pl.Float64),
        "low": pl.Series(low.tolist(), dtype=pl.Float64),
        "close": pl.Series(close.tolist(), dtype=pl.Float64),
        "volume": pl.Series(volume.tolist(), dtype=pl.Int64),
        "symbol": pl.Series(["RELIANCE"] * n, dtype=pl.Utf8),
    })


# ---------------------------------------------------------------------------
# TechnicalFeatures tests
# ---------------------------------------------------------------------------

# Canonical feature column names as output by TechnicalFeatures.compute()
_TECHNICAL_FEAT_COLS = [
    "feat_rsi",
    "feat_macd_hist",
    "feat_bb_pct_b",
    "feat_bb_width",
    "feat_atr",
    "feat_adx",
    "feat_stoch_k",
    "feat_stoch_d",
    "feat_obv",
    "feat_supertrend_dir",
    "feat_ema_cross",
    "feat_ichimoku_pos",
    "feat_vwap_dev",
    "feat_ha_dir",
    "feat_pivot_dist",
]


@pytest.fixture(scope="module")
def sample_df() -> pl.DataFrame:
    """Module-scoped synthetic OHLCV DataFrame (200 bars)."""
    return make_sample_ohlcv(n=200)


@pytest.fixture(scope="module")
def technical_result(sample_df: pl.DataFrame) -> pl.DataFrame:
    """Module-scoped TechnicalFeatures computation result."""
    tf = TechnicalFeatures()
    return tf.compute(sample_df)


def test_technical_features_returns_polars_df(technical_result: pl.DataFrame) -> None:
    """TechnicalFeatures.compute() must return a Polars DataFrame."""
    assert isinstance(technical_result, pl.DataFrame), (
        f"Expected pl.DataFrame, got {type(technical_result)}"
    )


def test_technical_features_has_required_columns(technical_result: pl.DataFrame) -> None:
    """All 15 technical feature columns must be present in the output."""
    missing = [col for col in _TECHNICAL_FEAT_COLS if col not in technical_result.columns]
    assert not missing, f"Missing technical feature columns: {missing}"


def test_technical_features_row_count_preserved(
    sample_df: pl.DataFrame,
    technical_result: pl.DataFrame,
) -> None:
    """Output DataFrame must have the same number of rows as the input."""
    assert len(technical_result) == len(sample_df), (
        f"Row count changed: input={len(sample_df)}, output={len(technical_result)}"
    )


def test_technical_features_no_all_null_columns(technical_result: pl.DataFrame) -> None:
    """No feature column should be 100% null (would indicate a broken indicator)."""
    for col in _TECHNICAL_FEAT_COLS:
        null_count = technical_result[col].null_count()
        total = len(technical_result)
        assert null_count < total, (
            f"Column '{col}' is entirely null ({null_count}/{total} rows)"
        )


def test_rsi_range(technical_result: pl.DataFrame) -> None:
    """RSI values (where not null) must lie in [0, 100]."""
    rsi = technical_result["feat_rsi"].drop_nulls()
    if len(rsi) == 0:
        pytest.skip("feat_rsi has no non-null values — lookback not met")
    arr = rsi.to_numpy()
    assert arr.min() >= 0.0, f"RSI below 0: min={arr.min()}"
    assert arr.max() <= 100.0, f"RSI above 100: max={arr.max()}"


def test_supertrend_dir_binary(technical_result: pl.DataFrame) -> None:
    """Supertrend direction values (where not null) must be +1 or -1."""
    st = technical_result["feat_supertrend_dir"].drop_nulls()
    if len(st) == 0:
        pytest.skip("feat_supertrend_dir has no non-null values — lookback not met")
    unique_vals = set(st.cast(pl.Int32).to_list())
    assert unique_vals <= {1, -1}, (
        f"Supertrend direction contains unexpected values: {unique_vals}"
    )


# ---------------------------------------------------------------------------
# StatisticalFeatures tests
# ---------------------------------------------------------------------------

try:
    from algo_trading.features.statistical import StatisticalFeatures as _StatFeat  # noqa: F401
    _STAT_AVAILABLE = True
except ImportError:
    _STAT_AVAILABLE = False

_STAT_FEAT_COLS = [
    "feat_kalman_price",
    "feat_hmm_regime",
    "feat_garch_vol",
]


@pytest.fixture(scope="module")
def statistical_result(sample_df: pl.DataFrame) -> pl.DataFrame:
    """Module-scoped StatisticalFeatures computation result."""
    if not _STAT_AVAILABLE:
        pytest.skip("optional dependency missing (filterpy, hmmlearn)")
    try:
        from algo_trading.features.statistical import StatisticalFeatures
        sf = StatisticalFeatures()
        return sf.compute(sample_df)
    except ImportError:
        pytest.skip("optional dependency missing (filterpy, hmmlearn)")


def test_statistical_features_returns_polars_df(statistical_result: pl.DataFrame) -> None:
    """StatisticalFeatures.compute() must return a Polars DataFrame."""
    if not _STAT_AVAILABLE:
        pytest.skip("optional dependency missing")
    assert isinstance(statistical_result, pl.DataFrame), (
        f"Expected pl.DataFrame, got {type(statistical_result)}"
    )


def test_statistical_has_hmm_regime(statistical_result: pl.DataFrame) -> None:
    """feat_hmm_regime column must be present; non-null values must be in {0, 1, 2}."""
    if not _STAT_AVAILABLE:
        pytest.skip("optional dependency missing")
    assert "feat_hmm_regime" in statistical_result.columns, (
        "feat_hmm_regime column not found"
    )
    non_null = statistical_result["feat_hmm_regime"].drop_nulls()
    if len(non_null) > 0:
        # HMM not fitted → all NaN; when fitted values should be 0/1/2
        unique_vals = set(non_null.cast(pl.Int32).to_list())
        assert unique_vals <= {0, 1, 2}, (
            f"hmm_regime values outside {{0,1,2}}: {unique_vals}"
        )


def test_statistical_has_garch_vol(statistical_result: pl.DataFrame) -> None:
    """feat_garch_vol column must be present; non-null values must be > 0."""
    if not _STAT_AVAILABLE:
        pytest.skip("optional dependency missing")
    assert "feat_garch_vol" in statistical_result.columns, (
        "feat_garch_vol column not found"
    )
    non_null = statistical_result["feat_garch_vol"].drop_nulls()
    if len(non_null) > 0:
        arr = non_null.to_numpy()
        # GARCH not fitted → column may be all NaN; skip value check if so
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) > 0:
            assert (finite_vals > 0).all(), (
                f"feat_garch_vol has non-positive values: min={finite_vals.min()}"
            )


def test_statistical_has_kalman_price(statistical_result: pl.DataFrame) -> None:
    """feat_kalman_price column must be present in the output."""
    if not _STAT_AVAILABLE:
        pytest.skip("optional dependency missing")
    assert "feat_kalman_price" in statistical_result.columns, (
        "feat_kalman_price column not found"
    )


# ---------------------------------------------------------------------------
# CustomFeatures tests
# ---------------------------------------------------------------------------

try:
    from algo_trading.features.custom import CustomFeatures as _CustomFeat  # noqa: F401
    _CUSTOM_AVAILABLE = True
except ImportError:
    _CUSTOM_AVAILABLE = False


def _get_custom_features() -> "_CustomFeat":
    """Helper: return a CustomFeatures instance or skip the test."""
    if not _CUSTOM_AVAILABLE:
        pytest.skip("optional dependency missing (numba)")
    try:
        from algo_trading.features.custom import CustomFeatures
        return CustomFeatures(bar_type="volume")
    except ImportError:
        pytest.skip("optional dependency missing (numba)")


def test_custom_features_has_tod_encoding() -> None:
    """encode_time_of_day() must return (sin, cos) both in [-1, +1]."""
    cf = _get_custom_features()

    # Sample several times of day across the NSE session
    test_cases = [
        (9, 15),   # session open
        (12, 0),   # midday
        (15, 30),  # session close
    ]
    for hour, minute in test_cases:
        sin_val, cos_val = cf.encode_time_of_day(hour, minute)
        assert -1.0 <= sin_val <= 1.0, (
            f"tod_sin out of [-1,1] at {hour}:{minute:02d}: {sin_val}"
        )
        assert -1.0 <= cos_val <= 1.0, (
            f"tod_cos out of [-1,1] at {hour}:{minute:02d}: {cos_val}"
        )


def test_custom_features_spread_percentile() -> None:
    """compute_spread_percentile() must return a value in [0, 1]."""
    cf = _get_custom_features()
    rng = np.random.default_rng(99)

    for _ in range(20):
        current_spread = float(rng.uniform(0.05, 1.5))
        recent_spreads = rng.uniform(0.05, 1.5, 20).astype("float64")
        result = cf.compute_spread_percentile(current_spread, recent_spreads)
        assert 0.0 <= result <= 1.0, (
            f"spread_percentile out of [0,1]: {result} "
            f"(current={current_spread:.4f})"
        )


# ---------------------------------------------------------------------------
# General cross-feature tests
# ---------------------------------------------------------------------------

def test_no_inf_values_in_any_feature(technical_result: pl.DataFrame) -> None:
    """No technical feature column should contain infinity values."""
    for col in _TECHNICAL_FEAT_COLS:
        series = technical_result[col]
        # Cast to float and check for inf
        arr = series.cast(pl.Float64).to_numpy()
        has_inf = np.any(np.isinf(arr[~np.isnan(arr)]))
        assert not has_inf, (
            f"Column '{col}' contains infinity values"
        )


def test_feature_names_list_length() -> None:
    """FEATURE_NAMES from engine.py must contain exactly 25 entries."""
    assert len(FEATURE_NAMES) == 25, (
        f"Expected 25 feature names, got {len(FEATURE_NAMES)}: {FEATURE_NAMES}"
    )
