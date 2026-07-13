"""Tests for WalkForwardSplitter (algo_trading/labeling/walk_forward.py).

All test data is generated in-memory — no file I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from algo_trading.labeling.walk_forward import (
    SplitMetadata,
    WalkForwardConfig,
    WalkForwardSplitter,
)

# ---------------------------------------------------------------------------
# Helper: synthetic DataFrame with timestamp + label columns
# ---------------------------------------------------------------------------

def make_large_df(n: int = 5000) -> pl.DataFrame:
    """Return a Polars DataFrame with ``timestamp`` and ``label`` columns.

    Timestamps are spaced 15 minutes apart starting from 2023-01-02 09:15.
    Labels are uniformly random from {-1, 0, 1}.

    Args:
        n: Number of rows.

    Returns:
        Polars DataFrame.
    """
    rng = np.random.default_rng(42)
    start = datetime(2023, 1, 2, 9, 15)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n)]
    labels = rng.choice([-1, 0, 1], size=n).tolist()

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "label":     labels,
        },
        schema={
            "timestamp": pl.Datetime,
            "label":     pl.Int8,
        },
    )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_splits():
    """Return splits produced by the default WalkForwardConfig on a 5000-row df."""
    df = make_large_df(n=5000)
    splitter = WalkForwardSplitter()
    return splitter.split(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNFoldsProduced:
    """test_n_folds_produced — split returns exactly n_folds tuples."""

    def test_n_folds_produced(self, default_splits) -> None:
        assert len(default_splits) == WalkForwardConfig().n_folds


class TestNoTemporalLeakage:
    """test_no_temporal_leakage — validate_no_future_leakage returns True."""

    def test_no_temporal_leakage(self, default_splits) -> None:
        splitter = WalkForwardSplitter()
        result = splitter.validate_no_future_leakage(default_splits)
        assert result is True


class TestTrainBeforeTest:
    """test_train_before_test — max(train.timestamp) < min(test.timestamp) per fold."""

    def test_train_before_test(self, default_splits) -> None:
        for train_df, test_df, meta in default_splits:
            max_train = train_df["timestamp"].max()
            min_test  = test_df["timestamp"].min()
            assert max_train < min_test, (
                f"Fold {meta.fold_index}: max(train.timestamp)={max_train} "
                f">= min(test.timestamp)={min_test}"
            )


class TestPurgeGapRespected:
    """test_purge_gap_respected — test_start - train_end >= purge_bars per fold."""

    def test_purge_gap_respected(self, default_splits) -> None:
        for _train_df, _test_df, meta in default_splits:
            actual_gap = meta.test_start - meta.train_end
            assert actual_gap >= meta.purge_bars, (
                f"Fold {meta.fold_index}: gap={actual_gap} < purge_bars={meta.purge_bars}"
            )


class TestMetadataFields:
    """test_metadata_fields — SplitMetadata has all expected fields."""

    EXPECTED_FIELDS = {
        "fold_index",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "purge_bars",
        "embargo_bars",
        "bar_type",
        "train_size",
        "test_size",
    }

    def test_metadata_fields(self, default_splits) -> None:
        for _train, _test, meta in default_splits:
            actual = set(vars(meta).keys())
            missing = self.EXPECTED_FIELDS - actual
            assert not missing, f"SplitMetadata missing fields: {missing}"


class TestNFoldsLessThan5Raises:
    """test_n_folds_less_than_5_raises — WalkForwardConfig(n_folds=3) raises ValueError."""

    def test_n_folds_less_than_5_raises(self) -> None:
        with pytest.raises(ValueError, match="[Mm]inimum 5|n_folds"):
            WalkForwardConfig(n_folds=3)


class TestDatasetTooSmallRaises:
    """test_dataset_too_small_raises — 500-row df with default config raises ValueError."""

    def test_dataset_too_small_raises(self) -> None:
        df = make_large_df(n=500)
        splitter = WalkForwardSplitter()
        with pytest.raises(ValueError, match="[Tt]oo small|[Dd]ataset"):
            splitter.split(df)


class TestEmbargoBarsZeroRaises:
    """test_embargo_bars_zero_raises — embargo_bars=0 raises ValueError."""

    def test_embargo_bars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="embargo_bars"):
            WalkForwardConfig(embargo_bars=0)


class TestExpandingWindowSplits:
    """test_expanding_window_splits — get_expanding_window_splits returns same count."""

    def test_expanding_window_splits(self) -> None:
        df = make_large_df(n=5000)
        splitter = WalkForwardSplitter()
        rolling_splits  = splitter.split(df)
        expanding_splits = splitter.get_expanding_window_splits(df)
        assert len(expanding_splits) == len(rolling_splits), (
            f"Expanding splits ({len(expanding_splits)}) != "
            f"rolling splits ({len(rolling_splits)})"
        )


class TestFoldTestSizesRoughlyEqual:
    """test_fold_test_sizes_roughly_equal — max(test_size) - min(test_size) <= 1."""

    def test_fold_test_sizes_roughly_equal(self, default_splits) -> None:
        test_sizes = [meta.test_size for _t, _v, meta in default_splits]
        spread = max(test_sizes) - min(test_sizes)
        assert spread <= 1, (
            f"Test fold sizes differ by {spread} (expected ≤ 1): {test_sizes}"
        )


class TestPurgeFloor51:
    """test_purge_floor_51 — purge_bars >= 51 even with short max_feature_lookback."""

    def test_purge_floor_51(self) -> None:
        # max_holding_bars=1, max_feature_lookback_bars=1, embargo_bars=1
        # computed = 1+1+1 = 3, but floor is 51.
        cfg = WalkForwardConfig(
            n_folds=5,
            max_holding_bars=1,
            max_feature_lookback_bars=1,
            embargo_bars=1,
            min_test_trades=1,
        )
        splitter = WalkForwardSplitter(cfg)
        assert splitter.purge_bars >= 51, (
            f"purge_bars={splitter.purge_bars} should be >= 51 (hard floor)"
        )


class TestMinTestTradesEnforced:
    """test_min_test_trades_enforced — very large min_test_trades raises ValueError."""

    def test_min_test_trades_enforced(self) -> None:
        # With n=5000 rows and min_test_trades=5000, each fold would need 5000
        # labeled rows which is impossible.
        df = make_large_df(n=5000)
        cfg = WalkForwardConfig(
            n_folds=5,
            min_test_trades=5000,
        )
        splitter = WalkForwardSplitter(cfg)
        with pytest.raises(ValueError):
            splitter.split(df)
