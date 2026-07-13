"""Tests for TripleBarrierLabeler (algo_trading/labeling/triple_barrier.py).

All test data is generated in-memory — no file I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from algo_trading.labeling.triple_barrier import BarrierSpec, TripleBarrierLabeler

# ---------------------------------------------------------------------------
# Helper: synthetic OHLCV DataFrame
# ---------------------------------------------------------------------------

def make_ohlcv_df(
    n: int = 500,
    base_price: float = 2800.0,
    atr: float = 15.0,
) -> pl.DataFrame:
    """Create a minimal OHLCV Polars DataFrame for labeler testing.

    Prices follow a simple flat random walk; ATR is constant for easy
    barrier arithmetic.

    Args:
        n: Number of bars.
        base_price: Starting close price.
        atr: Constant ATR value assigned to every row.

    Returns:
        Polars DataFrame with columns:
            timestamp, open, high, low, close, volume, atr14
    """
    rng = np.random.default_rng(0)

    # Timestamps: 15-min bars from an arbitrary start.
    start = datetime(2023, 1, 2, 9, 15)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n)]

    # Close: random walk.
    steps = rng.uniform(-1.0, 1.0, size=n)
    closes = base_price + np.cumsum(steps)
    closes = np.clip(closes, base_price * 0.5, base_price * 2.0)

    # Open: previous close + tiny gap.
    opens = np.empty(n)
    opens[0] = base_price
    opens[1:] = closes[:-1] * (1.0 + rng.uniform(-0.0003, 0.0003, size=n - 1))

    # High / Low: open/close ± half ATR.
    highs = np.maximum(opens, closes) + rng.uniform(0.0, atr * 0.5, size=n)
    lows  = np.minimum(opens, closes) - rng.uniform(0.0, atr * 0.5, size=n)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows,  np.minimum(opens, closes))

    volumes = rng.integers(10_000, 1_000_000, size=n)
    atrs    = np.full(n, atr, dtype=np.float64)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open":      opens.astype(np.float64),
            "high":      highs.astype(np.float64),
            "low":       lows.astype(np.float64),
            "close":     closes.astype(np.float64),
            "volume":    volumes.astype(np.int64),
            "atr14":     atrs,
        },
        schema={
            "timestamp": pl.Datetime,
            "open":      pl.Float64,
            "high":      pl.Float64,
            "low":       pl.Float64,
            "close":     pl.Float64,
            "volume":    pl.Int64,
            "atr14":     pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def labeled_df() -> pl.DataFrame:
    """Return a labeled DataFrame shared across tests that need a valid result."""
    df = make_ohlcv_df(n=500, base_price=2800.0, atr=15.0)
    return TripleBarrierLabeler().label(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLabelOutputColumns:
    """test_label_output_columns — all required output columns are present."""

    EXPECTED_OUTPUT_COLUMNS = {
        "label",
        "upper_barrier",
        "lower_barrier",
        "time_barrier",
        "barrier_hit",
        "sample_weight",
    }

    def test_label_output_columns(self, labeled_df: pl.DataFrame) -> None:
        missing = self.EXPECTED_OUTPUT_COLUMNS - set(labeled_df.columns)
        assert not missing, f"Output is missing columns: {missing}"


class TestLabelValuesValid:
    """test_label_values_valid — every label is in {-1, 0, +1}."""

    def test_label_values_valid(self, labeled_df: pl.DataFrame) -> None:
        unique_labels = set(labeled_df["label"].to_list())
        assert unique_labels.issubset({-1, 0, 1}), (
            f"Unexpected label values found: {unique_labels - {-1, 0, 1}}"
        )


class TestSampleWeightsPositive:
    """test_sample_weights_positive — all weights are strictly > 0."""

    def test_sample_weights_positive(self, labeled_df: pl.DataFrame) -> None:
        weights = labeled_df["sample_weight"].to_list()
        assert all(w > 0 for w in weights), "Some sample weights are not positive."


class TestSampleWeightsMeanOne:
    """test_sample_weights_mean_one — mean weight ≈ 1.0 (±0.01)."""

    def test_sample_weights_mean_one(self, labeled_df: pl.DataFrame) -> None:
        mean_w = labeled_df["sample_weight"].mean()
        assert mean_w is not None
        assert abs(mean_w - 1.0) < 0.01, (
            f"Mean sample weight = {mean_w:.6f}, expected ≈ 1.0 (±0.01)"
        )


class TestNoLookahead:
    """test_no_lookahead — validate_no_lookahead returns True."""

    def test_no_lookahead(self, labeled_df: pl.DataFrame) -> None:
        labeler = TripleBarrierLabeler()
        result = labeler.validate_no_lookahead(labeled_df)
        assert result is True


class TestSymmetricSpecEnforced:
    """test_symmetric_spec_enforced — asymmetric BarrierSpec raises ValueError."""

    def test_symmetric_spec_enforced(self) -> None:
        with pytest.raises(ValueError, match="[Aa]symmetric"):
            BarrierSpec(upper_atr_mult=2.0, lower_atr_mult=1.5)


class TestEmptyDfRaises:
    """test_empty_df_raises — empty DataFrame raises ValueError."""

    def test_empty_df_raises(self) -> None:
        empty_df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "open":      pl.Series([], dtype=pl.Float64),
                "high":      pl.Series([], dtype=pl.Float64),
                "low":       pl.Series([], dtype=pl.Float64),
                "close":     pl.Series([], dtype=pl.Float64),
                "volume":    pl.Series([], dtype=pl.Int64),
                "atr14":     pl.Series([], dtype=pl.Float64),
            }
        )
        with pytest.raises(ValueError, match="[Ee]mpty"):
            TripleBarrierLabeler().label(empty_df)


class TestMissingColumnsRaises:
    """test_missing_columns_raises — DataFrame without atr14 raises ValueError."""

    def test_missing_columns_raises(self) -> None:
        df = make_ohlcv_df(n=50).drop("atr14")
        with pytest.raises(ValueError, match="atr14"):
            TripleBarrierLabeler().label(df)


class TestUpperBarrierCorrect:
    """test_upper_barrier_correct — upper_barrier[i] == close[i] + 1.5 * atr14[i]."""

    def test_upper_barrier_correct(self) -> None:
        atr_val = 15.0
        mult = 1.5
        df = make_ohlcv_df(n=100, base_price=2800.0, atr=atr_val)
        labeled = TripleBarrierLabeler(BarrierSpec(upper_atr_mult=mult, lower_atr_mult=mult)).label(df)

        closes   = labeled["close"].to_list()
        barriers = labeled["upper_barrier"].to_list()
        atrs     = labeled["atr14"].to_list()

        for i, (c, ub, a) in enumerate(zip(closes, barriers, atrs)):
            expected = c + mult * max(a, BarrierSpec().min_atr_floor)
            assert abs(ub - expected) < 1e-9, (
                f"Row {i}: upper_barrier={ub:.6f} != expected={expected:.6f}"
            )


class TestLowerBarrierCorrect:
    """test_lower_barrier_correct — lower_barrier[i] == close[i] - 1.5 * atr14[i]."""

    def test_lower_barrier_correct(self) -> None:
        atr_val = 15.0
        mult = 1.5
        df = make_ohlcv_df(n=100, base_price=2800.0, atr=atr_val)
        labeled = TripleBarrierLabeler(BarrierSpec(upper_atr_mult=mult, lower_atr_mult=mult)).label(df)

        closes   = labeled["close"].to_list()
        barriers = labeled["lower_barrier"].to_list()
        atrs     = labeled["atr14"].to_list()

        for i, (c, lb, a) in enumerate(zip(closes, barriers, atrs)):
            expected = c - mult * max(a, BarrierSpec().min_atr_floor)
            assert abs(lb - expected) < 1e-9, (
                f"Row {i}: lower_barrier={lb:.6f} != expected={expected:.6f}"
            )


class TestLabelDistributionWarns:
    """test_label_distribution_warns — tiny ATR (many 0 labels) doesn't crash."""

    def test_label_distribution_warns(self) -> None:
        # ATR so small barriers are essentially at close; almost all timeout.
        df = make_ohlcv_df(n=200, base_price=2800.0, atr=0.00001)
        labeler = TripleBarrierLabeler()
        labeled = labeler.label(df)

        # Distribution computation must not raise even with extreme imbalance.
        dist = labeler.compute_label_distribution(labeled["label"])
        assert set(dist.keys()) == {-1, 0, 1}
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, f"Distribution doesn't sum to 1.0: {total}"


class TestTrendingUpProducesUpperHits:
    """test_trending_up_produces_upper_hits — strong uptrend → many +1 labels."""

    def test_trending_up_produces_upper_hits(self) -> None:
        # Each bar close moves up by 5 % of ATR so the upper barrier is quickly hit.
        n = 300
        atr = 15.0
        base = 2800.0
        rng_arr = np.arange(n)

        # Strongly upward close series: +2.0 per bar (>> 1.5 * 15 = 22.5 over ~12 bars).
        closes = base + rng_arr * 2.0

        starts = datetime(2023, 1, 2, 9, 15)
        timestamps = [starts + timedelta(minutes=15 * i) for i in range(n)]

        # High = close + 1*ATR (enough to frequently cross upper barrier).
        highs = closes + atr
        lows  = closes - atr * 0.1
        opens = closes - 0.5

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open":      opens.astype(np.float64),
                "high":      highs.astype(np.float64),
                "low":       lows.astype(np.float64),
                "close":     closes.astype(np.float64),
                "volume":    np.ones(n, dtype=np.int64) * 100_000,
                "atr14":     np.full(n, atr),
            },
            schema={
                "timestamp": pl.Datetime,
                "open":      pl.Float64,
                "high":      pl.Float64,
                "low":       pl.Float64,
                "close":     pl.Float64,
                "volume":    pl.Int64,
                "atr14":     pl.Float64,
            },
        )

        labeled = TripleBarrierLabeler().label(df)
        label_counts = labeled["label"].value_counts()
        count_map: dict[int, int] = {
            row["label"]: row["count"]
            for row in label_counts.iter_rows(named=True)
        }
        upper_hits = count_map.get(1, 0)
        total = len(labeled)

        assert upper_hits / total > 0.5, (
            f"Expected > 50 % upper hits in strong uptrend, got {upper_hits}/{total} "
            f"({upper_hits/total:.1%}). Distribution: {count_map}"
        )


class TestTrendingDownProducesLowerHits:
    """test_trending_down_produces_lower_hits — strong downtrend → many -1 labels."""

    def test_trending_down_produces_lower_hits(self) -> None:
        n = 300
        atr = 15.0
        base = 5000.0   # High start so prices stay positive after decline.
        rng_arr = np.arange(n)

        # Strongly downward close series: -2.0 per bar.
        closes = base - rng_arr * 2.0
        closes = np.clip(closes, 1.0, None)

        starts = datetime(2023, 1, 2, 9, 15)
        timestamps = [starts + timedelta(minutes=15 * i) for i in range(n)]

        # Low = close - 1*ATR (enough to frequently cross lower barrier).
        highs = closes + atr * 0.1
        lows  = closes - atr
        opens = closes + 0.5

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open":      opens.astype(np.float64),
                "high":      highs.astype(np.float64),
                "low":       lows.astype(np.float64),
                "close":     closes.astype(np.float64),
                "volume":    np.ones(n, dtype=np.int64) * 100_000,
                "atr14":     np.full(n, atr),
            },
            schema={
                "timestamp": pl.Datetime,
                "open":      pl.Float64,
                "high":      pl.Float64,
                "low":       pl.Float64,
                "close":     pl.Float64,
                "volume":    pl.Int64,
                "atr14":     pl.Float64,
            },
        )

        labeled = TripleBarrierLabeler().label(df)
        label_counts = labeled["label"].value_counts()
        count_map: dict[int, int] = {
            row["label"]: row["count"]
            for row in label_counts.iter_rows(named=True)
        }
        lower_hits = count_map.get(-1, 0)
        total = len(labeled)

        assert lower_hits / total > 0.5, (
            f"Expected > 50 % lower hits in strong downtrend, got {lower_hits}/{total} "
            f"({lower_hits/total:.1%}). Distribution: {count_map}"
        )
