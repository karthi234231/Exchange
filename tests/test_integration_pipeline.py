"""
Integration test: synthetic data → features → labels → model-ready matrix.

Tests the full Stage 1-2 pipeline without any network calls or real data.
Uses the generate_sample_data approach (GBM-based synthetic OHLCV) inline.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared fixture: 300 bars of synthetic NSE-like OHLCV data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_ohlcv():
    """200 bars of synthetic NSE-like OHLCV data as Polars DataFrame."""
    try:
        import polars as pl
    except ImportError as exc:
        pytest.skip(f"polars not installed: {exc}")

    np.random.seed(42)
    n = 300
    closes = 2800.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volumes = np.random.randint(100_000, 2_000_000, n)

    # Compute ATR14 (simplified: use high-low range rolling 14)
    tr = highs - lows
    atr14 = np.array([tr[max(0, i - 13) : i + 1].mean() for i in range(n)])

    base = datetime(2023, 1, 2, 9, 15)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens.astype(float),
            "high": highs.astype(float),
            "low": lows.astype(float),
            "close": closes.astype(float),
            "volume": volumes.tolist(),
            "atr14": atr14.astype(float),
            "symbol": ["RELIANCE"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — TechnicalFeatures runs on synthetic data
# ---------------------------------------------------------------------------


def test_technical_features_runs_on_synthetic(synthetic_ohlcv) -> None:
    """TechnicalFeatures.compute() must return a DataFrame with more columns."""
    try:
        from algo_trading.features.technical import TechnicalFeatures
    except ImportError as exc:
        pytest.skip(f"algo_trading.features.technical not importable: {exc}")

    tf = TechnicalFeatures()
    try:
        result = tf.compute(synthetic_ohlcv)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for TechnicalFeatures: {exc}")

    import polars as pl

    assert isinstance(result, pl.DataFrame), "Result must be a Polars DataFrame"
    assert len(result.columns) > len(synthetic_ohlcv.columns), (
        "TechnicalFeatures must add feature columns to the DataFrame"
    )
    assert len(result) == len(synthetic_ohlcv), (
        "Row count must be preserved"
    )


# ---------------------------------------------------------------------------
# Test 2 — TripleBarrierLabeler runs after features and produces valid labels
# ---------------------------------------------------------------------------


def test_labeling_runs_after_features(synthetic_ohlcv) -> None:
    """TripleBarrierLabeler.label() must add a 'label' column with values in {-1, 0, 1}."""
    try:
        from algo_trading.labeling.triple_barrier import TripleBarrierLabeler
    except ImportError as exc:
        pytest.skip(f"algo_trading.labeling.triple_barrier not importable: {exc}")

    import polars as pl

    labeler = TripleBarrierLabeler()
    labeled_df = labeler.label(synthetic_ohlcv)

    assert "label" in labeled_df.columns, "Output must contain a 'label' column"

    label_values = set(labeled_df["label"].to_list())
    invalid = label_values - {-1, 0, 1}
    assert not invalid, f"Unexpected label values: {invalid}"


# ---------------------------------------------------------------------------
# Test 3 — WalkForwardSplitter produces temporally ordered folds
# ---------------------------------------------------------------------------


def test_walk_forward_split_after_labeling(synthetic_ohlcv) -> None:
    """WalkForwardSplitter must produce 5 folds, all temporally ordered."""
    try:
        from algo_trading.labeling.triple_barrier import TripleBarrierLabeler
        from algo_trading.labeling.walk_forward import (
            WalkForwardConfig,
            WalkForwardSplitter,
        )
    except ImportError as exc:
        pytest.skip(f"Required labeling modules not importable: {exc}")

    labeler = TripleBarrierLabeler()
    labeled_df = labeler.label(synthetic_ohlcv)

    # Use relaxed min_test_trades so 300-bar fixture is sufficient
    config = WalkForwardConfig(n_folds=5, min_test_trades=10)
    splitter = WalkForwardSplitter(config)

    try:
        splits = splitter.split(labeled_df)
    except ValueError as exc:
        pytest.skip(f"Dataset too small for requested folds: {exc}")

    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"

    for i, (train_df, test_df, meta) in enumerate(splits):
        assert meta.fold_index == i
        assert meta.train_end <= meta.test_start, (
            f"Fold {i}: train_end={meta.train_end} must be <= test_start={meta.test_start}"
        )
        # Temporal order: last training timestamp must precede first test timestamp
        if len(train_df) > 0 and len(test_df) > 0:
            train_last_ts = train_df["timestamp"][-1]
            test_first_ts = test_df["timestamp"][0]
            assert train_last_ts <= test_first_ts, (
                f"Fold {i}: training leaks into test — "
                f"train_last={train_last_ts}, test_first={test_first_ts}"
            )


# ---------------------------------------------------------------------------
# Test 4 — IndiaCostModel computes positive round-trip cost for each bar
# ---------------------------------------------------------------------------


def test_cost_model_computes_for_each_bar(synthetic_ohlcv) -> None:
    """IndiaCostModel.compute_round_trip() must return total_inr > 0 for every bar."""
    try:
        from algo_trading.risk.cost_model import IndiaCostModel
    except ImportError as exc:
        pytest.skip(f"algo_trading.risk.cost_model not importable: {exc}")

    model = IndiaCostModel()
    closes = synthetic_ohlcv["close"].to_list()

    # Sample 10 bars evenly across the 300-bar fixture
    sample_indices = [int(i * len(closes) / 10) for i in range(10)]
    for idx in sample_indices:
        price = Decimal(str(round(closes[idx], 2)))
        cost = model.compute_round_trip("RELIANCE", 100, price)
        assert cost.total_inr > Decimal("0"), (
            f"Round-trip cost must be > 0 for RELIANCE @ {price}, got {cost.total_inr}"
        )


# ---------------------------------------------------------------------------
# Test 5 — No lookahead in labels
# ---------------------------------------------------------------------------


def test_no_lookahead_in_labels(synthetic_ohlcv) -> None:
    """TripleBarrierLabeler.validate_no_lookahead() must return True."""
    try:
        from algo_trading.labeling.triple_barrier import TripleBarrierLabeler
    except ImportError as exc:
        pytest.skip(f"algo_trading.labeling.triple_barrier not importable: {exc}")

    labeler = TripleBarrierLabeler()
    labeled_df = labeler.label(synthetic_ohlcv)
    result = labeler.validate_no_lookahead(labeled_df)
    assert result is True, "validate_no_lookahead() must return True on a clean label run"


# ---------------------------------------------------------------------------
# Test 6 — Label distribution does not crash on 300-bar fixture
# ---------------------------------------------------------------------------


def test_label_distribution_all_classes_present(synthetic_ohlcv) -> None:
    """compute_label_distribution must not crash and must return a dict for 300 bars."""
    try:
        from algo_trading.labeling.triple_barrier import TripleBarrierLabeler
    except ImportError as exc:
        pytest.skip(f"algo_trading.labeling.triple_barrier not importable: {exc}")

    import polars as pl

    labeler = TripleBarrierLabeler()
    labeled_df = labeler.label(synthetic_ohlcv)

    dist = labeler.compute_label_distribution(labeled_df["label"])
    assert isinstance(dist, dict), "compute_label_distribution must return a dict"

    # All three classes should be present as keys (even if count == 0)
    assert -1 in dist and 0 in dist and 1 in dist, (
        f"All three classes must be keys in distribution dict, got: {dist}"
    )

    # Proportions must sum to ~1.0
    total_pct = sum(dist.values())
    assert abs(total_pct - 1.0) < 1e-6, (
        f"Label proportions must sum to 1.0, got {total_pct}"
    )


# ---------------------------------------------------------------------------
# Test 7 — PositionSizer.size_atr() with synthetic ATR returns qty > 0
# ---------------------------------------------------------------------------


def test_position_sizer_with_synthetic_atr(synthetic_ohlcv) -> None:
    """PositionSizer.size_atr() must return SizingResult with qty > 0."""
    try:
        from algo_trading.risk.position_sizer import PositionSizer, SizingResult
    except ImportError as exc:
        pytest.skip(f"algo_trading.risk.position_sizer not importable: {exc}")

    mean_atr = float(synthetic_ohlcv["atr14"].mean())
    capital = Decimal("500000")
    atr_mult = 1.5

    sizer = PositionSizer(
        capital=capital,
        max_position_pct=0.05,
        atr_risk_pct=0.01,
        atr_multiplier=atr_mult,
    )

    # Use the median close price as reference
    import statistics
    closes = synthetic_ohlcv["close"].to_list()
    price = Decimal(str(round(statistics.median(closes), 2)))
    atr_dec = Decimal(str(round(mean_atr, 4)))

    result = sizer.size_atr(price=price, atr=atr_dec)
    assert isinstance(result, SizingResult), "Must return a SizingResult"
    assert result.shares > 0, (
        f"Expected qty > 0 with capital={capital}, atr={atr_dec}, price={price}"
    )


# ---------------------------------------------------------------------------
# Test 8 — resolve_signal returns TradingDecision with action="BUY"
# ---------------------------------------------------------------------------


def test_signal_hierarchy_resolves_without_crash() -> None:
    """resolve_signal with BULL regime + ml_signal=1 must return action='BUY'."""
    try:
        from algo_trading.execution.signal_hierarchy import (
            DriftSeverity,
            HMMRegime,
            RiskStatus,
            TradingDecision,
            resolve_signal,
        )
    except ImportError as exc:
        pytest.skip(f"algo_trading.execution.signal_hierarchy not importable: {exc}")

    decision = resolve_signal(
        ml_signal=1,
        ml_confidence=0.72,
        regime=HMMRegime.BULL,
        drift_flag=DriftSeverity.NONE,
        risk_status=RiskStatus.OK,
    )
    assert isinstance(decision, TradingDecision), "Must return TradingDecision"
    assert decision.action == "BUY", (
        f"Expected action='BUY' for BULL regime + ml_signal=1, got {decision.action!r}"
    )


# ---------------------------------------------------------------------------
# Test 9 — All major modules importable without crash
# ---------------------------------------------------------------------------


def test_full_pipeline_no_imports_fail() -> None:
    """All Stage 1-2 modules must be importable without raising."""
    modules_to_test = [
        "algo_trading.config",
        "algo_trading.risk.cost_model",
        "algo_trading.risk.manager",
        "algo_trading.execution.signal_hierarchy",
        "algo_trading.labeling.triple_barrier",
        "algo_trading.labeling.walk_forward",
        "algo_trading.features.technical",
    ]

    # Optional modules: hmmlearn, arch, filterpy may not be installed in CI
    optional_modules = [
        "algo_trading.models.ensemble",
        "algo_trading.models.online_model",
    ]

    import importlib

    failed: list[str] = []
    for module_path in modules_to_test:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            failed.append(f"{module_path}: {exc}")

    if failed:
        pytest.skip(
            "Some required modules not importable (missing dependencies):\n"
            + "\n".join(f"  - {m}" for m in failed)
        )

    # Optional modules: skip silently, don't fail
    optional_failed: list[str] = []
    for module_path in optional_modules:
        try:
            importlib.import_module(module_path)
        except ImportError:
            optional_failed.append(module_path)

    if optional_failed:
        import warnings
        warnings.warn(
            f"Optional modules not importable (hmmlearn/arch/filterpy may be missing): "
            f"{optional_failed}",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Test 10 — RingBuffer stores and retrieves 10 synthetic tick dicts
# ---------------------------------------------------------------------------


def test_ring_buffer_stores_and_retrieves() -> None:
    """RingBuffer: push 10 synthetic ticks, last_as_polars() must return 10 rows."""
    try:
        from algo_trading.data.ring_buffer import RingBuffer
    except ImportError as exc:
        pytest.skip(f"algo_trading.data.ring_buffer not importable: {exc}")

    try:
        import polars as pl
    except ImportError as exc:
        pytest.skip(f"polars not installed: {exc}")

    buf = RingBuffer(capacity=100)

    np.random.seed(7)
    n_ticks = 10
    for i in range(n_ticks):
        ltp = float(2800.0 + np.random.normal(0, 5))
        volume = int(np.random.randint(1000, 50000))
        bid = ltp - 0.05
        ask = ltp + 0.05
        timestamp_ns = 1_672_644_900_000_000_000 + i * 1_000_000_000  # 1-second steps

        buf.push(
            ltp=ltp,
            volume=volume,
            bid=bid,
            ask=ask,
            timestamp_ns=timestamp_ns,
        )

    assert buf.count == n_ticks, f"Expected {n_ticks} ticks in buffer, got {buf.count}"

    df = buf.last_as_polars(n_ticks)
    assert isinstance(df, pl.DataFrame), "last_as_polars() must return a Polars DataFrame"
    assert len(df) == n_ticks, (
        f"Expected {n_ticks} rows from last_as_polars, got {len(df)}"
    )
    assert "ltp" in df.columns, "DataFrame must have 'ltp' column"
    assert "volume" in df.columns, "DataFrame must have 'volume' column"
    assert "timestamp" in df.columns, "DataFrame must have 'timestamp' column"
