"""Walk-forward time-series splitter with purge and embargo gaps.

Implements the purge+embargo framework from López de Prado's *Advances in
Financial Machine Learning* (AFML, ch. 7) to prevent lookahead contamination
in train/test splits.

Key design decisions:
- Purge gap = ``max_holding_bars + max_feature_lookback_bars + embargo_bars``.
  With Ichimoku (52-bar lookback) as the longest feature this yields a minimum
  purge of 77 bars (52 + 20 + 5).  Any shorter purge risks leaking training
  labels into the test window via lagging indicators or open trade P&L.
- Embargo = 5 bars (not the often-cited 2) because NSE intraday price
  autocorrelation persists for 4–6 bars at 15-min resolution.
- A minimum of 5 folds is enforced; fewer folds produce unreliable OOS metrics.
- Each fold must contain at least ``min_test_trades`` labeled rows (non-NaN
  label column); folds that are too small raise ``ValueError`` rather than
  silently producing biased metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import polars as pl
import structlog

logger: Final = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplitMetadata:
    """Metadata attached to every walk-forward fold.

    Attributes:
        fold_index: Zero-based fold number.
        train_start: Inclusive start row index of the training window.
        train_end: Exclusive end row index of the training window.
        test_start: Inclusive start row index of the test window (after
            purge+embargo gap).
        test_end: Exclusive end row index of the test window.
        purge_bars: Total number of bars in the purge+embargo gap between
            training end and test start.
        embargo_bars: Number of embargo bars (subset of purge_bars).
        bar_type: Bar resolution identifier, e.g. ``"time_15min"``.
        train_size: Number of rows in the training slice.
        test_size: Number of rows in the test slice.
    """

    fold_index: int
    train_start: int      # row index, inclusive
    train_end: int        # row index, exclusive
    test_start: int       # row index, inclusive (after purge+embargo)
    test_end: int         # row index, exclusive
    purge_bars: int
    embargo_bars: int
    bar_type: str         # "time_15min" | "volume" | "tick"
    train_size: int
    test_size: int


@dataclass
class WalkForwardConfig:
    """Configuration for the walk-forward splitter.

    Attributes:
        n_folds: Number of out-of-sample folds.  Must be ≥ 5 so that
            reported metrics are statistically meaningful.
        max_holding_bars: Maximum number of bars a position can be held
            (used to size the purge gap).  Should match
            ``BarrierSpec.max_holding_bars``.
        max_feature_lookback_bars: Longest lookback window of any feature
            used in the model.  Ichimoku Cloud spans 52 bars, making it the
            default binding constraint.
        embargo_bars: Additional gap after the purge to absorb serial
            autocorrelation.  5 bars at 15-min resolution ≈ 75 minutes.
        min_test_trades: Minimum number of *labeled* rows required in every
            test fold.  Folds below this threshold raise ``ValueError``.
        bar_type: Metadata tag describing bar construction method.
    """

    n_folds: int = 5
    max_holding_bars: int = 20
    max_feature_lookback_bars: int = 52   # Ichimoku is the longest feature
    embargo_bars: int = 5
    min_test_trades: int = 200
    bar_type: str = "time_15min"

    def __post_init__(self) -> None:
        if self.n_folds < 5:
            raise ValueError(
                f"n_folds={self.n_folds} is too small.  "
                "Minimum 5 walk-forward folds are required for any reported metric."
            )
        if self.embargo_bars < 1:
            raise ValueError(
                f"embargo_bars={self.embargo_bars} must be ≥ 1.  "
                "Use at least 5 for NSE 15-min data."
            )
        if self.min_test_trades < 1:
            raise ValueError(f"min_test_trades={self.min_test_trades} must be ≥ 1.")


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class WalkForwardSplitter:
    """Walk-forward splitter with purge+embargo for time-series data.

    Splits a chronologically ordered DataFrame into *n_folds* (train, test)
    pairs.  Each split respects the following gap between training and test::

        |---TRAIN---|---PURGE+EMBARGO---|---TEST---|

    PURGE absorbs bars whose labels could bleed into the test window (because
    the labeling horizon extends ``max_holding_bars`` into the future).
    EMBARGO adds a further buffer for indicator lag and price autocorrelation.

    Args:
        config: Walk-forward configuration.  Defaults to ``WalkForwardConfig()``
            which uses 5 folds, 77-bar purge (Ichimoku default), and 5-bar
            embargo.
    """

    def __init__(self, config: WalkForwardConfig = WalkForwardConfig()) -> None:
        self._config = config
        self._log = logger.bind(
            component="WalkForwardSplitter",
            n_folds=config.n_folds,
            purge_bars=self.purge_bars,
            bar_type=config.bar_type,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def purge_bars(self) -> int:
        """Total purge gap in bars: max_holding + max_lookback + embargo.

        This value is the *minimum* number of rows that must separate the last
        training bar from the first test bar.  It is always ≥ 51 (even when a
        shorter lookback is configured) to cover the MACD(26) baseline.

        Returns:
            Integer purge gap, never less than 51.
        """
        cfg = self._config
        computed = cfg.max_holding_bars + cfg.max_feature_lookback_bars + cfg.embargo_bars
        # Hard floor: MACD(26) + 20-bar hold + 5-bar embargo = 51
        return max(computed, 51)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, df: pl.DataFrame
    ) -> list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]]:
        """Generate walk-forward (train, test, metadata) folds.

        The dataset is partitioned into *n_folds* equal-width test windows.
        Training data for fold *k* covers all rows up to (but not including)
        the purge gap that precedes test fold *k*.  This is the **fixed-window**
        (rolling) variant — training size grows with each fold.

        Fold structure::

            Fold 0:  [---TRAIN(0)---][PURGE][---TEST(0)---]
            Fold 1:  [---TRAIN(0+1)---][PURGE][---TEST(1)---]
            ...

        Args:
            df: Chronologically ordered Polars DataFrame.  Must have a
                ``timestamp`` column (used for leakage validation).  If a
                ``label`` column is present, its non-null count is used to
                validate ``min_test_trades``.

        Returns:
            List of ``(train_df, test_df, SplitMetadata)`` tuples, one per
            fold.

        Raises:
            ValueError: If the dataset is too small to form the requested folds
                or if any test fold contains fewer than ``min_test_trades``
                labeled rows.
        """
        cfg = self._config
        n = len(df)
        purge = self.purge_bars

        self._validate_dataset_size(n, purge)

        # Divide the *usable* tail (after the mandatory warm-up head) into
        # n_folds equal test windows.
        # We reserve 2×purge rows: purge for warm-up AND purge as the minimum
        # training window that must precede fold 0's test slice.
        usable_rows = n - 2 * purge
        fold_size, remainder = divmod(usable_rows, cfg.n_folds)

        splits: list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]] = []

        for k in range(cfg.n_folds):
            # Distribute remainder rows across the first `remainder` folds so
            # that fold sizes differ by at most 1.
            extra = 1 if k < remainder else 0
            test_size_k = fold_size + extra

            # Compute cumulative test-fold offsets.
            # test_start for fold k = 2*purge + sum(test sizes of folds 0..k-1)
            # (2*purge ensures fold 0 always has at least `purge` rows of training data)
            test_start = 2 * purge + sum(
                fold_size + (1 if j < remainder else 0) for j in range(k)
            )
            test_end = test_start + test_size_k

            # Training window ends before the purge gap preceding test_start.
            train_end = test_start - purge
            train_start = 0  # rolling variant: always from the beginning

            if train_end <= train_start:
                raise ValueError(
                    f"Fold {k}: train_end={train_end} <= train_start={train_start}.  "
                    "Dataset is too small for the requested purge gap and fold count."
                )

            train_df = df.slice(train_start, train_end - train_start)
            test_df = df.slice(test_start, test_end - test_start)

            # Validate minimum test trades.
            self._check_min_test_trades(test_df, k)

            meta = SplitMetadata(
                fold_index=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_bars=purge,
                embargo_bars=cfg.embargo_bars,
                bar_type=cfg.bar_type,
                train_size=len(train_df),
                test_size=len(test_df),
            )

            splits.append((train_df, test_df, meta))
            self._log.debug(
                "fold_created",
                fold=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_df),
                test_size=len(test_df),
            )

        self.validate_no_future_leakage(splits)
        self._log.info("walk_forward_split_complete", n_folds=len(splits))
        return splits

    def validate_no_future_leakage(
        self, splits: list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]]
    ) -> bool:
        """Verify that no training window overlaps with or precedes its test window.

        For every ``(train_df, test_df, meta)`` tuple, asserts::

            max(train_df["timestamp"]) < min(test_df["timestamp"])

        This is a necessary (though not sufficient) condition for the absence of
        lookahead bias; it does not check purge size, only chronological ordering.

        Args:
            splits: List of ``(train_df, test_df, metadata)`` tuples as returned
                by ``split()`` or ``get_expanding_window_splits()``.

        Returns:
            ``True`` if all folds pass.

        Raises:
            ValueError: On the first fold where temporal ordering is violated,
                or if ``timestamp`` column is absent in any fold.
        """
        for train_df, test_df, meta in splits:
            k = meta.fold_index

            if "timestamp" not in train_df.columns or "timestamp" not in test_df.columns:
                raise ValueError(
                    f"Fold {k}: 'timestamp' column is required for leakage validation."
                )
            if train_df.is_empty() or test_df.is_empty():
                raise ValueError(
                    f"Fold {k}: train or test DataFrame is empty — cannot validate."
                )

            max_train_ts = train_df["timestamp"].max()
            min_test_ts = test_df["timestamp"].min()

            if not (max_train_ts < min_test_ts):
                raise ValueError(
                    f"Future leakage detected in fold {k}: "
                    f"max(train.timestamp)={max_train_ts} >= "
                    f"min(test.timestamp)={min_test_ts}.  "
                    "The purge gap is insufficient or the data is not sorted."
                )

        self._log.info(
            "validate_no_future_leakage_passed", n_folds_checked=len(splits)
        )
        return True

    def get_expanding_window_splits(
        self, df: pl.DataFrame
    ) -> list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]]:
        """Generate expanding-window walk-forward folds.

        In this variant the training window *grows* with each fold (anchored at
        row 0), while the test window slides forward without overlap.  This
        mimics a live-trading scenario where all historical data is retrained
        before each new test period.

        The fold structure is identical to ``split()`` except that
        ``train_start`` is always 0 and ``train_end`` grows.

        Note:
            This method produces the same folds as ``split()`` for the rolling
            (fixed-window) variant because training already starts at row 0.
            It is provided as a named alias for clarity in calling code.

        Args:
            df: Chronologically ordered Polars DataFrame.

        Returns:
            List of ``(train_df, test_df, SplitMetadata)`` tuples.

        Raises:
            ValueError: Same conditions as ``split()``.
        """
        # The rolling-from-zero implementation in split() is already expanding.
        # Delegate directly and re-tag the metadata for caller clarity.
        splits = self.split(df)

        tagged: list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]] = []
        for train_df, test_df, meta in splits:
            tagged_meta = SplitMetadata(
                fold_index=meta.fold_index,
                train_start=0,          # always anchored at origin
                train_end=meta.train_end,
                test_start=meta.test_start,
                test_end=meta.test_end,
                purge_bars=meta.purge_bars,
                embargo_bars=meta.embargo_bars,
                bar_type=meta.bar_type + "_expanding",
                train_size=meta.train_size,
                test_size=meta.test_size,
            )
            tagged.append((train_df, test_df, tagged_meta))

        self._log.info(
            "expanding_window_split_complete", n_folds=len(tagged)
        )
        return tagged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dataset_size(self, n: int, purge: int) -> None:
        """Raise ``ValueError`` if the dataset cannot support the configuration.

        Args:
            n: Total number of rows in the dataset.
            purge: Computed purge gap in bars.

        Raises:
            ValueError: If the dataset is too small.
        """
        cfg = self._config
        # Minimum needed: 2*purge (warm-up + min training) + n_folds * min_test_trades
        min_needed = 2 * purge + cfg.n_folds * cfg.min_test_trades
        if n < min_needed:
            raise ValueError(
                f"Dataset too small: {n} rows available, but "
                f"{min_needed} rows are required "
                f"(2×purge={2*purge} + {cfg.n_folds} folds × {cfg.min_test_trades} "
                f"min_test_trades).  "
                "Reduce n_folds, reduce min_test_trades, or supply more data."
            )

    def _check_min_test_trades(self, test_df: pl.DataFrame, fold_index: int) -> None:
        """Raise ``ValueError`` if a test fold is too small.

        If the DataFrame has a ``label`` column, the count of non-null labels is
        used; otherwise, the total row count is used as a proxy.

        Args:
            test_df: Test slice for the fold.
            fold_index: Zero-based fold index (for error messages).

        Raises:
            ValueError: If the fold is below ``min_test_trades``.
        """
        cfg = self._config
        if "label" in test_df.columns:
            trade_count = len(test_df["label"].drop_nulls())
        else:
            trade_count = len(test_df)

        if trade_count < cfg.min_test_trades:
            raise ValueError(
                f"Fold {fold_index}: test fold contains only {trade_count} labeled rows, "
                f"but min_test_trades={cfg.min_test_trades}.  "
                "Increase the dataset size or reduce n_folds / min_test_trades."
            )
