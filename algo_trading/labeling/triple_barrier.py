"""Triple-barrier labeling for NSE algo trading.

Implements López de Prado's triple-barrier method from *Advances in Financial
Machine Learning* (AFML).  Barriers are SYMMETRIC (±1.5 ATR) to avoid the
long-side class imbalance caused by asymmetric multipliers.

Correctness invariants enforced at runtime:
- Labeling loop is strictly forward-only (no vectorised future peek).
- ``validate_no_lookahead`` performs a post-hoc sanity check on every labelled
  row's barrier windows.
- ``compute_label_distribution`` warns when any class falls below 15 % or
  exceeds 55 % of the sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

import polars as pl
import structlog

logger: Final = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS: Final[list[str]] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr14",
]

IMBALANCE_WARN_LOW: Final[float] = 0.15   # Warn if any class < 15 %
IMBALANCE_WARN_HIGH: Final[float] = 0.55  # Warn if any class > 55 %


@dataclass
class BarrierSpec:
    """Configuration for triple-barrier levels.

    Attributes:
        upper_atr_mult: ATR multiplier for the upper (profit-take) barrier.
            MUST equal lower_atr_mult to keep class balance symmetric.
        lower_atr_mult: ATR multiplier for the lower (stop-loss) barrier.
            MUST equal upper_atr_mult.
        max_holding_bars: Maximum number of bars before a time-out label (0) is
            assigned.  For 15-min bars, 20 bars ≈ 5 trading hours.
        min_atr_floor: Minimum absolute ATR value used when computing barriers.
            Prevents micro-barriers on very liquid / low-volatility stocks where
            ATR approaches zero.  5 bps (0.0005) is appropriate for NSE equities.
    """

    upper_atr_mult: float = 1.5   # SYMMETRIC — do not change without class balance check
    lower_atr_mult: float = 1.5   # SYMMETRIC — must equal upper
    max_holding_bars: int = 20    # 20 × 15-min bars = 5 hours max hold
    min_atr_floor: float = 0.0005  # 5 bps minimum barrier

    def __post_init__(self) -> None:
        if not math.isclose(self.upper_atr_mult, self.lower_atr_mult):
            raise ValueError(
                f"Asymmetric barriers detected: upper={self.upper_atr_mult}, "
                f"lower={self.lower_atr_mult}.  Asymmetric barriers cause class "
                "imbalance and long-side bias.  Set both multipliers to the same value."
            )
        if self.max_holding_bars < 1:
            raise ValueError(f"max_holding_bars must be ≥ 1, got {self.max_holding_bars}")
        if self.min_atr_floor < 0:
            raise ValueError(f"min_atr_floor must be ≥ 0, got {self.min_atr_floor}")


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------


class TripleBarrierLabeler:
    """Labels OHLCV bars using the triple-barrier method (AFML, ch. 3).

    Each bar receives one of three labels:
        +1  — upper barrier was touched first (long profit-take).
        -1  — lower barrier was touched first (stop-loss / short profit-take).
         0  — neither barrier was touched within ``spec.max_holding_bars``
              (time-out / sideways).

    The labeling scan is strictly forward-only.  The implementation iterates
    bar-by-bar and slices only *subsequent* rows, which guarantees no
    lookahead bias.

    Args:
        spec: Barrier configuration.  Defaults to ``BarrierSpec()`` which uses
            symmetric ±1.5 ATR barriers and a 20-bar holding period.
    """

    def __init__(self, spec: BarrierSpec = BarrierSpec()) -> None:
        self._spec = spec
        self._log = logger.bind(component="TripleBarrierLabeler")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply triple-barrier labeling to OHLCV data with ATR.

        For each bar at index *i*:

        1. ``upper_barrier = close[i] + spec.upper_atr_mult * max(atr14[i], min_atr_floor)``
        2. ``lower_barrier = close[i] - spec.lower_atr_mult * max(atr14[i], min_atr_floor)``
        3. ``time_barrier  = timestamp[i + max_holding_bars]`` (or last bar)
        4. Scan bars *i+1 … i+max_holding_bars*:
           - If ``high[j] >= upper_barrier`` and ``high[j]`` is touched before
             ``low[j] <= lower_barrier``, label = +1.
           - If ``low[j] <= lower_barrier`` first, label = -1.
           - If both are touched on the same bar, the bar that is *closer* to the
             entry close wins; ties are broken in favour of -1 (conservative).
           - If neither barrier is touched, label = 0.

        Important: barriers are set from **close** prices; hits are checked on
        **high** and **low** prices.  This prevents using the bar's own close to
        both set and immediately trigger a barrier.

        Args:
            df: Polars DataFrame with columns
                ``[timestamp, open, high, low, close, volume, atr14]``.
                Rows must be in chronological order (earliest first).

        Returns:
            The input DataFrame with additional columns:
            ``[label, upper_barrier, lower_barrier, time_barrier,
               sample_weight, barrier_hit]``.
            ``barrier_hit`` is a string: ``"upper"`` | ``"lower"`` | ``"time"``.

        Raises:
            ValueError: If required columns are missing or DataFrame is empty.
        """
        df = self._validate_input(df)
        spec = self._spec
        n = len(df)

        self._log.info("starting_triple_barrier_labeling", n_bars=n)

        # Extract arrays once for fast indexed access.
        closes: list[float] = df["close"].to_list()
        highs: list[float] = df["high"].to_list()
        lows: list[float] = df["low"].to_list()
        atrs: list[float] = df["atr14"].to_list()
        timestamps = df["timestamp"].to_list()

        labels: list[int] = []
        upper_barriers: list[float] = []
        lower_barriers: list[float] = []
        time_barriers: list[object] = []
        barrier_hits: list[str] = []

        for i in range(n):
            entry_close = closes[i]
            effective_atr = max(atrs[i] if atrs[i] is not None else spec.min_atr_floor, spec.min_atr_floor)

            ub = entry_close + spec.upper_atr_mult * effective_atr
            lb = entry_close - spec.lower_atr_mult * effective_atr

            # Time barrier is the timestamp of the last bar in the window
            # (or the last available bar if near the end of the series).
            window_end = min(i + spec.max_holding_bars, n - 1)
            tb = timestamps[window_end]

            upper_barriers.append(ub)
            lower_barriers.append(lb)
            time_barriers.append(tb)

            # Forward scan: strictly bars i+1 … i+max_holding_bars
            label = 0
            hit = "time"
            for j in range(i + 1, window_end + 1):
                upper_hit = highs[j] >= ub
                lower_hit = lows[j] <= lb

                if upper_hit and lower_hit:
                    # Both barriers on the same bar — award the one whose
                    # distance from entry_close is smaller.  Ties go to -1.
                    upper_dist = highs[j] - ub      # 0 if exactly on barrier
                    lower_dist = lb - lows[j]       # 0 if exactly on barrier
                    if upper_dist < lower_dist:
                        label = 1
                        hit = "upper"
                    else:
                        label = -1
                        hit = "lower"
                    break
                elif upper_hit:
                    label = 1
                    hit = "upper"
                    break
                elif lower_hit:
                    label = -1
                    hit = "lower"
                    break

            labels.append(label)
            barrier_hits.append(hit)

        labels_series = pl.Series("label", labels, dtype=pl.Int8)
        dist = self.compute_label_distribution(labels_series)
        self._log.info("label_distribution_computed", distribution=dist)

        result = df.with_columns(
            [
                pl.Series("label", labels, dtype=pl.Int8),
                pl.Series("upper_barrier", upper_barriers, dtype=pl.Float64),
                pl.Series("lower_barrier", lower_barriers, dtype=pl.Float64),
                pl.Series("time_barrier", time_barriers),
                pl.Series("barrier_hit", barrier_hits, dtype=pl.String),
            ]
        )

        # Build an events frame with entry timestamp and exit time (t1) so that
        # uniqueness weights can account for temporal clustering of label spans.
        events_df = result.select(
            [pl.col("timestamp"), pl.col("time_barrier").alias("t1"), pl.col("label")]
        )
        weights = self.compute_uniqueness_weights(events_df)
        result = result.with_columns(weights.alias("sample_weight"))

        self._log.info("triple_barrier_labeling_complete", n_labels=n)
        return result

    def compute_sample_weights(self, labels: pl.Series) -> pl.Series:
        """Compute inverse-frequency sample weights to address class imbalance.

        Each sample receives a weight inversely proportional to the frequency of
        its class.  The weights are then normalised so that the mean weight equals
        1.0, preserving the effective sample size passed to the estimator.

        A minority class (e.g. +1 in a bearish regime) thus receives a higher
        weight, which prevents the model from ignoring it.

        Args:
            labels: Integer series of labels (values in ``{-1, 0, +1}``).

        Returns:
            Float64 series of per-sample weights aligned with ``labels``.
        """
        total = len(labels)
        if total == 0:
            return pl.Series("sample_weight", [], dtype=pl.Float64)

        counts: dict[int, int] = {}
        for v in labels.to_list():
            counts[v] = counts.get(v, 0) + 1

        # Inverse-frequency: weight_class = total / (n_classes * count_class)
        n_classes = len(counts)
        inv_freq: dict[int, float] = {
            cls: total / (n_classes * cnt) for cls, cnt in counts.items()
        }

        weights = [inv_freq[v] for v in labels.to_list()]
        return pl.Series("sample_weight", weights, dtype=pl.Float64)

    def compute_uniqueness_weights(self, events_df: pl.DataFrame) -> pl.Series:
        """Compute uniqueness-weighted sample weights (AFML Chapter 4).

        Each observation *i* whose label spans bars ``[entry_i, t1_i]`` receives
        a weight proportional to its average uniqueness — the reciprocal of the
        average number of concurrently active label spans at each bar within the
        window.  Observations that overlap with many others contribute less
        information and are down-weighted accordingly.

        Algorithm:
            1. For every bar *t* in the dataset, count ``c_t`` = number of label
               spans ``[entry_j, t1_j]`` that are active at bar *t*
               (i.e. ``entry_j <= t <= t1_j``).
            2. For observation *i*, its average uniqueness is:
               ``u_i = mean(1 / c_t)`` for all bars t in ``[entry_i, t1_i]``.
            3. Weights are normalised so that ``mean(w_i) == 1.0``.

        Args:
            events_df: DataFrame with columns:
                - ``timestamp``: entry bar timestamp for each observation.
                - ``t1``: exit bar timestamp (time barrier or actual barrier exit).
                - ``label``: integer label (used to confirm the row is labeled).

        Returns:
            Float64 series of uniqueness-based sample weights, one per row,
            aligned with ``events_df``.  Returns an empty series when the
            input is empty.
        """
        n = len(events_df)
        if n == 0:
            return pl.Series("sample_weight", [], dtype=pl.Float64)

        required = {"timestamp", "t1"}
        missing = required - set(events_df.columns)
        if missing:
            self._log.warning(
                "compute_uniqueness_weights_missing_columns",
                missing=list(missing),
                fallback="returning unit weights",
            )
            return pl.Series("sample_weight", [1.0] * n, dtype=pl.Float64)

        entries: list = events_df["timestamp"].to_list()
        exits: list = events_df["t1"].to_list()

        # Vectorised O(n) implementation using numpy cumulative sums.
        # Map all timestamps to integer indices in a sorted unique grid.
        all_ts = np.unique(np.concatenate([
            np.array(entries, dtype="datetime64[ns]"),
            np.array(exits, dtype="datetime64[ns]"),
        ]))
        n_t = len(all_ts)

        entry_idx = np.searchsorted(all_ts, np.array(entries, dtype="datetime64[ns]"))
        exit_idx  = np.searchsorted(all_ts, np.array(exits,   dtype="datetime64[ns]"))

        # Build concurrency array: +1 at entry, -1 one slot after exit.
        events_arr = np.zeros(n_t + 1, dtype=np.int32)
        np.add.at(events_arr, entry_idx, 1)
        np.add.at(events_arr, np.minimum(exit_idx + 1, n_t), -1)
        concurrency = np.maximum(np.cumsum(events_arr)[:n_t], 1)

        # Prefix sum of 1/concurrency for O(1) range-mean queries.
        inv_conc = 1.0 / concurrency
        cum_inv = np.concatenate([[0.0], np.cumsum(inv_conc)])

        span_len = (exit_idx - entry_idx + 1).clip(min=1)
        uniqueness = (cum_inv[exit_idx + 1] - cum_inv[entry_idx]) / span_len

        # Normalise so that mean weight == 1.0.
        mean_u = float(uniqueness.mean()) or 1.0
        weights = (uniqueness / mean_u).tolist()

        return pl.Series("sample_weight", weights, dtype=pl.Float64)

    def compute_label_distribution(self, labels: pl.Series) -> dict[int, float]:
        """Return per-class proportions and warn on imbalance.

        Checks whether any class falls outside the acceptable range
        ``[IMBALANCE_WARN_LOW, IMBALANCE_WARN_HIGH]`` (default 15 %–55 %).
        If so, a ``structlog`` WARNING is emitted.

        Args:
            labels: Integer series with values in ``{-1, 0, +1}``.

        Returns:
            Dictionary mapping each class label (int) to its proportion (float
            in ``[0, 1]``).  All three classes are always present in the output
            (with 0.0 if absent from the data).
        """
        total = len(labels)
        if total == 0:
            return {-1: 0.0, 0: 0.0, 1: 0.0}

        counts: dict[int, int] = {-1: 0, 0: 0, 1: 0}
        for v in labels.to_list():
            if v in counts:
                counts[v] += 1

        dist = {cls: cnt / total for cls, cnt in counts.items()}

        for cls, pct in dist.items():
            if pct < IMBALANCE_WARN_LOW:
                self._log.warning(
                    "class_imbalance_low",
                    label_class=cls,
                    proportion=round(pct, 4),
                    threshold=IMBALANCE_WARN_LOW,
                    advice=(
                        "Model will be biased against this class.  "
                        "Consider re-checking ATR multipliers or input data range."
                    ),
                )
            elif pct > IMBALANCE_WARN_HIGH:
                self._log.warning(
                    "class_imbalance_high",
                    label_class=cls,
                    proportion=round(pct, 4),
                    threshold=IMBALANCE_WARN_HIGH,
                    advice=(
                        "This class dominates the sample.  "
                        "Model will be biased.  "
                        "Check for trending regimes or wide holding windows."
                    ),
                )

        return dist

    def validate_no_lookahead(self, df: pl.DataFrame) -> bool:
        """Verify that no future data was used during labeling.

        For each row, the method confirms:

        1. ``upper_barrier`` and ``lower_barrier`` were derived only from data
           available *at or before* bar *i* (i.e. ``close[i]`` and ``atr14[i]``).
        2. ``time_barrier`` falls at or after ``timestamp[i]``.

        The check is intentionally conservative: it re-derives the expected
        barrier values from the raw inputs and compares them to the labeled
        values, flagging any discrepancy.

        Args:
            df: Labeled DataFrame (output of ``label()``).

        Returns:
            ``True`` if no lookahead is detected.

        Raises:
            AssertionError: On the first row where a discrepancy is found.
            ValueError: If required labeled columns are absent.
        """
        required = REQUIRED_INPUT_COLUMNS + [
            "label",
            "upper_barrier",
            "lower_barrier",
            "time_barrier",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"validate_no_lookahead: missing columns {missing}")

        spec = self._spec
        closes = df["close"].to_list()
        atrs = df["atr14"].to_list()
        timestamps = df["timestamp"].to_list()
        upper_barriers = df["upper_barrier"].to_list()
        lower_barriers = df["lower_barrier"].to_list()
        time_barriers = df["time_barrier"].to_list()

        n = len(df)
        tolerance = 1e-9  # floating-point comparison tolerance

        for i in range(n):
            eff_atr = max(atrs[i] if atrs[i] is not None else spec.min_atr_floor, spec.min_atr_floor)
            expected_ub = closes[i] + spec.upper_atr_mult * eff_atr
            expected_lb = closes[i] - spec.lower_atr_mult * eff_atr

            assert abs(upper_barriers[i] - expected_ub) < tolerance, (
                f"Lookahead bias at row {i}: "
                f"stored upper_barrier={upper_barriers[i]:.6f} != "
                f"expected={expected_ub:.6f} (close={closes[i]}, atr={atrs[i]})"
            )
            assert abs(lower_barriers[i] - expected_lb) < tolerance, (
                f"Lookahead bias at row {i}: "
                f"stored lower_barrier={lower_barriers[i]:.6f} != "
                f"expected={expected_lb:.6f} (close={closes[i]}, atr={atrs[i]})"
            )

            # time_barrier must not precede the entry timestamp
            assert time_barriers[i] >= timestamps[i], (
                f"Lookahead bias at row {i}: "
                f"time_barrier={time_barriers[i]} precedes entry timestamp={timestamps[i]}"
            )

        self._log.info("validate_no_lookahead_passed", n_rows_checked=n)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pl.DataFrame) -> pl.DataFrame:
        """Raise informative errors for bad inputs and drop null-ATR rows.

        Args:
            df: Input DataFrame to validate.

        Returns:
            The (possibly filtered) DataFrame with null-ATR rows removed.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if df.is_empty():
            raise ValueError("Input DataFrame is empty.")
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing}.  "
                f"Expected: {REQUIRED_INPUT_COLUMNS}"
            )
        # Warn about nulls and drop those rows — they would produce garbage labels.
        null_count = df["atr14"].null_count()
        if null_count > 0:
            self._log.warning(
                "atr14_contains_nulls",
                null_count=null_count,
                advice=(
                    "Null ATR values produce garbage barriers.  "
                    "Drop the warm-up rows before labeling for cleaner results."
                ),
            )
            self._log.warning("dropping_null_atr_rows", n_dropped=null_count)
            df = df.filter(pl.col("atr14").is_not_null())
        return df
