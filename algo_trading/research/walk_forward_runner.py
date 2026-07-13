"""
algo_trading/research/walk_forward_runner.py — Walk-forward test harness.

Implements the exact walk-forward methodology specified in
FO_SYSTEM_DESIGN.md for NSE F&O strategies:

- **Rolling window**: train on N years, test on M months, step forward.
- **Purge window**: removes samples within
  ``holding_period + max_feature_lookback + embargo`` bars of the
  train/test boundary to prevent label leakage.
- **Embargo**: skips the first K bars after each test period boundary to
  suppress NSE autocorrelation leakage (5 bars — 2 was found insufficient).
- **Minimum 5 folds** for any metric to be reported.
- **15 % final holdout** reserved and tested only once, after all
  hyperparameter decisions are finalised.

Phase 2 exit criteria evaluated automatically
---------------------------------------------
1. ``passes_min_folds``         — >= 5 walk-forward folds
2. ``passes_min_trades``        — >= 500 OOS trades in aggregate
3. ``passes_oos_is_ratio``      — OOS Sharpe >= 0.5 × IS Sharpe in every fold
4. ``passes_concentration``     — no single fold contributes > 40 % of total PnL
5. ``passes_holdout``           — final holdout Sharpe >= 0.5

Usage
-----
::

    from algo_trading.research.walk_forward_runner import (
        WalkForwardRunner, WalkForwardConfig
    )

    runner = WalkForwardRunner(WalkForwardConfig(n_folds=5))
    result = runner.run(
        strategy_fn=my_strategy_fn,
        data_provider=my_provider,
        start_date=date(2022, 1, 1),
        end_date=date(2024, 12, 31),
        symbol="NIFTY",
    )
    result.print_summary()

The *strategy_fn* must have the signature::

    def my_strategy(events_iter, *, mode: str) -> list[TradeRecord]:
        ...

where *mode* is ``"train"``, ``"test"``, or ``"holdout"``, and the function
returns a list of :class:`~algo_trading.research.metrics_engine.TradeRecord`
objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtesting.

    Attributes
    ----------
    train_years:
        Length of each training window in years (rolling — not expanding).
    test_years:
        Length of each out-of-sample test window in years.
    n_folds:
        Minimum number of folds.  Phase 2 requires >= 5.
    purge_bars:
        Number of bars at the *end* of the training window to drop due to
        the holding-period overlap with the test window.  Default 20 bars
        = 5 hours at 15-minute granularity.
    embargo_bars:
        Number of bars at the *start* of the test window to skip.  5 bars
        is the NSE-specific value (2 was found insufficient given the
        autocorrelation structure of NIFTY and BANKNIFTY).
    max_feature_lookback_bars:
        Worst-case feature lookback in bars (MACD uses 26 bars).  Added to
        ``purge_bars`` to give the total purge window.
    bar_interval_minutes:
        Bar granularity used to convert bar counts to calendar days.
        60-minute bars are the default for F&O chain snapshots.
    min_trades_per_fold:
        Minimum expected OOS trades per fold.  500 total / 5 folds = 100.
    min_sharpe_oos:
        Minimum OOS Sharpe ratio threshold (Phase 2).
    max_single_fold_pnl_contribution:
        Maximum fraction of total PnL that one fold may contribute.
    holdout_sharpe_min:
        Minimum Sharpe required on the final holdout period.
    """

    train_years: float = 1.0
    test_years: float = 0.5
    n_folds: int = 5

    # Purge + embargo (FO_SYSTEM_DESIGN.md + CLAUDE.md)
    purge_bars: int = 20
    embargo_bars: int = 5              # 5 bars — NSE autocorrelation buffer
    max_feature_lookback_bars: int = 26  # MACD(26) worst-case

    @property
    def total_purge_bars(self) -> int:
        """Total purge window = holding_period + lookback + embargo."""
        return self.purge_bars + self.max_feature_lookback_bars + self.embargo_bars

    bar_interval_minutes: int = 60

    # Phase 2 exit criteria
    min_trades_per_fold: int = 100
    min_sharpe_oos: float = 0.5
    max_single_fold_pnl_contribution: float = 0.40
    holdout_sharpe_min: float = 0.5


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results for one train-test fold.

    Attributes
    ----------
    fold_idx:
        Zero-based fold index.
    purge_start:
        First date of the purge zone (rows after this date and before
        ``embargo_end`` are excluded from both train and test).
    embargo_end:
        First date that is safe to include in the test window.
    oos_is_ratio:
        ``test_metrics.sharpe_ratio / train_metrics.sharpe_ratio``.
        Phase 2 requires this to be >= 0.5 for every fold.
    """

    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    purge_start: date
    embargo_end: date
    train_metrics: Any   # MetricsSuite — typed Any to avoid circular import
    test_metrics: Any    # MetricsSuite
    trades_in_fold: list = field(default_factory=list)

    @property
    def oos_is_ratio(self) -> float:
        """OOS Sharpe / IS Sharpe.  Phase 2 requires >= 0.5.

        Returns 0.0 when IS Sharpe is near zero to avoid division by
        near-zero values producing misleading large ratios.
        """
        is_sharpe = (
            self.train_metrics.sharpe_ratio if self.train_metrics is not None else 0.0
        )
        oos_sharpe = (
            self.test_metrics.sharpe_ratio if self.test_metrics is not None else 0.0
        )
        if abs(is_sharpe) < 1e-6:
            return 0.0
        return oos_sharpe / is_sharpe


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes
    ----------
    aggregate_metrics:
        :class:`~algo_trading.research.metrics_engine.MetricsSuite` computed
        over all OOS trades concatenated across every fold.
    stability_stats:
        Output of :func:`~algo_trading.research.metrics_engine.per_fold_stability`.
    holdout_metrics:
        Metrics on the 15 % final holdout period, or ``None`` if the
        holdout evaluation failed.
    passes_*:
        Phase 2 exit criteria flags.  All five must be ``True`` for a
        strategy to advance to live paper-trading.
    """

    config: WalkForwardConfig
    folds: list[FoldResult]
    aggregate_metrics: Any           # MetricsSuite
    stability_stats: dict
    holdout_metrics: Any | None      # MetricsSuite or None

    # Phase 2 exit criteria
    passes_min_folds: bool
    passes_min_trades: bool
    passes_oos_is_ratio: bool
    passes_concentration: bool
    passes_holdout: bool

    @property
    def passes_all(self) -> bool:
        """True only when every Phase 2 exit criterion is satisfied."""
        return all([
            self.passes_min_folds,
            self.passes_min_trades,
            self.passes_oos_is_ratio,
            self.passes_concentration,
            self.passes_holdout,
        ])

    def print_summary(self) -> None:
        """Print a human-readable walk-forward summary to stdout."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Walk-Forward Summary: {len(self.folds)} folds")
        print(sep)

        if self.aggregate_metrics is not None:
            print(f"  {self.aggregate_metrics.summary_str()}")

        print("\nPhase 2 Exit Criteria:")
        print(f"  Min folds (>=5):        {'PASS' if self.passes_min_folds else 'FAIL'}")
        print(f"  Min trades (>=500):     {'PASS' if self.passes_min_trades else 'FAIL'}")
        print(f"  OOS/IS ratio (>=0.5):   {'PASS' if self.passes_oos_is_ratio else 'FAIL'}")
        print(f"  Concentration (<=40%):  {'PASS' if self.passes_concentration else 'FAIL'}")
        print(f"  Holdout Sharpe (>=0.5): {'PASS' if self.passes_holdout else 'FAIL'}")

        print("\nPer-Fold Sharpe:")
        for fold in self.folds:
            is_s = (
                fold.train_metrics.sharpe_ratio if fold.train_metrics is not None else 0.0
            )
            oos_s = (
                fold.test_metrics.sharpe_ratio if fold.test_metrics is not None else 0.0
            )
            print(
                f"  Fold {fold.fold_idx + 1}: "
                f"IS={is_s:.2f}, OOS={oos_s:.2f}, "
                f"ratio={fold.oos_is_ratio:.2f}, "
                f"trades={len(fold.trades_in_fold)}"
            )

        if self.holdout_metrics is not None:
            print(
                f"\nHoldout: Sharpe={self.holdout_metrics.sharpe_ratio:.2f}, "
                f"Trades={self.holdout_metrics.total_trades}"
            )

        overall = "PASS — ready for paper trading" if self.passes_all else "FAIL — not ready"
        print(f"\nOverall: {overall}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class WalkForwardRunner:
    """Walk-forward test harness for NSE F&O strategies.

    Implements the exact methodology from FO_SYSTEM_DESIGN.md:

    1. Compute rolling (train, test) date splits with purge + embargo.
    2. For each fold: call ``strategy_fn`` in ``"train"`` mode, then in
       ``"test"`` mode; record both IS and OOS metrics.
    3. Reserve 15 % of the data as an untouched final holdout.
    4. Evaluate all Phase 2 exit criteria on the aggregated OOS results.
    5. Optionally log the full run to an :class:`~.audit_store.AuditStore`.

    Parameters
    ----------
    config:
        Walk-forward configuration.  Defaults to :class:`WalkForwardConfig`
        with NSE F&O–appropriate defaults.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def generate_splits(
        self,
        start_date: date,
        end_date: date,
    ) -> list[tuple[date, date, date, date, date, date]]:
        """Generate train/test date splits with purge and embargo zones.

        Each tuple is:
        ``(train_start, train_end, purge_start, embargo_end, test_start, test_end)``

        - Rows in ``[purge_start, train_end)`` are excluded from training
          (they overlap with the test label horizon).
        - Rows in ``[train_end, embargo_end)`` are excluded from both
          train and test (NSE autocorrelation buffer).
        - The actual test window is ``[test_start, test_end)`` which equals
          ``[embargo_end, embargo_end + test_days)``.

        The cursor advances by 50 % of the test window on each iteration
        (overlapping folds), which is common practice for NSE intraday
        strategies where data is expensive.

        Parameters
        ----------
        start_date:
            First date of the available data range.
        end_date:
            Last date of the available data range (exclusive — the holdout
            end date is passed here, not the raw dataset end).

        Returns
        -------
        list of 6-tuple
            Chronologically ordered (train_start, train_end, purge_start,
            embargo_end, test_start, test_end) tuples.
        """
        train_days = int(self.config.train_years * 252)
        test_days = int(self.config.test_years * 252)

        # Convert bar counts to approximate calendar days.
        # NSE continuous session: 375 minutes / bar_interval_minutes bars per day.
        bars_per_day = 375.0 / self.config.bar_interval_minutes
        purge_days = math.ceil(
            (self.config.purge_bars + self.config.max_feature_lookback_bars) / bars_per_day
        )
        embargo_days = math.ceil(self.config.embargo_bars / bars_per_day)

        splits: list[tuple[date, date, date, date, date, date]] = []
        cursor = start_date

        # Safety limit: never generate more than 3× the requested fold count.
        max_splits = self.config.n_folds * 3

        while len(splits) < max_splits:
            train_start = cursor
            train_end = train_start + timedelta(days=train_days)
            purge_start = train_end - timedelta(days=purge_days)
            embargo_end = train_end + timedelta(days=embargo_days)
            test_start = embargo_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > end_date:
                break

            splits.append(
                (train_start, train_end, purge_start, embargo_end, test_start, test_end)
            )

            # Advance cursor by 50 % of the test window (overlapping folds).
            step = max(1, test_days // 2)
            cursor = test_start + timedelta(days=step)

        log.info("walk_forward.splits_generated", n_splits=len(splits))
        return splits

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_fn: Callable,
        data_provider: Any,
        start_date: date,
        end_date: date,
        symbol: str,
        capital: float = 100_000.0,
        holdout_pct: float = 0.15,
        audit_store: Any = None,
        strategy_name: str = "unnamed",
        strategy_version: str = "v1.0",
        strategy_params: dict | None = None,
    ) -> WalkForwardResult:
        """Run a full walk-forward backtest.

        Parameters
        ----------
        strategy_fn:
            Callable with signature
            ``strategy_fn(events_iter, *, mode: str) -> list[TradeRecord]``.
            *mode* is ``"train"``, ``"test"``, or ``"holdout"``.
        data_provider:
            A :class:`~algo_trading.research.chain_replay_engine.HistoricalDataProvider`
            (or compatible) instance that serves chain snapshots.
        start_date:
            First date of the data range.
        end_date:
            Last date of the data range.
        symbol:
            NSE ticker, e.g. ``"NIFTY"`` or ``"BANKNIFTY"``.
        capital:
            Starting capital in INR for percentage-return and Sharpe
            computations.
        holdout_pct:
            Fraction of the total date range reserved as the final holdout
            (evaluated once, after all hyperparameter decisions).
        audit_store:
            Optional :class:`~algo_trading.research.audit_store.AuditStore`
            instance.  If provided, the run is logged automatically.
        strategy_name:
            Human-readable name used in audit logs.
        strategy_version:
            Semver string used in audit logs.
        strategy_params:
            Dict of all strategy parameters used in audit logs.

        Returns
        -------
        WalkForwardResult
            Fully populated result with per-fold details, aggregate metrics,
            stability statistics, holdout metrics, and Phase 2 flags.
        """
        from .metrics_engine import compute_metrics, per_fold_stability
        from .chain_replay_engine import ChainReplayEngine, ReplayConfig

        # ---- Reserve final holdout ------------------------------------------
        total_days = (end_date - start_date).days
        holdout_days = max(1, int(total_days * holdout_pct))
        wf_end = end_date - timedelta(days=holdout_days)
        holdout_start = wf_end + timedelta(days=1)

        log.info(
            "walk_forward.start",
            symbol=symbol,
            start=str(start_date),
            end=str(end_date),
            holdout_start=str(holdout_start),
            n_folds=self.config.n_folds,
        )

        # ---- Generate splits ------------------------------------------------
        splits = self.generate_splits(start_date, wf_end)

        # Keep only the most recent n_folds splits when more were generated.
        if len(splits) > self.config.n_folds:
            splits = splits[-self.config.n_folds :]

        fold_results: list[FoldResult] = []
        all_oos_trades: list = []

        # ---- Fold loop ------------------------------------------------------
        for fold_idx, (
            train_start, train_end,
            purge_start, embargo_end,
            test_start, test_end,
        ) in enumerate(splits):

            log.info(
                "walk_forward.fold",
                fold=fold_idx + 1,
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
            )

            # Train (IS)
            train_config = ReplayConfig(
                symbol=symbol,
                start_date=train_start,
                end_date=purge_start,  # stop before the purge zone
            )
            train_engine = ChainReplayEngine(data_provider, train_config)

            try:
                train_trades = strategy_fn(
                    train_engine.replay(), mode="train"
                )
            except Exception as exc:
                log.error(
                    "walk_forward.train_error",
                    fold=fold_idx + 1,
                    error=str(exc),
                )
                train_trades = []

            train_metrics = compute_metrics(train_trades, capital=capital)

            # Test (OOS)
            test_config = ReplayConfig(
                symbol=symbol,
                start_date=test_start,
                end_date=test_end,
            )
            test_engine = ChainReplayEngine(data_provider, test_config)

            try:
                test_trades = strategy_fn(
                    test_engine.replay(), mode="test"
                )
            except Exception as exc:
                log.error(
                    "walk_forward.test_error",
                    fold=fold_idx + 1,
                    error=str(exc),
                )
                test_trades = []

            test_metrics = compute_metrics(test_trades, capital=capital)
            all_oos_trades.extend(test_trades)

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                embargo_end=embargo_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                trades_in_fold=test_trades,
            ))

            log.info(
                "walk_forward.fold_complete",
                fold=fold_idx + 1,
                is_sharpe=f"{train_metrics.sharpe_ratio:.2f}",
                oos_sharpe=f"{test_metrics.sharpe_ratio:.2f}",
                oos_is_ratio=f"{fold_results[-1].oos_is_ratio:.2f}",
                oos_trades=len(test_trades),
            )

        # ---- Aggregate metrics across all OOS folds -------------------------
        aggregate_metrics = compute_metrics(all_oos_trades, capital=capital)
        stability = per_fold_stability([f.test_metrics for f in fold_results])

        # Backfill fold_sharpe_std into aggregate (set to 0.0 by compute_metrics).
        aggregate_metrics.fold_sharpe_std = stability.get("fold_sharpe_std", 0.0)

        # ---- Final holdout --------------------------------------------------
        holdout_metrics = None
        if holdout_start < end_date:
            holdout_config = ReplayConfig(
                symbol=symbol,
                start_date=holdout_start,
                end_date=end_date,
            )
            holdout_engine = ChainReplayEngine(data_provider, holdout_config)
            try:
                holdout_trades = strategy_fn(
                    holdout_engine.replay(), mode="holdout"
                )
                holdout_metrics = compute_metrics(holdout_trades, capital=capital)
                log.info(
                    "walk_forward.holdout_complete",
                    sharpe=f"{holdout_metrics.sharpe_ratio:.2f}",
                    trades=holdout_metrics.total_trades,
                )
            except Exception as exc:
                log.error("walk_forward.holdout_error", error=str(exc))

        # ---- Phase 2 exit criteria ------------------------------------------
        # OOS/IS ratio: every fold must have ratio >= 0.5.
        # Folds where IS Sharpe is near zero (abs < 1e-6) return ratio=0.0
        # from oos_is_ratio, which correctly counts as a failure.
        passes_oos_is = (
            all(f.oos_is_ratio >= 0.5 for f in fold_results)
            if fold_results
            else False
        )

        result = WalkForwardResult(
            config=self.config,
            folds=fold_results,
            aggregate_metrics=aggregate_metrics,
            stability_stats=stability,
            holdout_metrics=holdout_metrics,
            passes_min_folds=len(fold_results) >= 5,
            passes_min_trades=aggregate_metrics.total_trades >= 500,
            passes_oos_is_ratio=passes_oos_is,
            passes_concentration=stability.get("passes_concentration_limit", False),
            passes_holdout=(
                holdout_metrics is not None
                and holdout_metrics.sharpe_ratio >= self.config.holdout_sharpe_min
            ),
        )

        # ---- Audit logging --------------------------------------------------
        if audit_store is not None:
            try:
                audit_store.log_experiment(
                    strategy_name=strategy_name,
                    strategy_version=strategy_version,
                    strategy_params=strategy_params or {},
                    data_params={
                        "symbol": symbol,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "n_folds": len(fold_results),
                        "holdout_pct": holdout_pct,
                    },
                    results={
                        "aggregate_sharpe": aggregate_metrics.sharpe_ratio,
                        "aggregate_profit_factor": aggregate_metrics.profit_factor,
                        "aggregate_win_rate": aggregate_metrics.win_rate,
                        "total_oos_trades": aggregate_metrics.total_trades,
                        "max_drawdown_pct": aggregate_metrics.max_drawdown_pct,
                        "holdout_sharpe": (
                            holdout_metrics.sharpe_ratio
                            if holdout_metrics is not None
                            else None
                        ),
                        "passes_all_criteria": result.passes_all,
                    },
                )
            except Exception as exc:
                log.error("walk_forward.audit_log_failed", error=str(exc))

        return result
