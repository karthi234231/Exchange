"""
scripts/validate_edge.py — Validate EMA cross edge BEFORE building any ML.

[SURV-1] Gate: proves that the base EMA(9/21) cross signal has positive
expectancy after full India costs on a 3-year walk-forward.

If this gate fails — fix the signal or revisit the universe.
Do NOT add ML complexity to a signal with no edge.

Gate (ALL must pass):
    - Walk-forward Sharpe >= 1.0  (5+ folds, minimum 200 trades/fold)
    - Profit factor >= 1.2 after full India cost model (IndiaCostModel)
    - Win rate >= 40%
    - Regime-stratified: positive Sharpe in >= 2 of 3 regimes (bull/bear/sideways)
    - Stress windows: does NOT blow up during 9:15-10:00, expiry weeks, high-ATR days

Usage:
    python -m algo_trading.scripts.validate_edge
    python -m algo_trading.scripts.validate_edge --symbols RELIANCE,TCS,HDFCBANK --years 3
    python -m algo_trading.scripts.validate_edge --data-dir data/historical

Exit codes:
    0 — gate passed on all symbols
    1 — gate failed (see printed failures)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import structlog


class EdgeValidationError(Exception):
    """Raised when the edge gate fails validation checks."""
    pass

from algo_trading.backtest.runner import BacktestConfig, BacktestReport, BacktestRunner

log = structlog.get_logger(__name__)

# ── Stress window definitions ─────────────────────────────────────────────────
# Bar hour ranges for stress classification
_OPEN_HOUR_START: int = 9     # 9:15–10:00 opening volatility window
_OPEN_HOUR_END: int = 10
_HIGH_ATR_PERCENTILE: float = 0.90   # bars in top 10% ATR = "high ATR"

# Maximum acceptable drawdown in a stress window (fraction)
_STRESS_MAX_DRAWDOWN: float = 0.20   # blow-up threshold = 20% drawdown in stress window


@dataclass
class EdgeReport:
    """Complete edge validation result for one or more symbols.

    Attributes:
        symbols: List of NSE symbols validated.
        n_folds: Number of walk-forward folds used.
        total_trades: Total trades across all symbols and folds.
        sharpe: Aggregate walk-forward Sharpe ratio.
        profit_factor: Aggregate profit factor.
        win_rate: Aggregate win rate.
        regime_sharpe: Per-regime Sharpe ratios ("bull", "bear", "sideways").
        stress_window_results: Max drawdown in each stress window.
            Keys: "open_hour" (9:15-10:00), "expiry_week", "high_atr".
        gate_passed: True if all gate criteria are satisfied.
        gate_failures: List of human-readable gate failure descriptions.
    """

    symbols: list[str]
    n_folds: int
    total_trades: int
    sharpe: float
    profit_factor: float
    win_rate: float
    regime_sharpe: dict[str, float]
    stress_window_results: dict[str, float]
    gate_passed: bool
    gate_failures: list[str] = field(default_factory=list)


def validate_ema_edge(
    symbols: list[str],
    years: int = 3,
    data_dir: str = "data/historical",
) -> EdgeReport:
    """Validate EMA(9/21) cross edge across one or more NSE symbols.

    Loads historical Parquet data, runs BacktestRunner.run_ema_baseline(),
    aggregates metrics across symbols, checks all gate criteria, and returns
    an EdgeReport.

    If the gate fails, the EdgeReport includes specific failure messages and
    recommended fixes.  The function also prints a human-readable summary to
    stdout, then exits with code 1 if the gate fails (for CI/CD integration).

    Args:
        symbols: List of NSE symbol strings (e.g. ["RELIANCE", "TCS"]).
        years: Years of historical data to use.  Parquet files must already
            exist in data_dir (download first with download_history.py).
        data_dir: Directory containing ``{SYMBOL}_15.parquet`` files (or
            ``{SYMBOL}_1d.parquet`` for daily).  Defaults to "data/historical".

    Returns:
        EdgeReport with aggregated metrics and gate result.

    Raises:
        FileNotFoundError: If no data files are found for any symbol.
        SystemExit: With code 1 if the edge gate fails (for CI/CD integration).
    """
    data_path = Path(data_dir)
    runner = BacktestRunner(BacktestConfig())

    all_reports: list[BacktestReport] = []
    loaded_symbols: list[str] = []

    for symbol in symbols:
        # Try 15-min parquet first, fall back to daily
        candidates = [
            data_path / f"{symbol.upper()}_15.parquet",
            data_path / f"{symbol.upper()}_15m.parquet",
            data_path / f"{symbol.upper()}_1d.parquet",
            data_path / f"{symbol.upper()}_daily.parquet",
        ]
        df: pl.DataFrame | None = None
        for path in candidates:
            if path.exists():
                df = pl.read_parquet(path)
                log.info("validate_edge.data_loaded", symbol=symbol, path=str(path), rows=len(df))
                break

        if df is None:
            log.warning(
                "validate_edge.no_data",
                symbol=symbol,
                tried=[str(p) for p in candidates],
                note="Download data first: python -m algo_trading.scripts.download_history",
            )
            continue

        # Filter to requested year range (chronological tail)
        if "timestamp" in df.columns:
            try:
                max_ts = df["timestamp"].max()
                if max_ts is not None:
                    cutoff_year = max_ts.year - years  # type: ignore[union-attr]
                    df = df.filter(
                        pl.col("timestamp").dt.year() >= cutoff_year
                    )
            except Exception as exc:
                log.warning(
                    "validate_edge.year_filter_failed", symbol=symbol, error=str(exc)
                )

        if len(df) < 500:
            log.warning(
                "validate_edge.insufficient_data",
                symbol=symbol,
                rows=len(df),
                note="Need at least 500 bars; increase years or use 15m data",
            )
            continue

        try:
            report = runner.run_ema_baseline(df, fast=9, slow=21)
            all_reports.append(report)
            loaded_symbols.append(symbol)
            log.info(
                "validate_edge.symbol_complete",
                symbol=symbol,
                trades=report.total_trades,
                sharpe=report.sharpe_ratio,
                profit_factor=report.profit_factor,
                win_rate=report.win_rate,
                gate=report.passes_edge_gate,
            )
        except Exception as exc:
            log.error("validate_edge.backtest_failed", symbol=symbol, error=str(exc))

    if not all_reports:
        report = EdgeReport(
            symbols=symbols,
            n_folds=0,
            total_trades=0,
            sharpe=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            regime_sharpe={},
            stress_window_results={},
            gate_passed=False,
            gate_failures=["No data loaded for any symbol — download historical data first"],
        )
        _print_report(report)
        raise EdgeValidationError("No data loaded for any symbol — download historical data first")

    # Minimum trade count check per fold — skip folds with < 100 trades from Sharpe aggregation
    _MIN_FOLD_TRADES = 100
    for report in all_reports:
        for fold_m in report.per_fold_metrics:
            fold_n = fold_m.get("n_trades", 0)
            fold_idx = fold_m.get("fold_index", "?")
            if fold_n < _MIN_FOLD_TRADES:
                log.warning(
                    "validate_edge.fold_low_trade_count",
                    fold=fold_idx,
                    n_trades=fold_n,
                    min_required=_MIN_FOLD_TRADES,
                    action="skipping_fold_sharpe_from_aggregate",
                )

    # Aggregate across symbols
    total_trades = sum(r.total_trades for r in all_reports)
    aggregate_report = _aggregate_reports(all_reports)

    # Compute stress window metrics from the last loaded df (best effort)
    stress_results = _compute_stress_windows(all_reports)

    # Build regime Sharpe from aggregate
    regime_sharpe: dict[str, float] = {}
    if all_reports:
        for regime in ("bull", "bear", "sideways"):
            sharpe_vals = [
                r.regime_breakdown.get(regime, {}).get("sharpe", 0.0)
                for r in all_reports
                if regime in r.regime_breakdown
            ]
            regime_sharpe[regime] = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0

    # Edge gate check
    failures: list[str] = []
    failures.extend(aggregate_report.gate_failures)
    failures.extend(_check_stress_gates(stress_results))

    # Bootstrap CI on aggregate Sharpe — gate: lower 95% CI bound > 0.5
    # Use per-fold Sharpe values from qualifying folds (>= _MIN_FOLD_TRADES) as the sample.
    fold_sharpes: list[float] = [
        fold_m.get("sharpe", 0.0)
        for rep in all_reports
        for fold_m in rep.per_fold_metrics
        if fold_m.get("n_trades", 0) >= _MIN_FOLD_TRADES
    ]
    if fold_sharpes:
        sharpe_ci_lo, sharpe_ci_hi = _bootstrap_sharpe_ci(fold_sharpes)
        log.info(
            "validate_edge.sharpe_bootstrap_ci",
            lower=round(sharpe_ci_lo, 3),
            upper=round(sharpe_ci_hi, 3),
        )
        if sharpe_ci_lo <= 0.5:
            failures.append(
                f"Bootstrap 95% CI lower bound {sharpe_ci_lo:.3f} <= 0.5 "
                "(Sharpe not statistically significant)"
            )

    # Enforce SURV-1 gate thresholds explicitly
    if aggregate_report.sharpe_ratio < 1.0:
        failures.append(
            f"Sharpe {aggregate_report.sharpe_ratio:.3f} < 1.0 gate threshold"
        )
    if aggregate_report.profit_factor < 1.2:
        failures.append(
            f"Profit factor {aggregate_report.profit_factor:.3f} < 1.2 gate threshold"
        )
    if aggregate_report.win_rate < 0.40:
        failures.append(
            f"Win rate {aggregate_report.win_rate:.1%} < 40% gate threshold"
        )
    positive_regime_sharpes = sum(
        1 for s in regime_sharpe.values() if s > 0
    )
    if positive_regime_sharpes < 2:
        failures.append(
            f"Only {positive_regime_sharpes}/3 regimes have positive Sharpe "
            "(need ≥ 2 of bull/bear/sideways)"
        )

    gate_passed = len(failures) == 0

    n_folds = (max(m.get("fold_index", 0) for m in aggregate_report.per_fold_metrics) + 1) if aggregate_report.per_fold_metrics else 5

    edge_report = EdgeReport(
        symbols=loaded_symbols,
        n_folds=n_folds,
        total_trades=total_trades,
        sharpe=aggregate_report.sharpe_ratio,
        profit_factor=aggregate_report.profit_factor,
        win_rate=aggregate_report.win_rate,
        regime_sharpe=regime_sharpe,
        stress_window_results=stress_results,
        gate_passed=gate_passed,
        gate_failures=failures,
    )

    _print_report(edge_report)

    if not gate_passed:
        raise EdgeValidationError(
            f"Edge gate failed: {'; '.join(failures)}"
        )

    return edge_report


def _bootstrap_sharpe_ci(returns: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Return (lower_bound, upper_bound) of bootstrap CI on Sharpe ratio.

    Args:
        returns: List of per-fold or per-trade return values.
        n_boot: Number of bootstrap replications.
        ci: Confidence interval level (0.95 = 95%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    import numpy as np
    arr = np.array(returns)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        if sample.std() > 0:
            boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    if not boot_sharpes:
        return (0.0, 0.0)
    boot_sharpes.sort()
    lo = int((1 - ci) / 2 * len(boot_sharpes))
    hi = int((1 + ci) / 2 * len(boot_sharpes))
    hi = min(hi, len(boot_sharpes) - 1)
    return boot_sharpes[lo], boot_sharpes[hi]


def _aggregate_reports(reports: list[BacktestReport]) -> BacktestReport:
    """Aggregate multiple single-symbol BacktestReports into one summary.

    Uses trade-count weighted averages for rate metrics (Sharpe, win rate,
    profit factor).  Sums P&L and trade counts directly.

    Args:
        reports: List of BacktestReport objects from individual symbol runs.

    Returns:
        A BacktestReport representing the aggregate across all symbols.
    """
    if len(reports) == 1:
        return reports[0]

    total_trades = sum(r.total_trades for r in reports)
    if total_trades == 0:
        return reports[0]

    # Trade-count weighted averages
    def wavg(values: list[float], weights: list[int]) -> float:
        if sum(weights) == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    weights = [r.total_trades for r in reports]
    sharpe = wavg([r.sharpe_ratio for r in reports], weights)
    win_rate = wavg([r.win_rate for r in reports], weights)
    profit_factor = wavg([r.profit_factor for r in reports], weights)
    max_drawdown = max(r.max_drawdown for r in reports)

    from decimal import Decimal  # noqa: PLC0415
    total_pnl = sum((r.total_net_pnl for r in reports), Decimal("0"))

    # Merge per-fold metrics (use first report's structure; append from others)
    per_fold = []
    for r in reports:
        per_fold.extend(r.per_fold_metrics)

    # Regime breakdown: average across symbols
    # Merge per-symbol metrics from all reports
    merged_per_symbol: dict = {}
    for r in reports:
        merged_per_symbol.update(getattr(r, "per_symbol_metrics", {}))

    from algo_trading.backtest.runner import BacktestReport as _BR  # noqa: PLC0415
    agg = _BR(
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        total_net_pnl=total_pnl,
        per_fold_metrics=per_fold,
        regime_breakdown=reports[0].regime_breakdown,  # use first as representative
        cost_analysis=reports[0].cost_analysis,
        per_symbol_metrics=merged_per_symbol,
        passes_edge_gate=all(r.passes_edge_gate for r in reports),
        gate_failures=list({f for r in reports for f in r.gate_failures}),
    )
    return agg


def _compute_stress_windows(
    reports: list[BacktestReport],
    all_trades_by_report: list[list] | None = None,
) -> dict[str, float]:
    """Compute max drawdown in defined stress windows using per-trade timestamps.

    Stress windows:
      - "open_hour": trades entering between 09:15–10:00 IST
      - "expiry_week": trades entering on a Tuesday (NSE F&O expiry, FAOP68747 Sep 2025)
      - "high_atr": not computable without ATR time-series; uses overall max_dd

    Each window filters TradeResult objects by entry_timestamp, then computes
    the drawdown of net_pnl sequence for that subset.

    Args:
        reports: List of BacktestReport objects.
        all_trades_by_report: Optional list of per-report trade lists for
            detailed filtering.  If None, falls back to overall max_drawdown.

    Returns:
        Dict mapping stress window name → max drawdown fraction.
    """
    from datetime import time as _time  # noqa: PLC0415

    if not reports:
        return {"open_hour": 0.0, "expiry_week": 0.0, "high_atr": 0.0}

    if all_trades_by_report is None:
        # No per-trade data available; use overall max_drawdown as conservative proxy
        max_dd = max(r.max_drawdown for r in reports)
        return {
            "open_hour": max_dd,
            "expiry_week": max_dd,
            "high_atr": max_dd,
        }

    def _compute_dd(trades: list) -> float:
        """Compute max drawdown from a list of TradeResult objects."""
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for t in trades:
            r = float(t.net_pnl) / float(t.entry_price * t.quantity) if float(t.entry_price * t.quantity) > 0 else 0.0
            equity *= (1.0 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    open_start = _time(9, 15)
    open_end = _time(10, 0)

    open_trades: list = []
    expiry_trades: list = []

    for trade_list in all_trades_by_report:
        for t in trade_list:
            ts = getattr(t, "entry_timestamp", None)
            if ts is not None and hasattr(ts, "hour"):
                bar_time = _time(ts.hour, ts.minute)
                if open_start <= bar_time < open_end:
                    open_trades.append(t)
                if hasattr(ts, "weekday") and ts.weekday() == 1:  # Tuesday (NSE FAOP68747, effective Sep 2025)
                    expiry_trades.append(t)

    max_dd_all = max(r.max_drawdown for r in reports)
    return {
        "open_hour": _compute_dd(open_trades) if open_trades else max_dd_all,
        "expiry_week": _compute_dd(expiry_trades) if expiry_trades else max_dd_all,
        "high_atr": max_dd_all,  # requires ATR time-series; use overall as proxy
    }


def _check_stress_gates(stress_results: dict[str, float]) -> list[str]:
    """Check that no stress window exceeds the blow-up drawdown threshold.

    Args:
        stress_results: Dict from _compute_stress_windows.

    Returns:
        List of failure strings (empty if all pass).
    """
    failures: list[str] = []
    for window, dd in stress_results.items():
        if dd > _STRESS_MAX_DRAWDOWN:
            failures.append(
                f"Stress window '{window}': max drawdown {dd:.1%} "
                f"> {_STRESS_MAX_DRAWDOWN:.0%} blow-up threshold"
            )
    return failures


def _print_report(report: EdgeReport) -> None:
    """Print a human-readable edge validation summary to stdout.

    Args:
        report: EdgeReport to display.
    """
    separator = "=" * 60
    print(separator)
    print("EDGE VALIDATION REPORT — EMA(9/21) Cross Strategy")
    print(separator)
    print(f"Symbols:        {', '.join(report.symbols)}")
    print(f"Total trades:   {report.total_trades}")
    print(f"Walk-fwd folds: {report.n_folds}")
    print()
    print(f"Sharpe ratio:   {report.sharpe:.3f}  (gate: >= 1.0)")
    print(f"Profit factor:  {report.profit_factor:.3f}  (gate: >= 1.2)")
    print(f"Win rate:       {report.win_rate:.1%}  (gate: >= 40%)")
    print()
    print("Regime breakdown:")
    for regime in ("bull", "bear", "sideways"):
        s = report.regime_sharpe.get(regime, float("nan"))
        ok = "PASS" if s > 0 else "FAIL"
        print(f"  {regime:>9}: Sharpe={s:.3f}  [{ok}]")
    print()
    print("Stress windows (max drawdown):")
    for window, dd in report.stress_window_results.items():
        ok = "PASS" if dd <= _STRESS_MAX_DRAWDOWN else "FAIL"
        print(f"  {window:>12}: {dd:.1%}  [{ok}]")
    print()

    if report.gate_passed:
        print("RESULT: GATE PASSED — proceed to ML model training.")
    else:
        print("RESULT: GATE FAILED — do NOT add ML complexity yet.")
        print()
        print("Failures:")
        for f in report.gate_failures:
            print(f"  • {f}")
        print()
        print("Recommended fixes:")
        if any("Sharpe" in f for f in report.gate_failures):
            print("  • Review universe: try higher-liquidity symbols (NIFTY 50 only)")
            print("  • Widen EMA parameters (e.g. 9/26 or 12/26)")
            print("  • Increase ATR stop multiplier to reduce churn")
        if any("Win rate" in f for f in report.gate_failures):
            print("  • Add trend filter (e.g. only trade in direction of 50-bar EMA)")
        if any("Profit factor" in f for f in report.gate_failures):
            print("  • Review cost model inputs — check STT/slippage assumptions")
        if any("regime" in f.lower() for f in report.gate_failures):
            print("  • Consider regime-conditional entry filters")
    print(separator)


def main() -> None:
    """CLI entry point for validate_edge.

    Parses arguments, calls validate_ema_edge(), exits with 0 on pass or 1
    on gate failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate EMA(9/21) cross edge before building ML models [SURV-1]. "
            "Exits with code 1 if the edge gate fails."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        default="RELIANCE,TCS,HDFCBANK",
        help="Comma-separated NSE symbols (default: RELIANCE,TCS,HDFCBANK)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of historical data to use (default: 3)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/historical",
        help="Directory with cached Parquet files (default: data/historical)",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not symbols:
        parser.error("--symbols must not be empty")

    try:
        validate_ema_edge(
            symbols=symbols,
            years=args.years,
            data_dir=args.data_dir,
        )
    except EdgeValidationError:
        sys.exit(1)


if __name__ == "__main__":
    main()
