"""
algo_trading/research/metrics_engine.py — Performance metrics with confidence intervals.

Computes Sharpe, Sortino, Calmar, maximum drawdown, VaR/CVaR, win rate, profit
factor, and max consecutive losses from a list of completed trades.  Every
metric is accompanied by a 95% confidence interval where applicable.

Used by :class:`~algo_trading.research.walk_forward_runner.WalkForwardRunner`
to evaluate both in-sample (IS) and out-of-sample (OOS) performance on each
walk-forward fold.

Mathematical conventions
------------------------
- Sharpe ratio is annualised with sqrt(252) using *daily* P&L aggregation
  (not per-trade aggregation, which over-inflates N for intraday strategies).
- Risk-free rate is 0 — standard for intraday and short-duration F&O.
- Confidence intervals use *block bootstrap* (block = 5 days) to respect
  autocorrelation in daily returns.
- CAGR is computed from first-trade entry to last-trade exit.
- Drawdown percentages are relative to the running equity-curve peak.

Phase 2 exit criteria (FO_SYSTEM_DESIGN.md)
-------------------------------------------
:attr:`MetricsSuite.passes_min_trade_count`  — >= 500 OOS trades total
:attr:`MetricsSuite.passes_sharpe_threshold` — Sharpe >= 0.5
:attr:`MetricsSuite.passes_profit_factor`    — Profit Factor > 1.0
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single completed trade used for metrics computation.

    All cost components (commission, slippage, STT, etc.) must already be
    netted into ``pnl_inr`` before passing to :func:`compute_metrics`.
    """

    pnl_inr: float           # Net P&L after all transaction costs
    entry_time: object       # datetime (or any object with .year/.month/.date())
    exit_time: object        # datetime
    holding_days: float      # Calendar days between entry and exit
    strategy_name: str = ""
    notes: str = ""


@dataclass
class MetricsSuite:
    """Complete performance metrics suite with confidence intervals.

    All metrics are computed on out-of-sample data only.  No metric is
    reported without an associated quality indicator (CI or stability flag).

    Attributes
    ----------
    total_pnl_inr:
        Gross P&L in INR summed across all trades.
    annualized_return_pct:
        CAGR as a percentage (252 trading-day basis).
    sharpe_ratio:
        Annualised Sharpe ratio (rf=0, sqrt(252) scaling).
    sharpe_ci_95:
        (lower, upper) 95% block-bootstrap confidence interval on Sharpe.
    calmar_ratio:
        ``annualised_return_pct / max_drawdown_pct`` (avoids divide-by-zero
        by clamping denominator to 1e-6).
    fold_sharpe_std:
        Standard deviation of per-fold Sharpe ratios.  Set to 0.0 by
        :func:`compute_metrics`; populated by the walk-forward runner.
    """

    # Return metrics
    total_pnl_inr: float
    annualized_return_pct: float      # % p.a. (252 trading-day basis)

    # Risk-adjusted
    sharpe_ratio: float               # annualised; rf=0
    sharpe_ci_95: tuple[float, float] # 95% CI via block bootstrap
    calmar_ratio: float               # annualised_return_pct / max_drawdown_pct

    # Drawdown
    max_drawdown_pct: float           # % of running equity-curve peak
    max_drawdown_inr: float           # in INR
    avg_drawdown_pct: float           # average over all drawdown episodes
    max_drawdown_duration_days: float # calendar days of the longest drawdown

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float                   # 0–1
    avg_win_inr: float
    avg_loss_inr: float
    profit_factor: float              # gross_profit / abs(gross_loss)
    expectancy_per_trade_inr: float   # average P&L per trade

    # Downside risk
    sortino_ratio: float              # annualised; rf=0, downside threshold=0
    var_95_inr: float                 # empirical 1-day VaR at 95% confidence (positive=loss)
    cvar_95_inr: float                # conditional VaR / Expected Shortfall at 95%

    # Trade run analysis
    max_consecutive_losses: int       # longest streak of consecutive losing trades

    # Stability
    pct_profitable_months: float      # % of calendar months with positive P&L
    fold_sharpe_std: float            # std of per-fold Sharpes (lower = more stable)

    # Phase 2 exit criteria checks (FO_SYSTEM_DESIGN.md)
    passes_min_trade_count: bool      # total_trades >= 500
    passes_sharpe_threshold: bool     # sharpe_ratio >= min_sharpe_threshold
    passes_profit_factor: bool        # profit_factor > 1.0

    def summary_str(self) -> str:
        """One-line human-readable summary."""
        sortino_str = (
            f"{self.sortino_ratio:.2f}"
            if self.sortino_ratio != float("inf") else "∞"
        )
        return (
            f"Sharpe={self.sharpe_ratio:.2f} "
            f"[{self.sharpe_ci_95[0]:.2f}, {self.sharpe_ci_95[1]:.2f}], "
            f"Sortino={sortino_str}, "
            f"PF={self.profit_factor:.2f}, WR={self.win_rate:.1%}, "
            f"MaxDD={self.max_drawdown_pct:.1f}%, "
            f"VaR95=₹{self.var_95_inr:,.0f}, "
            f"Trades={self.total_trades}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: list[TradeRecord],
    capital: float = 100_000.0,
    n_bootstrap: int = 1000,
    min_sharpe_threshold: float = 0.5,
) -> MetricsSuite:
    """Compute a full :class:`MetricsSuite` from a list of completed trades.

    Parameters
    ----------
    trades:
        Completed trades (out-of-sample only; all costs netted into
        ``TradeRecord.pnl_inr``).
    capital:
        Starting capital in INR used for percentage-return and Sharpe
        computations.  Must be > 0.
    n_bootstrap:
        Number of bootstrap resamples for the Sharpe 95% CI.  1 000 gives
        stable CIs; reduce to 200 for fast unit tests.
    min_sharpe_threshold:
        Minimum Sharpe ratio for the Phase 2 ``passes_sharpe_threshold``
        flag.  Default 0.5 matches FO_SYSTEM_DESIGN.md.

    Returns
    -------
    MetricsSuite
        Fully populated metrics object.  Returns an all-zero object if
        *trades* is empty.
    """
    if not trades:
        return _empty_metrics(capital, min_sharpe_threshold)

    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")

    pnls = np.array([t.pnl_inr for t in trades], dtype=float)

    # Sort by entry time for equity curve construction.
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    cumulative_pnl = np.cumsum([t.pnl_inr for t in sorted_trades])
    equity_curve = capital + cumulative_pnl

    # ---- Drawdown -------------------------------------------------------
    max_dd_pct, max_dd_inr, avg_dd_pct, max_dd_days = _compute_drawdown(
        equity_curve, sorted_trades
    )

    # ---- Trade statistics -----------------------------------------------
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls)
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # ---- Sharpe (annualised, rf=0) --------------------------------------
    # Aggregate to daily P&L first — using per-trade P&L inflates the
    # degrees of freedom and gives an artificially high Sharpe.
    daily_pnls = _aggregate_to_daily(sorted_trades)
    sharpe = _annualized_sharpe(daily_pnls, capital)
    sharpe_ci = _bootstrap_sharpe_ci(daily_pnls, capital, n=n_bootstrap)

    # ---- CAGR + Calmar --------------------------------------------------
    annualized_ret = _annualized_return(equity_curve, sorted_trades, capital)
    calmar = annualized_ret / max(max_dd_pct, 1e-6)

    # ---- Monthly win % --------------------------------------------------
    monthly_pct = _pct_profitable_months(sorted_trades)

    # ---- Sortino ratio --------------------------------------------------
    sortino = _sortino_ratio(daily_pnls, capital)

    # ---- VaR / CVaR (empirical) -----------------------------------------
    var_95, cvar_95 = _var_cvar_95(daily_pnls)

    # ---- Max consecutive losses ------------------------------------------
    max_consec_losses = _max_consecutive_losses(sorted_trades)

    return MetricsSuite(
        total_pnl_inr=float(np.sum(pnls)),
        annualized_return_pct=annualized_ret * 100.0,
        sharpe_ratio=sharpe,
        sharpe_ci_95=sharpe_ci,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_inr=max_dd_inr,
        avg_drawdown_pct=avg_dd_pct,
        max_drawdown_duration_days=max_dd_days,
        total_trades=len(trades),
        winning_trades=int(len(wins)),
        losing_trades=int(len(losses)),
        win_rate=win_rate,
        avg_win_inr=float(np.mean(wins)) if len(wins) > 0 else 0.0,
        avg_loss_inr=float(np.mean(losses)) if len(losses) > 0 else 0.0,
        profit_factor=profit_factor,
        expectancy_per_trade_inr=float(np.mean(pnls)),
        sortino_ratio=sortino,
        var_95_inr=var_95,
        cvar_95_inr=cvar_95,
        max_consecutive_losses=max_consec_losses,
        pct_profitable_months=monthly_pct,
        fold_sharpe_std=0.0,    # populated by walk_forward_runner across folds
        passes_min_trade_count=len(trades) >= 500,
        passes_sharpe_threshold=sharpe >= min_sharpe_threshold,
        passes_profit_factor=profit_factor > 1.0,
    )


def per_fold_stability(fold_metrics: list[MetricsSuite]) -> dict:
    """Compute cross-fold stability statistics from a list of fold metrics.

    Implements the Phase 2 exit criteria from FO_SYSTEM_DESIGN.md:

    - No single fold may contribute more than 40 % of total P&L.
    - OOS Sharpe >= 0.5 × IS Sharpe (evaluated by the runner, not here).

    Parameters
    ----------
    fold_metrics:
        One :class:`MetricsSuite` per fold, in chronological order.

    Returns
    -------
    dict
        Keys: ``fold_sharpe_mean``, ``fold_sharpe_std``, ``fold_sharpe_min``,
        ``fold_sharpe_max``, ``max_fold_pnl_contribution_pct``,
        ``passes_concentration_limit``, ``all_folds_positive``.
        Returns empty dict for an empty input.
    """
    if not fold_metrics:
        return {}

    sharpes = [m.sharpe_ratio for m in fold_metrics]
    pnls = [m.total_pnl_inr for m in fold_metrics]
    total_pnl = sum(pnls)
    denom = max(abs(total_pnl), 1e-6)

    max_fold_contribution = max(abs(p) / denom for p in pnls)

    fold_sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    return {
        "fold_sharpe_mean": float(np.mean(sharpes)),
        "fold_sharpe_std": fold_sharpe_std,
        "sharpe_std": fold_sharpe_std,  # alias for backward compatibility
        "fold_sharpe_min": float(np.min(sharpes)),
        "fold_sharpe_max": float(np.max(sharpes)),
        "max_fold_pnl_contribution_pct": max_fold_contribution * 100.0,
        "passes_concentration_limit": max_fold_contribution <= 0.40,
        "all_folds_positive": all(s > 0 for s in sharpes),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_drawdown(
    equity_curve: np.ndarray,
    trades: list[TradeRecord],
) -> tuple[float, float, float, float]:
    """Compute drawdown statistics from an equity curve.

    Returns
    -------
    tuple[float, float, float, float]
        ``(max_dd_pct, max_dd_inr, avg_dd_pct, max_dd_duration_days)``

        - *max_dd_pct*: maximum drawdown as a percentage of the running peak.
        - *max_dd_inr*: the INR depth of that maximum drawdown.
        - *avg_dd_pct*: mean of all per-bar drawdown percentages.
        - *max_dd_duration_days*: calendar days of the longest drawdown episode.
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0, 0.0, 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_inr = 0.0
    drawdowns: list[float] = []
    dd_start_idx: int | None = None
    max_dd_duration = 0

    for i, val in enumerate(equity_curve):
        if val > peak:
            # Exited a drawdown — measure its duration.
            if dd_start_idx is not None and trades:
                end_trade_idx = min(i, len(trades) - 1)
                try:
                    duration_days = (
                        trades[end_trade_idx].exit_time
                        - trades[dd_start_idx].entry_time
                    ).days
                    max_dd_duration = max(max_dd_duration, duration_days)
                except (AttributeError, TypeError):
                    # entry_time / exit_time may not support subtraction in tests.
                    max_dd_duration = max(max_dd_duration, i - dd_start_idx)
            peak = val
            dd_start_idx = None

        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > 0.0:
            if dd_start_idx is None:
                dd_start_idx = i
            drawdowns.append(dd)
            if dd > max_dd:
                max_dd = dd
                max_dd_inr = peak - val

    avg_dd = float(np.mean(drawdowns)) if drawdowns else 0.0
    return max_dd * 100.0, max_dd_inr, avg_dd * 100.0, float(max_dd_duration)


def _annualized_sharpe(daily_pnls: np.ndarray, capital: float) -> float:
    """Compute annualised Sharpe ratio from daily P&L.

    Formula: ``sqrt(252) * mean(r) / std(r, ddof=1)``
    where ``r = daily_pnl / capital``.

    Returns 0.0 when fewer than 2 observations are available or when the
    standard deviation is effectively zero (no variance in returns).
    """
    if len(daily_pnls) < 2:
        return 0.0
    daily_returns = daily_pnls / capital
    mean_r = float(np.mean(daily_returns))
    std_r = float(np.std(daily_returns, ddof=1))
    if std_r < 1e-10:
        # Zero variance: return sign-based large value or 0 if mean is also zero
        if mean_r > 1e-10:
            return float("inf")
        if mean_r < -1e-10:
            return float("-inf")
        return 0.0
    return float(math.sqrt(252) * mean_r / std_r)


def _bootstrap_sharpe_ci(
    daily_pnls: np.ndarray,
    capital: float,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a 95% block-bootstrap confidence interval on the Sharpe ratio.

    Block bootstrap is used (rather than i.i.d. bootstrap) because daily
    equity-strategy returns exhibit autocorrelation.  A fixed block length
    of 5 days is appropriate for weekly mean-reversion horizons typical of
    NSE F&O.

    Parameters
    ----------
    daily_pnls:
        Array of daily net P&L values.
    capital:
        Starting capital (denominator for return computation).
    n:
        Bootstrap iterations.  1 000 is sufficient for stable CIs.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        ``(2.5th percentile, 97.5th percentile)`` of the bootstrap Sharpe
        distribution.  Returns ``(0.0, 0.0)`` when insufficient data.
    """
    if len(daily_pnls) < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    block_length = 5
    n_blocks = max(1, len(daily_pnls) // block_length)

    sharpes: list[float] = []
    for _ in range(n):
        # Sample n_blocks blocks with replacement.
        block_indices = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([
            daily_pnls[
                b * block_length : min((b + 1) * block_length, len(daily_pnls))
            ]
            for b in block_indices
        ])
        if len(sample) >= 2:
            sharpes.append(_annualized_sharpe(sample, capital))

    if not sharpes:
        return (0.0, 0.0)

    arr = np.array(sharpes)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def _annualized_return(
    equity_curve: np.ndarray,
    trades: list[TradeRecord],
    capital: float,
) -> float:
    """Compute CAGR from first-trade entry to last-trade exit.

    Uses the standard formula ``(final_equity / initial_capital)^(1/years) - 1``.
    The minimum duration is clamped to 1 day to avoid division by zero for
    same-day test runs.

    Returns 0.0 when the equity curve has fewer than 2 points.
    """
    if len(equity_curve) < 2 or not trades:
        return 0.0

    total_return = equity_curve[-1] / capital - 1.0

    try:
        duration_days = (trades[-1].exit_time - trades[0].entry_time).days
    except (AttributeError, TypeError):
        duration_days = len(trades)

    years = max(duration_days / 365.25, 1.0 / 365.25)
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _aggregate_to_daily(trades: list[TradeRecord]) -> np.ndarray:
    """Aggregate trade-level P&L to calendar-day P&L.

    Groups trades by ``exit_time.date()`` (the day the trade closes).
    Used to avoid inflating the number of observations in Sharpe computation
    when many trades exit on the same day.

    Returns a 1-D float array of daily net P&L values.
    """
    daily: dict = defaultdict(float)
    for t in trades:
        try:
            day = t.exit_time.date()
        except AttributeError:
            # exit_time is already a date object.
            day = t.exit_time
        daily[day] += t.pnl_inr
    return np.array(list(daily.values()), dtype=float)


def _pct_profitable_months(trades: list[TradeRecord]) -> float:
    """Return the fraction of calendar months with net positive P&L.

    A month is identified by ``(year, month)`` of the trade's ``exit_time``.
    Returns 0.0 if *trades* is empty.
    """
    monthly: dict = defaultdict(float)
    for t in trades:
        et = t.exit_time
        key = (getattr(et, "year", 0), getattr(et, "month", 0))
        monthly[key] += t.pnl_inr

    if not monthly:
        return 0.0

    positive = sum(1 for v in monthly.values() if v > 0)
    return positive / len(monthly)


def _sortino_ratio(daily_pnls: np.ndarray, capital: float) -> float:
    """Compute annualised Sortino ratio (rf=0, downside threshold=0).

    Formula:
        sortino = sqrt(252) * mean_daily_return / downside_deviation

    Downside deviation is computed from the semi-variance of daily returns:
        downside_deviation = sqrt(mean(min(r_i, 0)^2))

    This uses all N observations in the denominator (not just negative ones),
    which is the standard Sortino formulation (Sortino & van der Meer 1991).

    Returns 0.0 when fewer than 2 observations are available.
    """
    if len(daily_pnls) < 2:
        return 0.0
    daily_returns = daily_pnls / capital
    mean_r = float(np.mean(daily_returns))
    # Semi-variance: mean of squared below-zero returns across all N points
    neg_returns = np.minimum(daily_returns, 0.0)
    downside_var = float(np.mean(neg_returns ** 2))
    if downside_var < 1e-20:
        # No downside variance → either all positive (return inf for positive mean)
        # or flat (return 0.0)
        return float("inf") if mean_r > 1e-10 else 0.0
    downside_std = math.sqrt(downside_var)
    return float(math.sqrt(252) * mean_r / downside_std)


def _var_cvar_95(daily_pnls: np.ndarray) -> tuple[float, float]:
    """Compute empirical 1-day VaR and CVaR at 95% confidence level.

    Requires at least 20 daily observations for a meaningful estimate.
    Returns (0.0, 0.0) for insufficient data.

    Returns
    -------
    tuple[float, float]
        ``(var_95_inr, cvar_95_inr)`` — both positive numbers representing
        expected loss magnitude.

        - *var_95_inr*: the 5th percentile of the daily P&L distribution,
          negated (so a positive number represents a loss).
          Interpretation: "We lose at least this much on 5% of trading days."
        - *cvar_95_inr*: the mean of daily P&Ls at or below the VaR threshold.
          Interpretation: "On the worst 5% of days, we lose this much on average."
    """
    if len(daily_pnls) < 20:
        return (0.0, 0.0)
    var_threshold = float(np.percentile(daily_pnls, 5))
    var_inr = -var_threshold          # positive = loss magnitude
    tail = daily_pnls[daily_pnls <= var_threshold]
    cvar_inr = -float(np.mean(tail)) if len(tail) > 0 else var_inr
    return (var_inr, cvar_inr)


def _max_consecutive_losses(trades: list[TradeRecord]) -> int:
    """Count the longest streak of consecutive losing trades.

    A trade is a loss if ``pnl_inr < 0``.  Break-even trades (pnl_inr == 0)
    do not extend a losing streak but also do not reset it — they are treated
    as neutral (streak resets on any non-negative P&L).
    """
    if not trades:
        return 0
    max_streak = 0
    current_streak = 0
    for t in trades:
        if t.pnl_inr < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def _empty_metrics(capital: float, min_sharpe_threshold: float) -> MetricsSuite:
    """Return an all-zero :class:`MetricsSuite` for the empty-trade case."""
    return MetricsSuite(
        total_pnl_inr=0.0,
        annualized_return_pct=0.0,
        sharpe_ratio=0.0,
        sharpe_ci_95=(0.0, 0.0),
        calmar_ratio=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_inr=0.0,
        avg_drawdown_pct=0.0,
        max_drawdown_duration_days=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        avg_win_inr=0.0,
        avg_loss_inr=0.0,
        profit_factor=0.0,
        expectancy_per_trade_inr=0.0,
        sortino_ratio=0.0,
        var_95_inr=0.0,
        cvar_95_inr=0.0,
        max_consecutive_losses=0,
        pct_profitable_months=0.0,
        fold_sharpe_std=0.0,
        passes_min_trade_count=False,
        passes_sharpe_threshold=False,
        passes_profit_factor=False,
    )
