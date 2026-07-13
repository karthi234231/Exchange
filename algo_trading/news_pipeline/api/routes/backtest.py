"""Backtest endpoints — /backtest."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from algo_trading.news_pipeline.api.schemas import BacktestSummaryResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary", response_model=BacktestSummaryResponse)
async def get_backtest_summary(
    request: Request,
    days: int = Query(30, ge=7, le=365, description="Evaluation window in calendar days"),
) -> BacktestSummaryResponse:
    """Get signal performance summary for the last *days* calendar days.

    Delegates to ``SignalBacktester.summary_report(days)`` which reads from
    the ``signal_performance`` table in PostgreSQL.

    Response fields
    ---------------
    - **period_days**: requested evaluation window.
    - **total_evaluated**: number of directional signals evaluated (NEUTRAL excluded).
    - **win_rate**: fraction of signals that predicted direction correctly (0–1).
    - **avg_return_pct**: average per-trade return in percent.
    - **sharpe_ratio**: annualised Sharpe ratio (mean/std * sqrt(252)).
    - **max_drawdown**: maximum peak-to-trough drawdown of the equity curve.
    - **by_signal_type**: per-type breakdown (STRONG_BUY, BUY, SELL, STRONG_SELL).
    - **by_symbol**: per-symbol breakdown.
    - **best_signal_type**: signal type with highest win_rate.
    - **best_symbol**: symbol with highest win_rate.
    - **recent_10**: list of the 10 most recent ``SignalPerformance`` records.
    """
    try:
        backtester = request.app.state.backtester
        report = await backtester.summary_report(days=days)

        return BacktestSummaryResponse(
            period_days=report.get("period_days", days),
            total_evaluated=report.get("total_evaluated", 0),
            win_rate=float(report.get("win_rate", 0.0)),
            avg_return_pct=float(report.get("avg_return_pct", 0.0)),
            sharpe_ratio=float(report.get("sharpe_ratio", 0.0)),
            max_drawdown=float(report.get("max_drawdown", 0.0)),
            by_signal_type=report.get("by_signal_type", {}),
            by_symbol=report.get("by_symbol", {}),
            best_signal_type=report.get("best_signal_type", ""),
            best_symbol=report.get("best_symbol", ""),
            recent_10=report.get("recent_10", []),
        )
    except Exception as exc:
        log.error("api.get_backtest_summary.failed", days=days, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to compute backtest summary") from exc
