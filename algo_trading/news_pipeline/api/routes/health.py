"""Health endpoints — /health."""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, Request

from algo_trading.news_pipeline.api.schemas import HealthResponse
from algo_trading.news_pipeline.monitoring.tracker import PipelineHealthSummary

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    """Pipeline health check.

    Queries ``PipelineHealthSummary.compute(pg)`` which inspects recent
    pipeline run records from PostgreSQL and the in-process
    ``FailureCounter`` singletons.

    Response fields
    ---------------
    - **status**: ``"healthy"`` when all flows are green, ``"degraded"`` otherwise.
    - **flows**: per-flow breakdown with last_run, last_status, success_rate_pct,
      avg_latency_ms, and consecutive_failures.
    - **overall_healthy**: mirrors the ``overall_healthy`` flag from the summary.
    - **checked_at**: UTC timestamp of this health check.
    """
    try:
        pg = request.app.state.pg
        summary = await PipelineHealthSummary.compute(pg)

        overall_healthy: bool = summary.get("overall_healthy", False)
        status = "healthy" if overall_healthy else "degraded"

        return HealthResponse(
            status=status,
            flows=summary.get("flows", {}),
            overall_healthy=overall_healthy,
            checked_at=datetime.now(tz=timezone.utc),
        )
    except Exception as exc:
        log.error("api.get_health.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Health check failed") from exc


@router.get("/flows/{flow_name}")
async def get_flow_health(flow_name: str, request: Request) -> dict:
    """Health detail for a specific pipeline flow.

    Returns the per-flow dict from ``PipelineHealthSummary`` for the named flow.
    Returns 404 if no data exists for *flow_name*.

    Response fields
    ---------------
    - **flow_name**: echo of the requested flow.
    - **last_run**: ISO-8601 timestamp of the last run start (or null).
    - **last_status**: ``"completed"`` | ``"failed"`` | ``"running"`` | ``"unknown"``.
    - **success_rate_pct**: success percentage over the sampled window (0–100).
    - **avg_latency_ms**: average wall-clock duration in milliseconds.
    - **consecutive_failures**: current count from the in-process failure counter.
    - **healthy**: derived boolean (True when last_status is completed and recent).
    """
    try:
        pg = request.app.state.pg
        summary = await PipelineHealthSummary.compute(pg)
        flows: dict = summary.get("flows", {})

        if flow_name not in flows:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for flow '{flow_name}'",
            )

        flow_data: dict = flows[flow_name]
        # Annotate with flow_name and derived healthy flag
        return {
            "flow_name": flow_name,
            **flow_data,
            "healthy": flow_data.get("last_status") == "completed"
                       and flow_data.get("consecutive_failures", 0) == 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("api.get_flow_health.failed", flow_name=flow_name, error=str(exc))
        raise HTTPException(status_code=500, detail="Flow health check failed") from exc
