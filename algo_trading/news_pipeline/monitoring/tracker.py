"""
PipelineTracker: tracks Prefect flow runs and sends Telegram alerts on failure.

Usage in Prefect flows:
    async with PipelineTracker.run("ingest_flow", pg_store=store) as tracker:
        items = await fetch_something()
        tracker.add_records(len(items))
        # on exception: tracker auto-records failure
    # on exit: auto-writes to DB and checks alert threshold
"""
from __future__ import annotations

import time
import uuid
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog

from algo_trading.news_pipeline.models import PipelineRun, make_id
from algo_trading.news_pipeline.config import cfg

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------

class LatencyProfiler:
    """
    Synchronous context manager for measuring wall-clock latency of a code block.

    Usage::

        with LatencyProfiler("fetch_pulse") as lp:
            data = fetch()
        print(lp.elapsed_ms)   # float, milliseconds
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "LatencyProfiler":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()
        log.debug(
            "latency_profiler.finished",
            name=self.name,
            elapsed_ms=round(self.elapsed_ms, 3),
        )

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock duration in milliseconds. 0.0 if context has not exited yet."""
        end = self._end if self._end > 0.0 else time.perf_counter()
        return (end - self._start) * 1_000.0


# ---------------------------------------------------------------------------
# FailureCounter
# ---------------------------------------------------------------------------

class FailureCounter:
    """
    Thread-safe consecutive failure counter keyed by flow name.
    Resets to 0 on success.  Used to gate Telegram alert delivery after
    cfg.MAX_CONSECUTIVE_FAILURES consecutive failures.
    """

    _counts: dict[str, int] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def increment(cls, flow_name: str) -> int:
        """Increment the counter for *flow_name* and return the new value."""
        with cls._lock:
            cls._counts[flow_name] = cls._counts.get(flow_name, 0) + 1
            return cls._counts[flow_name]

    @classmethod
    def reset(cls, flow_name: str) -> None:
        """Reset the counter to 0 (called on a successful run)."""
        with cls._lock:
            cls._counts[flow_name] = 0

    @classmethod
    def get(cls, flow_name: str) -> int:
        """Return the current consecutive failure count for *flow_name*."""
        with cls._lock:
            return cls._counts.get(flow_name, 0)


# ---------------------------------------------------------------------------
# PipelineTracker
# ---------------------------------------------------------------------------

class PipelineTracker:
    """
    Tracks a single Prefect (or equivalent) flow run end-to-end.

    Responsibilities
    ----------------
    - Generates a unique run_id (UUID4) on construction.
    - Records start time, end time, status, records processed, and error messages.
    - Writes a ``PipelineRun`` record to PostgreSQL via *pg_store* when provided.
    - Sends a Telegram alert via ``algo_trading.alerts.telegram_bot.send_alert``
      if consecutive failures reach ``cfg.MAX_CONSECUTIVE_FAILURES``.

    Manual usage::

        tracker = PipelineTracker("ingest_flow", pg_store=store)
        tracker.start()
        tracker.add_records(50)
        tracker.add_error("fetch failed: timeout")
        run = await tracker.finish(success=False)

    Context-manager usage::

        async with PipelineTracker.run("ingest_flow", pg_store=store) as t:
            items = await fetch()
            t.add_records(len(items))
    """

    def __init__(
        self,
        flow_name: str,
        pg_store: Any = None,   # PGStore | None — typed as Any to avoid circular import
        run_id: str | None = None,
    ) -> None:
        self.flow_name: str = flow_name
        self._pg_store: Any = pg_store
        self.run_id: str = run_id if run_id is not None else str(uuid.uuid4())

        self._started_at: datetime | None = None
        self._records_processed: int = 0
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def start(self) -> "PipelineTracker":
        """Record the start timestamp and log the run as begun. Returns self for chaining."""
        self._started_at = datetime.now(tz=timezone.utc)
        log.info(
            "pipeline_tracker.started",
            flow_name=self.flow_name,
            run_id=self.run_id,
            started_at=self._started_at.isoformat(),
        )
        return self

    def add_records(self, count: int) -> None:
        """Increment the running total of records processed by *count*."""
        self._records_processed += count

    def add_error(self, msg: str) -> None:
        """Append a human-readable error message to the error list."""
        self._errors.append(msg)
        log.warning(
            "pipeline_tracker.error_recorded",
            flow_name=self.flow_name,
            run_id=self.run_id,
            error=msg,
        )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    async def finish(self, success: bool = True) -> PipelineRun:
        """
        Finalize the run record.

        - Computes ``latency_ms`` from start to now.
        - Determines ``status`` as "completed" or "failed".
        - Writes the ``PipelineRun`` to the DB when *pg_store* is available.
        - Updates ``FailureCounter`` (reset on success, increment on failure).
        - Sends a Telegram alert if the consecutive failure threshold is met.
        - Returns the completed ``PipelineRun`` dataclass.
        """
        if self._started_at is None:
            # Guard against calling finish() without start().
            self._started_at = datetime.now(tz=timezone.utc)
            log.warning(
                "pipeline_tracker.finish_called_without_start",
                flow_name=self.flow_name,
                run_id=self.run_id,
            )

        completed_at = datetime.now(tz=timezone.utc)
        latency_ms = (completed_at - self._started_at).total_seconds() * 1_000.0
        status = "completed" if success else "failed"

        run = PipelineRun(
            run_id=self.run_id,
            flow_name=self.flow_name,
            started_at=self._started_at,
            completed_at=completed_at,
            status=status,
            records_processed=self._records_processed,
            errors=list(self._errors),
            latency_ms=round(latency_ms, 3),
        )

        # Persist to PostgreSQL
        if self._pg_store is not None:
            try:
                await self._pg_store.insert_pipeline_run(run)
            except Exception as exc:
                log.error(
                    "pipeline_tracker.db_write_failed",
                    flow_name=self.flow_name,
                    run_id=self.run_id,
                    error=str(exc),
                )

        # Update failure counter and conditionally alert
        if success:
            FailureCounter.reset(self.flow_name)
            log.info(
                "pipeline_tracker.finished",
                flow_name=self.flow_name,
                run_id=self.run_id,
                status=status,
                records_processed=self._records_processed,
                latency_ms=run.latency_ms,
            )
        else:
            count = FailureCounter.increment(self.flow_name)
            last_error = self._errors[-1] if self._errors else "unknown error"
            log.error(
                "pipeline_tracker.finished",
                flow_name=self.flow_name,
                run_id=self.run_id,
                status=status,
                consecutive_failures=count,
                records_processed=self._records_processed,
                latency_ms=run.latency_ms,
            )
            await self._send_alert_if_needed(self.flow_name, last_error)

        return run

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def run(
        cls,
        flow_name: str,
        pg_store: Any = None,
    ) -> AsyncIterator["PipelineTracker"]:
        """
        Async context manager.

        - Calls ``start()`` on entry.
        - Calls ``finish(success=True)`` on clean exit.
        - Calls ``finish(success=False)`` and records the exception message on error.
        - Re-raises the original exception after recording, so Prefect/calling code
          can observe the failure normally.
        """
        tracker = cls(flow_name=flow_name, pg_store=pg_store)
        tracker.start()
        try:
            yield tracker
        except Exception as exc:
            tracker.add_error(f"{type(exc).__name__}: {exc}")
            await tracker.finish(success=False)
            raise
        else:
            await tracker.finish(success=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_run(self, status: str) -> PipelineRun:
        """
        Construct a ``PipelineRun`` snapshot using the current tracker state.

        ``completed_at`` is set to now; ``latency_ms`` is computed from ``_started_at``.
        This is a pure helper — it does NOT write to the DB or update counters.
        """
        now = datetime.now(tz=timezone.utc)
        started = self._started_at if self._started_at is not None else now
        latency_ms = (now - started).total_seconds() * 1_000.0
        return PipelineRun(
            run_id=self.run_id,
            flow_name=self.flow_name,
            started_at=started,
            completed_at=now,
            status=status,
            records_processed=self._records_processed,
            errors=list(self._errors),
            latency_ms=round(latency_ms, 3),
        )

    async def _send_alert_if_needed(self, flow_name: str, error_msg: str) -> None:
        """
        Send a Telegram alert when consecutive failures reach the configured threshold.

        Uses ``algo_trading.alerts.telegram_bot.send_alert`` (ZMQ fire-and-forget).
        If the bot token is empty or the module is unavailable, logs and returns silently.
        """
        count = FailureCounter.get(flow_name)
        if count < cfg.MAX_CONSECUTIVE_FAILURES:
            return

        # Skip alerting if token is not configured — avoids spurious errors in CI/CD.
        if not cfg.TELEGRAM_BOT_TOKEN:
            log.warning(
                "pipeline_tracker.alert_skipped_no_token",
                flow_name=flow_name,
                consecutive_failures=count,
            )
            return

        message = (
            f"Pipeline failure: {flow_name} failed {count} times in a row. "
            f"Last error: {error_msg[:200]}"
        )

        try:
            from algo_trading.alerts.telegram_bot import send_alert  # noqa: PLC0415
            send_alert(message, level="CRITICAL")
            log.info(
                "pipeline_tracker.alert_sent",
                flow_name=flow_name,
                consecutive_failures=count,
            )
        except Exception as exc:
            # Alert failures must never propagate — just log and continue.
            log.warning(
                "pipeline_tracker.alert_failed",
                flow_name=flow_name,
                error=str(exc),
                alert_message=message,
            )


# ---------------------------------------------------------------------------
# PipelineHealthSummary
# ---------------------------------------------------------------------------

class PipelineHealthSummary:
    """
    Computes an aggregated health summary from recent pipeline run records.
    Intended for consumption by the FastAPI ``/health`` endpoint.
    """

    @staticmethod
    async def compute(pg_store: Any) -> dict:
        """
        Query ``pg_store.get_recent_runs()`` for each known flow and build a
        health summary dict.

        Returns
        -------
        {
            "flows": {
                "<flow_name>": {
                    "last_run":            "<ISO-8601 timestamp> | null",
                    "last_status":         "completed" | "failed" | "running" | "unknown",
                    "success_rate_pct":    float,   # 0.0 – 100.0
                    "avg_latency_ms":      float,
                    "consecutive_failures": int,
                },
                ...
            },
            "overall_healthy": bool,  # True when every flow succeeded within the last hour
        }

        A flow is considered *unhealthy* if its most recent run has status
        ``"failed"`` or if no run has been recorded in the last 60 minutes.
        The ``overall_healthy`` flag is ``True`` only when all known flows are healthy.
        If *pg_store* is unavailable or raises, the method returns a safe empty dict.
        """
        from datetime import timedelta  # local import keeps top-level clean

        now_utc = datetime.now(tz=timezone.utc)
        one_hour_ago = now_utc - timedelta(hours=1)

        try:
            # Fetch up to 20 recent runs across all flows (no flow_name filter)
            all_runs: list[dict] = await pg_store.get_recent_runs(limit=20)
        except Exception as exc:
            log.error("pipeline_health_summary.fetch_failed", error=str(exc))
            return {"flows": {}, "overall_healthy": False}

        # Group runs by flow_name
        grouped: dict[str, list[dict]] = {}
        for row in all_runs:
            fn = row.get("flow_name", "unknown")
            grouped.setdefault(fn, []).append(row)

        # Also include flows that have in-memory failure counts but no DB rows yet.
        for fn in list(FailureCounter._counts.keys()):
            grouped.setdefault(fn, [])

        flows_summary: dict[str, dict] = {}
        all_healthy = True

        for flow_name, runs in grouped.items():
            if not runs:
                # Flow is known (via FailureCounter) but has no DB records yet.
                flows_summary[flow_name] = {
                    "last_run": None,
                    "last_status": "unknown",
                    "success_rate_pct": 0.0,
                    "avg_latency_ms": 0.0,
                    "consecutive_failures": FailureCounter.get(flow_name),
                }
                all_healthy = False
                continue

            # Most-recent run is first (pg_store orders by started_at DESC)
            latest = runs[0]
            last_status: str = latest.get("status", "unknown")

            # Parse last_run timestamp
            last_run_raw = latest.get("started_at")
            last_run_iso: str | None = None
            last_run_dt: datetime | None = None
            if last_run_raw:
                try:
                    if isinstance(last_run_raw, str):
                        last_run_dt = datetime.fromisoformat(last_run_raw)
                    elif isinstance(last_run_raw, datetime):
                        last_run_dt = last_run_raw
                    if last_run_dt is not None:
                        # Normalise to UTC-aware for comparison
                        if last_run_dt.tzinfo is None:
                            last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
                        last_run_iso = last_run_dt.isoformat()
                except (ValueError, TypeError):
                    last_run_dt = None

            # Compute success rate across the sample of rows
            total = len(runs)
            successes = sum(
                1 for r in runs if r.get("status") == "completed"
            )
            success_rate_pct = round((successes / total) * 100.0, 2) if total > 0 else 0.0

            # Average latency across available runs (skip None/zero)
            latencies = [
                float(r["latency_ms"])
                for r in runs
                if r.get("latency_ms") is not None
            ]
            avg_latency_ms = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

            # Determine whether this flow is healthy:
            #   healthy = last status is "completed" AND it ran within the last hour
            flow_healthy = (
                last_status == "completed"
                and last_run_dt is not None
                and last_run_dt >= one_hour_ago
            )
            if not flow_healthy:
                all_healthy = False

            flows_summary[flow_name] = {
                "last_run": last_run_iso,
                "last_status": last_status,
                "success_rate_pct": success_rate_pct,
                "avg_latency_ms": avg_latency_ms,
                "consecutive_failures": FailureCounter.get(flow_name),
            }

        return {
            "flows": flows_summary,
            "overall_healthy": all_healthy,
        }
