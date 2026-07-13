"""algo_trading.news_pipeline.ingestion.base — Abstract base for all fetchers.

All concrete fetchers inherit BaseFetcher and implement _fetch_internal().
The public fetch() method adds a 3-attempt retry loop with exponential
back-off (5 s / 15 s / 30 s) plus ±30% random jitter and measures
wall-clock latency.

Jitter prevents thundering-herd when several fetchers fail simultaneously
(e.g. a brief NSE outage) and all retry at exactly the same second.
"""
from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

_IST = ZoneInfo("Asia/Kolkata")
_BACKOFF_SCHEDULE = (5, 15, 30)   # base seconds between retry attempts
_JITTER_FACTOR    = 0.30          # ±30% random jitter applied to each backoff


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Outcome record returned by every fetch() call alongside the data list."""

    source: str
    success: bool
    records: int
    error: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseFetcher(ABC):
    """Base class for all data fetchers.

    Subclasses implement ``_fetch_internal()`` which contains the raw I/O
    logic.  The public ``fetch()`` wraps that with:
    - Up to ``max_retries`` attempts (default 3).
    - Exponential back-off of 5 s / 15 s / 30 s between attempts.
    - Wall-clock latency measurement.
    - A ``FetchResult`` summary returned alongside the data.

    Usage::

        fetcher = ConcreteSubclass()
        data, result = await fetcher.fetch()
        if result.success:
            process(data)
    """

    max_retries: int = 3

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable identifier for this source, e.g. "pulse"."""

    @abstractmethod
    async def _fetch_internal(self) -> list:
        """Perform the actual I/O and return a list of data objects.

        Implementations MUST NOT raise — they should catch all exceptions
        internally and return an empty list with appropriate logging.
        """

    async def fetch(self) -> tuple[list, FetchResult]:
        """Retry wrapper.  Returns ``(data, FetchResult)``.

        Tries ``_fetch_internal()`` up to ``max_retries`` times.  Each
        failure waits for the corresponding back-off delay from
        ``_BACKOFF_SCHEDULE`` before retrying.  If all attempts fail the
        returned list is empty and ``FetchResult.success`` is False.
        """
        last_error: str | None = None
        t_start = time.monotonic()

        for attempt in range(1, self.max_retries + 1):
            try:
                data = await self._fetch_internal()
                latency_ms = (time.monotonic() - t_start) * 1_000
                log.debug(
                    "fetch_success",
                    source=self.source_name,
                    attempt=attempt,
                    records=len(data),
                    latency_ms=round(latency_ms, 1),
                )
                return data, FetchResult(
                    source=self.source_name,
                    success=True,
                    records=len(data),
                    latency_ms=round(latency_ms, 1),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "fetch_attempt_failed",
                    source=self.source_name,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=last_error,
                )
                if attempt < self.max_retries:
                    base = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
                    jitter = random.uniform(-base * _JITTER_FACTOR, base * _JITTER_FACTOR)
                    delay = max(1.0, base + jitter)
                    await asyncio.sleep(delay)

        latency_ms = (time.monotonic() - t_start) * 1_000
        log.error(
            "fetch_all_attempts_failed",
            source=self.source_name,
            max_retries=self.max_retries,
            error=last_error,
        )
        return [], FetchResult(
            source=self.source_name,
            success=False,
            records=0,
            error=last_error,
            latency_ms=round(latency_ms, 1),
        )

    def is_market_hours(self) -> bool:
        """Return True if the current IST wall-clock time is 09:00–15:35.

        Uses the Asia/Kolkata timezone.  Returns False on any error
        (e.g. missing tzdata) so that callers default to safe behaviour.
        """
        try:
            now_ist = datetime.now(tz=_IST)
            market_open = now_ist.replace(hour=9,  minute=0,  second=0, microsecond=0)
            market_close = now_ist.replace(hour=15, minute=35, second=0, microsecond=0)
            return market_open <= now_ist <= market_close
        except Exception as exc:  # noqa: BLE001
            log.warning("is_market_hours_error", error=str(exc))
            return False
