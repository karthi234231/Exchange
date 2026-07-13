"""
algo_trading/monitoring/alert_deduplicator.py — Critical alert deduplication.

Prevents alert spam: if the same alert fires 10 times in 5 minutes (e.g., repeated
WebSocket disconnects), only send 1 alert, then suppress duplicates.

Deduplication key: (alert_type, symbol) — same type on same symbol = duplicate.

Policy:
  - First occurrence: send immediately
  - Repeats within window: suppress, increment counter
  - After window expires: send one summary "Alert fired N times in last X minutes"
  - Critical alerts (HALT, DRAWDOWN): never deduplicate — always send
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Default deduplication window.  ASSUMED: 5 minutes (300 seconds).
_DEFAULT_DEDUP_WINDOW_SECONDS: float = 300.0


class AlertPriority(str, Enum):
    """Priority level for an alert.

    CRITICAL alerts bypass deduplication and are always delivered.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertRecord:
    """State tracked for a single (alert_type, symbol) deduplication key.

    Attributes:
        alert_type: Logical category of the alert (e.g., "BROKER_DISCONNECT").
        symbol: NSE instrument symbol associated with the alert, or "" for
            system-level alerts not tied to a specific instrument.
        message: Most recent alert message text.
        priority: Priority at first fire.
        first_fired_at: Monotonic timestamp (seconds) when first seen.
        fire_count: Total number of times this alert has fired.
        last_fired_at: Monotonic timestamp of the most recent fire.
        suppressed_count: How many fires were suppressed (not sent).
    """

    alert_type: str
    symbol: str
    message: str
    priority: AlertPriority
    first_fired_at: float
    fire_count: int = 1
    last_fired_at: float = field(default=0.0)
    suppressed_count: int = 0

    def __post_init__(self) -> None:
        if self.last_fired_at == 0.0:
            self.last_fired_at = self.first_fired_at


class AlertDeduplicator:
    """Stateful alert deduplicator with configurable suppression window.

    Thread-safe: all state mutations are serialised through an internal lock.

    Args:
        dedup_window_seconds: Duration (seconds) of the suppression window.
            Within this window, duplicate (alert_type, symbol) pairs are
            suppressed. ASSUMED default: 300.0 (5 minutes).

    Usage::

        dedup = AlertDeduplicator()
        should_send, msg = dedup.should_send(
            "BROKER_DISCONNECT", "NIFTY25JUNFUT",
            "WebSocket dropped", AlertPriority.WARNING
        )
        if should_send:
            telegram.send(msg)

        # Periodically flush expired summaries
        for summary in dedup.get_suppression_summary():
            telegram.send(summary)
        dedup.clear_expired()
    """

    def __init__(self, dedup_window_seconds: float = _DEFAULT_DEDUP_WINDOW_SECONDS) -> None:
        self._window = dedup_window_seconds
        self._records: dict[tuple[str, str], AlertRecord] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_send(
        self,
        alert_type: str,
        symbol: str,
        message: str,
        priority: AlertPriority,
    ) -> tuple[bool, str]:
        """Determine whether this alert should be delivered.

        CRITICAL alerts are always delivered — deduplication is bypassed.
        For all other priorities:
          - First occurrence within the window: send immediately.
          - Subsequent occurrences within the window: suppress.
          - First occurrence after window expiry: send (treated as new event).

        Args:
            alert_type: Logical alert category (e.g., "BROKER_DISCONNECT").
            symbol: NSE instrument symbol, or "" for system-level alerts.
            message: Human-readable alert text.
            priority: Priority level; CRITICAL bypasses suppression.

        Returns:
            (True, message) if the alert should be sent.
            (False, "") if the alert is suppressed.
        """
        # CRITICAL: always send, never track
        if priority == AlertPriority.CRITICAL:
            log.debug(
                "alert_dedup.critical_bypass",
                alert_type=alert_type,
                symbol=symbol,
            )
            return True, message

        key = (alert_type, symbol)
        now = time.monotonic()

        with self._lock:
            record = self._records.get(key)

            if record is None:
                # First occurrence — record and send
                self._records[key] = AlertRecord(
                    alert_type=alert_type,
                    symbol=symbol,
                    message=message,
                    priority=priority,
                    first_fired_at=now,
                    fire_count=1,
                    last_fired_at=now,
                    suppressed_count=0,
                )
                log.debug(
                    "alert_dedup.first_occurrence",
                    alert_type=alert_type,
                    symbol=symbol,
                )
                return True, message

            # Check if window has expired
            window_age = now - record.first_fired_at
            if window_age > self._window:
                # Window expired — treat this as a fresh event
                suppressed = record.suppressed_count
                self._records[key] = AlertRecord(
                    alert_type=alert_type,
                    symbol=symbol,
                    message=message,
                    priority=priority,
                    first_fired_at=now,
                    fire_count=1,
                    last_fired_at=now,
                    suppressed_count=0,
                )
                log.debug(
                    "alert_dedup.window_expired",
                    alert_type=alert_type,
                    symbol=symbol,
                    prev_suppressed=suppressed,
                )
                return True, message

            # Within window — suppress
            record.fire_count += 1
            record.last_fired_at = now
            record.suppressed_count += 1
            record.message = message  # keep latest message text
            log.debug(
                "alert_dedup.suppressed",
                alert_type=alert_type,
                symbol=symbol,
                fire_count=record.fire_count,
            )
            return False, ""

    def get_suppression_summary(self) -> list[str]:
        """Return summary messages for alerts whose window has just expired.

        Only returns summaries for records that have at least one suppressed
        fire (i.e., were deduplicated at least once).

        Returns:
            List of human-readable summary strings ready to send as alerts.
        """
        summaries: list[str] = []
        now = time.monotonic()

        with self._lock:
            for key, record in list(self._records.items()):
                window_age = now - record.first_fired_at
                if window_age > self._window and record.suppressed_count > 0:
                    window_minutes = self._window / 60.0
                    sym_part = f" [{record.symbol}]" if record.symbol else ""
                    summaries.append(
                        f"[DEDUP SUMMARY] {record.alert_type}{sym_part} fired "
                        f"{record.fire_count} times in the last "
                        f"{window_minutes:.0f} min "
                        f"({record.suppressed_count} suppressed). "
                        f"Last message: {record.message}"
                    )
                    log.info(
                        "alert_dedup.summary_generated",
                        alert_type=record.alert_type,
                        symbol=record.symbol,
                        fire_count=record.fire_count,
                        suppressed=record.suppressed_count,
                    )

        return summaries

    def clear_expired(self) -> None:
        """Remove all records whose deduplication window has fully expired.

        Call this periodically (e.g., every minute) to prevent unbounded
        memory growth.
        """
        now = time.monotonic()
        with self._lock:
            expired_keys = [
                key
                for key, record in self._records.items()
                if (now - record.first_fired_at) > self._window
            ]
            for key in expired_keys:
                del self._records[key]
            if expired_keys:
                log.debug(
                    "alert_dedup.cleared_expired",
                    count=len(expired_keys),
                )

    def active_record_count(self) -> int:
        """Return the number of currently tracked alert keys (for monitoring).

        Returns:
            Number of (alert_type, symbol) pairs currently in the window.
        """
        with self._lock:
            return len(self._records)
