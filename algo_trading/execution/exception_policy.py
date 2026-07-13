"""
execution/exception_policy.py — P2-FIX-97: Explicit fail-open/fail-closed policy.

Every component in the trading system must have a documented and enforced
exception policy.  The wrong default (fail-open when fail-closed is required,
or fail-closed when fail-open is acceptable) can cause either capital loss or
unnecessary trading halts.

Policy table:
  MARKET DATA (entries)      → FAIL CLOSED
    - If tick/LTP is missing, block new entries.  A degraded price signal is
      worse than no signal.  Exits remain allowed via a separate path.

  RISK GUARD (can_trade)     → FAIL CLOSED
    - If risk checks raise, block the order.  Never let an unchecked order through.

  FAT-FINGER GUARD           → FAIL CLOSED
    - If guard raises, block the order.  A missing check is as dangerous as a
      failed check.

  ORDER PLACEMENT (entries)  → FAIL CLOSED
    - Broker exceptions map to REJECT FSM state.  Never retry silently.

  EXITS / STOP-LOSS          → FAIL OPERATIONALLY SAFE
    - Stop-loss and force-close orders use MARKET type with no price cap.
    - If the first cancel attempt fails, log and alert — do not give up silently.
    - Exits can operate even during data quality halt (kills only NEW ENTRIES).

  STORAGE (QuestDB)          → CONTINUE TRADING IF AUDIT LOG IS DURABLE
    - QuestDB write failures are logged but do not block trading.
    - The local JSONL audit log (state/order_audit.log) must be writable.
    - If audit log write fails, log at ERROR — operator review required.

  ALERTS (Telegram)          → FAIL OPEN
    - Telegram failures must never affect trading.  Alerts are best-effort.
    - The ZMQ PUSH sender is fire-and-forget with NOBLOCK flag.

  MONITORING (Prometheus)    → FAIL OPEN
    - Counter/gauge increment failures are silently ignored.
    - Monitoring failure must never halt trading.
"""
from __future__ import annotations

import functools
import logging
from enum import Enum, auto
from typing import Callable, TypeVar, Any

import structlog

log = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class FailPolicy(Enum):
    """Enumeration of exception policies for trading system components."""

    CLOSED = auto()   # Exception → block the action. Returns None/False.
    OPEN = auto()     # Exception → allow the action. Logs at WARNING.
    SAFE = auto()     # Exception → log + alert, attempt graceful fallback.


def fail_closed(component: str) -> Callable[[F], F]:
    """
    Decorator: exception → return None and log at ERROR.

    Use for: risk checks, fat-finger guard, order placement.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log.error(
                    "exception_policy.fail_closed",
                    component=component,
                    function=func.__qualname__,
                    error=str(exc),
                    note="Action BLOCKED (fail-closed policy)",
                )
                return None
        return wrapper  # type: ignore[return-value]
    return decorator


def fail_open(component: str) -> Callable[[F], F]:
    """
    Decorator: exception → swallow and log at WARNING. Action proceeds.

    Use for: Telegram alerts, Prometheus metrics, non-critical logging.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log.warning(
                    "exception_policy.fail_open",
                    component=component,
                    function=func.__qualname__,
                    error=str(exc),
                    note="Action ALLOWED despite exception (fail-open policy)",
                )
                return None
        return wrapper  # type: ignore[return-value]
    return decorator


def require_audit_log_writable(audit_path: str) -> bool:
    """
    P2-FIX-97: Verify the SEBI compliance audit log path is writable.

    Checks that the parent directory exists and is writable.
    If the audit log is not writable, trading must not proceed — a missing
    audit trail violates SEBI PFUTP §22 (5-year trade record retention).

    Returns True if writable, False if not.
    """
    from pathlib import Path
    try:
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file to verify write access without truncating existing content
        p.touch(exist_ok=True)
        # Attempt a zero-byte write to confirm the filesystem isn't read-only
        with p.open("a", encoding="utf-8") as fh:
            fh.write("")
        return True
    except Exception as exc:
        log.error(
            "exception_policy.audit_log_not_writable",
            path=audit_path,
            error=str(exc),
            note="Trading must not proceed — SEBI audit trail is not writable",
        )
        return False


def safe_exit_action(action_name: str, alert_sender: Any, action_fn: Callable[[], Any]) -> bool:
    """
    P2-FIX-97: Execute a safety-critical exit action (stop-loss / force-close).

    Wraps the action in try/except with:
      1. First attempt — standard execution
      2. On failure — log at CRITICAL, alert operator, return False (caller retries)

    Exits are operationally safe: we attempt but do not suppress failures.
    The system must stay running to attempt cleanup even when individual
    exit attempts fail.

    Returns True on success, False on failure.
    """
    try:
        action_fn()
        return True
    except Exception as exc:
        log.critical(
            "exception_policy.exit_action_failed",
            action=action_name,
            error=str(exc),
            note="EXIT ACTION FAILED — operator intervention may be required",
        )
        try:
            if alert_sender is not None:
                alert_sender.send_risk_alert(
                    f"CRITICAL: Exit action failed: {action_name} — {exc}"
                )
        except Exception:
            pass  # Alert failure must never suppress the original error log
        return False
