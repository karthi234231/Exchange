"""
algo_trading/risk/kill_switch_manager.py — Multi-level kill switch with audit log.

Kill switch hierarchy (FO_SYSTEM_DESIGN.md Section 9.5):
  Level 1 — Strategy kill:  halts one strategy, others continue
  Level 2 — Portfolio kill: halts ALL trading on ALL strategies
  Level 3 — Human override: operator-triggered, requires manual reset

Once triggered, a kill switch BLOCKS all can_trade() calls until explicitly
reset (with operator confirmation for Level 3).

All activations and resets are written to an append-only audit log. This log
is critical for SEBI compliance (7-year retention requirement).

Integration:
  - RiskManager.can_trade() must call KillSwitchManager.is_trading_allowed()
  - OrderExecutor must call KillSwitchManager.is_trading_allowed() as first check
  - AlertSink must receive CRITICAL alert on any kill switch trigger

This module is intentionally stateless on its own — state is persisted to
the audit log and reconstructed at startup via load_state().
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class KillLevel(str, Enum):
    STRATEGY  = "STRATEGY"   # Halt one strategy only
    PORTFOLIO = "PORTFOLIO"  # Halt ALL strategies
    HUMAN     = "HUMAN"      # Human override — requires manual reset


class KillReason(str, Enum):
    DAILY_LOSS_LIMIT     = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN         = "MAX_DRAWDOWN"
    DATA_QUALITY         = "DATA_QUALITY"
    BROKER_ERROR         = "BROKER_ERROR"
    STUCK_LEG            = "STUCK_LEG"
    PARTIAL_ABORT_FAILED = "PARTIAL_ABORT_FAILED"
    VIX_EMERGENCY        = "VIX_EMERGENCY"
    MANUAL               = "MANUAL"
    STARTUP_RECONCILE    = "STARTUP_RECONCILE"
    SESSION_END          = "SESSION_END"
    UNKNOWN              = "UNKNOWN"


@dataclass
class KillEvent:
    """Single append-only audit record."""
    event_id: str
    timestamp: datetime
    event_type: str           # "TRIGGERED" | "RESET"
    level: KillLevel
    reason: KillReason
    strategy_id: str | None   # None for portfolio-level kills
    triggered_by: str         # "SYSTEM" | "HUMAN" | operator name
    details: dict             # free-form details (daily_loss_pct, etc.)
    schema_version: int = 1

    def to_json_line(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "level": self.level.value,
            "reason": self.reason.value,
            "strategy_id": self.strategy_id,
            "triggered_by": self.triggered_by,
            "details": self.details,
            "schema_version": self.schema_version,
        })

    @classmethod
    def from_json_line(cls, line: str) -> "KillEvent":
        d = json.loads(line)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["level"] = KillLevel(d["level"])
        d["reason"] = KillReason(d["reason"])
        return cls(**d)


class KillSwitchManager:
    """
    Multi-level kill switch manager.

    Manages strategy-level and portfolio-level kill switches. Every
    activation and reset is appended to an audit log for SEBI compliance.

    Parameters
    ----------
    audit_log_path : Path
        Path to the JSONL audit log. Append-only.
    alert_callback : callable | None
        Called with (message: str, level: str) on any kill switch event.
        Should route to Telegram/ZMQ alert sink.

    Thread Safety
    -------------
    All public methods are thread-safe via an internal RLock.

    Usage
    -----
    ks = KillSwitchManager(audit_log_path=Path("audit/kill_switch.jsonl"),
                           alert_callback=send_alert)

    # Check before trading
    allowed, reason = ks.is_trading_allowed("iron_condor_v1")
    if not allowed:
        return  # do not trade

    # Trigger from risk manager
    ks.trigger(KillLevel.PORTFOLIO, KillReason.DAILY_LOSS_LIMIT,
               strategy_id=None, triggered_by="SYSTEM",
               details={"daily_loss_pct": -2.1})
    """

    def __init__(
        self,
        audit_log_path: Path | str = "audit/kill_switch.jsonl",
        alert_callback: Any = None,
    ) -> None:
        self._audit_path = Path(audit_log_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._alert_cb = alert_callback
        self._lock = threading.RLock()

        # State (reconstructed from audit log at startup)
        self._portfolio_killed: bool = False
        self._portfolio_kill_reason: KillReason | None = None
        self._strategy_kills: dict[str, KillReason] = {}  # strategy_id → reason
        self._human_override: bool = False

        self._load_state()
        log.info("kill_switch.init",
                 portfolio_killed=self._portfolio_killed,
                 strategy_kills=list(self._strategy_kills.keys()))

    # ── Public API ────────────────────────────────────────────────────────────

    def is_trading_allowed(
        self, strategy_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Parameters
        ----------
        strategy_id : str | None
            Strategy to check. None checks portfolio level only.

        Returns
        -------
        (allowed: bool, reason: str)
            allowed=False means DO NOT TRADE.
        """
        with self._lock:
            if self._human_override:
                return False, "Human override active — manual reset required"
            if self._portfolio_killed:
                return False, f"Portfolio kill switch active: {self._portfolio_kill_reason}"
            if strategy_id and strategy_id in self._strategy_kills:
                return False, f"Strategy '{strategy_id}' kill active: {self._strategy_kills[strategy_id]}"
            return True, "OK"

    def trigger(
        self,
        level: KillLevel,
        reason: KillReason,
        strategy_id: str | None = None,
        triggered_by: str = "SYSTEM",
        details: dict | None = None,
    ) -> None:
        """
        Trigger a kill switch.

        A PORTFOLIO kill overrides all strategy kills.
        A HUMAN kill overrides everything and requires manual reset.

        Parameters
        ----------
        level : KillLevel
            Which level to trigger.
        reason : KillReason
            Why the kill was triggered.
        strategy_id : str | None
            Required for STRATEGY-level kills. Ignored for PORTFOLIO/HUMAN.
        triggered_by : str
            "SYSTEM" or operator name for human overrides.
        details : dict | None
            Supplementary details (loss amounts, etc.) for the audit log.
        """
        import uuid

        with self._lock:
            if level == KillLevel.STRATEGY:
                if not strategy_id:
                    raise ValueError("strategy_id required for STRATEGY-level kill")
                self._strategy_kills[strategy_id] = reason
                msg = f"STRATEGY KILL: {strategy_id} — {reason.value}"

            elif level == KillLevel.PORTFOLIO:
                self._portfolio_killed = True
                self._portfolio_kill_reason = reason
                msg = f"PORTFOLIO KILL — {reason.value} (all strategies halted)"

            elif level == KillLevel.HUMAN:
                self._human_override = True
                self._portfolio_killed = True
                self._portfolio_kill_reason = reason
                msg = f"HUMAN OVERRIDE — {reason.value} — manual reset required"

            event = KillEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(timezone.utc),
                event_type="TRIGGERED",
                level=level,
                reason=reason,
                strategy_id=strategy_id,
                triggered_by=triggered_by,
                details=details or {},
            )
            self._append_audit(event)

        log.critical("kill_switch.triggered",
                     level=level.value, reason=reason.value,
                     strategy=strategy_id, by=triggered_by)
        self._send_alert(msg, "CRITICAL")

    def reset(
        self,
        level: KillLevel,
        strategy_id: str | None = None,
        reset_by: str = "SYSTEM",
        confirm_human: bool = False,
    ) -> bool:
        """
        Reset a kill switch.

        HUMAN-level kills require confirm_human=True.
        Returns True if reset was successful.

        Parameters
        ----------
        confirm_human : bool
            Must be True to reset a HUMAN-level kill. Prevents accidental reset.
        """
        import uuid

        with self._lock:
            if level == KillLevel.HUMAN and not confirm_human:
                log.error("kill_switch.reset_rejected",
                          reason="confirm_human=False required for HUMAN override")
                return False

            if level == KillLevel.STRATEGY:
                if not strategy_id:
                    return False
                removed = self._strategy_kills.pop(strategy_id, None)
                if not removed:
                    return False

            elif level in (KillLevel.PORTFOLIO, KillLevel.HUMAN):
                self._portfolio_killed = False
                self._portfolio_kill_reason = None
                self._human_override = False

            event = KillEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(timezone.utc),
                event_type="RESET",
                level=level,
                reason=KillReason.MANUAL,
                strategy_id=strategy_id,
                triggered_by=reset_by,
                details={"confirm_human": confirm_human},
            )
            self._append_audit(event)

        log.warning("kill_switch.reset",
                    level=level.value, strategy=strategy_id, by=reset_by)
        self._send_alert(
            f"Kill switch RESET: level={level.value}, by={reset_by}", "WARNING"
        )
        return True

    def status(self) -> dict:
        """Return current state summary."""
        with self._lock:
            return {
                "portfolio_killed": self._portfolio_killed,
                "portfolio_kill_reason": (
                    self._portfolio_kill_reason.value
                    if self._portfolio_kill_reason else None
                ),
                "human_override": self._human_override,
                "strategy_kills": {k: v.value for k, v in self._strategy_kills.items()},
            }

    def load_audit_log(self) -> list[KillEvent]:
        """Read the full audit log (for review/compliance)."""
        if not self._audit_path.exists():
            return []
        events = []
        with open(self._audit_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(KillEvent.from_json_line(line))
                    except Exception as e:
                        log.warning("kill_switch.audit_parse_error",
                                    line=line[:80], error=str(e))
        return events

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Reconstruct current state by replaying audit log."""
        events = self.load_audit_log()
        for event in events:
            if event.event_type == "TRIGGERED":
                if event.level == KillLevel.HUMAN:
                    self._human_override = True
                    self._portfolio_killed = True
                    self._portfolio_kill_reason = event.reason
                elif event.level == KillLevel.PORTFOLIO:
                    self._portfolio_killed = True
                    self._portfolio_kill_reason = event.reason
                elif event.level == KillLevel.STRATEGY and event.strategy_id:
                    self._strategy_kills[event.strategy_id] = event.reason
            elif event.event_type == "RESET":
                if event.level in (KillLevel.PORTFOLIO, KillLevel.HUMAN):
                    self._portfolio_killed = False
                    self._portfolio_kill_reason = None
                    self._human_override = False
                elif event.level == KillLevel.STRATEGY and event.strategy_id:
                    self._strategy_kills.pop(event.strategy_id, None)

    def _append_audit(self, event: KillEvent) -> None:
        """Append event to audit log. Thread-safe (called under lock)."""
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")
        except OSError as e:
            # Audit log write failure is critical — alert and raise
            log.critical("kill_switch.audit_write_failed", error=str(e))
            raise  # Do not silently swallow — trading must halt if audit fails

    def _send_alert(self, message: str, level: str = "CRITICAL") -> None:
        if self._alert_cb:
            try:
                self._alert_cb(message, level)
            except Exception:
                pass  # Alert failure must never crash kill switch logic
