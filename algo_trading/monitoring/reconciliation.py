"""
algo_trading/monitoring/reconciliation.py — Broker position reconciliation.

Compares internal system state (from EventStore) against broker-reported
positions. Any mismatch = HALT until operator confirms.

Used at:
  1. System startup (mandatory — see SURV-6)
  2. After broker disconnect/reconnect
  3. On-demand by operator

Reconciliation result actions:
  OK_TO_TRADE: positions match, daily limits not hit
  STATE_MISMATCH: positions differ — halt, alert, require manual confirmation
  DAILY_LIMIT_HIT: daily loss limit already reached — halt trading for today
  STALE_STATE: last heartbeat > 30 minutes ago — treat as mismatch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Stale heartbeat threshold — treat as mismatch if last beat older than this.
# ASSUMED: 30 minutes
_STALE_HEARTBEAT_MINUTES: int = 30


class ReconcileStatus(str, Enum):
    """Outcome of a broker reconciliation check."""

    OK_TO_TRADE = "OK_TO_TRADE"
    STATE_MISMATCH = "STATE_MISMATCH"
    DAILY_LIMIT_HIT = "DAILY_LIMIT_HIT"
    STALE_STATE = "STALE_STATE"


@dataclass
class ReconcileResult:
    """Full result of one reconciliation pass.

    Attributes:
        status: Overall outcome; drives halt/resume decision.
        mismatches: Human-readable description of each discrepancy found.
        reconstructed_positions: Internal positions from EventStore.
        broker_positions: Positions reported by broker API.
        checked_at: UTC datetime when reconciliation was performed.
    """

    status: ReconcileStatus
    mismatches: list[str]
    reconstructed_positions: dict[str, int]
    broker_positions: dict[str, int]
    checked_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class BrokerReconciler:
    """Compares EventStore state to live broker positions.

    This class is stateless — every call to ``reconcile`` is independent.
    Instantiate once and call as needed.

    Usage::

        reconciler = BrokerReconciler()
        result = reconciler.reconcile(
            event_store_positions=event_store.reconstruct_positions(),
            broker_positions=broker.get_positions(),
            daily_pnl=-8500.0,
            daily_limit=-10000.0,
            last_heartbeat=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        )
        if reconciler.needs_operator_confirmation(result):
            telegram.send(reconciler.format_report(result))
            sys.exit(1)
    """

    def reconcile(
        self,
        event_store_positions: dict[str, int],
        broker_positions: dict[str, int],
        daily_pnl: float,
        daily_limit: float,
        last_heartbeat: Optional[datetime],
    ) -> ReconcileResult:
        """Run one reconciliation pass.

        Args:
            event_store_positions: Net qty per symbol reconstructed from events.
            broker_positions: Net qty per symbol from broker API (0-qty symbols
                may be omitted).
            daily_pnl: Realised + unrealised P&L for today (negative = loss).
            daily_limit: Daily loss limit (negative value, e.g. -10000.0).
                Trading is halted when daily_pnl <= daily_limit.
            last_heartbeat: UTC datetime of the last successful broker heartbeat.
                Pass None to force STALE_STATE.

        Returns:
            ReconcileResult describing the outcome.
        """
        mismatches: list[str] = []

        # ── Check 1: Stale state ───────────────────────────────────────────────
        now = datetime.now(tz=timezone.utc)
        if last_heartbeat is None:
            mismatches.append("No heartbeat recorded — broker connection never confirmed.")
            log.warning("reconcile.stale_state", reason="no_heartbeat")
            return ReconcileResult(
                status=ReconcileStatus.STALE_STATE,
                mismatches=mismatches,
                reconstructed_positions=event_store_positions,
                broker_positions=broker_positions,
                checked_at=now,
            )

        # Ensure last_heartbeat is timezone-aware
        hb = last_heartbeat
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=timezone.utc)

        age_minutes = (now - hb).total_seconds() / 60.0
        if age_minutes > _STALE_HEARTBEAT_MINUTES:
            mismatches.append(
                f"Last heartbeat {age_minutes:.1f} minutes ago "
                f"(threshold: {_STALE_HEARTBEAT_MINUTES} min)."
            )
            log.warning("reconcile.stale_state", age_minutes=round(age_minutes, 1))
            return ReconcileResult(
                status=ReconcileStatus.STALE_STATE,
                mismatches=mismatches,
                reconstructed_positions=event_store_positions,
                broker_positions=broker_positions,
                checked_at=now,
            )

        # ── Check 2: Daily loss limit ──────────────────────────────────────────
        if daily_pnl <= daily_limit:
            mismatches.append(
                f"Daily P&L {daily_pnl:.2f} has reached or exceeded "
                f"daily limit {daily_limit:.2f}."
            )
            log.warning(
                "reconcile.daily_limit_hit",
                daily_pnl=daily_pnl,
                daily_limit=daily_limit,
            )
            return ReconcileResult(
                status=ReconcileStatus.DAILY_LIMIT_HIT,
                mismatches=mismatches,
                reconstructed_positions=event_store_positions,
                broker_positions=broker_positions,
                checked_at=now,
            )

        # ── Check 3: Position mismatch ─────────────────────────────────────────
        all_symbols = set(event_store_positions) | set(broker_positions)
        for sym in sorted(all_symbols):
            internal_qty = event_store_positions.get(sym, 0)
            broker_qty = broker_positions.get(sym, 0)
            if internal_qty != broker_qty:
                mismatches.append(
                    f"{sym}: internal={internal_qty}, broker={broker_qty} "
                    f"(delta={broker_qty - internal_qty:+d})"
                )

        if mismatches:
            log.error(
                "reconcile.state_mismatch",
                mismatch_count=len(mismatches),
                symbols=[m.split(":")[0] for m in mismatches],
            )
            return ReconcileResult(
                status=ReconcileStatus.STATE_MISMATCH,
                mismatches=mismatches,
                reconstructed_positions=event_store_positions,
                broker_positions=broker_positions,
                checked_at=now,
            )

        log.info(
            "reconcile.ok",
            symbols=list(event_store_positions.keys()),
            daily_pnl=daily_pnl,
        )
        return ReconcileResult(
            status=ReconcileStatus.OK_TO_TRADE,
            mismatches=[],
            reconstructed_positions=event_store_positions,
            broker_positions=broker_positions,
            checked_at=now,
        )

    def format_report(self, result: ReconcileResult) -> str:
        """Format a human-readable reconciliation report for Telegram.

        Args:
            result: Output of :meth:`reconcile`.

        Returns:
            Multi-line text string suitable for Telegram alert message.
        """
        lines = [
            f"[RECONCILIATION] Status: {result.status.value}",
            f"Checked at: {result.checked_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if result.mismatches:
            lines.append(f"Mismatches ({len(result.mismatches)}):")
            for m in result.mismatches:
                lines.append(f"  • {m}")
        else:
            lines.append("No mismatches detected.")

        lines.append("Internal positions:")
        if result.reconstructed_positions:
            for sym, qty in sorted(result.reconstructed_positions.items()):
                lines.append(f"  {sym}: {qty:+d}")
        else:
            lines.append("  (none)")

        lines.append("Broker positions:")
        if result.broker_positions:
            for sym, qty in sorted(result.broker_positions.items()):
                lines.append(f"  {sym}: {qty:+d}")
        else:
            lines.append("  (none)")

        if result.status == ReconcileStatus.OK_TO_TRADE:
            lines.append("ACTION: OK to trade.")
        elif result.status == ReconcileStatus.DAILY_LIMIT_HIT:
            lines.append("ACTION: HALT — daily loss limit hit. No new trades today.")
        elif result.status == ReconcileStatus.STATE_MISMATCH:
            lines.append("ACTION: HALT — operator confirmation required before resuming.")
        elif result.status == ReconcileStatus.STALE_STATE:
            lines.append("ACTION: HALT — stale broker connection. Reconnect and re-reconcile.")

        return "\n".join(lines)

    def needs_operator_confirmation(self, result: ReconcileResult) -> bool:
        """Return True if a human operator must confirm before trading resumes.

        Args:
            result: Output of :meth:`reconcile`.

        Returns:
            True for any non-OK status.
        """
        return result.status != ReconcileStatus.OK_TO_TRADE
