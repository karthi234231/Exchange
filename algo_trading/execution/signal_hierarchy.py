"""
signal_hierarchy.py — Explicit signal priority chain for NSE AlgoTrader.

Resolves conflicts between HMM regime, ML ensemble, ADWIN drift, and risk
manager according to spec [SURV-4].  The priority order is:

    4 (highest)  RISK_MANAGER  — absolute veto, always wins
    3            HMM_REGIME    — blocks trades in sideways regime
    2            ML_ENSEMBLE   — primary signal source
    1 (lowest)   ADWIN_DRIFT   — advisory only; reduces size, never vetoes

RL integration is intentionally excluded (Phase 4 only).

Usage::

    from algo_trading.execution.signal_hierarchy import (
        resolve_signal,
        HMMRegime,
        DriftSeverity,
        RiskStatus,
        SignalConflictLogger,
    )

    decision = resolve_signal(
        ml_signal=1,
        ml_confidence=0.72,
        regime=HMMRegime.BULL,
        drift_flag=DriftSeverity.NONE,
        risk_status=RiskStatus.OK,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal

import structlog

from algo_trading.config import settings  # noqa: F401  (imported for side-effects / validators)

log: structlog.BoundLogger = structlog.get_logger(__name__)

_VETO_AUDIT_PATH = Path("state/veto_audit.jsonl")


def log_veto(
    timestamp: datetime,
    symbol: str,
    signal_direction: int,
    veto_reason: str,
    veto_source: str,
    signal_confidence: float,
) -> None:
    """Append a structured veto record to state/veto_audit.jsonl.

    Each call writes one JSON object on its own line.  The file is created
    (including any missing parent directories) on first write.

    Args:
        timestamp: When the veto occurred (timezone-aware recommended).
        symbol: NSE symbol being traded, e.g. ``"RELIANCE"``.
        signal_direction: Raw ML signal that was vetoed (``+1``, ``0``, or ``-1``).
        veto_reason: Human-readable explanation from the veto source.
        veto_source: Name of the priority level that issued the veto, e.g.
            ``"RISK_MANAGER"`` or ``"HMM_REGIME"``.
        signal_confidence: ML ensemble confidence score at veto time.
    """
    record = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "signal_direction": signal_direction,
        "veto_reason": veto_reason,
        "veto_source": veto_source,
        "signal_confidence": round(signal_confidence, 6),
    }
    try:
        _VETO_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _VETO_AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error("veto_audit.write_failed", error=str(exc), record=record)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SignalPriority(IntEnum):
    """Numeric priority levels — higher value wins.

    Attributes:
        ADWIN_DRIFT: Advisory; reduces position size only, never vetoes.
        ML_ENSEMBLE: Primary signal source.
        HMM_REGIME: Blocks new trades when regime is SIDEWAYS.
        RISK_MANAGER: Absolute veto over all other signals.
    """

    ADWIN_DRIFT = 1
    ML_ENSEMBLE = 2
    HMM_REGIME = 3
    RISK_MANAGER = 4


class HMMRegime(str, Enum):
    """Hidden Markov Model market-regime classification.

    Attributes:
        BULL: Trending upward — entries allowed.
        BEAR: Trending downward — entries allowed (short side).
        SIDEWAYS: Range-bound — all new entries blocked by priority 3.
        UNKNOWN: Model not yet warm — treated as SIDEWAYS for safety.
    """

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class DriftSeverity(str, Enum):
    """ADWIN concept-drift severity from the feature stream.

    Attributes:
        NONE: No drift detected; full position size.
        MODERATE: Drift detected; reduce size to 50 % of nominal.
        SEVERE: Strong drift; reduce size to 25 % of nominal.
    """

    NONE = "none"
    MODERATE = "moderate"
    SEVERE = "severe"


class RiskStatus(str, Enum):
    """Real-time risk-manager status.

    Attributes:
        OK: Within all risk limits.
        WARNING: Approaching daily-loss limit; still trading.
        HALT: Kill-switch or daily-loss-limit breach; all trading stopped.
    """

    OK = "ok"
    WARNING = "warning"
    HALT = "halt"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TradingDecision:
    """Fully resolved trading decision emitted by :func:`resolve_signal`.

    Attributes:
        action: Final instruction for the order executor.
        size_multiplier: Fraction of nominal position size to use.
            Only ADWIN_DRIFT may set this below 1.0.
            Possible values: ``1.0`` (full), ``0.5`` (MODERATE drift),
            ``0.25`` (SEVERE drift).
        veto_reason: Human-readable explanation when action is ``HALT`` or
            ``HOLD`` due to a veto.  Empty string when action is ``BUY``/
            ``SELL``.
        signal_strength: ML ensemble confidence score, passed through
            unchanged for downstream use.
        resolved_by: Name of the :class:`SignalPriority` level that made
            the final decision (e.g. ``"RISK_MANAGER"``).
    """

    action: Literal["BUY", "SELL", "HOLD", "HALT"]
    size_multiplier: float = 1.0
    veto_reason: str = ""
    signal_strength: float = 0.0
    resolved_by: str = ""


# ---------------------------------------------------------------------------
# Size-multiplier helpers
# ---------------------------------------------------------------------------

_DRIFT_MULTIPLIER: dict[DriftSeverity, float] = {
    DriftSeverity.NONE: 1.0,
    DriftSeverity.MODERATE: 0.5,
    DriftSeverity.SEVERE: 0.25,
}

_DRIFT_REASON: dict[DriftSeverity, str] = {
    DriftSeverity.MODERATE: "ADWIN drift MODERATE — size reduced 50 %",
    DriftSeverity.SEVERE: "ADWIN drift SEVERE — size reduced 75 %",
}


# ---------------------------------------------------------------------------
# Core resolution function
# ---------------------------------------------------------------------------


def resolve_signal(
    ml_signal: int,
    ml_confidence: float,
    regime: HMMRegime,
    drift_flag: DriftSeverity,
    risk_status: RiskStatus,
    symbol: str = "",
) -> TradingDecision:
    """Resolve a raw ML signal through the four-level priority chain.

    Priority order (highest wins):
      4. RISK_MANAGER — absolute veto; returns HALT immediately.
      3. HMM_REGIME   — blocks BUY/SELL when regime is SIDEWAYS or UNKNOWN.
      2. ML_ENSEMBLE  — converts ``ml_signal`` (+1/0/-1) to BUY/SELL/HOLD.
      1. ADWIN_DRIFT  — only adjusts ``size_multiplier``; never changes action.

    RL is intentionally absent (Phase 4 only).

    Args:
        ml_signal: Direction from ML ensemble.  ``+1`` = BUY, ``-1`` = SELL,
            ``0`` = no opinion (yields HOLD).
        ml_confidence: Confidence score from the ML ensemble in ``[0.0, 1.0]``.
        regime: Current HMM market-regime classification.
        drift_flag: ADWIN drift severity for the current feature stream.
        risk_status: Real-time risk-manager status.
        symbol: NSE trading symbol (optional; used for veto audit trail).

    Returns:
        A :class:`TradingDecision` containing the final action, adjusted
        size multiplier, and audit fields.

    Raises:
        ValueError: If ``ml_signal`` is not in ``{-1, 0, 1}``.
    """
    if ml_signal not in (-1, 0, 1):
        raise ValueError(
            f"ml_signal must be -1, 0, or +1; received {ml_signal!r}"
        )

    bound_log = log.bind(
        ml_signal=ml_signal,
        ml_confidence=round(ml_confidence, 4),
        regime=regime.value,
        drift_flag=drift_flag.value,
        risk_status=risk_status.value,
    )

    # ------------------------------------------------------------------
    # Priority 4 — RISK_MANAGER (absolute veto)
    # ------------------------------------------------------------------
    if risk_status == RiskStatus.HALT:
        decision = TradingDecision(
            action="HALT",
            size_multiplier=0.0,
            veto_reason="RISK_MANAGER: kill-switch or daily-loss-limit active",
            signal_strength=ml_confidence,
            resolved_by=SignalPriority.RISK_MANAGER.name,
        )
        bound_log.warning(
            "signal_resolved",
            action=decision.action,
            resolved_by=decision.resolved_by,
            veto_reason=decision.veto_reason,
        )
        log_veto(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_direction=ml_signal,
            veto_reason=decision.veto_reason,
            veto_source=SignalPriority.RISK_MANAGER.name,
            signal_confidence=ml_confidence,
        )
        return decision

    # ------------------------------------------------------------------
    # Priority 3 — HMM_REGIME (blocks entries in sideways/unknown)
    # ------------------------------------------------------------------
    if regime in (HMMRegime.SIDEWAYS, HMMRegime.UNKNOWN) and ml_signal != 0:
        decision = TradingDecision(
            action="HOLD",
            size_multiplier=0.0,
            veto_reason=f"HMM_REGIME: regime={regime.value} blocks new entries",
            signal_strength=ml_confidence,
            resolved_by=SignalPriority.HMM_REGIME.name,
        )
        bound_log.info(
            "signal_resolved",
            action=decision.action,
            resolved_by=decision.resolved_by,
            veto_reason=decision.veto_reason,
        )
        log_veto(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_direction=ml_signal,
            veto_reason=decision.veto_reason,
            veto_source=SignalPriority.HMM_REGIME.name,
            signal_confidence=ml_confidence,
        )
        return decision

    # ------------------------------------------------------------------
    # Priority 2 — ML_ENSEMBLE (primary signal)
    # ------------------------------------------------------------------
    if ml_signal == 1:
        action: Literal["BUY", "SELL", "HOLD", "HALT"] = "BUY"
    elif ml_signal == -1:
        action = "SELL"
    else:
        action = "HOLD"

    # ------------------------------------------------------------------
    # Priority 1 — ADWIN_DRIFT (advisory size reduction only)
    # ------------------------------------------------------------------
    size_multiplier = _DRIFT_MULTIPLIER[drift_flag] if action != "HOLD" else 0.0
    veto_reason = _DRIFT_REASON.get(drift_flag, "")

    # RISK_MANAGER WARNING does not halt but is surfaced in the log
    if risk_status == RiskStatus.WARNING:
        veto_reason = (
            f"{veto_reason}; RISK_MANAGER: WARNING — near daily-loss limit"
            if veto_reason
            else "RISK_MANAGER: WARNING — near daily-loss limit"
        )

    decision = TradingDecision(
        action=action,
        size_multiplier=size_multiplier,
        veto_reason=veto_reason,
        signal_strength=ml_confidence,
        resolved_by=SignalPriority.ML_ENSEMBLE.name
        if drift_flag == DriftSeverity.NONE
        else SignalPriority.ADWIN_DRIFT.name,
    )

    bound_log.info(
        "signal_resolved",
        action=decision.action,
        size_multiplier=decision.size_multiplier,
        resolved_by=decision.resolved_by,
        veto_reason=decision.veto_reason or None,
    )
    return decision


# ---------------------------------------------------------------------------
# Conflict / activation logger for weekly calibration review
# ---------------------------------------------------------------------------


@dataclass
class SignalConflictLogger:
    """Tracks how often each priority level fires for calibration audits.

    Counters are incremented every time :meth:`record` is called.
    Call :meth:`reset` after each calibration review cycle (e.g. weekly).
    Call :meth:`log_summary` to emit the current counts to structlog.

    Attributes:
        _counters: Mapping from :class:`SignalPriority` name to fire count.
        _total: Total calls recorded since last reset.
    """

    _counters: dict[str, int] = field(
        default_factory=lambda: {p.name: 0 for p in SignalPriority}
    )
    _total: int = field(default=0, init=False)

    def record(self, decision: TradingDecision) -> None:
        """Record a resolved decision and increment the relevant counter.

        Args:
            decision: The :class:`TradingDecision` returned by
                :func:`resolve_signal`.
        """
        self._total += 1
        resolved_by = decision.resolved_by
        if resolved_by in self._counters:
            self._counters[resolved_by] += 1
        else:
            # Defensive: unknown key — still track it
            self._counters[resolved_by] = self._counters.get(resolved_by, 0) + 1

    def get_counts(self) -> dict[str, int]:
        """Return a shallow copy of the current fire counts.

        Returns:
            Dictionary mapping priority-level name to count.
        """
        return dict(self._counters)

    def get_rates(self) -> dict[str, float]:
        """Return each priority level's fraction of total decisions.

        Returns:
            Dictionary mapping priority-level name to rate in ``[0.0, 1.0]``.
            All zeros when ``_total`` is 0.
        """
        if self._total == 0:
            return {k: 0.0 for k in self._counters}
        return {k: v / self._total for k, v in self._counters.items()}

    def log_summary(self) -> None:
        """Emit current counts and rates to structlog at INFO level."""
        log.info(
            "signal_conflict_summary",
            total_decisions=self._total,
            counts=self.get_counts(),
            rates={k: round(v, 4) for k, v in self.get_rates().items()},
        )

    def reset(self) -> None:
        """Reset all counters to zero (call after each calibration cycle)."""
        self._counters = {p.name: 0 for p in SignalPriority}
        self._total = 0
        log.info("signal_conflict_logger_reset")


class SignalHierarchy:
    """Object-oriented facade for the signal hierarchy resolver.

    Older call sites and import smoke tests expect a ``SignalHierarchy`` class,
    while the implementation is intentionally a pure ``resolve_signal`` function.
    This wrapper keeps that public contract without duplicating resolution logic.
    """

    def resolve(
        self,
        ml_signal: int,
        ml_confidence: float,
        regime: HMMRegime,
        drift_flag: DriftSeverity,
        risk_status: RiskStatus,
        symbol: str = "",
    ) -> TradingDecision:
        """Resolve a signal through the standard hierarchy."""
        return resolve_signal(
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            regime=regime,
            drift_flag=drift_flag,
            risk_status=risk_status,
            symbol=symbol,
        )
