"""ADWIN drift adaptation with guardrails for NSE AlgoTrader.

Spec reference: SURV-12.
Prevents false-positive drift adaptations during volatile market events
by enforcing four guardrails before any adaptation is accepted.

Design principle:
    If ADWIN fires more than once per week the model has fundamental problems.
    In that case investigate signal quality — do NOT increase adaptation frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import structlog
from river.drift import ADWIN  # type: ignore[import]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AdaptationResult(str, Enum):
    """Outcome of a single :meth:`SafeADWINAdapter.update` call.

    Attributes:
        NO_CHANGE: ADWIN did not detect drift; no adaptation triggered.
        ADAPTED: Drift detected and all guardrails passed; model adapted.
        BLOCKED_INSUFFICIENT_DATA: Fewer than ``MIN_BARS_BEFORE_ADAPT`` bars
            have been observed since last reset; adaptation blocked.
        BLOCKED_COOLDOWN: Fewer than ``MIN_DAYS_BETWEEN_ADAPT`` days have
            elapsed since the last adaptation; adaptation blocked.
        BLOCKED_HIGH_VOL: Current ATR exceeds ``FREEZE_WHEN_ATR_RATIO`` times
            the 20-day baseline ATR; adaptation frozen to prevent false
            positives during volatile events.
        BLOCKED_EDGE_VALIDATION: ADWIN signalled drift but the edge
            re-validation check (signal quality gate) failed.
    """

    NO_CHANGE = "no_change"
    ADAPTED = "adapted"
    BLOCKED_INSUFFICIENT_DATA = "blocked_insufficient_data"
    BLOCKED_COOLDOWN = "blocked_cooldown"
    BLOCKED_HIGH_VOL = "blocked_high_vol"
    BLOCKED_EDGE_VALIDATION = "blocked_edge_validation"


class _DriftSeverity(str, Enum):
    """Internal drift severity level, used to set the position-size multiplier.

    Attributes:
        NONE: No drift detected.
        MODERATE: Drift detected but adaptation is cooling down.
        SEVERE: Multiple unresolved drift signals.
    """

    NONE = "none"
    MODERATE = "moderate"
    SEVERE = "severe"


# ---------------------------------------------------------------------------
# Guardrails configuration
# ---------------------------------------------------------------------------


@dataclass
class ADWINGuardrails:
    """Configuration parameters for the four drift-adaptation guardrails.

    Attributes:
        MIN_BARS_BEFORE_ADAPT: Minimum number of bars (prediction errors) that
            must be observed after the last reset before any adaptation is
            allowed.  Default 200 ≈ 10 trading days of 15-min bars.
        MIN_DAYS_BETWEEN_ADAPT: Minimum calendar days that must elapse between
            consecutive adaptations.  Enforces at most ~1 adaptation per week.
        FREEZE_WHEN_ATR_RATIO: If ``current_atr / baseline_atr_20d`` exceeds
            this ratio the adapter is frozen.  Default 2.0 means the adapter
            freezes when ATR doubles its 20-day baseline.
        REQUIRE_EDGE_REVALIDATION: When ``True``, a simple edge re-validation
            heuristic must pass before any adaptation is committed.  Set
            ``False`` only in backtesting environments.
    """

    MIN_BARS_BEFORE_ADAPT: int = 200
    MIN_DAYS_BETWEEN_ADAPT: int = 5
    FREEZE_WHEN_ATR_RATIO: float = 2.0
    REQUIRE_EDGE_REVALIDATION: bool = True


# ---------------------------------------------------------------------------
# ADWIN adapter
# ---------------------------------------------------------------------------


class SafeADWINAdapter:
    """ADWIN concept-drift detector wrapped with four safety guardrails.

    The ``river`` ADWIN implementation is used as the underlying drift
    detector.  Prediction errors (0 = correct, 1 = wrong) are fed via
    :meth:`update`.  Each call returns an :class:`AdaptationResult` that
    informs the caller whether the model was adapted or why adaptation was
    blocked.

    Guardrails (evaluated in order when ADWIN fires):
    1. ``BLOCKED_INSUFFICIENT_DATA`` — need at least ``MIN_BARS_BEFORE_ADAPT``
       observations since the last reset.
    2. ``BLOCKED_COOLDOWN`` — at least ``MIN_DAYS_BETWEEN_ADAPT`` days since
       last adaptation.
    3. ``BLOCKED_HIGH_VOL`` — current ATR must not exceed
       ``FREEZE_WHEN_ATR_RATIO × baseline_atr_20d``.
    4. ``BLOCKED_EDGE_VALIDATION`` — if ``REQUIRE_EDGE_REVALIDATION`` is
       ``True``, the rolling accuracy of the last ``MIN_BARS_BEFORE_ADAPT``
       bars must exceed 50 % (better than random).

    Position-size multiplier:
    - ``NONE`` severity → 1.0 (full size)
    - ``MODERATE`` severity → 0.5 (half size)
    - ``SEVERE`` severity → 0.25 (quarter size)

    Attributes:
        guardrails: Immutable guardrails configuration.
    """

    # Size multipliers per severity level
    _SIZE_MULT: dict[_DriftSeverity, float] = {
        _DriftSeverity.NONE: 1.0,
        _DriftSeverity.MODERATE: 0.5,
        _DriftSeverity.SEVERE: 0.25,
    }

    def __init__(self, guardrails: ADWINGuardrails = ADWINGuardrails()) -> None:
        """Initialise the safe ADWIN adapter.

        Args:
            guardrails: Guardrail thresholds.  Uses sensible production
                defaults if not provided.
        """
        self.guardrails = guardrails
        self._adwin = ADWIN()

        # Rolling error buffer for edge re-validation
        # Uses a fixed-size circular buffer implemented with a plain list
        self._error_buffer: list[float] = []
        self._bar_count: int = 0

        # Adaptation tracking
        self._last_adapt_date: date | None = None
        self._drift_count_this_week: int = 0
        self._severity: _DriftSeverity = _DriftSeverity.NONE

        # Total lifetime counters
        self._total_adaptations: int = 0
        self._total_blocked: int = 0

        logger.info(
            "safe_adwin_initialised",
            min_bars=guardrails.MIN_BARS_BEFORE_ADAPT,
            min_days_cooldown=guardrails.MIN_DAYS_BETWEEN_ADAPT,
            freeze_atr_ratio=guardrails.FREEZE_WHEN_ATR_RATIO,
            require_edge_revalidation=guardrails.REQUIRE_EDGE_REVALIDATION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        prediction_error: float,
        current_atr: float,
        baseline_atr_20d: float,
    ) -> AdaptationResult:
        """Feed one prediction error into ADWIN and apply guardrails.

        Args:
            prediction_error: Error value for this bar.  Should be ``0.0``
                when the model was correct and ``1.0`` when wrong.  Values
                in ``[0.0, 1.0]`` are accepted for soft error signals.
            current_atr: ATR value for the most recent bar.
            baseline_atr_20d: Rolling 20-day baseline ATR for volatility
                scaling context.

        Returns:
            :class:`AdaptationResult` indicating what happened.
        """
        self._bar_count += 1
        self._error_buffer.append(prediction_error)
        # Keep buffer bounded to MIN_BARS_BEFORE_ADAPT
        max_buf = self.guardrails.MIN_BARS_BEFORE_ADAPT
        if len(self._error_buffer) > max_buf:
            self._error_buffer.pop(0)

        # Feed ADWIN
        self._adwin.update(prediction_error)

        if not self._adwin.drift_detected:
            logger.debug(
                "adwin_no_drift",
                bar=self._bar_count,
                error=prediction_error,
            )
            return AdaptationResult.NO_CHANGE

        # ADWIN detected drift — run guardrails
        logger.info(
            "adwin_drift_detected",
            bar=self._bar_count,
            total_adaptations=self._total_adaptations,
        )

        # ---- Guardrail 1: insufficient data ----
        if self._bar_count < self.guardrails.MIN_BARS_BEFORE_ADAPT:
            self._total_blocked += 1
            logger.warning(
                "adaptation_blocked_insufficient_data",
                bar_count=self._bar_count,
                required=self.guardrails.MIN_BARS_BEFORE_ADAPT,
            )
            self._adwin = ADWIN()  # reset detector without adapting
            return AdaptationResult.BLOCKED_INSUFFICIENT_DATA

        # ---- Guardrail 2: cooldown ----
        today = date.today()
        if self._last_adapt_date is not None:
            days_since = (today - self._last_adapt_date).days
            if days_since < self.guardrails.MIN_DAYS_BETWEEN_ADAPT:
                self._total_blocked += 1
                logger.warning(
                    "adaptation_blocked_cooldown",
                    days_since_last=days_since,
                    required_days=self.guardrails.MIN_DAYS_BETWEEN_ADAPT,
                    last_adapt=str(self._last_adapt_date),
                )
                self._adwin = ADWIN()
                return AdaptationResult.BLOCKED_COOLDOWN

        # ---- Guardrail 3: high volatility freeze ----
        if baseline_atr_20d > 0.0:
            atr_ratio = current_atr / baseline_atr_20d
        else:
            # Baseline is zero (start of trading day); treat as neutral
            atr_ratio = 1.0

        if atr_ratio >= self.guardrails.FREEZE_WHEN_ATR_RATIO:
            self._total_blocked += 1
            logger.warning(
                "adaptation_blocked_high_volatility",
                atr_ratio=round(atr_ratio, 3),
                threshold=self.guardrails.FREEZE_WHEN_ATR_RATIO,
                current_atr=current_atr,
                baseline_atr=baseline_atr_20d,
            )
            self._adwin = ADWIN()
            return AdaptationResult.BLOCKED_HIGH_VOL

        # ---- Guardrail 4: edge re-validation ----
        if self.guardrails.REQUIRE_EDGE_REVALIDATION:
            if not self._passes_edge_validation():
                self._total_blocked += 1
                logger.warning(
                    "adaptation_blocked_edge_validation",
                    rolling_accuracy=self._rolling_accuracy(),
                )
                self._adwin = ADWIN()
                return AdaptationResult.BLOCKED_EDGE_VALIDATION

        # ---- All guardrails passed — accept adaptation ----
        self._total_adaptations += 1
        self._drift_count_this_week += 1
        self._last_adapt_date = today
        self._severity = self._compute_severity()
        self._adwin = ADWIN()  # reset ADWIN after adaptation
        self._error_buffer.clear()
        self._bar_count = 0

        if self._drift_count_this_week > 1:
            logger.warning(
                "frequent_drift_warning",
                drift_count_this_week=self._drift_count_this_week,
                message=(
                    "ADWIN is firing >1x/week. "
                    "Investigate signal quality rather than increasing adaptation."
                ),
            )

        logger.info(
            "adaptation_accepted",
            total_adaptations=self._total_adaptations,
            severity=self._severity.value,
            size_multiplier=self.get_size_multiplier(),
        )
        return AdaptationResult.ADAPTED

    def get_size_multiplier(self) -> float:
        """Return the current position-size multiplier based on drift severity.

        Size schedule:
        - ``NONE`` (no unresolved drift): ``1.0``
        - ``MODERATE`` (one drift event, adapting): ``0.5``
        - ``SEVERE`` (multiple consecutive drifts): ``0.25``

        Returns:
            Multiplier in ``{0.25, 0.5, 1.0}``.
        """
        return self._SIZE_MULT[self._severity]

    def reset(self) -> None:
        """Reset all state after a manual Sunday retrain.

        Clears ADWIN history, error buffer, bar counter, adaptation date,
        drift count, and severity.  Call this after a new :class:`FrozenModelSpec`
        has been deployed to the registry.
        """
        self._adwin = ADWIN()
        self._error_buffer.clear()
        self._bar_count = 0
        self._last_adapt_date = None
        self._drift_count_this_week = 0
        self._severity = _DriftSeverity.NONE
        logger.info(
            "safe_adwin_reset",
            total_lifetime_adaptations=self._total_adaptations,
            total_lifetime_blocked=self._total_blocked,
        )

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def total_adaptations(self) -> int:
        """Total number of successful adaptations since creation.

        Returns:
            Lifetime adaptation count.
        """
        return self._total_adaptations

    @property
    def total_blocked(self) -> int:
        """Total number of adaptation attempts blocked by guardrails.

        Returns:
            Lifetime blocked-adaptation count.
        """
        return self._total_blocked

    @property
    def severity(self) -> str:
        """Current drift severity level as a string.

        Returns:
            One of ``"none"``, ``"moderate"``, or ``"severe"``.
        """
        return self._severity.value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_accuracy(self) -> float:
        """Compute rolling accuracy over the error buffer.

        Accuracy = proportion of errors that are 0.0 (correct predictions).
        A value above 0.5 means the model is still better than random.

        Returns:
            Accuracy in ``[0.0, 1.0]``, or ``0.0`` if the buffer is empty.
        """
        if not self._error_buffer:
            return 0.0
        # Treat error < 0.5 as "correct" to handle soft error values
        correct = sum(1 for e in self._error_buffer if e < 0.5)
        return correct / len(self._error_buffer)

    def _passes_edge_validation(self) -> bool:
        """Return True if the rolling accuracy exceeds 50 % (better than random).

        Used as a lightweight signal-quality gate in Guardrail 4.

        Returns:
            ``True`` if accuracy > 0.50, ``False`` otherwise.
        """
        acc = self._rolling_accuracy()
        return acc > 0.50

    def _compute_severity(self) -> _DriftSeverity:
        """Determine drift severity from the weekly drift count.

        Schedule:
        - ``drift_count_this_week == 1`` → ``MODERATE``
        - ``drift_count_this_week >= 2`` → ``SEVERE``

        Returns:
            The appropriate :class:`_DriftSeverity` level.
        """
        if self._drift_count_this_week >= 2:
            return _DriftSeverity.SEVERE
        return _DriftSeverity.MODERATE
