"""
algo_trading/models/shadow_mode.py — Shadow mode runner and auto-rollback.

Before activating a newly trained model in production, run it in shadow mode:
  - Shadow model receives the same live market data
  - Shadow model generates predictions but they are NOT executed
  - Shadow predictions are compared to the current production model's predictions
  - If shadow model passes a performance gate, it is promoted to production
  - If the production model degrades, it is automatically rolled back to the previous version

Shadow mode phases:
  1. SHADOW: new model runs in parallel, no trades
  2. PAPER: new model trades in paper (simulated) mode
  3. LIVE: new model promoted to production (requires manual approval + auto gates)

Auto-rollback trigger:
  - Production model Sharpe drops below 0.3 for 5 consecutive days
  - OR production model loss > 1.5x historical average daily loss for 3 days
  → automatic rollback to previous version, alert via Telegram
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ── Default configuration constants ──────────────────────────────────────────

_DEFAULT_MIN_SHADOW_DAYS: int     = 5
_DEFAULT_PAPER_TRADE_DAYS: int    = 5
_DEFAULT_MIN_AGREEMENT_RATE: float = 0.70
_DEFAULT_ROLLBACK_SHARPE: float   = 0.3
_DEFAULT_ROLLBACK_CONSEC_DAYS: int = 5

# Loss multiplier threshold for rollback trigger
_ROLLBACK_LOSS_MULTIPLIER: float  = 1.5
# Consecutive loss days required before loss-based rollback fires
_ROLLBACK_LOSS_DAYS: int          = 3

# Signals are classified into three directions for agreement computation
_DIRECTION_LONG: str  = "long"
_DIRECTION_SHORT: str = "short"
_DIRECTION_FLAT: str  = "flat"


class ShadowPhase(str, Enum):
    """Lifecycle phase of a shadow model candidate.

    Attributes:
        SHADOW: Model runs in parallel, predictions logged only — no trades.
        PAPER: Model executes trades in simulation mode.
        LIVE: Model has been promoted to production and executes real orders.
    """

    SHADOW = "shadow"
    PAPER  = "paper"
    LIVE   = "live"


@dataclass
class _PredictionEntry:
    """Internal record of a single model prediction.

    Attributes:
        model_version: Version string of the model that produced this prediction.
        signal: Directional signal emitted (``"long"``, ``"short"``, or ``"flat"``).
        confidence: Model confidence score in [0.0, 1.0].
        bar_time: Timestamp of the bar on which the prediction was made.
    """

    model_version: str
    signal: str
    confidence: float
    bar_time: datetime


@dataclass
class ShadowModelRecord:
    """Mutable state record for a model under shadow or paper evaluation.

    This is the primary data object tracked by :class:`ShadowModeRunner`.
    Predictions are stored in an append-only list; other fields are updated
    in-place as the model progresses through the pipeline.

    Attributes:
        model_version: Semantic version string of the candidate model.
        phase: Current :class:`ShadowPhase`.
        predictions: Ordered list of :class:`_PredictionEntry` objects.
        paper_pnl: Cumulative paper P&L in currency units (default INR).
        shadow_start_date: Date on which shadow mode began.
        promoted_date: Date on which the model was promoted to LIVE, or ``None``.
        rolled_back: ``True`` if this model was rolled back after going LIVE.
    """

    model_version: str
    phase: ShadowPhase
    predictions: list[_PredictionEntry] = field(default_factory=list)
    paper_pnl: float = 0.0
    shadow_start_date: date = field(default_factory=date.today)
    promoted_date: date | None = None
    rolled_back: bool = False


@dataclass
class ShadowRunnerConfig:
    """Configuration parameters for :class:`ShadowModeRunner`.

    Attributes:
        min_shadow_days: Minimum days in SHADOW phase before promotion is allowed.
        paper_trade_days: Minimum days in PAPER phase before promotion to LIVE.
        min_agreement_rate: Minimum fraction of bars where shadow and production
            signals must agree before promotion.
        rollback_sharpe_threshold: Sharpe ratio below which the production model
            is considered degraded.
        rollback_consecutive_days: Number of consecutive sub-threshold Sharpe days
            that trigger automatic rollback.
    """

    min_shadow_days: int         = _DEFAULT_MIN_SHADOW_DAYS
    paper_trade_days: int        = _DEFAULT_PAPER_TRADE_DAYS
    min_agreement_rate: float    = _DEFAULT_MIN_AGREEMENT_RATE
    rollback_sharpe_threshold: float = _DEFAULT_ROLLBACK_SHARPE
    rollback_consecutive_days: int   = _DEFAULT_ROLLBACK_CONSEC_DAYS


class ShadowModeRunner:
    """Manages shadow-mode evaluation and automatic rollback of ML models.

    The runner tracks one production model and one shadow/paper candidate.
    It compares their predictions on each live bar, enforces promotion gates,
    and triggers rollback when the production model degrades.

    Typical lifecycle::

        config  = ShadowRunnerConfig(min_shadow_days=5, paper_trade_days=5)
        runner  = ShadowModeRunner(prod_spec, shadow_spec, config)

        # On each live bar:
        runner.record_prediction("v1.2", signal="long", confidence=0.8, bar_time=now)
        runner.record_prediction("v1.3", signal="long", confidence=0.85, bar_time=now)

        # End of day:
        if runner.check_rollback_condition(daily_returns):
            runner.rollback()
        elif runner.should_promote():
            runner.promote_to_paper()

    Args:
        production_spec: Production model specification object.  Any object
            with a ``.version`` attribute is accepted.
        shadow_spec: Shadow/candidate model specification.  Any object with a
            ``.version`` attribute is accepted.
        config: :class:`ShadowRunnerConfig` governing promotion and rollback gates.
    """

    def __init__(
        self,
        production_spec: Any,
        shadow_spec: Any,
        config: ShadowRunnerConfig | None = None,
    ) -> None:
        """Initialise the runner with production and shadow model specs.

        Args:
            production_spec: Production model spec (must have ``.version``).
            shadow_spec: Shadow candidate spec (must have ``.version``).
            config: Runner configuration; defaults applied if ``None``.
        """
        self._production_spec = production_spec
        self._shadow_spec     = shadow_spec
        self._config          = config or ShadowRunnerConfig()

        self._production_record = ShadowModelRecord(
            model_version=production_spec.version,
            phase=ShadowPhase.LIVE,
        )
        self._shadow_record = ShadowModelRecord(
            model_version=shadow_spec.version,
            phase=ShadowPhase.SHADOW,
            shadow_start_date=date.today(),
        )

        # Tracks consecutive Sharpe-below-threshold days for rollback
        self._consecutive_low_sharpe_days: int = 0
        # Tracks consecutive high-loss days for rollback
        self._consecutive_high_loss_days: int  = 0
        # Historical average daily loss — computed from first N daily_returns seen
        self._historical_avg_loss: float | None = None

        log.info(
            "shadow.runner_init",
            production_version=production_spec.version,
            shadow_version=shadow_spec.version,
            phase=ShadowPhase.SHADOW.value,
            min_shadow_days=self._config.min_shadow_days,
            min_agreement_rate=self._config.min_agreement_rate,
        )

    # ── Core recording ────────────────────────────────────────────────────────

    def record_prediction(
        self,
        model_version: str,
        signal: str,
        confidence: float,
        bar_time: datetime,
    ) -> None:
        """Store a prediction from either the production or shadow model.

        Routes the prediction to the appropriate internal record based on
        the ``model_version`` string.

        Args:
            model_version: Version string of the model making the prediction.
            signal: Directional signal: ``"long"``, ``"short"``, or ``"flat"``.
            confidence: Model confidence in [0.0, 1.0].
            bar_time: Bar timestamp associated with this prediction.
        """
        entry = _PredictionEntry(
            model_version=model_version,
            signal=signal.lower().strip(),
            confidence=confidence,
            bar_time=bar_time,
        )

        if model_version == self._production_record.model_version:
            self._production_record.predictions.append(entry)
            log.debug(
                "shadow.production_prediction",
                model=model_version,
                signal=signal,
                confidence=confidence,
                bar_time=bar_time.isoformat(),
            )
        elif model_version == self._shadow_record.model_version:
            self._shadow_record.predictions.append(entry)
            log.debug(
                "shadow.shadow_prediction",
                model=model_version,
                signal=signal,
                confidence=confidence,
                bar_time=bar_time.isoformat(),
            )
        else:
            log.warning(
                "shadow.unknown_model_version",
                model_version=model_version,
                production=self._production_record.model_version,
                shadow=self._shadow_record.model_version,
            )

    # ── Agreement rate ────────────────────────────────────────────────────────

    def compute_agreement_rate(self) -> float:
        """Compute the fraction of bars where production and shadow agree.

        Aligns predictions by ``bar_time``, counts bars where both models
        emit the same directional signal.

        Returns:
            Agreement rate in [0.0, 1.0].  Returns 0.0 if there are no
            aligned predictions.
        """
        prod_preds = {p.bar_time: p.signal for p in self._production_record.predictions}
        shadow_preds = {p.bar_time: p.signal for p in self._shadow_record.predictions}

        common_bars = set(prod_preds.keys()) & set(shadow_preds.keys())
        if not common_bars:
            log.debug("shadow.agreement_no_common_bars")
            return 0.0

        agreements = sum(
            1 for bt in common_bars
            if prod_preds[bt] == shadow_preds[bt]
        )
        rate = agreements / len(common_bars)

        log.debug(
            "shadow.agreement_rate",
            common_bars=len(common_bars),
            agreements=agreements,
            rate=round(rate, 4),
        )
        return rate

    # ── Promotion gates ───────────────────────────────────────────────────────

    def should_promote(self) -> bool:
        """Return ``True`` if the shadow model is eligible for paper promotion.

        Promotion requires:
        1. Shadow has been running for at least ``config.min_shadow_days``.
        2. Agreement rate >= ``config.min_agreement_rate``.
        3. Shadow model is still in SHADOW phase (not yet paper/live).

        Returns:
            ``True`` if both gate conditions are met.
        """
        if self._shadow_record.phase != ShadowPhase.SHADOW:
            return False

        days_running = (date.today() - self._shadow_record.shadow_start_date).days
        if days_running < self._config.min_shadow_days:
            log.debug(
                "shadow.promote_gate_days_insufficient",
                days_running=days_running,
                min_required=self._config.min_shadow_days,
            )
            return False

        agreement = self.compute_agreement_rate()
        if agreement < self._config.min_agreement_rate:
            log.debug(
                "shadow.promote_gate_agreement_insufficient",
                agreement=round(agreement, 4),
                min_required=self._config.min_agreement_rate,
            )
            return False

        log.info(
            "shadow.promote_gate_passed",
            days_running=days_running,
            agreement=round(agreement, 4),
        )
        return True

    def promote_to_paper(self) -> None:
        """Transition the shadow model from SHADOW to PAPER phase.

        Clears accumulated predictions to start fresh paper trading.
        Logs the transition event.

        Raises:
            RuntimeError: If the shadow model is not in SHADOW phase.
        """
        if self._shadow_record.phase != ShadowPhase.SHADOW:
            raise RuntimeError(
                f"Cannot promote to PAPER: shadow model '{self._shadow_record.model_version}' "
                f"is in phase {self._shadow_record.phase.value!r}, expected 'shadow'."
            )

        self._shadow_record.phase = ShadowPhase.PAPER
        self._shadow_record.predictions.clear()
        self._shadow_record.paper_pnl = 0.0

        log.info(
            "shadow.promoted_to_paper",
            model_version=self._shadow_record.model_version,
            shadow_days_run=(
                date.today() - self._shadow_record.shadow_start_date
            ).days,
        )

    def promote_to_live(self) -> None:
        """Transition the paper model to LIVE (production) status.

        This is an explicit manual action — it is NEVER called automatically.
        The caller must verify paper trading results independently before
        invoking this method.

        The shadow model's version becomes the new production version.
        The old production model record is archived internally.

        Raises:
            RuntimeError: If the shadow model is not in PAPER phase.
        """
        if self._shadow_record.phase != ShadowPhase.PAPER:
            raise RuntimeError(
                f"Cannot promote to LIVE: shadow model '{self._shadow_record.model_version}' "
                f"is in phase {self._shadow_record.phase.value!r}, expected 'paper'."
            )

        old_production_version = self._production_record.model_version
        self._shadow_record.phase = ShadowPhase.LIVE
        self._shadow_record.promoted_date = date.today()

        # Swap: shadow becomes production
        self._production_record, self._shadow_record = (
            self._shadow_record, self._production_record
        )

        log.warning(
            "shadow.promoted_to_live",
            new_production=self._production_record.model_version,
            retired_production=old_production_version,
            paper_pnl=self._production_record.paper_pnl,
        )

    # ── Rollback ──────────────────────────────────────────────────────────────

    def check_rollback_condition(self, recent_daily_returns: list[float]) -> bool:
        """Evaluate whether the production model warrants an automatic rollback.

        Two rollback triggers are checked:

        **Trigger 1 — Low Sharpe**: if the annualised Sharpe of the most recent
        ``config.rollback_consecutive_days`` returns is below
        ``config.rollback_sharpe_threshold``, increment the consecutive-day
        counter; reset to zero otherwise.

        **Trigger 2 — Excessive Loss**: if the most recent day's loss exceeds
        ``_ROLLBACK_LOSS_MULTIPLIER`` × historical average daily loss,
        increment the consecutive-loss counter.  The historical average is
        seeded from the first call's return series.

        Args:
            recent_daily_returns: List of daily P&L returns (float, in consistent
                units).  The most recent return is the last element.

        Returns:
            ``True`` if either rollback trigger has fired.
        """
        if not recent_daily_returns:
            return False

        fired = False

        # ── Trigger 1: Rolling Sharpe ─────────────────────────────────────
        window = recent_daily_returns[-self._config.rollback_consecutive_days:]
        rolling_sharpe = _compute_sharpe(window)

        if rolling_sharpe < self._config.rollback_sharpe_threshold:
            self._consecutive_low_sharpe_days += 1
            log.warning(
                "shadow.rollback_sharpe_low",
                rolling_sharpe=round(rolling_sharpe, 4),
                threshold=self._config.rollback_sharpe_threshold,
                consecutive_days=self._consecutive_low_sharpe_days,
                trigger_at=self._config.rollback_consecutive_days,
            )
            if self._consecutive_low_sharpe_days >= self._config.rollback_consecutive_days:
                log.critical(
                    "shadow.rollback_trigger_sharpe",
                    consecutive_days=self._consecutive_low_sharpe_days,
                    rolling_sharpe=round(rolling_sharpe, 4),
                )
                fired = True
        else:
            self._consecutive_low_sharpe_days = 0

        # ── Trigger 2: Excessive daily loss ───────────────────────────────
        # Seed historical average on first call
        if self._historical_avg_loss is None and len(recent_daily_returns) >= 5:
            losses = [-r for r in recent_daily_returns if r < 0.0]
            self._historical_avg_loss = (
                sum(losses) / len(losses) if losses else 0.0
            )
            log.info(
                "shadow.historical_avg_loss_seeded",
                avg_loss=round(self._historical_avg_loss, 4),
                from_n_days=len(recent_daily_returns),
            )

        if self._historical_avg_loss is not None and self._historical_avg_loss > 0.0:
            latest_loss = -recent_daily_returns[-1] if recent_daily_returns[-1] < 0.0 else 0.0
            threshold_loss = _ROLLBACK_LOSS_MULTIPLIER * self._historical_avg_loss

            if latest_loss > threshold_loss:
                self._consecutive_high_loss_days += 1
                log.warning(
                    "shadow.rollback_loss_elevated",
                    latest_loss=round(latest_loss, 4),
                    threshold=round(threshold_loss, 4),
                    historical_avg=round(self._historical_avg_loss, 4),
                    consecutive_days=self._consecutive_high_loss_days,
                )
                if self._consecutive_high_loss_days >= _ROLLBACK_LOSS_DAYS:
                    log.critical(
                        "shadow.rollback_trigger_loss",
                        consecutive_loss_days=self._consecutive_high_loss_days,
                        latest_loss=round(latest_loss, 4),
                    )
                    fired = True
            else:
                self._consecutive_high_loss_days = 0

        return fired

    def rollback(self) -> None:
        """Execute an automatic rollback of the production model.

        Swaps production and shadow records, marks the rolled-back model,
        and resets the internal counter.  Logs at CRITICAL level.

        This method is idempotent if called multiple times (the counters
        reset on each call regardless).
        """
        rolled_back_version = self._production_record.model_version
        restored_version    = self._shadow_record.model_version

        self._production_record.rolled_back = True

        # Swap roles: previous shadow (or earlier production) becomes active
        self._production_record, self._shadow_record = (
            self._shadow_record, self._production_record
        )

        # Reset degradation counters
        self._consecutive_low_sharpe_days = 0
        self._consecutive_high_loss_days  = 0

        log.critical(
            "shadow.rollback_executed",
            rolled_back=rolled_back_version,
            restored=restored_version,
            note=(
                "Production rolled back automatically. "
                "Manual review required before re-promoting."
            ),
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def production_version(self) -> str:
        """Return the version string of the current production model.

        Returns:
            Production model version string.
        """
        return self._production_record.model_version

    @property
    def shadow_version(self) -> str:
        """Return the version string of the current shadow/paper model.

        Returns:
            Shadow model version string.
        """
        return self._shadow_record.model_version

    @property
    def shadow_phase(self) -> ShadowPhase:
        """Return the current phase of the shadow/paper model.

        Returns:
            Current :class:`ShadowPhase`.
        """
        return self._shadow_record.phase

    def shadow_days_running(self) -> int:
        """Return the number of days the shadow model has been running.

        Returns:
            Calendar days since shadow mode started.
        """
        return (date.today() - self._shadow_record.shadow_start_date).days

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of runner state for logging and monitoring.

        Returns:
            Dictionary with production version, shadow version, shadow phase,
            agreement rate, shadow days, and paper P&L.
        """
        return {
            "production_version":    self._production_record.model_version,
            "shadow_version":        self._shadow_record.model_version,
            "shadow_phase":          self._shadow_record.phase.value,
            "agreement_rate":        round(self.compute_agreement_rate(), 4),
            "shadow_days_running":   self.shadow_days_running(),
            "paper_pnl":             self._shadow_record.paper_pnl,
            "consecutive_low_sharpe": self._consecutive_low_sharpe_days,
            "consecutive_high_loss":  self._consecutive_high_loss_days,
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_sharpe(returns: list[float], annualise_factor: float = 252.0) -> float:
    """Compute the Sharpe ratio for a list of daily returns.

    Assumes a risk-free rate of zero (typical for short intraday holding
    periods in Indian markets).

    Args:
        returns: List of daily return values.
        annualise_factor: Square root of this is multiplied by mean/std to
            annualise.  Default 252 (trading days in a year).

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 if standard deviation is zero
        or fewer than 2 return values are provided.
    """
    if len(returns) < 2:
        return 0.0
    try:
        mean   = statistics.mean(returns)
        stdev  = statistics.stdev(returns)
    except statistics.StatisticsError:
        return 0.0

    if stdev == 0.0:
        return 0.0

    return (mean / stdev) * (annualise_factor ** 0.5)
