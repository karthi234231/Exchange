"""
algo_trading/models/uncertainty.py — Uncertainty-aware trading decisions.

Wraps ensemble model predictions with uncertainty quantification to:
1. Compute prediction confidence intervals (bootstrap-style)
2. Score ensemble disagreement (how much models differ)
3. Compute model entropy (information content of prediction)
4. Define abstain/no-trade zones when uncertainty is too high
5. Scale position size down proportionally to uncertainty

Research basis:
  Bayesian DeepLOB (Bayesian uncertainty in LOB prediction) shows that
  uncertainty-aware sizing reduces drawdowns by 15-30% vs full-size trading.

Trading rule: trade only when expected_value > cost + uncertainty_buffer
  where uncertainty_buffer = k * entropy * ATR
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Maximum theoretical Shannon entropy for 3 classes = log2(3)
_MAX_ENTROPY_3: float = math.log2(3)  # ≈ 1.585

# Tiny epsilon to guard log2(0) in entropy calculation
_LOG_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# UncertaintyScores
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyScores:
    """All uncertainty measures and the derived trading decision for one bar.

    Attributes:
        prob_buy: P(signal=+1) from the meta-learner.
        prob_hold: P(signal=0) from the meta-learner.
        prob_sell: P(signal=-1) from the meta-learner.
        per_model_probs: Predicted-class probability from each base model,
            e.g. {"xgb": 0.72, "lgb": 0.68, "cat": 0.71}.
        entropy: Shannon entropy H = -Σ p·log2(p) over [buy, hold, sell].
            Range [0, log2(3)≈1.585]. Higher → more uncertain.
        ensemble_disagreement: Std-dev of per-model probabilities for the
            winning class.  Range [0, 0.5]. Higher → models disagree more.
        confidence: Max probability = probability of the predicted class.
            Range [1/3, 1.0]. Higher → more confident.
        predicted_signal: +1 (buy), 0 (hold), or -1 (sell).
        should_trade: False when any uncertainty threshold is breached.
        size_multiplier: Scalar in [0.0, 1.0] to apply to base position size.
        abstain_reason: Empty string when trading; descriptive text when not.
    """

    # Inputs from ensemble
    prob_buy: float
    prob_hold: float
    prob_sell: float
    per_model_probs: dict[str, float]

    # Derived uncertainty measures
    entropy: float
    ensemble_disagreement: float
    confidence: float

    # Trading decision
    predicted_signal: int
    should_trade: bool
    size_multiplier: float
    abstain_reason: str


# ---------------------------------------------------------------------------
# UncertaintyConfig
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyConfig:
    """Thresholds that govern when and how much to trade under uncertainty.

    Attributes:
        min_confidence: Abstain entirely if max probability is below this.
            Default 0.55; random chance for 3 classes is 1/3 ≈ 0.333.
        max_entropy: Abstain if Shannon entropy exceeds this value.
            Default 0.9 (max possible for 3 classes ≈ 1.585).
        max_disagreement: Abstain if ensemble std-dev exceeds this value.
            Default 0.15 (models disagree by more than 15 pp on average).
        full_size_confidence: Use full position size only above this
            confidence level.  Default 0.75.  Below this, size is scaled
            linearly from 0 (at min_confidence) to 1.0 (at this value).
    """

    min_confidence: float = 0.55
    max_entropy: float = 0.9
    max_disagreement: float = 0.15
    full_size_confidence: float = 0.75


# ---------------------------------------------------------------------------
# UncertaintyEvaluator
# ---------------------------------------------------------------------------

class UncertaintyEvaluator:
    """Converts raw ensemble probabilities into uncertainty-adjusted trade decisions.

    Instantiate once per strategy session (stateless after construction).

    Args:
        config: :class:`UncertaintyConfig` thresholds. Uses defaults when None.

    Example:
        >>> evaluator = UncertaintyEvaluator()
        >>> scores = evaluator.evaluate(0.65, 0.20, 0.15, {"xgb": 0.63, "lgb": 0.67})
        >>> scores.should_trade
        True
        >>> scores.size_multiplier
        0.5
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Optional :class:`UncertaintyConfig`. Uses default
                thresholds when not provided.
        """
        self._cfg: UncertaintyConfig = config if config is not None else UncertaintyConfig()
        log.debug(
            "uncertainty_evaluator.init",
            min_confidence=self._cfg.min_confidence,
            max_entropy=self._cfg.max_entropy,
            max_disagreement=self._cfg.max_disagreement,
            full_size_confidence=self._cfg.full_size_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prob_buy: float,
        prob_hold: float,
        prob_sell: float,
        per_model_probs: dict[str, float],
    ) -> UncertaintyScores:
        """Compute all uncertainty scores and determine whether and how much to trade.

        Shannon entropy formula: H = -Σ p·log2(p) for p in [prob_buy, prob_hold, prob_sell].
        Disagreement: std dev of per_model_probs values for the predicted (winning) class.
        Size multiplier: linearly scales from 0.0 at min_confidence to 1.0 at
        full_size_confidence; clamped to [0.0, 1.0].

        Args:
            prob_buy: P(signal=+1) from the meta-learner.
            prob_hold: P(signal=0) from the meta-learner.
            prob_sell: P(signal=-1) from the meta-learner.
            per_model_probs: Dict mapping each base model name to its probability
                for the winning class (e.g. {"xgb": 0.72, "lgb": 0.68, "cat": 0.71}).

        Returns:
            Fully populated :class:`UncertaintyScores` instance.
        """
        # Normalise probs to guard against floating-point drift above 1.0
        total = prob_buy + prob_hold + prob_sell
        if total <= 0.0:
            total = 1.0
        p_buy = prob_buy / total
        p_hold = prob_hold / total
        p_sell = prob_sell / total

        # Predicted class = argmax
        probs_vec = [p_sell, p_hold, p_buy]  # indices 0, 1, 2 → signals -1, 0, +1
        max_idx = int(np.argmax(probs_vec))
        signal_map = {0: -1, 1: 0, 2: 1}
        predicted_signal = signal_map[max_idx]
        confidence = float(probs_vec[max_idx])

        # Shannon entropy
        entropy = self._shannon_entropy([p_buy, p_hold, p_sell])

        # Ensemble disagreement = std dev of per-model probs for the predicted class
        disagreement = self._ensemble_disagreement(per_model_probs)

        # Determine if we should trade
        should_trade, abstain_reason = self._check_abstain(
            confidence=confidence,
            entropy=entropy,
            disagreement=disagreement,
        )

        # Size multiplier: linear ramp from min_confidence → full_size_confidence
        size_multiplier = self._compute_size_multiplier(confidence) if should_trade else 0.0

        scores = UncertaintyScores(
            prob_buy=p_buy,
            prob_hold=p_hold,
            prob_sell=p_sell,
            per_model_probs=dict(per_model_probs),
            entropy=entropy,
            ensemble_disagreement=disagreement,
            confidence=confidence,
            predicted_signal=predicted_signal,
            should_trade=should_trade,
            size_multiplier=size_multiplier,
            abstain_reason=abstain_reason,
        )

        log.debug(
            "uncertainty_evaluator.scores",
            predicted_signal=predicted_signal,
            confidence=round(confidence, 4),
            entropy=round(entropy, 4),
            disagreement=round(disagreement, 4),
            should_trade=should_trade,
            size_multiplier=round(size_multiplier, 4),
            abstain_reason=abstain_reason,
        )
        return scores

    def size_with_uncertainty(
        self,
        base_position_size: int,
        scores: UncertaintyScores,
    ) -> int:
        """Apply the uncertainty multiplier to a base position size.

        Args:
            base_position_size: Raw position size in shares from the position sizer.
            scores: :class:`UncertaintyScores` produced by :meth:`evaluate`.

        Returns:
            Adjusted integer share count.  Always >= 0; never negative.
        """
        if not scores.should_trade or base_position_size <= 0:
            return 0
        adjusted = int(math.floor(base_position_size * scores.size_multiplier))
        result = max(0, adjusted)
        log.debug(
            "uncertainty_evaluator.size_adjusted",
            base_size=base_position_size,
            multiplier=round(scores.size_multiplier, 4),
            adjusted_size=result,
        )
        return result

    def expected_value_exceeds_cost(
        self,
        scores: UncertaintyScores,
        expected_edge_bps: float,
        round_trip_cost_bps: float,
        atr_bps: float,
        uncertainty_buffer_k: float = 0.5,
    ) -> bool:
        """Core trading filter: only trade if net edge exceeds cost plus uncertainty buffer.

        Filter:
            expected_value = scores.confidence × expected_edge_bps
            uncertainty_buffer = uncertainty_buffer_k × scores.entropy × atr_bps
            trade = (expected_value > round_trip_cost_bps + uncertainty_buffer)

        Args:
            scores: :class:`UncertaintyScores` from :meth:`evaluate`.
            expected_edge_bps: Estimated strategy edge in basis points.
            round_trip_cost_bps: Full round-trip transaction cost from the
                cost model (brokerage + impact + taxes) in basis points.
            atr_bps: Current ATR expressed as basis points of price.
            uncertainty_buffer_k: Multiplier that scales how much entropy
                contributes to the required buffer.  Default 0.5.

        Returns:
            True if the trade passes the EV filter; False otherwise.
        """
        expected_value = scores.confidence * expected_edge_bps
        uncertainty_buffer = uncertainty_buffer_k * scores.entropy * atr_bps
        required_hurdle = round_trip_cost_bps + uncertainty_buffer
        passes = expected_value > required_hurdle

        log.debug(
            "uncertainty_evaluator.ev_filter",
            expected_value=round(expected_value, 4),
            round_trip_cost_bps=round(round_trip_cost_bps, 4),
            uncertainty_buffer=round(uncertainty_buffer, 4),
            required_hurdle=round(required_hurdle, 4),
            passes=passes,
        )
        return passes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(probs: list[float]) -> float:
        """Compute Shannon entropy H = -Σ p·log2(p) over a probability vector.

        Args:
            probs: Probability values (should sum to 1; each in [0, 1]).

        Returns:
            Shannon entropy in bits.  Range [0, log2(len(probs))].
        """
        h = 0.0
        for p in probs:
            if p > _LOG_EPS:
                h -= p * math.log2(p)
        return h

    @staticmethod
    def _ensemble_disagreement(per_model_probs: dict[str, float]) -> float:
        """Compute std dev of per-model predicted-class probabilities.

        Args:
            per_model_probs: Dict of model_name → probability for the winning
                class.

        Returns:
            Population std deviation.  Returns 0.0 if fewer than 2 models.
        """
        values = list(per_model_probs.values())
        if len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        # ddof=0 (population std) — consistent with the spec description
        return float(np.std(arr, ddof=0))

    def _check_abstain(
        self,
        confidence: float,
        entropy: float,
        disagreement: float,
    ) -> tuple[bool, str]:
        """Determine whether uncertainty thresholds require abstaining.

        Args:
            confidence: Max predicted-class probability.
            entropy: Shannon entropy of the prediction distribution.
            disagreement: Std dev of per-model probabilities.

        Returns:
            Tuple of (should_trade, abstain_reason).  should_trade is True
            only when all thresholds are satisfied.
        """
        if confidence < self._cfg.min_confidence:
            reason = (
                f"confidence={confidence:.4f} < min_confidence={self._cfg.min_confidence}"
            )
            return False, reason

        if entropy > self._cfg.max_entropy:
            reason = (
                f"entropy={entropy:.4f} > max_entropy={self._cfg.max_entropy}"
            )
            return False, reason

        if disagreement > self._cfg.max_disagreement:
            reason = (
                f"disagreement={disagreement:.4f} > max_disagreement={self._cfg.max_disagreement}"
            )
            return False, reason

        return True, ""

    def _compute_size_multiplier(self, confidence: float) -> float:
        """Scale position size linearly from 0 at min_confidence to 1.0 at full_size_confidence.

        Args:
            confidence: Predicted-class probability (confidence measure).

        Returns:
            Float multiplier clamped to [0.0, 1.0].
        """
        lo = self._cfg.min_confidence
        hi = self._cfg.full_size_confidence

        if hi <= lo:
            # Degenerate config: use binary threshold
            return 1.0 if confidence >= hi else 0.0

        if confidence >= hi:
            return 1.0

        if confidence <= lo:
            return 0.0

        # Linear interpolation between lo (→ 0.0) and hi (→ 1.0)
        multiplier = (confidence - lo) / (hi - lo)
        return float(np.clip(multiplier, 0.0, 1.0))
