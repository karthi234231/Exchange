"""
AlphaScorer: computes per-component scores and combines them into a single alpha score.

Fusion formula:
    alpha = (
        cfg.WEIGHT_NEWS      * news_score        +   # FinBERT sentiment -1..1
        cfg.WEIGHT_BULK_DEAL * bulk_deal_score   +   # institutional direction
        cfg.WEIGHT_DELIVERY  * delivery_score    +   # delivery z-score
        cfg.WEIGHT_VIX       * vix_adjustment        # dampening when market is fearful
    )

    Clamp final alpha to [-1.0, 1.0].

Signal labels (from config thresholds):
    alpha >= STRONG_BUY_THRESHOLD  (0.60) → STRONG_BUY
    alpha >= BUY_THRESHOLD         (0.30) → BUY
    alpha >= SELL_THRESHOLD        (-0.30) → NEUTRAL
    alpha >= STRONG_SELL_THRESHOLD (-0.60) → SELL
    else                                   → STRONG_SELL
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

from algo_trading.news_pipeline.config import cfg

log = structlog.get_logger(__name__)

# Impact multipliers for news event scoring.
_IMPACT_MULTIPLIERS: dict[str, float] = {
    "CRITICAL": 2.0,
    "HIGH": 1.5,
    "MEDIUM": 1.0,
    "LOW": 0.5,
}

# Bulk/block deal scale (crore INR) for normalisation.
_BULK_DEAL_SCALE_CR: float = 500.0

# Block deals are 3× more significant than bulk deals.
_BLOCK_MULTIPLIER: float = 3.0

# Minimum history points required before computing a delivery z-score.
_MIN_DELIVERY_HISTORY: int = 5

# Denominator used when mapping a z-score to [-1, 1] via tanh.
_ZSCORE_TANH_DIVISOR: float = 2.0


@dataclass
class ComponentScores:
    news_score: float        # -1.0 to 1.0
    bulk_deal_score: float   # -1.0 to 1.0
    delivery_score: float    # -1.0 to 1.0
    vix_score: float         # -1.0 to 1.0 (dampening, always <= 0)
    alpha_score: float       # final combined score
    signal_type: str         # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL


class AlphaScorer:
    """
    Stateless scorer — takes pre-loaded data, returns ComponentScores.
    No DB calls here — that is SignalFuser's job.
    """

    # ------------------------------------------------------------------
    # News score
    # ------------------------------------------------------------------

    def compute_news_score(self, events: list[dict]) -> float:
        """
        Aggregate sentiment_score across multiple news events.

        Weighting rules:
        - Recency: exponential decay based on age in hours.
          An event published exactly now → time_weight = 1.0.
          An event published 4 hours ago → time_weight ≈ 0.25.
          Formula: time_weight = exp(-ln(4) * age_hours / 4)
          which gives 1.0 at 0 h and 0.25 at 4 h.
        - Impact multiplier: CRITICAL → 2.0, HIGH → 1.5, MEDIUM → 1.0, LOW → 0.5.
        - Final weight per event: time_weight * impact_multiplier.
        - Weighted average; clamped per event score to [-1.0, 1.0].
        - Returns 0.0 when no events are supplied.

        events: list of dicts with keys:
            sentiment_score (float), impact (str), published_at (ISO str or datetime)
        """
        if not events:
            return 0.0

        now_utc = datetime.now(tz=timezone.utc)
        total_weight: float = 0.0
        weighted_sum: float = 0.0

        for event in events:
            # --- parse published_at ---
            pub_raw = event.get("published_at")
            if pub_raw is None:
                age_hours = 0.0
            elif isinstance(pub_raw, datetime):
                pub_dt = pub_raw
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                age_hours = max(0.0, (now_utc - pub_dt).total_seconds() / 3600.0)
            else:
                try:
                    pub_dt = datetime.fromisoformat(str(pub_raw))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    age_hours = max(0.0, (now_utc - pub_dt).total_seconds() / 3600.0)
                except (ValueError, TypeError):
                    age_hours = 0.0

            # exp(-ln(4) * age_hours / 4) → 1.0 at age=0, 0.25 at age=4
            _ln4 = math.log(4.0)
            time_weight = math.exp(-_ln4 * age_hours / 4.0)

            # --- impact multiplier ---
            impact = str(event.get("impact", "MEDIUM")).upper().strip()
            impact_mult = _IMPACT_MULTIPLIERS.get(impact, 1.0)

            # --- sentiment score clamped ---
            raw_score = float(event.get("sentiment_score", 0.0))
            score = max(-1.0, min(1.0, raw_score * impact_mult))

            final_weight = time_weight * impact_mult
            weighted_sum += score * final_weight
            total_weight += final_weight

        if total_weight == 0.0:
            return 0.0

        result = weighted_sum / total_weight
        return max(-1.0, min(1.0, result))

    # ------------------------------------------------------------------
    # Bulk deal score
    # ------------------------------------------------------------------

    def compute_bulk_deal_score(self, deals: list[dict]) -> float:
        """
        Score from today's bulk and block deals.

        Rules:
        - BUY deal  → positive contribution equal to deal_value_cr.
        - SELL deal → negative contribution equal to deal_value_cr.
        - Block deals count 3× vs bulk deals (larger, institutionally significant).
        - net_value = sum(buy_contributions) − sum(sell_contributions)
        - Normalise: net_value / 500 Cr.
        - Clamp to [-1.0, 1.0].
        - Returns 0.0 when no deals are supplied.

        deals: list of dicts with keys:
            direction ("BUY"|"SELL"), deal_type ("bulk"|"block"), deal_value_cr (float)
        """
        if not deals:
            return 0.0

        net_value: float = 0.0
        for deal in deals:
            direction = str(deal.get("direction", "BUY")).upper().strip()
            deal_type = str(deal.get("deal_type", "bulk")).lower().strip()
            value_cr = float(deal.get("deal_value_cr", 0.0))

            multiplier = _BLOCK_MULTIPLIER if deal_type == "block" else 1.0
            contribution = value_cr * multiplier

            if direction == "BUY":
                net_value += contribution
            else:
                net_value -= contribution

        normalised = net_value / _BULK_DEAL_SCALE_CR
        return max(-1.0, min(1.0, normalised))

    # ------------------------------------------------------------------
    # Delivery score
    # ------------------------------------------------------------------

    def compute_delivery_score(self, delivery_history: list[dict]) -> float:
        """
        Z-score of today's delivery percentage vs 30-day history.

        delivery_history: list of dicts with keys delivery_pct, trade_date
                          sorted descending by trade_date (most recent first).
        - Today  = delivery_history[0]
        - History = delivery_history[1 : cfg.DELIVERY_ZSCORE_WINDOW_DAYS]
        - z = (today_pct − mean(history)) / stdev(history)
        - Maps to [-1.0, 1.0] via tanh(z / 2.0).
        - High delivery is a bullish signal (smart money holding).
        - Returns 0.0 when fewer than 5 data points are available.
        """
        if len(delivery_history) < _MIN_DELIVERY_HISTORY + 1:
            # Need at least today + 5 history points.
            return 0.0

        today_pct = float(delivery_history[0].get("delivery_pct", 0.0))
        window = delivery_history[1: cfg.DELIVERY_ZSCORE_WINDOW_DAYS]

        if len(window) < _MIN_DELIVERY_HISTORY:
            return 0.0

        history_pcts = [float(r.get("delivery_pct", 0.0)) for r in window]

        try:
            mean_pct = statistics.mean(history_pcts)
            stdev_pct = statistics.stdev(history_pcts)
        except statistics.StatisticsError:
            return 0.0

        if stdev_pct == 0.0:
            # No variance in history — cannot compute meaningful z-score.
            return 0.0

        z = (today_pct - mean_pct) / stdev_pct
        mapped = math.tanh(z / _ZSCORE_TANH_DIVISOR)
        return max(-1.0, min(1.0, mapped))

    # ------------------------------------------------------------------
    # VIX adjustment
    # ------------------------------------------------------------------

    def compute_vix_adjustment(self, vix_value: float | None) -> float:
        """
        VIX adjustment (dampening) factor for the alpha score.

        - None        → 0.0  (no adjustment)
        - vix < 20   → 0.0  (calm market, no dampening)
        - vix >= 20  → −(vix − 20) / 40   (linear dampening)
          e.g. VIX = 30 → −0.25, VIX = 40 → −0.50, VIX = 60 → −1.0
        - Clamped to [−1.0, 0.0]: VIX can only dampen, never amplify.
        """
        if vix_value is None:
            return 0.0

        threshold = cfg.VIX_HIGH_THRESHOLD  # default 20.0
        if vix_value < threshold:
            return 0.0

        raw = -(vix_value - threshold) / 40.0
        return max(-1.0, min(0.0, raw))

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    def combine(
        self,
        news_score: float,
        bulk_deal_score: float,
        delivery_score: float,
        vix_adjustment: float,
    ) -> ComponentScores:
        """Apply the weighted fusion formula, clamp, and label the result."""
        raw_alpha = (
            cfg.WEIGHT_NEWS      * news_score
            + cfg.WEIGHT_BULK_DEAL * bulk_deal_score
            + cfg.WEIGHT_DELIVERY  * delivery_score
            + cfg.WEIGHT_VIX       * vix_adjustment
        )
        alpha = max(-1.0, min(1.0, raw_alpha))
        label = self._label(alpha)

        log.debug(
            "alpha_scorer.combine",
            news_score=round(news_score, 4),
            bulk_deal_score=round(bulk_deal_score, 4),
            delivery_score=round(delivery_score, 4),
            vix_adjustment=round(vix_adjustment, 4),
            alpha=round(alpha, 4),
            signal_type=label,
        )

        return ComponentScores(
            news_score=news_score,
            bulk_deal_score=bulk_deal_score,
            delivery_score=delivery_score,
            vix_score=vix_adjustment,
            alpha_score=alpha,
            signal_type=label,
        )

    # ------------------------------------------------------------------
    # Label helper
    # ------------------------------------------------------------------

    def _label(self, alpha: float) -> str:
        """Map an alpha score to a signal-type string."""
        if alpha >= cfg.STRONG_BUY_THRESHOLD:
            return "STRONG_BUY"
        if alpha >= cfg.BUY_THRESHOLD:
            return "BUY"
        if alpha >= cfg.SELL_THRESHOLD:
            return "NEUTRAL"
        if alpha >= cfg.STRONG_SELL_THRESHOLD:
            return "SELL"
        return "STRONG_SELL"
