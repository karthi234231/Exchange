"""
algo_trading/fo/signal_generator.py — F&O strategy entry signal generator.

Combines IV rank, VRP, PCR, and days-to-expiry into a single actionable
signal that recommends which multi-leg strategy to enter (or no trade).

Design
------
- Inputs: raw scalars (iv_rank, pcr, vrp, days_to_expiry) + VRPEstimate
- Output: FOSignal dataclass with strategy_name, confidence, and reasoning
- No I/O: pure function logic — the orchestrator feeds live data and
  passes the FOSignal to StrategyBuilder for order construction.

All thresholds below are ASSUMED starting points.
Calibrate against at least 12 months of live data before trusting signals.

Entry hierarchy (precedence order):
  1. DTE out-of-range → NO_TRADE (calendar filter first)
  2. High IV (>60) + favorable VRP + valid PCR → short-premium strategy
  3. Low IV (<30) + unfavorable VRP → long straddle (buy vol)
  4. Else → NO_TRADE

Short-premium strategy selection (when condition 2 fires):
  - PCR > 1.5: bearish skew → Short Strangle (wider wings, more credit)
  - PCR 0.8–1.5: balanced → Iron Condor (defined risk, both sides)
  - PCR < 0.8: bullish skew → Bull Put Spread (sell downside only)

References
----------
Cohen, G. (2005) "The Bible of Options Strategies", Ch. 7–11.
Sinclair, E. (2010) "Option Trading: Pricing and Volatility Strategies", Ch. 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from algo_trading.fo.portfolio_risk import VRPEstimate

log = structlog.get_logger(__name__)

# ── Configurable thresholds (all ASSUMED — validate on historical data) ───────

IV_RANK_HIGH_THRESHOLD: float = 60.0   # ASSUMED: IV rank > 60 → elevated vol, sell premium
IV_RANK_LOW_THRESHOLD:  float = 30.0   # ASSUMED: IV rank < 30 → depressed vol, buy vol

PCR_BEARISH_THRESHOLD:  float = 1.5    # ASSUMED: PCR > 1.5 → fear-driven put buying → strangle
PCR_BULLISH_THRESHOLD:  float = 0.8    # ASSUMED: PCR < 0.8 → complacent call buying → bull put

MIN_DTE: int = 7     # ASSUMED: don't enter within 7 DTE (gamma risk too high)
MAX_DTE: int = 45    # ASSUMED: don't enter beyond 45 DTE (theta too slow to realize)

# Minimum confidence threshold below which NO_TRADE is returned
MIN_CONFIDENCE: float = 0.25   # ASSUMED

# Confidence scoring weights
_W_IV_RANK: float   = 0.40
_W_VRP:     float   = 0.35
_W_PCR:     float   = 0.25


# ── Output types ─────────────────────────────────────────────────────────────

STRATEGY_NO_TRADE        = "no_trade"
STRATEGY_SHORT_STRANGLE  = "short_strangle"
STRATEGY_IRON_CONDOR     = "iron_condor"
STRATEGY_BULL_PUT_SPREAD = "bull_put_spread"
STRATEGY_BEAR_CALL_SPREAD = "bear_call_spread"
STRATEGY_LONG_STRADDLE   = "long_straddle"

DIRECTION_SHORT_PREMIUM = "short_premium"
DIRECTION_LONG_PREMIUM  = "long_premium"
DIRECTION_NEUTRAL       = "neutral"


@dataclass
class FOSignal:
    """
    Recommended F&O trade entry signal.

    Attributes
    ----------
    strategy_name : str
        One of the STRATEGY_* constants.  "no_trade" means no entry.
    trade_direction : str
        "short_premium", "long_premium", or "neutral".
    confidence : float
        Score in [0.0, 1.0].  Higher = more aligned indicators.
        Below MIN_CONFIDENCE the strategy_name is forced to "no_trade".
    iv_rank : float
        Current IV rank (0–100) used for this signal.
    vrp : float
        Variance Risk Premium at signal time.
    vrp_signal : str
        VRPEstimate.signal string for audit logging.
    pcr : float
        Put-Call Ratio at signal time.
    days_to_expiry : int
        DTE of the target expiry used for this signal.
    reasoning : list[str]
        Ordered list of human-readable justifications (for audit log / Telegram).
    """
    strategy_name:   str
    trade_direction: str
    confidence:      float
    iv_rank:         float
    vrp:             float
    vrp_signal:      str
    pcr:             float
    days_to_expiry:  int
    reasoning:       list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """True when a real trade is recommended (not no_trade)."""
        return self.strategy_name != STRATEGY_NO_TRADE

    def summary(self) -> str:
        """Single-line summary for Telegram / logging."""
        return (
            f"[FOSignal] {self.strategy_name.upper()} | "
            f"direction={self.trade_direction} | "
            f"confidence={self.confidence:.0%} | "
            f"iv_rank={self.iv_rank:.1f} | vrp={self.vrp:+.3f} | pcr={self.pcr:.2f} | "
            f"dte={self.days_to_expiry}"
        )


# ── Generator ─────────────────────────────────────────────────────────────────

class FOSignalGenerator:
    """
    Combine IV rank, VRP, and PCR into an actionable F&O entry signal.

    This class is stateless — all state is in the inputs.  Instantiate once
    per session (or change thresholds between sessions by re-instantiating).

    Parameters
    ----------
    iv_rank_high : float
        IV rank above which short-premium strategies are considered.
        ASSUMED default: 60.
    iv_rank_low : float
        IV rank below which long-premium strategies (buy vol) are considered.
        ASSUMED default: 30.
    pcr_bearish : float
        PCR above which bearish/strangle strategy is preferred over condor.
        ASSUMED default: 1.5.
    pcr_bullish : float
        PCR below which bullish/put-spread strategy is preferred.
        ASSUMED default: 0.8.
    min_dte : int
        Minimum days to expiry for entry.  ASSUMED default: 7.
    max_dte : int
        Maximum days to expiry for entry.  ASSUMED default: 45.
    min_confidence : float
        Minimum composite confidence score; signals below this → no_trade.
        ASSUMED default: 0.40.
    """

    def __init__(
        self,
        iv_rank_high:    float = IV_RANK_HIGH_THRESHOLD,
        iv_rank_low:     float = IV_RANK_LOW_THRESHOLD,
        pcr_bearish:     float = PCR_BEARISH_THRESHOLD,
        pcr_bullish:     float = PCR_BULLISH_THRESHOLD,
        min_dte:         int   = MIN_DTE,
        max_dte:         int   = MAX_DTE,
        min_confidence:  float = MIN_CONFIDENCE,
    ) -> None:
        self.iv_rank_high   = iv_rank_high
        self.iv_rank_low    = iv_rank_low
        self.pcr_bearish    = pcr_bearish
        self.pcr_bullish    = pcr_bullish
        self.min_dte        = min_dte
        self.max_dte        = max_dte
        self.min_confidence = min_confidence

        # Validate that confidence-scoring weights sum to 1.0 (within tolerance)
        weight_sum = _W_IV_RANK + _W_VRP + _W_PCR
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"Signal confidence weights must sum to 1.0 (got {weight_sum:.4f}). "
                f"_W_IV_RANK={_W_IV_RANK}, _W_VRP={_W_VRP}, _W_PCR={_W_PCR}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        iv_rank:        float,
        vrp_estimate:   VRPEstimate,
        pcr:            float,
        days_to_expiry: int,
    ) -> FOSignal:
        """
        Generate an F&O entry signal from current market conditions.

        Parameters
        ----------
        iv_rank : float
            Current IV rank in [0, 100].  Computed by FOFeaturesEngine.
        vrp_estimate : VRPEstimate
            Output of PortfolioRiskEngine.compute_vrp().
        pcr : float
            Put-Call Ratio of open interest for the target expiry.
            Computed by FOFeaturesEngine.
        days_to_expiry : int
            Calendar days to the target expiry contract.

        Returns
        -------
        FOSignal
            Recommended strategy and metadata.
        """
        # ── Input validation ─────────────────────────────────────────────────
        if not 0.0 <= iv_rank <= 100.0:
            raise ValueError(f"iv_rank must be in [0, 100], got {iv_rank}")

        reasons: list[str] = []
        vrp = vrp_estimate.vrp

        # ── Step 1: DTE gate ─────────────────────────────────────────────────
        if not (self.min_dte <= days_to_expiry <= self.max_dte):
            reasons.append(
                f"DTE={days_to_expiry} outside [{self.min_dte},{self.max_dte}] — no entry."
            )
            return self._no_trade(iv_rank, vrp, vrp_estimate.signal, pcr, days_to_expiry, reasons)

        # ── Step 2: Short-premium conditions ────────────────────────────────
        if iv_rank >= self.iv_rank_high:
            reasons.append(f"IV rank={iv_rank:.1f} >= {self.iv_rank_high} (elevated vol — sell premium).")
            return self._short_premium_signal(
                iv_rank, vrp, vrp_estimate, pcr, days_to_expiry, reasons
            )

        # ── Step 3: Long-premium conditions ─────────────────────────────────
        if iv_rank <= self.iv_rank_low and vrp_estimate.signal == "unfavorable_short_premium":
            reasons.append(
                f"IV rank={iv_rank:.1f} <= {self.iv_rank_low} and VRP unfavorable "
                f"({vrp:+.3f}) — buy volatility."
            )
            confidence = self._long_premium_confidence(iv_rank, vrp)
            if confidence < self.min_confidence:
                reasons.append(f"Confidence {confidence:.0%} below {self.min_confidence:.0%} — no entry.")
                return self._no_trade(iv_rank, vrp, vrp_estimate.signal, pcr, days_to_expiry, reasons)
            return FOSignal(
                strategy_name=STRATEGY_LONG_STRADDLE,
                trade_direction=DIRECTION_LONG_PREMIUM,
                confidence=round(confidence, 3),
                iv_rank=iv_rank,
                vrp=vrp,
                vrp_signal=vrp_estimate.signal,
                pcr=pcr,
                days_to_expiry=days_to_expiry,
                reasoning=reasons,
            )

        # ── Step 4: Neutral — no edge ────────────────────────────────────────
        reasons.append(
            f"No edge: iv_rank={iv_rank:.1f}, vrp={vrp:+.3f}, pcr={pcr:.2f}."
        )
        return self._no_trade(iv_rank, vrp, vrp_estimate.signal, pcr, days_to_expiry, reasons)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _short_premium_signal(
        self,
        iv_rank:      float,
        vrp:          float,
        vrp_estimate: VRPEstimate,
        pcr:          float,
        dte:          int,
        reasons:      list[str],
    ) -> FOSignal:
        """Select specific short-premium strategy and compute confidence."""
        vrp_favorable = vrp_estimate.signal == "favorable_short_premium"

        if vrp_favorable:
            reasons.append(f"VRP={vrp:+.3f} > {0.03} (favorable short premium — implied > realized).")
        else:
            reasons.append(f"VRP={vrp:+.3f} — not fully favorable (signal={vrp_estimate.signal}).")

        confidence = self._short_premium_confidence(iv_rank, vrp, pcr, vrp_estimate)

        if confidence < self.min_confidence:
            reasons.append(
                f"Composite confidence {confidence:.0%} below {self.min_confidence:.0%} — no entry."
            )
            return self._no_trade(iv_rank, vrp, vrp_estimate.signal, pcr, dte, reasons)

        # Strategy selection by PCR
        if pcr > self.pcr_bearish:
            strategy = STRATEGY_SHORT_STRANGLE
            reasons.append(
                f"PCR={pcr:.2f} > {self.pcr_bearish} (bearish skew) → Short Strangle."
            )
        elif pcr < self.pcr_bullish:
            strategy = STRATEGY_BULL_PUT_SPREAD
            reasons.append(
                f"PCR={pcr:.2f} < {self.pcr_bullish} (bullish skew) → Bull Put Spread."
            )
        else:
            strategy = STRATEGY_IRON_CONDOR
            reasons.append(
                f"PCR={pcr:.2f} in [{self.pcr_bullish},{self.pcr_bearish}] (balanced) → Iron Condor."
            )

        sig = FOSignal(
            strategy_name=strategy,
            trade_direction=DIRECTION_SHORT_PREMIUM,
            confidence=round(confidence, 3),
            iv_rank=iv_rank,
            vrp=vrp,
            vrp_signal=vrp_estimate.signal,
            pcr=pcr,
            days_to_expiry=dte,
            reasoning=reasons,
        )
        log.info(
            "fo_signal.generated",
            strategy=strategy,
            confidence=round(confidence, 3),
            iv_rank=round(iv_rank, 1),
            vrp=round(vrp, 4),
            pcr=round(pcr, 3),
            dte=dte,
        )
        return sig

    def _short_premium_confidence(
        self,
        iv_rank:      float,
        vrp:          float,
        pcr:          float,
        vrp_estimate: VRPEstimate,
    ) -> float:
        """
        Composite confidence score for short-premium entry.

        Weights:
          40% — IV rank: linearly scaled from iv_rank_high (0.5) to 100 (1.0)
          35% — VRP signal strength (from VRPEstimate.signal_strength)
          25% — PCR alignment: best when PCR is far from 1.0 (strong skew)

        Returns float in [0.0, 1.0].
        """
        # IV rank component: 0.5 at threshold, 1.0 at 100
        iv_score = 0.5 + 0.5 * max(0.0, (iv_rank - self.iv_rank_high) / (100.0 - self.iv_rank_high))
        iv_score = min(iv_score, 1.0)

        # VRP component: signal_strength is already [0, 1], 0 if unfavorable
        vrp_score = vrp_estimate.signal_strength if vrp > 0 else 0.0

        # PCR component: linear distance from 1.0 (neutral), capped at 1.0
        pcr_score = min(abs(pcr - 1.0), 1.0)

        composite = _W_IV_RANK * iv_score + _W_VRP * vrp_score + _W_PCR * pcr_score
        return round(min(composite, 1.0), 4)

    def _long_premium_confidence(self, iv_rank: float, vrp: float) -> float:
        """Confidence for long-premium (buy vol) entry."""
        # IV rank component: stronger the closer to 0
        iv_score = max(0.0, 1.0 - iv_rank / self.iv_rank_low)
        # VRP component: more negative VRP = higher confidence to buy vol
        vrp_score = min(abs(vrp) / 0.20, 1.0) if vrp < 0 else 0.0
        composite = _W_IV_RANK * iv_score + _W_VRP * vrp_score
        return round(min(composite, 1.0), 4)

    @staticmethod
    def _no_trade(
        iv_rank: float,
        vrp: float,
        vrp_signal: str,
        pcr: float,
        dte: int,
        reasons: list[str],
    ) -> FOSignal:
        log.debug(
            "fo_signal.no_trade",
            iv_rank=round(iv_rank, 1),
            vrp=round(vrp, 4),
            pcr=round(pcr, 3),
            dte=dte,
        )
        return FOSignal(
            strategy_name=STRATEGY_NO_TRADE,
            trade_direction=DIRECTION_NEUTRAL,
            confidence=0.0,
            iv_rank=iv_rank,
            vrp=vrp,
            vrp_signal=vrp_signal,
            pcr=pcr,
            days_to_expiry=dte,
            reasoning=reasons,
        )
