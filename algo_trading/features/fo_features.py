"""
Stage 4: F&O chain-derived features for the ML model.

Features computed from a live or synthetic OptionChain:
  - Put-Call Ratio (PCR) by OI and by volume
  - Max Pain strike
  - IV Rank / IV Percentile (rolling 252-day window)
  - IV Skew (25-delta proxy: 90% ATM put vs 110% ATM call)
  - ATM IV
  - OI Buildup signal
  - Call Wall / Put Wall (highest-OI strikes)
  - PCR sentiment signal
"""
from __future__ import annotations

STAGE = "4"

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque

import structlog

from algo_trading.data.fo_chain import OptionChain, OptionContract

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_VALID_OI_BUILDUP = frozenset(
    {"long_buildup", "short_buildup", "long_unwinding", "short_unwinding"}
)
_VALID_PCR_SIGNAL = frozenset({"bullish", "neutral", "bearish"})


@dataclass
class FOFeaturesResult:
    timestamp: datetime
    symbol: str
    pcr_oi: float           # put OI / call OI
    pcr_volume: float       # put volume / call volume
    max_pain: float         # strike where option sellers lose least
    iv_rank: float          # current ATM IV position in 52-week range [0, 1]
    iv_percentile: float    # fraction of history below current ATM IV
    iv_skew: float          # OTM put IV - OTM call IV (25-delta proxy)
    atm_iv: float           # average of nearest CE and PE IV
    oi_buildup_signal: str  # long_buildup / short_buildup / long_unwinding / short_unwinding
    call_wall: float        # strike with highest call OI (resistance)
    put_wall: float         # strike with highest put OI  (support)
    price_vs_max_pain_pct: float  # (underlying - max_pain) / max_pain × 100
    pcr_signal: str         # bullish / neutral / bearish


# ---------------------------------------------------------------------------
# Feature engine
# ---------------------------------------------------------------------------

class FOFeatureEngine:
    """
    Stateful feature engine for F&O chain analysis.

    Maintains a rolling history of ATM IV values (default 252 days) used
    to compute IV Rank and IV Percentile.
    """

    def __init__(self, iv_history_window: int = 252) -> None:
        self._iv_history: Deque[float] = deque(maxlen=iv_history_window)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        chain: OptionChain,
        price_change_pct: float = 0.0,
        oi_change_pct: float = 0.0,
    ) -> FOFeaturesResult:
        """
        Compute all F&O features from the provided OptionChain.

        Parameters
        ----------
        chain            : OptionChain with greeks already populated
        price_change_pct : % change in underlying price vs prior session
        oi_change_pct    : % change in total OI vs prior session
        """
        atm_strike = chain.atm_strike

        # ATM IV: average of nearest CE and PE IVs
        atm_iv = self._compute_atm_iv(chain, atm_strike)

        pcr_oi, pcr_volume = self._compute_pcr(chain)
        max_pain = self._compute_max_pain(chain)
        iv_rank, iv_percentile = self._compute_iv_rank(atm_iv)
        iv_skew = self._compute_iv_skew(chain, atm_strike)
        oi_signal = self._compute_oi_buildup(chain, price_change_pct, oi_change_pct)
        call_wall, put_wall = self._compute_walls(chain)

        price_vs_max_pain_pct = (
            (chain.underlying_price - max_pain) / max_pain * 100.0
            if max_pain != 0.0
            else 0.0
        )

        if pcr_oi < 0.8:
            pcr_signal = "bullish"
        elif pcr_oi > 1.2:
            pcr_signal = "bearish"
        else:
            pcr_signal = "neutral"

        result = FOFeaturesResult(
            timestamp=chain.timestamp,
            symbol=chain.symbol,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_volume,
            max_pain=max_pain,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_skew=iv_skew,
            atm_iv=atm_iv,
            oi_buildup_signal=oi_signal,
            call_wall=call_wall,
            put_wall=put_wall,
            price_vs_max_pain_pct=price_vs_max_pain_pct,
            pcr_signal=pcr_signal,
        )

        logger.debug(
            "fo_features_computed",
            symbol=chain.symbol,
            pcr_oi=round(pcr_oi, 4),
            max_pain=max_pain,
            iv_rank=round(iv_rank, 4),
            atm_iv=round(atm_iv, 4),
            oi_signal=oi_signal,
            pcr_signal=pcr_signal,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_atm_iv(self, chain: OptionChain, atm_strike: float) -> float:
        """Average IV of the CE and PE closest to ATM strike."""
        atm_calls = [c for c in chain.calls if c.strike == atm_strike and c.iv > 0]
        atm_puts = [p for p in chain.puts if p.strike == atm_strike and p.iv > 0]

        ivs: list[float] = []
        if atm_calls:
            ivs.append(atm_calls[0].iv)
        if atm_puts:
            ivs.append(atm_puts[0].iv)

        if not ivs:
            # Fall back: nearest contracts with positive IV
            all_with_iv = [c for c in chain.contracts if c.iv > 0]
            if all_with_iv:
                nearest = min(all_with_iv, key=lambda c: abs(c.strike - chain.underlying_price))
                return nearest.iv
            return 0.25  # hard fallback

        return sum(ivs) / len(ivs)

    def _compute_pcr(self, chain: OptionChain) -> tuple[float, float]:
        """
        Put-Call Ratio by OI and by volume.
        A +1 denominator guard prevents division by zero.
        """
        total_put_oi = sum(p.oi for p in chain.puts)
        total_call_oi = sum(c.oi for c in chain.calls)
        total_put_vol = sum(p.volume for p in chain.puts)
        total_call_vol = sum(c.volume for c in chain.calls)

        pcr_oi = total_put_oi / (total_call_oi + 1)
        pcr_volume = total_put_vol / (total_call_vol + 1)
        return pcr_oi, pcr_volume

    def _compute_max_pain(self, chain: OptionChain) -> float:
        """
        Max pain: strike where total option-seller loss is minimised.

        For each candidate strike K:
          call_loss(K) = Σ_{K_i > K}  (K_i - K) × call_OI(K_i)
          put_loss(K)  = Σ_{K_i < K}  (K  - K_i) × put_OI(K_i)
          total_loss   = call_loss + put_loss

        Returns the strike with the smallest total_loss.
        """
        # Build OI maps keyed by strike
        call_oi: dict[float, int] = {}
        for c in chain.calls:
            call_oi[c.strike] = call_oi.get(c.strike, 0) + c.oi

        put_oi: dict[float, int] = {}
        for p in chain.puts:
            put_oi[p.strike] = put_oi.get(p.strike, 0) + p.oi

        strikes = sorted(set(call_oi) | set(put_oi))
        if not strikes:
            return chain.underlying_price

        min_loss = math.inf
        max_pain_strike = strikes[0]

        for k in strikes:
            # ITM calls lose money to seller when underlying settles > K
            call_loss = sum(
                (ki - k) * call_oi.get(ki, 0)
                for ki in strikes
                if ki > k
            )
            # ITM puts lose money to seller when underlying settles < K
            put_loss = sum(
                (k - ki) * put_oi.get(ki, 0)
                for ki in strikes
                if ki < k
            )
            total = call_loss + put_loss
            if total < min_loss:
                min_loss = total
                max_pain_strike = k

        return max_pain_strike

    def _compute_iv_rank(self, atm_iv: float) -> tuple[float, float]:
        """
        Update the rolling IV history and compute rank / percentile.

        iv_rank       = (current - min) / (max - min + ε)   ∈ [0, 1]
        iv_percentile = fraction of historical IVs < current ∈ [0, 1]
        """
        self._iv_history.append(atm_iv)
        history = list(self._iv_history)

        low = min(history)
        high = max(history)
        iv_rank = (atm_iv - low) / (high - low + 1e-8)
        iv_rank = float(max(0.0, min(1.0, iv_rank)))  # clamp to [0, 1]

        below = sum(1 for v in history if v < atm_iv)
        iv_percentile = below / len(history)

        return iv_rank, iv_percentile

    def _compute_iv_skew(self, chain: OptionChain, atm_strike: float) -> float:
        """
        25-delta skew proxy:
          OTM put  ≈ strike closest to 90% of ATM
          OTM call ≈ strike closest to 110% of ATM

        Returns otm_put_iv - otm_call_iv.
        Positive skew = market fears downside (normal for equities).
        """
        otm_put_target = atm_strike * 0.90
        otm_call_target = atm_strike * 1.10

        puts_with_iv = [p for p in chain.puts if p.iv > 0]
        calls_with_iv = [c for c in chain.calls if c.iv > 0]

        if not puts_with_iv or not calls_with_iv:
            return 0.0

        otm_put = min(puts_with_iv, key=lambda p: abs(p.strike - otm_put_target))
        otm_call = min(calls_with_iv, key=lambda c: abs(c.strike - otm_call_target))

        return otm_put.iv - otm_call.iv

    def _compute_oi_buildup(
        self,
        chain: OptionChain,
        price_change_pct: float,
        oi_change_pct: float,
    ) -> str:
        """
        Classify OI buildup based on price and OI direction.

        Thresholds:
          |price_change_pct| > 0.1%
          |oi_change_pct|    > 1.0%
        """
        price_up = price_change_pct > 0.1
        price_dn = price_change_pct < -0.1
        oi_up = oi_change_pct > 1.0
        oi_dn = oi_change_pct < -1.0

        if price_up and oi_up:
            return "long_buildup"
        if price_dn and oi_up:
            return "short_buildup"
        if price_dn and oi_dn:
            return "long_unwinding"
        if price_up and oi_dn:
            return "short_unwinding"

        # Changes below threshold — classify by dominant direction
        if oi_up:
            return "short_buildup" if price_dn else "long_buildup"
        return "long_unwinding"  # neutral / inconclusive

    def _compute_walls(self, chain: OptionChain) -> tuple[float, float]:
        """
        Call wall = strike with highest call OI (acts as resistance).
        Put wall  = strike with highest put OI  (acts as support).
        """
        calls = chain.calls
        puts = chain.puts

        if calls:
            call_wall = max(calls, key=lambda c: c.oi).strike
        else:
            call_wall = chain.underlying_price

        if puts:
            put_wall = max(puts, key=lambda p: p.oi).strike
        else:
            put_wall = chain.underlying_price

        return call_wall, put_wall
