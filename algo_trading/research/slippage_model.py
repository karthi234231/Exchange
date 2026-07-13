"""
algo_trading/research/slippage_model.py — Tiered slippage model for NSE F&O backtesting.

This module provides the slippage cost model used during replay/backtesting.
It is SEPARATE from fill_simulator.py — slippage_model handles the cost side,
fill_simulator handles the fill probability and execution side.

Tiered slippage (FO_SYSTEM_DESIGN.md Section 9.3):
  Normal trading:  realistic bid-ask spread model
  Expiry day:      final hour spreads blow out (OTM options especially)
  High-VIX:        IV spike → wider spreads
  Event day:       RBI / Budget → extreme spread widening

All parameters are [ASSUMED] starting values calibrated from general NSE
market observations. MUST be validated against real fills in Phase 4.

Sources:
  [S] CLAUDE.md: "LIMIT 1-tick inside: 3–8 bps avg; MARKET: 8–20 bps"
  [S] NSE Market Data: typical Nifty ATM option spread ~₹0.30–0.50 on normal days
  [S] NSE Options analytics: OTM spread widens 3–5x on expiry day final hour
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Sequence

import structlog

log = structlog.get_logger(__name__)


class SlippageRegime(str, Enum):
    """Market condition that determines slippage tier."""
    NORMAL       = "NORMAL"        # Normal trading day
    HIGH_VIX     = "HIGH_VIX"      # India VIX > 20
    EXPIRY_EARLY = "EXPIRY_EARLY"  # Expiry day but before final hour
    EXPIRY_FINAL = "EXPIRY_FINAL"  # Expiry day last 60 min (14:30–15:30)
    EVENT_DAY    = "EVENT_DAY"     # RBI policy / Budget / major macro
    OPEN_AUCTION = "OPEN_AUCTION"  # 09:15–09:30 (opening auction spillover)


@dataclass(frozen=True)
class SlippageTier:
    """
    Slippage parameters for one market regime.

    Attributes
    ----------
    regime : SlippageRegime
    limit_slippage_bps : float
        Expected slippage for LIMIT orders (inside or at best bid/ask).
        [ASSUMED] Basis points of premium/notional.
    market_slippage_bps : float
        Expected slippage for MARKET orders.
        [ASSUMED] Basis points of premium/notional.
    atm_spread_multiplier : float
        Multiplier on baseline ATM spread (1.0 = no change).
    otm_spread_multiplier : float
        Multiplier for OTM strikes (usually higher than ATM).
    fill_probability_limit : float
        Probability that a LIMIT order fills in one bar [ASSUMED].
    notes : str
    """
    regime: SlippageRegime
    limit_slippage_bps: float      # bps of option premium
    market_slippage_bps: float     # bps of option premium
    atm_spread_multiplier: float
    otm_spread_multiplier: float
    fill_probability_limit: float  # 0.0–1.0
    notes: str


# ── Tiered slippage table ────────────────────────────────────────────────────
# All values are [ASSUMED] — calibrate from Phase 4 real fill data.

SLIPPAGE_TIERS: dict[SlippageRegime, SlippageTier] = {
    SlippageRegime.NORMAL: SlippageTier(
        regime=SlippageRegime.NORMAL,
        limit_slippage_bps=5.0,        # 3–8 bps avg per CLAUDE.md [S]
        market_slippage_bps=12.0,      # 8–20 bps per CLAUDE.md [S]
        atm_spread_multiplier=1.0,
        otm_spread_multiplier=2.0,     # OTM spreads ~2x ATM [ASSUMED]
        fill_probability_limit=0.80,
        notes="Normal trading day, liquid hours",
    ),
    SlippageRegime.OPEN_AUCTION: SlippageTier(
        regime=SlippageRegime.OPEN_AUCTION,
        limit_slippage_bps=20.0,       # Opening auction: wide spreads [ASSUMED]
        market_slippage_bps=40.0,
        atm_spread_multiplier=4.0,     # 3–5x normal per CLAUDE.md SURV-8 [S]
        otm_spread_multiplier=6.0,
        fill_probability_limit=0.50,
        notes="09:15–09:30 opening auction spillover — avoid trading [SURV-8]",
    ),
    SlippageRegime.HIGH_VIX: SlippageTier(
        regime=SlippageRegime.HIGH_VIX,
        limit_slippage_bps=12.0,
        market_slippage_bps=25.0,
        atm_spread_multiplier=2.0,
        otm_spread_multiplier=3.5,
        fill_probability_limit=0.65,
        notes="India VIX > 20 — elevated spreads [ASSUMED]",
    ),
    SlippageRegime.EXPIRY_EARLY: SlippageTier(
        regime=SlippageRegime.EXPIRY_EARLY,
        limit_slippage_bps=8.0,
        market_slippage_bps=18.0,
        atm_spread_multiplier=1.5,
        otm_spread_multiplier=3.0,
        fill_probability_limit=0.70,
        notes="Expiry day morning — OTM spreads widen as positions close",
    ),
    SlippageRegime.EXPIRY_FINAL: SlippageTier(
        regime=SlippageRegime.EXPIRY_FINAL,
        limit_slippage_bps=25.0,       # Very wide for OTM [ASSUMED]
        market_slippage_bps=60.0,
        atm_spread_multiplier=2.0,
        otm_spread_multiplier=5.0,     # OTM near-zero premium: huge spread% [ASSUMED]
        fill_probability_limit=0.45,
        notes="Expiry final 60 min — OTM near worthless, extreme spreads",
    ),
    SlippageRegime.EVENT_DAY: SlippageTier(
        regime=SlippageRegime.EVENT_DAY,
        limit_slippage_bps=30.0,
        market_slippage_bps=70.0,
        atm_spread_multiplier=3.5,     # 3–5x per CLAUDE.md [S]
        otm_spread_multiplier=6.0,
        fill_probability_limit=0.40,
        notes="RBI policy / Budget / major event — extreme uncertainty",
    ),
}


@dataclass
class SlippageEstimate:
    """Slippage estimate for one trade leg."""
    regime: SlippageRegime
    order_type: str             # "LIMIT" or "MARKET"
    is_atm: bool
    slippage_bps: float         # bps of premium
    spread_multiplier: float
    fill_probability: float
    notes: str

    def slippage_inr(self, premium_inr: float) -> float:
        """Slippage cost in INR for given premium amount."""
        return premium_inr * self.slippage_bps / 10_000.0


class SlippageModel:
    """
    Tiered slippage model for NSE F&O backtesting.

    Selects the appropriate SlippageTier based on current market conditions
    and computes expected slippage for a trade.

    Parameters
    ----------
    event_dates : set[date] | None
        Set of event dates (RBI policy, Budget) to flag EVENT_DAY regime.
        If None, event day detection is disabled.

    Usage
    -----
    model = SlippageModel()

    estimate = model.estimate(
        order_type="LIMIT",
        is_atm=True,
        event_time=datetime(2026, 6, 19, 10, 30),
        is_expiry_day=False,
        india_vix=16.0,
    )
    cost_inr = estimate.slippage_inr(premium=150.0 * 65)  # Nifty lot
    """

    def __init__(self, event_dates: set | None = None) -> None:
        self._event_dates = event_dates or set()

    def detect_regime(
        self,
        event_time: datetime,
        is_expiry_day: bool,
        india_vix: float | None = None,
    ) -> SlippageRegime:
        """
        Determine the current slippage regime.

        Priority (highest to lowest):
          EVENT_DAY > EXPIRY_FINAL > EXPIRY_EARLY > OPEN_AUCTION > HIGH_VIX > NORMAL
        """
        t = event_time.time()
        d = event_time.date()

        # Event day check (calendar-based)
        if d in self._event_dates:
            return SlippageRegime.EVENT_DAY

        # Opening auction
        if t < time(9, 30):
            return SlippageRegime.OPEN_AUCTION

        # Expiry final hour
        if is_expiry_day and t >= time(14, 30):
            return SlippageRegime.EXPIRY_FINAL

        # Expiry early (rest of expiry day)
        if is_expiry_day:
            return SlippageRegime.EXPIRY_EARLY

        # High VIX
        if india_vix is not None and india_vix > 20.0:
            return SlippageRegime.HIGH_VIX

        return SlippageRegime.NORMAL

    def estimate(
        self,
        order_type: str,           # "LIMIT" or "MARKET"
        is_atm: bool,
        event_time: datetime,
        is_expiry_day: bool = False,
        india_vix: float | None = None,
        regime_override: SlippageRegime | None = None,
    ) -> SlippageEstimate:
        """
        Estimate slippage for a single option trade leg.

        Parameters
        ----------
        order_type : str
            "LIMIT" or "MARKET"
        is_atm : bool
            True for ATM options; False for OTM (higher spread multiplier)
        event_time : datetime
            Timestamp of the trade (exchange time)
        is_expiry_day : bool
            True if this is the expiry day for the contract
        india_vix : float | None
            Current India VIX level (e.g. 14.5)
        regime_override : SlippageRegime | None
            Force a specific regime (for testing / scenario analysis)
        """
        regime = regime_override or self.detect_regime(
            event_time, is_expiry_day, india_vix
        )
        tier = SLIPPAGE_TIERS[regime]

        if order_type.upper() == "LIMIT":
            base_bps = tier.limit_slippage_bps
        else:
            base_bps = tier.market_slippage_bps

        spread_mult = tier.atm_spread_multiplier if is_atm else tier.otm_spread_multiplier
        # Scale slippage by spread multiplier (wider spread → more slippage)
        scaled_bps = base_bps * spread_mult / tier.atm_spread_multiplier

        return SlippageEstimate(
            regime=regime,
            order_type=order_type,
            is_atm=is_atm,
            slippage_bps=scaled_bps,
            spread_multiplier=spread_mult,
            fill_probability=tier.fill_probability_limit,
            notes=(
                f"{regime.value}: {tier.notes}, "
                f"spread_mult={spread_mult}×, base_bps={base_bps:.1f}"
            ),
        )

    def round_trip_cost_bps(
        self,
        order_type: str,
        is_atm: bool,
        entry_time: datetime,
        exit_time: datetime,
        is_entry_expiry: bool = False,
        is_exit_expiry: bool = False,
        india_vix: float | None = None,
    ) -> float:
        """
        Total round-trip slippage in bps (entry + exit).

        Parameters
        ----------
        entry_time : datetime
            Entry trade time
        exit_time : datetime
            Exit trade time

        Returns
        -------
        float
            Total slippage in bps (entry + exit combined)
        """
        entry_est = self.estimate(order_type, is_atm, entry_time, is_entry_expiry, india_vix)
        exit_est = self.estimate(order_type, is_atm, exit_time, is_exit_expiry, india_vix)
        return entry_est.slippage_bps + exit_est.slippage_bps

    @staticmethod
    def regime_summary() -> None:
        """Print slippage tier table for review."""
        print(f"\n{'Regime':<18} {'Limit(bps)':>10} {'Market(bps)':>12} "
              f"{'ATM mult':>9} {'OTM mult':>9} {'Fill prob':>10}")
        print("-" * 72)
        for regime, tier in SLIPPAGE_TIERS.items():
            print(f"{regime.value:<18} {tier.limit_slippage_bps:>10.1f} "
                  f"{tier.market_slippage_bps:>12.1f} "
                  f"{tier.atm_spread_multiplier:>9.1f} "
                  f"{tier.otm_spread_multiplier:>9.1f} "
                  f"{tier.fill_probability_limit:>9.0%}")
        print("\n⚠ All values are [ASSUMED] — calibrate from Phase 4 real fills")
