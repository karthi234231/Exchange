"""
algo_trading/fo/margin.py — SPAN-like margin approximation for NSE F&O.

[APPROX-NOT-OFFICIAL-SPAN]
============================================================
THIS MODULE IMPLEMENTS A SPAN APPROXIMATION, NOT OFFICIAL NSE SPAN.

Background
----------
CME invented SPAN (Standard Portfolio Analysis of Risk) in 1988.
NSE Clearing uses a SPAN variant. The OFFICIAL NSE SPAN margin requires
NSE Clearing's proprietary daily risk-parameter files (not publicly available).

This module APPROXIMATES the key SPAN logic for pre-trade position sizing.
DO NOT use this for actual margin calls, collateral management, or risk reporting.
Always obtain official margin requirements from your broker's API or NSE Clearing.

ASSUMED constants (verify against current SEBI/NSE circulars before going live):
  - SOMC rate:           3% of max(underlying, strike) × lot_size
  - Exposure margin:     1.5% of notional (index options)
  - Vol scenario range:  ±25% relative vol change for normal scenarios
  - Extreme scenario:    ±2σ price moves at 35% loss weight
  - Capital utilisation: 50% default for max-lots sizing

NSE SPAN Reference:
  NSE Clearing SPAN documentation (requires login):
  https://www.nseclearing.com/clearing-settlement/risk-management/span-margin/
  SEBI circular CIR/MRD/DRMNP/26/2013 (framework for portfolio margining)

Author: NSE AlgoTrader project
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import structlog

from algo_trading.fo.pricing import BSMEngine, option_price

log = structlog.get_logger(__name__)

# ── Approximation constants — ALL ASSUMED; verify before live use ────────────

_APPROX_TAG = "[APPROX-NOT-OFFICIAL-SPAN]"
_APPROX_WARNING = (
    "[APPROX-NOT-OFFICIAL-SPAN] Use official NSE Clearing SPAN files for actual margin."
)

# ASSUMED: Short Option Minimum Charge as % of max(underlying, strike).
# Historical NSE rate was 3%; verify SEBI/NSE circulars for current value.
_SOMC_PCT: float = 0.03  # ASSUMED: 3%

# ASSUMED: Exposure margin rate for index options = 1.5% of notional.
# Source: NSE circular (percentage; verify for non-index instruments).
_EXPOSURE_MARGIN_PCT: float = 0.015  # ASSUMED: 1.5%

# ASSUMED: Relative vol change for SPAN vol scenarios (±25%).
_VOL_UP_FACTOR:   float = 1.25  # vol up  = sigma × 1.25
_VOL_DOWN_FACTOR: float = 0.75  # vol down = sigma × 0.75

# ASSUMED: Extreme scenario price-move multiplier (scenarios 15 & 16).
# Using 3σ (daily) to ensure extreme scenarios generate worst-case losses for
# short-premium strategies.  2σ daily moves are often dominated by vol-up scenarios
# for close-to-ATM positions; 3σ ensures genuine tail-risk capture.
_EXTREME_SIGMA_MULT: float = 3.0   # ±3σ price move (ASSUMED; verify vs NSE SPAN files)
_EXTREME_WEIGHT:     float = 0.35  # 35% fractional credit for extreme scenarios

# ASSUMED: Default capital utilisation for max_lots_for_capital.
_DEFAULT_CAPITAL_UTIL: float = 0.50  # use at most 50% of capital for margin

# Safety floor for T_years to avoid BSM domain errors at/near expiry.
_MIN_T_YEARS: float = 1.0 / 365.0


# ── Strategy leg interface ────────────────────────────────────────────────────
# StrategyLeg is defined in algo_trading.fo.strategies (not yet committed).
# We use a duck-typed dataclass here to avoid a circular import if strategies.py
# itself imports from margin.py, and to keep this module self-contained if
# strategies.py is absent.  When strategies.py exists the real dataclass is
# assignment-compatible (same field names/types).

@dataclass
class StrategyLeg:
    """
    Minimal representation of one leg in an F&O strategy.

    This mirrors the interface defined in algo_trading.fo.strategies.StrategyLeg.
    Import from strategies when available; use this stub for standalone use.

    Fields
    ------
    option_type    : "CE" (call), "PE" (put), or "FUT" (futures)
    strike         : Strike price in INR (ignored for FUT legs)
    expiry         : Expiry date of the contract
    direction      : +1 = long, -1 = short
    lots           : Number of lots
    lot_size       : Units per lot (from NSE contract spec, e.g. 65 for Nifty)
    entry_premium  : Premium/price per unit at trade entry (INR)
    """
    option_type:   str    # "CE", "PE", "FUT"
    strike:        float
    expiry:        date
    direction:     int    # +1 long, -1 short
    lots:          int
    lot_size:      int
    entry_premium: float


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ScenarioPnL:
    """
    P&L for the entire portfolio under one SPAN price/vol scenario.

    Attributes
    ----------
    scenario_id       : 1–16, matching standard SPAN numbering
    price_move_sigma  : Price move in σ units (e.g. +1.0, -0.667, etc.)
    vol_multiplier    : Volatility multiplier (1.25 = vol up, 0.75 = vol down,
                        1.0 = flat, only used for extreme scenarios)
    price_move_inr    : Actual INR price move applied to underlying
    pnl               : Aggregate P&L across all legs (positive = profit, negative = loss)
    """
    scenario_id:      int
    price_move_sigma: float
    vol_multiplier:   float
    price_move_inr:   float
    pnl:              float


@dataclass
class SPANResult:
    """
    [APPROX-NOT-OFFICIAL-SPAN] Estimated SPAN margin for an F&O strategy.

    All monetary values are in INR.

    Attributes
    ----------
    span_margin           : max(scan_risk across 16 scenarios, SOMC)
    somc                  : Short Option Minimum Charge aggregate [ASSUMED: 3%]
    exposure_margin       : 1.5% of total notional [ASSUMED: 1.5%]
    total_initial_margin  : span_margin + exposure_margin
    worst_scenario        : The scenario driving the maximum loss
    all_scenarios         : Full list of all 16 ScenarioPnL results
    WARNING               : Always set to the APPROX disclaimer string
    """
    span_margin:          float
    somc:                 float
    exposure_margin:      float   # [ASSUMED: 1.5% of notional]
    total_initial_margin: float   # span_margin + exposure_margin
    worst_scenario:       ScenarioPnL
    all_scenarios:        list[ScenarioPnL]
    WARNING:              str = field(
        default=_APPROX_WARNING,
        init=False,
    )

    def __post_init__(self) -> None:
        # Enforce the warning tag — value is fixed regardless of any future
        # changes to _APPROX_WARNING at call time.
        self.WARNING = _APPROX_WARNING


# ── SPAN margin engine ────────────────────────────────────────────────────────

class SPANMarginEngine:
    """
    [APPROX-NOT-OFFICIAL-SPAN] Simulated SPAN margin estimator for NSE F&O.

    Approximates the CME/NSE SPAN methodology using 16 price/vol scenarios:
      - Scenarios 1–14: seven price-move levels (±1σ, ±2/3σ, ±1/3σ, 0) each
        paired with vol up (+25%) and vol down (−25%).
      - Scenarios 15–16: extreme ±2σ price moves with 35% loss weighting.

    SPAN margin = max(worst scenario loss across all 16, SOMC)
    Total initial margin = SPAN margin + exposure margin (1.5% of notional)

    IMPORTANT LIMITATIONS
    ---------------------
    1.  This is an APPROXIMATION.  Official NSE SPAN uses proprietary risk arrays
        from NSE Clearing that are not publicly available.
    2.  SOMC rate (3%) and exposure margin (1.5%) are ASSUMED based on historical
        NSE practice — verify against current SEBI/NSE circulars before live use.
    3.  Futures legs are repriced using a linear (delta-only) approximation:
        ΔP_futures = direction × lots × lot_size × price_move_inr.
        No funding/carry adjustment is applied across scenarios.
    4.  Deep-in-the-money options near expiry may have BSM pricing anomalies;
        the engine floors T at 1/365 years to prevent domain errors.
    5.  Cross-margin netting (portfolio SPAN) between unrelated positions is NOT
        implemented — each strategy is treated as an isolated portfolio.

    Parameters
    ----------
    bsm_engine : BSMEngine
        A configured BSMEngine instance carrying r and q defaults.
        Used to reprice each leg across the 16 SPAN scenarios.

    Usage
    -----
    bsm    = BSMEngine(r=0.065, q=0.015)
    engine = SPANMarginEngine(bsm_engine=bsm)

    legs = [
        StrategyLeg("CE", strike=24500, expiry=date(2026,6,26),
                    direction=-1, lots=1, lot_size=65, entry_premium=50.0),
        StrategyLeg("PE", strike=23500, expiry=date(2026,6,26),
                    direction=-1, lots=1, lot_size=65, entry_premium=45.0),
    ]
    result = engine.compute_span(legs, underlying=24000, annual_vol=0.14, T_years=7/365)
    print(result.total_initial_margin)
    print(result.WARNING)
    """

    def __init__(self, bsm_engine: BSMEngine) -> None:
        self._bsm = bsm_engine

    # ── 16 SPAN scenario definitions ─────────────────────────────────────────

    @staticmethod
    def _build_scenarios() -> list[tuple[int, float, float, bool]]:
        """
        Return the 16 standard SPAN scenarios as a list of:
            (scenario_id, price_move_sigma, vol_multiplier, is_extreme)

        Standard 16 SPAN scenarios:
          Scenario  1: +1σ,    vol up   (×1.25)
          Scenario  2: +1σ,    vol down (×0.75)
          Scenario  3: +2/3σ,  vol up
          Scenario  4: +2/3σ,  vol down
          Scenario  5: +1/3σ,  vol up
          Scenario  6: +1/3σ,  vol down
          Scenario  7:  0,     vol up
          Scenario  8:  0,     vol down
          Scenario  9: -1/3σ,  vol up
          Scenario 10: -1/3σ,  vol down
          Scenario 11: -2/3σ,  vol up
          Scenario 12: -2/3σ,  vol down
          Scenario 13: -1σ,    vol up
          Scenario 14: -1σ,    vol down
          Scenario 15: +2σ extreme, no vol shift (35% weight applied to loss)
          Scenario 16: -2σ extreme, no vol shift (35% weight applied to loss)
        """
        price_moves = [
            +1.0,
            +2.0 / 3.0,
            +1.0 / 3.0,
            0.0,
            -1.0 / 3.0,
            -2.0 / 3.0,
            -1.0,
        ]
        scenarios: list[tuple[int, float, float, bool]] = []
        scenario_id = 1
        for pm in price_moves:
            for vol_mult in (_VOL_UP_FACTOR, _VOL_DOWN_FACTOR):
                scenarios.append((scenario_id, pm, vol_mult, False))
                scenario_id += 1
        # Scenarios 15 and 16: extreme moves
        scenarios.append((15, +_EXTREME_SIGMA_MULT, 1.0, True))
        scenarios.append((16, -_EXTREME_SIGMA_MULT, 1.0, True))
        return scenarios

    # ── Per-leg repricing ─────────────────────────────────────────────────────

    def _reprice_leg(
        self,
        leg: StrategyLeg,
        S_new: float,
        sigma_new: float,
        T_years: float,
    ) -> float:
        """
        Return the mark-to-market P&L for one leg under a scenario.

        For option legs: full BSM reprice using (S_new, sigma_new).
        For futures legs: linear P&L = direction × lots × lot_size × (S_new − entry_premium).
        The entry_premium for a futures leg should be the futures price at entry.

        Returns
        -------
        float
            P&L in INR for this leg (positive = profit, negative = loss).
        """
        qty = leg.lots * leg.lot_size  # total units

        if leg.option_type == "FUT":
            # Linear futures P&L — no vol sensitivity
            pnl = leg.direction * qty * (S_new - leg.entry_premium)
        else:
            T_safe = max(T_years, _MIN_T_YEARS)
            new_px = option_price(
                S=S_new,
                K=leg.strike,
                T=T_safe,
                r=self._bsm.r,
                q=self._bsm.q,
                sigma=max(sigma_new, 1e-6),
                option_type=leg.option_type,
            )
            # P&L = direction × qty × (new_price − entry_premium)
            # For a short (direction=-1) option: premium received upfront;
            # an increase in option price is a loss.
            pnl = leg.direction * qty * (new_px - leg.entry_premium)

        return pnl

    # ── SOMC calculation ──────────────────────────────────────────────────────

    @staticmethod
    def _somc_for_leg(leg: StrategyLeg, underlying: float) -> float:
        """
        Short Option Minimum Charge for one short option leg.

        [ASSUMED: 3% rate — verify against current SEBI/NSE circulars]

        SOMC = 3% × max(underlying, strike) × lot_size × lots

        Only applies to short option positions (direction == -1).
        FUT legs and long option legs return 0.
        """
        if leg.option_type == "FUT" or leg.direction != -1:
            return 0.0
        # ASSUMED: 3% of max(underlying, strike) × qty
        base = max(underlying, leg.strike)
        qty  = leg.lots * leg.lot_size
        return _SOMC_PCT * base * qty  # ASSUMED: 3%

    # ── Main compute_span ─────────────────────────────────────────────────────

    def compute_span(
        self,
        legs: list[StrategyLeg],
        underlying: float,
        annual_vol: float,
        T_years: float,
        lots: Optional[int] = None,
    ) -> SPANResult:
        """
        [APPROX-NOT-OFFICIAL-SPAN] Compute estimated SPAN initial margin.

        Algorithm
        ---------
        1.  Compute 1-day sigma = underlying × annual_vol / sqrt(252).
        2.  For each of the 16 standard scenarios:
              a. New underlying = underlying + price_move_sigma × sigma_1day
              b. New vol        = annual_vol × vol_multiplier
              c. Reprice each leg with (S_new, sigma_new, T_years)
              d. Sum P&L across all legs → scenario P&L
              e. If extreme scenario (15 or 16): loss = loss × _EXTREME_WEIGHT
        3.  scan_risk = max loss across all 16 scenarios (a positive number).
        4.  SOMC = sum of SOMC for each short option leg.
        5.  SPAN margin = max(scan_risk, SOMC).
        6.  Exposure margin = 1.5% × total notional  [ASSUMED].
            notional = lots × lot_size × underlying  (summed over all legs).
        7.  Total initial margin = SPAN margin + exposure margin.

        Parameters
        ----------
        legs        : List of StrategyLeg objects describing the portfolio.
        underlying  : Current underlying index level (INR).
        annual_vol  : Annualised volatility estimate (e.g. 0.14 for 14%).
                      ASSUMED: caller provides current IV as annual_vol proxy.
        T_years     : Time to nearest expiry in years.
        lots        : Optional scalar to override the lots field of ALL legs
                      (useful during binary-search sizing in max_lots_for_capital).
                      When provided, each leg's lots is replaced by this value.

        Returns
        -------
        SPANResult
            Always contains WARNING = "[APPROX-NOT-OFFICIAL-SPAN] ..."
        """
        if not legs:
            # Empty portfolio: return zero margin with 16 zero-P&L scenarios
            zero_scenarios = [
                ScenarioPnL(
                    scenario_id=sid,
                    price_move_sigma=pm_sigma,
                    vol_multiplier=vol_mult,
                    price_move_inr=0.0,
                    pnl=0.0,
                )
                for sid, pm_sigma, vol_mult, _ in self._build_scenarios()
            ]
            worst = zero_scenarios[0]
            return SPANResult(
                span_margin=0.0,
                somc=0.0,
                exposure_margin=0.0,
                total_initial_margin=0.0,
                worst_scenario=worst,
                all_scenarios=zero_scenarios,
            )
        if underlying <= 0.0:
            raise ValueError(f"underlying must be positive, got {underlying}")
        if annual_vol <= 0.0:
            raise ValueError(f"annual_vol must be positive, got {annual_vol}")
        if T_years <= 0.0:
            T_years = _MIN_T_YEARS
            log.warning(
                "span.T_years_floor",
                msg="T_years <= 0; floored to 1/365",
                T_years=_MIN_T_YEARS,
                tag=_APPROX_TAG,
            )

        # Apply lots override if requested
        effective_legs: list[StrategyLeg]
        if lots is not None:
            if lots < 0:
                raise ValueError(f"lots override must be >= 0, got {lots}")
            effective_legs = [
                StrategyLeg(
                    option_type=lg.option_type,
                    strike=lg.strike,
                    expiry=lg.expiry,
                    direction=lg.direction,
                    lots=lots,
                    lot_size=lg.lot_size,
                    entry_premium=lg.entry_premium,
                )
                for lg in legs
            ]
        else:
            effective_legs = list(legs)

        # ── Step 1: 1-day sigma in INR ────────────────────────────────────────
        sigma_1day_inr: float = underlying * annual_vol / math.sqrt(252.0)

        log.debug(
            "span.inputs",
            underlying=underlying,
            annual_vol=annual_vol,
            T_years=T_years,
            sigma_1day_inr=round(sigma_1day_inr, 2),
            n_legs=len(effective_legs),
            tag=_APPROX_TAG,
        )

        # ── Step 2: 16 scenario P&L ───────────────────────────────────────────
        scenarios_def = self._build_scenarios()
        scenario_results: list[ScenarioPnL] = []

        for sid, pm_sigma, vol_mult, is_extreme in scenarios_def:
            price_move_inr = pm_sigma * sigma_1day_inr
            S_new          = underlying + price_move_inr
            # Ensure S_new stays positive (deep crash scenario)
            S_new          = max(S_new, 1.0)
            sigma_new      = annual_vol * vol_mult

            total_pnl = sum(
                self._reprice_leg(lg, S_new, sigma_new, T_years)
                for lg in effective_legs
            )

            # Store the raw (unweighted) P&L in the scenario record so that
            # callers can observe the full P&L impact of each scenario.
            scenario_results.append(
                ScenarioPnL(
                    scenario_id=sid,
                    price_move_sigma=pm_sigma,
                    vol_multiplier=vol_mult,
                    price_move_inr=price_move_inr,
                    pnl=total_pnl,
                )
            )

        # ── Step 3: scan_risk = max loss (worst P&L is most negative) ─────────
        # worst_scenario tracks the scenario with the minimum raw (unweighted) P&L.
        # scan_risk uses the weighted P&L for extreme scenarios (35% of loss).
        # [ASSUMED: 35% weight per SPAN specification for scenarios 15 & 16]
        worst: ScenarioPnL = min(scenario_results, key=lambda s: s.pnl)
        weighted_losses: list[float] = []
        for s in scenario_results:
            if s.scenario_id in (15, 16) and s.pnl < 0.0:
                weighted_losses.append(s.pnl * _EXTREME_WEIGHT)  # ASSUMED: 35%
            else:
                weighted_losses.append(s.pnl)
        scan_risk: float = max(-min(weighted_losses), 0.0)   # convert loss to positive charge

        # ── Step 4: SOMC ──────────────────────────────────────────────────────
        somc: float = sum(
            self._somc_for_leg(lg, underlying) for lg in effective_legs
        )  # ASSUMED: 3% rate

        # ── Step 5: SPAN margin ───────────────────────────────────────────────
        span_margin: float = max(scan_risk, somc)

        # ── Step 6: Exposure margin ───────────────────────────────────────────
        # [ASSUMED: 1.5% × total notional where notional = lots × lot_size × underlying]
        total_notional: float = sum(
            lg.lots * lg.lot_size * underlying for lg in effective_legs
        )
        exposure_margin: float = _EXPOSURE_MARGIN_PCT * total_notional  # ASSUMED: 1.5%

        # ── Step 7: Total initial margin ──────────────────────────────────────
        total_initial_margin: float = span_margin + exposure_margin

        log.info(
            "span.result",
            scan_risk=round(scan_risk, 2),
            somc=round(somc, 2),
            span_margin=round(span_margin, 2),
            exposure_margin=round(exposure_margin, 2),
            total_initial_margin=round(total_initial_margin, 2),
            worst_scenario=worst.scenario_id,
            worst_pnl=round(worst.pnl, 2),
            tag=_APPROX_TAG,
        )

        return SPANResult(
            span_margin=round(span_margin, 2),
            somc=round(somc, 2),
            exposure_margin=round(exposure_margin, 2),
            total_initial_margin=round(total_initial_margin, 2),
            worst_scenario=worst,
            all_scenarios=scenario_results,
        )

    # ── max_lots_for_capital ──────────────────────────────────────────────────

    def max_lots_for_capital(
        self,
        strategy_template: Callable[..., list[StrategyLeg]],
        strategy_kwargs: dict,
        available_capital: float,
        underlying: float,
        annual_vol: float,
        T_years: float,
        capital_utilization_pct: float = _DEFAULT_CAPITAL_UTIL,  # ASSUMED: 50%
    ) -> int:
        """
        [APPROX-NOT-OFFICIAL-SPAN] Find the maximum lots for a given capital budget.

        Uses binary search to find the largest lot count where:
            total_initial_margin ≤ available_capital × capital_utilization_pct

        The strategy_template callable must accept a 'lots' keyword argument
        (plus any other kwargs supplied via strategy_kwargs) and return a list
        of StrategyLeg objects.  All legs are assumed to scale proportionally
        with lots.

        Parameters
        ----------
        strategy_template       : Callable that builds strategy legs.
                                  Signature: f(lots=N, **strategy_kwargs) → list[StrategyLeg]
        strategy_kwargs         : Additional keyword arguments forwarded to the template.
        available_capital       : Total available capital in INR.
        underlying              : Current underlying price (INR).
        annual_vol              : Annualised volatility (e.g. 0.14 for 14%).
        T_years                 : Time to expiry in years.
        capital_utilization_pct : Maximum fraction of capital for margin.
                                  [ASSUMED: 0.50 = 50%]

        Returns
        -------
        int
            Maximum number of lots that fits within the capital budget.
            Returns 0 if even 1 lot exceeds the capital limit.

        Notes
        -----
        - Binary search upper bound starts at 1,000 lots; extend if needed.
        - The strategy_template is called with lots=1 first to detect errors.
        """
        if available_capital <= 0.0:
            raise ValueError(f"available_capital must be positive, got {available_capital}")
        if not (0.0 < capital_utilization_pct <= 1.0):
            raise ValueError(
                f"capital_utilization_pct must be in (0, 1], got {capital_utilization_pct}"
            )

        margin_budget: float = available_capital * capital_utilization_pct  # ASSUMED: 50%

        def _margin_for_lots(n: int) -> float:
            """Return total_initial_margin for n lots of the strategy template."""
            legs = strategy_template(lots=n, **strategy_kwargs)
            result = self.compute_span(legs, underlying, annual_vol, T_years)
            return result.total_initial_margin

        # ── Quick check: can we even afford 1 lot? ────────────────────────────
        try:
            margin_1_lot = _margin_for_lots(1)
        except Exception as exc:
            log.error(
                "span.max_lots.template_error",
                lots=1,
                error=str(exc),
                tag=_APPROX_TAG,
            )
            raise

        if margin_1_lot > margin_budget:
            log.info(
                "span.max_lots.insufficient_capital",
                margin_1_lot=round(margin_1_lot, 2),
                margin_budget=round(margin_budget, 2),
                result=0,
                tag=_APPROX_TAG,
            )
            return 0

        # ── Binary search ─────────────────────────────────────────────────────
        # Margin scales approximately linearly with lots; binary search is exact.
        lo: int = 1
        hi: int = 1000  # upper bound; extend if strategies with many lots are expected

        # Expand upper bound if hi is not sufficient
        while _margin_for_lots(hi) <= margin_budget:
            hi *= 2
            if hi > 100_000:
                log.warning(
                    "span.max_lots.hi_cap_reached",
                    hi=hi,
                    tag=_APPROX_TAG,
                )
                break

        best: int = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid < 1:
                break
            m = _margin_for_lots(mid)
            if m <= margin_budget:
                best = mid
                lo   = mid + 1
            else:
                hi   = mid - 1

        log.info(
            "span.max_lots.result",
            best_lots=best,
            margin_at_best=round(_margin_for_lots(best), 2),
            margin_budget=round(margin_budget, 2),
            capital_utilization_pct=capital_utilization_pct,
            tag=_APPROX_TAG,
        )
        return best

    # ── stress_test ───────────────────────────────────────────────────────────

    def stress_test(
        self,
        legs: list[StrategyLeg],
        underlying: float,
        annual_vol: float,
        T_years: float,
        extreme_moves: list[float] | None = None,
    ) -> dict[float, float]:
        """
        P&L sensitivity to large underlying spot moves (additional stress analysis).

        This is NOT part of official SPAN — it is supplementary stress testing
        for understanding tail risk outside the standard 16 SPAN scenarios.

        For each percentage move in extreme_moves:
          - S_new = underlying × (1 + pct_move)
          - vol is held constant at annual_vol (flat-vol assumption for stress)
          - Reprice all legs at (S_new, annual_vol, T_years)
          - Compute aggregate P&L

        Parameters
        ----------
        legs          : Strategy legs to stress.
        underlying    : Current underlying price (INR).
        annual_vol    : Annualised volatility (held constant during stress).
        T_years       : Time to expiry in years.
        extreme_moves : List of fractional spot moves (e.g. [-0.10, 0.10]).
                        Default: [-0.10, -0.05, -0.03, 0.03, 0.05, 0.10]

        Returns
        -------
        dict[float, float]
            {pct_move → aggregate P&L in INR}
            Positive P&L = profit; negative = loss.
            Keys are rounded to 4 decimal places for dict hashing stability.
        """
        if extreme_moves is None:
            extreme_moves = [-0.10, -0.05, -0.03, 0.03, 0.05, 0.10]

        if not legs:
            raise ValueError("legs must be a non-empty list")
        if underlying <= 0.0:
            raise ValueError(f"underlying must be positive, got {underlying}")

        T_safe = max(T_years, _MIN_T_YEARS)
        result: dict[float, float] = {}

        for pct in extreme_moves:
            S_new    = max(underlying * (1.0 + pct), 1.0)
            total_pnl = sum(
                self._reprice_leg(lg, S_new, annual_vol, T_safe)
                for lg in legs
            )
            rounded_pct = round(pct, 4)
            result[rounded_pct] = round(total_pnl, 2)
            log.debug(
                "span.stress_test",
                pct_move=rounded_pct,
                S_new=round(S_new, 2),
                pnl=round(total_pnl, 2),
                tag=_APPROX_TAG,
            )

        log.info(
            "span.stress_test.complete",
            n_moves=len(extreme_moves),
            worst_pnl=round(min(result.values()), 2),
            best_pnl=round(max(result.values()), 2),
            note="NOT-OFFICIAL-SPAN; supplementary stress analysis only",
        )
        return result


# ── Convenience factory ───────────────────────────────────────────────────────

def make_span_engine(
    r: float = 0.065,    # ASSUMED: RBI repo rate proxy — verify current rate
    q: float = 0.015,    # ASSUMED: Nifty trailing dividend yield ~1.5%
) -> SPANMarginEngine:
    """
    Convenience factory: construct a SPANMarginEngine with default BSM parameters.

    Parameters
    ----------
    r : float
        Risk-free rate (continuously compounded).
        ASSUMED: 6.5% (RBI repo rate proxy) — verify current RBI policy rate.
    q : float
        Continuous dividend yield.
        ASSUMED: 1.5% (Nifty trailing yield) — update periodically.

    Returns
    -------
    SPANMarginEngine
        Ready-to-use engine with the given r, q parameters.
    """
    bsm = BSMEngine(r=r, q=q)
    return SPANMarginEngine(bsm_engine=bsm)
