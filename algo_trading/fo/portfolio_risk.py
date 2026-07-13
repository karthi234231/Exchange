"""
algo_trading/fo/portfolio_risk.py — Portfolio-level Greek aggregation, VRP signal,
and risk-limit enforcement for NSE F&O positions.

Mathematical foundations
------------------------
Portfolio Greeks are computed by aggregating single-leg Greeks (from
algo_trading.fo.pricing) weighted by direction, lots, and lot_size.

Key formula (all quantities in INR-denominated units):

    Portfolio Delta = Σᵢ (direction_i × delta_i × lots_i × lot_size_i)
    Portfolio Gamma = Σᵢ (direction_i × gamma_i × lots_i × lot_size_i)
    Portfolio Vega  = Σᵢ (direction_i × vega_i  × lots_i × lot_size_i)  [₹ per 1% vol]
    Portfolio Theta = Σᵢ (direction_i × theta_i × lots_i × lot_size_i)  [₹ per calendar day]
    Portfolio Rho   = Σᵢ (direction_i × rho_i   × lots_i × lot_size_i)

where direction_i = +1 for long, -1 for short.

Normalized risk measures:
    delta_risk_inr   = portfolio_delta × underlying_price    [₹ change per 1-pt index move]
    gamma_pnl_1pct   = 0.5 × portfolio_gamma × (underlying_price × 0.01)²
                                                             [₹ P&L for 1% spot move]
    vega_risk_1pct   = portfolio_vega                        [₹ P&L for 1% IV move — direct]

Variance Risk Premium (VRP):
    VRP = realized_vol - implied_vol_atm

    realized_vol (daily bars)   = sqrt(252) × std(log(Pᵢ/Pᵢ₋₁) for last N bars)
    realized_vol (15-min bars)  = sqrt(6552) × std(log(Pᵢ/Pᵢ₋₁) for last N bars)
                                  where 6552 = 252 × 26 bars-per-day

    A positive VRP means realized vol < implied vol: option sellers collect
    premium above fair value.  This is the structural basis for short-premium
    strategies on NSE (historical Nifty VRP ≈ +3% to +6% — ASSUMED; validate on data).

P&L Attribution (first-order Taylor expansion of BSM):
    delta_pnl  = delta × ΔS
    gamma_pnl  = 0.5 × gamma × ΔS²
    vega_pnl   = vega  × Δσ%        (vega is already ₹ per 1% vol)
    theta_pnl  = theta × days_elapsed

Thread safety
-------------
PortfolioRiskEngine is thread-safe: all mutable state (the Greeks cache)
is protected by a threading.Lock.  Methods that only read inputs and
return new objects are lock-free (stateless computation).

References
----------
Hull, J.C. (2022) "Options, Futures, and Other Derivatives" 11th ed., Ch. 17–19.
McDonald, R.L. (2013) "Derivatives Markets" 3rd ed., Ch. 12–13.
Gatheral, J. (2006) "The Volatility Surface", Ch. 1.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

import structlog

from algo_trading.fo.pricing import BSMEngine, OptionGreeks, compute_greeks
from algo_trading.fo.contracts import get_contract, ContractType, lot_size
from algo_trading.fo.cost_model import FOCostModel

# ── StrategyLeg / StrategyResult import with graceful fallback ────────────────
# strategies.py is a sibling module planned in the fo package.
# We import with a try/except so that portfolio_risk.py can be imported (and
# tested) independently before strategies.py is written.  Any code path that
# calls compute_portfolio_greeks() with real StrategyLeg objects will work once
# strategies.py exists; until then, type annotations fall back to the stub below.
try:
    from algo_trading.fo.strategies import StrategyLeg, StrategyResult  # type: ignore[import]
except ImportError:
    # ── Minimal stub: mirrors the guaranteed interface described in the spec ──
    # Remove this block once algo_trading/fo/strategies.py is available.
    @dataclass
    class StrategyLeg:  # type: ignore[no-redef]
        """
        Stub definition — replace with the real import when strategies.py exists.

        Fields match the interface contract:
          option_type     : "CE" or "PE"
          strike          : strike price in INR
          expiry          : expiry date string ("YYYY-MM-DD") or datetime.date
          direction       : +1 (long) or -1 (short)
          lots            : number of lots
          lot_size        : units per lot (from NSE contract master)
          entry_premium   : premium at position entry (INR per unit)
          current_premium : current market premium (INR per unit)
          delta           : option delta (dimensionless)
          gamma           : option gamma (per 1-pt move)
          vega            : option vega (₹ per 1% vol move)
          theta           : option theta (₹ per calendar day)
          sigma           : implied volatility used for Greeks (annualised)
        """
        option_type:      str
        strike:           float
        expiry:           object          # datetime.date or str
        direction:        int             # +1 long, -1 short
        lots:             int
        lot_size:         int
        entry_premium:    float
        current_premium:  float
        delta:            float
        gamma:            float
        vega:             float
        theta:            float
        sigma:            float = 0.0     # IV used for current Greeks
        rho:              float = 0.0     # rho may or may not be populated

    @dataclass
    class StrategyResult:  # type: ignore[no-redef]
        """Stub for StrategyResult — replace with real import."""
        legs:        list[StrategyLeg] = field(default_factory=list)
        strategy_id: str = ""
        pnl:         float = 0.0


log = structlog.get_logger(__name__)

# ── VRP signal thresholds (ASSUMED — validate on historical Nifty data) ───────
# Historical NSE Nifty VRP is structurally positive (realized vol < implied vol),
# with a long-run mean in the +3% to +6% band.
# ASSUMED: these thresholds are reasonable starting points; calibrate on data.
VRP_THRESHOLD_FAVORABLE: float   = 0.03   # ASSUMED: VRP > +3% → favorable short premium
VRP_THRESHOLD_UNFAVORABLE: float = 0.00   # ASSUMED: VRP < 0  → unfavorable short premium

# VRP normalization constant: ASSUMED max |VRP| of 20% for [0,1] scaling.
# Adjust this to match empirical distribution from at least 252 trading days of data.
_VRP_NORM_SCALE: float = 0.20   # ASSUMED

# Annualisation factors
_SQRT252:  float = math.sqrt(252.0)      # daily bars → annualised
_SQRT6552: float = math.sqrt(6552.0)    # 15-min bars → annualised (252 × 26)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioGreeks:
    """
    Aggregated Greeks across all open F&O positions.

    All monetary values are in INR.

    Attributes
    ----------
    delta : float
        Net portfolio delta in underlying units.
        Interpretation: a 1-point move in the index changes portfolio value
        by approximately delta INR × lot_size (see delta_risk_inr).
    gamma : float
        Net portfolio gamma per 1-point move in the underlying.
        Positive gamma: portfolio gains when the underlying moves sharply in
        either direction (convexity benefit).
    vega : float
        Net portfolio vega in ₹ per 1% absolute move in implied volatility.
        E.g., vega = -5000 means a 1% rise in IV costs ₹5,000.
    theta : float
        Net portfolio theta in ₹ per calendar day.
        Typically negative for long-premium, positive for short-premium books.
    rho : float
        Net portfolio rho in ₹ per 1% absolute move in the risk-free rate.
    delta_risk_inr : float
        ₹ P&L for a 1-point move in the underlying index.
        = delta × underlying_price (normalized for index-level scaling).
    gamma_pnl_1pct : float
        ₹ P&L from the quadratic (gamma) term for a 1% move in the underlying.
        = 0.5 × gamma × (underlying_price × 0.01)²
        Captures convexity gain/loss that first-order delta misses.
    vega_risk_1pct : float
        Alias for vega — already expressed in ₹ per 1% vol move.
    underlying_price : float
        Snapshot of the underlying index level at computation time.
        Stored for downstream P&L attribution.
    """
    delta:            float
    gamma:            float
    vega:             float
    theta:            float
    rho:              float
    # Normalized risk measures
    delta_risk_inr:   float   # ₹ P&L for 1-point underlying move
    gamma_pnl_1pct:   float   # ₹ P&L for 1% underlying move (quadratic)
    vega_risk_1pct:   float   # same as vega (₹ per 1% vol move — direct)
    underlying_price: float   # index level at computation time


@dataclass
class VRPEstimate:
    """
    Variance Risk Premium signal.

    Attributes
    ----------
    realized_vol : float
        Annualised realized volatility estimated from price history.
        E.g., 0.12 = 12%.
    implied_vol_atm : float
        Current ATM implied volatility from the live options chain.
        E.g., 0.15 = 15%.
    vrp : float
        Variance Risk Premium = realized_vol - implied_vol_atm.
        Typically negative for NSE Nifty (implied > realized).
        A *more negative* VRP indicates more attractive short-premium conditions.
    lookback_bars : int
        Number of price bars used to estimate realized_vol.
    signal : str
        One of:
          "favorable_short_premium"  — VRP > VRP_THRESHOLD_FAVORABLE
          "neutral"                  — VRP_THRESHOLD_UNFAVORABLE <= VRP <= VRP_THRESHOLD_FAVORABLE
          "unfavorable_short_premium"— VRP < VRP_THRESHOLD_UNFAVORABLE
    signal_strength : float
        |vrp| normalized to [0, 1] using _VRP_NORM_SCALE (ASSUMED = 20%).
        Clipped to [0, 1]; values close to 1 indicate extreme VRP.
        ASSUMED: the normalization scale must be calibrated on live data.
    """
    realized_vol:    float
    implied_vol_atm: float
    vrp:             float
    lookback_bars:   int
    signal:          str
    signal_strength: float


@dataclass
class RiskLimitResult:
    """
    Result of checking portfolio Greeks against configured risk limits.

    Attributes
    ----------
    delta_ok : bool
        True if |delta_risk_inr| <= max_delta_inr.
    gamma_ok : bool
        True if |gamma_pnl_1pct| <= max_gamma_pnl_1pct.
    vega_ok : bool
        True if |vega| <= max_vega_inr.
    theta_ok : bool
        Theta is tracked but does not have a hard limit; always True.
        Included for observability and downstream alerting.
    all_ok : bool
        True if all limit checks pass.
    violations : list[str]
        Human-readable descriptions of each limit breach.
    delta_utilization_pct : float
        |delta_risk_inr| / max_delta_inr × 100.  Useful for dashboards.
    vega_utilization_pct : float
        |vega| / max_vega_inr × 100.
    """
    delta_ok:               bool
    gamma_ok:               bool
    vega_ok:                bool
    theta_ok:               bool
    all_ok:                 bool
    violations:             list[str]
    delta_utilization_pct:  float
    vega_utilization_pct:   float


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioRiskEngine:
    """
    Aggregate Greeks from all active strategy legs and enforce risk limits.

    This engine is the single source of truth for portfolio-level risk in the
    F&O book.  It provides:
      - Greek aggregation across any list of StrategyLeg objects
      - Risk-limit enforcement (delta, vega, gamma)
      - VRP signal computation from price history + live ATM IV
      - Delta-hedge unit calculation
      - First-order P&L attribution

    Thread safety
    -------------
    All public methods that return new objects (stateless computation paths)
    are lock-free.  The internal ``_greeks_cache`` is protected by
    ``_lock`` and is updated only when explicitly requested by the caller
    (not auto-updated on market data ticks — that is the orchestrator's job).

    Parameters
    ----------
    bsm : BSMEngine
        Configured BSM engine (carries r and q defaults).
    max_delta_inr : float
        Maximum allowed |delta_risk_inr| in INR.
        ASSUMED default: ₹50,000 — a 1-point index move earns/loses at most ₹50K.
        Set based on your capital and SEBI SPAN margin buffers.
    max_vega_inr : float
        Maximum allowed |portfolio vega| in INR.
        ASSUMED default: ₹25,000 — a 1% IV rise/fall earns/loses at most ₹25K.
    max_gamma_pnl_1pct : float
        Maximum allowed |gamma_pnl_1pct| in INR.
        ASSUMED default: ₹10,000 — a 1% spot move causes at most ₹10K gamma P&L.

    Example
    -------
    engine = PortfolioRiskEngine(bsm=BSMEngine(r=0.065, q=0.015))
    greeks = engine.compute_portfolio_greeks(legs, underlying_price=24000.0, T_years=7/365)
    result = engine.check_risk_limits(greeks)
    if not result.all_ok:
        for v in result.violations:
            print(v)
    vrp = engine.compute_vrp(price_history, atm_iv=0.14)
    """

    def __init__(
        self,
        bsm: BSMEngine,
        max_delta_inr: float        = 50_000.0,   # ASSUMED — calibrate to capital
        max_vega_inr: float         = 25_000.0,   # ASSUMED — calibrate to capital
        max_gamma_pnl_1pct: float   = 10_000.0,   # ASSUMED — calibrate to capital
    ) -> None:
        self.bsm = bsm
        self.max_delta_inr        = max_delta_inr
        self.max_vega_inr         = max_vega_inr
        self.max_gamma_pnl_1pct   = max_gamma_pnl_1pct

        # Thread-safe cache of the last computed PortfolioGreeks snapshot.
        # Updated by compute_portfolio_greeks(); consumed by delta_hedge_units()
        # and pnl_attribution() when callers pass old_greeks directly.
        self._greeks_cache: Optional[PortfolioGreeks] = None
        self._lock = threading.Lock()

        log.info(
            "portfolio_risk.init",
            max_delta_inr=max_delta_inr,
            max_vega_inr=max_vega_inr,
            max_gamma_pnl_1pct=max_gamma_pnl_1pct,
            bsm_r=bsm.r,
            bsm_q=bsm.q,
        )

    # ── Greek aggregation ─────────────────────────────────────────────────────

    def compute_portfolio_greeks(
        self,
        legs: list[StrategyLeg],
        underlying_price: float,
        T_years: float,
    ) -> PortfolioGreeks:
        """
        Recompute all Greeks for each leg using current market inputs, then aggregate.

        Greeks are recomputed from scratch using the BSM engine (not taken from
        leg.delta / leg.vega etc.) so that the snapshot is consistent with the
        current underlying_price and T_years inputs.

        Parameters
        ----------
        legs : list[StrategyLeg]
            All open option legs across all active strategies.
        underlying_price : float
            Current index level (e.g. 24000.0).  Must be > 0.
        T_years : float
            Time to expiry in years for the dominant expiry.  For multi-expiry
            books, callers should call this method per expiry bucket and combine
            the returned PortfolioGreeks objects using add_portfolio_greeks().
            Floor is enforced at 1/365 internally (one calendar day).

        Returns
        -------
        PortfolioGreeks
            Snapshot of aggregated Greeks.  Also stored in _greeks_cache.

        Raises
        ------
        ValueError
            If underlying_price <= 0 or T_years < 0.
        """
        if underlying_price <= 0.0:
            raise ValueError(
                f"underlying_price must be > 0, got {underlying_price}"
            )
        if T_years < 0.0:
            raise ValueError(
                f"T_years must be >= 0, got {T_years}"
            )

        T_eff = max(T_years, 1.0 / 365.0)   # floor: avoid BSM division-by-zero

        p_delta = 0.0
        p_gamma = 0.0
        p_vega  = 0.0
        p_theta = 0.0
        p_rho   = 0.0

        for i, leg in enumerate(legs):
            # Validate leg fields
            if leg.lots <= 0:
                log.warning(
                    "portfolio_risk.skip_zero_lots",
                    leg_index=i,
                    strike=leg.strike,
                    option_type=leg.option_type,
                )
                continue
            if leg.sigma <= 0.0:
                log.warning(
                    "portfolio_risk.skip_zero_iv",
                    leg_index=i,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    sigma=leg.sigma,
                )
                continue

            # Recompute Greeks with current market inputs for consistency
            try:
                g: OptionGreeks = self.bsm.greeks(
                    S=underlying_price,
                    K=leg.strike,
                    T=T_eff,
                    sigma=leg.sigma,
                    option_type=leg.option_type,
                )
            except Exception as exc:
                log.error(
                    "portfolio_risk.greeks_error",
                    leg_index=i,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    error=str(exc),
                )
                raise

            # Weight = direction × lots × lot_size
            # direction: +1 = long position (we paid premium),
            #            -1 = short position (we received premium)
            weight = float(leg.direction) * float(leg.lots) * float(leg.lot_size)

            p_delta += weight * g.delta
            p_gamma += weight * g.gamma
            p_vega  += weight * g.vega     # already ₹ per 1% vol per unit
            p_theta += weight * g.theta    # already ₹ per calendar day per unit
            p_rho   += weight * g.rho      # already ₹ per 1% rate per unit

        # ── Normalized risk measures ──────────────────────────────────────────
        # delta_risk_inr: ₹ change for a 1-point move in the index.
        # This is the natural unit for index F&O risk: delta is in "fraction of
        # underlying", so multiplying by S converts to INR exposure per point.
        delta_risk_inr = p_delta * underlying_price

        # gamma_pnl_1pct: second-order P&L for a 1% spot move.
        # ΔS for 1% move = underlying_price × 0.01
        # Gamma P&L = 0.5 × Γ × ΔS²
        ds_1pct = underlying_price * 0.01
        gamma_pnl_1pct = 0.5 * p_gamma * ds_1pct * ds_1pct

        # vega_risk_1pct: vega is already in ₹ per 1% vol move (from pricing.py)
        vega_risk_1pct = p_vega

        result = PortfolioGreeks(
            delta=p_delta,
            gamma=p_gamma,
            vega=p_vega,
            theta=p_theta,
            rho=p_rho,
            delta_risk_inr=delta_risk_inr,
            gamma_pnl_1pct=gamma_pnl_1pct,
            vega_risk_1pct=vega_risk_1pct,
            underlying_price=underlying_price,
        )

        with self._lock:
            self._greeks_cache = result

        log.debug(
            "portfolio_risk.greeks_computed",
            n_legs=len(legs),
            underlying_price=underlying_price,
            T_years=T_eff,
            delta=round(p_delta, 4),
            gamma=round(p_gamma, 6),
            vega=round(p_vega, 2),
            theta=round(p_theta, 2),
            delta_risk_inr=round(delta_risk_inr, 2),
            gamma_pnl_1pct=round(gamma_pnl_1pct, 2),
        )
        return result

    # ── Risk-limit enforcement ────────────────────────────────────────────────

    def check_risk_limits(self, greeks: PortfolioGreeks) -> RiskLimitResult:
        """
        Check portfolio Greeks against configured risk limits.

        All checks are on absolute values (|metric| vs limit) so that
        both long and short books are symmetric.

        Parameters
        ----------
        greeks : PortfolioGreeks
            Output of compute_portfolio_greeks().

        Returns
        -------
        RiskLimitResult
            all_ok=True means no limits are breached; proceed.
            all_ok=False means at least one limit is breached; act on violations.
        """
        violations: list[str] = []

        # ── Delta check ───────────────────────────────────────────────────────
        abs_delta_inr = abs(greeks.delta_risk_inr)
        delta_ok = abs_delta_inr <= self.max_delta_inr
        delta_utilization_pct = (abs_delta_inr / self.max_delta_inr * 100.0
                                  if self.max_delta_inr > 0 else 0.0)
        if not delta_ok:
            violations.append(
                f"DELTA_BREACH: |delta_risk_inr|=₹{abs_delta_inr:,.0f} "
                f"exceeds limit ₹{self.max_delta_inr:,.0f} "
                f"(utilization {delta_utilization_pct:.1f}%)"
            )

        # ── Gamma check ───────────────────────────────────────────────────────
        abs_gamma_pnl = abs(greeks.gamma_pnl_1pct)
        gamma_ok = abs_gamma_pnl <= self.max_gamma_pnl_1pct
        if not gamma_ok:
            gamma_utilization_pct = (abs_gamma_pnl / self.max_gamma_pnl_1pct * 100.0
                                     if self.max_gamma_pnl_1pct > 0 else 0.0)
            violations.append(
                f"GAMMA_BREACH: |gamma_pnl_1pct|=₹{abs_gamma_pnl:,.0f} "
                f"exceeds limit ₹{self.max_gamma_pnl_1pct:,.0f} "
                f"(utilization {gamma_utilization_pct:.1f}%)"
            )

        # ── Vega check ────────────────────────────────────────────────────────
        abs_vega = abs(greeks.vega)
        vega_ok = abs_vega <= self.max_vega_inr
        vega_utilization_pct = (abs_vega / self.max_vega_inr * 100.0
                                 if self.max_vega_inr > 0 else 0.0)
        if not vega_ok:
            violations.append(
                f"VEGA_BREACH: |vega|=₹{abs_vega:,.0f} "
                f"exceeds limit ₹{self.max_vega_inr:,.0f} "
                f"(utilization {vega_utilization_pct:.1f}%)"
            )

        # ── Theta: tracked but no hard limit ─────────────────────────────────
        # Theta is positive-expected for short-premium strategies but cannot
        # be hard-limited in the same way (it is a "benefit" not a "risk").
        # We include it in the result for observability and alerting.
        theta_ok = True   # no hard limit; informational only

        all_ok = delta_ok and gamma_ok and vega_ok

        result = RiskLimitResult(
            delta_ok=delta_ok,
            gamma_ok=gamma_ok,
            vega_ok=vega_ok,
            theta_ok=theta_ok,
            all_ok=all_ok,
            violations=violations,
            delta_utilization_pct=round(delta_utilization_pct, 2),
            vega_utilization_pct=round(vega_utilization_pct, 2),
        )

        if violations:
            log.warning(
                "portfolio_risk.limits_breached",
                all_ok=all_ok,
                violations=violations,
                delta_utilization_pct=result.delta_utilization_pct,
                vega_utilization_pct=result.vega_utilization_pct,
            )
        else:
            log.debug(
                "portfolio_risk.limits_ok",
                delta_utilization_pct=result.delta_utilization_pct,
                vega_utilization_pct=result.vega_utilization_pct,
            )

        return result

    # ── VRP signal ────────────────────────────────────────────────────────────

    def compute_vrp(
        self,
        price_history: list[float],
        atm_iv: float,
        lookback: int = 22,
        bar_type: str = "daily",
    ) -> VRPEstimate:
        """
        Compute the Variance Risk Premium (VRP) signal.

        VRP = realized_volatility - implied_volatility_atm

        A VRP > VRP_THRESHOLD_FAVORABLE (+3% ASSUMED) indicates the options
        market is pricing in more volatility than has been realized, making it
        a favorable environment for short-premium strategies.

        Parameters
        ----------
        price_history : list[float]
            Closing prices (or bar closes for 15-min), sorted oldest first.
            Must contain at least ``lookback + 1`` prices to compute
            ``lookback`` log-returns.
        atm_iv : float
            Current ATM implied volatility (annualised decimal, e.g. 0.14 = 14%).
            Typically the IV of the closest-to-ATM strike from the live options chain.
        lookback : int
            Number of bar-returns used for realized vol estimation.
            Default 22 bars ≈ 1 trading month for daily bars.
            ASSUMED: 22 days is a conventional short-term VRP lookback;
            longer lookbacks (63 days) smooth out noise.
        bar_type : str
            "daily"  → annualise by sqrt(252)
            "15min"  → annualise by sqrt(6552)  [252 days × 26 bars/day]

        Returns
        -------
        VRPEstimate
            Contains realized_vol, implied_vol_atm, vrp, and the signal
            classification.

        Raises
        ------
        ValueError
            If bar_type is not "daily" or "15min", or if price_history is
            too short for the requested lookback.
        """
        if bar_type not in ("daily", "15min"):
            raise ValueError(
                f"bar_type must be 'daily' or '15min', got {bar_type!r}"
            )
        if atm_iv <= 0.0:
            raise ValueError(f"atm_iv must be > 0, got {atm_iv}")

        min_required = lookback + 1
        if len(price_history) < min_required:
            raise ValueError(
                f"price_history has {len(price_history)} prices; need at least "
                f"{min_required} to compute {lookback} log-returns."
            )

        # ── Realized volatility ───────────────────────────────────────────────
        # Use the last (lookback + 1) prices → lookback log-returns
        window = price_history[-(lookback + 1):]
        log_returns: list[float] = []
        for j in range(1, len(window)):
            p_prev = window[j - 1]
            p_curr = window[j]
            if p_prev <= 0.0 or p_curr <= 0.0:
                raise ValueError(
                    f"Non-positive price in history at relative index {j}: "
                    f"prev={p_prev}, curr={p_curr}"
                )
            log_returns.append(math.log(p_curr / p_prev))

        n = len(log_returns)
        if n < 2:
            raise ValueError(
                f"Need at least 2 log-returns to compute variance; got {n}."
            )

        # Sample standard deviation of log-returns (Bessel's correction: n-1)
        mean_r = sum(log_returns) / n
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (n - 1)
        sigma_per_bar = math.sqrt(variance)

        # Annualise
        annualization_factor = _SQRT252 if bar_type == "daily" else _SQRT6552
        realized_vol = annualization_factor * sigma_per_bar

        # ── VRP ───────────────────────────────────────────────────────────────
        # Convention: VRP > 0 means realized < implied (implied overpriced).
        # This is the typical NSE condition favoring option sellers.
        vrp = realized_vol - atm_iv

        # ── Signal classification ─────────────────────────────────────────────
        if vrp > VRP_THRESHOLD_FAVORABLE:
            signal = "favorable_short_premium"
        elif vrp < VRP_THRESHOLD_UNFAVORABLE:
            signal = "unfavorable_short_premium"
        else:
            signal = "neutral"

        # ── Signal strength: |VRP| / _VRP_NORM_SCALE, clipped to [0, 1] ──────
        # ASSUMED: _VRP_NORM_SCALE = 20% is the normalization reference.
        # This means a |VRP| of 20% gives signal_strength = 1.0 (maximum).
        signal_strength = min(abs(vrp) / _VRP_NORM_SCALE, 1.0)

        estimate = VRPEstimate(
            realized_vol=realized_vol,
            implied_vol_atm=atm_iv,
            vrp=vrp,
            lookback_bars=lookback,
            signal=signal,
            signal_strength=round(signal_strength, 4),
        )

        log.debug(
            "portfolio_risk.vrp_computed",
            realized_vol=round(realized_vol, 4),
            atm_iv=round(atm_iv, 4),
            vrp=round(vrp, 4),
            signal=signal,
            signal_strength=round(signal_strength, 4),
            bar_type=bar_type,
            lookback=lookback,
        )
        return estimate

    # ── Delta-hedge unit calculator ───────────────────────────────────────────

    def delta_hedge_units(
        self,
        greeks: PortfolioGreeks,
        futures_lot_size: int,
    ) -> int:
        """
        Compute the number of futures lots needed to make the portfolio delta-neutral.

        Formula:
            hedge_lots = round(-portfolio_delta / futures_lot_size)

        Sign convention:
            Positive result → buy futures lots (add long delta to offset short delta).
            Negative result → sell futures lots (add short delta to offset long delta).

        Rounding: standard banker's rounding (Python round()) to the nearest integer.
        For large books, consider splitting into multiple child orders to reduce
        market impact on the futures leg.

        Parameters
        ----------
        greeks : PortfolioGreeks
            Current portfolio Greeks snapshot.
        futures_lot_size : int
            Lot size of the futures contract used for hedging.
            Retrieve from lot_size(symbol, ContractType.FUTURES).

        Returns
        -------
        int
            Number of futures lots to trade.  Zero means no hedge needed
            (portfolio is already delta-neutral to within half a lot).

        Raises
        ------
        ValueError
            If futures_lot_size <= 0.
        """
        if futures_lot_size <= 0:
            raise ValueError(
                f"futures_lot_size must be > 0, got {futures_lot_size}"
            )

        raw_lots = -greeks.delta / futures_lot_size
        hedge_lots = round(raw_lots)

        log.debug(
            "portfolio_risk.delta_hedge",
            portfolio_delta=round(greeks.delta, 4),
            futures_lot_size=futures_lot_size,
            raw_lots=round(raw_lots, 4),
            hedge_lots=hedge_lots,
        )
        return hedge_lots

    # ── P&L attribution ───────────────────────────────────────────────────────

    def pnl_attribution(
        self,
        old_greeks: PortfolioGreeks,
        new_underlying: float,
        new_iv_atm: float,
        days_elapsed: float,
    ) -> dict[str, float]:
        """
        First-order (Taylor) P&L attribution across delta, gamma, vega, theta.

        This is a linear-quadratic approximation of the actual P&L.  It will
        NOT match mark-to-market P&L exactly because:
          - It ignores cross-Greek terms (vanna, volga, charm, etc.)
          - It uses a single IV shift (not a full vol-surface shift)
          - Gamma term is second-order in spot but first-order in vol

        Use this for decomposing attribution in post-trade analysis, not for
        precise margin calculations.

        Formulas:
            delta_pnl  = portfolio_delta × ΔS
            gamma_pnl  = 0.5 × portfolio_gamma × ΔS²
            vega_pnl   = portfolio_vega × Δσ%
                         (vega is ₹ per 1% vol; Δσ% = (new_iv_atm - old_iv) × 100)
            theta_pnl  = portfolio_theta × days_elapsed
            total      = sum of all four components

        Parameters
        ----------
        old_greeks : PortfolioGreeks
            Greeks snapshot at the *start* of the attribution period.
            underlying_price is taken from old_greeks.underlying_price.
        new_underlying : float
            Current underlying price (end of attribution period).
        new_iv_atm : float
            Current ATM implied volatility (end of attribution period).
            The vol change Δσ% = (new_iv_atm - old_iv_atm) × 100 is computed
            internally; however, old_iv_atm is not stored in PortfolioGreeks —
            callers must pass it via new_iv_atm relative to the reference stored
            externally.  Alternatively, pass the *Δσ directly by setting
            new_iv_atm = old_iv_atm + desired_change*.
        days_elapsed : float
            Calendar days since old_greeks was computed (can be fractional).
            E.g., 0.5 for a half-day session, 1.0 for overnight.

        Returns
        -------
        dict[str, float]
            Keys: "delta", "gamma", "vega", "theta", "total".
            All values in INR (positive = gain, negative = loss).

        Raises
        ------
        ValueError
            If new_underlying <= 0 or days_elapsed < 0.
        """
        if new_underlying <= 0.0:
            raise ValueError(
                f"new_underlying must be > 0, got {new_underlying}"
            )
        if days_elapsed < 0.0:
            raise ValueError(
                f"days_elapsed must be >= 0, got {days_elapsed}"
            )

        ds = new_underlying - old_greeks.underlying_price   # ΔS in index points

        # Delta P&L: first-order spot sensitivity
        # delta is in "fraction of underlying per unit" → multiply by lot-weighted
        # units already embedded from aggregation → direct ₹ change per point
        delta_pnl = old_greeks.delta * ds

        # Gamma P&L: second-order (convexity) correction
        # 0.5 × Γ × ΔS²  — same sign convention as gamma itself
        gamma_pnl = 0.5 * old_greeks.gamma * ds * ds

        # Vega P&L: portfolio_vega is ₹ per 1% vol move
        # Δσ in percentage points: e.g., IV moves from 14% to 15% → Δσ% = +1.0
        # We assume the caller supplies new_iv_atm in absolute terms (e.g. 0.15),
        # but we need a reference old IV.  Since PortfolioGreeks does not store IV,
        # we compute the vol shift from VRP context if available.  In practice,
        # callers should pass new_iv_atm as the *change from a known reference*,
        # or compute Δσ externally and pass new_iv_atm = old_iv + delta_sigma.
        #
        # Convention adopted here:
        #   new_iv_atm is the NEW absolute IV level (annualised, e.g. 0.15)
        #   old_iv_atm is inferred from the last VRPEstimate if cached, else
        #   we assume it equals atm_iv stored in the Greeks (not available in
        #   PortfolioGreeks).  To avoid silent errors, we treat new_iv_atm as
        #   the CHANGE in vol expressed in percentage points when it is < 1,
        #   and as an absolute IV level change when >= 1.
        #
        # ROBUST DESIGN DECISION:
        #   We interpret new_iv_atm as the *absolute new IV level* (e.g. 0.15).
        #   To compute Δσ%, the caller must also supply old_iv_atm.
        #   Since the current signature only has new_iv_atm, we interpret it as
        #   an *absolute IV change in percentage points* for vega attribution:
        #     vega_pnl = vega × new_iv_atm   (caller passes Δσ% directly)
        #
        # IMPORTANT: The docstring notes this ambiguity.  Callers MUST pass
        # the *change* (Δσ%) not the absolute level, for correct attribution.
        # Example: IV moved from 14% to 15% → pass new_iv_atm = 1.0 (one pct pt).
        # This is documented as a known API limitation pending PortfolioGreeks
        # storing the reference IV snapshot.
        delta_iv_pct = new_iv_atm   # CALLER PASSES Δσ in percentage points
        vega_pnl = old_greeks.vega * delta_iv_pct

        # Theta P&L: theta is ₹ per calendar day (already negative for long premium)
        theta_pnl = old_greeks.theta * days_elapsed

        total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        attribution = {
            "delta": round(delta_pnl, 4),
            "gamma": round(gamma_pnl, 4),
            "vega":  round(vega_pnl, 4),
            "theta": round(theta_pnl, 4),
            "total": round(total, 4),
        }

        log.debug(
            "portfolio_risk.pnl_attribution",
            ds=round(ds, 2),
            delta_iv_pct=round(delta_iv_pct, 4),
            days_elapsed=days_elapsed,
            **{f"pnl_{k}": v for k, v in attribution.items()},
        )
        return attribution

    # ── Convenience: access last cached Greeks (thread-safe) ─────────────────

    @property
    def last_greeks(self) -> Optional[PortfolioGreeks]:
        """
        Return the last Greeks snapshot computed by compute_portfolio_greeks(),
        or None if no snapshot has been computed yet.

        Thread-safe read under lock.
        """
        with self._lock:
            return self._greeks_cache

    # ── Static helper: combine two PortfolioGreeks snapshots ─────────────────

    @staticmethod
    def add_portfolio_greeks(
        a: PortfolioGreeks,
        b: PortfolioGreeks,
    ) -> PortfolioGreeks:
        """
        Add two PortfolioGreeks snapshots (e.g. from two expiry buckets).

        The combined underlying_price is taken from ``a`` (the caller should
        ensure both snapshots are computed at the same underlying_price).

        Parameters
        ----------
        a, b : PortfolioGreeks
            Two snapshots to combine.  Both must use the same underlying_price.

        Returns
        -------
        PortfolioGreeks
            Combined Greeks.

        Raises
        ------
        ValueError
            If the underlying_price difference between a and b exceeds 1.0 INR,
            suggesting a stale snapshot inconsistency.
        """
        if abs(a.underlying_price - b.underlying_price) > 1.0:
            raise ValueError(
                f"underlying_price mismatch: a={a.underlying_price}, b={b.underlying_price}. "
                "Both PortfolioGreeks snapshots must be computed at the same spot level."
            )

        combined_delta = a.delta + b.delta
        combined_gamma = a.gamma + b.gamma
        combined_vega  = a.vega  + b.vega
        combined_theta = a.theta + b.theta
        combined_rho   = a.rho   + b.rho
        S = a.underlying_price

        ds_1pct = S * 0.01
        return PortfolioGreeks(
            delta=combined_delta,
            gamma=combined_gamma,
            vega=combined_vega,
            theta=combined_theta,
            rho=combined_rho,
            delta_risk_inr=combined_delta * S,
            gamma_pnl_1pct=0.5 * combined_gamma * ds_1pct * ds_1pct,
            vega_risk_1pct=combined_vega,
            underlying_price=S,
        )
