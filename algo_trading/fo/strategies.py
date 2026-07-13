"""
algo_trading/fo/strategies.py — Multi-leg F&O strategy templates for NSE index options.

All payoff formulas are mathematically exact for European cash-settled options.
No formula is ASSUMED — all are derived from standard option theory.

Supported strategies:
  1. Short Strangle       — sell OTM CE + sell OTM PE (different strikes)
  2. Short Straddle       — sell ATM CE + sell ATM PE (same strike)
  3. Iron Condor          — buy PE (K1) + sell PE (K2) + sell CE (K3) + buy CE (K4)
  4. Bull Put Spread      — sell higher PE + buy lower PE (net credit)
  5. Bear Call Spread     — sell lower CE + buy higher CE (net credit)
  6. Long Straddle        — buy ATM CE + buy ATM PE (same strike)
  7. Covered Future + Short Call — long futures + short OTM CE (synthetic covered call)

Payoff formulas at expiry (per unit, before lot-size scaling):
  CE intrinsic = max(0, S_T − K)
  PE intrinsic = max(0, K − S_T)
  FUT payoff   = S_T − entry_price          (no optionality)

For a leg with direction d (+1 long / −1 short) and entry_premium p_entry:
  Option net cash flow = d × (intrinsic − p_entry)
  Future  net cash flow = d × (S_T − p_entry)   [p_entry is the futures entry price]

Total P&L = Σ over legs of: direction × (payoff − entry_premium) × lots × lot_size
         (futures: direction × (S_T − entry_premium) × lots × lot_size)

References:
  McDonald, R.L. (2013) "Derivatives Markets" 3rd ed., Chapters 3–4.
  Hull, J.C. (2022) "Options, Futures, and Other Derivatives" 11th ed., Chapters 10–12.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Optional

import structlog

from algo_trading.fo.contracts import (
    ContractType,
    get_contract,
)
from algo_trading.fo.cost_model import FOCostModel, FOInstrumentType  # noqa: F401
from algo_trading.fo.pricing import BSMEngine, OptionGreeks  # noqa: F401 — public API

log = structlog.get_logger(__name__)

# ── Session-time constants ────────────────────────────────────────────────────
# Source: algo_trading/fo/contracts.py (verified NSE circulars)
_FO_NO_NEW_TRADES = time(15, 30)   # conservative cutoff; F&O session ends 15:40 from Aug 3 2026

# ── Minimum net credit guard (ASSUMED) ───────────────────────────────────────
# Strategies that receive a net credit must receive at least this many INR per lot
# above total round-trip transaction costs to be considered viable.
# This is a judgement threshold — not an NSE rule.
# Tune based on your risk tolerance.
_MIN_CREDIT_ABOVE_COSTS_INR: float = 50.0  # ASSUMED: ₹50 above round-trip costs per lot

# ── Iron condor wing-width minimum (ASSUMED) ─────────────────────────────────
# Each wing (put spread or call spread) must be at least 1 strike interval wide.
# This is a sanity check — not a regulatory requirement.
_IC_MIN_WING_STRIKES: int = 1  # ASSUMED: minimum number of strike intervals per wing


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StrategyLeg:
    """
    A single option or futures leg in a multi-leg F&O strategy.

    Attributes
    ----------
    symbol        : NSE trading symbol (e.g. "NIFTY", "BANKNIFTY")
    option_type   : "CE" (call), "PE" (put), or "FUT" (futures)
    strike        : strike price in INR; 0.0 for futures legs
    expiry        : contract expiry date
    direction     : +1 = long (buy), −1 = short (sell)
    lots          : number of contracts
    lot_size      : units per contract (from NSE contract master)
    entry_premium : premium paid/received per unit at entry (INR)
                    For futures: this is the entry futures price per unit
    current_premium : current market premium per unit (mark-to-market)
    delta, gamma, vega, theta : BSM Greeks per unit (zero if not computed)
    """
    symbol: str
    option_type: str          # "CE", "PE", or "FUT"
    strike: float             # 0.0 for futures
    expiry: date
    direction: int            # +1 = long, −1 = short
    lots: int
    lot_size: int
    entry_premium: float      # INR per unit at entry
    current_premium: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    @property
    def notional_entry(self) -> float:
        """Total notional value at entry: entry_premium × lots × lot_size."""
        return self.entry_premium * self.lots * self.lot_size

    @property
    def side(self) -> str:
        """Return 'BUY' or 'SELL' from direction."""
        return "BUY" if self.direction == +1 else "SELL"


@dataclass
class StrategyResult:
    """
    Complete description of a multi-leg F&O strategy at entry time.

    Monetary units: INR.

    Attributes
    ----------
    name            : human-readable strategy name
    legs            : ordered list of StrategyLeg objects
    net_premium     : algebraic sum of (direction × entry_premium × lots × lot_size)
                      Positive = net credit received; negative = net debit paid.
    max_profit      : maximum possible P&L at expiry (INR); math.inf = unlimited
    max_loss        : maximum possible loss magnitude at expiry (INR, positive number);
                      math.inf = unlimited
    break_even_upper: upper break-even spot price at expiry (None if not applicable)
    break_even_lower: lower break-even spot price at expiry (None if not applicable)
    margin_required : approximate SEBI initial margin in INR
                      [ASSUMED approximation; actual SPAN margin will differ]
    current_pnl     : mark-to-market P&L using current_premium on each leg
    portfolio_delta : sum of (direction × delta × lots × lot_size) across legs
    portfolio_vega  : sum of (direction × vega  × lots × lot_size) across legs
    portfolio_theta : sum of (direction × theta × lots × lot_size) across legs
    """
    name: str
    legs: list[StrategyLeg]
    net_premium: float
    max_profit: float
    max_loss: float
    break_even_upper: Optional[float]
    break_even_lower: Optional[float]
    margin_required: float         # ASSUMED approximation — not official SPAN
    current_pnl: float
    portfolio_delta: float
    portfolio_vega: float
    portfolio_theta: float

    def summary(self) -> str:
        """Single-line summary string for logging."""
        mp = "∞" if math.isinf(self.max_profit) else f"₹{self.max_profit:,.0f}"
        ml = "∞" if math.isinf(self.max_loss)   else f"₹{self.max_loss:,.0f}"
        return (
            f"{self.name} | credit=₹{self.net_premium:,.2f}"
            f" | max_profit={mp} | max_loss={ml}"
            f" | BE=[{self.break_even_lower},{self.break_even_upper}]"
            f" | delta={self.portfolio_delta:.4f}"
        )


# ── Payoff functions ──────────────────────────────────────────────────────────

def _leg_payoff_per_unit(leg: StrategyLeg, S_T: float) -> float:
    """
    Compute intrinsic value per unit at expiry for a single leg.

    CE: max(0, S_T − K)
    PE: max(0, K   − S_T)
    FUT: S_T − entry_premium  (futures PnL; entry_premium holds entry price)

    Returns the per-unit payoff (before direction and lot-size).
    For options this is the intrinsic value at S_T.
    For futures this is the profit per unit.
    """
    if leg.option_type == "CE":
        return max(0.0, S_T - leg.strike)
    if leg.option_type == "PE":
        return max(0.0, leg.strike - S_T)
    if leg.option_type == "FUT":
        return S_T - leg.entry_premium
    raise ValueError(f"Unknown option_type {leg.option_type!r}; expected 'CE', 'PE', or 'FUT'")


def payoff_at_expiry(legs: list[StrategyLeg], S_T: float) -> float:
    """
    Compute total strategy P&L at expiry given settlement price S_T.

    For each option leg:
        P&L_leg = direction × (intrinsic(S_T) − entry_premium) × lots × lot_size

    For each futures leg:
        P&L_leg = direction × (S_T − entry_premium) × lots × lot_size
        [entry_premium stores the futures entry price]

    Summing across all legs gives total strategy P&L in INR.

    Note: transaction costs are NOT subtracted here — they are computed separately
    via FOCostModel and should be subtracted by the caller when computing net P&L.

    Parameters
    ----------
    legs : list[StrategyLeg]
        All legs of the strategy (in any order).
    S_T  : float
        Underlying settlement price at expiry (index level).

    Returns
    -------
    float
        Net strategy P&L in INR (positive = profit, negative = loss).
    """
    total = 0.0
    for leg in legs:
        if leg.option_type == "FUT":
            # Futures: profit = direction × (S_T − entry_price) × lots × lot_size
            pnl_leg = leg.direction * (S_T - leg.entry_premium) * leg.lots * leg.lot_size
        else:
            # Options: profit = direction × (intrinsic − premium_paid) × lots × lot_size
            intrinsic = _leg_payoff_per_unit(leg, S_T)
            pnl_leg = leg.direction * (intrinsic - leg.entry_premium) * leg.lots * leg.lot_size
        total += pnl_leg
    return total


def payoff_curve(
    legs: list[StrategyLeg],
    S_range: tuple[float, float],
    n_points: int = 200,
) -> tuple[list[float], list[float]]:
    """
    Compute a payoff diagram for the strategy across a range of settlement prices.

    Returns (spot_prices, pnl_values) — two equal-length lists.
    Suitable for plotting or analysis.

    Parameters
    ----------
    legs     : list[StrategyLeg]
    S_range  : (S_min, S_max) — range of spot prices to evaluate
    n_points : number of evenly-spaced points (default 200)

    Returns
    -------
    tuple[list[float], list[float]]
        spot_prices : list of n_points prices from S_min to S_max
        pnl_values  : corresponding total strategy P&L values
    """
    S_min, S_max = S_range
    if S_min >= S_max:
        raise ValueError(f"S_range must satisfy S_min < S_max; got {S_range}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")

    step = (S_max - S_min) / (n_points - 1)
    spot_prices: list[float] = [S_min + i * step for i in range(n_points)]
    pnl_values: list[float] = [payoff_at_expiry(legs, s) for s in spot_prices]
    return spot_prices, pnl_values


# ── Validation ────────────────────────────────────────────────────────────────

def validate_strategy_legs(legs: list[StrategyLeg]) -> list[str]:
    """
    Validate a set of strategy legs and return a list of error messages.

    An empty return list means all checks passed.

    Checks performed
    ----------------
    1. At least one leg present.
    2. All legs reference the same underlying symbol.
    3. All non-cross-expiry legs share the same expiry date.
       (Cross-expiry is allowed but emits a WARNING, not an error.)
    4. All option strikes are positive.
    5. Strike is a multiple of the contract's strike_interval.
    6. direction is +1 or −1.
    7. lots ≥ 1.
    8. lot_size matches the NSE contract master.
    9. entry_premium > 0 for options; entry_premium > 0 for futures (entry price).
    10. option_type is "CE", "PE", or "FUT".
    """
    errors: list[str] = []

    if not legs:
        errors.append("No legs provided — strategy must have at least one leg.")
        return errors

    # ── Check 2: same symbol ─────────────────────────────────────────────────
    symbols = {leg.symbol.upper() for leg in legs}
    if len(symbols) > 1:
        msg = f"Legs reference multiple symbols: {sorted(symbols)}. Mixed-symbol strategies not supported."
        errors.append(msg)
        log.warning("strategy.validate.mixed_symbols", symbols=sorted(symbols))

    # ── Check 3: expiry consistency ──────────────────────────────────────────
    expiries = {leg.expiry for leg in legs}
    if len(expiries) > 1:
        # Not an error — calendar spreads are legitimate — but warn clearly
        log.warning(
            "strategy.validate.cross_expiry",
            expiries=sorted(str(e) for e in expiries),
            note="Cross-expiry strategy detected — payoff_at_expiry assumes single expiry",
        )

    for leg in legs:
        # ── Check 10: option_type ────────────────────────────────────────────
        if leg.option_type not in ("CE", "PE", "FUT"):
            errors.append(
                f"Leg (strike={leg.strike}, type={leg.option_type!r}): "
                f"option_type must be 'CE', 'PE', or 'FUT'."
            )
            log.warning("strategy.validate.bad_option_type", option_type=leg.option_type)

        # ── Check 6: direction ───────────────────────────────────────────────
        if leg.direction not in (+1, -1):
            errors.append(
                f"Leg (strike={leg.strike}, type={leg.option_type!r}): "
                f"direction must be +1 or −1, got {leg.direction}."
            )

        # ── Check 7: lots ────────────────────────────────────────────────────
        if leg.lots < 1:
            errors.append(
                f"Leg (strike={leg.strike}, type={leg.option_type!r}): "
                f"lots must be ≥ 1, got {leg.lots}."
            )

        # ── Check 9: entry premium ────────────────────────────────────────────
        if leg.entry_premium <= 0.0:
            errors.append(
                f"Leg (strike={leg.strike}, type={leg.option_type!r}): "
                f"entry_premium must be > 0, got {leg.entry_premium}."
            )

        if leg.option_type in ("CE", "PE"):
            # ── Check 4: strike positive ─────────────────────────────────────
            if leg.strike <= 0.0:
                errors.append(
                    f"Leg (type={leg.option_type!r}): strike must be > 0, got {leg.strike}."
                )
                log.warning("strategy.validate.nonpositive_strike", strike=leg.strike)

            # ── Check 5: strike on grid ───────────────────────────────────────
            try:
                spec = get_contract(leg.symbol, ContractType.OPTIONS)
                interval = spec.strike_interval
                remainder = leg.strike % interval
                # Allow floating-point tolerance of 0.1 INR
                if remainder > 0.1 and (interval - remainder) > 0.1:
                    errors.append(
                        f"Leg (symbol={leg.symbol!r}, strike={leg.strike}): "
                        f"strike is not a multiple of strike_interval={interval}."
                    )
                    log.warning(
                        "strategy.validate.strike_off_grid",
                        symbol=leg.symbol, strike=leg.strike, interval=interval,
                    )

                # ── Check 8: lot_size matches registry ─────────────────────
                if leg.lot_size != spec.lot_size:
                    errors.append(
                        f"Leg (symbol={leg.symbol!r}, strike={leg.strike}): "
                        f"lot_size={leg.lot_size} does not match registry lot_size={spec.lot_size}."
                    )
                    log.warning(
                        "strategy.validate.lot_size_mismatch",
                        symbol=leg.symbol, provided=leg.lot_size, registry=spec.lot_size,
                    )
            except KeyError:
                errors.append(
                    f"Leg (symbol={leg.symbol!r}): symbol not found in contract registry."
                )

        elif leg.option_type == "FUT":
            # ── Check 8 for futures ────────────────────────────────────────
            try:
                spec = get_contract(leg.symbol, ContractType.FUTURES)
                if leg.lot_size != spec.lot_size:
                    errors.append(
                        f"Futures leg (symbol={leg.symbol!r}): "
                        f"lot_size={leg.lot_size} does not match registry lot_size={spec.lot_size}."
                    )
                    log.warning(
                        "strategy.validate.fut_lot_size_mismatch",
                        symbol=leg.symbol, provided=leg.lot_size, registry=spec.lot_size,
                    )
            except KeyError:
                errors.append(
                    f"Futures leg (symbol={leg.symbol!r}): symbol not found in contract registry."
                )

    return errors


def _check_entry_time() -> Optional[str]:
    """
    Return an error string if current IST time is past FO_NO_NEW_TRADES (15:30),
    or None if within session.

    ASSUMED: server clock is IST or correctly offset to IST via UTC+5:30.
    """
    # Convert current UTC time to IST (UTC+5:30 = UTC+19800s)
    now_utc = datetime.now(timezone.utc)
    ist_offset_seconds = 5 * 3600 + 30 * 60
    # Compute IST hour/minute directly from UTC
    total_minutes_utc = now_utc.hour * 60 + now_utc.minute
    total_minutes_ist = (total_minutes_utc + ist_offset_seconds // 60) % (24 * 60)
    ist_hour = total_minutes_ist // 60
    ist_minute = total_minutes_ist % 60
    now_ist_time = time(ist_hour, ist_minute)

    if now_ist_time >= _FO_NO_NEW_TRADES:
        return (
            f"Entry rejected: current IST time {now_ist_time.strftime('%H:%M')} "
            f"is at or after FO_NO_NEW_TRADES ({_FO_NO_NEW_TRADES.strftime('%H:%M')} IST)."
        )
    return None


def _compute_margin(
    symbol: str,
    lot_size: int,
    lots: int,
    underlying_price: float,
    margin_pct: float,
    strategy_name: str,
) -> float:
    """
    Approximate SEBI initial margin for a short-options strategy.

    Formula: notional × margin_pct × lots
      where notional = underlying_price × lot_size

    [ASSUMED] This uses the FuturesSpec.margin_pct from the contract registry as a
    proxy for the option margin.  Actual SPAN margin from NSE Clearing will differ
    — it accounts for position offsets, volatility scenarios, etc.
    This is a conservative upper-bound estimate only.
    """
    notional_per_lot = underlying_price * lot_size
    margin = notional_per_lot * margin_pct * lots
    log.debug(
        "strategy.margin_estimate",
        strategy=strategy_name, symbol=symbol, lots=lots,
        notional_per_lot=notional_per_lot, margin_pct=margin_pct,
        margin_estimated=margin,
        note="ASSUMED approximation — not official NSE SPAN margin",
    )
    return margin


def _net_greeks(legs: list[StrategyLeg]) -> tuple[float, float, float]:
    """
    Return (portfolio_delta, portfolio_vega, portfolio_theta) for the leg list.
    Each Greek is weighted by direction × lots × lot_size.
    """
    delta = sum(leg.direction * leg.delta * leg.lots * leg.lot_size for leg in legs)
    vega  = sum(leg.direction * leg.vega  * leg.lots * leg.lot_size for leg in legs)
    theta = sum(leg.direction * leg.theta * leg.lots * leg.lot_size for leg in legs)
    return delta, vega, theta


def _current_pnl(legs: list[StrategyLeg]) -> float:
    """
    Mark-to-market P&L: current_premium vs entry_premium.

    For each leg:
        MTM P&L = direction × (current_premium − entry_premium) × lots × lot_size
    """
    return sum(
        leg.direction * (leg.current_premium - leg.entry_premium) * leg.lots * leg.lot_size
        for leg in legs
    )


# ── StrategyBuilder ───────────────────────────────────────────────────────────

class StrategyBuilder:
    """
    Factory for creating validated F&O strategy templates.

    Each method validates the input, constructs StrategyLeg objects,
    computes payoff metrics, and returns a StrategyResult.

    Parameters
    ----------
    lot_size  : int
        DEPRECATED parameter — do not use.  Lot sizes are now always fetched
        from the contract registry via get_contract().  This parameter is accepted
        for backwards compatibility but ignored.
    cost_model : FOCostModel
        Used to compute total round-trip transaction costs for the minimum-credit check.
    symbol    : str
        NSE trading symbol (e.g. "NIFTY", "BANKNIFTY").  Defaults to "NIFTY".
    """

    def __init__(
        self,
        cost_model: FOCostModel,
        lot_size: int = 0,       # ignored — always read from registry
        symbol: str = "NIFTY",
    ) -> None:
        self._cost_model = cost_model
        self._symbol = symbol.upper()
        # Fetch verified lot size from registry — never use the constructor parameter
        opt_spec = get_contract(self._symbol, ContractType.OPTIONS)
        fut_spec = get_contract(self._symbol, ContractType.FUTURES)
        self._lot_size: int = opt_spec.lot_size
        self._margin_pct: float = fut_spec.margin_pct
        self._strike_interval: float = opt_spec.strike_interval
        log.info(
            "strategy_builder.init",
            symbol=self._symbol,
            lot_size=self._lot_size,
            margin_pct=self._margin_pct,
            strike_interval=self._strike_interval,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_credit_vs_costs(
        self,
        net_credit_inr: float,
        lots: int,
        strategy_name: str,
        leg_premiums: list[tuple[str, float]],  # list of (side, premium)
    ) -> Optional[str]:
        """
        Return an error string if the net credit per lot is insufficient to
        cover round-trip transaction costs plus the minimum threshold.

        Returns None if the credit is adequate.
        """
        # Estimate round-trip costs as sum of entry + exit for all legs
        total_rt_costs = 0.0
        for side, prem in leg_premiums:
            entry_c = self._cost_model.compute_option(
                side=side, lots=lots, lot_size=self._lot_size,
                ltp=prem, underlying=0.0,  # underlying not needed for premium cost
            )
            exit_side = "BUY" if side == "SELL" else "SELL"
            exit_c = self._cost_model.compute_option(
                side=exit_side, lots=lots, lot_size=self._lot_size,
                ltp=prem, underlying=0.0,
            )
            total_rt_costs += float(entry_c.total) + float(exit_c.total)

        threshold = total_rt_costs + _MIN_CREDIT_ABOVE_COSTS_INR * lots
        if net_credit_inr < threshold:
            msg = (
                f"{strategy_name}: net credit ₹{net_credit_inr:.2f} < "
                f"threshold ₹{threshold:.2f} (costs ₹{total_rt_costs:.2f} + "
                f"min_buffer ₹{_MIN_CREDIT_ABOVE_COSTS_INR * lots:.2f}). "
                "Strategy may not be viable after costs."
            )
            log.warning(
                "strategy.validate.insufficient_credit",
                strategy=strategy_name,
                net_credit=net_credit_inr,
                threshold=threshold,
                costs=total_rt_costs,
            )
            return msg
        return None

    def _build_result(
        self,
        name: str,
        legs: list[StrategyLeg],
        max_profit: float,
        max_loss: float,
        break_even_upper: Optional[float],
        break_even_lower: Optional[float],
        S: float,
    ) -> StrategyResult:
        """Assemble a StrategyResult from computed components."""
        # Net premium = algebraic sum: positive → net credit received.
        # Negated so that short legs (direction=-1) contribute positively:
        # selling a leg receives premium → credit to the trader (positive).
        net_premium = -sum(
            leg.direction * leg.entry_premium * leg.lots * leg.lot_size
            for leg in legs
        )
        margin = _compute_margin(
            symbol=self._symbol,
            lot_size=self._lot_size,
            lots=max(leg.lots for leg in legs),
            underlying_price=S,
            margin_pct=self._margin_pct,
            strategy_name=name,
        )
        delta, vega, theta = _net_greeks(legs)
        cur_pnl = _current_pnl(legs)
        result = StrategyResult(
            name=name,
            legs=legs,
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=break_even_upper,
            break_even_lower=break_even_lower,
            margin_required=margin,
            current_pnl=cur_pnl,
            portfolio_delta=delta,
            portfolio_vega=vega,
            portfolio_theta=theta,
        )
        log.info("strategy.built", **{
            "strategy": name,
            "net_premium": round(net_premium, 2),
            "max_profit": "inf" if math.isinf(max_profit) else round(max_profit, 2),
            "max_loss": "inf" if math.isinf(max_loss) else round(max_loss, 2),
            "be_lower": break_even_lower,
            "be_upper": break_even_upper,
            "delta": round(delta, 4),
        })
        return result

    # ── Strategy 1: Short Strangle ────────────────────────────────────────────

    def short_strangle(
        self,
        S: float,
        call_strike: float,
        put_strike: float,
        call_premium: float,
        put_premium: float,
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Short Strangle: sell OTM CE at call_strike + sell OTM PE at put_strike.

        Entry condition: put_strike < S < call_strike (both legs OTM at entry).

        Payoff at expiry (per unit):
          P&L = (call_premium + put_premium)
                − max(0, S_T − call_strike)
                − max(0, put_strike − S_T)

        Max profit  : call_premium + put_premium (both expire worthless)
                      Achieved when put_strike ≤ S_T ≤ call_strike.
        Max loss    : unlimited (theoretically) in both directions.
        Break-evens:
          Upper BE  : call_strike + (call_premium + put_premium) / lot_size
          Lower BE  : put_strike  − (call_premium + put_premium) / lot_size

        Parameters
        ----------
        S            : current spot (index level)
        call_strike  : OTM call strike (must be > S)
        put_strike   : OTM put strike  (must be < S)
        call_premium : CE premium received per unit (INR)
        put_premium  : PE premium received per unit (INR)
        expiry       : contract expiry date
        lots         : number of lots to trade (same for both legs)
        """
        errors: list[str] = []

        # Validation: call_strike > S > put_strike
        if call_strike <= S:
            msg = (
                f"Short Strangle: call_strike={call_strike} must be > spot={S} (OTM call). "
                "Use a higher strike."
            )
            errors.append(msg)
            log.warning("strategy.short_strangle.call_not_otm", call_strike=call_strike, S=S)
        if put_strike >= S:
            msg = (
                f"Short Strangle: put_strike={put_strike} must be < spot={S} (OTM put). "
                "Use a lower strike."
            )
            errors.append(msg)
            log.warning("strategy.short_strangle.put_not_otm", put_strike=put_strike, S=S)
        if call_strike <= put_strike:
            errors.append(
                f"Short Strangle: call_strike={call_strike} must be > put_strike={put_strike}."
            )

        time_err = _check_entry_time()
        if time_err:
            errors.append(time_err)
            log.warning("strategy.short_strangle.time_check_failed", error=time_err)

        if errors:
            raise ValueError("Short Strangle validation failed:\n" + "\n".join(errors))

        # Legs: short CE + short PE
        call_leg = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=call_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_premium,
        )
        put_leg = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=put_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=put_premium,
        )
        legs = [call_leg, put_leg]

        # Validate all legs via universal validator
        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Short Strangle leg validation failed:\n" + "\n".join(leg_errors))

        # Payoff metrics
        # net_credit per unit = call_premium + put_premium
        # max profit = net_credit per lot × lots × lot_size
        net_credit_per_unit = call_premium + put_premium
        net_credit_inr = net_credit_per_unit * lots * self._lot_size

        max_profit = net_credit_inr
        max_loss   = math.inf  # unlimited in both directions

        # Break-evens are per-unit prices, not INR totals
        # Premium_received per lot / lot_size = net_credit_per_unit (already per unit)
        be_upper = call_strike + net_credit_per_unit
        be_lower = put_strike  - net_credit_per_unit

        # Credit adequacy check (warning only, not blocking)
        credit_warn = self._check_credit_vs_costs(
            net_credit_inr=net_credit_inr, lots=lots,
            strategy_name="Short Strangle",
            leg_premiums=[("SELL", call_premium), ("SELL", put_premium)],
        )
        if credit_warn:
            log.warning("strategy.short_strangle.credit_check", warning=credit_warn)

        return self._build_result(
            name="Short Strangle",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 2: Short Straddle ────────────────────────────────────────────

    def short_straddle(
        self,
        S: float,
        strike: float,
        call_premium: float,
        put_premium: float,
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Short Straddle: sell ATM CE + sell ATM PE at the same strike K.

        Payoff at expiry (per unit):
          P&L = (call_premium + put_premium)
                − max(0, S_T − K)
                − max(0, K  − S_T)

        Max profit  : call_premium + put_premium (achieved only when S_T = K exactly)
        Max loss    : unlimited in both directions
        Break-evens:
          Upper BE  : K + (call_premium + put_premium) / lot_size = K + net_premium_per_unit
          Lower BE  : K − (call_premium + put_premium) / lot_size

        Note: Unlike a strangle, max profit for a straddle is achieved at exactly
        one point S_T = K, making it a higher-conviction, higher-risk position.
        """
        time_err = _check_entry_time()
        if time_err:
            log.warning("strategy.short_straddle.time_check_failed", error=time_err)
            raise ValueError(f"Short Straddle validation failed: {time_err}")

        # ATM guideline: strike should be close to spot (within 1 strike interval)
        dist = abs(strike - S)
        if dist > self._strike_interval:
            log.warning(
                "strategy.short_straddle.not_atm",
                strike=strike, S=S, distance=dist,
                strike_interval=self._strike_interval,
                note="ASSUMED: ATM tolerance is 1 strike interval",
            )

        call_leg = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_premium,
        )
        put_leg = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=put_premium,
        )
        legs = [call_leg, put_leg]

        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Short Straddle leg validation failed:\n" + "\n".join(leg_errors))

        net_credit_per_unit = call_premium + put_premium
        net_credit_inr = net_credit_per_unit * lots * self._lot_size

        max_profit = net_credit_inr
        max_loss   = math.inf

        be_upper = strike + net_credit_per_unit
        be_lower = strike - net_credit_per_unit

        credit_warn = self._check_credit_vs_costs(
            net_credit_inr=net_credit_inr, lots=lots,
            strategy_name="Short Straddle",
            leg_premiums=[("SELL", call_premium), ("SELL", put_premium)],
        )
        if credit_warn:
            log.warning("strategy.short_straddle.credit_check", warning=credit_warn)

        return self._build_result(
            name="Short Straddle",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 3: Iron Condor ───────────────────────────────────────────────

    def iron_condor(
        self,
        S: float,
        put_buy_strike: float,   # K1 (lowest)
        put_sell_strike: float,  # K2
        call_sell_strike: float, # K3
        call_buy_strike: float,  # call buy strike K4 (highest)
        put_buy_prem: float,     # P1: cost of long put at K1
        put_sell_prem: float,    # P2: premium received for short put at K2
        call_sell_prem: float,   # C3: premium received for short call at K3
        call_buy_prem: float,    # C4: cost of long call at K4
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Iron Condor: buy PE(K1) + sell PE(K2) + sell CE(K3) + buy CE(K4).

        Constraint: K1 < K2 ≤ S ≤ K3 < K4

        Net credit = (P2 + C3) − (P1 + C4)
        (P2 and C3 are received; P1 and C4 are paid)

        Payoff at expiry (per unit):
          S_T ≤ K1:          −(K2−K1) + net_credit   (max loss on put side)
          K1 < S_T ≤ K2:     −(K2−S_T) + net_credit
          K2 < S_T ≤ K3:     net_credit               (max profit zone)
          K3 < S_T ≤ K4:     −(S_T−K3) + net_credit
          S_T > K4:          −(K4−K3) + net_credit   (max loss on call side)

        Max profit : net_credit (in INR = net_credit_per_unit × lots × lot_size)
        Max loss   : max((K2−K1), (K4−K3)) − net_credit per unit, × lots × lot_size
                     For symmetric wings (K2−K1 == K4−K3), both sides are equal.
        Break-evens:
          Lower BE  : K2 − net_credit_per_unit
          Upper BE  : K3 + net_credit_per_unit
        """
        errors: list[str] = []

        # Validate strike ordering: K1 < K2 ≤ S ≤ K3 < K4
        if not (put_buy_strike < put_sell_strike):
            errors.append(
                f"Iron Condor: K1 (put_buy={put_buy_strike}) must be < K2 (put_sell={put_sell_strike})."
            )
        if not (put_sell_strike <= S):
            errors.append(
                f"Iron Condor: K2 (put_sell={put_sell_strike}) must be ≤ spot={S}. "
                "Spot should be in or above the put spread at entry."
            )
        if not (S <= call_sell_strike):
            errors.append(
                f"Iron Condor: K3 (call_sell={call_sell_strike}) must be ≥ spot={S}. "
                "Spot should be in or below the call spread at entry."
            )
        if not (call_sell_strike < call_buy_strike):
            errors.append(
                f"Iron Condor: K3 (call_sell={call_sell_strike}) must be < K4 (call_buy={call_buy_strike})."
            )

        # Minimum wing widths
        put_wing = put_sell_strike - put_buy_strike
        call_wing = call_buy_strike - call_sell_strike
        if put_wing < self._strike_interval * _IC_MIN_WING_STRIKES:
            errors.append(
                f"Iron Condor: put wing width={put_wing} is less than "
                f"minimum {self._strike_interval * _IC_MIN_WING_STRIKES} "
                f"(strike_interval={self._strike_interval} × {_IC_MIN_WING_STRIKES}). "
                "[ASSUMED minimum]"
            )
        if call_wing < self._strike_interval * _IC_MIN_WING_STRIKES:
            errors.append(
                f"Iron Condor: call wing width={call_wing} is less than "
                f"minimum {self._strike_interval * _IC_MIN_WING_STRIKES}. [ASSUMED minimum]"
            )

        time_err = _check_entry_time()
        if time_err:
            errors.append(time_err)
            log.warning("strategy.iron_condor.time_check_failed", error=time_err)

        if errors:
            raise ValueError("Iron Condor validation failed:\n" + "\n".join(errors))

        # Build legs (long = +1, short = -1)
        # K1: buy put
        leg_put_buy = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=put_buy_strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=put_buy_prem,
        )
        # K2: sell put
        leg_put_sell = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=put_sell_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=put_sell_prem,
        )
        # K3: sell call
        leg_call_sell = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=call_sell_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_sell_prem,
        )
        # K4: buy call
        leg_call_buy = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=call_buy_strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_buy_prem,
        )
        legs = [leg_put_buy, leg_put_sell, leg_call_sell, leg_call_buy]

        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Iron Condor leg validation failed:\n" + "\n".join(leg_errors))

        # Net credit per unit = (P2 + C3) − (P1 + C4)
        net_credit_per_unit = (put_sell_prem + call_sell_prem) - (put_buy_prem + call_buy_prem)
        net_credit_inr = net_credit_per_unit * lots * self._lot_size

        if net_credit_per_unit <= 0:
            log.warning(
                "strategy.iron_condor.no_credit",
                net_credit_per_unit=net_credit_per_unit,
                note="Iron condor should normally generate a net credit",
            )

        max_profit = net_credit_inr

        # Max loss: larger of put-wing loss or call-wing loss (minus net credit)
        # Max loss per unit = max(K2−K1, K4−K3) − net_credit_per_unit
        # (For symmetric condor both wings are equal)
        max_wing_per_unit = max(put_wing, call_wing)
        max_loss_per_unit = max_wing_per_unit - net_credit_per_unit
        max_loss = max(0.0, max_loss_per_unit) * lots * self._lot_size

        # Break-evens
        be_lower = put_sell_strike  - net_credit_per_unit
        be_upper = call_sell_strike + net_credit_per_unit

        credit_warn = self._check_credit_vs_costs(
            net_credit_inr=net_credit_inr, lots=lots,
            strategy_name="Iron Condor",
            leg_premiums=[
                ("BUY",  put_buy_prem),
                ("SELL", put_sell_prem),
                ("SELL", call_sell_prem),
                ("BUY",  call_buy_prem),
            ],
        )
        if credit_warn:
            log.warning("strategy.iron_condor.credit_check", warning=credit_warn)

        return self._build_result(
            name="Iron Condor",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 4: Bull Put Spread ───────────────────────────────────────────

    def bull_put_spread(
        self,
        S: float,
        sell_strike: float,   # K2: higher-strike put, SOLD (credit leg)
        buy_strike: float,    # K1: lower-strike put,  BOUGHT (hedge leg)
        sell_premium: float,  # P2: premium received
        buy_premium: float,   # P1: premium paid
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Bull Put Spread: sell higher-strike PE (K2) + buy lower-strike PE (K1).

        Constraint: K1 < K2 (sell_strike > buy_strike).

        Net credit = P2 − P1

        Payoff at expiry (per unit):
          S_T ≥ K2:           net_credit              (max profit — both PEs expire OTM)
          K1 < S_T < K2:      net_credit − (K2 − S_T)
          S_T ≤ K1:           net_credit − (K2 − K1)  (max loss)

        Max profit : net_credit × lots × lot_size
        Max loss   : (K2 − K1 − net_credit) × lots × lot_size
        Break-even : K2 − net_credit_per_unit

        Bullish strategy: profitable if spot stays above K2 at expiry.
        """
        errors: list[str] = []

        if sell_strike <= buy_strike:
            errors.append(
                f"Bull Put Spread: sell_strike={sell_strike} must be > buy_strike={buy_strike}."
            )
            log.warning(
                "strategy.bull_put_spread.invalid_strikes",
                sell_strike=sell_strike, buy_strike=buy_strike,
            )

        time_err = _check_entry_time()
        if time_err:
            errors.append(time_err)
            log.warning("strategy.bull_put_spread.time_check_failed", error=time_err)

        if errors:
            raise ValueError("Bull Put Spread validation failed:\n" + "\n".join(errors))

        # Short higher PE (K2) + Long lower PE (K1)
        leg_sell = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=sell_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=sell_premium,
        )
        leg_buy = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=buy_strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=buy_premium,
        )
        legs = [leg_sell, leg_buy]

        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Bull Put Spread leg validation failed:\n" + "\n".join(leg_errors))

        net_credit_per_unit = sell_premium - buy_premium
        net_credit_inr = net_credit_per_unit * lots * self._lot_size

        spread_per_unit = sell_strike - buy_strike
        max_profit_per_unit = net_credit_per_unit
        max_loss_per_unit   = spread_per_unit - net_credit_per_unit

        max_profit = max_profit_per_unit * lots * self._lot_size
        max_loss   = max(0.0, max_loss_per_unit) * lots * self._lot_size

        be_upper = None                                 # no upper break-even (profit above K2)
        be_lower = sell_strike - net_credit_per_unit   # the single break-even

        if net_credit_per_unit <= 0:
            log.warning(
                "strategy.bull_put_spread.no_credit",
                net_credit_per_unit=net_credit_per_unit,
                note="Bull put spread should normally generate a net credit",
            )

        credit_warn = self._check_credit_vs_costs(
            net_credit_inr=net_credit_inr, lots=lots,
            strategy_name="Bull Put Spread",
            leg_premiums=[("SELL", sell_premium), ("BUY", buy_premium)],
        )
        if credit_warn:
            log.warning("strategy.bull_put_spread.credit_check", warning=credit_warn)

        return self._build_result(
            name="Bull Put Spread",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 5: Bear Call Spread ──────────────────────────────────────────

    def bear_call_spread(
        self,
        S: float,
        sell_strike: float,   # K1: lower-strike call, SOLD (credit leg)
        buy_strike: float,    # K2: higher-strike call, BOUGHT (hedge leg)
        sell_premium: float,  # C1: premium received
        buy_premium: float,   # C2: premium paid
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Bear Call Spread: sell lower-strike CE (K1) + buy higher-strike CE (K2).

        Constraint: K1 < K2 (sell_strike < buy_strike).

        Net credit = C1 − C2

        Payoff at expiry (per unit):
          S_T ≤ K1:           net_credit              (max profit — both CEs expire OTM)
          K1 < S_T < K2:      net_credit − (S_T − K1)
          S_T ≥ K2:           net_credit − (K2 − K1)  (max loss)

        Max profit : net_credit × lots × lot_size
        Max loss   : (K2 − K1 − net_credit) × lots × lot_size
        Break-even : K1 + net_credit_per_unit

        Bearish strategy: profitable if spot stays below K1 at expiry.
        """
        errors: list[str] = []

        if sell_strike >= buy_strike:
            errors.append(
                f"Bear Call Spread: sell_strike={sell_strike} must be < buy_strike={buy_strike}."
            )
            log.warning(
                "strategy.bear_call_spread.invalid_strikes",
                sell_strike=sell_strike, buy_strike=buy_strike,
            )

        time_err = _check_entry_time()
        if time_err:
            errors.append(time_err)
            log.warning("strategy.bear_call_spread.time_check_failed", error=time_err)

        if errors:
            raise ValueError("Bear Call Spread validation failed:\n" + "\n".join(errors))

        # Short lower CE (K1) + Long higher CE (K2)
        leg_sell = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=sell_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=sell_premium,
        )
        leg_buy = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=buy_strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=buy_premium,
        )
        legs = [leg_sell, leg_buy]

        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Bear Call Spread leg validation failed:\n" + "\n".join(leg_errors))

        net_credit_per_unit = sell_premium - buy_premium
        net_credit_inr = net_credit_per_unit * lots * self._lot_size

        spread_per_unit = buy_strike - sell_strike
        max_profit_per_unit = net_credit_per_unit
        max_loss_per_unit   = spread_per_unit - net_credit_per_unit

        max_profit = max_profit_per_unit * lots * self._lot_size
        max_loss   = max(0.0, max_loss_per_unit) * lots * self._lot_size

        be_upper = sell_strike + net_credit_per_unit  # single break-even
        be_lower = None                               # no lower break-even (profit below K1)

        if net_credit_per_unit <= 0:
            log.warning(
                "strategy.bear_call_spread.no_credit",
                net_credit_per_unit=net_credit_per_unit,
                note="Bear call spread should normally generate a net credit",
            )

        credit_warn = self._check_credit_vs_costs(
            net_credit_inr=net_credit_inr, lots=lots,
            strategy_name="Bear Call Spread",
            leg_premiums=[("SELL", sell_premium), ("BUY", buy_premium)],
        )
        if credit_warn:
            log.warning("strategy.bear_call_spread.credit_check", warning=credit_warn)

        return self._build_result(
            name="Bear Call Spread",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 6: Long Straddle ─────────────────────────────────────────────

    def long_straddle(
        self,
        S: float,
        strike: float,
        call_premium: float,
        put_premium: float,
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Long Straddle: buy ATM CE + buy ATM PE at the same strike K.

        This is the reverse of a short straddle — a net debit strategy.
        Payoff at expiry (per unit):
          P&L = max(0, S_T − K) + max(0, K − S_T) − (call_premium + put_premium)
              = |S_T − K| − net_debit_per_unit

        Max profit  : unlimited (S_T rises above upper BE) or large
                      (index cannot go below zero, so max put profit = K − net_debit)
        Max loss    : call_premium + put_premium (if S_T = K at expiry — both expire worthless)
        Break-evens:
          Upper BE  : K + net_debit_per_unit
          Lower BE  : K − net_debit_per_unit

        Volatility strategy: profits from large moves in either direction.
        net_premium is NEGATIVE (debit paid).
        """
        time_err = _check_entry_time()
        if time_err:
            log.warning("strategy.long_straddle.time_check_failed", error=time_err)
            raise ValueError(f"Long Straddle validation failed: {time_err}")

        dist = abs(strike - S)
        if dist > self._strike_interval:
            log.warning(
                "strategy.long_straddle.not_atm",
                strike=strike, S=S, distance=dist,
                note="ASSUMED: ATM tolerance is 1 strike interval",
            )

        call_leg = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_premium,
        )
        put_leg = StrategyLeg(
            symbol=self._symbol, option_type="PE", strike=strike,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=self._lot_size, entry_premium=put_premium,
        )
        legs = [call_leg, put_leg]

        leg_errors = validate_strategy_legs(legs)
        if leg_errors:
            raise ValueError("Long Straddle leg validation failed:\n" + "\n".join(leg_errors))

        net_debit_per_unit = call_premium + put_premium
        net_debit_inr = net_debit_per_unit * lots * self._lot_size

        # Max loss: the debit paid (if both legs expire worthless)
        max_loss   = net_debit_inr

        # Max profit: unlimited on the upside; on downside bounded by K (index ≥ 0)
        # Upper side: unlimited; lower side: K - net_debit (when S_T = 0, unlikely)
        # Convention: report max_profit = math.inf (as spot has no ceiling)
        max_profit = math.inf

        # Break-evens (same formula as short straddle, opposite profit/loss zones)
        be_upper = strike + net_debit_per_unit
        be_lower = strike - net_debit_per_unit

        log.info(
            "strategy.long_straddle.debit",
            net_debit_inr=round(net_debit_inr, 2),
            be_upper=round(be_upper, 2),
            be_lower=round(be_lower, 2),
        )

        return self._build_result(
            name="Long Straddle",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )

    # ── Strategy 7: Covered Future + Short Call ───────────────────────────────

    def covered_future_short_call(
        self,
        S: float,
        futures_entry: float,    # price at which futures were bought
        call_strike: float,      # OTM CE strike (above current futures price)
        call_premium: float,     # CE premium received per unit
        expiry: date,
        lots: int = 1,
    ) -> StrategyResult:
        """
        Synthetic Covered Call using Futures: long futures + short OTM CE.

        This replicates the payoff of a covered call (long stock + short call)
        using NSE index futures instead of the physical index.

        Legs:
          1. Long futures at futures_entry  (direction=+1)
          2. Short CE at call_strike         (direction=−1)

        Payoff at expiry (per unit):
          Futures P&L   = S_T − futures_entry
          Short CE P&L  = call_premium − max(0, S_T − call_strike)
          Total         = (S_T − futures_entry) + call_premium − max(0, S_T − call_strike)

        Zones:
          S_T ≤ call_strike:  (S_T − futures_entry) + call_premium  (capped by CE)
          S_T > call_strike:  (call_strike − futures_entry) + call_premium  (profit capped)

        Max profit : (call_strike − futures_entry + call_premium) × lots × lot_size
                     [profit is capped at call_strike — upside sold away]
        Max loss   : unlimited downside from the long futures leg
                     [if index crashes, there is no put protection]
        Break-even : futures_entry − call_premium  (per unit)

        Constraint : call_strike > futures_entry (CE should be OTM at entry)

        Note: Futures lot sizes are used for both legs. futures_entry is stored as
        entry_premium in the FUT leg (per the StrategyLeg convention).
        """
        errors: list[str] = []

        if call_strike <= futures_entry:
            errors.append(
                f"Covered Future + Short Call: call_strike={call_strike} must be > "
                f"futures_entry={futures_entry} (call must be OTM at entry)."
            )
            log.warning(
                "strategy.covered_future_short_call.call_not_otm",
                call_strike=call_strike, futures_entry=futures_entry,
            )

        time_err = _check_entry_time()
        if time_err:
            errors.append(time_err)
            log.warning("strategy.covered_future_short_call.time_check_failed", error=time_err)

        if errors:
            raise ValueError(
                "Covered Future + Short Call validation failed:\n" + "\n".join(errors)
            )

        # Lot sizes: use futures lot_size for the FUT leg
        fut_spec = get_contract(self._symbol, ContractType.FUTURES)
        fut_lot_size = fut_spec.lot_size

        # FUT leg: long futures (entry price stored as entry_premium per convention)
        leg_fut = StrategyLeg(
            symbol=self._symbol, option_type="FUT", strike=0.0,
            expiry=expiry, direction=+1, lots=lots,
            lot_size=fut_lot_size, entry_premium=futures_entry,
        )
        # Short CE leg
        leg_call = StrategyLeg(
            symbol=self._symbol, option_type="CE", strike=call_strike,
            expiry=expiry, direction=-1, lots=lots,
            lot_size=self._lot_size, entry_premium=call_premium,
        )
        legs = [leg_fut, leg_call]

        # Validate only the option leg via standard validator
        # (futures leg has strike=0 so we validate it separately)
        opt_errors = validate_strategy_legs([leg_call])
        if opt_errors:
            raise ValueError(
                "Covered Future + Short Call (CE leg) validation failed:\n"
                + "\n".join(opt_errors)
            )

        # Payoff metrics
        # Max profit (capped by short call): per unit
        max_profit_per_unit = (call_strike - futures_entry) + call_premium
        max_profit = max(0.0, max_profit_per_unit) * lots * self._lot_size

        # Max loss: unlimited (long futures with no put hedge)
        max_loss = math.inf

        # Break-even: futures_entry - call_premium (where total = 0)
        be_lower = futures_entry - call_premium
        be_upper = None  # no upper break-even; profit is capped above call_strike

        log.info(
            "strategy.covered_future_short_call",
            futures_entry=futures_entry, call_strike=call_strike,
            call_premium=call_premium,
            max_profit_per_lot=round(max_profit_per_unit * self._lot_size, 2),
            be_lower=round(be_lower, 2),
        )

        return self._build_result(
            name="Covered Future + Short Call",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_upper=be_upper,
            break_even_lower=be_lower,
            S=S,
        )
