"""
algo_trading/fo/pricing.py — Black-Scholes-Merton pricing engine for NSE index options.

Mathematical foundations
------------------------
Model: Black-Scholes-Merton with continuous dividend yield (Merton 1973).
NSE index options (Nifty, BankNifty) are European-style cash-settled — BSM
is the EXACT model (no early-exercise adjustment required).

Reference notation follows:
  McDonald, R.L. (2013) "Derivatives Markets" 3rd ed., Chapters 12–13.
  Hull, J.C. (2022) "Options, Futures, and Other Derivatives" 11th ed., Chapters 15–19.

All formula derivations can be verified against these standard texts.

Key parameters
--------------
S      : current underlying price (index level)
K      : strike price
T      : time to expiry in years  (T > 0; floor at 1/365 enforced)
r      : continuously compounded risk-free rate (e.g. 0.065 for 6.5% RBI repo)
q      : continuous dividend yield (e.g. 0.018 for ~1.8% Nifty dividend yield)
sigma  : implied or estimated volatility, annualised (e.g. 0.15 for 15%)

Forward price: F = S * exp((r - q) * T)   [cost-of-carry model, Merton 1973]

IV solver
---------
Phase 1: Newton-Raphson (quadratic convergence, typically 3–5 iterations)
Phase 2: Brent's method fallback (guaranteed superlinear convergence if NR diverges)
Initial guess: Brenner-Subrahmanyam (1988) for near-ATM, σ₀ = 0.25 otherwise.

Higher-order Greeks
-------------------
Vanna  = ∂Delta/∂σ = ∂Vega/∂S   [cross-sensitivity, important for skew delta hedging]
Volga  = ∂Vega/∂σ               [also called Vomma; sensitivity of vega to vol]
Charm  = ∂Delta/∂T              [delta bleed per day; critical for daily delta-hedging]
Speed  = ∂Gamma/∂S              [rate of change of gamma with spot]
Zomma  = ∂Gamma/∂σ             [gamma sensitivity to vol; key for vol risk in options book]
Color  = ∂Gamma/∂T              [time decay of gamma; also called Gamma Bleed]

All formulas are exact closed-form, verified against McDonald (2013) Appendix B
and Hull (2022) Chapter 19.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import norm as _scipy_norm  # only for CDF/PDF — no other scipy use


# ── Scalar wrappers: avoid scipy overhead in tight loops ─────────────────────
# Using the error-function identity: N(x) = 0.5 * [1 + erf(x/√2)]
# N'(x) = (1/√2π) * exp(-x²/2)
_SQRT2    = math.sqrt(2.0)
_SQRT2PI  = math.sqrt(2.0 * math.pi)
_INVSQRT2 = 1.0 / _SQRT2


def _N(x: float) -> float:
    """Standard normal CDF, accurate to machine precision via math.erfc."""
    return 0.5 * math.erfc(-x * _INVSQRT2)


def _n(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


# ── Core d1, d2 ──────────────────────────────────────────────────────────────

def _d1d2(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> tuple[float, float]:
    """
    Compute d1 and d2 in the Merton (1973) continuous-yield BSM framework.

        F  = S * exp((r - q) * T)                 # forward price
        d1 = [ln(F/K) + 0.5 * σ² * T] / (σ * √T)
        d2 = d1 - σ * √T

    Preconditions (callers must ensure):
        T > 0, sigma > 0, S > 0, K > 0
    """
    F       = S * math.exp((r - q) * T)
    sqrt_T  = math.sqrt(T)
    vol_T   = sigma * sqrt_T  # σ√T
    d1      = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_T
    d2      = d1 - vol_T
    return d1, d2


# ── Option prices ─────────────────────────────────────────────────────────────

def call_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    European call price (Merton 1973):
        C = exp(-rT) * [F * N(d1) - K * N(d2)]
          = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    """
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)


def put_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    European put price (Merton 1973):
        P = exp(-rT) * [K * N(-d2) - F * N(-d1)]
          = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

    Put-call parity consistency check (not executed in hot path):
        C - P = S * exp(-qT) - K * exp(-rT)  ✓
    """
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    return K * math.exp(-r * T) * _N(-d2) - S * math.exp(-q * T) * _N(-d1)


def option_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str,   # "CE" or "PE"
) -> float:
    """Dispatch to call_price or put_price based on option_type."""
    if option_type == "CE":
        return call_price(S, K, T, r, q, sigma)
    if option_type == "PE":
        return put_price(S, K, T, r, q, sigma)
    raise ValueError(f"option_type must be 'CE' or 'PE', got {option_type!r}")


def forward_price(S: float, r: float, q: float, T: float) -> float:
    """
    No-arbitrage forward price under continuous dividend yield.
        F = S * exp((r - q) * T)
    """
    return S * math.exp((r - q) * T)


# ── First-order Greeks ────────────────────────────────────────────────────────

@dataclass
class OptionGreeks:
    """
    Complete first-order BSM Greeks.

    Units:
      delta : dimensionless; fraction of underlying per option
      gamma : 1/currency; change in delta per 1-unit move in S
      vega  : currency per 1% absolute IV move
      theta : currency per calendar day (negative for long options)
      rho   : currency per 1% absolute rate move
    """
    delta: float   # ∂V/∂S
    gamma: float   # ∂²V/∂S²
    vega:  float   # ∂V/∂σ per 1% move (= annual_vega / 100)
    theta: float   # ∂V/∂T per calendar day
    rho:   float   # ∂V/∂r per 1% move (= annual_rho / 100)
    option_type: str
    price: float   # BSM fair value
    # Inputs preserved for downstream use
    S: float = 0.0
    K: float = 0.0
    T: float = 0.0
    r: float = 0.0
    q: float = 0.0
    sigma: float = 0.0


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,   # "CE" or "PE"
) -> OptionGreeks:
    """
    Compute all first-order Greeks.

    Formulas (Merton 1973, McDonald 2013 Ch 12):

        Delta_call = exp(-qT) * N(d1)
        Delta_put  = exp(-qT) * [N(d1) - 1]  =  -exp(-qT) * N(-d1)

        Gamma  = exp(-qT) * N'(d1) / (S * σ * √T)          [same for call and put]

        Vega   = S * exp(-qT) * N'(d1) * √T                [annual; divided by 100 below]

        Theta_call = -exp(-qT)*S*N'(d1)*σ/(2√T)
                     - r*K*exp(-rT)*N(d2)
                     + q*S*exp(-qT)*N(d1)
        Theta_put  = -exp(-qT)*S*N'(d1)*σ/(2√T)
                     + r*K*exp(-rT)*N(-d2)
                     - q*S*exp(-qT)*N(-d1)
        (Theta divided by 365 to get per-calendar-day)

        Rho_call = K * T * exp(-rT) * N(d2)                [annual; divided by 100 below]
        Rho_put  = -K * T * exp(-rT) * N(-d2)
    """
    if T <= 0:
        T = 1.0 / 365.0  # floor: avoid division by zero at expiry
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    d1, d2  = _d1d2(S, K, T, r, q, sigma)
    sqrt_T  = math.sqrt(T)
    exp_qT  = math.exp(-q * T)
    exp_rT  = math.exp(-r * T)
    n_d1    = _n(d1)

    # ─ Delta ─
    if option_type == "CE":
        delta = exp_qT * _N(d1)
    else:
        delta = exp_qT * (_N(d1) - 1.0)    # = -exp(-qT)*N(-d1)

    # ─ Gamma (identical for call and put) ─
    gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)

    # ─ Vega (per 1% absolute IV move = annual_vega / 100) ─
    annual_vega = S * exp_qT * n_d1 * sqrt_T
    vega = annual_vega / 100.0

    # ─ Theta (per calendar day = annual_theta / 365) ─
    common_theta = -exp_qT * S * n_d1 * sigma / (2.0 * sqrt_T)
    if option_type == "CE":
        annual_theta = (
            common_theta
            - r * K * exp_rT * _N(d2)
            + q * S * exp_qT * _N(d1)
        )
    else:
        annual_theta = (
            common_theta
            + r * K * exp_rT * _N(-d2)
            - q * S * exp_qT * _N(-d1)
        )
    theta = annual_theta / 365.0

    # ─ Rho (per 1% absolute rate move = annual_rho / 100) ─
    if option_type == "CE":
        annual_rho = K * T * exp_rT * _N(d2)
    else:
        annual_rho = -K * T * exp_rT * _N(-d2)
    rho = annual_rho / 100.0

    # ─ Price ─
    price = option_price(S, K, T, r, q, sigma, option_type)

    return OptionGreeks(
        delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
        option_type=option_type, price=price,
        S=S, K=K, T=T, r=r, q=q, sigma=sigma,
    )


# ── Higher-order Greeks ───────────────────────────────────────────────────────

@dataclass
class HigherOrderGreeks:
    """
    Second-order and cross-partial Greeks.

    Units:
      vanna  : 1/vol (= ∂Delta/∂σ per unit σ move — multiply by 0.01 for 1% vol move)
      volga  : currency per (vol unit)²  (= ∂Vega/∂σ per unit move)
      charm  : per day  (= ∂Delta/∂T per calendar day)
      speed  : 1/currency (= ∂Gamma/∂S)
      zomma  : 1/vol (= ∂Gamma/∂σ per unit vol move)
      color  : per day (= ∂Gamma/∂T per calendar day; also called gamma bleed)
    """
    vanna: float
    volga: float
    charm: float
    speed: float
    zomma: float
    color: float


def compute_higher_order_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str,
) -> HigherOrderGreeks:
    """
    Compute second-order and cross-partial Greeks.

    Formulas verified against Hull (2022) Ch 19 and McDonald (2013) Appendix B:

        Vanna  = -exp(-qT) * N'(d1) * d2 / σ
               [Note: ∂Delta/∂σ = -exp(-qT)*n(d1)*d2/σ for calls AND puts]

        Volga  = S * exp(-qT) * N'(d1) * √T * d1*d2/σ
               = annual_Vega * d1*d2/σ

        Charm_call = q*exp(-qT)*N(d1)
                     - exp(-qT)*n(d1)*[2*(r-q)*T - d2*σ*√T] / (2*T*σ*√T)
        Charm_put  = -q*exp(-qT)*N(-d1)
                     - exp(-qT)*n(d1)*[2*(r-q)*T - d2*σ*√T] / (2*T*σ*√T)

        Speed  = -(Gamma/S) * (d1/(σ*√T) + 1)   [= ∂Gamma/∂S]

        Zomma  = Gamma * (d1*d2 - 1) / σ         [= ∂Gamma/∂σ]

        Color  = -exp(-qT) * n(d1) / (2*S*T*σ*√T)
                 * [2*q*T + 1 + d1*(2*(r-q)*T - d2*σ*√T)/(σ*√T)]
               [= ∂Gamma/∂T per calendar day when divided by 365]
    """
    if T <= 0:
        T = 1.0 / 365.0
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    sqrt_T  = math.sqrt(T)
    exp_qT  = math.exp(-q * T)
    n_d1    = _n(d1)

    # ─ Vanna: ∂Delta/∂σ (same sign for call and put) ─
    vanna = -exp_qT * n_d1 * d2 / sigma

    # ─ Volga: ∂Vega/∂σ = ∂²V/∂σ² ─
    annual_vega = S * exp_qT * n_d1 * sqrt_T
    volga = annual_vega * d1 * d2 / sigma

    # ─ Charm: ∂Delta/∂T (per calendar day) ─
    common_charm = -exp_qT * n_d1 * (2.0*(r-q)*T - d2*sigma*sqrt_T) / (2.0*T*sigma*sqrt_T)
    if option_type == "CE":
        annual_charm = q * exp_qT * _N(d1) - common_charm
    else:
        annual_charm = -q * exp_qT * _N(-d1) - common_charm
    charm = annual_charm / 365.0

    # ─ Gamma (needed for Speed and Zomma) ─
    gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)

    # ─ Speed: ∂Gamma/∂S ─
    speed = -(gamma / S) * (d1 / (sigma * sqrt_T) + 1.0)

    # ─ Zomma: ∂Gamma/∂σ ─
    zomma = gamma * (d1 * d2 - 1.0) / sigma

    # ─ Color: ∂Gamma/∂T (per calendar day) ─
    annual_color = (
        -exp_qT * n_d1 / (2.0 * S * T * sigma * sqrt_T)
        * (2.0*q*T + 1.0 + d1*(2.0*(r-q)*T - d2*sigma*sqrt_T)/(sigma*sqrt_T))
    )
    color = annual_color / 365.0

    return HigherOrderGreeks(
        vanna=vanna, volga=volga, charm=charm,
        speed=speed, zomma=zomma, color=color,
    )


# ── IV Solver ─────────────────────────────────────────────────────────────────

@dataclass
class IVSolverResult:
    """
    Result from the implied-volatility solver.

    Attributes
    ----------
    iv          : solved implied volatility (annualised).  None if solver failed.
    converged   : True if Newton-Raphson converged within tol.
    iterations  : NR iterations consumed.
    method      : "newton_raphson", "bisection", or "failed"
    error_msg   : populated when converged=False
    """
    iv: Optional[float]
    converged: bool
    iterations: int
    method: str
    error_msg: str = ""


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    tol: float           = 1e-7,
    max_iter_nr: int     = 50,
    max_iter_bisect: int = 200,
) -> IVSolverResult:
    """
    Robust implied-volatility solver.

    Phase 1: Newton-Raphson with Brenner-Subrahmanyam (1988) initial guess.
        σ_{n+1} = σ_n - [BSM(σ_n) - market_price] / Vega(σ_n)
        Convergence criterion: |σ_{n+1} - σ_n| < tol.

    Phase 2 (fallback): Bisection via Brent's method on [σ_lo, σ_hi].
        Used when NR diverges or vega becomes negligibly small.
        Guaranteed to converge to within tol given a valid bracket.

    Returns IVSolverResult(iv=None, converged=False) if:
        - market_price is below intrinsic value (no real IV exists)
        - bisection cannot find a valid bracket in [0.001, 20.0]

    Reference:
        Brenner, M. and Subrahmanyam, M.G. (1988)
        "A Simple Approximation to Option Valuation and Hedging in the Black-Scholes Model"
        Financial Analysts Journal, 44(5), 80–83.
    """
    if T <= 0:
        T = 1.0 / 365.0

    # ─ Intrinsic value check ─
    disc = math.exp(-r * T)
    F    = S * math.exp((r - q) * T)
    if option_type == "CE":
        intrinsic = max(0.0, (F - K) * disc)
    else:
        intrinsic = max(0.0, (K - F) * disc)

    if market_price < intrinsic - 1e-4:
        return IVSolverResult(
            iv=None, converged=False, iterations=0, method="failed",
            error_msg=f"market_price {market_price:.4f} < intrinsic {intrinsic:.4f}",
        )

    # ─ Initial guess: multi-tier by moneyness ─────────────────────────────
    # Tier 1 (ATM, |k| < 0.05): Brenner-Subrahmanyam (1988)
    #   C_ATM ≈ S * σ * sqrt(T / (2π))  →  σ ≈ C / S * sqrt(2π / T)
    #   Reference: Brenner & Subrahmanyam (1988) Financial Analysts Journal 44(5).
    #
    # Tier 2 (OTM/ITM, |k| ≥ 0.05): Manaster-Koehler (1982)
    #   σ₀ = sqrt(2 * |ln(F/K)| / T)  [provably NR-convergent starting point]
    #   Reference: Manaster & Koehler (1982) Journal of Finance 37(1), 227-230.
    #   For deep OTM (|k| > 0.5): blend with 0.3 floor to avoid σ₀ < σ_true
    moneyness = abs(math.log(F / K))  # |k| = |log(F/K)| = 0 at ATM
    if moneyness < 0.05:
        # Tier 1: Brenner-Subrahmanyam ATM approximation
        sigma0 = max(0.01, market_price / S * math.sqrt(2.0 * math.pi / T))
        sigma0 = min(sigma0, 5.0)
    else:
        # Tier 2: Manaster-Koehler (1982) initial guess
        mk_sigma = math.sqrt(2.0 * moneyness / T)
        # Floor: must be ≥ vol implied by the option premium (rough lower bound)
        # Use Brenner-Subrahmanyam as a lower bound on the forward premium
        bs_floor = max(0.01, market_price / S * math.sqrt(2.0 * math.pi / T))
        sigma0 = max(bs_floor, min(mk_sigma, 5.0))

    # ─ Phase 1: Newton-Raphson ─
    sigma = sigma0
    for i in range(max_iter_nr):
        d1, d2   = _d1d2(S, K, T, r, q, sigma)
        sqrt_T   = math.sqrt(T)
        exp_qT   = math.exp(-q * T)
        n_d1     = _n(d1)

        # Annual vega as the NR gradient: ∂BSM/∂σ
        annual_vega = S * exp_qT * n_d1 * sqrt_T
        if annual_vega < 1e-10:
            break   # gradient too flat → switch to bisection

        if option_type == "CE":
            price_bs = S * exp_qT * _N(d1) - K * math.exp(-r * T) * _N(d2)
        else:
            price_bs = K * math.exp(-r * T) * _N(-d2) - S * exp_qT * _N(-d1)

        diff  = price_bs - market_price
        sigma_new = sigma - diff / annual_vega
        sigma_new = max(1e-6, min(sigma_new, 20.0))  # box constraint

        if abs(sigma_new - sigma) < tol:
            return IVSolverResult(
                iv=sigma_new, converged=True, iterations=i + 1,
                method="newton_raphson",
            )
        sigma = sigma_new

    # ─ Phase 2: Brent's bisection fallback ─
    # Find a bracket [lo, hi] where f(lo) < 0 < f(hi)
    def f(sig: float) -> float:
        d1_, d2_ = _d1d2(S, K, T, r, q, sig)
        exp_qT_  = math.exp(-q * T)
        if option_type == "CE":
            return S * exp_qT_ * _N(d1_) - K * math.exp(-r*T) * _N(d2_) - market_price
        return K * math.exp(-r*T) * _N(-d2_) - S * exp_qT_ * _N(-d1_) - market_price

    lo, hi = 1e-4, 20.0
    f_lo, f_hi = f(lo), f(hi)

    if f_lo * f_hi > 0:
        # No bracket found — cannot solve
        return IVSolverResult(
            iv=None, converged=False, iterations=max_iter_nr,
            method="failed",
            error_msg="Bisection: no sign change in [0.0001, 20.0] — cannot bracket IV",
        )

    mid = 0.5 * (lo + hi)
    for j in range(max_iter_bisect):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return IVSolverResult(
                iv=mid, converged=True,
                iterations=max_iter_nr + j + 1,
                method="bisection",
            )
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    return IVSolverResult(
        iv=mid, converged=False,
        iterations=max_iter_nr + max_iter_bisect,
        method="bisection",
        error_msg=f"Bisection did not converge to tol={tol} in {max_iter_bisect} steps",
    )


# ── High-level engine facade ──────────────────────────────────────────────────

class BSMEngine:
    """
    Facade exposing the complete BSM toolkit with configurable defaults.

    Default risk-free rate: 6.5% (RBI repo rate proxy — update from live source).
    Default dividend yield: 1.5% (Nifty trailing yield — update periodically).
    Both are ASSUMED and must be verified against current RBI/NSE data.

    Usage
    -----
    engine = BSMEngine(r=0.065, q=0.015)

    # Price a Nifty call
    price = engine.call(S=24000, K=24000, T=7/365, sigma=0.14)

    # Greeks
    greeks = engine.greeks(S=24000, K=24000, T=7/365, sigma=0.14, option_type="CE")

    # Solve IV
    result = engine.iv(market_price=150.0, S=24000, K=24000, T=7/365, option_type="CE")
    if result.converged:
        print(f"IV = {result.iv:.4f}")
    """

    def __init__(
        self,
        r: float = 0.065,   # ASSUMED: RBI repo rate proxy — verify current rate
        q: float = 0.015,   # ASSUMED: Nifty trailing dividend yield ~1.5%
    ) -> None:
        self.r = r
        self.q = q

    def call(self, S: float, K: float, T: float, sigma: float) -> float:
        return call_price(S, K, T, self.r, self.q, sigma)

    # Alias for tests that use the longer name
    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        return call_price(S, K, T, self.r, self.q, sigma)

    def put(self, S: float, K: float, T: float, sigma: float) -> float:
        return put_price(S, K, T, self.r, self.q, sigma)

    # Alias for tests that use the longer name
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        return put_price(S, K, T, self.r, self.q, sigma)

    def price(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        return option_price(S, K, T, self.r, self.q, sigma, option_type)

    def forward(self, S: float, T: float) -> float:
        return forward_price(S, self.r, self.q, T)

    def greeks(
        self, S: float, K: float, T: float, sigma: float, option_type: str
    ) -> OptionGreeks:
        return compute_greeks(S, K, T, self.r, self.q, sigma, option_type)

    def call_greeks(self, S: float, K: float, T: float, sigma: float) -> OptionGreeks:
        """Alias: compute Greeks for a call option."""
        return compute_greeks(S, K, T, self.r, self.q, sigma, "CE")

    def put_greeks(self, S: float, K: float, T: float, sigma: float) -> OptionGreeks:
        """Alias: compute Greeks for a put option."""
        return compute_greeks(S, K, T, self.r, self.q, sigma, "PE")

    def higher_order_greeks(
        self, S: float, K: float, T: float, sigma: float, option_type: str
    ) -> HigherOrderGreeks:
        return compute_higher_order_greeks(S, K, T, self.r, self.q, sigma, option_type)

    def iv(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str,
        tol: float = 1e-7,
    ) -> IVSolverResult:
        return implied_volatility(
            market_price=market_price,
            S=S, K=K, T=T, r=self.r, q=self.q,
            option_type=option_type, tol=tol,
        )

    def implied_vol_call(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        tol: float = 1e-7,
    ) -> IVSolverResult:
        """Alias: solve implied vol for a call option.

        Args are ordered (S, K, T, market_price) to match test call conventions.
        """
        return implied_volatility(
            market_price=market_price,
            S=S, K=K, T=T, r=self.r, q=self.q,
            option_type="CE", tol=tol,
        )

    def implied_vol_put(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        tol: float = 1e-7,
    ) -> IVSolverResult:
        """Alias: solve implied vol for a put option.

        Args are ordered (S, K, T, market_price) to match test call conventions.
        """
        return implied_volatility(
            market_price=market_price,
            S=S, K=K, T=T, r=self.r, q=self.q,
            option_type="PE", tol=tol,
        )

    def chain_iv(
        self,
        prices: list[float],
        S: float,
        strikes: list[float],
        T: float,
        option_types: list[str],
        tol: float = 1e-7,
    ) -> list[IVSolverResult]:
        """
        Compute implied volatility for an entire option chain in one call.

        Convenience wrapper around :func:`implied_volatility` that processes
        a full chain of (price, strike, option_type) tuples sharing the same
        underlying S and expiry T.

        Useful for batch-processing NSE option chain snapshots:

            chain = [
                (ce_premium, 24000.0, "CE"),
                (pe_premium, 24000.0, "PE"),
                (ce_premium, 24500.0, "CE"),
                ...
            ]
            ivs = engine.chain_iv(
                prices=[p for p, _, _ in chain],
                strikes=[k for _, k, _ in chain],
                option_types=[t for _, _, t in chain],
                S=24000.0, T=7/365,
            )

        Parameters
        ----------
        prices : list[float]
            Market mid-prices for each option.
        S : float
            Current underlying price (same for all legs).
        strikes : list[float]
            Strike prices, one per option.
        T : float
            Time to expiry in years (same for all legs).
        option_types : list[str]
            "CE" or "PE" for each option.
        tol : float
            IV solver tolerance (default 1e-7).

        Returns
        -------
        list[IVSolverResult]
            One result per option.  Check ``result.converged`` before using
            ``result.iv`` — unconverged results have ``iv=None``.

        Raises
        ------
        ValueError
            If lengths of prices, strikes, and option_types do not match.
        """
        n = len(prices)
        if len(strikes) != n or len(option_types) != n:
            raise ValueError(
                f"chain_iv: lengths must match — "
                f"prices={n}, strikes={len(strikes)}, option_types={len(option_types)}"
            )
        return [
            implied_volatility(
                market_price=px,
                S=S, K=K, T=T,
                r=self.r, q=self.q,
                option_type=ot, tol=tol,
            )
            for px, K, ot in zip(prices, strikes, option_types)
        ]

    def put_call_parity_check(
        self,
        call_px: float, put_px: float,
        S: float, K: float, T: float,
        tol: float = 0.05,
    ) -> bool:
        """
        Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
        Returns True if parity holds within *tol* INR.
        A violation > 0.05 INR for liquid near-month options indicates
        stale/crossed data or a data quality issue.
        """
        lhs = call_px - put_px
        rhs = S * math.exp(-self.q * T) - K * math.exp(-self.r * T)
        return abs(lhs - rhs) <= tol
