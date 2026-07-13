"""
algo_trading/fo/vol_surface.py — SVI (Stochastic Volatility Inspired) parametric
implied-volatility surface for NSE index options.

Mathematical Model
------------------
SVI raw parametrization (Gatheral 2004):

    w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))

where:
    k      = log(K/F) = log-moneyness  (NOTE: k > 0 for OTM calls, k < 0 for OTM puts)
    F      = S * exp((r - q) * T)  = forward price
    w      = σ_impl² * T           = total implied variance
    T      = time to expiry in years
    σ_impl = sqrt(w / T)            = implied volatility

    Parameters:
        a      : vertical shift; controls overall level of total variance
        b      : slope; controls the steepness/bandwidth of the smile (≥ 0)
        rho    : correlation skew parameter (-1 < rho < 1); negative = left skew
        m      : log-moneyness at the minimum of the smile (horizontal shift)
        sigma  : width parameter, the "smile" curvature (σ_svi > 0)

No-Static-Arbitrage Parameter Constraints (Gatheral 2004, Lee 2004)
--------------------------------------------------------------------
(C1) b ≥ 0
(C2) |ρ| < 1
(C3) σ_svi > 0
(C4) a + b * σ_svi * sqrt(1 - ρ²) ≥ 0         (non-negative total variance at k=-∞)
(C5) 0 ≤ a + b * σ_svi * sqrt(1 - ρ²) ≤ 4/T   (T-bill bound; upper wing bounded)

Butterfly Arbitrage-Free Condition (Lee 2004)
---------------------------------------------
The smile is free of butterfly arbitrage iff g(k) ≥ 0 for all k, where:

    g(k) = (1 - k*w'/(2w))² - (w'/2)² * (1/w + 1/4) + w''/2

For SVI:
    w'(k)  = b * (ρ + (k-m) / sqrt((k-m)² + σ²))
    w''(k) = b * σ² / ((k-m)² + σ²)^(3/2)

Calendar Spread Arbitrage-Free Condition
-----------------------------------------
For two maturities T1 < T2:
    w_T1(k) ≤ w_T2(k)  for all k  (total variance non-decreasing in T)

SVI Calibration
---------------
Minimize weighted sum of squared errors between SVI and observed market total variances:
    loss = Σᵢ wᵢ * (w_market_i - w_svi_i)²
where wᵢ = vega²_i (vega-squared weighting).

Optimizer: scipy.optimize.minimize with L-BFGS-B and 5 random restarts.
Arbitrage violations penalized with 1000 * violation² added to objective.

References
----------
Gatheral, J. (2004) "A Parsimonious Arbitrage-Free Implied Volatility Parametrization
    with Application to the Valuation of Volatility Derivatives."
    Presentation at Global Derivatives & Risk Management, Madrid.

Lee, R. (2004) "The Moment Formula for Implied Volatility at Extreme Strikes."
    Mathematical Finance 14(3), 469–480.

Parkinson, M. (1980) "The Extreme Value Method for Estimating the Variance of the Rate
    of Return." Journal of Business 53(1), 61–65.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from scipy.optimize import minimize

from algo_trading.fo.pricing import BSMEngine, implied_volatility

log = structlog.get_logger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

_IV_FLOOR: float = 0.001          # minimum implied vol to avoid sqrt(0) issues
_IV_CAP:   float = 5.0            # maximum credible annualised IV (500%)
_W_FLOOR:  float = _IV_FLOOR ** 2 # minimum total variance (for g(k) denominator)
_ARBIT_PENALTY: float = 1_000.0   # penalty multiplier for arbitrage violation in calibration

# SVI parameter bounds for L-BFGS-B
_BOUNDS = [
    (-1.0,  5.0),     # a:     can be negative (shifts level)
    ( 0.0,  2.0),     # b:     non-negative slope
    (-0.999, 0.999),  # rho:   strict interior of (-1, 1)
    (-2.0,  2.0),     # m:     ATM shift in log-moneyness units
    ( 0.001, 2.0),    # sigma: strictly positive width
]

_N_RESTARTS: int = 5  # ASSUMED: 5 restarts balances speed vs. global optimum coverage


# ── Pure functions ────────────────────────────────────────────────────────────

def svi_total_variance(
    k: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float:
    """
    SVI raw parametrization: total implied variance w(k).

    Formula (Gatheral 2004):
        w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))

    Parameters
    ----------
    k     : log-moneyness = log(K/F) where F is the forward price
    a     : level shift
    b     : ATM slope (≥ 0)
    rho   : spot–vol correlation (-1 < rho < 1)
    m     : horizontal shift (log-moneyness of smile minimum when rho=0)
    sigma : width of the smile (σ_svi > 0)

    Returns
    -------
    w : total implied variance = σ_impl² * T  (≥ 0 when parameters satisfy no-arb)
    """
    km = k - m
    return a + b * (rho * km + math.sqrt(km * km + sigma * sigma))


def svi_total_variance_vec(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """
    Vectorised SVI total variance for numpy arrays of log-moneyness.

    Equivalent to svi_total_variance() applied element-wise but uses numpy
    broadcasting for performance in calibration inner loops.
    """
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))


def svi_implied_vol(
    k: float,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float:
    """
    SVI implied volatility: σ_impl = sqrt(w(k) / T).

    Floored at _IV_FLOOR (0.001) to prevent sqrt(0) numerical issues near
    deep OTM strikes where SVI can produce w ≈ 0.

    Parameters
    ----------
    k     : log-moneyness = log(K/F)
    T     : time to expiry in years (> 0)
    a,b,rho,m,sigma : SVI parameters

    Returns
    -------
    Annualised implied volatility (≥ 0.001)
    """
    w = svi_total_variance(k, a, b, rho, m, sigma)
    if w <= 0.0 or T <= 0.0:
        return _IV_FLOOR
    return max(_IV_FLOOR, math.sqrt(w / T))


# ── Arbitrage checks ──────────────────────────────────────────────────────────

def _svi_g(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """
    Compute the butterfly-arbitrage indicator g(k) for the SVI smile.

    Lee (2004) condition: smile is butterfly-arbitrage-free iff g(k) ≥ 0 ∀k.

        g(k) = (1 - k*w'/(2w))² - (w'/2)² * (1/w + 1/4) + w''/2

    SVI derivatives:
        w'(k)  = b * (ρ + (k-m) / sqrt((k-m)² + σ²))
        w''(k) = b * σ² / ((k-m)² + σ²)^(3/2)

    Returns g(k); negative values indicate butterfly arbitrage.
    """
    km = k - m
    denom_sqrt = math.sqrt(km * km + sigma * sigma)

    w    = a + b * (rho * km + denom_sqrt)
    wp   = b * (rho + km / denom_sqrt)                    # w'(k)
    wpp  = b * sigma * sigma / (denom_sqrt ** 3)          # w''(k)

    w    = max(w, _W_FLOOR)   # floor prevents division by zero at far wings

    g = (1.0 - k * wp / (2.0 * w)) ** 2 \
        - (wp / 2.0) ** 2 * (1.0 / w + 0.25) \
        + wpp / 2.0
    return g


def _check_static_arbitrage(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    T: float,
) -> tuple[bool, list[str]]:
    """
    Check all static no-arbitrage parameter constraints.

    Returns (is_valid, list_of_violations).

    Constraints checked (Gatheral 2004):
        (C1) b ≥ 0
        (C2) |ρ| < 1
        (C3) σ_svi > 0
        (C4) a + b*σ*sqrt(1-ρ²) ≥ 0
        (C5) a + b*σ*sqrt(1-ρ²) ≤ 4/T  (requires T > 0)
    """
    violations: list[str] = []

    if b < 0.0:
        violations.append(f"(C1) b={b:.6f} < 0")
    if abs(rho) >= 1.0:
        violations.append(f"(C2) |ρ|={abs(rho):.6f} ≥ 1")
    if sigma <= 0.0:
        violations.append(f"(C3) σ_svi={sigma:.6f} ≤ 0")

    if b >= 0.0 and abs(rho) < 1.0 and sigma > 0.0:
        wing = a + b * sigma * math.sqrt(1.0 - rho * rho)
        if wing < 0.0:
            violations.append(f"(C4) a+b*σ*sqrt(1-ρ²)={wing:.6f} < 0")
        if T > 0.0 and wing > 4.0 / T:
            violations.append(
                f"(C5) a+b*σ*sqrt(1-ρ²)={wing:.6f} > 4/T={4.0/T:.6f}"
            )

    return len(violations) == 0, violations


# ── SVIParams dataclass ───────────────────────────────────────────────────────

@dataclass
class SVIParams:
    """
    SVI raw parametrization parameters (Gatheral 2004).

    Attributes
    ----------
    a     : level shift — total variance at k→+∞ behaves as
            a + b*σ*sqrt(1-ρ²) + b*ρ*(k-m) (asymptotically linear in k)
    b     : ATM vol-of-vol proxy; controls slope and curvature (b ≥ 0)
    rho   : correlation between spot and vol (-1 < rho < 1).
            Negative rho ⟹ left skew (typical for equity indices).
    m     : log-moneyness at which the smile minimum occurs when rho=0.
            Positive m means the minimum is shifted to OTM calls.
    sigma : width (curvature) of the smile (σ_svi > 0).
            Larger σ ⟹ flatter, more symmetric smile.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> list[float]:
        """Return [a, b, rho, m, sigma] for use with scipy.optimize."""
        return [self.a, self.b, self.rho, self.m, self.sigma]

    @classmethod
    def from_array(cls, x: list[float] | np.ndarray) -> "SVIParams":
        """Construct from a 5-element array [a, b, rho, m, sigma]."""
        return cls(a=float(x[0]), b=float(x[1]), rho=float(x[2]),
                   m=float(x[3]), sigma=float(x[4]))

    def __repr__(self) -> str:
        return (
            f"SVIParams(a={self.a:.6f}, b={self.b:.6f}, ρ={self.rho:.6f}, "
            f"m={self.m:.6f}, σ={self.sigma:.6f})"
        )


# ── SVISmile ──────────────────────────────────────────────────────────────────

class SVISmile:
    """
    Fitted SVI smile for a single expiry.

    Encapsulates a calibrated SVIParams, the expiry T, and the forward price F.
    Provides total_variance, implied_vol, and butterfly-arbitrage checks.

    Usage
    -----
    smile = SVISmile(params, T=30/365, forward=24000.0)
    iv = smile.implied_vol(K=23500.0)
    ok, msg = smile.is_arbitrage_free(strikes=[22000, 23000, 24000, 25000, 26000])
    """

    def __init__(self, params: SVIParams, T: float, forward: float) -> None:
        """
        Parameters
        ----------
        params  : calibrated SVI parameters
        T       : time to expiry in years (> 0)
        forward : forward price F = S * exp((r-q)*T)
        """
        if T <= 0.0:
            raise ValueError(f"T must be > 0, got {T}")
        if forward <= 0.0:
            raise ValueError(f"forward must be > 0, got {forward}")
        self.params  = params
        self.T       = T
        self.forward = forward

    def _log_moneyness(self, K: float) -> float:
        """k = log(K/F): log-moneyness convention matching SVI calibration."""
        if K <= 0.0:
            raise ValueError(f"Strike K must be > 0, got {K}")
        return math.log(K / self.forward)

    def total_variance(self, K: float) -> float:
        """
        Total implied variance w(K) = σ_impl(K)² * T.

        Parameters
        ----------
        K : strike price (> 0)

        Returns
        -------
        w : total implied variance (≥ 0)
        """
        k = self._log_moneyness(K)
        p = self.params
        return svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma)

    def implied_vol(self, K: float) -> float:
        """
        Implied volatility σ_impl(K) = sqrt(w(K) / T).

        Floored at 0.001 to handle far-wing numerical edge cases.

        Parameters
        ----------
        K : strike price (> 0)

        Returns
        -------
        Annualised implied volatility (≥ 0.001)
        """
        k = self._log_moneyness(K)
        p = self.params
        return svi_implied_vol(k, self.T, p.a, p.b, p.rho, p.m, p.sigma)

    def is_arbitrage_free(
        self,
        strikes: list[float],
        n_butterfly_checks: int = 200,
    ) -> tuple[bool, str]:
        """
        Check whether this SVI smile is free of static arbitrage.

        Performs two checks:
          1. Parameter-level no-arb constraints (C1–C5, Gatheral 2004).
          2. Butterfly arbitrage: sample g(k) over the strike range and
             over an extended log-moneyness range [-3, 3] to catch wing violations.

        Parameters
        ----------
        strikes            : list of observed strikes; used to set the sampling range
        n_butterfly_checks : number of equally-spaced k values to evaluate g(k)

        Returns
        -------
        (is_valid, message)
            is_valid : True iff no arbitrage found
            message  : "OK" or human-readable description of the first violation
        """
        p = self.params

        # ── Parameter constraints ──
        valid, violations = _check_static_arbitrage(p.a, p.b, p.rho, p.m, p.sigma, self.T)
        if not valid:
            return False, "; ".join(violations)

        # ── Butterfly check over extended range ──
        if strikes:
            k_min = math.log(min(strikes) / self.forward)
            k_max = math.log(max(strikes) / self.forward)
            # Extend by 0.5 log-moneyness units on each side to catch wing arb
            k_min = min(k_min - 0.5, -3.0)
            k_max = max(k_max + 0.5,  3.0)
        else:
            k_min, k_max = -3.0, 3.0

        k_grid = np.linspace(k_min, k_max, n_butterfly_checks)
        for k_val in k_grid:
            g_val = _svi_g(float(k_val), p.a, p.b, p.rho, p.m, p.sigma)
            if g_val < -1e-8:   # small tolerance for floating-point
                return (
                    False,
                    f"Butterfly arbitrage: g(k={k_val:.4f})={g_val:.6e} < 0",
                )

        return True, "OK"

    def __repr__(self) -> str:
        return (
            f"SVISmile(T={self.T:.6f}, F={self.forward:.2f}, {self.params})"
        )


# ── SVISurface ────────────────────────────────────────────────────────────────

class SVISurface:
    """
    Collection of calibrated SVISmiles, one per expiry.

    Supports:
    - Adding smiles per expiry via add_smile()
    - Querying implied vol at arbitrary (K, T) via linear interpolation of
      total variances between the two bracketing expiries
    - Calibrating a full surface from a raw option chain via calibrate()

    Interpolation convention (calendar-spread-safe)
    ------------------------------------------------
    For a target maturity T with T1 ≤ T ≤ T2:

        w_interp(k) = w1(k) + (T - T1) / (T2 - T1) * (w2(k) - w1(k))
        σ_impl = sqrt(w_interp / T)

    This is linear interpolation in total variance, which guarantees calendar-
    spread-arbitrage-free interpolation provided w1(k) ≤ w2(k) for all k
    (i.e., the per-expiry calibrations are themselves consistent).

    Extrapolation: flat (use nearest smile) beyond the last or before the first
    expiry — marks log a warning.
    """

    def __init__(self) -> None:
        # Sorted list of expiries (ascending T)
        self._expiries: list[float] = []
        # Map T -> SVISmile
        self._smiles: dict[float, SVISmile] = {}

    def add_smile(self, T: float, smile: SVISmile) -> None:
        """
        Add (or replace) the smile for expiry T.

        Parameters
        ----------
        T     : time to expiry in years (> 0)
        smile : calibrated SVISmile for this expiry
        """
        if T <= 0.0:
            raise ValueError(f"T must be > 0, got {T}")
        if T not in self._smiles:
            self._expiries.append(T)
            self._expiries.sort()
        self._smiles[T] = smile
        log.debug("svi_surface.add_smile", T=T, params=repr(smile.params))

    def implied_vol(self, K: float, T: float) -> float:
        """
        Implied volatility at strike K and maturity T via surface interpolation.

        Algorithm
        ---------
        1. Find bracket expiries T1 ≤ T ≤ T2.
        2. Compute total variances w1(k), w2(k) from the two bracket smiles.
        3. Linearly interpolate: w(k) = w1 + α*(w2-w1),  α = (T-T1)/(T2-T1).
        4. Return sqrt(max(w, 0) / T).

        Extrapolation: if T < T_min or T > T_max, uses the nearest smile (flat).

        Parameters
        ----------
        K : strike price (> 0)
        T : target maturity in years (> 0)

        Returns
        -------
        Annualised implied volatility (≥ 0.001)
        """
        if not self._expiries:
            raise RuntimeError("SVISurface has no smiles; call add_smile() first")
        if T <= 0.0:
            raise ValueError(f"T must be > 0, got {T}")

        # ── Extrapolation guard ──
        if T <= self._expiries[0]:
            if T < self._expiries[0]:
                log.warning("svi_surface.extrapolate_near",
                            T=T, T_min=self._expiries[0])
            return self._smiles[self._expiries[0]].implied_vol(K)

        if T >= self._expiries[-1]:
            if T > self._expiries[-1]:
                log.warning("svi_surface.extrapolate_far",
                            T=T, T_max=self._expiries[-1])
            return self._smiles[self._expiries[-1]].implied_vol(K)

        # ── Find bracket ──
        T1, T2 = self._expiries[0], self._expiries[1]
        for i in range(len(self._expiries) - 1):
            if self._expiries[i] <= T <= self._expiries[i + 1]:
                T1 = self._expiries[i]
                T2 = self._expiries[i + 1]
                break

        smile1 = self._smiles[T1]
        smile2 = self._smiles[T2]

        # k must be computed consistently from the *target* forward, not per-slice
        # forward. For interpolation we use each smile's own forward for w(k).
        # The log-moneyness k = log(K/F) depends on T through F(T); we use each
        # smile's native forward so that w values are on the same smile domain.
        w1 = smile1.total_variance(K)
        w2 = smile2.total_variance(K)

        alpha = (T - T1) / (T2 - T1)
        w_interp = w1 + alpha * (w2 - w1)

        # Floor to zero before sqrt
        w_interp = max(0.0, w_interp)
        return max(_IV_FLOOR, math.sqrt(w_interp / T))

    def calendar_spread_check(
        self,
        strikes: list[float],
    ) -> tuple[bool, str]:
        """
        Verify that total variances are non-decreasing across expiries for all strikes.

        For each consecutive pair (T1, T2) and each strike K:
            w_T1(K) ≤ w_T2(K)  [calendar-spread-arbitrage-free condition]

        Parameters
        ----------
        strikes : list of strikes to check

        Returns
        -------
        (is_valid, message)
        """
        if len(self._expiries) < 2:
            return True, "OK (fewer than 2 expiries)"

        for i in range(len(self._expiries) - 1):
            T1 = self._expiries[i]
            T2 = self._expiries[i + 1]
            s1 = self._smiles[T1]
            s2 = self._smiles[T2]
            for K in strikes:
                w1 = s1.total_variance(K)
                w2 = s2.total_variance(K)
                if w1 > w2 + 1e-8:
                    return (
                        False,
                        f"Calendar spread arb: w(K={K},T1={T1:.4f})={w1:.6f} > "
                        f"w(K={K},T2={T2:.4f})={w2:.6f}",
                    )
        return True, "OK"

    def calibrate(
        self,
        chain_data: list[dict],
        r: float,
        q: float,
    ) -> dict[float, SVISmile]:
        """
        Calibrate a complete SVI surface from raw option chain data.

        For each distinct expiry T in chain_data:
          1. Compute forward F = S * exp((r-q)*T).
          2. Solve implied vols from mid prices using BSMEngine.iv().
          3. Convert to total variance: w_i = iv_i² * T.
          4. Compute vega² weights.
          5. Call fit_svi() to calibrate SVI parameters.
          6. Construct SVISmile and add to surface.

        Parameters
        ----------
        chain_data : list of dicts, each with keys:
            {
                "T"           : float   — time to expiry in years
                "K"           : float   — strike
                "option_type" : str     — "CE" or "PE"
                "mid_price"   : float   — mid of bid/ask
                "S"           : float   — spot price at observation time
                "r"           : float   — (optional; overridden by r arg)
                "q"           : float   — (optional; overridden by q arg)
            }
        r : continuously compounded risk-free rate (e.g. 0.065)
        q : continuous dividend yield (e.g. 0.015)

        Returns
        -------
        dict mapping T -> SVISmile (same as what was added to self._smiles)
        """
        # ── Group by expiry ──
        expiry_map: dict[float, list[dict]] = {}
        for row in chain_data:
            T = float(row["T"])
            expiry_map.setdefault(T, []).append(row)

        bsm = BSMEngine(r=r, q=q)
        result: dict[float, SVISmile] = {}

        for T in sorted(expiry_map.keys()):
            rows = expiry_map[T]
            log.info("svi_surface.calibrate_expiry",
                     T=T, n_options=len(rows))

            # ── Compute forward ──
            # Use the first row's spot price (they should be the same for a given T)
            S_ref = float(rows[0]["S"])
            F = S_ref * math.exp((r - q) * T)

            # ── Solve IVs and compute total variances ──
            log_moneyness_list: list[float] = []
            total_var_list:     list[float] = []
            vega_list:          list[float] = []
            valid_strikes:      list[float] = []

            for row in rows:
                K          = float(row["K"])
                mid_price  = float(row["mid_price"])
                S          = float(row["S"])
                opt_type   = str(row["option_type"])

                iv_result = implied_volatility(
                    market_price=mid_price,
                    S=S, K=K, T=T,
                    r=r, q=q,
                    option_type=opt_type,
                )
                if not iv_result.converged or iv_result.iv is None:
                    log.warning(
                        "svi_surface.iv_failed",
                        K=K, T=T, option_type=opt_type,
                        mid_price=mid_price,
                        error=iv_result.error_msg,
                    )
                    continue

                iv = max(_IV_FLOOR, iv_result.iv)
                w  = iv * iv * T         # total implied variance
                k  = math.log(K / F)    # log-moneyness

                # Vega (annual, unnormalised) for weighting
                sqrt_T = math.sqrt(T)
                # d1 for vega: recompute from solved IV
                try:
                    d1 = (math.log(F / K) + 0.5 * iv * iv * T) / (iv * sqrt_T)
                    n_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
                    exp_qT = math.exp(-q * T)
                    vega = S * exp_qT * n_d1 * sqrt_T
                except (ValueError, ZeroDivisionError):
                    vega = 1.0  # fallback to unit weight on numerical failure

                log_moneyness_list.append(k)
                total_var_list.append(w)
                vega_list.append(vega * vega)  # vega² weight
                valid_strikes.append(K)

            if len(log_moneyness_list) < 3:
                log.error(
                    "svi_surface.insufficient_data",
                    T=T,
                    n_valid=len(log_moneyness_list),
                )
                continue

            # ── Fit SVI ──
            try:
                params, rmse = fit_svi(
                    log_moneyness=log_moneyness_list,
                    total_variance=total_var_list,
                    weights=vega_list,
                )
            except Exception as exc:
                log.error("svi_surface.fit_failed", T=T, error=str(exc))
                continue

            smile = SVISmile(params=params, T=T, forward=F)
            arb_ok, arb_msg = smile.is_arbitrage_free(valid_strikes)
            if not arb_ok:
                log.warning(
                    "svi_surface.arbitrage_detected",
                    T=T, reason=arb_msg,
                )

            self.add_smile(T, smile)
            result[T] = smile
            log.info(
                "svi_surface.calibrated",
                T=T, rmse=rmse, params=repr(params),
                arbitrage_free=arb_ok,
            )

        return result

    def __repr__(self) -> str:
        n = len(self._expiries)
        return f"SVISurface(n_expiries={n}, T={self._expiries})"


# ── fit_svi ───────────────────────────────────────────────────────────────────

def fit_svi(
    log_moneyness: list[float],
    total_variance: list[float],
    weights: Optional[list[float]] = None,
    initial_params: Optional[SVIParams] = None,
) -> tuple[SVIParams, float]:
    """
    Fit SVI raw parametrization to observed (k, w) pairs.

    Minimises the weighted sum of squared errors:
        loss = Σᵢ wᵢ * (w_market_i - w_svi_i)²
             + penalty_for_arbitrage_violations

    Arbitrage penalty (added when parameter constraints are violated):
        For each violated constraint, add _ARBIT_PENALTY * max(0, -violation)²

    Optimiser: scipy.optimize.minimize with L-BFGS-B.
    Strategy : _N_RESTARTS (5) random restarts; best solution retained.

    Parameters
    ----------
    log_moneyness   : k_i = log(K_i / F) for each observed strike
    total_variance  : w_i = iv_i² * T for each observation
    weights         : vega²_i weights; uniform weights if None
    initial_params  : if provided, used as one of the starting points (plus restarts)

    Returns
    -------
    (SVIParams, rmse)
        SVIParams : fitted parameters with lowest calibration error
        rmse      : root-mean-squared error (unweighted, in total-variance units)

    Raises
    ------
    ValueError : if fewer than 3 observations are provided (underdetermined)
    """
    n = len(log_moneyness)
    if n < 3:
        raise ValueError(
            f"fit_svi requires at least 3 observations, got {n}"
        )
    if len(total_variance) != n:
        raise ValueError(
            f"log_moneyness and total_variance must have the same length "
            f"({n} vs {len(total_variance)})"
        )

    k_arr = np.array(log_moneyness, dtype=float)
    w_arr = np.array(total_variance, dtype=float)

    if weights is not None:
        if len(weights) != n:
            raise ValueError(
                f"weights length mismatch: expected {n}, got {len(weights)}"
            )
        wt = np.array(weights, dtype=float)
        # Normalise weights to sum to n (preserves scale of objective)
        wt_sum = wt.sum()
        wt = wt * n / wt_sum if wt_sum > 0 else np.ones(n)
    else:
        wt = np.ones(n)

    # ── ATM/wing statistics for heuristic initial guesses ──
    w_mean  = float(w_arr.mean())
    w_std   = float(w_arr.std()) if n > 1 else 0.01
    k_range = float(k_arr.max() - k_arr.min())

    def _objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = x
        w_fit = svi_total_variance_vec(k_arr, a, b, rho, m, sigma)
        diff  = w_arr - w_fit
        obj   = float(np.sum(wt * diff * diff))

        # ── Arbitrage penalty ──
        # (C1) b ≥ 0
        if b < 0.0:
            obj += _ARBIT_PENALTY * b * b

        # (C2) |ρ| < 1 — bounds already enforce this, but penalty adds smoothness
        rho_excess = abs(rho) - 0.999
        if rho_excess > 0.0:
            obj += _ARBIT_PENALTY * rho_excess * rho_excess

        # (C3) σ_svi > 0
        if sigma <= 0.0:
            obj += _ARBIT_PENALTY * sigma * sigma

        # (C4) a + b*σ*sqrt(1-ρ²) ≥ 0
        if b >= 0.0 and abs(rho) < 1.0 and sigma > 0.0:
            wing = a + b * sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
            if wing < 0.0:
                obj += _ARBIT_PENALTY * wing * wing

        return obj

    bounds = _BOUNDS

    # ── Candidate starting points ──
    # Heuristic seed: ATM total variance ≈ a + b*sigma (at k=m≈0, rho=0)
    starting_points: list[list[float]] = []

    if initial_params is not None:
        starting_points.append(initial_params.to_array())

    # Heuristic: flat smile centred at ATM
    starting_points.append([
        max(w_mean * 0.5, 0.001),   # a
        max(w_std * 2.0,  0.01),    # b
        -0.2,                        # rho — mild left skew typical for indices
        0.0,                         # m
        max(k_range * 0.5, 0.1),    # sigma
    ])

    # Random restarts to fill up to _N_RESTARTS
    rng = random.Random(42)  # deterministic seed for reproducibility  # ASSUMED: seed=42
    while len(starting_points) < _N_RESTARTS:
        a0     = rng.uniform(-0.1 * w_mean, 2.0 * w_mean)
        b0     = rng.uniform(0.01, 0.5)
        rho0   = rng.uniform(-0.7, 0.0)   # ASSUMED: equity indices lean negative rho
        m0     = rng.uniform(-0.2, 0.2)
        sigma0 = rng.uniform(0.05, 0.5)
        starting_points.append([a0, b0, rho0, m0, sigma0])

    # ── Run optimisation from each starting point ──
    best_params: Optional[SVIParams] = None
    best_obj    = math.inf
    best_result = None

    for x0 in starting_points:
        # Clip initial point to bounds
        x0_clipped = [
            max(lo, min(hi, v))
            for v, (lo, hi) in zip(x0, bounds)
        ]
        try:
            res = minimize(
                _objective,
                x0_clipped,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 2000,
                    "ftol":    1e-12,
                    "gtol":    1e-8,
                },
            )
        except Exception as exc:
            log.debug("fit_svi.restart_failed", error=str(exc))
            continue

        if res.fun < best_obj:
            best_obj    = res.fun
            best_result = res
            best_params = SVIParams.from_array(res.x)

    if best_params is None or best_result is None:
        raise RuntimeError(
            "fit_svi: all optimisation restarts failed — check input data"
        )

    # ── Compute unweighted RMSE ──
    p = best_params
    w_fit_arr = svi_total_variance_vec(k_arr, p.a, p.b, p.rho, p.m, p.sigma)
    rmse = float(np.sqrt(np.mean((w_arr - w_fit_arr) ** 2)))

    log.debug(
        "fit_svi.done",
        n_obs=n,
        rmse=rmse,
        best_obj=best_obj,
        params=repr(best_params),
        n_restarts=len(starting_points),
        optimizer_success=best_result.success,
    )

    return best_params, rmse


# ── realized_volatility ───────────────────────────────────────────────────────

def realized_volatility(
    log_returns: list[float],
    annualization: float = 252.0,
) -> float:
    """
    Parkinson (1980) close-to-close realized volatility estimator.

    Formula:
        RV = sqrt(annualization) * std(log_returns, ddof=1)

    where log_returns are close-to-close log returns:
        r_t = log(C_t / C_{t-1})

    The estimator is unbiased for the diffusion volatility under GBM.

    Annualization factors
    ---------------------
    Daily bars (1D):     annualization = 252  (NSE trading days per year)
    15-minute bars:      annualization = 252 * 26 = 6552
                         (26 = number of 15-min bars per NSE session of 6h30m)
    5-minute bars:       annualization = 252 * 78 = 19656  # ASSUMED: 78 5-min bars/day
    1-minute bars:       annualization = 252 * 375 = 94500 # ASSUMED: 375 min/day (09:15–15:30)

    Parameters
    ----------
    log_returns   : list of log returns r_t = log(C_t / C_{t-1})
    annualization : annualization factor (default 252 for daily bars)

    Returns
    -------
    Annualised realised volatility (≥ 0).
    Returns 0.0 if fewer than 2 returns are provided.

    Reference
    ---------
    Parkinson, M. (1980) "The Extreme Value Method for Estimating the Variance
    of the Rate of Return." Journal of Business 53(1), 61–65.
    """
    if len(log_returns) < 2:
        log.warning(
            "realized_volatility.insufficient_data",
            n=len(log_returns),
        )
        return 0.0

    r = np.array(log_returns, dtype=float)
    # Use ddof=1 (sample std dev, unbiased)
    sigma_period = float(np.std(r, ddof=1))
    return math.sqrt(annualization) * sigma_period


# ── VolSurfaceRouter ──────────────────────────────────────────────────────────

class VolSurfaceRouter:
    """
    Intelligent implied-volatility surface router.

    Selects SABR for short-dated expiries (≤ 21 DTE) and SVI for longer-dated
    expiries (> 21 DTE), providing a unified ``get_iv(T, K)`` interface.

    Motivation
    ----------
    NSE-specific model selection (see also sabr.py module docstring):

    - SVI for short-dated (≤ 21 DTE):
      The Hagan (2002) SABR expansion is an O(T) asymptotic approximation.
      For T < 21 days with high vol-of-vol ν > 0.5 (common in NSE Nifty
      weeklies), the residual error can exceed 1% IV. SVI's parametric form
      does not have this short-T instability.

    - SABR for longer-dated (> 21 DTE):
      SABR captures the skew dynamics of NSE monthly options more naturally
      via its stochastic vol formulation. The z/χ(z) correction provides better
      skew calibration for maturities 1–3 months than SVI, which fits the smile
      shape but does not model the vol dynamics.

    The 21-day threshold aligns with the NSE F&O weekly/monthly boundary:
      - Nifty weekly options expire on Tuesdays
      - 3 weekly expiry cycles ≈ 21 calendar days

    [APPROX-SABR-HAGAN2002]: SABR accuracy degrades for short T < 0.05y;
    see sabr.py for full discussion.

    [APPROX-SABR-HAGAN2002]: For T < 0.05 years, the SABR expansion is an
    approximation (see SABR module warning). For T > 0.5 years, SVI is preferred.

    Usage
    -----
    router = VolSurfaceRouter()
    router.add_sabr_smile(T=7/365, smile=sabr_smile_7d)
    router.add_svi_smile(T=30/365, smile=svi_smile_30d)

    iv = router.get_iv(T=7/365, K=24500.0)   # uses SABR
    iv = router.get_iv(T=30/365, K=24000.0)  # uses SVI

    model_name = router.model_for_expiry(T=7/365)  # → "SABR"
    expiries   = router.available_expiries()
    """

    _SHORT_DTE_THRESHOLD: float = 21.0 / 365.0  # 21 calendar days in years

    def __init__(
        self,
        svi_smiles: Optional[dict] = None,   # dict[float, SVISmile]
        sabr_smiles: Optional[dict] = None,  # dict[float, SABRSmile]
    ) -> None:
        from algo_trading.fo.sabr import SABRSmile as _SABRSmile
        self._SABRSmile = _SABRSmile
        self._svi: dict[float, SVISmile] = dict(svi_smiles or {})
        self._sabr: dict = dict(sabr_smiles or {})

    # ── Model registration ────────────────────────────────────────────────────

    def add_svi_smile(self, T: float, smile: SVISmile) -> None:
        """Register an SVI smile for expiry T."""
        if not isinstance(smile, SVISmile):
            raise TypeError(f"Expected SVISmile, got {type(smile)}")
        self._svi[T] = smile
        log.debug("vol_surface_router.svi_added", T=f"{T:.4f}")

    def add_sabr_smile(self, T: float, smile) -> None:
        """Register a SABR smile for expiry T."""
        if not isinstance(smile, self._SABRSmile):
            raise TypeError(f"Expected SABRSmile, got {type(smile)}")
        self._sabr[T] = smile
        log.debug("vol_surface_router.sabr_added", T=f"{T:.4f}")

    # ── Model selection ───────────────────────────────────────────────────────

    def model_for_expiry(self, T: float) -> str:
        """
        Return the preferred model name for expiry T.

        Rules:
          - If T ≤ _SHORT_DTE_THRESHOLD (21 DTE): prefer "SVI"
            (Hagan SABR expansion is less accurate for short T; see sabr.py)
          - If T > _SHORT_DTE_THRESHOLD:          prefer "SABR"
            (SABR captures NSE monthly skew dynamics better)
          - If the preferred model has no smile registered for T, fall back to
            the other model (if available).
          - If neither is registered, returns "NONE".

        Returns
        -------
        str
            "SABR", "SVI", or "NONE".
        """
        if T <= self._SHORT_DTE_THRESHOLD:
            if T in self._svi:
                return "SVI"
            if T in self._sabr:
                return "SABR"
        else:
            if T in self._sabr:
                return "SABR"
            if T in self._svi:
                return "SVI"
        return "NONE"

    # ── IV lookup ─────────────────────────────────────────────────────────────

    def get_iv(self, T: float, K: float) -> Optional[float]:
        """
        Return implied volatility σ(T, K) from the appropriate model.

        Requires an exact match on T. For interpolation across expiries,
        use :meth:`interpolated_iv`.

        Returns None if no model is registered for T.

        Parameters
        ----------
        T : float
            Time to expiry in years.
        K : float
            Strike price in INR.

        Returns
        -------
        float | None
            Annualised implied volatility, or None if T not registered.
        """
        model = self.model_for_expiry(T)
        if model == "SABR":
            return self._sabr[T].implied_vol(K)
        if model == "SVI":
            return self._svi[T].implied_vol(K)
        log.warning("vol_surface_router.no_model", T=T, K=K)
        return None

    def interpolated_iv(self, T: float, K: float) -> Optional[float]:
        """
        Return IV(T, K) with flat extrapolation / linear interpolation between
        the two nearest registered expiries.

        Algorithm:
          1. Collect all registered expiries (combined SVI + SABR).
          2. Find (T1, T2) bracketing T; use flat extrapolation at edges.
          3. Interpolate: σ(T) = w1 * σ(T1, K) + w2 * σ(T2, K)
             where w1 = (T2 - T) / (T2 - T1), w2 = (T - T1) / (T2 - T1).
             [Linear interpolation in vol space — [APPROX]: calendar-time
              interpolation in total variance would be more arbitrage-consistent,
              but requires matched SVI surfaces; acceptable for vol monitoring.]

        Returns None if no expiries are registered.
        """
        expiries = self.available_expiries()
        if not expiries:
            return None

        if len(expiries) == 1:
            return self.get_iv(expiries[0], K)

        # Find bracket
        if T <= expiries[0]:
            return self.get_iv(expiries[0], K)   # flat left extrapolation
        if T >= expiries[-1]:
            return self.get_iv(expiries[-1], K)  # flat right extrapolation

        # Binary search for bracket
        lo_idx = 0
        for i, t in enumerate(expiries):
            if t <= T:
                lo_idx = i

        T1 = expiries[lo_idx]
        T2 = expiries[lo_idx + 1]

        iv1 = self.get_iv(T1, K)
        iv2 = self.get_iv(T2, K)

        if iv1 is None:
            return iv2
        if iv2 is None:
            return iv1

        w1 = (T2 - T) / (T2 - T1)
        w2 = (T - T1) / (T2 - T1)
        return w1 * iv1 + w2 * iv2

    def smile_dict(
        self, T: float, strikes: list[float]
    ) -> dict[float, float]:
        """
        Return a strike → IV mapping for a specific expiry using the routed model.

        Returns an empty dict if T is not registered.

        Parameters
        ----------
        T       : expiry in years (must be an exact registered expiry)
        strikes : list of strike prices

        Returns
        -------
        dict[float, float]
            {K: σ(K)} for each K in strikes.
        """
        return {
            K: iv
            for K in strikes
            if (iv := self.get_iv(T, K)) is not None
        }

    def available_expiries(self) -> list[float]:
        """Return all registered expiries (SVI + SABR), sorted ascending."""
        all_expiries = set(self._svi.keys()) | set(self._sabr.keys())
        return sorted(all_expiries)

    def coverage_summary(self) -> dict:
        """
        Return a summary of all registered smiles and their assigned models.

        Returns
        -------
        dict
            Keys: expiry (float), values: {"T": float, "model": str, "dte_days": float}
        """
        return {
            T: {
                "T": T,
                "model": self.model_for_expiry(T),
                "dte_days": round(T * 365.0, 1),
            }
            for T in self.available_expiries()
        }

    def __repr__(self) -> str:
        n = len(self.available_expiries())
        return (
            f"VolSurfaceRouter("
            f"n_expiries={n}, "
            f"svi={sorted(self._svi)}, "
            f"sabr={sorted(self._sabr)}, "
            f"threshold={self._SHORT_DTE_THRESHOLD:.4f}yr)"
        )
