"""
algo_trading/fo/sabr.py — SABR stochastic volatility model for NSE index options.

SABR Model (Hagan, Kumar, Lesniewski, Woodward 2002)
-----------------------------------------------------
Reference:
  Hagan, P.S., Kumar, D., Lesniewski, A.S. and Woodward, D.E. (2002)
  "Managing Smile Risk." Wilmott Magazine, September, pp. 84–108.

Process description:
  dF = σ · F^β · dW₁
  dσ = ν · σ · dW₂
  corr(dW₁, dW₂) = ρ · dt

Parameters:
  α   > 0 : initial volatility — controls ATM vol level
  β ∈ [0,1]: CEV exponent (β=1 = log-normal, β=0 = normal, β=0.5 = intermediate)
             For NSE index options: β=1.0 is standard (log-normal backbone)
  ρ ∈ (-1,1): spot-vol correlation; ρ < 0 gives left (equity) skew
  ν   > 0 : vol-of-vol — controls smile curvature/width

Hagan (2002) approximation for BSM-equivalent implied vol (Eq 2.17b):
  For F ≠ K:
    z     = (ν/α) · (FK)^((1-β)/2) · ln(F/K)
    χ(z)  = ln([ √(1 - 2ρz + z²) + z - ρ ] / (1-ρ))
    A     = 1 / ((FK)^((1-β)/2) · [1 + (1-β)²/24 · ln²(F/K)
                                      + (1-β)⁴/1920 · ln⁴(F/K)])
    B     = 1 + [(1-β)²α²/(24(FK)^(1-β)) + ρβνα/(4(FK)^((1-β)/2))
                 + (2-3ρ²)ν²/24] · T
    σ_B   = α · A · (z/χ(z)) · B

  For F = K (ATM case):
    σ_B_ATM = (α / F^(1-β)) · [1 + ((1-β)²α²/(24F^(2(1-β)))
                                     + ρβνα/(4F^(1-β))
                                     + (2-3ρ²)ν²/24) · T]

Accuracy limitations
--------------------
[APPROX-SABR-HAGAN2002]: The Hagan approximation is an O(T) asymptotic
expansion.  Accuracy degrades for:
  - Short T < 0.05 years (T < ~18 days) with high vol-of-vol ν > 0.5
  - Deep OTM: |ln(F/K)| > 0.5

For NSE weeklies (T ≈ 7/365 ≈ 0.019), we recommend SVI (vol_surface.py)
over SABR because the short-T accuracy issue can cause errors > 1% IV.
SABR is more appropriate for NSE monthly options (T ≈ 0.08–0.25 years).

Calibration
-----------
SABR calibration minimizes the weighted sum of squared IV errors:
  loss = Σᵢ wᵢ · (σ_market_i − σ_SABR_i)²

For NSE options, β is typically fixed at 1.0 (log-normal backbone) and
only (α, ρ, ν) are calibrated (3-parameter fit to the smile).
Fixing β avoids the well-known SABR calibration degeneracy.

Usage
-----
  from algo_trading.fo.sabr import SABRParams, SABRSmile, fit_sabr

  # Calibrate to market smile
  params = fit_sabr(
      strikes=[22000, 23000, 24000, 25000, 26000],
      market_ivs=[0.17, 0.16, 0.15, 0.16, 0.18],
      F=24000.0, T=30/365, beta=1.0,
  )
  smile = SABRSmile(params, F=24000.0, T=30/365)
  iv_at_23500 = smile.implied_vol(23500.0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from scipy.optimize import minimize

log = structlog.get_logger(__name__)

# Minimum vol floor
_VOL_FLOOR: float = 0.001
_VOL_CAP:   float = 5.0


# ── SABR Parameters ───────────────────────────────────────────────────────────

@dataclass
class SABRParams:
    """
    SABR model parameters (Hagan 2002).

    Attributes
    ----------
    alpha : float
        Initial vol level.  Controls ATM implied vol.  Must be > 0.
    beta  : float
        CEV exponent ∈ [0, 1].  Typically fixed at 1.0 for NSE equity indices.
        β=1 = log-normal backbone (standard for index options).
        β=0 = normal/Bachelier backbone (rarely used for equity).
    rho   : float
        Spot-vol correlation ∈ (-1, 1).  Negative for left (equity) skew.
        Typical NSE range: [-0.6, -0.1].
    nu    : float
        Vol-of-vol > 0.  Controls smile curvature.
        Typical NSE range: [0.2, 1.5].
    """
    alpha: float
    beta:  float
    rho:   float
    nu:    float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"SABRParams: alpha must be > 0, got {self.alpha}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"SABRParams: beta must be in [0,1], got {self.beta}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"SABRParams: rho must be in (-1,1), got {self.rho}")
        if self.nu <= 0:
            raise ValueError(f"SABRParams: nu must be > 0, got {self.nu}")

    def to_array(self) -> list[float]:
        """Return [alpha, rho, nu] for calibration (beta fixed)."""
        return [self.alpha, self.rho, self.nu]

    def __repr__(self) -> str:
        return (
            f"SABRParams(α={self.alpha:.6f}, β={self.beta:.3f}, "
            f"ρ={self.rho:.4f}, ν={self.nu:.4f})"
        )


# ── Core formula ──────────────────────────────────────────────────────────────

def sabr_implied_vol(
    K: float,
    F: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    SABR BSM-equivalent implied volatility (Hagan 2002, Eq. 2.17b).

    [APPROX-SABR-HAGAN2002]: Asymptotic approximation; less accurate for
    short T (< 0.05 years) with high nu (> 0.5). See module docstring.

    Parameters
    ----------
    K     : strike price (> 0)
    F     : forward price = S * exp((r-q)*T) (> 0)
    T     : time to expiry in years (> 0)
    alpha : initial vol (> 0)
    beta  : CEV exponent ∈ [0, 1]
    rho   : spot-vol correlation ∈ (-1, 1)
    nu    : vol-of-vol (> 0)

    Returns
    -------
    BSM-equivalent implied vol (annualised; floored at 0.001)
    """
    if K <= 0 or F <= 0 or T <= 0:
        return _VOL_FLOOR

    # ATM case — use the closed-form limit
    if abs(K - F) < 1e-8 * F:
        return _sabr_atm(F, T, alpha, beta, rho, nu)

    # General case (F ≠ K)
    log_FK = math.log(F / K)
    FK_mid = math.sqrt(F * K)           # geometric mean
    one_minus_beta = 1.0 - beta

    # ── Factor A: (FK)^((1-β)/2) expanded ──────────────────────────────────
    # Denominator: (FK)^((1-β)/2) * [ 1 + (1-β)²/24 * ln²(F/K)
    #                                    + (1-β)⁴/1920 * ln⁴(F/K) ]
    FK_pow = FK_mid ** one_minus_beta
    log_sq = log_FK * log_FK
    log_4th = log_sq * log_sq
    denom_A = FK_pow * (
        1.0
        + (one_minus_beta ** 2) / 24.0 * log_sq
        + (one_minus_beta ** 4) / 1920.0 * log_4th
    )

    # z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
    z = (nu / alpha) * FK_pow * log_FK

    # χ(z) = ln([ √(1 - 2ρz + z²) + z - ρ ] / (1-ρ))
    # Guard against z ≈ 0 (L'Hopital: z/χ(z) → 1)
    if abs(z) < 1e-8:
        z_over_chi = 1.0
    else:
        disc = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 1e-14))
        numerator = disc + z - rho
        if numerator <= 0 or (1.0 - rho) <= 0:
            z_over_chi = 1.0
        else:
            chi = math.log(numerator / (1.0 - rho))
            z_over_chi = z / chi if abs(chi) > 1e-12 else 1.0

    # ── Factor B: T-correction ──────────────────────────────────────────────
    # B = 1 + [(1-β)²α²/(24(FK)^(1-β)) + ρβνα/(4(FK)^((1-β)/2))
    #          + (2-3ρ²)ν²/24] * T
    term1 = (one_minus_beta ** 2) * alpha * alpha / (24.0 * FK_pow ** 2)
    term2 = rho * beta * nu * alpha / (4.0 * FK_pow)
    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    B = 1.0 + (term1 + term2 + term3) * T

    sigma_B = alpha / denom_A * z_over_chi * B
    return max(_VOL_FLOOR, min(sigma_B, _VOL_CAP))


def _sabr_atm(
    F: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    ATM SABR implied vol (K = F limit).

    Exact expression as K→F:
      σ_B_ATM = α/F^(1-β) * [1 + ((1-β)²α²/(24F^(2(1-β)))
                                    + ρβνα/(4F^(1-β))
                                    + (2-3ρ²)ν²/24) * T]
    """
    one_minus_beta = 1.0 - beta
    F_pow   = F ** one_minus_beta
    F_pow2  = F ** (2.0 * one_minus_beta)

    term1 = (one_minus_beta ** 2) * alpha * alpha / (24.0 * F_pow2)
    term2 = rho * beta * nu * alpha / (4.0 * F_pow)
    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0

    sigma_atm = (alpha / F_pow) * (1.0 + (term1 + term2 + term3) * T)
    return max(_VOL_FLOOR, min(sigma_atm, _VOL_CAP))


# ── SABRSmile ─────────────────────────────────────────────────────────────────

class SABRSmile:
    """
    Calibrated SABR smile for a single expiry.

    Parameters
    ----------
    params  : SABRParams (calibrated α, β, ρ, ν)
    F       : forward price F = S * exp((r-q)*T)
    T       : time to expiry in years (> 0)
    """

    def __init__(self, params: SABRParams, F: float, T: float) -> None:
        if F <= 0:
            raise ValueError(f"Forward F must be > 0, got {F}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        self.params = params
        self.F = F
        self.T = T

    def implied_vol(self, K: float) -> float:
        """
        SABR BSM-equivalent implied vol at strike K.

        [APPROX-SABR-HAGAN2002]: See module-level accuracy limitations.

        Parameters
        ----------
        K : strike price (> 0)

        Returns
        -------
        Annualised implied volatility (≥ 0.001)
        """
        p = self.params
        return sabr_implied_vol(K, self.F, self.T, p.alpha, p.beta, p.rho, p.nu)

    def smile(self, strikes: list[float]) -> dict[float, float]:
        """Return {K: IV} for a list of strikes."""
        return {K: self.implied_vol(K) for K in strikes}

    def atm_vol(self) -> float:
        """ATM implied vol at K = F."""
        p = self.params
        return _sabr_atm(self.F, self.T, p.alpha, p.beta, p.rho, p.nu)

    def risk_reversal(self, delta: float = 0.25) -> float:
        """
        Risk reversal ≈ IV(25Δ call) - IV(25Δ put).

        Uses a moneyness approximation for the 25-delta strikes:
          K_call ≈ F * exp( N⁻¹(1-δ) * σ_ATM * √T )
          K_put  ≈ F * exp(-N⁻¹(1-δ) * σ_ATM * √T )
        where N⁻¹ is the probit (inverse normal CDF).

        [APPROX]: Uses ATM vol to estimate delta strikes rather than exact
        delta inversion. Accurate to within ~1% for typical NSE smile shapes.

        Parameters
        ----------
        delta : delta level (default 0.25 = 25Δ risk reversal)

        Returns
        -------
        RR = IV_call_25d − IV_put_25d  (positive = right skew, negative = left skew)
        """
        from scipy.stats import norm
        sigma_atm = self.atm_vol()
        sqrt_T = math.sqrt(self.T)
        d_move = norm.ppf(1.0 - delta) * sigma_atm * sqrt_T   # [APPROX]
        K_call = self.F * math.exp(d_move)
        K_put  = self.F * math.exp(-d_move)
        return self.implied_vol(K_call) - self.implied_vol(K_put)

    def butterfly(self, delta: float = 0.25) -> float:
        """
        Butterfly ≈ 0.5 * (IV(25Δ call) + IV(25Δ put)) - IV(ATM).

        Positive butterfly = smile has convexity (wings above ATM level).

        [APPROX]: Same delta-strike approximation as risk_reversal().

        Parameters
        ----------
        delta : delta level (default 0.25 = 25Δ butterfly)

        Returns
        -------
        BF = 0.5 * (IV_call_25d + IV_put_25d) - IV_ATM
        """
        from scipy.stats import norm
        sigma_atm = self.atm_vol()
        sqrt_T = math.sqrt(self.T)
        d_move = norm.ppf(1.0 - delta) * sigma_atm * sqrt_T   # [APPROX]
        K_call = self.F * math.exp(d_move)
        K_put  = self.F * math.exp(-d_move)
        iv_call = self.implied_vol(K_call)
        iv_put  = self.implied_vol(K_put)
        return 0.5 * (iv_call + iv_put) - sigma_atm

    def __repr__(self) -> str:
        return f"SABRSmile(F={self.F:.2f}, T={self.T:.4f}, {self.params})"


# ── Calibration ───────────────────────────────────────────────────────────────

@dataclass
class SABRCalibrationResult:
    """
    Result from fit_sabr().

    Attributes
    ----------
    params      : calibrated SABRParams
    rmse        : root-mean-squared error (in IV units, not bps)
    converged   : True if optimizer converged
    n_iter      : number of optimizer iterations
    warning     : any calibration warnings (short T, insufficient data, etc.)
    """
    params:    SABRParams
    rmse:      float
    converged: bool
    n_iter:    int
    warning:   str = ""


def fit_sabr(
    strikes:    list[float],
    market_ivs: list[float],
    F:          float,
    T:          float,
    beta:       float = 1.0,
    weights:    Optional[list[float]] = None,
) -> SABRCalibrationResult:
    """
    Calibrate SABR parameters (α, ρ, ν) to an observed NSE option smile.

    Beta is fixed (not calibrated) to avoid the well-known SABR degeneracy
    in (α, β) space.  For NSE index options, β=1.0 (log-normal backbone) is
    standard.

    Objective: minimize Σᵢ wᵢ * (σ_market_i − σ_SABR(K_i))²

    Optimizer: scipy.optimize.minimize with L-BFGS-B.
    Three restarts with different initial guesses for (α, ρ, ν) to avoid
    local minima.

    [APPROX-SABR-HAGAN2002]: See module-level accuracy notes for short T.

    Parameters
    ----------
    strikes    : observed strike prices (aligned with market_ivs)
    market_ivs : observed implied vols (annualised, e.g. 0.15 for 15%)
    F          : forward price
    T          : time to expiry in years
    beta       : fixed CEV exponent (default 1.0 = log-normal backbone)
    weights    : optional per-strike weights; default = uniform

    Returns
    -------
    SABRCalibrationResult with calibrated (α, ρ, ν) and RMSE

    Raises
    ------
    ValueError : if fewer than 3 strikes provided
    """
    n = len(strikes)
    if n < 3:
        raise ValueError(f"fit_sabr requires at least 3 strikes, got {n}")
    if len(market_ivs) != n:
        raise ValueError(
            f"strikes and market_ivs must have the same length ({n} vs {len(market_ivs)})"
        )

    wt = np.ones(n) if weights is None else np.array(weights, dtype=float)
    wt = wt / wt.sum() * n  # normalise

    K_arr  = np.array(strikes, dtype=float)
    iv_arr = np.array(market_ivs, dtype=float)

    warning = ""
    if T < 0.05:
        warning = (
            f"[APPROX-SABR-HAGAN2002]: T={T:.4f} yr < 0.05 (< ~18 days). "
            "Hagan approximation may have >1% IV error. Use SVI for weeklies."
        )
        log.warning("sabr.short_t_warning", T=T)

    def _objective(x: np.ndarray) -> float:
        alpha_fit, rho_fit, nu_fit = float(x[0]), float(x[1]), float(x[2])
        # Guard parameter range
        if alpha_fit <= 0 or nu_fit <= 0 or abs(rho_fit) >= 1.0:
            return 1e10
        err_sq = 0.0
        for i, K in enumerate(K_arr):
            iv_model = sabr_implied_vol(K, F, T, alpha_fit, beta, rho_fit, nu_fit)
            diff = iv_model - float(iv_arr[i])
            err_sq += float(wt[i]) * diff * diff
        return err_sq

    # Bounds: α > 0, ρ ∈ (-1, 1), ν > 0
    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]

    # ATM IV as initial α guess
    atm_idx = int(np.argmin(np.abs(K_arr - F)))
    sigma_atm_guess = float(iv_arr[atm_idx])

    # Initial guesses for (α, ρ, ν)
    initial_guesses = [
        [sigma_atm_guess, -0.3, 0.5],
        [sigma_atm_guess * 0.8, -0.5, 0.8],
        [sigma_atm_guess * 1.2, -0.1, 0.3],
    ]

    best_result = None
    best_val = float("inf")

    for x0 in initial_guesses:
        try:
            res = minimize(
                _objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
        except Exception as e:
            log.debug("sabr.calibration_attempt_failed", error=str(e))

    if best_result is None:
        # All attempts failed — return default params
        return SABRCalibrationResult(
            params=SABRParams(alpha=sigma_atm_guess, beta=beta, rho=-0.3, nu=0.5),
            rmse=float("inf"),
            converged=False,
            n_iter=0,
            warning="All calibration attempts failed; using default params.",
        )

    alpha_opt, rho_opt, nu_opt = best_result.x
    # Clamp to valid ranges
    alpha_opt = max(1e-4, float(alpha_opt))
    rho_opt   = max(-0.999, min(0.999, float(rho_opt)))
    nu_opt    = max(1e-4, float(nu_opt))

    fitted_params = SABRParams(alpha=alpha_opt, beta=beta, rho=rho_opt, nu=nu_opt)

    # RMSE in IV units
    rmse = math.sqrt(float(best_result.fun) / n)

    log.info(
        "sabr.calibrated",
        alpha=round(alpha_opt, 6),
        rho=round(rho_opt, 4),
        nu=round(nu_opt, 4),
        rmse_bps=round(rmse * 10000, 2),
        converged=bool(best_result.success),
    )

    return SABRCalibrationResult(
        params=fitted_params,
        rmse=rmse,
        converged=bool(best_result.success),
        n_iter=int(best_result.nit),
        warning=warning,
    )
