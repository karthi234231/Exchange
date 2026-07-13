"""
algo_trading/fo/vix_estimator.py — India VIX approximation using CBOE methodology.

Computes a VIX-style implied volatility index from a Nifty option chain snapshot.

CBOE VIX Methodology (exact)
-----------------------------
Reference: CBOE (2009) "The CBOE Volatility Index — VIX" white paper.
NSE India VIX follows the same methodology applied to Nifty 50 options.

Formula
-------
For each expiry T:

    σ²(T) = (2/T) * Σᵢ [ΔKᵢ/Kᵢ²] * e^(rT) * Q(Kᵢ) − (1/T) * (F/K₀ − 1)²

where:
  T      : time to expiry in years  (calendar_days / 365)
  r      : risk-free rate (RBI repo rate proxy)
  F      : implied forward price from put-call parity:
             F = K* + e^(rT) * |C(K*) − P(K*)|
             where K* = strike with minimum |C - P| difference
  K₀     : first listed strike at or below F
  Q(Kᵢ)  : midpoint of bid/ask for OTM option at Kᵢ:
             • K < K₀  → put price
             • K > K₀  → call price
             • K = K₀  → average of call and put
  ΔKᵢ    : strike interval:
             • interior strikes: (K_{i+1} − K_{i−1}) / 2
             • lowest/highest   : distance to the next/previous strike
  Only include strikes with non-zero bid (Q > 0 and bid > 0).

30-day interpolation
--------------------
Choose near-term T₁ < 30d and next-term T₂ > 30d.
If no next-term expiry available, use just T₁.

  N_T₁  = calendar minutes to T₁ expiry
  N_T₂  = calendar minutes to T₂ expiry
  N₃₀   = 30 * 1440 calendar minutes
  N₃₆₅  = 365 * 1440 calendar minutes

  VIX²  = (N_T₁ * σ₁² * (N_T₂ − N₃₀)/(N_T₂ − N_T₁)
         + N_T₂ * σ₂² * (N₃₀ − N_T₁)/(N_T₂ − N_T₁))
         × N₃₆₅ / N₃₀

  VIX   = 100 * sqrt(VIX²)

When only one expiry is available, no interpolation is performed and σ
is computed directly from that expiry.

IMPORTANT notes
---------------
- Results are tagged [APPROX-VIXMODEL] when < 5 valid strikes exist.
- This is an approximation of India VIX. The actual NSE India VIX computation
  uses real-time bid/ask data across all listed strikes and official rate curves.
  This implementation is for research/risk-monitoring purposes only.
- Zero-bid options are excluded (they are untradeable and distort the estimate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# NSE convention: calendar days / 365 (same as our time_to_expiry_years)
_DAYS_PER_YEAR = 365.0
_MINUTES_PER_DAY = 1440.0
_N30  = 30.0  * _MINUTES_PER_DAY   # 30 calendar days in minutes
_N365 = 365.0 * _MINUTES_PER_DAY   # 365 calendar days in minutes

# Minimum valid strikes to report a confident VIX estimate
_MIN_VALID_STRIKES = 5


@dataclass
class VIXEstimate:
    """
    Result of the India VIX estimation.

    Attributes
    ----------
    vix            : VIX value as a percentage (e.g. 15.2 = 15.2%).
                     -1.0 if estimation failed.
    near_term_vix  : σ(T₁) × 100 for the near-term expiry (% units).
    next_term_vix  : σ(T₂) × 100 for the next-term expiry (% units).
                     0.0 if only one expiry was used.
    near_expiry    : near-term expiry date used.
    next_expiry    : next-term expiry date used (None if only one expiry).
    n_near_strikes : number of valid OTM strikes used for near-term.
    n_next_strikes : number of valid OTM strikes used for next-term.
    is_approx      : True when < _MIN_VALID_STRIKES were available
                     (result is tagged as [APPROX]).
    warning        : any quality warning message.
    """
    vix:            float
    near_term_vix:  float
    next_term_vix:  float
    near_expiry:    date
    next_expiry:    date | None
    n_near_strikes: int
    n_next_strikes: int
    is_approx:      bool
    warning:        str = ""

    def to_log_dict(self) -> dict:
        return {
            "vix": round(self.vix, 4),
            "near_vix": round(self.near_term_vix, 4),
            "next_vix": round(self.next_term_vix, 4),
            "near_expiry": str(self.near_expiry),
            "next_expiry": str(self.next_expiry) if self.next_expiry else None,
            "n_near": self.n_near_strikes,
            "n_next": self.n_next_strikes,
            "is_approx": self.is_approx,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class _StrikeData:
    """Internal: one strike's contribution to the VIX sum."""
    K: float       # strike
    Q: float       # OTM option midpoint price (bid/ask mid)
    bid: float     # bid (to filter zero-bid)
    is_call: bool  # True = call side, False = put side


def _midpoint(bid: float, ask: float, ltp: float) -> float:
    """Return (bid+ask)/2 when both > 0, else ltp."""
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    return ltp


def _compute_vix_sq_for_expiry(
    strike_data: list[_StrikeData],
    T: float,
    r: float,
    F: float,
    K0: float,
) -> tuple[float, int]:
    """
    Compute σ²(T) for one expiry using the CBOE VIX formula.

    Parameters
    ----------
    strike_data : valid OTM strikes sorted ascending by K
    T           : time to expiry in years
    r           : risk-free rate
    F           : implied forward price
    K0          : first listed strike at or below F

    Returns
    -------
    (sigma_sq, n_valid_strikes)
    """
    if not strike_data or T <= 0:
        return 0.0, 0

    strikes = sorted(strike_data, key=lambda s: s.K)
    n = len(strikes)

    # ── Compute ΔK for each strike ────────────────────────────────────────
    delta_K: list[float] = []
    for i, s in enumerate(strikes):
        if i == 0:
            # Lowest strike: Δ = K₁ - K₀ (distance to next)
            dK = strikes[1].K - strikes[0].K if n > 1 else 0.0
        elif i == n - 1:
            # Highest strike: Δ = K_{n-1} - K_{n-2} (distance to prev)
            dK = strikes[-1].K - strikes[-2].K
        else:
            # Interior: Δ = (K_{i+1} - K_{i-1}) / 2
            dK = (strikes[i + 1].K - strikes[i - 1].K) / 2.0
        delta_K.append(dK)

    # ── Sum: Σ (ΔKᵢ / Kᵢ²) * Q(Kᵢ) ──────────────────────────────────────
    total = 0.0
    for s, dK in zip(strikes, delta_K):
        if dK <= 0 or s.bid <= 0:
            continue
        total += (dK / (s.K * s.K)) * s.Q

    e_rT = math.exp(r * T)
    sigma_sq = (2.0 / T) * e_rT * total - (1.0 / T) * ((F / K0) - 1.0) ** 2
    sigma_sq = max(sigma_sq, 0.0)   # numerical floor

    return sigma_sq, n


def _select_otm_strikes(
    snapshot: Any,
    expiry: date,
    F: float,
    K0: float,
) -> list[_StrikeData]:
    """
    Select all OTM strikes for a given expiry from a ChainSnapshot.

    Selection rules:
      - K < K₀  : use put (OTM put)
      - K > K₀  : use call (OTM call)
      - K = K₀  : use average of call and put
      - Q > 0   : exclude if midpoint is zero
      - bid > 0 : exclude zero-bid (untradeable)
      - Stop collecting when two consecutive strikes have zero bid
        (they mark the edge of the visible chain)

    Returns sorted list of _StrikeData.
    """
    # Get all strikes for this expiry
    if hasattr(snapshot, "strikes_by_expiry"):
        all_strikes = snapshot.strikes_by_expiry(expiry)
    elif isinstance(snapshot, dict):
        quotes = snapshot.get("quotes", {})
        all_strikes = sorted({s for (e, s, _) in quotes if e == expiry})
    else:
        return []

    if not all_strikes:
        return []

    result: list[_StrikeData] = []
    # Track consecutive zero-bid counts (stop after 2 consecutive)
    consec_zero_lower = 0
    consec_zero_upper = 0

    def _get_quote(K: float, opt_type: str):
        if hasattr(snapshot, "get_quote"):
            return snapshot.get_quote(expiry, K, opt_type)
        elif isinstance(snapshot, dict):
            return snapshot.get("quotes", {}).get((expiry, K, opt_type))
        return None

    # ── Collect strikes from K₀ downward (puts) ──────────────────────────
    below = [K for K in sorted(all_strikes, reverse=True) if K <= K0]
    for K in below:
        pe = _get_quote(K, "PE")
        if pe is None:
            consec_zero_lower += 1
            if consec_zero_lower >= 2:
                break
            continue
        bid = getattr(pe, "bid", None) or 0.0
        ask = getattr(pe, "ask", None) or 0.0
        ltp = getattr(pe, "ltp", None) or 0.0
        if isinstance(pe, dict):
            bid = pe.get("bid", 0.0) or 0.0
            ask = pe.get("ask", 0.0) or 0.0
            ltp = pe.get("ltp", 0.0) or 0.0
        mid = _midpoint(bid, ask, ltp)
        if bid <= 0:
            consec_zero_lower += 1
            if consec_zero_lower >= 2:
                break
            continue
        consec_zero_lower = 0
        result.append(_StrikeData(K=K, Q=mid, bid=bid, is_call=False))

    # ── Collect strikes from K₀ upward (calls) ────────────────────────────
    above = [K for K in sorted(all_strikes) if K >= K0]
    for K in above:
        if K == K0:
            # Average of call and put at K₀
            ce = _get_quote(K, "CE")
            pe = _get_quote(K, "PE")
            bids_c = asks_c = ltps_c = 0.0
            bids_p = asks_p = ltps_p = 0.0
            if ce:
                bids_c = getattr(ce, "bid", 0.0) or 0.0
                asks_c = getattr(ce, "ask", 0.0) or 0.0
                ltps_c = getattr(ce, "ltp", 0.0) or 0.0
            if pe:
                bids_p = getattr(pe, "bid", 0.0) or 0.0
                asks_p = getattr(pe, "ask", 0.0) or 0.0
                ltps_p = getattr(pe, "ltp", 0.0) or 0.0
            mid_c = _midpoint(bids_c, asks_c, ltps_c)
            mid_p = _midpoint(bids_p, asks_p, ltps_p)
            Q = (mid_c + mid_p) / 2.0
            bid = max(bids_c, bids_p)
            if bid > 0 and Q > 0:
                result.append(_StrikeData(K=K, Q=Q, bid=bid, is_call=True))
            continue

        ce = _get_quote(K, "CE")
        if ce is None:
            consec_zero_upper += 1
            if consec_zero_upper >= 2:
                break
            continue
        bid = getattr(ce, "bid", None) or 0.0
        ask = getattr(ce, "ask", None) or 0.0
        ltp = getattr(ce, "ltp", None) or 0.0
        if isinstance(ce, dict):
            bid = ce.get("bid", 0.0) or 0.0
            ask = ce.get("ask", 0.0) or 0.0
            ltp = ce.get("ltp", 0.0) or 0.0
        mid = _midpoint(bid, ask, ltp)
        if bid <= 0:
            consec_zero_upper += 1
            if consec_zero_upper >= 2:
                break
            continue
        consec_zero_upper = 0
        result.append(_StrikeData(K=K, Q=mid, bid=bid, is_call=True))

    return result


def _implied_forward(
    snapshot: Any,
    expiry: date,
    T: float,
    r: float,
    all_strikes: list[float],
) -> tuple[float, float]:
    """
    Compute the implied forward price F and K₀ for a given expiry.

    Method: Find the strike K* with the smallest |Call(K*) - Put(K*)|,
    then:
        F = K* + e^(rT) * (Call(K*) - Put(K*))
        K₀ = first strike at or below F

    Returns (F, K0).  On failure, returns (S_proxy, K0_approx) where
    S_proxy is estimated from ATM call/put average.
    """
    def _get_quote(K: float, opt_type: str):
        if hasattr(snapshot, "get_quote"):
            return snapshot.get_quote(expiry, K, opt_type)
        elif isinstance(snapshot, dict):
            return snapshot.get("quotes", {}).get((expiry, K, opt_type))
        return None

    best_K_star = None
    best_diff = float("inf")

    for K in all_strikes:
        ce = _get_quote(K, "CE")
        pe = _get_quote(K, "PE")
        if ce is None or pe is None:
            continue
        c_bid = getattr(ce, "bid", 0.0) or 0.0
        c_ask = getattr(ce, "ask", 0.0) or 0.0
        p_bid = getattr(pe, "bid", 0.0) or 0.0
        p_ask = getattr(pe, "ask", 0.0) or 0.0
        if c_bid <= 0 or p_bid <= 0:
            continue
        c_mid = (c_bid + c_ask) / 2.0
        p_mid = (p_bid + p_ask) / 2.0
        diff = abs(c_mid - p_mid)
        if diff < best_diff:
            best_diff = diff
            best_K_star = K
            best_c_mid = c_mid
            best_p_mid = p_mid

    if best_K_star is None:
        # Fallback: use underlying price if available
        S = getattr(snapshot, "underlying_price", 0.0) or 0.0
        if S <= 0:
            return 0.0, 0.0
        K0 = max((K for K in all_strikes if K <= S), default=all_strikes[0])
        return S, K0

    F = best_K_star + math.exp(r * T) * (best_c_mid - best_p_mid)
    # K0 = first strike at or below F
    candidates = [K for K in all_strikes if K <= F]
    K0 = max(candidates) if candidates else all_strikes[0]
    return F, K0


def estimate_india_vix(
    snapshot: Any,
    as_of_date: date,
    r: float = 0.065,
    near_expiry: date | None = None,
    next_expiry: date | None = None,
) -> VIXEstimate:
    """
    Estimate the India VIX from a Nifty option chain snapshot.

    Uses the CBOE VIX methodology: CBOE (2009) white paper.

    Parameters
    ----------
    snapshot : ChainSnapshot or dict
        Current option chain.  Must support `strikes_by_expiry(expiry)` and
        `get_quote(expiry, strike, option_type)`.
    as_of_date : date
        Date of the snapshot (used to compute T = calendar_days / 365).
    r : float
        Risk-free rate (RBI repo rate proxy).  Default 6.5% [ASSUMED].
    near_expiry : date | None
        Near-term expiry.  If None, selected as the first Nifty expiry
        with T₁ < 30 calendar days from as_of_date.
    next_expiry : date | None
        Next-term expiry.  If None, selected as the first expiry with
        T₂ > 30 calendar days.

    Returns
    -------
    VIXEstimate
        Contains the VIX value and quality metadata.
        Returns VIX = -1.0 on fatal failures (no chain data, zero valid strikes).

    Notes
    -----
    [APPROX-VIXMODEL]: The NSE India VIX is computed by NSE using real-time
    data and official rate curves.  This approximation uses the same CBOE
    formula but with:
      - Risk-free rate r [ASSUMED] = 6.5% flat term structure
      - Chain snapshot at the moment of this call (not end-of-day settling)
    """

    # ── Get all expiries in the snapshot ──────────────────────────────────
    if hasattr(snapshot, "all_expiries"):
        all_expiries = snapshot.all_expiries()
    elif isinstance(snapshot, dict):
        quotes = snapshot.get("quotes", {})
        all_expiries = sorted({e for (e, _, _) in quotes})
    else:
        return VIXEstimate(
            vix=-1.0, near_term_vix=0.0, next_term_vix=0.0,
            near_expiry=as_of_date, next_expiry=None,
            n_near_strikes=0, n_next_strikes=0, is_approx=True,
            warning="No option chain data in snapshot",
        )

    if not all_expiries:
        return VIXEstimate(
            vix=-1.0, near_term_vix=0.0, next_term_vix=0.0,
            near_expiry=as_of_date, next_expiry=None,
            n_near_strikes=0, n_next_strikes=0, is_approx=True,
            warning="Empty expiry list in snapshot",
        )

    # ── Select near-term and next-term expiries ───────────────────────────
    future_expiries = [e for e in all_expiries if e > as_of_date]
    if not future_expiries:
        return VIXEstimate(
            vix=-1.0, near_term_vix=0.0, next_term_vix=0.0,
            near_expiry=as_of_date, next_expiry=None,
            n_near_strikes=0, n_next_strikes=0, is_approx=True,
            warning="No future expiries in snapshot",
        )

    if near_expiry is None:
        # Choose first expiry < 30 days if available, else the nearest one
        lt30 = [e for e in future_expiries if (e - as_of_date).days < 30]
        near_expiry = lt30[0] if lt30 else future_expiries[0]

    if next_expiry is None:
        gt30 = [
            e for e in future_expiries
            if (e - as_of_date).days > 30 and e != near_expiry
        ]
        next_expiry = gt30[0] if gt30 else None

    # ── Compute T₁, T₂ (calendar days / 365) ─────────────────────────────
    T1_days = (near_expiry - as_of_date).days
    T1 = max(T1_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)
    T2 = None
    if next_expiry:
        T2_days = (next_expiry - as_of_date).days
        T2 = max(T2_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR)

    # ── Process near-term expiry ──────────────────────────────────────────
    near_strikes = sorted(
        (hasattr(snapshot, "strikes_by_expiry") and snapshot.strikes_by_expiry(near_expiry))
        or (isinstance(snapshot, dict) and sorted({s for (e, s, _) in snapshot.get("quotes", {}) if e == near_expiry}))
        or []
    )
    if not near_strikes:
        return VIXEstimate(
            vix=-1.0, near_term_vix=0.0, next_term_vix=0.0,
            near_expiry=near_expiry, next_expiry=next_expiry,
            n_near_strikes=0, n_next_strikes=0, is_approx=True,
            warning=f"No strikes for near expiry {near_expiry}",
        )

    F1, K01 = _implied_forward(snapshot, near_expiry, T1, r, near_strikes)
    if F1 <= 0:
        return VIXEstimate(
            vix=-1.0, near_term_vix=0.0, next_term_vix=0.0,
            near_expiry=near_expiry, next_expiry=next_expiry,
            n_near_strikes=0, n_next_strikes=0, is_approx=True,
            warning=f"Could not compute forward price for near expiry {near_expiry}",
        )

    near_strike_data = _select_otm_strikes(snapshot, near_expiry, F1, K01)
    sigma1_sq, n_near = _compute_vix_sq_for_expiry(near_strike_data, T1, r, F1, K01)

    near_vix_pct = 100.0 * math.sqrt(sigma1_sq) if sigma1_sq > 0 else 0.0

    # ── Process next-term expiry (optional) ──────────────────────────────
    sigma2_sq = 0.0
    n_next = 0
    next_vix_pct = 0.0

    if next_expiry and T2 is not None:
        next_strikes = sorted(
            (hasattr(snapshot, "strikes_by_expiry") and snapshot.strikes_by_expiry(next_expiry))
            or (isinstance(snapshot, dict) and sorted({s for (e, s, _) in snapshot.get("quotes", {}) if e == next_expiry}))
            or []
        )
        if next_strikes:
            F2, K02 = _implied_forward(snapshot, next_expiry, T2, r, next_strikes)
            if F2 > 0:
                next_strike_data = _select_otm_strikes(snapshot, next_expiry, F2, K02)
                sigma2_sq, n_next = _compute_vix_sq_for_expiry(
                    next_strike_data, T2, r, F2, K02
                )
                next_vix_pct = 100.0 * math.sqrt(sigma2_sq) if sigma2_sq > 0 else 0.0

    # ── 30-day interpolation ──────────────────────────────────────────────
    if T2 is not None and sigma2_sq > 0 and T1_days < 30:
        N_T1 = T1_days * _MINUTES_PER_DAY
        N_T2 = (next_expiry - as_of_date).days * _MINUTES_PER_DAY  # type: ignore

        if N_T2 > N_T1:
            w1 = (N_T2 - _N30) / (N_T2 - N_T1)
            w2 = (_N30 - N_T1) / (N_T2 - N_T1)
            vix_sq = (N_T1 * sigma1_sq * w1 + N_T2 * sigma2_sq * w2) * (_N365 / _N30)
            vix_sq = max(vix_sq, 0.0)
            vix = 100.0 * math.sqrt(vix_sq)
        else:
            # Degenerate case: both expiries on the same day
            vix = near_vix_pct
    else:
        # Only one expiry or near-term >= 30d: use near-term directly
        vix = near_vix_pct

    is_approx = (n_near < _MIN_VALID_STRIKES) or (near_vix_pct <= 0)
    warning = ""
    if is_approx:
        warning = (
            f"[APPROX-VIXMODEL] Only {n_near} valid near-term strikes. "
            f"VIX estimate is unreliable."
        )
    if vix <= 0:
        warning += " VIX computed as 0 or negative — insufficient chain data."

    log.info(
        "vix_estimator.result",
        vix=round(vix, 4),
        near=str(near_expiry),
        n_near=n_near,
        n_next=n_next,
        is_approx=is_approx,
    )

    return VIXEstimate(
        vix=vix,
        near_term_vix=near_vix_pct,
        next_term_vix=next_vix_pct,
        near_expiry=near_expiry,
        next_expiry=next_expiry,
        n_near_strikes=n_near,
        n_next_strikes=n_next,
        is_approx=is_approx,
        warning=warning,
    )
