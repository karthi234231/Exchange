"""Strike-price utilities for NSE F&O contracts.

NSE lists options at fixed strike intervals for each index.  This module
provides pure, stateless functions to enumerate valid strikes, identify
ATM, and classify moneyness.

Verified NSE index strike intervals (as of 2026-06-19)
-------------------------------------------------------
  NIFTY 50    : 50 points   [P] NSE F&O contract specification
  BANK NIFTY  : 100 points  [P] NSE F&O contract specification

These values are **not** hard-coded inside the functions below — callers
always supply ``strike_interval`` explicitly so the module remains
instrument-agnostic.  Common values live in
``algo_trading.fo.instrument_master.STRIKE_INTERVALS`` for convenience.

Moneyness categories
--------------------
The thresholds below follow common NSE market convention:

  DEEP_ITM   : |moneyness| > 5 strike intervals from ATM
  ITM        : within 1–5 strike intervals of ATM (ITM side)
  ATM        : within 0.5 strike intervals of the ATM strike
  OTM        : 1–5 strike intervals OTM
  DEEP_OTM   : > 5 strike intervals OTM
"""

from __future__ import annotations

import math
import structlog
from typing import Literal

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical strike intervals — add new instruments here.
# [P] NSE F&O contract specifications.
# ---------------------------------------------------------------------------
STRIKE_INTERVALS: dict[str, float] = {
    "NIFTY": 50.0,       # [P] NSE F&O contract spec for NIFTY 50
    "BANKNIFTY": 100.0,  # [P] NSE F&O contract spec for BANK NIFTY
}

# Moneyness literal type for static checking
MoneynessCategory = Literal["DEEP_ITM", "ITM", "ATM", "OTM", "DEEP_OTM"]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def atm_strike(underlying: float, strike_interval: float) -> float:
    """Return the at-the-money (ATM) strike nearest to *underlying*.

    ATM is defined as ``round(underlying / interval) * interval``, using
    standard half-round-up (Python's ``round()`` uses banker's rounding;
    this function uses ``math.floor(x + 0.5)`` for consistent behaviour).

    Args:
        underlying: Current spot price of the index or stock.
        strike_interval: Distance between consecutive valid strikes
            (e.g. 50 for NIFTY, 100 for BANKNIFTY).

    Returns:
        Nearest valid strike as a float.

    Raises:
        ValueError: If *strike_interval* is not positive.

    Examples::

        >>> atm_strike(24350.0, 50)
        24350.0
        >>> atm_strike(24375.0, 50)
        24400.0   # rounds up at exact midpoint
        >>> atm_strike(24349.9, 50)
        24350.0
    """
    if strike_interval <= 0:
        raise ValueError(
            f"strike_interval must be positive, got {strike_interval!r}"
        )
    # Use floor(x + 0.5) for unambiguous midpoint behaviour.
    atm = math.floor(underlying / strike_interval + 0.5) * strike_interval
    log.debug(
        "atm_strike_computed",
        underlying=underlying,
        strike_interval=strike_interval,
        atm=atm,
    )
    return float(atm)


def generate_strikes(
    underlying: float = 0.0,
    strike_interval: float = 0.0,
    n_strikes_each_side: int = 20,
    underlying_price: float = 0.0,
) -> list[float]:
    """Return a sorted list of valid NSE strikes centred around ATM.

    Generates ``2 * n_strikes_each_side + 1`` strikes: *n* below ATM,
    the ATM strike itself, and *n* above ATM.  All strikes are multiples
    of *strike_interval*.

    Args:
        underlying: Current spot price of the index or stock.
        strike_interval: Distance between consecutive valid strikes.
        n_strikes_each_side: Number of strikes to generate above *and*
            below ATM (default 20, giving a total of 41 strikes).  NSE
            lists a broader range but the liquid range is well within ±20
            intervals.

    Returns:
        Sorted (ascending) list of strike prices as floats.

    Raises:
        ValueError: If *strike_interval* is not positive, or
            *n_strikes_each_side* is negative.

    Examples::

        >>> strikes = generate_strikes(24350.0, 50, n_strikes_each_side=2)
        >>> strikes
        [24250.0, 24300.0, 24350.0, 24400.0, 24450.0]
    """
    # Accept underlying_price as an alias for underlying (backward-compatible)
    if underlying == 0.0 and underlying_price != 0.0:
        underlying = underlying_price

    if strike_interval <= 0:
        raise ValueError(
            f"strike_interval must be positive, got {strike_interval!r}"
        )
    if n_strikes_each_side < 0:
        raise ValueError(
            f"n_strikes_each_side must be non-negative, got {n_strikes_each_side!r}"
        )

    atm = atm_strike(underlying, strike_interval)
    strikes = [
        atm + i * strike_interval
        for i in range(-n_strikes_each_side, n_strikes_each_side + 1)
    ]
    log.debug(
        "strikes_generated",
        underlying=underlying,
        strike_interval=strike_interval,
        atm=atm,
        n_strikes_each_side=n_strikes_each_side,
        total_strikes=len(strikes),
        lowest=strikes[0],
        highest=strikes[-1],
    )
    return strikes


def get_moneyness_category(
    underlying: float,
    strike: float,
    option_type: str,
    strike_interval: float,
) -> MoneynessCategory:
    """Classify a strike as DEEP_ITM, ITM, ATM, OTM, or DEEP_OTM.

    Thresholds (in multiples of *strike_interval* from ATM):

    +-----------+----------------------------+
    | Category  | Distance from ATM          |
    +===========+============================+
    | ATM       | <= 0.5 × strike_interval   |
    | ITM/OTM   | 0.5 – 5 × strike_interval  |
    | DEEP_*    | > 5 × strike_interval      |
    +-----------+----------------------------+

    Args:
        strike: The option strike price to classify.
        underlying: Current spot price of the index or stock.
        option_type: ``"CE"`` (call) or ``"PE"`` (put).  Case-insensitive.
        strike_interval: Distance between consecutive valid strikes, used
            to define ATM and moneyness thresholds.

    Returns:
        One of ``"DEEP_ITM"``, ``"ITM"``, ``"ATM"``, ``"OTM"``,
        ``"DEEP_OTM"``.

    Raises:
        ValueError: If *option_type* is not ``"CE"`` or ``"PE"``, or if
            *strike_interval* is not positive.

    Examples::

        >>> get_moneyness_category(24350.0, 24350.0, "CE", 50)
        'ATM'
        >>> get_moneyness_category(24200.0, 24350.0, "CE", 50)  # 3 intervals below
        'OTM'
        >>> get_moneyness_category(24600.0, 24350.0, "CE", 50)  # 5 intervals above
        'ITM'
    """
    if strike_interval <= 0:
        raise ValueError(
            f"strike_interval must be positive, got {strike_interval!r}"
        )
    upper_type = option_type.upper()
    if upper_type not in {"CE", "PE"}:
        raise ValueError(
            f"option_type must be 'CE' or 'PE', got {option_type!r}"
        )

    atm = atm_strike(underlying, strike_interval)
    distance = abs(strike - atm)

    # ATM band: within half a strike interval of the ATM strike.
    if distance <= 0.5 * strike_interval:
        category: MoneynessCategory = "ATM"
    else:
        # Determine whether this strike is ITM or OTM for the given option type.
        if upper_type == "CE":
            # Call: ITM if strike < underlying (strike below spot)
            is_itm = strike < underlying
        else:
            # Put: ITM if strike > underlying (strike above spot)
            is_itm = strike > underlying

        if distance > 5 * strike_interval:
            category = "DEEP_ITM" if is_itm else "DEEP_OTM"
        else:
            category = "ITM" if is_itm else "OTM"

    log.debug(
        "moneyness_classified",
        strike=strike,
        underlying=underlying,
        option_type=upper_type,
        atm=atm,
        distance=distance,
        category=category,
    )
    return category


def otm_strikes(
    underlying: float,
    strike_interval: float,
    option_type: str,
    n: int,
) -> list[float]:
    """Return the *n* nearest OTM strikes for a CE or PE option.

    OTM strikes for a CE are **above** the ATM strike; OTM strikes for a PE
    are **below** the ATM strike.  The ATM strike itself is excluded.

    Args:
        underlying: Current spot price of the index or stock.
        strike_interval: Distance between consecutive valid strikes.
        option_type: ``"CE"`` (call) or ``"PE"`` (put).  Case-insensitive.
        n: Number of OTM strikes to return.  Must be positive.

    Returns:
        List of *n* OTM strike prices, ordered from nearest-to-ATM to
        furthest-from-ATM.

        - For CE: ascending order (24400, 24450, 24500, …)
        - For PE: descending order (24300, 24250, 24200, …)

    Raises:
        ValueError: If *option_type* is not ``"CE"`` or ``"PE"``, or if
            *n* is not positive, or if *strike_interval* is not positive.

    Examples::

        >>> otm_strikes(24350.0, 50, "CE", 3)
        [24400.0, 24450.0, 24500.0]
        >>> otm_strikes(24350.0, 50, "PE", 3)
        [24300.0, 24250.0, 24200.0]
    """
    if strike_interval <= 0:
        raise ValueError(
            f"strike_interval must be positive, got {strike_interval!r}"
        )
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    upper_type = option_type.upper()
    if upper_type not in {"CE", "PE"}:
        raise ValueError(
            f"option_type must be 'CE' or 'PE', got {option_type!r}"
        )

    atm = atm_strike(underlying, strike_interval)

    if upper_type == "CE":
        # OTM calls are above ATM
        strikes = [atm + i * strike_interval for i in range(1, n + 1)]
    else:
        # OTM puts are below ATM
        strikes = [atm - i * strike_interval for i in range(1, n + 1)]

    log.debug(
        "otm_strikes_generated",
        underlying=underlying,
        strike_interval=strike_interval,
        option_type=upper_type,
        atm=atm,
        n=n,
        strikes=strikes,
    )
    return strikes
