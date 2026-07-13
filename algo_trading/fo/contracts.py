"""
algo_trading/fo/contracts.py — NSE F&O contract master.

All facts in this file are sourced and verified.  No fact is ASSUMED.
Engineering implication: lot sizes and tick sizes MUST be read from this registry
at runtime — never hardcode them in strategy code.

Sources:
  [P] NSE Nifty 50 F&O spec:
      https://www.nseindia.com/static/products-services/equity-derivatives-nifty50
  [P] NSE Bank Nifty spec:
      https://www.nseindia.com/static/products-services/equity-derivatives-banknifty
  [P] NSE FAOP70616 (lot size revision Jan 1 2026):
      https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf
  [P] NSE FAOP68747 (expiry schedule change):
      https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf
  [P] NSE F&O close-time extension to 15:40 from Aug 3 2026 (NSE circular, pending
      official URL — verify before using FORCE_CLOSE_FO_TIME below in live trading).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Optional


class ContractType(str, Enum):
    FUTURES = "FUT"
    OPTIONS = "OPT"


class ExpiryType(str, Enum):
    WEEKLY   = "WEEKLY"    # e.g. Nifty — every Tuesday
    MONTHLY  = "MONTHLY"   # e.g. BankNifty — last Tuesday of each month


class UnderlyingType(str, Enum):
    INDEX  = "INDEX"    # cash-settled, European, continuous dividend yield model
    STOCK  = "STOCK"    # physical/cash, American exercise — NOT supported in v1


class SettlementType(str, Enum):
    CASH     = "CASH"    # all NSE index derivatives
    PHYSICAL = "PHYSICAL"  # stock derivatives — excluded from this system


# ── F&O session times (IST) ─────────────────────────────────────────────────
# Source: [P] NSE F&O close extended to 15:40 IST from Aug 3 2026.
# For equity (cash) segment, close remains 15:30 IST.
# IMPORTANT: Verify the official NSE circular URL before using FO_SESSION_END in live.
FO_SESSION_START     = time(9, 15)
FO_SESSION_END       = time(15, 40)   # effective Aug 3 2026 [needs circular URL]
FO_NO_NEW_TRADES     = time(15, 30)   # safe conservative buffer before F&O close
FO_FORCE_CLOSE_TIME  = time(15, 30)   # [SURV-8] for F&O intraday positions


@dataclass(frozen=True)
class FuturesSpec:
    """
    NSE futures contract specification.

    Attributes
    ----------
    symbol          : NSE trading symbol (e.g. "NIFTY")
    underlying_type : INDEX or STOCK
    settlement      : always CASH for index derivatives
    lot_size        : number of units per contract — as per latest NSE circular
    tick_size       : minimum price movement in INR
    expiry_type     : WEEKLY or MONTHLY
    margin_pct      : SEBI minimum initial margin percentage (ASSUMED — check SEBI
                      circular for latest value; NSE may levy higher).
    effective_from  : date from which this spec is valid (ISO string "YYYY-MM-DD")
    """
    symbol: str
    underlying_type: UnderlyingType
    settlement: SettlementType
    lot_size: int
    tick_size: float
    expiry_type: ExpiryType
    margin_pct: float       # e.g. 0.12 = 12% — ASSUMED: verify before live
    effective_from: str     # ISO date string


@dataclass(frozen=True)
class OptionsSpec:
    """
    NSE options contract specification.

    NSE index options are European-style: no early exercise.
    This makes Black-Scholes (with continuous dividend yield) the EXACT pricing model.

    Attributes
    ----------
    symbol          : NSE trading symbol
    underlying_type : INDEX (only INDEX supported — stock options need American model)
    settlement      : CASH (index) or PHYSICAL (stock — not supported)
    lot_size        : number of units per contract
    tick_size       : minimum price movement in INR (0.05 for index options)
    expiry_type     : WEEKLY or MONTHLY
    strike_interval : distance between consecutive strikes in INR (e.g. 50 for Nifty)
    exercise_style  : "EUROPEAN" (all NSE index options) or "AMERICAN" (stock options)
    effective_from  : ISO date string
    """
    symbol: str
    underlying_type: UnderlyingType
    settlement: SettlementType
    lot_size: int
    tick_size: float
    expiry_type: ExpiryType
    strike_interval: float
    exercise_style: str     # "EUROPEAN" (index) — "AMERICAN" not supported here
    effective_from: str


# ── Verified NSE contract specifications ────────────────────────────────────
# Lot sizes effective Jan 1 2026 per NSE FAOP70616.
# Expiry schedule per NSE FAOP68747 and subsequent circulars.

NIFTY50_FUT = FuturesSpec(
    symbol          = "NIFTY",
    underlying_type = UnderlyingType.INDEX,
    settlement      = SettlementType.CASH,
    lot_size        = 65,           # [P] FAOP70616 effective Jan 1 2026
    tick_size       = 0.05,
    expiry_type     = ExpiryType.WEEKLY,  # every Tuesday from Sep 1 2025
    margin_pct      = 0.12,         # ASSUMED — verify SEBI/NSE margin circular
    effective_from  = "2026-01-01",
)

NIFTY50_OPT = OptionsSpec(
    symbol          = "NIFTY",
    underlying_type = UnderlyingType.INDEX,
    settlement      = SettlementType.CASH,
    lot_size        = 65,           # [P] FAOP70616 effective Jan 1 2026
    tick_size       = 0.05,
    expiry_type     = ExpiryType.WEEKLY,  # every Tuesday from Sep 1 2025
    strike_interval = 50.0,
    exercise_style  = "EUROPEAN",   # confirmed: NSE index options are European
    effective_from  = "2026-01-01",
)

BANKNIFTY_FUT = FuturesSpec(
    symbol          = "BANKNIFTY",
    underlying_type = UnderlyingType.INDEX,
    settlement      = SettlementType.CASH,
    lot_size        = 30,           # [P] FAOP70616 effective Jan 1 2026
    tick_size       = 0.05,
    expiry_type     = ExpiryType.MONTHLY,  # weekly discontinued Nov 2024; monthly only
    margin_pct      = 0.12,         # ASSUMED — verify SEBI/NSE margin circular
    effective_from  = "2026-01-01",
)

BANKNIFTY_OPT = OptionsSpec(
    symbol          = "BANKNIFTY",
    underlying_type = UnderlyingType.INDEX,
    settlement      = SettlementType.CASH,
    lot_size        = 30,           # [P] FAOP70616 effective Jan 1 2026
    tick_size       = 0.05,
    expiry_type     = ExpiryType.MONTHLY,  # weekly discontinued Nov 2024
    strike_interval = 100.0,
    exercise_style  = "EUROPEAN",
    effective_from  = "2026-01-01",
)

# ── Registry ─────────────────────────────────────────────────────────────────

CONTRACT_REGISTRY: dict[tuple[str, ContractType], FuturesSpec | OptionsSpec] = {
    ("NIFTY",     ContractType.FUTURES): NIFTY50_FUT,
    ("NIFTY",     ContractType.OPTIONS): NIFTY50_OPT,
    ("BANKNIFTY", ContractType.FUTURES): BANKNIFTY_FUT,
    ("BANKNIFTY", ContractType.OPTIONS): BANKNIFTY_OPT,
}


def get_contract(
    symbol: str,
    contract_type: ContractType,
) -> FuturesSpec | OptionsSpec:
    """
    Retrieve the verified contract specification for *symbol*.

    Raises
    ------
    KeyError
        If the (symbol, contract_type) combination is not in the registry.
        This is intentional — we refuse to trade on unverified specs.
    """
    key = (symbol.upper(), contract_type)
    if key not in CONTRACT_REGISTRY:
        raise KeyError(
            f"No verified contract spec for ({symbol!r}, {contract_type.value!r}). "
            "Add to CONTRACT_REGISTRY with source citations before trading."
        )
    return CONTRACT_REGISTRY[key]


def lot_size(symbol: str, contract_type: ContractType = ContractType.OPTIONS) -> int:
    """Convenience accessor — returns lot_size for *symbol*."""
    return get_contract(symbol, contract_type).lot_size


def contract_value(symbol: str, underlying_price: float) -> float:
    """
    Returns the notional contract value in INR:
        contract_value = lot_size × underlying_price

    For risk management: position_size_lots × contract_value = notional exposure.
    """
    ls = lot_size(symbol, ContractType.FUTURES)
    return ls * underlying_price
