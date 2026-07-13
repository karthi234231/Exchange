"""instrument_master — date-aware lot sizes and strike utilities for NSE F&O.

Public API
----------
LotSizeRecord        : Immutable record of a single lot-size epoch.
LotSizeHistory       : Registry that resolves lot sizes by date.
LotSizeNotFoundError : Raised when no record covers a requested date.
get_lot_size         : Convenience wrapper — resolves lot size for a symbol/date.
generate_strikes     : Generate sorted list of valid NSE strikes around ATM.
atm_strike           : Nearest valid strike for a given underlying price.
get_moneyness_category : Classify a strike as DEEP_ITM / ITM / ATM / OTM / DEEP_OTM.
otm_strikes          : Return n OTM strikes for a CE or PE option.
"""

from __future__ import annotations

from algo_trading.fo.instrument_master.lot_size_history import (
    LotSizeHistory,
    LotSizeNotFoundError,
    LotSizeRecord,
    get_lot_size,
)
from algo_trading.fo.instrument_master.strike_generator import (
    atm_strike,
    generate_strikes,
    get_moneyness_category,
    otm_strikes,
)

__all__ = [
    # lot size
    "LotSizeRecord",
    "LotSizeHistory",
    "LotSizeNotFoundError",
    "get_lot_size",
    # strikes
    "generate_strikes",
    "atm_strike",
    "get_moneyness_category",
    "otm_strikes",
]
