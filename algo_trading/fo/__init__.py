"""
algo_trading/fo — NSE Futures & Options system package.

Sub-modules (import order respects dependencies):
  contracts       — NSE contract master: verified lot sizes, tick sizes, expiry rules
  expiry          — F&O expiry calendar (Nifty Tuesday, BankNifty monthly last Tuesday)
  pricing         — Black-Scholes-Merton with continuous dividend yield, all Greeks,
                    robust Newton-Raphson + bisection IV solver
  cost_model      — F&O transaction-cost model (NSE April 2026 STT rates)
  vol_surface     — SVI parametric implied-volatility surface (Gatheral 2004) + VolSurfaceRouter
  strategies      — Multi-leg strategy definitions and at-expiry payoff
  portfolio_risk  — Portfolio Greek aggregation, VRP signal, delta/vega/gamma limits
  margin          — SPAN-like initial margin approximation [APPROX-NOT-OFFICIAL-SPAN]
  execution       — Multi-leg execution with legging risk and stuck-leg recovery
  interfaces      — Abstract base classes (SnapshotStore, BrokerGateway, etc.)
  instrument_master — Date-aware lot sizes, strike generation, moneyness classification
"""
from algo_trading.fo.contracts import (
    ContractType,
    ExpiryType,
    FuturesSpec,
    OptionsSpec,
    NIFTY50_FUT,
    NIFTY50_OPT,
    BANKNIFTY_FUT,
    BANKNIFTY_OPT,
    CONTRACT_REGISTRY,
    get_contract,
)
from algo_trading.fo.expiry import (
    ExpiryCalendar,
    nifty_expiry_calendar,
    banknifty_expiry_calendar,
)
from algo_trading.fo.pricing import (
    BSMEngine,
    OptionGreeks,
    HigherOrderGreeks,
    IVSolverResult,
)
from algo_trading.fo.cost_model import (
    FOCostModel,
    FOTradeCost,
    FOInstrumentType,
)
from algo_trading.fo.interfaces import (
    SnapshotStore,
    BrokerGateway,
    HistoricalDataProvider,
    ExecutionVenueAdapter,
    MarginModelProvider,
)
from algo_trading.fo.instrument_master import (
    LotSizeRecord,
    LotSizeHistory,
    LotSizeNotFoundError,
    get_lot_size,
    generate_strikes,
    atm_strike,
    get_moneyness_category,
    otm_strikes,
)
from algo_trading.fo.vix_estimator import (
    VIXEstimate,
    estimate_india_vix,
)
from algo_trading.fo.sabr import (
    SABRParams,
    SABRSmile,
    SABRCalibrationResult,
    sabr_implied_vol,
    fit_sabr,
)
from algo_trading.fo.vol_surface import (
    SVIParams,
    SVISmile,
    SVISurface,
    VolSurfaceRouter,
    fit_svi,
    realized_volatility,
)

__all__ = [
    # contracts
    "ContractType", "ExpiryType", "FuturesSpec", "OptionsSpec",
    "NIFTY50_FUT", "NIFTY50_OPT", "BANKNIFTY_FUT", "BANKNIFTY_OPT",
    "CONTRACT_REGISTRY", "get_contract",
    # expiry
    "ExpiryCalendar", "nifty_expiry_calendar", "banknifty_expiry_calendar",
    # pricing
    "BSMEngine", "OptionGreeks", "HigherOrderGreeks", "IVSolverResult",
    # cost_model
    "FOCostModel", "FOTradeCost", "FOInstrumentType",
    # interfaces
    "SnapshotStore", "BrokerGateway", "HistoricalDataProvider",
    "ExecutionVenueAdapter", "MarginModelProvider",
    # instrument_master
    "LotSizeRecord", "LotSizeHistory", "LotSizeNotFoundError", "get_lot_size",
    "generate_strikes", "atm_strike", "get_moneyness_category", "otm_strikes",
    # vix_estimator
    "VIXEstimate", "estimate_india_vix",
    # sabr
    "SABRParams", "SABRSmile", "SABRCalibrationResult",
    "sabr_implied_vol", "fit_sabr",
    # vol_surface
    "SVIParams", "SVISmile", "SVISurface", "VolSurfaceRouter",
    "fit_svi", "realized_volatility",
]
