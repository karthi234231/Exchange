"""
algo_trading/fo/cost_model.py — NSE F&O transaction-cost model.

All tax rates are sourced and dated.  No rate is ASSUMED.

NSE F&O cost structure (effective April 1 2026, Union Budget 2026 revisions):
  STT:
    Futures  — sell side only — 0.05% of traded value (turnover = qty × price)
    Options  — sell side (premium) — 0.15% of premium value
    Options  — exercise/assignment — 0.15% of intrinsic value at settlement
  NSE Exchange Transaction Charge:
    Futures  — 0.0019% of turnover  (₹190 per crore)
    Options  — 0.0500% of premium   (₹5,000 per crore)
  SEBI Turnover Fees: ₹10 per crore (0.00001%) — both segments
  GST: 18% on (exchange charge + SEBI fee)  — NOT on STT or stamp duty
  Stamp Duty: 0.002% on buy-side premium / turnover  (Maharashtra; varies by state)
  Brokerage: ₹0 (Shoonya zero-brokerage platform)

Sources:
  [P] NSE STT page: https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax
  [P] NSE Clearing STT: https://www.nseclearing.in/clearing-settlement/equity-derivatives/securities-transaction-tax
  [S] ClearTax STT summary: https://cleartax.in/s/securities-transaction-tax-stt
  [S] 1finance Budget 2026 analysis: https://1finance.co.in/blog/stt-futures-options-increased-budget-2026-for-fno-investors/
  [P] NSE Exchange Transaction Charges (F&O):
      https://www.nseindia.com/static/products-services/equity-derivatives-transaction-charges

IMPORTANT: Stamp duty rate used here (0.002%) is for Maharashtra (NSE primary).
Other state-domiciled accounts may have different stamp duty.  Verify for your state.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class FOInstrumentType(str, Enum):
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


# ── Verified F&O Tax Rates (effective April 1 2026) ──────────────────────────

# STT
_STT_FUT_SELL_PCT     = Decimal("0.0005")    # 0.05% on sell-side turnover [P]
_STT_OPT_SELL_PCT     = Decimal("0.0015")    # 0.15% on premium (sell side) [P]
_STT_OPT_EXERCISE_PCT = Decimal("0.0015")    # 0.15% on intrinsic value at exercise [P]

# NSE Exchange Transaction Charges
_EXCHANGE_FUT_PCT     = Decimal("0.0000190") # 0.0019% of turnover [P]
_EXCHANGE_OPT_PCT     = Decimal("0.0005000") # 0.0500% of premium [P]

# SEBI fee
_SEBI_FEE_PER_CRORE   = Decimal("10")        # ₹10 per crore = 0.00001%

# GST on exchange charges + SEBI fee (NOT on STT or stamp duty)
_GST_PCT              = Decimal("0.18")

# Stamp duty (Maharashtra, buy-side only)
_STAMP_DUTY_PCT       = Decimal("0.00002")   # 0.002% on buy-side

# Brokerage
_BROKERAGE            = Decimal("0")          # Shoonya zero-brokerage


@dataclass(frozen=True)
class FOTradeCost:
    """
    Itemised cost breakdown for a single F&O trade leg.

    All monetary values are Decimal (INR).
    """
    instrument_type: FOInstrumentType
    side: str                # "BUY" or "SELL"
    brokerage:   Decimal
    stt:         Decimal
    exchange:    Decimal
    sebi_fee:    Decimal
    gst:         Decimal
    stamp_duty:  Decimal
    slippage:    Decimal     # market-impact estimate (see below)
    total:       Decimal     # sum of all components
    turnover:    Decimal     # notional traded (qty × price_per_unit × lot_size)
    premium:     Decimal     # premium value (for options: qty × ltp × lot_size)

    def as_bps_of_premium(self) -> Decimal:
        """Total cost as basis points of premium value (relevant for options)."""
        if self.premium == 0:
            return Decimal(0)
        return (self.total / self.premium * 10_000).quantize(Decimal("0.01"))

    def as_bps_of_turnover(self) -> Decimal:
        """Total cost as basis points of notional turnover."""
        if self.turnover == 0:
            return Decimal(0)
        return (self.total / self.turnover * 10_000).quantize(Decimal("0.01"))

    def round_trip_estimate(self) -> Decimal:
        """Rough round-trip estimate: 2× this cost (assumes symmetric legs)."""
        return (self.total * 2).quantize(Decimal("0.01"))


class FOCostModel:
    """
    Compute itemised transaction costs for NSE F&O trades.

    Parameters
    ----------
    slippage_bps_futures : float
        Slippage budget for futures in basis points of notional.
        Default 2.0 bps = ₹2 per ₹1,00,000 notional.  Calibrate from live data.
    slippage_bps_options : float
        Slippage budget for options in basis points of PREMIUM.
        Default 25.0 bps = ₹0.25 per ₹1,000 premium.  Wide: options spreads can be large.

    Usage
    -----
    model = FOCostModel()

    # Sell 1 lot of Nifty 24000 CE at ₹150 premium, 7 DTE
    cost = model.compute_option(
        side="SELL", lots=1, lot_size=65, ltp=150.0, underlying=24000.0
    )
    print(f"STT: ₹{cost.stt}  Total: ₹{cost.total}")

    # Buy 1 Nifty futures lot at 24000
    cost = model.compute_futures(
        side="BUY", lots=1, lot_size=65, price=24000.0
    )
    """

    def __init__(
        self,
        slippage_bps_futures: float = 2.0,
        slippage_bps_options: float = 25.0,
    ) -> None:
        self._slip_fut = Decimal(str(slippage_bps_futures)) / Decimal("10000")
        self._slip_opt = Decimal(str(slippage_bps_options)) / Decimal("10000")

    # ── Options ───────────────────────────────────────────────────────────────

    def compute_option(
        self,
        side: str,          # "BUY" or "SELL"
        lots: int,
        lot_size: int,
        ltp: float,         # last traded price (option premium per unit)
        underlying: float,  # underlying price (index level) — used for notional
        is_exercise: bool = False,  # True only for exercise/settlement legs
        intrinsic_value: Optional[float] = None,  # required if is_exercise=True
    ) -> FOTradeCost:
        """
        Compute all-in cost for an options trade leg.

        STT applies to:
          - Sell side: 0.15% of premium value
          - Exercise/assignment: 0.15% of intrinsic value (not premium)
          - Buy side: no STT on options premium
        """
        qty           = lots * lot_size
        prem_per_unit = Decimal(str(ltp))
        premium       = Decimal(str(qty)) * prem_per_unit
        turnover      = Decimal(str(qty)) * Decimal(str(underlying))

        brokerage = _BROKERAGE

        # STT
        if is_exercise and intrinsic_value is not None:
            intrinsic_v = Decimal(str(intrinsic_value)) * Decimal(str(qty))
            stt = _STT_OPT_EXERCISE_PCT * intrinsic_v
        elif side == "SELL":
            stt = _STT_OPT_SELL_PCT * premium
        else:
            stt = Decimal("0")

        # Exchange transaction charge (on premium for options)
        exchange = _EXCHANGE_OPT_PCT * premium

        # SEBI fee (on notional turnover = qty × underlying price for index)
        sebi_fee = (turnover / Decimal("10000000")) * _SEBI_FEE_PER_CRORE

        # GST: 18% on (exchange + SEBI fee)  — not on STT or stamp
        gst = _GST_PCT * (exchange + sebi_fee)

        # Stamp duty: only on buy side
        stamp_duty = _STAMP_DUTY_PCT * premium if side == "BUY" else Decimal("0")

        # Slippage: on premium (bid-ask spread for options can be wide)
        slippage = self._slip_opt * premium

        total = (
            brokerage + stt + exchange + sebi_fee + gst + stamp_duty + slippage
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        _Q = Decimal("0.01")
        cost = FOTradeCost(
            instrument_type = FOInstrumentType.OPTIONS,
            side            = side,
            brokerage       = brokerage.quantize(_Q, rounding=ROUND_HALF_UP),
            stt             = stt.quantize(_Q, rounding=ROUND_HALF_UP),
            exchange        = exchange.quantize(_Q, rounding=ROUND_HALF_UP),
            sebi_fee        = sebi_fee.quantize(_Q, rounding=ROUND_HALF_UP),
            gst             = gst.quantize(_Q, rounding=ROUND_HALF_UP),
            stamp_duty      = stamp_duty.quantize(_Q, rounding=ROUND_HALF_UP),
            slippage        = slippage.quantize(_Q, rounding=ROUND_HALF_UP),
            total           = total,
            turnover        = turnover.quantize(_Q, rounding=ROUND_HALF_UP),
            premium         = premium.quantize(_Q, rounding=ROUND_HALF_UP),
        )
        log.debug(
            "fo_cost.option",
            side=side, lots=lots, ltp=ltp, total=str(cost.total),
            stt=str(cost.stt), bps_premium=str(cost.as_bps_of_premium()),
        )
        return cost

    # ── Futures ───────────────────────────────────────────────────────────────

    def compute_futures(
        self,
        side: str,    # "BUY" or "SELL"
        lots: int,
        lot_size: int,
        price: float,  # futures price per unit
    ) -> FOTradeCost:
        """
        Compute all-in cost for a futures trade leg.

        STT: 0.05% on sell-side only (on turnover = qty × price).
        Exchange charge: 0.0019% of turnover.
        """
        qty      = lots * lot_size
        turnover = Decimal(str(qty)) * Decimal(str(price))

        brokerage = _BROKERAGE

        # STT: only on sell side for futures
        stt = _STT_FUT_SELL_PCT * turnover if side == "SELL" else Decimal("0")

        # Exchange charge (on turnover for futures)
        exchange = _EXCHANGE_FUT_PCT * turnover

        # SEBI fee
        sebi_fee = (turnover / Decimal("10000000")) * _SEBI_FEE_PER_CRORE

        # GST
        gst = _GST_PCT * (exchange + sebi_fee)

        # Stamp duty: on buy side
        stamp_duty = _STAMP_DUTY_PCT * turnover if side == "BUY" else Decimal("0")

        # Slippage
        slippage = self._slip_fut * turnover

        total = (
            brokerage + stt + exchange + sebi_fee + gst + stamp_duty + slippage
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        _Q = Decimal("0.01")
        cost = FOTradeCost(
            instrument_type = FOInstrumentType.FUTURES,
            side            = side,
            brokerage       = brokerage.quantize(_Q, rounding=ROUND_HALF_UP),
            stt             = stt.quantize(_Q, rounding=ROUND_HALF_UP),
            exchange        = exchange.quantize(_Q, rounding=ROUND_HALF_UP),
            sebi_fee        = sebi_fee.quantize(_Q, rounding=ROUND_HALF_UP),
            gst             = gst.quantize(_Q, rounding=ROUND_HALF_UP),
            stamp_duty      = stamp_duty.quantize(_Q, rounding=ROUND_HALF_UP),
            slippage        = slippage.quantize(_Q, rounding=ROUND_HALF_UP),
            total           = total,
            turnover        = turnover.quantize(_Q, rounding=ROUND_HALF_UP),
            premium         = turnover.quantize(_Q, rounding=ROUND_HALF_UP),  # for futures, premium=turnover
        )
        log.debug(
            "fo_cost.futures",
            side=side, lots=lots, price=price, total=str(cost.total),
            stt=str(cost.stt), bps_turnover=str(cost.as_bps_of_turnover()),
        )
        return cost

    # ── Round-trip helpers ────────────────────────────────────────────────────

    def option_round_trip(
        self,
        lots: int,
        lot_size: int,
        entry_ltp: float,
        exit_ltp: float,
        underlying: float,
        initial_side: str = "SELL",   # "SELL" for premium-selling strategies
    ) -> tuple[FOTradeCost, FOTradeCost]:
        """
        Return (entry_cost, exit_cost) for a round-trip options position.
        For short-premium strategies (initial_side="SELL"):
          - Entry: sell at entry_ltp
          - Exit : buy back at exit_ltp
        """
        entry_cost = self.compute_option(initial_side, lots, lot_size, entry_ltp, underlying)
        exit_side  = "BUY" if initial_side == "SELL" else "SELL"
        exit_cost  = self.compute_option(exit_side, lots, lot_size, exit_ltp, underlying)
        return entry_cost, exit_cost

    def strategy_break_even_bps(
        self,
        lots: int,
        lot_size: int,
        ltp: float,
        underlying: float,
    ) -> float:
        """
        Minimum P&L in INR needed to break even on a short-options round trip.
        Returns the value in basis points of premium received.

        Use this to filter strategies with insufficient premium vs. total costs.
        """
        entry, exit_ = self.option_round_trip(lots, lot_size, ltp, ltp, underlying)
        total_rt = float(entry.total + exit_.total)
        premium_received = float(lots * lot_size * ltp)
        if premium_received == 0:
            return 0.0
        return total_rt / premium_received * 10_000
