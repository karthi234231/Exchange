"""
risk/cost_model.py — Full India cost model for NSE intraday equity trades.

All values are ASSUMED (see verified_facts.md) — verify before live trading.

India equity intraday cost breakdown (Shoonya, NSE):
  - Brokerage: 0% (Shoonya zero-brokerage)
  - STT: 0.025% on sell-side turnover (intraday), 0.1% both sides (delivery)
  - NSE Exchange transaction charge: ~0.00345% of turnover
  - SEBI turnover fees: ₹10/crore = 0.00001% of turnover
  - GST: 18% on (brokerage + exchange charges + SEBI fee) — NOT on STT or stamp
  - Stamp duty: 0.003% on buy-side only
  - Dynamic slippage: ATR-based, scaled by time-of-day and spread width

Constraint: No static 0.05% slippage anywhere — dynamic model required [R-11].
Constraint: Decimal for all monetary calculations — never float arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

import structlog
import yaml

log = structlog.get_logger(__name__)


def _load_tax_rates() -> dict:
    """Load tax rates from config/tax_rates.yaml if it exists, else return {}."""
    path = Path("config/tax_rates.yaml")
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


_TAX_RATES: dict = _load_tax_rates()

# ── Tax Rates (ASSUMED — verify before live) ──────────────────────────────────
# Values loaded from config/tax_rates.yaml when available; hardcoded defaults
# used as fallback so the module works without the config file.
_STT_INTRADAY_SELL_PCT = Decimal(str(
    _TAX_RATES.get("stt_intraday_sell_pct", 0.025) / 100
))   # 0.025% → 0.00025
_STT_DELIVERY_BOTH_PCT = Decimal("0.001")      # 0.1% on both sides (not in YAML yet)
_EXCHANGE_CHARGE_PCT   = Decimal(str(
    _TAX_RATES.get("exchange_charge_pct", 0.00345) / 100
))   # 0.00345% → 0.0000345
_SEBI_FEE_PER_CRORE    = Decimal(str(
    _TAX_RATES.get("sebi_fee_per_crore", 10.0)
))   # ₹10 per crore
_GST_PCT               = Decimal(str(
    _TAX_RATES.get("gst_pct", 18.0) / 100
))   # 18% → 0.18
_STAMP_DUTY_PCT        = Decimal(str(
    _TAX_RATES.get("stamp_duty_buy_pct", 0.003) / 100
))   # 0.003% → 0.00003
_BROKERAGE_PCT         = Decimal("0")          # Shoonya: zero brokerage

# Slippage model parameters
_ATR_SLIPPAGE_BASE_PCT = Decimal("0.30")       # 30% of ATR as base slippage
_OPEN_CLOSE_MULTIPLIER = Decimal("1.5")        # 1.5× slippage in first/last 30 min
_SPREAD_SLIPPAGE_HALF  = Decimal("0.5")        # 50% of bid-ask spread as slippage

# P1-FIX-26: Static fallback — used ONLY when neither ATR nor spread is available.
# Named explicitly so reports can identify when this was applied.
# 5 bps is a conservative lower bound for liquid NSE intraday stocks.
# If this fires frequently it means the dynamic model is not being supplied inputs —
# fix the upstream caller rather than relying on this fallback.
_SLIPPAGE_STATIC_FALLBACK_BPS = Decimal("5")   # 5 bps = 0.05% of price


@dataclass(frozen=True)
class TradeCost:
    """Breakdown of all costs for a single trade side."""
    brokerage: Decimal
    stt: Decimal
    exchange_charge: Decimal
    sebi_fee: Decimal
    gst: Decimal
    stamp_duty: Decimal
    slippage_inr: Decimal
    slippage_bps: Decimal
    total_inr: Decimal        # sum of all above
    turnover: Decimal         # quantity × price

    def as_bps_of_turnover(self) -> Decimal:
        """Total cost as basis points of turnover."""
        if self.turnover == 0:
            return Decimal(0)
        return (self.total_inr / self.turnover * 10_000).quantize(Decimal("0.01"))


class IndiaCostModel:
    """
    Computes full India equity cost for a single trade.

    Usage:
        model = IndiaCostModel()
        cost = model.compute("RELIANCE", "BUY", quantity=100, price=Decimal("2850"))
        print(f"Total cost: ₹{cost.total_inr:.2f} ({cost.as_bps_of_turnover():.2f} bps)")
    """

    def compute(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        price: Decimal,
        is_delivery: bool = False,
        atr: Optional[Decimal] = None,
        hour_of_day: Optional[int] = None,
        spread: Optional[Decimal] = None,
    ) -> TradeCost:
        """
        Compute all-in trade cost for one side of a trade.

        For a round-trip, call compute() twice (once for BUY, once for SELL)
        and sum the total_inr values.

        Parameters:
            symbol: trading symbol (for logging)
            side: "BUY" or "SELL"
            quantity: number of shares
            price: execution price (Decimal)
            is_delivery: False = MIS intraday (default), True = CNC delivery
            atr: ATR(14) in price units — used for dynamic slippage (optional)
            hour_of_day: IST hour (9–15) — for time-of-day slippage multiplier
            spread: bid-ask spread in price units — additional slippage component
        """
        turnover = Decimal(str(quantity)) * price

        brokerage = _BROKERAGE_PCT * turnover  # ₹0 for Shoonya

        stt = self._compute_stt(turnover, side, is_delivery)

        exchange_charge = _EXCHANGE_CHARGE_PCT * turnover

        sebi_fee = (turnover / Decimal("10000000")) * _SEBI_FEE_PER_CRORE  # ₹10/crore

        # GST applies to brokerage + exchange + SEBI fee (NOT to STT or stamp duty)
        gst_base = brokerage + exchange_charge + sebi_fee
        gst = _GST_PCT * gst_base

        stamp_duty = self._compute_stamp_duty(turnover, side)

        slippage_inr, slippage_bps = self._compute_slippage(
            price=price,
            quantity=quantity,
            atr=atr,
            hour_of_day=hour_of_day,
            spread=spread,
        )

        total = (
            brokerage + stt + exchange_charge + sebi_fee
            + gst + stamp_duty + slippage_inr
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        cost = TradeCost(
            brokerage=brokerage.quantize(Decimal("0.01")),
            stt=stt.quantize(Decimal("0.01")),
            exchange_charge=exchange_charge.quantize(Decimal("0.01")),
            sebi_fee=sebi_fee.quantize(Decimal("0.01")),
            gst=gst.quantize(Decimal("0.01")),
            stamp_duty=stamp_duty.quantize(Decimal("0.01")),
            slippage_inr=slippage_inr.quantize(Decimal("0.01")),
            slippage_bps=slippage_bps.quantize(Decimal("0.01")),
            total_inr=total,
            turnover=turnover.quantize(Decimal("0.01")),
        )

        log.debug(
            "cost_model.computed",
            symbol=symbol,
            side=side,
            qty=quantity,
            price=str(price),
            total_inr=str(cost.total_inr),
            total_bps=str(cost.as_bps_of_turnover()),
        )
        return cost

    def compute_round_trip(
        self,
        symbol: str,
        quantity: int,
        entry_price: Decimal,
        exit_price: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        direction: int = 1,
    ) -> TradeCost:
        """
        Compute total cost for a complete round-trip.

        For a LONG trade (direction=+1): entry is BUY, exit is SELL.
        For a SHORT trade (direction=-1): entry is SELL, exit is BUY.

        exit_price defaults to entry_price (break-even analysis).
        """
        ep = exit_price or entry_price
        if direction >= 0:
            # LONG: buy in, sell out
            entry_side, exit_side = "BUY", "SELL"
        else:
            # SHORT: sell in, buy out
            entry_side, exit_side = "SELL", "BUY"
        buy_cost = self.compute(symbol, entry_side, quantity, entry_price, atr=atr)
        sell_cost = self.compute(symbol, exit_side, quantity, ep, atr=atr)

        total = buy_cost.total_inr + sell_cost.total_inr
        combined_turnover = buy_cost.turnover + sell_cost.turnover

        return TradeCost(
            brokerage=buy_cost.brokerage + sell_cost.brokerage,
            stt=buy_cost.stt + sell_cost.stt,
            exchange_charge=buy_cost.exchange_charge + sell_cost.exchange_charge,
            sebi_fee=buy_cost.sebi_fee + sell_cost.sebi_fee,
            gst=buy_cost.gst + sell_cost.gst,
            stamp_duty=buy_cost.stamp_duty + sell_cost.stamp_duty,
            slippage_inr=buy_cost.slippage_inr + sell_cost.slippage_inr,
            slippage_bps=(buy_cost.slippage_bps + sell_cost.slippage_bps) / 2,
            total_inr=total,
            turnover=combined_turnover,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_stt(turnover: Decimal, side: str, is_delivery: bool) -> Decimal:
        """
        STT (Securities Transaction Tax).
        Intraday: 0.025% on sell-side only.
        Delivery: 0.1% on both buy and sell.
        """
        if is_delivery:
            return _STT_DELIVERY_BOTH_PCT * turnover
        else:
            # Intraday: STT only on sell side
            if side == "SELL":
                return _STT_INTRADAY_SELL_PCT * turnover
            return Decimal(0)

    @staticmethod
    def _compute_stamp_duty(turnover: Decimal, side: str) -> Decimal:
        """Stamp duty: 0.003% on buy-side only (Maharashtra rate)."""
        if side == "BUY":
            return _STAMP_DUTY_PCT * turnover
        return Decimal(0)

    @staticmethod
    def _compute_slippage(
        price: Decimal,
        quantity: int,
        atr: Optional[Decimal],
        hour_of_day: Optional[int],
        spread: Optional[Decimal],
    ) -> tuple[Decimal, Decimal]:
        """
        Dynamic slippage model.

        Components:
          1. ATR-based: 30% of ATR per trade (models market impact).
          2. Time-of-day multiplier: 1.5× during opening (9-9:30) and closing (15-15:30).
          3. Spread component: 50% of bid-ask spread.

        Returns (slippage_inr, slippage_bps).
        """
        slippage_per_share = Decimal(0)

        # Component 1: ATR-based slippage
        if atr is not None and atr > 0:
            tod_mult = Decimal(1)
            if hour_of_day is not None and (hour_of_day < 10 or hour_of_day >= 15):
                tod_mult = _OPEN_CLOSE_MULTIPLIER
            slippage_per_share += _ATR_SLIPPAGE_BASE_PCT * atr * tod_mult

        # Component 2: Spread component
        if spread is not None and spread > 0:
            slippage_per_share += _SPREAD_SLIPPAGE_HALF * spread

        # P1-FIX-26: If no dynamic inputs available, apply explicit named fallback.
        # Log a warning so callers know the estimate is coarse — fix upstream if frequent.
        if slippage_per_share == 0:
            slippage_per_share = price * (_SLIPPAGE_STATIC_FALLBACK_BPS / Decimal("10000"))
            log.warning(
                "cost_model.slippage_fallback_used",
                fallback_bps=str(_SLIPPAGE_STATIC_FALLBACK_BPS),
                reason="no ATR or spread supplied — provide atr/spread for dynamic model",
            )

        slippage_inr = slippage_per_share * quantity
        slippage_bps = (slippage_per_share / price * 10_000) if price > 0 else Decimal(0)

        return slippage_inr, slippage_bps
