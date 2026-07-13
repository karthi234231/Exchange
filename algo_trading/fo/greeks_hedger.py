"""
algo_trading/fo/greeks_hedger.py — Delta-neutral rebalancer for NSE F&O book.

Monitors portfolio delta from PortfolioRiskEngine and computes the futures order
needed to maintain delta-neutrality.  The hedger returns HedgeOrder objects —
it does NOT place orders itself.  The caller (orchestrator) submits them to
OrderExecutor.  This keeps I/O out of the hedger and makes it testable in isolation.

Design
------
- Delta threshold: |portfolio_delta| triggers hedge when rounded lot count >= min_lots.
- Hedge instrument: Nifty/BankNifty front-month futures (lot size from contract registry).
- Hedge size: delegated to PortfolioRiskEngine.delta_hedge_units(), then clamped.
- Position tracking: maintains a running net futures lot count for P&L attribution.
- Unwind path: unwind_hedge() computes the closing trade to flatten the hedge book.

The hedger does NOT generate alpha — it is purely defensive.
Call it after any of: new strategy entered, existing strategy closed,
or on a scheduled rebalance heartbeat (e.g. every 15-min bar close).

Thread safety
-------------
NOT internally thread-safe.  Call from the single-threaded main bar consumer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from algo_trading.fo.contracts import ContractType, get_contract
from algo_trading.fo.portfolio_risk import PortfolioGreeks, PortfolioRiskEngine

log = structlog.get_logger(__name__)

# ── Defaults (ASSUMED — tune to capital size and transaction-cost profile) ────
_DEFAULT_MIN_LOTS: int   = 1   # ASSUMED: don't place a hedge smaller than 1 lot
_DEFAULT_MAX_LOTS: int   = 10  # ASSUMED: cap any single hedge at 10 lots (market impact)


@dataclass(slots=True)
class HedgeRecord:
    """Immutable record of a single delta-hedge trade."""
    timestamp:              datetime
    symbol:                 str
    direction:              int      # +1 = bought futures, -1 = sold futures
    lots:                   int      # absolute lot count
    futures_price:          float    # mark price at order submission (ASSUMED execution)
    portfolio_delta_before: float    # portfolio delta before the hedge
    portfolio_delta_after:  float    # approximate delta after the hedge


class DeltaHedger:
    """
    Delta-neutral rebalancer for NSE F&O short-premium books.

    Usage (in the orchestrator bar loop)::

        hedger = DeltaHedger(symbol="NIFTY", risk_engine=engine)

        # After computing portfolio Greeks:
        if hedger.needs_hedge(greeks):
            lots = hedger.compute_hedge_lots(greeks)
            # caller places the futures order and gets back fill_price
            record = hedger.record_hedge(greeks, lots_traded=lots, futures_price=fill_price)

        # At day end (before position EOD close):
        unwind = hedger.unwind_hedge()
        # caller places unwind order, then:
        hedger.reset()

    Parameters
    ----------
    symbol : str
        NSE underlying symbol ("NIFTY" or "BANKNIFTY").
    risk_engine : PortfolioRiskEngine
        Shared risk engine instance used for delta_hedge_units().
    min_lots : int
        Minimum hedge size.  Hedges smaller than this are suppressed.
        ASSUMED default: 1 lot.
    max_lots : int
        Maximum single hedge size.  Larger hedges are split across rebalances.
        ASSUMED default: 10 lots.
    """

    def __init__(
        self,
        symbol: str,
        risk_engine: PortfolioRiskEngine,
        min_lots: int = _DEFAULT_MIN_LOTS,
        max_lots: int = _DEFAULT_MAX_LOTS,
    ) -> None:
        if min_lots < 1:
            raise ValueError(f"min_lots must be >= 1, got {min_lots}")
        if max_lots < min_lots:
            raise ValueError(f"max_lots ({max_lots}) must be >= min_lots ({min_lots})")

        self._symbol       = symbol.upper()
        self._risk_engine  = risk_engine
        self._min_lots     = min_lots
        self._max_lots     = max_lots

        # Fetch authoritative lot size from contract registry
        fut_spec = get_contract(self._symbol, ContractType.FUTURES)
        self._futures_lot_size: int = fut_spec.lot_size

        # Net futures lots currently held as the hedge book
        # Positive = long hedge; negative = short hedge
        self._hedge_lots: int = 0

        # Full audit trail of all hedge trades since last reset()
        self._hedge_history: list[HedgeRecord] = []

        log.info(
            "delta_hedger.init",
            symbol=self._symbol,
            futures_lot_size=self._futures_lot_size,
            min_lots=min_lots,
            max_lots=max_lots,
        )

    # ── Core API ──────────────────────────────────────────────────────────────

    def needs_hedge(self, greeks: PortfolioGreeks) -> bool:
        """
        Return True if the portfolio delta requires a hedge trade.

        A hedge is needed when the rounded lot count exceeds min_lots.
        During warmup (portfolio is empty) this will return False.

        Parameters
        ----------
        greeks : PortfolioGreeks
            Current portfolio Greeks snapshot.
        """
        return self.compute_hedge_lots(greeks) != 0

    def compute_hedge_lots(self, greeks: PortfolioGreeks) -> int:
        """
        Compute the signed number of futures lots for delta neutrality.

        Delegates the core formula to PortfolioRiskEngine.delta_hedge_units(),
        then applies min/max lot filters.

        Returns
        -------
        int
            Signed lots to trade:
              Positive → buy futures (offset net short delta).
              Negative → sell futures (offset net long delta).
              Zero     → no hedge action (within min_lots threshold).
        """
        raw = self._risk_engine.delta_hedge_units(greeks, self._futures_lot_size)

        # ── Sanity-check the returned value before using it ───────────────────
        if not isinstance(raw, (int, float)) or math.isnan(raw) or math.isinf(raw):
            raise ValueError(f"delta_hedge_units returned invalid value: {raw}")
        # Preserve sign: positive = buy futures, negative = sell futures
        lots = int(raw)

        # Clamp to max_lots — remainder carries forward to next rebalance
        if lots > 0:
            clamped = min(lots, self._max_lots)
        elif lots < 0:
            clamped = max(lots, -self._max_lots)
        else:
            clamped = 0

        # Suppress hedges smaller than min_lots (sub-lot precision is not tradeable)
        if abs(clamped) < self._min_lots:
            clamped = 0

        log.debug(
            "delta_hedger.compute",
            symbol=self._symbol,
            portfolio_delta=round(greeks.delta, 4),
            raw_hedge_lots=raw,
            clamped_lots=clamped,
        )
        return clamped

    def record_hedge(
        self,
        greeks_before: PortfolioGreeks,
        lots_traded: int,
        futures_price: float,
    ) -> HedgeRecord:
        """
        Record a completed hedge trade and update the running position.

        Call this AFTER the broker confirms the fill.

        Parameters
        ----------
        greeks_before : PortfolioGreeks
            Greeks snapshot at the time the hedge decision was made.
        lots_traded : int
            Signed lots actually traded (positive = bought, negative = sold).
            Must be non-zero.
        futures_price : float
            Execution price of the futures order (INR).

        Returns
        -------
        HedgeRecord
            Immutable record stored in hedge_history.

        Raises
        ------
        ValueError
            If lots_traded is zero.
        """
        if lots_traded == 0:
            raise ValueError(
                "lots_traded must be non-zero — only call record_hedge after a confirmed fill."
            )

        direction = 1 if lots_traded > 0 else -1
        self._hedge_lots += lots_traded

        # Approximate portfolio delta after hedging (linear approximation)
        delta_after = greeks_before.delta + lots_traded * self._futures_lot_size

        record = HedgeRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=self._symbol,
            direction=direction,
            lots=abs(lots_traded),
            futures_price=futures_price,
            portfolio_delta_before=round(greeks_before.delta, 4),
            portfolio_delta_after=round(delta_after, 4),
        )
        self._hedge_history.append(record)

        log.info(
            "delta_hedger.hedged",
            symbol=self._symbol,
            side="BUY" if direction > 0 else "SELL",
            lots=abs(lots_traded),
            futures_price=round(futures_price, 2),
            delta_before=record.portfolio_delta_before,
            delta_after=record.portfolio_delta_after,
            cumulative_hedge_lots=self._hedge_lots,
        )
        return record

    def unwind_hedge(self) -> int:
        """
        Compute the lot count to flatten the current hedge book.

        Returns the opposite-signed lot count to _hedge_lots.  The caller
        places this order and then calls reset() to clear state.

        Returns
        -------
        int
            Lots to trade for full unwind (0 if already flat).
        """
        unwind = -self._hedge_lots
        if unwind != 0:
            log.info(
                "delta_hedger.unwind_required",
                symbol=self._symbol,
                current_hedge_lots=self._hedge_lots,
                unwind_lots=unwind,
            )
        return unwind

    def reset(self) -> None:
        """
        Reset hedge position and history (call at day end after full unwind).

        Does NOT place any orders — caller must ensure the hedge is flat first
        via unwind_hedge() before calling reset().
        """
        prev = self._hedge_lots
        self._hedge_lots = 0
        self._hedge_history.clear()
        log.info(
            "delta_hedger.reset",
            symbol=self._symbol,
            cleared_hedge_lots=prev,
        )

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def current_hedge_lots(self) -> int:
        """Net futures lots currently held as hedge (signed)."""
        return self._hedge_lots

    @property
    def hedge_history(self) -> list[HedgeRecord]:
        """Defensive copy of all hedge records since last reset()."""
        return list(self._hedge_history)

    @property
    def futures_lot_size(self) -> int:
        """NSE lot size for the hedging futures contract."""
        return self._futures_lot_size

    @property
    def min_lots(self) -> int:
        return self._min_lots

    @property
    def max_lots(self) -> int:
        return self._max_lots

    @property
    def symbol(self) -> str:
        return self._symbol
