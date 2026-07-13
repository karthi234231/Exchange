"""
algo_trading/research/strategy_simulator.py — Execute strategy logic against replayed chain.

This module wraps a strategy definition and runs it against ChainReplayEngine events,
recording all fills, P&L, and metrics. It is the bridge between the replay engine
and the metrics engine.

Design principles (FO_SYSTEM_DESIGN.md Section 9.3):
  - Strategy receives ReplayEvent objects and returns trade decisions
  - All fills are simulated via FillSimulator
  - All costs are applied via FOCostModel
  - Results are compatible with MetricsEngine (list[TradeRecord])
  - Audit trail: every run logged to AuditStore

Strategy interface:
  A strategy is a callable with signature:
    on_bar(event: ReplayEvent, state: StrategyState) -> list[OrderIntent]

  OrderIntent specifies: symbol, expiry, strike, option_type, side, qty_lots, order_type
  Simulator converts OrderIntent → SimulatedFill → TradeRecord

Position tracking:
  Simulator maintains an internal position ledger.
  Closing fills are matched to opening fills using FIFO.
  P&L is computed at position close (not mark-to-market).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterator

import structlog

log = structlog.get_logger(__name__)


@dataclass
class OrderIntent:
    """
    A trading intention returned by the strategy on each bar.

    The simulator converts this to a SimulatedFill.
    """
    symbol: str
    expiry: date
    strike: float
    option_type: str          # "CE" or "PE"
    side: str                 # "BUY" or "SELL"
    qty_lots: int
    order_type: str = "LIMIT" # "LIMIT" or "MARKET"
    strategy_tag: str = ""    # for attribution (e.g. "iron_condor.short_call")


@dataclass
class OpenPosition:
    """A currently open position in the simulator ledger."""
    position_id: str
    order_intent: OrderIntent
    fill_price: float
    fill_time: datetime
    qty_lots: int
    lot_size: int


@dataclass
class ClosedTrade:
    """A completed round-trip trade (open + close)."""
    position_id: str
    symbol: str
    expiry: date
    strike: float
    option_type: str
    opening_side: str         # "BUY" or "SELL"
    open_price: float
    close_price: float
    open_time: datetime
    close_time: datetime
    qty_lots: int
    lot_size: int
    pnl_gross: float          # before costs
    total_cost: float         # all statutory costs
    pnl_net: float            # after costs (= pnl_gross - total_cost)
    strategy_tag: str = ""
    fill_mode: str = ""

    def holding_days(self) -> float:
        return (self.close_time - self.open_time).total_seconds() / 86400.0


@dataclass
class StrategyState:
    """
    Mutable state passed to the strategy on each bar.

    The strategy can store any state it needs here.
    Simulator resets this at the start of each replay run.
    """
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    session_day: date | None = None
    bars_today: int = 0
    is_expiry_bar: bool = False
    custom: dict = field(default_factory=dict)  # strategy-specific state

    @property
    def net_delta_lots(self) -> int:
        """Net open lots across all positions (positive = net long)."""
        total = 0
        for pos in self.open_positions.values():
            sign = 1 if pos.order_intent.side == "BUY" else -1
            total += sign * pos.qty_lots
        return total


class StrategySimulator:
    """
    Runs a strategy function against historical chain replay.

    Parameters
    ----------
    fill_simulator : FillSimulator
        From research.fill_simulator — handles fill simulation.
    cost_model : FOCostModel
        From fo.cost_model — computes statutory charges.
    lot_size_fn : callable
        Callable(symbol: str, on_date: date) → int
        Returns the correct lot size for the contract on the given date.
        Must be date-aware (lot sizes change — see instrument_master).
    slippage_model : SlippageModel | None
        From research.slippage_model — tiered slippage.

    Usage
    -----
    def my_strategy(event: ReplayEvent, state: StrategyState) -> list[OrderIntent]:
        snapshot = event.snapshot
        if should_sell_strangle(snapshot):
            return [
                OrderIntent("NIFTY", expiry, call_strike, "CE", "SELL", 1),
                OrderIntent("NIFTY", expiry, put_strike, "PE", "SELL", 1),
            ]
        return []

    sim = StrategySimulator(fill_simulator, cost_model, lot_size_fn)
    trades, metrics = sim.run(my_strategy, replay_engine.replay())
    """

    def __init__(
        self,
        fill_simulator: Any,       # research.fill_simulator.FillSimulator
        cost_model: Any,           # fo.cost_model.FOCostModel
        lot_size_fn: Callable,     # (symbol, date) → int
        slippage_model: Any = None,
    ) -> None:
        self._fill_sim = fill_simulator
        self._cost_model = cost_model
        self._lot_size_fn = lot_size_fn
        self._slippage_model = slippage_model

    def run(
        self,
        strategy_fn: Callable,
        replay_iter: Iterator,
        max_open_positions: int = 20,
    ) -> tuple[list[ClosedTrade], StrategyState]:
        """
        Run strategy against a replay stream.

        Parameters
        ----------
        strategy_fn : callable
            Called as strategy_fn(event, state) → list[OrderIntent]
        replay_iter : Iterator[ReplayEvent]
            From ChainReplayEngine.replay()
        max_open_positions : int
            Safety cap on number of concurrent open positions.

        Returns
        -------
        (closed_trades, final_state)
        """
        state = StrategyState()
        order_counter = 0

        for event in replay_iter:
            state.session_day = event.session_day
            state.bars_today = event.bars_elapsed_today
            state.is_expiry_bar = event.is_expiry_bar

            # Check if forced expiry close needed
            if event.is_expiry_bar and event.minutes_to_session_close < 5.0:
                self._force_close_expiry_positions(
                    state, event, order_counter
                )
                order_counter += len(state.open_positions)

            # Call strategy
            try:
                intents = strategy_fn(event, state)
            except Exception as e:
                log.error("simulator.strategy_error",
                          event_time=str(event.event_time), error=str(e))
                continue

            if not intents:
                continue

            # Process each order intent
            for intent in intents:
                if len(state.open_positions) >= max_open_positions:
                    log.warning("simulator.max_positions_reached",
                                count=max_open_positions)
                    break

                order_id = f"SIM_{order_counter:06d}"
                order_counter += 1

                # Get quote from snapshot
                snapshot = event.snapshot
                quote = self._get_quote(snapshot, intent)

                if quote is None:
                    log.debug("simulator.no_quote",
                              strike=intent.strike, opt_type=intent.option_type)
                    continue

                # Simulate fill
                fill = self._fill_sim.simulate_fill(
                    order_id=order_id,
                    symbol=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    option_type=intent.option_type,
                    side=intent.side,
                    qty_lots=intent.qty_lots,
                    quote=quote,
                    event_time=event.event_time,
                    is_expiry_day=event.is_expiry_bar,
                )

                if fill is None or fill.cancelled or fill.filled_qty == 0:
                    log.debug("simulator.fill_missed", order_id=order_id,
                              strike=intent.strike)
                    continue

                lot_size = self._lot_size_fn(intent.symbol, event.session_day)

                # Match against open positions (close) or open new
                matched = self._try_close(state, intent, fill, event.event_time, lot_size)
                if not matched:
                    # Open new position
                    pos = OpenPosition(
                        position_id=order_id,
                        order_intent=intent,
                        fill_price=fill.fill_price,
                        fill_time=event.event_time,
                        qty_lots=fill.filled_qty,
                        lot_size=lot_size,
                    )
                    state.open_positions[order_id] = pos
                    log.debug("simulator.opened",
                              order_id=order_id,
                              strike=intent.strike, price=f"₹{fill.fill_price:.2f}")

        # Mark remaining open positions as still open (not closed)
        if state.open_positions:
            log.warning("simulator.run_complete_with_open_positions",
                        count=len(state.open_positions))

        log.info("simulator.run_complete",
                 closed_trades=len(state.closed_trades),
                 open_remaining=len(state.open_positions))
        return state.closed_trades, state

    def _try_close(
        self,
        state: StrategyState,
        intent: OrderIntent,
        fill: Any,
        event_time: datetime,
        lot_size: int,
    ) -> bool:
        """
        Try to match intent against an existing open position (FIFO).
        Returns True if a closing trade was found and processed.
        """
        # Closing side is opposite of opening
        open_side = "SELL" if intent.side == "BUY" else "BUY"

        # Find matching open position (same contract, opposite side)
        for pos_id, pos in list(state.open_positions.items()):
            oi = pos.order_intent
            if (oi.symbol == intent.symbol
                    and oi.expiry == intent.expiry
                    and abs(oi.strike - intent.strike) < 0.01
                    and oi.option_type == intent.option_type
                    and oi.side == open_side):

                # Compute P&L
                if pos.order_intent.side == "SELL":
                    # Short position: profit = open_price - close_price
                    pnl_per_unit = pos.fill_price - fill.fill_price
                else:
                    # Long position: profit = close_price - open_price
                    pnl_per_unit = fill.fill_price - pos.fill_price

                qty = min(pos.qty_lots, fill.filled_qty)
                pnl_gross = pnl_per_unit * qty * lot_size

                # Compute costs for both legs
                cost_open = self._compute_cost(
                    pos.order_intent.symbol,
                    pos.order_intent.side,
                    qty, lot_size,
                    pos.fill_price,
                    pos.fill_time.date(),
                )
                cost_close = self._compute_cost(
                    intent.symbol,
                    intent.side,
                    qty, lot_size,
                    fill.fill_price,
                    event_time.date(),
                )
                total_cost = cost_open + cost_close

                trade = ClosedTrade(
                    position_id=pos_id,
                    symbol=intent.symbol,
                    expiry=intent.expiry,
                    strike=intent.strike,
                    option_type=intent.option_type,
                    opening_side=pos.order_intent.side,
                    open_price=pos.fill_price,
                    close_price=fill.fill_price,
                    open_time=pos.fill_time,
                    close_time=event_time,
                    qty_lots=qty,
                    lot_size=lot_size,
                    pnl_gross=pnl_gross,
                    total_cost=total_cost,
                    pnl_net=pnl_gross - total_cost,
                    strategy_tag=intent.strategy_tag,
                    fill_mode=getattr(fill, "fill_mode", ""),
                )
                state.closed_trades.append(trade)
                del state.open_positions[pos_id]

                log.debug("simulator.closed",
                          pos_id=pos_id, pnl_net=f"₹{trade.pnl_net:.2f}")
                return True

        return False

    def _force_close_expiry_positions(
        self, state: StrategyState, event: Any, counter_base: int
    ) -> None:
        """Force-close all expiry positions at settlement (zero-premium)."""
        for pos_id, pos in list(state.open_positions.items()):
            if pos.order_intent.expiry == event.session_day:
                close_intent = OrderIntent(
                    symbol=pos.order_intent.symbol,
                    expiry=pos.order_intent.expiry,
                    strike=pos.order_intent.strike,
                    option_type=pos.order_intent.option_type,
                    side="BUY" if pos.order_intent.side == "SELL" else "SELL",
                    qty_lots=pos.qty_lots,
                    order_type="MARKET",
                    strategy_tag="EXPIRY_FORCED_CLOSE",
                )
                lot_size = pos.lot_size
                settle_price = 0.0  # OTM expires worthless; ITM: intrinsic value
                snapshot = event.snapshot
                quote = self._get_quote(snapshot, close_intent)
                if quote:
                    settle_price = getattr(quote, "ltp", 0.0) or 0.0

                if pos.order_intent.side == "SELL":
                    pnl_gross = (pos.fill_price - settle_price) * pos.qty_lots * lot_size
                else:
                    pnl_gross = (settle_price - pos.fill_price) * pos.qty_lots * lot_size

                cost = self._compute_cost(
                    pos.order_intent.symbol, close_intent.side,
                    pos.qty_lots, lot_size, settle_price, event.session_day,
                )
                trade = ClosedTrade(
                    position_id=pos_id,
                    symbol=pos.order_intent.symbol,
                    expiry=pos.order_intent.expiry,
                    strike=pos.order_intent.strike,
                    option_type=pos.order_intent.option_type,
                    opening_side=pos.order_intent.side,
                    open_price=pos.fill_price,
                    close_price=settle_price,
                    open_time=pos.fill_time,
                    close_time=event.event_time,
                    qty_lots=pos.qty_lots,
                    lot_size=lot_size,
                    pnl_gross=pnl_gross,
                    total_cost=cost,
                    pnl_net=pnl_gross - cost,
                    strategy_tag="EXPIRY_FORCED_CLOSE",
                )
                state.closed_trades.append(trade)
                del state.open_positions[pos_id]
                log.info("simulator.expiry_close",
                         pos_id=pos_id, settle=f"₹{settle_price:.2f}",
                         pnl_net=f"₹{trade.pnl_net:.2f}")

    def _get_quote(self, snapshot: Any, intent: OrderIntent) -> Any | None:
        """Extract quote from snapshot for given strike/expiry/option_type."""
        if snapshot is None:
            return None
        try:
            # ChainSnapshot.get_quote() interface
            return snapshot.get_quote(intent.expiry, intent.strike, intent.option_type)
        except Exception:
            # Fallback: dict-based snapshot
            if isinstance(snapshot, dict):
                quotes = snapshot.get("quotes", {})
                key = (intent.expiry, intent.strike, intent.option_type)
                return quotes.get(key)
        return None

    def _compute_cost(
        self,
        symbol: str,
        side: str,
        qty_lots: int,
        lot_size: int,
        price: float,
        trade_date: date,
    ) -> float:
        """Compute statutory cost for one trade leg. Returns total cost in INR."""
        try:
            cost = self._cost_model.compute_option(
                side=side,
                lots=qty_lots,
                lot_size=lot_size,
                ltp=price,
                underlying=price * 1.0,  # approximation: use premium as underlying for cost
            )
            return float(cost.total)
        except Exception as e:
            log.warning("simulator.cost_error", error=str(e))
            return 0.0
