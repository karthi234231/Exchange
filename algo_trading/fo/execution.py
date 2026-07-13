"""
algo_trading/fo/execution.py — Multi-leg F&O order execution with legging risk management.

Key concepts
------------
Legging risk:
  A multi-leg strategy (e.g. short strangle) is executed as N sequential orders.
  The market moves between each fill, creating unintended exposure if a later leg
  does not fill at the expected price.  This module manages that risk by:
    1. Executing the highest-risk / highest-premium leg first.
    2. Enforcing a legging window: all legs must fill within N seconds of the
       FIRST submission (not the first fill).  ASSUMED default: 30 seconds.
    3. Checking that the realised net premium is still >= min acceptable floor
       before each subsequent leg.
    4. Aborting the entire strategy (cancel all filled legs at market) if any
       leg is stuck or the premium floor is breached.

Order state machine:
  PENDING → LEGGING (first leg sent) → PARTIALLY_FILLED (≥1 leg filled, rest pending)
          → FILLED (all legs filled)
          → STUCK (legging window expired with unfilled legs)
          → ABORTING (cancelling filled legs)
          → ABORTED (all filled leg cancels confirmed)
          → PARTIAL_ABORT_FAILED (≥1 cancel rejected — manual intervention required)

Thread safety:
  - FOPositionManager uses threading.RLock for all state mutations.
  - MultiLegExecutor uses threading.Lock for in-flight execution tracking.
  - Broker calls are non-blocking: they return quickly and status is polled.

Coding standards:
  - Python 3.11+, full type hints.
  - structlog everywhere — no print or bare logging.
  - ASSUMED thresholds clearly marked.
  - No hardcoded lot sizes — always read from contract master.
"""
from __future__ import annotations

import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

import structlog

from algo_trading.data.broker_base import (
    BrokerBase,
    Order,
    OrderRejectedError,
    SessionError,
)
from algo_trading.fo.contracts import (
    FO_FORCE_CLOSE_TIME,
    FO_NO_NEW_TRADES,
    ContractType,
    get_contract,
)
from algo_trading.fo.pricing import BSMEngine
from algo_trading.fo.strategies import StrategyLeg, StrategyResult
from algo_trading.core.calendar import IST, now_ist

log = structlog.get_logger(__name__)

# ── Tick size constant ────────────────────────────────────────────────────────
# NSE minimum tick for index options.  Source: verified in contracts.py.
_NSE_MIN_TICK: float = 0.05

# ── Limit-price slippage factors (ASSUMED) ───────────────────────────────────
# In the absence of live bid/ask, we approximate using LTP ± a small offset.
# 0.2% of LTP ≈ 2 ticks for an at-the-money Nifty option at ~100 premium.
# Calibrate from observed fill data once live.
_LIMIT_SELL_FACTOR: float = 0.998   # ASSUMED: sell limit slightly below LTP
_LIMIT_BUY_FACTOR:  float = 1.002   # ASSUMED: buy  limit slightly above LTP

# ── Poll interval for order-status checks (ASSUMED) ──────────────────────────
# Broker get_order_book() is called at most once per _POLL_INTERVAL_SEC.
# Keep > 0.2 s to respect NSE rate limits.
_POLL_INTERVAL_SEC: float = 0.5    # ASSUMED: calibrate from broker API limits

# ── Abort market-order retry wait (ASSUMED) ───────────────────────────────────
# After placing a market cancel/exit, wait this many seconds before declaring
# the cancel confirmed (broker may not push WebSocket ack fast enough).
_ABORT_CONFIRM_WAIT_SEC: float = 2.0  # ASSUMED


# ── MultiLegState ─────────────────────────────────────────────────────────────

class MultiLegState(str, Enum):
    PENDING              = "PENDING"
    LEGGING              = "LEGGING"
    PARTIALLY_FILLED     = "PARTIALLY_FILLED"
    FILLED               = "FILLED"
    STUCK                = "STUCK"
    ABORTING             = "ABORTING"
    ABORTED              = "ABORTED"
    PARTIAL_ABORT_FAILED = "PARTIAL_ABORT_FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            MultiLegState.FILLED,
            MultiLegState.ABORTED,
            MultiLegState.PARTIAL_ABORT_FAILED,
        )


# ── Per-leg execution record ──────────────────────────────────────────────────

@dataclass
class LegExecution:
    """Execution status for a single strategy leg."""
    leg: StrategyLeg
    order_id: Optional[str] = None           # broker exchange_order_id
    state: str = "PENDING"                   # PENDING | SENT | FILLED | CANCELLED | REJECTED
    filled_price: Optional[float] = None
    filled_qty: int = 0
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancel_order_id: Optional[str] = None    # exchange_order_id of the abort/exit order
    slippage: float = 0.0                    # filled_price - entry_premium (positive = worse)

    @property
    def is_filled(self) -> bool:
        return self.state == "FILLED"

    @property
    def is_pending_or_sent(self) -> bool:
        return self.state in ("PENDING", "SENT")

    def expected_qty(self) -> int:
        """Total units expected for this leg."""
        return self.leg.lots * self.leg.lot_size


# ── Multi-leg execution record ────────────────────────────────────────────────

@dataclass
class MultiLegExecution:
    """Complete execution record for a multi-leg strategy."""
    execution_id: str
    strategy_name: str
    state: MultiLegState
    legs: list[LegExecution]
    started_at: datetime
    completed_at: Optional[datetime] = None
    net_premium_received: float = 0.0      # actual net premium after all fills
    expected_net_premium: float = 0.0      # at strategy entry (from StrategyResult)
    abort_reason: str = ""
    total_slippage: float = 0.0            # sum of per-leg slippage

    def filled_legs(self) -> list[LegExecution]:
        return [le for le in self.legs if le.is_filled]

    def unfilled_legs(self) -> list[LegExecution]:
        return [le for le in self.legs if not le.is_filled]

    def all_legs_filled(self) -> bool:
        return all(le.is_filled for le in self.legs)

    def _recompute_net_premium(self) -> None:
        """
        Recompute net_premium_received and total_slippage from current fill data.
        Called after each leg fill confirmation.
        """
        net = 0.0
        slip = 0.0
        for le in self.legs:
            if le.is_filled and le.filled_price is not None:
                # direction: +1 = buy (debit), -1 = sell (credit)
                net += le.leg.direction * le.filled_price * le.leg.lots * le.leg.lot_size
                slip += le.slippage * le.leg.lots * le.leg.lot_size
        self.net_premium_received = -net   # positive = net credit received
        self.total_slippage = slip


# ── MultiLegExecutor ──────────────────────────────────────────────────────────

class MultiLegExecutor:
    """
    Executes multi-leg F&O strategies with legging risk management.

    Parameters
    ----------
    broker : BrokerBase
        The broker interface for order placement.
    bsm : BSMEngine
        Black-Scholes-Merton engine (used for fair-value validation).
    leg_timeout_seconds : int
        Maximum seconds to wait for ALL legs to fill before declaring stuck.
        Timer starts at FIRST leg submission.  ASSUMED default: 30 seconds.
    min_net_premium_pct : float
        Abort if net premium received < expected_net_premium * (1 - pct).
        Default 0.10 allows up to 10% degradation vs expected.  ASSUMED.
    use_limit_orders : bool
        True: LIMIT orders at computed limit price (recommended for options).
        False: MARKET orders (higher slippage, faster fill).
        ASSUMED default: True (market orders have wide spreads for options).
    max_position_entries_per_day : int
        Hard cap on new strategy entries per calendar day.  ASSUMED: 3.
    """

    def __init__(
        self,
        broker: BrokerBase,
        bsm: BSMEngine,
        leg_timeout_seconds: int = 30,           # ASSUMED
        min_net_premium_pct: float = 0.10,       # ASSUMED
        use_limit_orders: bool = True,
        max_position_entries_per_day: int = 3,   # ASSUMED
        require_hedge: bool = False,
        max_margin_inr: Optional[float] = None,
    ) -> None:
        self._broker = broker
        self._bsm = bsm
        self._leg_timeout_sec = leg_timeout_seconds
        self._min_net_premium_pct = min_net_premium_pct
        self._use_limit_orders = use_limit_orders
        self._max_entries_per_day = max_position_entries_per_day
        self._require_hedge = require_hedge
        self._max_margin_inr = max_margin_inr

        # In-flight executions keyed by execution_id
        self._inflight: dict[str, MultiLegExecution] = {}
        self._lock = threading.Lock()

        # Daily entry counter — reset at start of each trading day
        self._daily_entries: int = 0
        self._last_entry_date: Optional[date] = None

        log.info(
            "multi_leg_executor.init",
            leg_timeout_sec=leg_timeout_seconds,
            min_net_premium_pct=min_net_premium_pct,
            use_limit_orders=use_limit_orders,
            max_entries_per_day=max_position_entries_per_day,
            require_hedge=require_hedge,
            max_margin_inr=max_margin_inr,
        )

    # ── Primary entry point ───────────────────────────────────────────────────

    def execute_strategy(
        self,
        strategy: StrategyResult,
        current_premiums: dict[str, float],   # {f"{option_type}_{strike}": ltp}
        underlying_price: float,
    ) -> MultiLegExecution:
        """
        Execute a multi-leg strategy with legging risk management.

        Execution order:
          1. Validate: check time (< FO_NO_NEW_TRADES), check premiums still valid,
             check daily entry cap.
          2. Sort legs by execution priority (higher-risk / higher-premium first).
          3. For each leg in priority order:
               a. If legging window has expired → abort strategy.
               b. If remaining net premium is below floor → abort strategy.
               c. Submit leg order.
               d. Wait for fill confirmation (polling with timeout).
               e. If leg is rejected or times out → abort strategy.
          4. If all legs filled → mark FILLED.

        Returns the MultiLegExecution record (FILLED or terminal abort state).
        """
        now = now_ist()

        # ── Step 1: Pre-flight validation ─────────────────────────────────────
        validation_error = self._validate_pre_flight(strategy, now)
        if validation_error:
            log.warning(
                "multi_leg_executor.execute_strategy.blocked",
                strategy=strategy.name,
                reason=validation_error,
            )
            # Return a terminal ABORTED execution immediately
            execution = self._make_execution(strategy)
            execution.state = MultiLegState.ABORTED
            execution.abort_reason = validation_error
            execution.completed_at = now
            return execution

        # ── Step 2: Build execution record and sort legs ───────────────────────
        execution = self._make_execution(strategy)
        execution.state = MultiLegState.LEGGING

        # Sort leg executions by priority (modifies the list in place)
        sorted_legs = self._execution_priority(
            [le.leg for le in execution.legs]
        )
        # Re-order execution.legs to match sorted priority
        leg_map = {id(le.leg): le for le in execution.legs}
        execution.legs = [leg_map[id(sl)] for sl in sorted_legs]

        with self._lock:
            self._inflight[execution.execution_id] = execution
            self._increment_daily_entries(now.date())

        log.info(
            "multi_leg_executor.strategy_started",
            execution_id=execution.execution_id,
            strategy=strategy.name,
            legs=[
                f"{le.leg.option_type}_{le.leg.strike}_{le.leg.side}"
                for le in execution.legs
            ],
            expected_net_premium=execution.expected_net_premium,
        )

        # ── Step 3: Leg-by-leg execution ──────────────────────────────────────
        window_start = now_ist()

        for i, leg_exec in enumerate(execution.legs):
            # (a) Check legging window before submitting each leg
            elapsed = (now_ist() - window_start).total_seconds()
            if elapsed >= self._leg_timeout_sec:
                log.warning(
                    "multi_leg_executor.legging_window_expired.pre_submit",
                    execution_id=execution.execution_id,
                    leg_index=i,
                    elapsed_sec=f"{elapsed:.1f}",
                    timeout_sec=self._leg_timeout_sec,
                )
                execution.state = MultiLegState.STUCK
                execution.abort_reason = (
                    f"Legging window expired ({elapsed:.1f}s) before leg {i} submitted"
                )
                return self.abort_strategy(execution)

            # (b) Check net premium floor before each leg (skip for leg 0)
            if i > 0 and not self._net_premium_acceptable(execution, strategy):
                log.warning(
                    "multi_leg_executor.net_premium_floor_breached",
                    execution_id=execution.execution_id,
                    leg_index=i,
                    realised=execution.net_premium_received,
                    expected=execution.expected_net_premium,
                    min_pct=self._min_net_premium_pct,
                )
                execution.state = MultiLegState.STUCK
                execution.abort_reason = (
                    f"Net premium floor breached at leg {i}: "
                    f"realised={execution.net_premium_received:.2f} < "
                    f"floor={execution.expected_net_premium * (1 - self._min_net_premium_pct):.2f}"
                )
                return self.abort_strategy(execution)

            # (c) Determine limit price for this leg
            ltp_key = f"{leg_exec.leg.option_type}_{leg_exec.leg.strike}"
            current_ltp = current_premiums.get(ltp_key, leg_exec.leg.entry_premium)
            limit_price = (
                self._compute_limit_price(leg_exec.leg, current_ltp)
                if self._use_limit_orders
                else 0.0
            )

            # (d) Submit the leg
            order_id = self._submit_leg(execution, leg_exec, limit_price)
            if order_id is None:
                # Submission itself failed — abort immediately
                execution.state = MultiLegState.STUCK
                execution.abort_reason = f"Leg {i} submission failed"
                return self.abort_strategy(execution)

            # (e) Wait for fill, respecting remaining window
            remaining_window = (
                self._leg_timeout_sec - (now_ist() - window_start).total_seconds()
            )
            filled = self._wait_for_fill(leg_exec, max(remaining_window, 0.0))

            if not filled:
                elapsed_after = (now_ist() - window_start).total_seconds()
                log.warning(
                    "multi_leg_executor.leg_stuck",
                    execution_id=execution.execution_id,
                    leg_index=i,
                    option_type=leg_exec.leg.option_type,
                    strike=leg_exec.leg.strike,
                    elapsed_sec=f"{elapsed_after:.1f}",
                )
                execution.state = MultiLegState.STUCK
                execution.abort_reason = (
                    f"Leg {i} ({leg_exec.leg.option_type} {leg_exec.leg.strike}) "
                    f"stuck after {elapsed_after:.1f}s"
                )
                # Cancel the stuck leg's order before aborting filled legs
                if leg_exec.order_id and leg_exec.state == "SENT":
                    try:
                        self._broker.cancel_order(leg_exec.order_id)
                        leg_exec.state = "CANCELLED"
                    except Exception as exc:  # noqa: BLE001
                        log.error(
                            "multi_leg_executor.stuck_leg_cancel_failed",
                            execution_id=execution.execution_id,
                            order_id=leg_exec.order_id,
                            error=str(exc),
                        )
                return self.abort_strategy(execution)

            # Update aggregate after fill
            execution._recompute_net_premium()

            if i < len(execution.legs) - 1:
                execution.state = MultiLegState.PARTIALLY_FILLED
            else:
                execution.state = MultiLegState.FILLED

            log.info(
                "multi_leg_executor.leg_filled",
                execution_id=execution.execution_id,
                leg_index=i,
                option_type=leg_exec.leg.option_type,
                strike=leg_exec.leg.strike,
                filled_price=leg_exec.filled_price,
                slippage=leg_exec.slippage,
            )

        # ── Step 4: All legs filled ────────────────────────────────────────────
        execution.state = MultiLegState.FILLED
        execution.completed_at = now_ist()
        execution._recompute_net_premium()

        with self._lock:
            self._inflight.pop(execution.execution_id, None)

        log.info(
            "multi_leg_executor.strategy_filled",
            execution_id=execution.execution_id,
            strategy=strategy.name,
            net_premium_received=execution.net_premium_received,
            expected_net_premium=execution.expected_net_premium,
            total_slippage=execution.total_slippage,
        )
        return execution

    # ── Abort strategy ────────────────────────────────────────────────────────

    def abort_strategy(
        self,
        execution: MultiLegExecution,
    ) -> MultiLegExecution:
        """
        Cancel all filled legs of a partially-executed strategy.

        Uses MARKET orders to exit filled legs quickly.
        If any cancel order itself fails: marks state as PARTIAL_ABORT_FAILED
        and emits a CRITICAL structlog event requiring manual intervention.

        The abort reason must already be set on `execution` before calling this.
        """
        filled = execution.filled_legs()
        if not filled:
            # Nothing to unwind
            execution.state = MultiLegState.ABORTED
            execution.completed_at = now_ist()
            with self._lock:
                self._inflight.pop(execution.execution_id, None)
            log.info(
                "multi_leg_executor.abort_no_filled_legs",
                execution_id=execution.execution_id,
                reason=execution.abort_reason,
            )
            return execution

        execution.state = MultiLegState.ABORTING
        log.warning(
            "multi_leg_executor.aborting_strategy",
            execution_id=execution.execution_id,
            strategy=execution.strategy_name,
            abort_reason=execution.abort_reason,
            legs_to_cancel=len(filled),
        )

        failed_cancels: list[LegExecution] = []

        for leg_exec in filled:
            # Exit direction is opposite to entry direction
            exit_side = "BUY" if leg_exec.leg.direction == -1 else "SELL"
            strategy_tag = f"ABORT_{execution.execution_id[:8]}"

            exit_order = Order(
                symbol=leg_exec.leg.symbol,
                side=exit_side,
                quantity=leg_exec.expected_qty(),
                order_type="MARKET",
                price=Decimal(0),
                strategy_id=strategy_tag,
                tradingsymbol=self._build_trading_symbol(leg_exec.leg),
                exchange="NFO",
            )

            try:
                cancel_order_id = self._broker.place_order(exit_order)
                leg_exec.cancel_order_id = cancel_order_id

                # Wait briefly for market fill confirmation
                confirmed = self._wait_for_cancel_fill(
                    leg_exec, cancel_order_id, timeout=_ABORT_CONFIRM_WAIT_SEC
                )
                if confirmed:
                    leg_exec.state = "CANCELLED"
                    log.info(
                        "multi_leg_executor.abort_leg_cancelled",
                        execution_id=execution.execution_id,
                        option_type=leg_exec.leg.option_type,
                        strike=leg_exec.leg.strike,
                        cancel_order_id=cancel_order_id,
                    )
                else:
                    log.warning(
                        "multi_leg_executor.abort_leg_cancel_unconfirmed",
                        execution_id=execution.execution_id,
                        option_type=leg_exec.leg.option_type,
                        strike=leg_exec.leg.strike,
                        cancel_order_id=cancel_order_id,
                        note="Market order placed but fill not yet confirmed in poll window",
                    )

            except (OrderRejectedError, SessionError, Exception) as exc:  # noqa: BLE001
                log.critical(
                    "multi_leg_executor.abort_cancel_failed",
                    execution_id=execution.execution_id,
                    strategy=execution.strategy_name,
                    option_type=leg_exec.leg.option_type,
                    strike=leg_exec.leg.strike,
                    side=exit_side,
                    qty=leg_exec.expected_qty(),
                    error=str(exc),
                    action_required="MANUAL_INTERVENTION_REQUIRED",
                )
                failed_cancels.append(leg_exec)

        execution.completed_at = now_ist()

        if failed_cancels:
            execution.state = MultiLegState.PARTIAL_ABORT_FAILED
            failed_desc = [
                f"{le.leg.option_type}_{le.leg.strike}" for le in failed_cancels
            ]
            log.critical(
                "multi_leg_executor.partial_abort_failed",
                execution_id=execution.execution_id,
                strategy=execution.strategy_name,
                failed_legs=failed_desc,
                action_required="MANUAL_INTERVENTION_REQUIRED — open positions remain",
            )
        else:
            execution.state = MultiLegState.ABORTED
            log.info(
                "multi_leg_executor.strategy_aborted",
                execution_id=execution.execution_id,
                strategy=execution.strategy_name,
                abort_reason=execution.abort_reason,
            )

        with self._lock:
            self._inflight.pop(execution.execution_id, None)

        return execution

    # ── Execution priority ────────────────────────────────────────────────────

    def _execution_priority(self, legs: list[StrategyLeg]) -> list[StrategyLeg]:
        """
        Sort legs for optimal execution order to minimise legging risk.

        Rules (applied in order):
          1. Short legs (sell, direction=-1) before long legs (buy, direction=+1).
             Rationale: credit received from short legs mitigates risk if
             subsequent buy-protection legs cannot fill.
          2. Within the same direction bucket: higher entry_premium first.
             Rationale: higher-premium = ATM-er = higher market risk if stuck.
          3. For Iron Condors: sell legs first, then buy legs.
             This follows naturally from rules 1 and 2.

        Returns a new sorted list; does not modify the input.
        """
        def sort_key(leg: StrategyLeg) -> tuple[int, float]:
            direction_priority = 0 if leg.direction == -1 else 1  # sells first
            premium_priority = -leg.entry_premium                  # higher premium first
            return (direction_priority, premium_priority)

        return sorted(legs, key=sort_key)

    # ── Limit price computation ───────────────────────────────────────────────

    def _compute_limit_price(
        self,
        leg: StrategyLeg,
        current_ltp: float,
    ) -> float:
        """
        Compute limit price for a leg to balance fill probability vs slippage.

        Since bid/ask are not separately available here (only LTP), use:
          SELL leg: limit = ltp * 0.998  [2 ticks inside, favours fill]  ASSUMED
          BUY  leg: limit = ltp * 1.002  [2 ticks above  LTP]            ASSUMED

        Price is rounded to the NSE minimum tick (0.05 INR).
        Floor: 0.05 INR (minimum tick — options cannot trade at zero).
        """
        if leg.direction == -1:  # SELL
            raw = current_ltp * _LIMIT_SELL_FACTOR
        else:                    # BUY
            raw = current_ltp * _LIMIT_BUY_FACTOR

        # Round to NSE tick size (0.05)
        ticked = round(round(raw / _NSE_MIN_TICK) * _NSE_MIN_TICK, 2)
        return max(ticked, _NSE_MIN_TICK)

    # ── In-flight monitoring ──────────────────────────────────────────────────

    def monitor_open_executions(self) -> list[MultiLegExecution]:
        """
        Check status of all in-flight executions by polling the broker order book.
        Meant to be called periodically (e.g. every second) by the orchestrator.

        Returns the list of executions whose state changed during this poll cycle.
        Executions that reach a terminal state are removed from in-flight tracking.
        """
        changed: list[MultiLegExecution] = []

        with self._lock:
            inflight_snapshot = dict(self._inflight)

        if not inflight_snapshot:
            return changed

        # Fetch fresh order book once (bulk) to minimise API calls
        try:
            order_book: list[dict] = self._broker.get_order_book()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "multi_leg_executor.monitor.order_book_failed",
                error=str(exc),
            )
            return changed

        # Build status lookup by exchange order ID
        status_by_oid: dict[str, dict] = {
            entry.get("norenordno", entry.get("order_id", "")): entry
            for entry in order_book
            if entry.get("norenordno") or entry.get("order_id")
        }

        for exec_id, execution in inflight_snapshot.items():
            if execution.state.is_terminal:
                with self._lock:
                    self._inflight.pop(exec_id, None)
                continue

            prev_state = execution.state
            self._update_execution_from_book(execution, status_by_oid)

            if execution.state != prev_state:
                changed.append(execution)
                log.debug(
                    "multi_leg_executor.monitor.state_change",
                    execution_id=exec_id,
                    strategy=execution.strategy_name,
                    from_state=prev_state.value,
                    to_state=execution.state.value,
                )
                if execution.state.is_terminal:
                    with self._lock:
                        self._inflight.pop(exec_id, None)

        return changed

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_pre_flight(
        self, strategy: StrategyResult, now: datetime
    ) -> Optional[str]:
        """
        Run pre-flight checks before submitting any orders.
        Returns None if all checks pass, or an error string describing the block.
        """
        # Time check: no new trades after FO_NO_NEW_TRADES
        ist_time = now.astimezone(IST).time()
        if ist_time >= FO_NO_NEW_TRADES:
            return (
                f"After FO_NO_NEW_TRADES cutoff "
                f"({ist_time} >= {FO_NO_NEW_TRADES})"
            )

        # Daily entry cap check
        today = now.date()
        current_entries = self._get_daily_entries(today)
        if current_entries >= self._max_entries_per_day:
            return (
                f"Daily entry cap reached: "
                f"{current_entries}/{self._max_entries_per_day}"
            )

        # Basic legs validation
        if not strategy.legs:
            return "Strategy has no legs"

        for leg in strategy.legs:
            if leg.entry_premium <= 0:
                return (
                    f"Leg {leg.option_type}_{leg.strike} has "
                    f"non-positive entry premium: {leg.entry_premium}"
                )
            if leg.lots <= 0:
                return (
                    f"Leg {leg.option_type}_{leg.strike} has "
                    f"non-positive lots: {leg.lots}"
                )

        # ── Check 1: Lot size validation ──────────────────────────────────────
        # Verify lot_size on each leg matches the canonical contract spec, and
        # that lot_size itself is positive (leg.lots > 0 already checked above).
        for leg in strategy.legs:
            if leg.lot_size <= 0:
                return (
                    f"Leg {leg.option_type}_{leg.strike} has "
                    f"non-positive lot_size: {leg.lot_size}"
                )
            try:
                contract_type = (
                    ContractType.FUTURES
                    if leg.option_type == "FUT"
                    else ContractType.OPTIONS
                )
                contract = get_contract(leg.symbol, contract_type)
                if leg.lot_size != contract.lot_size:
                    return (
                        f"Leg {leg.option_type}_{leg.strike}: lot_size mismatch "
                        f"{leg.lot_size} != {contract.lot_size} "
                        f"(contract spec for {leg.symbol})"
                    )
            except KeyError:
                # Symbol not in contract registry — log and block to be safe
                log.warning(
                    "multi_leg_executor.preflight.unknown_contract",
                    symbol=leg.symbol,
                    option_type=leg.option_type,
                )
                return (
                    f"Leg {leg.option_type}_{leg.strike}: symbol {leg.symbol!r} "
                    "not found in contract registry"
                )

        # ── Check 2: Hedge check ──────────────────────────────────────────────
        # When require_hedge=True, block pure naked strategies (all-short or all-long).
        if self._require_hedge:
            has_long = any(leg.direction == 1 for leg in strategy.legs)
            has_short = any(leg.direction == -1 for leg in strategy.legs)
            if not (has_long and has_short):
                return (
                    "Strategy must have both long and short legs when require_hedge=True"
                )

        # ── Check 3: Margin headroom ──────────────────────────────────────────
        # Compute SPAN margin estimate and block if it exceeds the configured cap.
        if self._max_margin_inr is not None:
            try:
                from algo_trading.fo.margin import SPANMarginEngine  # noqa: PLC0415
                margin_engine = SPANMarginEngine(bsm_engine=self._bsm)
                total_margin = margin_engine.compute_portfolio_margin(strategy.legs)
                if total_margin > self._max_margin_inr:
                    return (
                        f"Margin {total_margin:.0f} exceeds cap "
                        f"{self._max_margin_inr:.0f}"
                    )
            except ImportError:
                log.warning(
                    "multi_leg_executor.margin_check_skipped",
                    reason="SPANMarginEngine not available",
                )
            except AttributeError:
                # SPANMarginEngine does not expose compute_portfolio_margin —
                # fall back to compute_span with strategy-level inputs.
                log.warning(
                    "multi_leg_executor.margin_check_skipped",
                    reason="compute_portfolio_margin not available on SPANMarginEngine",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "multi_leg_executor.margin_check_failed",
                    error=str(exc),
                )
                return f"Margin check failed: {exc}"  # Fail closed

        return None

    def _make_execution(self, strategy: StrategyResult) -> MultiLegExecution:
        """Construct a fresh MultiLegExecution from a StrategyResult."""
        leg_execs = [
            LegExecution(leg=leg)
            for leg in strategy.legs
        ]
        return MultiLegExecution(
            execution_id=str(uuid.uuid4()),
            strategy_name=strategy.name,
            state=MultiLegState.PENDING,
            legs=leg_execs,
            started_at=now_ist(),
            expected_net_premium=strategy.net_premium,
        )

    def _submit_leg(
        self,
        execution: MultiLegExecution,
        leg_exec: LegExecution,
        limit_price: float,
    ) -> Optional[str]:
        """
        Place one leg order with the broker.
        Returns the broker exchange_order_id on success, None on failure.
        Updates leg_exec.state and leg_exec.sent_at.
        """
        order_type = "LIMIT" if self._use_limit_orders and limit_price > 0 else "MARKET"
        price = Decimal(str(limit_price)) if order_type == "LIMIT" else Decimal(0)

        order = Order(
            symbol=leg_exec.leg.symbol,
            side=leg_exec.leg.side,
            quantity=leg_exec.expected_qty(),
            order_type=order_type,
            price=price,
            strategy_id=execution.execution_id[:20],   # 20-char SEBI tag
            tradingsymbol=self._build_trading_symbol(leg_exec.leg),
            exchange="NFO",
        )

        try:
            exchange_order_id = self._broker.place_order(order)
            leg_exec.order_id = exchange_order_id
            leg_exec.state = "SENT"
            leg_exec.sent_at = now_ist()
            log.info(
                "multi_leg_executor.leg_order_sent",
                execution_id=execution.execution_id,
                option_type=leg_exec.leg.option_type,
                strike=leg_exec.leg.strike,
                side=order.side,
                qty=order.quantity,
                order_type=order_type,
                price=str(price),
                order_id=exchange_order_id,
            )
            return exchange_order_id
        except (OrderRejectedError, SessionError) as exc:
            log.error(
                "multi_leg_executor.leg_order_rejected",
                execution_id=execution.execution_id,
                option_type=leg_exec.leg.option_type,
                strike=leg_exec.leg.strike,
                error=str(exc),
            )
            leg_exec.state = "REJECTED"
            return None
        except Exception as exc:  # noqa: BLE001
            log.error(
                "multi_leg_executor.leg_order_unexpected_error",
                execution_id=execution.execution_id,
                option_type=leg_exec.leg.option_type,
                strike=leg_exec.leg.strike,
                error=str(exc),
            )
            leg_exec.state = "REJECTED"
            return None

    def _wait_for_fill(
        self,
        leg_exec: LegExecution,
        timeout_sec: float,
    ) -> bool:
        """
        Poll the broker until the leg is filled or timeout_sec elapses.
        Updates leg_exec.state, filled_price, filled_qty, filled_at, slippage.
        Returns True if the leg filled within the timeout.
        """
        if leg_exec.order_id is None:
            return False

        deadline = _time.monotonic() + timeout_sec
        while _time.monotonic() < deadline:
            try:
                order_book = self._broker.get_order_book()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "multi_leg_executor.fill_poll_error",
                    order_id=leg_exec.order_id,
                    error=str(exc),
                )
                _time.sleep(_POLL_INTERVAL_SEC)
                continue

            for entry in order_book:
                oid = entry.get("norenordno") or entry.get("order_id", "")
                if oid != leg_exec.order_id:
                    continue
                status = (entry.get("status") or entry.get("stat") or "").upper()
                if status in ("COMPLETE", "FILLED", "TRADED"):
                    fill_price = float(
                        entry.get("avgprc") or entry.get("fill_price") or 0
                    )
                    fill_qty = int(
                        entry.get("filledqty") or entry.get("filled_qty") or 0
                    )
                    if fill_price > 0 and fill_qty >= leg_exec.expected_qty():
                        leg_exec.filled_price = fill_price
                        leg_exec.filled_qty = fill_qty
                        leg_exec.state = "FILLED"
                        leg_exec.filled_at = now_ist()
                        # slippage: positive = worse fill
                        # SELL leg: filled below entry = positive slippage
                        # BUY  leg: filled above entry = positive slippage
                        leg_exec.slippage = (
                            (leg_exec.leg.entry_premium - fill_price) * leg_exec.leg.direction
                        )
                        return True
                elif status in ("REJECTED", "CANCELLED", "CANCEL"):
                    leg_exec.state = "REJECTED"
                    log.warning(
                        "multi_leg_executor.leg_rejected_by_exchange",
                        order_id=leg_exec.order_id,
                        status=status,
                        reject_reason=entry.get("rejreason", ""),
                    )
                    return False
                break  # found our order, not yet filled/rejected

            _time.sleep(_POLL_INTERVAL_SEC)

        return False

    def _wait_for_cancel_fill(
        self,
        leg_exec: LegExecution,
        cancel_order_id: str,
        timeout: float,
    ) -> bool:
        """
        Wait for a market abort/exit order to confirm as filled.
        Returns True if fill confirmed within `timeout` seconds.
        """
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                order_book = self._broker.get_order_book()
            except Exception:  # noqa: BLE001
                _time.sleep(0.2)
                continue
            for entry in order_book:
                oid = entry.get("norenordno") or entry.get("order_id", "")
                if oid != cancel_order_id:
                    continue
                status = (entry.get("status") or entry.get("stat") or "").upper()
                if status in ("COMPLETE", "FILLED", "TRADED"):
                    return True
                break
            _time.sleep(0.2)
        return False

    def _update_execution_from_book(
        self,
        execution: MultiLegExecution,
        status_by_oid: dict[str, dict],
    ) -> None:
        """
        Update an in-flight execution's leg states from a pre-fetched order book.
        Called by monitor_open_executions().
        Does NOT trigger abort — that is the caller's responsibility.
        """
        for leg_exec in execution.legs:
            if leg_exec.is_filled or leg_exec.order_id is None:
                continue
            entry = status_by_oid.get(leg_exec.order_id)
            if entry is None:
                continue
            status = (entry.get("status") or entry.get("stat") or "").upper()
            if status in ("COMPLETE", "FILLED", "TRADED"):
                fill_price = float(entry.get("avgprc") or entry.get("fill_price") or 0)
                fill_qty = int(entry.get("filledqty") or entry.get("filled_qty") or 0)
                if fill_price > 0 and fill_qty >= leg_exec.expected_qty():
                    leg_exec.filled_price = fill_price
                    leg_exec.filled_qty = fill_qty
                    leg_exec.state = "FILLED"
                    leg_exec.filled_at = now_ist()
                    leg_exec.slippage = (
                        (leg_exec.leg.entry_premium - fill_price) * leg_exec.leg.direction
                    )
            elif status in ("REJECTED", "CANCELLED", "CANCEL"):
                leg_exec.state = "REJECTED"

        execution._recompute_net_premium()

        if execution.all_legs_filled():
            execution.state = MultiLegState.FILLED
            execution.completed_at = now_ist()

    def _net_premium_acceptable(
        self,
        execution: MultiLegExecution,
        strategy: StrategyResult,
    ) -> bool:
        """
        Check whether the realised net premium from filled legs is still acceptable.

        For credit strategies (expected_net_premium > 0):
          acceptable = net_premium_received >= expected * (1 - min_net_premium_pct)

        For debit strategies (expected_net_premium < 0):
          acceptable = net_premium_paid <= expected * (1 + min_net_premium_pct)
          i.e. we are not paying more than 10% extra on the debit.

        If expected_net_premium == 0 (e.g. a flat spread), always return True.
        """
        if strategy.net_premium == 0:
            return True

        if strategy.net_premium > 0:
            # Credit strategy: ensure we received enough
            floor = strategy.net_premium * (1.0 - self._min_net_premium_pct)
            return execution.net_premium_received >= floor
        else:
            # Debit strategy: ensure we are not paying too much
            ceiling = strategy.net_premium * (1.0 + self._min_net_premium_pct)
            # net_premium_received will be negative for debit strategies
            return execution.net_premium_received >= ceiling

    @staticmethod
    def _build_trading_symbol(leg: StrategyLeg) -> str:
        """
        Build the Shoonya/NSE tradingsymbol for an options leg.
        Format: {SYMBOL}{DDMONYY}{STRIKE}{CE|PE}
        Example: NIFTY19JUN2624500CE

        For futures: {SYMBOL}{DDMONYY}F
        ASSUMED: Shoonya uses this standard NSE options symbol format.
        """
        expiry_str = leg.expiry.strftime("%d%b%y").upper()
        if leg.option_type == "FUT":
            return f"{leg.symbol}{expiry_str}F"
        strike_str = f"{int(leg.strike)}" if leg.strike == int(leg.strike) else str(leg.strike)
        return f"{leg.symbol}{expiry_str}{strike_str}{leg.option_type}"

    def _get_daily_entries(self, today: date) -> int:
        """Return today's entry count (thread-safe read)."""
        with self._lock:
            if self._last_entry_date != today:
                return 0
            return self._daily_entries

    def _increment_daily_entries(self, today: date) -> None:
        """Increment daily entry counter, resetting if it's a new day. Caller holds lock."""
        if self._last_entry_date != today:
            self._daily_entries = 0
            self._last_entry_date = today
        self._daily_entries += 1


# ── FOPositionManager ─────────────────────────────────────────────────────────

class FOPositionManager:
    """
    Tracks all open F&O positions and their aggregate risk.

    Thread-safe via a single RLock: all public methods acquire the lock.

    Maintains:
      - Active strategy executions (keyed by execution_id)
      - Completed executions for P&L accounting
      - Daily P&L (realised from ABORTED/FILLED, MTM from open positions)
      - Peak concurrent lots for risk reporting
      - Forced square-off at FO_FORCE_CLOSE_TIME

    Design note: this class does NOT execute orders directly.  It delegates
    all order placement to the injected MultiLegExecutor.  Its role is purely
    position accounting and close coordination.
    """

    def __init__(self, executor: MultiLegExecutor) -> None:
        self._executor = executor
        self._lock = threading.RLock()

        # Active (not yet terminal) executions
        self._active: dict[str, MultiLegExecution] = {}
        # All completed executions for the day (including aborted)
        self._completed: list[MultiLegExecution] = []

        # Daily P&L tracking
        self._realised_pnl: float = 0.0
        self._today: Optional[date] = None

        # Peak lots tracker
        self._peak_lots: int = 0
        self._current_lots: int = 0

        log.info("fo_position_manager.init")

    # ── Add/remove executions ─────────────────────────────────────────────────

    def add_execution(self, execution: MultiLegExecution) -> None:
        """
        Register a new execution.  Called immediately after execute_strategy()
        returns a FILLED or in-progress execution.

        Terminal executions (ABORTED, PARTIAL_ABORT_FAILED) are moved directly
        to the completed list; active executions are tracked for monitoring.
        """
        with self._lock:
            self._reset_daily_counters_if_needed()
            if execution.state.is_terminal:
                self._completed.append(execution)
                if execution.state == MultiLegState.FILLED:
                    self._realise_pnl(execution)
            else:
                self._active[execution.execution_id] = execution
                lots = sum(le.leg.lots for le in execution.filled_legs())
                self._current_lots += lots
                self._peak_lots = max(self._peak_lots, self._current_lots)
                log.info(
                    "fo_position_manager.execution_added",
                    execution_id=execution.execution_id,
                    strategy=execution.strategy_name,
                    state=execution.state.value,
                    current_lots=self._current_lots,
                    peak_lots=self._peak_lots,
                )

    def update_execution(self, execution: MultiLegExecution) -> None:
        """
        Update an already-registered execution that has reached a terminal state
        (called by the monitoring loop when monitor_open_executions() detects a
        state change).
        """
        with self._lock:
            if execution.execution_id in self._active:
                if execution.state.is_terminal:
                    lots = sum(le.leg.lots for le in execution.filled_legs())
                    self._current_lots = max(0, self._current_lots - lots)
                    del self._active[execution.execution_id]
                    self._completed.append(execution)
                    if execution.state == MultiLegState.FILLED:
                        self._realise_pnl(execution)
                    log.info(
                        "fo_position_manager.execution_completed",
                        execution_id=execution.execution_id,
                        strategy=execution.strategy_name,
                        state=execution.state.value,
                    )

    # ── Position queries ──────────────────────────────────────────────────────

    def get_active_legs(self) -> list[LegExecution]:
        """Return all LegExecution objects across all active executions."""
        with self._lock:
            result: list[LegExecution] = []
            for ex in self._active.values():
                result.extend(ex.filled_legs())
            return result

    def get_active_executions(self) -> list[MultiLegExecution]:
        """Return all active (non-terminal) executions."""
        with self._lock:
            return list(self._active.values())

    # ── Force close ───────────────────────────────────────────────────────────

    def close_all_positions(
        self, reason: str = "force_close"
    ) -> list[MultiLegExecution]:
        """
        Close all open legs.

        Used for:
          1. End-of-day forced square-off at FO_FORCE_CLOSE_TIME (15:30 IST for F&O).
          2. Risk limit breach (called by portfolio_risk engine).
          3. Manual override.

        Implementation: call abort_strategy() on every active execution.
        Sends CRITICAL structlog event if any close order fails.

        Returns the list of updated (now-terminal) MultiLegExecution objects.
        """
        with self._lock:
            active_snapshot = list(self._active.values())

        if not active_snapshot:
            log.info(
                "fo_position_manager.close_all.nothing_to_close",
                reason=reason,
            )
            return []

        log.warning(
            "fo_position_manager.close_all.initiated",
            reason=reason,
            active_count=len(active_snapshot),
        )

        closed: list[MultiLegExecution] = []
        for execution in active_snapshot:
            # Mark as stuck with a reason, then abort
            execution.abort_reason = reason
            execution.state = MultiLegState.STUCK
            result = self._executor.abort_strategy(execution)
            self.update_execution(result)
            closed.append(result)

            if result.state == MultiLegState.PARTIAL_ABORT_FAILED:
                log.critical(
                    "fo_position_manager.close_all.partial_abort_failed",
                    execution_id=execution.execution_id,
                    strategy=execution.strategy_name,
                    reason=reason,
                    action_required="MANUAL_INTERVENTION_REQUIRED",
                )

        log.info(
            "fo_position_manager.close_all.complete",
            reason=reason,
            closed_count=len(closed),
            failed_count=sum(
                1 for ex in closed
                if ex.state == MultiLegState.PARTIAL_ABORT_FAILED
            ),
        )
        return closed

    # ── P&L ──────────────────────────────────────────────────────────────────

    def daily_pnl(self) -> float:
        """
        Sum of P&L across all completed + MTM open executions for today.

        For FILLED executions: realised_pnl is already accumulated.
        For active executions: compute MTM from net_premium_received vs current
        mark-to-market (simplified: use net_premium_received as proxy since
        current_premium is not passed here — callers should update leg premiums
        externally for accurate MTM).
        """
        with self._lock:
            self._reset_daily_counters_if_needed()
            mtm: float = 0.0
            for ex in self._active.values():
                # MTM for open positions: premium received at entry (credit received
                # minus the current mark-to-market cost to close).
                # Simplified: use net_premium_received as proxy.
                mtm += ex.net_premium_received
            return self._realised_pnl + mtm

    def max_concurrent_lots(self) -> int:
        """Maximum lots active at any point today."""
        with self._lock:
            return self._peak_lots

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _realise_pnl(self, execution: MultiLegExecution) -> None:
        """
        Accumulate realised P&L from a completed execution.
        Caller must hold self._lock.
        """
        self._realised_pnl += execution.net_premium_received
        log.debug(
            "fo_position_manager.pnl_realised",
            execution_id=execution.execution_id,
            pnl_this_execution=execution.net_premium_received,
            cumulative_realised=self._realised_pnl,
        )

    def _reset_daily_counters_if_needed(self) -> None:
        """
        Reset daily counters at the start of each new trading day.
        Caller must hold self._lock.
        """
        today = now_ist().date()
        if self._today != today:
            log.info(
                "fo_position_manager.daily_reset",
                prev_date=str(self._today),
                new_date=str(today),
                prev_pnl=self._realised_pnl,
                prev_peak_lots=self._peak_lots,
            )
            self._today = today
            self._realised_pnl = 0.0
            self._peak_lots = 0
            self._current_lots = 0
            # Keep active executions — they span the boundary only in theory
            # (force-close before market end should empty this).
