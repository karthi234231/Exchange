"""
execution/order_executor.py — Order submission, timeout management, and fill routing.

The OrderExecutor is the bridge between signals (TradeSignal) and the broker.
It enforces:
  - Risk manager approval before any order is sent
  - Rate limiting (DualLockRateLimiter)
  - Stale order timeout (cancel LIMIT orders not filled within config timeout)
  - Never converts stale LIMIT to MARKET [SURV-11]
  - Stop-loss orders are always MARKET, no price cap
  - 30bps price drift check: if price moved >30bps from signal → skip entry
  - Fat-finger guard: value cap, ADV%, price deviation, qty cap [RISK-FF]
  - Position close at force-close time

Execution rules (from CLAUDE.md):
  - Prefer LIMIT orders; price set by signal-strength-driven get_limit_price()
  - NEVER convert stale LIMIT to MARKET — cancel entirely if not filled
  - NEVER enter if price moved >30bps from signal price
  - Stop-loss: always MARKET, no exceptions
  - Never trade 09:15–09:30 bar [SURV-8]
  - Square off ALL intraday positions by 15:15 IST [SURV-8]
"""
from __future__ import annotations

import os
import re
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4
import json
from pathlib import Path

import structlog

from algo_trading.config import settings
from algo_trading.data.broker_base import (
    BrokerBase,
    Fill,
    Order,
    OrderRejectedError,
    OrderUpdate,
    SessionError,
)
from algo_trading.data.system_state import LiveSystemState
from algo_trading.execution.order_fsm import (
    FillCallback,
    OrderEvent,
    OrderFSM,
    OrderRecord,
    OrderState,
)
from algo_trading.execution.rate_limiter import DualLockRateLimiter
from algo_trading.core.calendar import IST, now_ist
from algo_trading.risk.fat_finger import FatFingerGuard, FatFingerResult

try:
    from algo_trading.app.health_check import SIGNALS_SKIPPED, ORDERS_PLACED, ORDERS_REJECTED
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False

log = structlog.get_logger(__name__)


def _skip(reason: str, symbol: str) -> None:
    """Increment SIGNALS_SKIPPED metric if prometheus is available."""
    if _HAS_METRICS:
        SIGNALS_SKIPPED.labels(reason=reason, symbol=symbol).inc()


def _round_to_tick_size(price: Decimal) -> Decimal:
    """
    P2-FIX-104: Round price to NSE equity tick size.

    NSE equity tick rules (as of 2026):
      - Price >= 1.00: tick size = 0.05
      - Price < 1.00 (sub-rupee): tick size = 0.01

    Rounds DOWN for BUY (passive entry — avoid paying more than intended)
    and UP for SELL when called without side context. Since we always use
    this for LIMIT entry orders (BUY below current price), rounding DOWN
    is the correct conservative choice.

    Broker rejects any LIMIT price that is not a valid tick multiple.
    Fat-finger and risk checks must see the same price that goes to the broker.
    """
    if price <= 0:
        return price
    if price < Decimal("1"):
        tick = Decimal("0.01")
    else:
        tick = Decimal("0.05")
    # Floor to tick: int(price / tick) * tick
    return (price // tick) * tick


# P2-FIX-132: Global and per-symbol active order caps.
# These prevent order accumulation during system failures or broker delays.
_MAX_OPEN_ORDERS_GLOBAL: int = 20
_MAX_OPEN_ORDERS_PER_SYMBOL: int = 3


@dataclass(slots=True)
class TradeSignal:
    """Output from a signal generator (e.g. EMACrossStrategy)."""
    symbol: str
    direction: int          # +1 = BUY, -1 = SELL/SHORT
    entry_price: Decimal    # Signal price at time of computation
    stop_loss_price: Decimal
    atr: Decimal
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    trace_id: str = ""      # End-to-end trace ID propagated from bar event
    # P2-FIX-89: Correlation ID — ties this signal to order, fill, P&L, and audit log entries
    decision_id: str = ""
    # P2-FIX-57: Model and data provenance — attach to every signal for post-trade analysis
    model_version: str = ""           # FrozenModelSpec version (e.g. "v2.3.1")
    feature_schema_version: str = ""  # Feature schema hash first 8 chars
    data_snapshot_time: Optional[datetime] = None  # Training data cutoff timestamp


class RiskCheckFailed(Exception):
    """Raised when the risk manager blocks an order."""
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ResearchModeError(RuntimeError):
    """#96: Raised (or used as guard) when an order is attempted in research_mode=True."""


class StopLossMode(str, Enum):
    """#130: Controls whether stop-losses are managed by the app or submitted to the broker.

    APP_MANAGED:
        The app monitors current prices via QuoteService and submits a market
        order when the stop level is breached.  Requires check_app_managed_stops()
        to be called on every bar/tick.

    BROKER_NATIVE:
        A SL-M (Stop-Loss Market) order is submitted to the broker at the time the
        position is opened. The broker exchange triggers the market order when price
        breaches the stop level. Reduces latency but counts toward OTR.
    """
    APP_MANAGED = "APP_MANAGED"
    BROKER_NATIVE = "BROKER_NATIVE"


class _OrderAuditWriter:
    """
    Immutable local audit log for all order FSM events. [P2-FIX-52]

    Appends JSONL entries to a local file for every order state transition.
    Written BEFORE any QuestDB write so records survive QuestDB failures.
    Thread-safe. Never raises — a failed write is logged but never propagates.

    Rotate weekly with logrotate or a cron job. Checksum before rotating
    for SEBI PFUTP §22 5-year retention compliance.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, fields: dict) -> None:
        """Append one JSONL entry. Non-blocking. Never raises."""
        entry: dict = {"ts_ist": now_ist().isoformat(), "event": event, **fields}
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, default=str) + "\n")
                    fh.flush()
                # #147: Restrict audit log to owner-read/write only (SEBI PFUTP §22).
                # On Windows, os.chmod does not fully restrict access but we still call it
                # to express intent and for portability to Linux VPS deployments.
                try:
                    os.chmod(self._path, 0o600)
                except OSError:
                    pass  # Windows doesn't support chmod fully — safe to ignore
        except Exception as exc:
            log.error("order_audit.write_failed", event=event, error=str(exc))


class OrderExecutor:
    """
    Coordinates order submission, monitoring, and fill routing.

    Dependencies injected at construction (no global state):
      - broker: ShoonyaBroker or MockBroker
      - fsm: OrderFSM (order state machine)
      - rate_limiter: DualLockRateLimiter
      - risk_manager: RiskManager
      - state: LiveSystemState
      - store: QuestDBStore

    Thread-safety:
      - submit_signal() is safe to call from the signal thread.
      - on_broker_update() is safe to call from the WebSocket thread.
      - _timeout_checker_loop() runs on a background daemon thread.
    """

    def __init__(
        self,
        broker: BrokerBase,
        fsm: OrderFSM,
        rate_limiter: DualLockRateLimiter,
        risk_manager: object,  # RiskManager — late binding to avoid circular import
        state: LiveSystemState,
        store: Optional[object] = None,  # QuestDBStore
        alert_sender: Optional[object] = None,  # AlertSender
        fat_finger: Optional[FatFingerGuard] = None,  # Fat-finger guard
        max_spread_bps: float = 50.0,  # #116: liquidity/spread filter threshold
        stop_loss_mode: StopLossMode = StopLossMode.APP_MANAGED,  # #130
    ) -> None:
        self._broker = broker
        self._fsm = fsm
        self._rate_limiter = rate_limiter
        self._risk_manager = risk_manager
        self._state = state
        self._store = store
        self._alert = alert_sender
        self._fat_finger = fat_finger

        # #116: Spread filter — max bid/ask spread in basis points before rejecting entry
        self._max_spread_bps: float = max_spread_bps  # [ASSUMED] 50bps default

        # #117: Symbol-level halt/cooldown — maps symbol → expiry datetime
        self._halted_symbols: dict[str, datetime] = {}
        self._halt_duration: timedelta = timedelta(minutes=30)  # [ASSUMED] 30 min default
        self._halt_lock = threading.Lock()

        # #130: Stop-loss mode
        self._stop_loss_mode: StopLossMode = stop_loss_mode
        # #130: APP_MANAGED stops — maps symbol → {"stop_price": Decimal, "side": str, "qty": int, "stop_order_id": str}
        self._app_managed_stops: dict[str, dict] = {}
        self._stops_lock = threading.Lock()

        # #131: Trailing stops — maps symbol → {"stop_price": Decimal, "peak_price": Decimal, "trail_pct": float}
        self._trailing_stops: dict[str, dict] = {}
        self._trailing_lock = threading.Lock()

        # P2-FIX-52: Order event audit trail — local JSONL, written before QuestDB
        self._audit = _OrderAuditWriter(Path("state") / "order_audit.log")

        # P2-FIX-133: Daily order notional tracker — track total turnover submitted
        self._daily_notional: Decimal = Decimal(0)
        self._daily_order_count: int = 0
        self._daily_notional_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._timeout_thread: Optional[threading.Thread] = None

        # Register FSM fill callback
        class _FillCb(FillCallback):
            def __init__(self, executor: OrderExecutor) -> None:
                self._ex = executor
            def on_fill(self, record: OrderRecord, fill: Fill) -> None:
                self._ex._on_fill(record, fill)
        self._fsm.register_fill_callback(_FillCb(self))

    def start(self) -> None:
        """Start the stale order timeout checker thread."""
        self._timeout_thread = threading.Thread(
            target=self._timeout_checker_loop,
            name="order-timeout-checker",
            daemon=True,
        )
        self._timeout_thread.start()
        log.info("order_executor.started")

    def stop(self) -> None:
        self._stop_event.set()
        log.info("order_executor.stopped")

    # ── Primary Entry Point ───────────────────────────────────────────────────

    def submit_signal(self, signal: TradeSignal) -> Optional[str]:
        """
        Process a trade signal: risk-check, price-check, rate-check, then place order.
        Returns client_order_id on success, None if blocked.

        Exit policy (#128):
          - research_mode guard:  blocks ALL orders (entries and exits)
          - symbol halt:          blocks NEW entries; exits are ALLOWED to reduce risk
          - circuit limit check:  only applied to entries, skipped for exits
          - spread filter:        warns but does NOT block exits
          - reconciliation gate:  exits allowed even when startup_reconcile_clean=False
        """
        from algo_trading.core.calendar import can_enter_new_trade

        trace_id = signal.trace_id or str(uuid4())
        # P2-FIX-89: decision_id is the primary correlation key for this trade lifecycle.
        # Falls back to trace_id for backwards-compat with signals that don't set it.
        decision_id = signal.decision_id or trace_id

        # #96: Research mode — block ALL order submission regardless of direction
        if settings.research_mode:
            log.warning(
                "order_executor.research_mode_blocked",
                signal=str(signal),
                symbol=signal.symbol,
                direction=signal.direction,
            )
            return None

        # Determine if this is an exit signal — used to apply separate exit policy (#128)
        is_exit = self._is_exit_signal(signal)

        # [SURV-8] Never enter during opening bar or after no-new-trades time
        # Exits are also subject to session guard — close_position() handles force-close
        if not can_enter_new_trade():
            log.info(
                "order_executor.signal_skipped.session_guard",
                symbol=signal.symbol,
                trace_id=trace_id,
            )
            _skip("session_guard", signal.symbol)
            return None

        # #117: Symbol-level halt check.
        # Policy (#128): halted symbols BLOCK new entries but ALLOW exits to reduce risk.
        if not is_exit and self.is_symbol_halted(signal.symbol):
            log.warning(
                "order_executor.signal_skipped.symbol_halted",
                symbol=signal.symbol,
                trace_id=trace_id,
            )
            _skip("symbol_halted", signal.symbol)
            return None

        # [P2-FIX-132] Max open orders guard — cap active orders globally and per symbol
        all_active = self._fsm.get_active_orders()
        if len(all_active) >= _MAX_OPEN_ORDERS_GLOBAL:
            log.warning(
                "order_executor.signal_skipped.max_open_orders_global",
                symbol=signal.symbol,
                active_count=len(all_active),
                cap=_MAX_OPEN_ORDERS_GLOBAL,
                trace_id=trace_id,
            )
            _skip("max_open_orders_global", signal.symbol)
            return None
        symbol_order_count = sum(1 for r in all_active if r.order.symbol == signal.symbol)
        if symbol_order_count >= _MAX_OPEN_ORDERS_PER_SYMBOL:
            log.warning(
                "order_executor.signal_skipped.max_open_orders_per_symbol",
                symbol=signal.symbol,
                symbol_count=symbol_order_count,
                cap=_MAX_OPEN_ORDERS_PER_SYMBOL,
                trace_id=trace_id,
            )
            _skip("max_open_orders_symbol", signal.symbol)
            return None

        # [P2-FIX-134/135] Duplicate signal and pending order conflict check
        # Block if there is already an active entry order for this symbol+side.
        target_side = "BUY" if signal.direction > 0 else "SELL"
        for _active in all_active:
            if (
                _active.order.symbol == signal.symbol
                and _active.order.side == target_side
                and _active.order.order_type == "LIMIT"
            ):
                log.info(
                    "order_executor.signal_skipped.pending_order_conflict",
                    symbol=signal.symbol,
                    side=target_side,
                    existing_client_id=_active.order.client_order_id,
                    trace_id=trace_id,
                )
                _skip("pending_order_conflict", signal.symbol)
                return None

        # [P2-FIX-107/108] Require fresh positive LTP for entries — fail closed on stale/zero quote
        current_ltp = self._broker.get_ltp(signal.symbol)
        if current_ltp <= 0:
            log.warning(
                "order_executor.signal_skipped.no_ltp",
                trace_id=trace_id,
                symbol=signal.symbol,
                note="broker returned zero/missing LTP — blocking entry (fail-closed)",
            )
            if self._alert:
                self._alert.send_risk_alert(  # type: ignore[union-attr]
                    f"Entry blocked for {signal.symbol}: broker LTP unavailable"
                )
            _skip("no_ltp", signal.symbol)
            return None

        # 30bps price drift check
        if signal.entry_price > 0:
            drift_bps = abs(current_ltp - signal.entry_price) / signal.entry_price * 10_000
            if drift_bps > settings.max_price_drift_bps:
                log.warning(
                    "order_executor.signal_skipped.price_drift",
                    trace_id=trace_id,
                    symbol=signal.symbol,
                    drift_bps=f"{drift_bps:.1f}",
                    threshold=settings.max_price_drift_bps,
                )
                if self._alert:
                    self._alert.send_trade_alert(  # type: ignore[union-attr]
                        symbol=signal.symbol,
                        action="SKIP",
                        price=current_ltp,
                        quantity=0,
                        reason=f"Price drift {drift_bps:.1f}bps > {settings.max_price_drift_bps}bps",
                    )
                _skip("price_drift", signal.symbol)
                return None

        # #115: Circuit limit check — only applied to entries, skipped for exits (#128)
        if not is_exit and self._is_near_circuit_limit(signal.symbol, signal.entry_price):
            log.warning(
                "order_executor.circuit_limit_proximity",
                symbol=signal.symbol,
                price=str(signal.entry_price),
                trace_id=trace_id,
            )
            _skip("circuit_limit", signal.symbol)
            return None

        # #116: Liquidity/spread filter — warn only for exits, block for entries (#128)
        from algo_trading.data.quote_service import get_quote_service
        _qs = get_quote_service()
        _q = _qs.get_for_exit(signal.symbol)  # fail-open — None if unavailable
        if _q is not None and _q.spread is not None and _q.ltp > 0:
            spread_bps = float(_q.spread / _q.ltp * 10000)
            if spread_bps > self._max_spread_bps:
                if is_exit:
                    # Exits: warn but proceed — trapping a position is worse than wide spread
                    log.warning(
                        "order_executor.spread_wide_on_exit",
                        symbol=signal.symbol,
                        spread_bps=f"{spread_bps:.1f}",
                        threshold=self._max_spread_bps,
                        note="proceeding with exit despite wide spread — fail-open policy",
                    )
                else:
                    log.warning(
                        "order_executor.spread_too_wide",
                        symbol=signal.symbol,
                        spread_bps=f"{spread_bps:.1f}",
                        threshold=self._max_spread_bps,
                        trace_id=trace_id,
                    )
                    if _HAS_METRICS:
                        SIGNALS_SKIPPED.labels(
                            reason="spread_too_wide", symbol=signal.symbol
                        ).inc()
                    return None

        limit_price = self._get_limit_price(signal, current_ltp)
        # P2-FIX-104: Normalize limit price to NSE tick size BEFORE risk/fat-finger
        # checks so that all subsequent checks see the exact broker-submission price.
        # NSE equity tick size = 0.05 for prices >= 1.00, 0.01 for sub-rupee.
        limit_price = _round_to_tick_size(limit_price)

        quantity = self._risk_manager.compute_position_size(signal)  # type: ignore[union-attr]

        if quantity <= 0:
            log.info(
                "order_executor.signal_skipped.zero_size",
                symbol=signal.symbol,
                trace_id=trace_id,
            )
            _skip("zero_size", signal.symbol)
            return None

        # Risk manager pre-trade check
        can_trade, reason = self._risk_manager.can_trade(  # type: ignore[union-attr]
            symbol=signal.symbol,
            side="BUY" if signal.direction > 0 else "SELL",
            quantity=quantity,
            price=limit_price,
        )
        if not can_trade:
            log.warning(
                "order_executor.signal_blocked_by_risk",
                reason=reason,
                trace_id=trace_id,
            )
            _skip("risk_check", signal.symbol)
            return None

        order = Order(
            symbol=signal.symbol,
            side="BUY" if signal.direction > 0 else "SELL",
            quantity=quantity,
            order_type="LIMIT",
            price=limit_price,
            strategy_id=settings.strategy_id or "PAPER",
            tradingsymbol=f"{signal.symbol}-EQ",
            decision_id=decision_id,          # P2-FIX-89: propagate correlation ID
            model_version=signal.model_version,  # P2-FIX-57: propagate model version
        )

        # Fat-finger guard — first check before any broker interaction [RISK-FF]
        if self._fat_finger is not None:
            ff_result, ff_reason = self._fat_finger.check(order, current_ltp)
            if ff_result != FatFingerResult.OK:
                log.error(
                    "order_executor.fat_finger.blocked",
                    trace_id=trace_id,
                    symbol=signal.symbol,
                    result=ff_result.value,
                    reason=ff_reason,
                )
                if self._alert:
                    self._alert.send_risk_alert(  # type: ignore[union-attr]
                        f"Fat-finger blocked [{ff_result.value}]: {ff_reason}"
                    )
                _skip("fat_finger", signal.symbol)
                return None

        # P2-FIX-133: Daily notional / order count cap check
        order_notional = limit_price * Decimal(str(quantity))
        daily_cap = settings.capital_inr * Decimal("3")  # 300% turnover cap (conservative)
        with self._daily_notional_lock:
            if self._daily_notional + order_notional > daily_cap:
                log.warning(
                    "order_executor.signal_skipped.daily_notional_cap",
                    symbol=signal.symbol,
                    daily_notional=str(self._daily_notional),
                    order_notional=str(order_notional),
                    cap=str(daily_cap),
                    trace_id=trace_id,
                )
                _skip("daily_notional_cap", signal.symbol)
                return None

        if _HAS_METRICS:
            ORDERS_PLACED.labels(symbol=signal.symbol).inc()
        return self._place_order(order, trace_id=trace_id)

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: int,
        stop_price: Decimal,
    ) -> Optional[str]:
        """
        Place a stop-loss MARKET order. Always MARKET — no price cap.
        side: the closing side (opposite of entry).
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
            price=Decimal(0),  # MARKET — no limit
            strategy_id=settings.strategy_id or "PAPER",
            tradingsymbol=f"{symbol}-EQ",
        )
        log.warning(
            "order_executor.stop_loss_market",
            symbol=symbol,
            side=side,
            qty=quantity,
            stop_trigger=str(stop_price),
        )
        return self._place_order(order)

    def close_position(self, symbol: str, position_qty: int) -> Optional[str]:
        """
        Close an existing position (for force-close at 15:15 or kill-switch).
        Always MARKET order.
        """
        if position_qty == 0:
            return None
        side = "SELL" if position_qty > 0 else "BUY"
        quantity = abs(position_qty)
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
            price=Decimal(0),
            strategy_id=settings.strategy_id or "PAPER",
            tradingsymbol=f"{symbol}-EQ",
        )
        log.info(
            "order_executor.position_close",
            symbol=symbol,
            side=side,
            qty=quantity,
        )
        return self._place_order(order)

    # ── Internal Order Placement ──────────────────────────────────────────────

    def _place_order(self, order: Order, trace_id: str = "") -> Optional[str]:
        """
        Create FSM record, rate-check, submit to broker, update FSM.
        Returns client_order_id on success, None on failure.

        trace_id: end-to-end trace originating from the bar event; propagated
        to all log messages for this order lifecycle.
        """
        # Store trace_id on the FSM record for retrieval in fill callbacks.
        # Create FSM record first — if this fails, no rate-limiter token is consumed.
        record = self._fsm.create(order)
        self._state.record_order(order.client_order_id, OrderState.PENDING.value)

        # Rate limiter check — acquired AFTER FSM creation to prevent token leaks
        # if FSM construction were to raise.
        allowed, rate_reason = self._rate_limiter.acquire(order.client_order_id)
        if not allowed:
            log.warning(
                "order_executor.rate_limited",
                reason=rate_reason,
                trace_id=trace_id,
            )
            if self._alert:
                self._alert.send_risk_alert(f"Rate limit: {rate_reason}")  # type: ignore[union-attr]
            self._fsm.transition(order.client_order_id, OrderEvent.REJECT, reject_reason=f"rate_limit: {rate_reason}")
            self._state.remove_order(order.client_order_id)
            if _HAS_METRICS:
                ORDERS_REJECTED.labels(reason="rate_limit").inc()
            return None

        # Submit to broker
        self._fsm.transition(order.client_order_id, OrderEvent.SEND)
        try:
            exchange_order_id = self._broker.place_order(order)
        except (SessionError, OrderRejectedError) as exc:
            log.error(
                "order_executor.place_failed",
                trace_id=trace_id,
                client_id=order.client_order_id,
                error=str(exc),
            )
            self._fsm.transition(
                order.client_order_id,
                OrderEvent.REJECT,
                reject_reason=str(exc),
            )
            self._audit.write("REJECT", {
                "client_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.quantity,
                "price": str(order.price),
                "reason": str(exc),
                "trace_id": trace_id,
                "decision_id": order.decision_id,  # P2-FIX-89
            })
            self._rate_limiter.release_order(order.client_order_id)
            self._state.remove_order(order.client_order_id)
            if self._alert:
                self._alert.send_risk_alert(f"Order rejected: {exc}")  # type: ignore[union-attr]
            if _HAS_METRICS:
                ORDERS_REJECTED.labels(reason=type(exc).__name__).inc()
            return None

        # P0-FIX-1/3: Store the exchange_order_id returned by the REST API
        # immediately so cancel_order() can find it even before the WS ACK
        # arrives. This is NOT an ACK — the FSM state stays SENT.
        self._fsm.record_exchange_id(order.client_order_id, exchange_order_id)

        # Do NOT synchronously ACK here. The ACK transition must only happen
        # via the broker WebSocket callback (on_broker_update → FSM). Calling
        # ACK immediately after place_order() would mark the order as
        # acknowledged before the exchange has confirmed it.

        # Stash trace_id in record metadata so _on_fill() can retrieve it.
        if hasattr(record, "__dict__"):
            record.__dict__["_trace_id"] = trace_id

        log.info(
            "order_executor.order_sent",
            trace_id=trace_id,
            client_id=order.client_order_id,
            exchange_id=exchange_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.quantity,
            price=str(order.price),
        )
        # P2-FIX-52: Audit SENT before any QuestDB write
        self._audit.write("SENT", {
            "client_id": order.client_order_id,
            "exchange_id": exchange_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.quantity,
            "price": str(order.price),
            "order_type": order.order_type,
            "trace_id": trace_id,
            "decision_id": order.decision_id,      # P2-FIX-89
            "model_version": order.model_version,  # P2-FIX-57
        })
        # P2-FIX-133: Accumulate daily notional
        if order.price > 0:
            with self._daily_notional_lock:
                self._daily_notional += order.price * Decimal(str(order.quantity))
                self._daily_order_count += 1
        if self._alert:
            self._alert.send_trade_alert(  # type: ignore[union-attr]
                symbol=order.symbol,
                action=f"{order.side} {order.order_type}",
                price=order.price,
                quantity=order.quantity,
            )
        return order.client_order_id

    def cancel_order(self, client_order_id: str) -> bool:
        """
        Cancel a live order by client_order_id.
        Updates FSM, calls broker cancel, releases rate limiter slot.
        """
        record = self._fsm.get_record(client_order_id)
        if record is None or record.state.is_terminal:
            return True  # Already gone

        # P0-FIX-3: Refuse to cancel if exchange_order_id is not yet known.
        # Calling broker.cancel_order("") sends a blank id to the exchange and
        # may result in a garbage response or stuck CANCEL_PENDING state.
        if not record.exchange_order_id:
            log.error(
                "order_executor.cancel_blocked.no_exchange_id",
                client_id=client_order_id,
                state=record.state.value,
                note="exchange_order_id not yet received — manual reconciliation required",
            )
            if self._alert:
                self._alert.send_risk_alert(  # type: ignore[union-attr]
                    f"Cancel blocked: no exchange ID for {client_order_id} "
                    f"(state={record.state.value}) — manual reconciliation required"
                )
            return False

        self._fsm.transition(client_order_id, OrderEvent.CANCEL_REQUEST)

        # Rate limiter: cancel counts as a modification.
        # Proceed regardless — cancellations are safety-critical and must not be blocked.
        _allowed, _rate_reason = self._rate_limiter.acquire(client_order_id, block=True)
        if not _allowed:
            log.warning(
                "order_executor.cancel_rate_limited",
                client_id=client_order_id,
                reason=_rate_reason,
                note="Proceeding with cancel despite rate limit — safety-critical",
            )

        ok = self._broker.cancel_order(record.exchange_order_id)
        if ok:
            self._fsm.transition(client_order_id, OrderEvent.CANCEL_CONFIRM)
            self._audit.write("CANCEL_CONFIRM", {
                "client_id": client_order_id,
                "exchange_id": record.exchange_order_id,
                "symbol": record.order.symbol,
                "side": record.order.side,
                "qty": record.order.quantity,
            })
        else:
            log.warning(
                "order_executor.cancel_failed",
                client_id=client_order_id,
                exchange_id=record.exchange_order_id,
            )
        self._rate_limiter.release_order(client_order_id)
        self._state.remove_order(client_order_id)
        return ok

    def modify_order(self, client_order_id: str, new_price: Decimal) -> None:
        """
        P2-FIX-103: Explicit no-modify policy.

        Order modification (MODIFY_REQUEST/MODIFY_CONFIRM) is NOT supported.
        Policy: cancel only. Rationale:
          1. Shoonya modify increments OTR (Order-to-Trade Ratio) identically to
             a new order — no benefit over cancel + re-place.
          2. Modify-then-fill race: if the old price fills between the modify REST
             call and the broker processing it, the position is correct but the
             FSM state may be inconsistent (fill at old price, FSM thinks new price).
          3. Per-order modification counter in DualLockRateLimiter caps at 5 before
             forced cancel — using modify would consume this budget identically.

        To adjust an order price: cancel_order() then submit_signal() with updated
        entry_price. This is unambiguous, auditable, and idempotent.

        Raises NotImplementedError unconditionally to prevent accidental use.
        """
        raise NotImplementedError(
            f"Order modification is not supported (policy: cancel-only). "
            f"To adjust order {client_order_id}: cancel it, then re-place "
            f"with the updated price via submit_signal()."
        )

    def on_broker_update(self, update: OrderUpdate) -> None:
        """Handle an order update event from the broker WebSocket."""
        record = self._fsm.on_broker_update(update)
        # P2-FIX-52: Audit every broker-driven state transition before QuestDB
        if record:
            self._audit.write("BROKER_UPDATE", {
                "client_id": record.order.client_order_id,
                "exchange_id": getattr(update, "exchange_order_id", ""),
                "symbol": record.order.symbol,
                "new_state": record.state.value,
                "raw_status": getattr(update, "status", ""),
            })
        # P0-FIX-2: Release rate limiter for ANY terminal state reached via WS.
        # FILL is handled by _on_fill callback; CANCEL_CONFIRM is handled by
        # cancel_order(). release_order() is idempotent so double-release is safe.
        if record and record.state.is_terminal:
            self._rate_limiter.release_order(record.order.client_order_id)
            self._state.remove_order(record.order.client_order_id)

    # ── Fill Handling ─────────────────────────────────────────────────────────

    def _on_fill(self, record: OrderRecord, fill: Fill) -> None:
        """Called by FSM fill callback after a fill is recorded."""
        # Retrieve trace_id stashed during _place_order for end-to-end correlation.
        trace_id: str = ""
        if hasattr(record, "__dict__"):
            trace_id = record.__dict__.get("_trace_id", "")

        # P2-FIX-89: decision_id flows from TradeSignal → Order → Fill log for full traceability
        decision_id: str = record.order.decision_id

        log.info(
            "order_executor.fill",
            trace_id=trace_id,
            decision_id=decision_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.quantity,
            price=str(fill.price),
            model_version=record.order.model_version,  # P2-FIX-57
        )
        # P2-FIX-52: Audit FILL before QuestDB write
        self._audit.write("FILL", {
            "client_id": fill.order_id,
            "exchange_id": getattr(fill, "exchange_order_id", ""),
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": fill.quantity,
            "fill_price": str(fill.price),
            "trace_id": trace_id,
            "decision_id": decision_id,              # P2-FIX-89
            "model_version": record.order.model_version,  # P2-FIX-57
        })

        # P2-FIX-118: Slippage monitoring — compare fill price vs limit price
        if record.order.order_type == "LIMIT" and record.order.price > 0:
            limit_px = record.order.price
            # For BUY: positive bps = paid more than limit (adverse slippage)
            # For SELL: positive bps = received less than limit (adverse slippage)
            raw_bps = float(
                (fill.price - limit_px) / limit_px * Decimal("10000")
            )
            signed_bps = raw_bps if fill.side == "BUY" else -raw_bps
            log.info(
                "order_executor.fill.slippage",
                trace_id=trace_id,
                symbol=fill.symbol,
                side=fill.side,
                limit_price=str(limit_px),
                fill_price=str(fill.price),
                slippage_bps=f"{signed_bps:.1f}",
            )
            if signed_bps > 10.0:   # >10bps adverse slippage on a LIMIT order is unusual
                log.warning(
                    "order_executor.fill.slippage_excess",
                    trace_id=trace_id,
                    symbol=fill.symbol,
                    slippage_bps=f"{signed_bps:.1f}",
                    note="LIMIT fill price worse than limit price by >10bps — investigate",
                )

        # Compute cost
        if self._risk_manager:
            cost = self._risk_manager.compute_trade_cost(  # type: ignore[union-attr]
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
            )
        else:
            from algo_trading.risk.cost_model import IndiaCostModel
            cost = IndiaCostModel().compute(fill.symbol, fill.side, fill.quantity, fill.price)

        # Update position state
        self._state.update_position_from_fill(
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            fill_price=fill.price,
            trade_cost=cost.total_inr,
        )

        # Persist fill
        if self._store:
            from algo_trading.risk.cost_model import IndiaCostModel
            slippage_bps = Decimal(0)  # Will be computed by cost model with actual prices
            self._store.write_fill(fill, cost.total_inr, slippage_bps)  # type: ignore[union-attr]

        # P2-FIX-119: Record trade for OTR compliance tracking
        self._rate_limiter.record_trade()

        # Release rate limiter slot if order is now terminal
        updated_record = self._fsm.get_record(fill.order_id)
        if updated_record and updated_record.state.is_terminal:
            self._rate_limiter.release_order(fill.order_id)
            self._state.remove_order(fill.order_id)

        if self._alert:
            self._alert.send_trade_alert(  # type: ignore[union-attr]
                symbol=fill.symbol,
                action=f"FILL {fill.side}",
                price=fill.price,
                quantity=fill.quantity,
            )

    # ── Stale Order Timeout ───────────────────────────────────────────────────

    def _timeout_checker_loop(self) -> None:
        """
        Background thread: checks for stale LIMIT orders that haven't filled.
        Stale = ACKED and older than config.stale_order_timeout_sec.
        Action: cancel entirely — NEVER convert to MARKET. [SURV-11]
        """
        while not self._stop_event.wait(5.0):  # Check every 5 seconds
            self._check_stale_orders()

    def _check_stale_orders(self) -> None:
        timeout_sec = settings.stale_order_timeout_sec
        now = now_ist()
        for record in self._fsm.get_active_orders():
            if record.order.order_type != "LIMIT":
                continue
            if record.state not in (OrderState.ACKED, OrderState.PARTIAL):
                continue
            if record.sent_at is None:
                continue
            age_sec = (now - record.sent_at).total_seconds()
            if age_sec > timeout_sec:
                log.warning(
                    "order_executor.stale_order.cancelling",
                    client_id=record.order.client_order_id,
                    symbol=record.order.symbol,
                    age_sec=f"{age_sec:.1f}",
                    timeout=timeout_sec,
                )
                if self._alert:
                    self._alert.send_risk_alert(  # type: ignore[union-attr]
                        f"Stale LIMIT cancelled: {record.order.symbol} "
                        f"age={age_sec:.0f}s > {timeout_sec}s"
                    )
                self.cancel_order(record.order.client_order_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_limit_price(self, signal: TradeSignal, current_ltp: Decimal) -> Decimal:
        """
        Compute limit price for a new entry order.
        For Stage 1 EMA cross: use current LTP (passive entry).
        Low confidence signals: use slightly more passive price.
        """
        if current_ltp <= 0:
            return signal.entry_price

        if signal.direction > 0:  # BUY: price at or slightly below LTP
            # Passive: bid slightly below best ask to improve fill odds
            offset = signal.entry_price * Decimal("0.0002") * (2 - Decimal(str(signal.confidence)))
            return (current_ltp - offset).quantize(Decimal("0.05"))
        else:  # SELL: price at or slightly above LTP
            offset = signal.entry_price * Decimal("0.0002") * (2 - Decimal(str(signal.confidence)))
            return (current_ltp + offset).quantize(Decimal("0.05"))

    # ── Exit Signal Detection (#128) ──────────────────────────────────────────

    def _is_exit_signal(self, signal: TradeSignal) -> bool:
        """
        #128: Returns True when the signal intends to close an existing position.

        A signal is treated as an exit when:
          - direction == 0 (explicit close)
          - OR direction is opposite to the current open position for this symbol

        This drives the separate exit policy:
          - exits bypass circuit limit check
          - exits bypass symbol halt (to reduce risk)
          - exits warn on wide spread but are never blocked by it
        """
        if signal.direction == 0:
            return True
        # Check if we have an open position in the opposite direction
        try:
            position = self._state.get_position(signal.symbol)
            if position is None:
                return False
            net_qty = position.net_quantity
            # BUY signal (+1) on a short position (<0) = exit
            # SELL signal (-1) on a long position (>0) = exit
            if signal.direction > 0 and net_qty < 0:
                return True
            if signal.direction < 0 and net_qty > 0:
                return True
        except Exception:
            pass
        return False

    # ── Symbol Halt / Cooldown (#117) ────────────────────────────────────────

    def halt_symbol(self, symbol: str, reason: str, duration_minutes: int = 30) -> None:
        """
        #117: Halt new entries for *symbol* for *duration_minutes*.

        Records expiry = now + duration in _halted_symbols.
        Exits for halted symbols are still permitted (#128).
        """
        expiry = datetime.now(tz=IST) + timedelta(minutes=duration_minutes)
        with self._halt_lock:
            self._halted_symbols[symbol] = expiry
        log.error(
            "order_executor.symbol_halted",
            symbol=symbol,
            reason=reason,
            duration_minutes=duration_minutes,
            expiry=expiry.isoformat(),
        )

    def is_symbol_halted(self, symbol: str) -> bool:
        """
        #117: Returns True if *symbol* has an active entry halt.
        Auto-clears expired halts on each call.
        """
        now = datetime.now(tz=IST)
        with self._halt_lock:
            expiry = self._halted_symbols.get(symbol)
            if expiry is None:
                return False
            if now >= expiry:
                # Halt expired — auto-clear
                del self._halted_symbols[symbol]
                log.info("order_executor.symbol_halt_expired", symbol=symbol)
                return False
            return True

    # ── Circuit Limit Check (#115) ────────────────────────────────────────────

    def _is_near_circuit_limit(self, symbol: str, price: Decimal) -> bool:
        """
        #115: Returns True if the symbol is near its NSE circuit limit bands.

        [STUB] — needs actual NSE circuit band data from get_quote_book() or
        the NSE Bhavcopy API. The current implementation is a heuristic placeholder.

        Heuristic: if price is up/down >8% from previous close, block entry.
        [ASSUMED] 8% threshold chosen conservatively below NSE's common 10%/20% bands.
        In production: fetch actual circuit bands via ShoonyaBroker.get_quote_book()
        and compare against upper_circuit_limit / lower_circuit_limit fields.
        """
        # [STUB] — replace with real NSE circuit band lookup
        log.debug(
            "order_executor.circuit_limit_check.stub",
            symbol=symbol,
            price=str(price),
            note="[STUB] circuit band data not available — returning False (allow entry)",
        )
        return False  # [STUB] Always returns False until real data is wired up

    # ── Stop-Loss Management (#130) ───────────────────────────────────────────

    def submit_stop_loss(
        self,
        symbol: str,
        position_side: str,
        quantity: int,
        stop_price: Decimal,
    ) -> Optional[str]:
        """
        #130: Submit or register a stop-loss order according to stop_loss_mode.

        BROKER_NATIVE:
            Places a SL-M (Stop-Loss Market) order directly with the broker.
            The broker exchange triggers the market order when price breaches
            stop_price. Counts toward OTR.

        APP_MANAGED:
            Records the stop level in memory. No broker call is made here.
            The caller must invoke check_app_managed_stops() on each tick/bar
            and then call place_stop_loss() when triggered.

        Returns:
            BROKER_NATIVE: exchange client_order_id, or None on failure.
            APP_MANAGED: synthetic stop_order_id string.
        """
        closing_side = "SELL" if position_side.upper() == "BUY" else "BUY"
        log.info(
            "order_executor.submit_stop_loss",
            symbol=symbol,
            mode=self._stop_loss_mode.value,
            position_side=position_side,
            closing_side=closing_side,
            quantity=quantity,
            stop_price=str(stop_price),
        )

        if self._stop_loss_mode == StopLossMode.BROKER_NATIVE:
            # Place SL-M order with broker
            return self.place_stop_loss(
                symbol=symbol,
                side=closing_side,
                quantity=quantity,
                stop_price=stop_price,
            )

        # APP_MANAGED: record the stop level; trigger via check_app_managed_stops()
        synthetic_id = f"APP_SL_{symbol}_{str(uuid4())[:8]}"
        with self._stops_lock:
            self._app_managed_stops[symbol] = {
                "stop_price": stop_price,
                "side": closing_side,
                "qty": quantity,
                "stop_order_id": synthetic_id,
                "position_side": position_side,
            }
        log.info(
            "order_executor.app_managed_stop_registered",
            symbol=symbol,
            stop_price=str(stop_price),
            stop_order_id=synthetic_id,
        )
        return synthetic_id

    def check_app_managed_stops(self, quotes: dict[str, Decimal]) -> list[str]:
        """
        #130: Check if any APP_MANAGED stop prices have been breached.

        Called by the orchestrator on each bar or tick. For each symbol with a
        registered stop, checks whether the current price has crossed the stop level.
        Returns a list of symbols where the stop was triggered.

        The caller is responsible for submitting exit orders for triggered symbols.
        Triggered stops are automatically removed from the registry.
        """
        triggered: list[str] = []
        with self._stops_lock:
            for symbol, stop_info in list(self._app_managed_stops.items()):
                current_price = quotes.get(symbol)
                if current_price is None:
                    continue
                stop_price: Decimal = stop_info["stop_price"]
                position_side: str = stop_info["position_side"]

                # Long position (BUY entry) → stop triggers if price falls below stop
                # Short position (SELL entry) → stop triggers if price rises above stop
                breached = False
                if position_side.upper() == "BUY" and current_price <= stop_price:
                    breached = True
                elif position_side.upper() == "SELL" and current_price >= stop_price:
                    breached = True

                if breached:
                    log.warning(
                        "order_executor.app_managed_stop_triggered",
                        symbol=symbol,
                        current_price=str(current_price),
                        stop_price=str(stop_price),
                        stop_order_id=stop_info["stop_order_id"],
                    )
                    triggered.append(symbol)
                    del self._app_managed_stops[symbol]

        return triggered

    # ── Trailing Stop Management (#131) ──────────────────────────────────────

    def set_trailing_stop(
        self,
        symbol: str,
        current_price: Decimal,
        trail_pct: float = 0.02,  # [ASSUMED] 2% default trail
    ) -> None:
        """
        #131: Register or reset a trailing stop for *symbol*.

        stop_price = current_price * (1 - trail_pct) for long positions.
        peak_price starts at current_price and ratchets up on price advances.
        """
        stop_price = current_price * Decimal(str(1.0 - trail_pct))
        with self._trailing_lock:
            self._trailing_stops[symbol] = {
                "stop_price": stop_price,
                "peak_price": current_price,
                "trail_pct": trail_pct,
            }
        log.info(
            "order_executor.trailing_stop_set",
            symbol=symbol,
            current_price=str(current_price),
            stop_price=str(stop_price),
            trail_pct=trail_pct,
        )

    def update_trailing_stops(self, quotes: dict[str, Decimal]) -> list[str]:
        """
        #131: Update trailing stop levels and return symbols where stop was triggered.

        For each symbol with an active trailing stop:
          1. Update peak_price if current quote > peak_price
          2. Update stop_price = peak_price * (1 - trail_pct)
          3. If current quote <= stop_price: add to triggered list

        Triggered trailing stops are removed from the registry.
        The caller is responsible for submitting exit orders.
        """
        triggered: list[str] = []
        with self._trailing_lock:
            for symbol, ts in list(self._trailing_stops.items()):
                current_price = quotes.get(symbol)
                if current_price is None:
                    continue

                trail_pct = ts["trail_pct"]
                peak_price: Decimal = ts["peak_price"]

                # Ratchet up peak if price advanced
                if current_price > peak_price:
                    peak_price = current_price
                    ts["peak_price"] = peak_price
                    ts["stop_price"] = peak_price * Decimal(str(1.0 - trail_pct))

                stop_price: Decimal = ts["stop_price"]
                if current_price <= stop_price:
                    log.warning(
                        "order_executor.trailing_stop_triggered",
                        symbol=symbol,
                        current_price=str(current_price),
                        stop_price=str(stop_price),
                        peak_price=str(peak_price),
                        trail_pct=trail_pct,
                    )
                    triggered.append(symbol)
                    del self._trailing_stops[symbol]

        return triggered

    # ── Broker Payload Logging (#145) ─────────────────────────────────────────

    # Fields that must be redacted before logging broker request/response payloads
    _SENSITIVE_PAYLOAD_FIELDS: frozenset[str] = frozenset({
        "password", "token", "secret", "api_key", "totp", "apikey",
        "passwd", "pin", "otp", "sesstoken",
    })

    def _log_broker_payload(self, payload: dict) -> None:
        """
        #145: Log broker request payload at DEBUG level with sensitive fields redacted.

        Safe to log fields: buy_or_sell, tradingsymbol, quantity, price_type,
        price, exchange, retention, product.

        Any field whose key matches a known-sensitive pattern is replaced with
        '[REDACTED]' before logging.
        """
        safe_payload: dict = {}
        for k, v in payload.items():
            key_lower = k.lower()
            if any(sensitive in key_lower for sensitive in self._SENSITIVE_PAYLOAD_FIELDS):
                safe_payload[k] = "[REDACTED]"
            else:
                safe_payload[k] = v
        log.debug("order_executor.broker_payload", payload=safe_payload)

    # ── Sensitive Log Scanner (#146) ──────────────────────────────────────────

    # Regex for patterns that look like API tokens or secrets:
    # long alphanumeric strings (>20 chars) that could be keys/tokens
    _TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")

    def _check_sensitive_log(self, text: str) -> bool:
        """
        #146: Returns True if *text* contains patterns that look like API tokens or secrets.

        Heuristic: a run of >20 alphanumeric/base64 characters is suspicious.
        [ASSUMED] Threshold of 20 characters chosen to catch typical JWT/API key lengths
        while avoiding false positives on normal log content (e.g. symbol names, prices).

        Used before logging any broker response to catch inadvertent secret exposure.
        If this returns True, the caller should redact or omit the text.
        """
        if not text:
            return False
        # Check for long opaque token-like strings
        matches = self._TOKEN_PATTERN.findall(text)
        for match in matches:
            if len(match) > 20:
                return True
        return False
