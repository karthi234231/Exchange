"""
orchestrator/fo_orchestrator.py — F&O session lifecycle manager.

Coordinates multi-leg NSE F&O trading for a full trading day:
  09:00 IST — Pre-market: login, warm feature engines
  09:15 IST — Market open: subscribe WebSocket, start collecting bars
  09:30 IST — Enable trading: signals active
  10:00 IST — First F&O chain snapshot: signal generation eligible
  11:00,12:00,13:00,14:00 IST — Delta rebalance heartbeat (every hour)
  15:25 IST — F&O force close: expiry-aware EOD close before NSE cutoff
  15:30 IST — Market close: flush bars, stop WebSocket
  15:45 IST — End of day: write session summary, unwind hedge book, logout

Data flow:
  ShoonyaBroker WebSocket → TimeBarBuilder
    → on_bar():
        FOFeatureEngine.compute(chain) → iv_rank, pcr, vrp_estimate
        FOSignalGenerator.generate(...)  → FOSignal
        StrategyBuilder.<strategy>()     → StrategyResult (multi-leg template)
        OrderExecutor.submit_signal()    × N legs (each leg as a TradeSignal)
        DeltaHedger.needs_hedge(greeks) → futures hedge leg via OrderExecutor

Design constraints (from DECISIONS.md):
  - No live order placement until 13 go-live gates pass (currently paper mode)
  - Hard F&O cutoff: 15:25 IST (NSE stops accepting new F&O at 15:30)
  - Delta rebalance: every completed 15-min bar (not just hourly heartbeat)
  - One symbol per FOOrchestrator instance: NIFTY or BANKNIFTY
  - StrategyBuilder is re-instantiated per signal (stateless per-call)
  - DeltaHedger is stateful and shared across bar callbacks

Thread safety:
  - APScheduler callbacks and on_bar() run on different threads.
  - _trading_enabled and _running are threading.Event (safe).
  - All mutation of hedge book and order state is serialised through the
    single-threaded bar consumer (on_bar is called from bar builder thread).
"""
from __future__ import annotations

import math
import signal as _signal
import statistics
import threading
import time as _time
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover
    BackgroundScheduler = None  # type: ignore[misc,assignment]
    CronTrigger = None  # type: ignore[misc,assignment]

from algo_trading.alerts.telegram_bot import AlertSender
from algo_trading.config import settings
from algo_trading.data.broker_base import Bar
from algo_trading.data.broker_shoonya import ShoonyaBroker
from algo_trading.data.candle_store import QuestDBStore, TimeBarBuilder
from algo_trading.data.fo_chain import OptionChain
from algo_trading.data.system_state import LiveSystemState
from algo_trading.execution.order_executor import OrderExecutor, TradeSignal
from algo_trading.execution.order_fsm import OrderFSM
from algo_trading.execution.rate_limiter import DualLockRateLimiter
from algo_trading.features.fo_features import FOFeatureEngine, FOFeaturesResult
from algo_trading.fo.cost_model import FOCostModel
from algo_trading.fo.greeks_hedger import DeltaHedger
from algo_trading.fo.portfolio_risk import PortfolioRiskEngine
from algo_trading.fo.signal_generator import (
    FOSignal,
    FOSignalGenerator,
    STRATEGY_SHORT_STRANGLE,
    STRATEGY_IRON_CONDOR,
    STRATEGY_BULL_PUT_SPREAD,
    STRATEGY_LONG_STRADDLE,
)
from algo_trading.features.statistical import GARCHVolForecaster
from algo_trading.fo.strategies import StrategyBuilder, StrategyResult
from algo_trading.fo.vix_estimator import VIXEstimate, estimate_india_vix
from algo_trading.core.calendar import (
    IST,
    is_trading_day,
    now_ist,
    today_ist,
)
from algo_trading.risk.manager import RiskManager

log = structlog.get_logger(__name__)

# F&O specific cutoffs (IST)
_FO_ENTRY_CUTOFF_HOUR   = 14   # ASSUMED: no new F&O entries after 14:00
_FO_ENTRY_CUTOFF_MINUTE = 0
_FO_FORCE_CLOSE_HOUR    = 15
_FO_FORCE_CLOSE_MINUTE  = 25   # NSE F&O cutoff is 15:30; close 5 min early

# Rebalance heartbeat (also fires on every bar, so this is a belt-and-suspenders)
_DELTA_REBALANCE_HOURS = [10, 11, 12, 13, 14]

# iv_rank from FOFeatureEngine is [0, 1]; FOSignalGenerator expects [0, 100]
_IV_RANK_SCALE = 100.0

_FLUSH_INTERVAL_SEC = 0.1

# ── GARCH lot-sizing thresholds (ASSUMED — calibrate on 12 months live data) ─
# Annualised vol from GARCH: low vol → sell more premium, high vol → reduce/skip
_GARCH_LOW_VOL:   float = 0.12   # below 12% ann. vol → lots = 2 (wider strangles OK)
_GARCH_MID_VOL:   float = 0.20   # 12–20%            → lots = 1 (standard)
_GARCH_HIGH_VOL:  float = 0.28   # 20–28%            → lots = 1 (cautious)
_GARCH_SKIP_VOL:  float = 0.35   # above 35%         → skip entry (vol regime too uncertain)

# ── VIX z-score thresholds for proactive delta hedge ─────────────────────────
_VIX_HISTORY_BARS: int  = 20     # rolling window for VIX z-score (ASSUMED: ~5 trading days)
_VIX_ZSCORE_TRIGGER: float = 2.5 # VIX spike > 2.5σ → force early hedge check


class FOOrchestrator:
    """
    F&O session lifecycle manager for a single NSE underlying.

    Initialise once per process. Call run() to block until session end.
    Handles graceful shutdown on SIGTERM/SIGINT.

    Parameters
    ----------
    symbol : str
        NSE underlying: "NIFTY" or "BANKNIFTY".
    days_to_expiry : int
        Calendar days to the target expiry being traded.
        Caller is responsible for rolling this to the next expiry.
    """

    def __init__(self, symbol: str = "NIFTY", days_to_expiry: int = 21) -> None:
        self._symbol = symbol.upper()
        self._days_to_expiry = days_to_expiry

        # ── Core infrastructure ────────────────────────────────────────────
        self._store = QuestDBStore()
        self._state = LiveSystemState()
        self._broker = ShoonyaBroker()
        self._alert = AlertSender()
        self._fsm = OrderFSM(store=self._store)
        self._rate_limiter = DualLockRateLimiter()
        self._risk_manager = RiskManager(
            state=self._state,
            store=self._store,
            alert_sender=self._alert,
        )
        self._executor = OrderExecutor(
            broker=self._broker,
            fsm=self._fsm,
            rate_limiter=self._rate_limiter,
            risk_manager=self._risk_manager,
            state=self._state,
            store=self._store,
            alert_sender=self._alert,
        )
        self._bar_builder = TimeBarBuilder(
            interval_minutes=settings.bar_interval_minutes,
        )

        # ── F&O components ─────────────────────────────────────────────────
        self._cost_model = FOCostModel()
        self._risk_engine = PortfolioRiskEngine()
        self._fo_features = FOFeatureEngine()
        self._signal_gen = FOSignalGenerator()
        self._hedger = DeltaHedger(
            symbol=self._symbol,
            risk_engine=self._risk_engine,
        )

        # Latest option chain snapshot — updated by broker subscription callback
        self._latest_chain: Optional[OptionChain] = None
        self._latest_fo_features: Optional[FOFeaturesResult] = None

        # Active strategy result (None if flat)
        self._active_strategy: Optional[StrategyResult] = None

        # ── GARCH vol forecaster (lot sizing) ──────────────────────────────
        self._garch = GARCHVolForecaster()

        # ── VIX history for z-score proactive hedge trigger ────────────────
        # Ring of recent VIX values (capped at _VIX_HISTORY_BARS)
        self._vix_history: list[float] = []
        self._latest_vix: Optional[VIXEstimate] = None

        # ── APScheduler ────────────────────────────────────────────────────
        self._scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

        # ── Threading events ───────────────────────────────────────────────
        self._running = threading.Event()
        self._trading_enabled = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the F&O orchestrator for the current trading day.
        Blocks until session end or SIGTERM/SIGINT.
        """
        log.info("fo_orchestrator.starting", symbol=self._symbol, dte=self._days_to_expiry)

        if not is_trading_day():
            log.warning("fo_orchestrator.not_a_trading_day", date=str(today_ist()))
            return

        _signal.signal(_signal.SIGTERM, self._handle_shutdown_signal)
        _signal.signal(_signal.SIGINT, self._handle_shutdown_signal)

        try:
            self._startup()
            self._schedule_daily_events()
            self._scheduler.start()
            self._running.set()
            log.info("fo_orchestrator.running — waiting for market events")

            while self._running.is_set():
                _time.sleep(1.0)

        except Exception as exc:
            log.error("fo_orchestrator.fatal_error", error=str(exc))
            self._alert.send_exception_alert(exc, context="fo_orchestrator.run")
            raise
        finally:
            self._shutdown()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _startup(self) -> None:
        log.info("fo_orchestrator.startup", symbol=self._symbol)

        self._alert.start()
        self._store.connect()
        self._store.create_tables()

        log.info("fo_orchestrator.broker.login")
        self._broker.login()
        self._broker.resolve_all_symbols()

        self._state.reset_for_new_day(today_ist())
        self._risk_manager.reset_for_new_day()
        self._bar_builder.reset()
        self._hedger.reset()
        self._active_strategy = None
        self._vix_history.clear()
        self._latest_vix = None

        self._executor.start()

        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="fo-questdb-flush",
            daemon=True,
        )
        self._flush_thread.start()

        log.info("fo_orchestrator.startup.complete")
        self._alert.send_session_alert(
            "F&O STARTUP COMPLETE",
            details=f"symbol={self._symbol} dte={self._days_to_expiry}",
        )

    # ── Scheduled Session Events ──────────────────────────────────────────────

    def _schedule_daily_events(self) -> None:
        tz = "Asia/Kolkata"

        self._scheduler.add_job(
            self._on_market_open,
            CronTrigger(hour=9, minute=15, timezone=tz),
            id="fo_market_open",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_trading_enabled,
            CronTrigger(hour=9, minute=30, timezone=tz),
            id="fo_trading_enabled",
            replace_existing=True,
        )

        # Hourly delta rebalance heartbeats (belt-and-suspenders — also fires on every bar)
        for hour in _DELTA_REBALANCE_HOURS:
            self._scheduler.add_job(
                self._on_delta_rebalance_heartbeat,
                CronTrigger(hour=hour, minute=0, timezone=tz),
                id=f"fo_rebalance_{hour:02d}",
                replace_existing=True,
            )

        # F&O force close: 15:25 IST (5 min before NSE cutoff)
        self._scheduler.add_job(
            self._on_fo_force_close,
            CronTrigger(
                hour=_FO_FORCE_CLOSE_HOUR,
                minute=_FO_FORCE_CLOSE_MINUTE,
                timezone=tz,
            ),
            id="fo_force_close",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_market_close,
            CronTrigger(hour=15, minute=30, timezone=tz),
            id="fo_market_close",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._on_end_of_day,
            CronTrigger(hour=15, minute=45, timezone=tz),
            id="fo_end_of_day",
            replace_existing=True,
        )

        log.info("fo_orchestrator.events_scheduled", symbol=self._symbol)

    # ── Session Lifecycle Callbacks ───────────────────────────────────────────

    def _on_market_open(self) -> None:
        log.info("fo_orchestrator.MARKET_OPEN", symbol=self._symbol)
        self._trading_enabled.clear()
        self._alert.send_session_alert("F&O MARKET OPEN", details=str(today_ist()))

        try:
            self._broker.subscribe_ws(
                symbols=[self._symbol],
                on_tick=self._on_tick,
                on_order_update=self._on_order_update,
            )
            # Subscribe to option chain feed if broker supports it
            if hasattr(self._broker, "start_option_chain"):
                self._broker.start_option_chain(
                    symbol=self._symbol,
                    expiry=None,  # broker selects front-month expiry
                    greek=True,
                )
        except Exception as exc:
            log.error("fo_orchestrator.ws_subscribe_failed", error=str(exc))
            self._alert.send_exception_alert(exc, "fo_market_open.subscribe_ws")

    def _on_trading_enabled(self) -> None:
        log.info("fo_orchestrator.TRADING_ENABLED", symbol=self._symbol)
        self._trading_enabled.set()
        self._alert.send_session_alert("F&O TRADING ENABLED — signals active")

    def _on_delta_rebalance_heartbeat(self) -> None:
        """Hourly scheduled delta rebalance check (independent of bar events)."""
        if not self._trading_enabled.is_set():
            return
        if self._risk_manager.is_kill_switch_active:
            return
        log.info("fo_orchestrator.delta_rebalance_heartbeat", symbol=self._symbol)
        self._attempt_delta_hedge()

    def _on_fo_force_close(self) -> None:
        """
        15:25 IST: F&O-specific EOD force close.

        Unwind the delta hedge book first (to avoid naked futures exposure),
        then close all option legs at market.
        """
        log.warning("fo_orchestrator.FO_FORCE_CLOSE", symbol=self._symbol)
        self._trading_enabled.clear()  # stop new entries

        # Step 1: unwind delta hedge futures
        unwind_lots = self._hedger.unwind_hedge()
        if unwind_lots != 0:
            log.info(
                "fo_orchestrator.unwind_hedge",
                symbol=self._symbol,
                lots=unwind_lots,
            )
            try:
                direction = 1 if unwind_lots > 0 else -1
                abs_lots = abs(unwind_lots)
                unwind_signal = TradeSignal(
                    symbol=f"{self._symbol}FUT",
                    direction=direction,
                    entry_price=Decimal("0"),      # market order — price filled by broker
                    stop_loss_price=Decimal("0"),
                    atr=Decimal("0"),
                    confidence=1.0,
                )
                self._executor.submit_signal(unwind_signal)
                self._hedger.reset()
            except Exception as exc:
                log.error("fo_orchestrator.unwind_hedge_failed", error=str(exc))
                self._alert.send_exception_alert(exc, "fo_force_close.unwind_hedge")

        # Step 2: close all option positions (market)
        positions = self._state.get_all_positions()
        closed = 0
        for pos in positions:
            if pos.is_flat:
                continue
            if pos.symbol == f"{self._symbol}FUT":
                continue  # already handled above
            try:
                self._executor.close_position(pos.symbol, pos.net_quantity)
                closed += 1
            except Exception as exc:
                log.error(
                    "fo_orchestrator.force_close_leg_failed",
                    symbol=pos.symbol,
                    error=str(exc),
                )
                self._alert.send_exception_alert(exc, f"fo_force_close.{pos.symbol}")

        self._active_strategy = None
        self._alert.send_session_alert(
            "F&O FORCE CLOSE COMPLETE",
            details=f"Closed {closed} option leg(s), unwind_lots={unwind_lots}",
        )

    def _on_market_close(self) -> None:
        log.info("fo_orchestrator.MARKET_CLOSE", symbol=self._symbol)
        try:
            self._broker.unsubscribe_ws([self._symbol])
            if hasattr(self._broker, "stop_option_chain"):
                self._broker.stop_option_chain(self._symbol)
        except Exception as exc:
            log.error("fo_orchestrator.unsubscribe_failed", error=str(exc))

        partial_bars = self._bar_builder.flush_all()
        for bar in partial_bars:
            self._store.write_bar(bar)
        self._store.flush()
        self._alert.send_session_alert(
            "F&O MARKET CLOSE",
            details=f"Flushed {len(partial_bars)} partial bars",
        )

    def _on_end_of_day(self) -> None:
        snapshot = self._state.snapshot()
        daily_pnl = self._state.daily_pnl
        log.info("fo_orchestrator.END_OF_DAY", daily_pnl=str(daily_pnl))
        self._alert.send_session_alert(
            "F&O SESSION COMPLETE",
            details=f"PnL=₹{daily_pnl:.2f}",
        )
        try:
            self._broker.logout()
        except Exception as exc:
            log.error("fo_orchestrator.logout.failed", error=str(exc))

        self._running.clear()

    # ── Bar Processing ────────────────────────────────────────────────────────

    def _on_tick(self, tick: object) -> None:
        """Broker tick callback → feed bar builder."""
        completed_bar = self._bar_builder.feed_tick(tick)  # type: ignore[arg-type]
        if completed_bar is not None:
            self._on_bar(completed_bar)

    def _on_bar(self, bar: Bar) -> None:
        """
        Called when a 15-min bar closes.
        Persist → compute F&O features → generate signal → submit legs → delta hedge.
        """
        log.info(
            "fo_orchestrator.bar",
            symbol=bar.symbol,
            ts=bar.timestamp.isoformat(),
            close=str(bar.close),
        )
        self._store.write_bar(bar)

        if not self._trading_enabled.is_set():
            return
        if self._risk_manager.is_kill_switch_active:
            return
        if not self._is_within_entry_window():
            # After entry cutoff: only allow delta rebalance, no new strategies
            self._attempt_delta_hedge()
            return

        # If already in a strategy, only rebalance delta — no double entries
        if self._active_strategy is not None:
            self._attempt_delta_hedge()
            return

        # Need a fresh chain snapshot to generate a signal
        if self._latest_chain is None:
            log.debug("fo_orchestrator.bar.no_chain_snapshot", symbol=bar.symbol)
            return

        try:
            self._generate_and_submit_signal()
        except Exception as exc:
            log.error("fo_orchestrator.bar.signal_error", error=str(exc))
            self._alert.send_exception_alert(exc, "fo_orchestrator.on_bar.signal")

        # Always attempt delta hedge after bar, even if no new signal
        self._attempt_delta_hedge()

    def _on_order_update(self, update: object) -> None:
        self._executor.on_broker_update(update)  # type: ignore[arg-type]

    # ── Signal Generation ─────────────────────────────────────────────────────

    def _generate_and_submit_signal(self) -> None:
        """
        Compute F&O features → generate FOSignal → build strategy legs → submit.
        Called from on_bar() when flat and within entry window.
        """
        chain = self._latest_chain
        assert chain is not None  # guarded by caller

        # Compute feature snapshot from chain
        fo_features = self._fo_features.compute(chain)
        self._latest_fo_features = fo_features

        # FOFeatureEngine returns iv_rank in [0, 1]; FOSignalGenerator expects [0, 100]
        iv_rank_pct = fo_features.iv_rank * _IV_RANK_SCALE

        # Compute VRP estimate from portfolio risk engine (uses realized vol vs ATM IV)
        vrp_estimate = self._risk_engine.compute_vrp(
            realized_vol_series=[],        # empty → engine uses cached realized vol
            implied_vol_atm=fo_features.atm_iv,
        )

        # Generate signal
        fo_signal = self._signal_gen.generate(
            iv_rank=iv_rank_pct,
            vrp_estimate=vrp_estimate,
            pcr=fo_features.pcr_oi,
            days_to_expiry=self._days_to_expiry,
        )

        log.info(
            "fo_orchestrator.signal",
            strategy=fo_signal.strategy_name,
            confidence=fo_signal.confidence,
            iv_rank=round(iv_rank_pct, 1),
            pcr=round(fo_features.pcr_oi, 3),
            dte=self._days_to_expiry,
            is_actionable=fo_signal.is_actionable,
        )

        if not fo_signal.is_actionable:
            return

        # GARCH vol check: skip entry if vol regime is too uncertain
        garch_vol = self._garch.forecast_one_step()
        if not math.isnan(garch_vol) and garch_vol >= _GARCH_SKIP_VOL:
            log.warning(
                "fo_orchestrator.garch_skip",
                garch_vol=round(garch_vol, 4),
                threshold=_GARCH_SKIP_VOL,
            )
            return

        # Build multi-leg strategy template
        builder = StrategyBuilder(cost_model=self._cost_model, symbol=self._symbol)
        lots = self._compute_lots_from_garch(garch_vol)
        strategy_result = self._build_strategy(fo_signal, builder, chain, lots=lots)

        if strategy_result is None:
            log.info("fo_orchestrator.strategy_build_failed", signal=fo_signal.strategy_name)
            return

        # Submit each leg as a TradeSignal to OrderExecutor
        self._submit_strategy_legs(strategy_result)
        self._active_strategy = strategy_result

        self._alert.send_session_alert(
            "F&O ENTRY",
            details=f"{fo_signal.summary()} | lots={lots} | garch_vol={garch_vol:.3f}",
        )

    def _build_strategy(
        self,
        fo_signal: object,
        builder: StrategyBuilder,
        chain: OptionChain,
        lots: int = 1,
    ) -> Optional[StrategyResult]:
        """
        Delegate to the correct StrategyBuilder method based on signal strategy name.
        Returns None if the builder raises (credit/margin/DTE validation failed).

        Parameters
        ----------
        lots : int
            GARCH-scaled lot count from _compute_lots_from_garch().
        """
        assert isinstance(fo_signal, FOSignal)

        # Derive ATM strike from chain (closest strike to current spot)
        atm_strike = chain.atm_strike if hasattr(chain, "atm_strike") else self._estimate_atm(chain)

        strategy_name = fo_signal.strategy_name
        expiry = chain.expiry

        try:
            if strategy_name == STRATEGY_SHORT_STRANGLE:
                return builder.short_strangle(
                    atm_strike=atm_strike,
                    expiry=expiry,
                    lots=lots,
                )
            elif strategy_name == STRATEGY_IRON_CONDOR:
                return builder.iron_condor(
                    atm_strike=atm_strike,
                    expiry=expiry,
                    lots=lots,
                )
            elif strategy_name == STRATEGY_BULL_PUT_SPREAD:
                return builder.bull_put_spread(
                    atm_strike=atm_strike,
                    expiry=expiry,
                    lots=lots,
                )
            elif strategy_name == STRATEGY_LONG_STRADDLE:
                return builder.long_straddle(
                    atm_strike=atm_strike,
                    expiry=expiry,
                    lots=lots,
                )
            else:
                log.warning("fo_orchestrator.unknown_strategy", name=strategy_name)
                return None
        except Exception as exc:
            log.warning(
                "fo_orchestrator.strategy_build_rejected",
                strategy=strategy_name,
                reason=str(exc),
            )
            return None

    def _submit_strategy_legs(self, strategy: StrategyResult) -> None:
        """
        Submit each leg of a StrategyResult to OrderExecutor as a TradeSignal.

        Each leg becomes an independent order — the executor handles each
        separately.  Multi-leg atomicity is an ASSUMED simplification;
        a real spread-order engine would submit all legs atomically.
        """
        for leg in strategy.legs:
            direction = 1 if leg.direction > 0 else -1
            signal = TradeSignal(
                symbol=leg.symbol,
                direction=direction,
                entry_price=Decimal(str(round(leg.entry_premium, 2))),
                stop_loss_price=Decimal("0"),   # F&O legs managed as a portfolio, not per-leg SL
                atr=Decimal("0"),
                confidence=1.0,
            )
            try:
                order_id = self._executor.submit_signal(signal)
                log.info(
                    "fo_orchestrator.leg_submitted",
                    symbol=leg.symbol,
                    direction=direction,
                    premium=leg.entry_premium,
                    order_id=order_id,
                )
            except Exception as exc:
                log.error(
                    "fo_orchestrator.leg_submit_failed",
                    symbol=leg.symbol,
                    error=str(exc),
                )
                self._alert.send_exception_alert(exc, f"submit_leg.{leg.symbol}")

    # ── Delta Hedging ─────────────────────────────────────────────────────────

    def _attempt_delta_hedge(self, force: bool = False) -> None:
        """
        Compute current portfolio Greeks and place a futures hedge if needed.
        Called on every bar close, hourly heartbeat, and VIX spike events.

        Parameters
        ----------
        force : bool
            When True (VIX z-score triggered), bypass the needs_hedge() threshold
            check and rebalance even if delta is within normal bounds.
        """
        if self._active_strategy is None:
            return  # flat book — nothing to hedge

        try:
            greeks = self._risk_engine.last_greeks
            if greeks is None:
                return

            if not force and not self._hedger.needs_hedge(greeks):
                return

            lots = self._hedger.compute_hedge_lots(greeks)
            if lots == 0:
                return

            direction = 1 if lots > 0 else -1
            hedge_signal = TradeSignal(
                symbol=f"{self._symbol}FUT",
                direction=direction,
                entry_price=Decimal("0"),   # market order
                stop_loss_price=Decimal("0"),
                atr=Decimal("0"),
                confidence=1.0,
            )
            order_id = self._executor.submit_signal(hedge_signal)

            # Record assuming market fill at latest known spot price (ASSUMED).
            # LiveSystemState does not track LTP; fall back to 0.0 — the real fill
            # price is reconciled later when the broker WebSocket confirms the fill.
            _get_ltp = getattr(self._state, "get_ltp", None)
            fill_price = float(_get_ltp(f"{self._symbol}FUT") if _get_ltp is not None else 0.0)
            record = self._hedger.record_hedge(
                greeks_before=greeks,
                lots_traded=lots,
                futures_price=fill_price,
            )
            log.info(
                "fo_orchestrator.hedge_placed",
                symbol=self._symbol,
                lots=lots,
                fill_price=fill_price,
                delta_before=record.portfolio_delta_before,
                delta_after=record.portfolio_delta_after,
            )
            self._alert.send_session_alert(
                "DELTA HEDGE",
                details=(
                    f"{'BUY' if lots > 0 else 'SELL'} {abs(lots)} lots "
                    f"Δ_before={record.portfolio_delta_before:.2f}"
                ),
            )
        except Exception as exc:
            log.error("fo_orchestrator.delta_hedge_failed", error=str(exc))
            self._alert.send_exception_alert(exc, "fo_orchestrator.attempt_delta_hedge")

    # ── Entry Window Guard ────────────────────────────────────────────────────

    def _is_within_entry_window(self) -> bool:
        """Return True if current IST time is before the F&O entry cutoff."""
        now = now_ist()
        cutoff_hour   = _FO_ENTRY_CUTOFF_HOUR
        cutoff_minute = _FO_ENTRY_CUTOFF_MINUTE
        return (now.hour, now.minute) < (cutoff_hour, cutoff_minute)

    # ── Option Chain Callback ─────────────────────────────────────────────────

    def on_chain_snapshot(self, chain: OptionChain) -> None:
        """
        Called by broker's option chain callback when a new snapshot arrives.
        Stores the latest chain for use in the next bar's signal generation.
        Also computes VIX from the chain and checks for spike-triggered hedges.

        This method is designed to be passed as a callback to TrueDataFeed or
        ShoonyaBroker.start_option_chain().
        """
        self._latest_chain = chain
        log.debug(
            "fo_orchestrator.chain_snapshot",
            symbol=self._symbol,
            expiry=str(chain.expiry),
            n_strikes=len(chain.calls) + len(chain.puts) if hasattr(chain, "calls") else 0,
        )

        # Compute VIX from chain and check for spike-triggered proactive hedge
        try:
            vix_est = estimate_india_vix(chain)
            self._latest_vix = vix_est
            if vix_est.vix > 0:
                self._update_vix_history(vix_est.vix)
        except Exception as exc:
            log.debug("fo_orchestrator.vix_compute_failed", error=str(exc))

    # ── GARCH lot sizing ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_lots_from_garch(garch_vol: float) -> int:
        """
        Map annualised GARCH vol forecast to an F&O lot count.

        Thresholds are ASSUMED starting points — calibrate on 12+ months live data.

        Returns
        -------
        int
            Lots to trade: 2 (low vol), 1 (mid/high vol).
            Callers skip entry entirely when garch_vol >= _GARCH_SKIP_VOL.
        """
        if math.isnan(garch_vol) or garch_vol < 0:
            return 1  # GARCH not yet fitted → conservative default
        if garch_vol < _GARCH_LOW_VOL:
            return 2  # Low vol: sell wider strangles, more premium capture
        return 1       # Mid or high vol: single lot (cautious)

    # ── VIX z-score hedge trigger ─────────────────────────────────────────────

    def _update_vix_history(self, vix_value: float) -> None:
        """
        Append latest VIX to rolling history and fire a proactive delta hedge
        if a z-score spike is detected (VIX > _VIX_ZSCORE_TRIGGER σ above mean).
        """
        self._vix_history.append(vix_value)
        # Keep only the last N bars
        if len(self._vix_history) > _VIX_HISTORY_BARS:
            self._vix_history.pop(0)

        if len(self._vix_history) < 5:
            return  # not enough history yet

        mean = statistics.mean(self._vix_history[:-1])  # exclude latest point
        stdev = statistics.stdev(self._vix_history[:-1])
        # When history is nearly flat (stdev ≈ 0), a large absolute deviation
        # is still a spike. Use a 1% floor so we can still detect extreme moves.
        effective_stdev = max(stdev, mean * 0.01) if mean > 0 else max(stdev, 0.1)
        if effective_stdev < 1e-9:
            return  # truly degenerate — cannot compute z-score

        z = (vix_value - mean) / effective_stdev
        if z >= _VIX_ZSCORE_TRIGGER:
            log.warning(
                "fo_orchestrator.vix_spike_hedge",
                vix=round(vix_value, 2),
                z_score=round(z, 2),
                mean=round(mean, 2),
            )
            self._alert.send_session_alert(
                "VIX SPIKE — proactive delta hedge",
                details=f"VIX={vix_value:.1f} z={z:.2f}σ",
            )
            # Force a hedge rebalance ahead of schedule only when a position is open
            if (
                self._active_strategy is not None
                and self._trading_enabled.is_set()
                and not self._risk_manager.is_kill_switch_active
            ):
                self._attempt_delta_hedge(force=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_atm(chain: OptionChain) -> float:
        """
        Estimate ATM strike from chain when chain.atm_strike is not available.
        Uses the midpoint of the strike range as a fallback.
        """
        try:
            all_strikes = sorted(
                set(c.strike for c in chain.calls) | set(p.strike for p in chain.puts)
            )
            if not all_strikes:
                return 0.0
            mid = all_strikes[len(all_strikes) // 2]
            return float(mid)
        except Exception:
            return 0.0

    # ── QuestDB Flush Loop ────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while self._running.is_set():
            try:
                self._store.flush()
            except Exception as exc:
                log.error("fo_orchestrator.flush_error", error=str(exc))
            _time.sleep(_FLUSH_INTERVAL_SEC)

    # ── Signal / Shutdown ─────────────────────────────────────────────────────

    def _handle_shutdown_signal(self, signum: int, frame: object) -> None:
        log.warning("fo_orchestrator.shutdown_signal", signum=signum)
        self._running.clear()

    def _shutdown(self) -> None:
        log.info("fo_orchestrator.shutdown.starting")
        self._trading_enabled.clear()
        self._executor.stop()

        # Final hedge unwind before full shutdown
        unwind = self._hedger.unwind_hedge()
        if unwind != 0:
            log.warning(
                "fo_orchestrator.shutdown.hedge_not_flat",
                symbol=self._symbol,
                unwind_lots=unwind,
            )

        # Force-close any remaining option positions
        positions = self._state.get_all_positions()
        for pos in positions:
            if not pos.is_flat:
                try:
                    self._executor.close_position(pos.symbol, pos.net_quantity)
                except Exception:
                    pass

        partial_bars = self._bar_builder.flush_all()
        for bar in partial_bars:
            self._store.write_bar(bar)
        self._store.flush()

        self._scheduler.shutdown(wait=False)
        self._alert.send_session_alert("F&O SHUTDOWN COMPLETE")
        self._alert.stop()
        self._store.close()

        log.info("fo_orchestrator.shutdown.complete")
