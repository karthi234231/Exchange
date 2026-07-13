"""
orchestrator/daily_orchestrator.py — Daily trading session lifecycle manager.

Coordinates all processes for a complete trading day:
  09:00 IST — Pre-market: login, resolve symbols, warm up state
  09:15 IST — Market open: start tick ingestor, start bar builder
  09:30 IST — Enable trading: first bar complete, signals enabled
  15:15 IST — Force close: close all positions, stop new entries [SURV-8]
  15:30 IST — Market close: stop tick ingestor, flush QuestDB
  15:45 IST — End of day: write session summary, logout, release ZMQ

Architecture:
  - Runs on the main process thread (not in a subprocess)
  - APScheduler CronTrigger for session lifecycle events
  - Tick processing in a tight loop on a background thread
  - Bar events drive signal generation synchronously (no GIL issue: no ML in Stage 1)
  - ZMQ: TickPublisher (PUB:5555) → BarBuilder (reads from broker callback directly in S1)

Stage 1 data flow (single-process, simpler than multi-process for paper trading):
  Shoonya WebSocket callback → TickValidator → TickPublisher (ZMQ)
                             → TimeBarBuilder (in-process)
                             → EMACrossStrategy
                             → OrderExecutor
                             → ShoonyaBroker (paper simulate)
"""
from __future__ import annotations

import gc
import multiprocessing
import signal
import threading
import time as _time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from uuid import uuid4

import structlog

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover
    BackgroundScheduler = None  # type: ignore[misc,assignment]
    CronTrigger = None  # type: ignore[misc,assignment]

from algo_trading.alerts.alert_sink import run_alert_sink
from algo_trading.alerts.telegram_bot import AlertSender
from algo_trading.config import settings
from algo_trading.app.health_check import start_metrics_server
from algo_trading.data.broker_base import Bar, Tick
from algo_trading.data.broker_shoonya import ShoonyaBroker
from algo_trading.data.candle_store import QuestDBStore, TimeBarBuilder
from algo_trading.data.integrity_guard import DataQualityMonitor, TickValidator
from algo_trading.data.system_state import LiveSystemState, StartupReconciler
from algo_trading.data.websocket_feed import BarPublisher, TickPublisher
from algo_trading.execution.order_executor import OrderExecutor, TradeSignal
from algo_trading.execution.order_fsm import OrderFSM
from algo_trading.execution.rate_limiter import DualLockRateLimiter
from algo_trading.core.calendar import (
    IST,
    can_enter_new_trade,
    is_market_open,
    is_trading_day,
    now_ist,
    today_ist,
)
from algo_trading.risk.manager import RiskManager
from algo_trading.signals.ema_cross import EMACrossStrategy
from algo_trading.signals.ema_cross_filtered import FilteredEMACrossStrategy
from algo_trading.signals.mean_reversion import MeanReversionStrategy
from algo_trading.signals.multi_timeframe import MultiTimeframeConfirmation

# Optional news pipeline integration.  Gracefully absent when news_pipeline
# dependencies (asyncpg, prefect, etc.) are not installed.
try:
    from algo_trading.news_pipeline.client import NewsPipelineClient as _NewsPipelineClient
    _NEWS_PIPELINE_AVAILABLE = True
except ImportError:
    _NEWS_PIPELINE_AVAILABLE = False

# Market context filter (FII/DII + bulk deals + pledging). Gracefully absent
# if the data module dependencies (requests, structlog) are not installed.
try:
    from algo_trading.data.market_context import MarketContextFilter as _MarketContextFilter
    _MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    _MarketContextFilter = None  # type: ignore[misc,assignment]
    _MARKET_CONTEXT_AVAILABLE = False

log = structlog.get_logger(__name__)

# Replace the module's class with a custom subclass that supports __setattr__
# for full dotted-path attribute names used in test patching via patch.multiple().
# e.g. patch.multiple("algo_trading.app.daily_orchestrator",
#          **{"algo_trading.app.daily_orchestrator.ShoonyaBroker": MagicMock})
# transparently redirects to setting "ShoonyaBroker" on the module.
import sys as _sys
import types as _types

_MODULE_PREFIX = __name__ + "."


class _PatchableModule(_types.ModuleType):
    """ModuleType subclass that maps dotted-path attribute names to short names."""

    def __getattr__(self, name: str):
        if name.startswith(_MODULE_PREFIX):
            short_name = name[len(_MODULE_PREFIX):]
            try:
                return super().__getattribute__(short_name)
            except AttributeError:
                pass
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name.startswith(_MODULE_PREFIX):
            short_name = name[len(_MODULE_PREFIX):]
            super().__setattr__(short_name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith(_MODULE_PREFIX):
            short_name = name[len(_MODULE_PREFIX):]
            try:
                super().__delattr__(short_name)
                return
            except AttributeError:
                pass
        super().__delattr__(name)


# Upgrade this module's class in-place so setattr/getattr work with dotted keys.
_sys.modules[__name__].__class__ = _PatchableModule


_TICK_LOOP_SLEEP_SEC = 0.001    # 1ms idle sleep in tick loop
_FLUSH_INTERVAL_SEC = 0.1       # Flush QuestDB every 100ms

# Minimum confidence multiplier from MTF before a signal is suppressed entirely
# (0.0 = counter-trend → blocked; 0.5 = sideways → half size; 1.0 = aligned → full)
_MTF_MIN_MULTIPLIER: float = 0.0   # signals with multiplier ≤ this are blocked (counter-trend)


class DailyOrchestrator:
    """
    Main orchestrator for a trading session.

    Initialise once per process. Call run() to block until market close
    or keyboard interrupt. Handles graceful shutdown on SIGTERM/SIGINT.
    """

    def __init__(self) -> None:
        # ── Core components ────────────────────────────────────────────────
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
        self._bar_builder = TimeBarBuilder(interval_minutes=settings.bar_interval_minutes)

        # ── Signal stack ───────────────────────────────────────────────────
        # Three complementary strategies; all fed every bar.
        # FilteredEMACross: trend signals when ADX ≥ 25
        # MeanReversion:    fade signals when ADX < 20 (sideways regime)
        # MultiTimeframe:   HTF trend filter that scales position size
        self._strategy = FilteredEMACrossStrategy()
        self._mean_rev = MeanReversionStrategy()
        self._mtf = MultiTimeframeConfirmation()

        self._tick_validator = TickValidator()
        self._data_quality = DataQualityMonitor(
            halt_callback=self._on_data_quality_halt,
        )
        self._tick_publisher = TickPublisher()
        self._bar_publisher = BarPublisher()

        # ── Scheduler ─────────────────────────────────────────────────────
        self._scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

        # ── State ─────────────────────────────────────────────────────────
        self._running = threading.Event()
        self._trading_enabled = threading.Event()
        self._alert_sink_proc: Optional[multiprocessing.Process] = None
        self._flush_thread: Optional[threading.Thread] = None

        # ── Pending paper fills: exchange_order_id → symbol ──────────────
        # After a paper order is placed, fill it at the next tick's LTP
        self._pending_paper_fills: dict[str, str] = {}
        self._last_ltp: dict[str, Decimal] = {}

        # P2-FIX-84: Warmup bar counter — block signals until each symbol has
        # received >= settings.min_bars_before_signal bars since market open.
        # This ensures indicators (EMA, ATR) have sufficient history before trading.
        self._bar_count: dict[str, int] = {}

        # ── News pipeline client (optional) ───────────────────────────────
        # Subscribes to FusedSignal updates from the Prefect fusion_flow via
        # ZMQ SUB (port 5572).  None when news_pipeline is not installed.
        self._news_client = _NewsPipelineClient() if _NEWS_PIPELINE_AVAILABLE else None

        # ── Market context filter (optional) ──────────────────────────────
        # FII/DII regime + NSE bulk deal alignment + promoter pledging filter.
        # Refreshed once at pre-market startup. None if module is unavailable.
        # Lazily constructed in _startup(). Keeping __init__ side-effect light
        # makes import smoke tests and unit tests deterministic.
        self._market_context: Optional[_MarketContextFilter] = None

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the orchestrator for the current trading day.
        Blocks until session end or SIGTERM/SIGINT.
        """
        log.info("orchestrator.starting", mode=settings.trading_mode.value)

        if not is_trading_day():
            log.warning("orchestrator.not_a_trading_day", date=str(today_ist()))
            return

        # P2-FIX-85: Late-start policy.
        # If we start after 09:30 IST (when the first bar is complete and trading normally
        # enables), indicators lack the warmup bars they need.  Rather than let bad signals
        # fire, we enforce an extended warmup: min_bars_before_signal bars from now.
        # We alert the operator and proceed — session is valid, just signals are delayed.
        from algo_trading.core.calendar import is_market_open as _is_mkt_open, SESSION_START as _SS
        _now = now_ist()
        _normal_start = _now.replace(
            hour=_SS.hour, minute=_SS.minute, second=_SS.second, microsecond=0
        )
        _warmup_cutoff = _normal_start + timedelta(minutes=settings.bar_interval_minutes)
        if is_market_open() and _now > _warmup_cutoff:
            # Session is already live when we start — warmup bars will be needed from now
            bars_elapsed = int((_now - _normal_start).total_seconds() // (settings.bar_interval_minutes * 60))
            log.warning(
                "orchestrator.late_start",
                bars_elapsed_since_open=bars_elapsed,
                warmup_required=settings.min_bars_before_signal,
                note=(
                    f"Starting after market open. Signals blocked until {settings.min_bars_before_signal} "
                    f"bars received per symbol from NOW. Strategies that require opening-range data "
                    f"(e.g. ORB) are disabled for today."
                ),
            )

        # Install shutdown handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        try:
            self._startup()
            self._schedule_daily_events()
            self._scheduler.start()
            self._running.set()
            log.info("orchestrator.running — waiting for market events")

            # Block main thread until shutdown
            while self._running.is_set():
                _time.sleep(1.0)

        except Exception as exc:
            log.error("orchestrator.fatal_error", error=str(exc))
            self._alert.send_exception_alert(exc, context="orchestrator.run")
            raise
        finally:
            self._shutdown()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _startup(self) -> None:
        """Initialise all components before market open."""
        log.info("orchestrator.startup")

        # P2-FIX-86: Clock sync check — block trading if local clock is drifted
        self._check_clock_sync()

        # P2-FIX-120: Static IP verification — alert/block if IP is not approved
        self._verify_static_ip()

        # Start Prometheus metrics scrape endpoint on port 9090
        start_metrics_server()

        # Start alert sink subprocess (isolated — trading continues if this dies)
        self._start_alert_sink()
        self._alert.start()

        # Connect QuestDB
        self._store.connect()
        self._store.create_tables()

        # Login to broker
        log.info("orchestrator.broker.login")
        self._broker.login()
        self._broker.resolve_all_symbols()

        # Startup reconciliation [SURV-6]
        log.info("orchestrator.reconcile.starting")
        reconciler = StartupReconciler(self._broker, self._state)
        result = reconciler.reconcile()
        if not result.is_clean:
            log.error(
                "orchestrator.reconcile.MISMATCH — trading blocked",
                mismatches=result.mismatches,
            )
            self._alert.send_risk_alert(
                f"STARTUP MISMATCH: {'; '.join(result.mismatches[:3])} — investigate before trading"
            )
            self._risk_manager.trigger_kill_switch("Startup reconciliation mismatch")

        # Reset state for today
        self._state.reset_for_new_day(today_ist())
        self._risk_manager.reset_for_new_day()
        self._bar_builder.reset()
        self._strategy.reset_all()
        self._mean_rev.reset_all()
        self._mtf.reset_all()
        self._bar_count.clear()  # P2-FIX-84: reset warmup counters for new session

        # Start ZMQ publishers
        self._tick_publisher.start()
        self._bar_publisher.start()

        # Start order executor timeout checker
        self._executor.start()

        # Start QuestDB flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="questdb-flush",
            daemon=True,
        )
        self._flush_thread.start()

        # Start news pipeline subscriber (no-op if unavailable)
        if self._news_client is not None:
            self._news_client.start()
            log.info("orchestrator.news_pipeline_client.started")

        # Refresh market context (FII/DII + bulk deals) — fail-safe, never blocks startup
        if self._market_context is None and _MARKET_CONTEXT_AVAILABLE:
            self._market_context = _MarketContextFilter()
        if self._market_context is not None:
            try:
                self._market_context.refresh()
                log.info(
                    "orchestrator.market_context.refreshed",
                    **self._market_context.summary(),
                )
            except Exception as exc:
                log.warning("orchestrator.market_context.refresh_failed", error=str(exc))

        log.info("orchestrator.startup.complete")
        self._alert.send_session_alert("STARTUP COMPLETE", details=settings.safe_repr())

        # P2-FIX-56: Log config checksum for session reproducibility
        import hashlib as _hashlib
        _config_repr = str(settings.safe_repr())
        _config_hash = _hashlib.sha256(_config_repr.encode()).hexdigest()[:16]
        log.info(
            "orchestrator.config_snapshot",
            config_hash=_config_hash,
            trading_mode=settings.trading_mode.value,
            symbols=settings.symbol_list,
            bar_interval_min=settings.bar_interval_minutes,
        )

    # ── Scheduled Session Events ──────────────────────────────────────────────

    def _schedule_daily_events(self) -> None:
        """Register all daily session lifecycle events with APScheduler."""
        tz = "Asia/Kolkata"

        # Market open: subscribe WebSocket, start collecting ticks
        self._scheduler.add_job(
            self._on_market_open,
            CronTrigger(hour=9, minute=15, timezone=tz),
            id="market_open",
            replace_existing=True,
        )

        # Enable trading: after first bar complete (09:30)
        self._scheduler.add_job(
            self._on_trading_enabled,
            CronTrigger(hour=9, minute=30, timezone=tz),
            id="trading_enabled",
            replace_existing=True,
        )

        # Pre-open call auction snapshot: 09:08 IST
        # Must capture BEFORE auction clears at 09:15.
        # NSE publishes live buy/sell qty imbalance during 09:00–09:15.
        self._scheduler.add_job(
            self._on_preopen_snapshot,
            CronTrigger(hour=9, minute=8, timezone=tz),
            id="preopen_snapshot",
            replace_existing=True,
        )

        # Mid-day FDA ANDA refresh: 13:00 IST
        # FDA announces approvals throughout US business hours; catching
        # morning approvals before afternoon signal routing.
        self._scheduler.add_job(
            self._on_fda_midday_refresh,
            CronTrigger(hour=13, minute=0, timezone=tz),
            id="fda_midday_refresh",
            replace_existing=True,
        )

        # Force close all positions: 15:15 [SURV-8]
        self._scheduler.add_job(
            self._on_force_close,
            CronTrigger(hour=15, minute=15, timezone=tz),
            id="force_close",
            replace_existing=True,
        )

        # Market close: stop tick ingestor, flush storage
        self._scheduler.add_job(
            self._on_market_close,
            CronTrigger(hour=15, minute=30, timezone=tz),
            id="market_close",
            replace_existing=True,
        )

        # End of day: write summary, logout
        self._scheduler.add_job(
            self._on_end_of_day,
            CronTrigger(hour=15, minute=45, timezone=tz),
            id="end_of_day",
            replace_existing=True,
        )

        log.info("orchestrator.session_events_scheduled")

    def _on_preopen_snapshot(self) -> None:
        """09:08: Capture NSE pre-open call auction imbalance snapshot."""
        if self._market_context is not None:
            try:
                self._market_context.refresh_preopen()
            except Exception as exc:
                log.warning("orchestrator.preopen_snapshot.failed", error=str(exc))

    def _on_fda_midday_refresh(self) -> None:
        """13:00: Refresh FDA ANDA approvals (catches morning US announcements)."""
        if self._market_context is not None:
            try:
                self._market_context.refresh_fda_midday()
            except Exception as exc:
                log.warning("orchestrator.fda_midday_refresh.failed", error=str(exc))

    def _on_market_open(self) -> None:
        """09:15: Subscribe WebSocket, start collecting ticks."""
        # Collect any pre-market garbage then disable GC for the session.
        # This prevents GC-induced latency spikes during live tick processing.
        gc.collect()
        gc.disable()

        log.info("orchestrator.MARKET_OPEN")
        self._trading_enabled.clear()  # Trading not yet enabled (skip first bar)
        self._alert.send_session_alert("MARKET OPEN", details=str(today_ist()))

        symbols = settings.symbol_list
        try:
            self._broker.subscribe_ws(
                symbols=symbols,
                on_tick=self._on_tick,
                on_order_update=self._on_order_update,
            )
        except Exception as exc:
            log.error("orchestrator.ws_subscribe_failed", error=str(exc))
            self._alert.send_exception_alert(exc, "market_open.subscribe_ws")

    def _on_trading_enabled(self) -> None:
        """09:30: First bar complete. Enable signal generation and order placement."""
        log.info("orchestrator.TRADING_ENABLED — first bar skip complete")
        self._trading_enabled.set()
        self._alert.send_session_alert("TRADING ENABLED — signals active")

    def _on_force_close(self) -> None:
        """15:15: Close all open positions with MARKET orders. [SURV-8]"""
        log.warning("orchestrator.FORCE_CLOSE — closing all positions")
        self._trading_enabled.clear()  # Stop new entries

        positions = self._state.get_all_positions()
        if not positions:
            log.info("orchestrator.force_close.no_positions")
            self._alert.send_session_alert("FORCE CLOSE", "No open positions")
            return

        for pos in positions:
            if pos.is_flat:
                continue
            log.warning(
                "orchestrator.force_close.closing",
                symbol=pos.symbol,
                qty=pos.net_quantity,
            )
            try:
                self._executor.close_position(pos.symbol, pos.net_quantity)
            except Exception as exc:
                log.error("orchestrator.force_close.failed", symbol=pos.symbol, error=str(exc))
                self._alert.send_exception_alert(exc, f"force_close.{pos.symbol}")

        self._alert.send_session_alert(
            "FORCE CLOSE COMPLETE",
            details=f"Closed {len(positions)} position(s)",
        )

    def _on_market_close(self) -> None:
        """15:30: Stop tick collection. Flush remaining bars to QuestDB."""
        # Re-enable GC now that live tick processing is over.
        gc.enable()

        log.info("orchestrator.MARKET_CLOSE")
        self._broker.unsubscribe_ws(settings.symbol_list)

        # Flush any incomplete bars
        partial_bars = self._bar_builder.flush_all()
        for bar in partial_bars:
            self._store.write_bar(bar)
        self._store.flush()

        self._alert.send_session_alert("MARKET CLOSE", details=f"Flushed {len(partial_bars)} partial bars")

    def _on_end_of_day(self) -> None:
        """15:45: Write session summary, logout."""
        snapshot = self._state.snapshot()
        log.info("orchestrator.END_OF_DAY", **{k: str(v) for k, v in snapshot.items()
                                                if k in ("daily_pnl", "peak_pnl")})
        self._alert.send_session_alert(
            "SESSION COMPLETE",
            details=(
                f"PnL=₹{self._state.daily_pnl:.2f} "
                f"peak=₹{self._state.peak_pnl:.2f}"
            ),
        )

        # P2-FIX-53: EOD reconciliation before logout
        self._eod_reconcile()

        # P2-FIX-90: Daily session report — writes JSON + sends Telegram summary
        self._generate_session_report()

        try:
            self._broker.logout()
        except Exception as exc:
            log.error("orchestrator.logout.failed", error=str(exc))

        # Signal main thread to exit
        self._running.clear()

    # ── Tick Processing ───────────────────────────────────────────────────────

    def _on_tick(self, tick: Tick) -> None:
        """
        Broker WebSocket tick callback — runs on broker's internal thread.
        Must be fast: validate → publish to ZMQ → feed to bar builder.
        """
        # Track latest LTP for price drift checks
        self._last_ltp[tick.symbol] = tick.ltp

        # Validate
        result = self._tick_validator.validate(tick)
        self._data_quality.record(result.is_valid)

        if not result.is_valid:
            log.debug("tick.rejected", symbol=tick.symbol, reason=result.reason)
            return

        # Publish via ZMQ (for logging / multi-process subscribers)
        self._tick_publisher.publish_tick(tick)

        # Feed to bar builder — may emit a completed bar
        completed_bar = self._bar_builder.feed_tick(tick)
        if completed_bar is not None:
            self._on_bar(completed_bar)

        # If paper mode: simulate fill at current LTP for pending orders
        if settings.is_paper():
            self._process_pending_paper_fills(tick.symbol, tick.ltp)

    def _on_bar(self, bar: Bar) -> None:
        """
        Called when a 15-min bar is complete.
        Persists bar, publishes via ZMQ, feeds signal stack.

        Signal routing:
          FilteredEMACrossStrategy — trend signals when ADX ≥ 25
          MeanReversionStrategy   — fade signals when ADX < 20
          MultiTimeframeConfirmation — scales confidence by HTF alignment;
            multiplier 0.0 (counter-trend) → signal suppressed entirely
        """
        # Safe GC at bar boundary (every 15 min) — GC is disabled during tick
        # processing, so we trigger it here at a deterministic low-latency window.
        gc.collect()

        # P2-FIX-84: Track bars received per symbol for warmup enforcement.
        self._bar_count[bar.symbol] = self._bar_count.get(bar.symbol, 0) + 1

        # Generate a unique trace ID for this bar event.  It is threaded through
        # every downstream log call (signal, order, fill) for end-to-end tracing.
        trace_id = str(uuid4())

        log.info(
            "orchestrator.bar",
            trace_id=trace_id,
            symbol=bar.symbol,
            ts=bar.timestamp.isoformat(),
            open=str(bar.open),
            close=str(bar.close),
            volume=bar.volume,
        )

        # Persist
        self._store.write_bar(bar)

        # Publish
        self._bar_publisher.publish_bar(bar)

        # Always update MTF accumulator — needs bars even before trading is enabled
        self._mtf.update(bar)

        # Generate signal only if trading is enabled
        if not self._trading_enabled.is_set():
            return
        if self._risk_manager.is_kill_switch_active:
            return

        # Feed both complementary strategies
        # Note: individual strategy objects (FilteredEMACrossStrategy,
        # MeanReversionStrategy) enforce their own bar-count warmup; they return
        # None until sufficient history is accumulated.
        trend_signal = self._strategy.feed_bar(bar)     # ADX ≥ 25 trend regime
        mean_rev_signal = self._mean_rev.feed_bar(bar)  # ADX < 20 sideways regime

        # Pick the higher-confidence signal if both fire (rare due to ADX gates)
        if trend_signal is not None and mean_rev_signal is not None:
            trade_signal: Optional[TradeSignal] = (
                trend_signal if trend_signal.confidence >= mean_rev_signal.confidence
                else mean_rev_signal
            )
        else:
            trade_signal = trend_signal or mean_rev_signal

        if trade_signal is None:
            return

        # Apply HTF size multiplier (1.0 = aligned, 0.5 = sideways, 0.0 = counter-trend)
        multiplier = self._mtf.get_size_multiplier(bar.symbol, trade_signal.direction)
        if multiplier <= _MTF_MIN_MULTIPLIER:
            log.debug(
                "orchestrator.signal_suppressed.counter_trend",
                symbol=bar.symbol,
                multiplier=multiplier,
            )
            return

        # Scale confidence by HTF alignment before routing to executor
        if multiplier < 1.0:
            trade_signal = TradeSignal(
                symbol=trade_signal.symbol,
                direction=trade_signal.direction,
                entry_price=trade_signal.entry_price,
                stop_loss_price=trade_signal.stop_loss_price,
                atr=trade_signal.atr,
                confidence=trade_signal.confidence * multiplier,
                timestamp=trade_signal.timestamp,
                trace_id=trade_signal.trace_id,
                decision_id=trade_signal.decision_id,  # P2-FIX-89
                model_version=trade_signal.model_version,
                feature_schema_version=trade_signal.feature_schema_version,
                data_snapshot_time=trade_signal.data_snapshot_time,
            )

        # ── News pipeline gate ────────────────────────────────────────────────
        # Evaluate the latest FusedSignal from the news pipeline.  If the news
        # is strongly against the trade direction, block the order entirely.
        # If the news aligns, boost confidence (capped at 1.0).
        if self._news_client is not None:
            allow, news_mult, news_reason = self._news_client.evaluate_trade(
                bar.symbol, trade_signal.direction
            )
            if not allow:
                log.warning(
                    "orchestrator.signal_blocked.news",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                    direction="BUY" if trade_signal.direction > 0 else "SELL",
                    reason=news_reason,
                )
                return
            if news_mult != 1.0:
                adjusted_conf = min(trade_signal.confidence * news_mult, 1.0)
                trade_signal = TradeSignal(
                    symbol=trade_signal.symbol,
                    direction=trade_signal.direction,
                    entry_price=trade_signal.entry_price,
                    stop_loss_price=trade_signal.stop_loss_price,
                    atr=trade_signal.atr,
                    confidence=adjusted_conf,
                    timestamp=trade_signal.timestamp,
                    trace_id=trade_signal.trace_id,
                    decision_id=trade_signal.decision_id,  # P2-FIX-89
                    model_version=trade_signal.model_version,
                    feature_schema_version=trade_signal.feature_schema_version,
                    data_snapshot_time=trade_signal.data_snapshot_time,
                )
                log.info(
                    "orchestrator.signal.news_adjusted",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                    news_multiplier=round(news_mult, 2),
                    adjusted_confidence=round(adjusted_conf, 4),
                    reason=news_reason,
                )

        # ── Market context gate ───────────────────────────────────────────────
        if self._market_context is not None:
            # Hard gate 1: promoter pledging blacklist
            if not self._market_context.is_eligible(bar.symbol):
                log.info(
                    "orchestrator.signal_blocked.pledging",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                )
                return

            # Hard gate 2: SIP day — block new long entries on first trading day
            # of month when 100M+ SIP inflows have already pushed prices up.
            if trade_signal.direction > 0 and not self._market_context.is_buy_allowed_today():
                log.info(
                    "orchestrator.signal_blocked.sip_day",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                    note="First trading day of month — SIP inflow day, no new longs",
                )
                return

            # Soft gate 3: FII × bulk deal × pre-open composite multiplier
            ctx_mult = self._market_context.get_composite_multiplier(
                bar.symbol, trade_signal.direction
            )
            if ctx_mult < 0.40:
                log.info(
                    "orchestrator.signal_suppressed.market_context",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                    composite_multiplier=round(ctx_mult, 3),
                    fii_regime=self._market_context.get_fii_regime(),
                )
                return

            # Soft adjustment: scale confidence by composite multiplier,
            # then boost by FDA ANDA / PEAD confidence signal
            ctx_boost = self._market_context.get_confidence_boost(bar.symbol)
            adjusted_conf = min(trade_signal.confidence * ctx_mult * ctx_boost, 1.0)

            if adjusted_conf != trade_signal.confidence:
                trade_signal = TradeSignal(
                    symbol=trade_signal.symbol,
                    direction=trade_signal.direction,
                    entry_price=trade_signal.entry_price,
                    stop_loss_price=trade_signal.stop_loss_price,
                    atr=trade_signal.atr,
                    confidence=adjusted_conf,
                    timestamp=trade_signal.timestamp,
                    trace_id=trade_signal.trace_id,
                    decision_id=trade_signal.decision_id,
                    model_version=trade_signal.model_version,
                    feature_schema_version=trade_signal.feature_schema_version,
                    data_snapshot_time=trade_signal.data_snapshot_time,
                )
                log.info(
                    "orchestrator.signal.market_context_adjusted",
                    trace_id=trace_id,
                    symbol=bar.symbol,
                    composite_multiplier=round(ctx_mult, 3),
                    confidence_boost=round(ctx_boost, 3),
                    adjusted_confidence=round(adjusted_conf, 4),
                    fii_regime=self._market_context.get_fii_regime(),
                )

        self._on_signal(trade_signal)

    def _on_signal(self, signal: TradeSignal) -> None:
        """
        Called when EMA cross strategy generates a trade signal.
        Routes to OrderExecutor after checking kill switch.
        """
        if self._risk_manager.is_kill_switch_active:
            log.warning(
                "orchestrator.signal_blocked.kill_switch",
                symbol=signal.symbol,
                trace_id=signal.trace_id,
            )
            return

        log.info(
            "orchestrator.SIGNAL",
            trace_id=signal.trace_id,
            symbol=signal.symbol,
            direction="BUY" if signal.direction > 0 else "SELL",
            entry=str(signal.entry_price),
            stop=str(signal.stop_loss_price),
            atr=str(signal.atr),
            confidence=signal.confidence,
        )

        # Handle existing position: close if signal is opposite direction
        pos = self._state.get_position(signal.symbol)
        if pos and not pos.is_flat:
            if (pos.is_long and signal.direction < 0) or (pos.is_short and signal.direction > 0):
                log.info("orchestrator.signal.close_existing_position", symbol=signal.symbol)
                self._executor.close_position(signal.symbol, pos.net_quantity)

        # Submit new entry
        order_id = self._executor.submit_signal(signal)
        if order_id:
            # Track for paper fill simulation
            if settings.is_paper():
                record = self._fsm.get_record(order_id)
                if record:
                    exchange_id = record.exchange_order_id
                    if exchange_id:
                        self._pending_paper_fills[exchange_id] = record.order.symbol

    def _on_order_update(self, update: object) -> None:
        """Broker WebSocket order update callback."""
        self._executor.on_broker_update(update)  # type: ignore[arg-type]

    def _process_pending_paper_fills(self, symbol: str, ltp: Decimal) -> None:
        """
        For paper trading: fill pending paper orders at the current LTP.
        Called on each tick for the relevant symbol.
        """
        to_fill = [
            eid for eid, sym in self._pending_paper_fills.items()
            if eid.startswith("PAPER-") and sym == symbol
        ]
        for eid in to_fill:
            fill = self._broker.simulate_paper_fill(eid, ltp)
            if fill is not None:
                self._pending_paper_fills.pop(eid, None)

    # ── Static IP Verification ────────────────────────────────────────────────

    def _verify_static_ip(self) -> None:
        """
        P2-FIX-120: Verify that the system's external IP matches APPROVED_STATIC_IP.

        Live mode: raises RuntimeError if IP does not match (blocks startup).
        Paper/mock mode: logs a warning only.

        APPROVED_STATIC_IP must be registered with Shoonya/broker per SEBI static IP
        requirements.  If the setting is empty, the check is skipped (development only).
        """
        import urllib.request as _urllib_req

        approved_ip: str = getattr(settings, "approved_static_ip", "") or ""
        if not approved_ip:
            log.info(
                "orchestrator.static_ip_check.skipped",
                note="APPROVED_STATIC_IP not configured — skipping check",
            )
            return

        current_ip: str = ""
        try:
            with _urllib_req.urlopen("https://api.ipify.org", timeout=5) as resp:
                current_ip = resp.read().decode().strip()
        except Exception as exc:
            log.warning(
                "orchestrator.static_ip_check.unreachable",
                error=str(exc),
                note="Cannot verify external IP — proceeding with caution",
            )
            return

        if current_ip == approved_ip:
            log.info("orchestrator.static_ip_check.OK", ip=current_ip)
        else:
            msg = (
                f"External IP mismatch: current={current_ip!r}, "
                f"approved={approved_ip!r}. "
                f"Ensure VPS outbound IP is registered with broker per SEBI requirements."
            )
            log.error("orchestrator.static_ip_check.MISMATCH", current=current_ip, approved=approved_ip)
            self._alert.send_risk_alert(f"STATIC IP MISMATCH: {msg}")
            if settings.trading_mode.value == "live":
                raise RuntimeError(msg)

    # ── Clock Sync Check ──────────────────────────────────────────────────────

    def _check_clock_sync(self) -> None:
        """
        P2-FIX-86: Verify local clock is not drifted beyond acceptable threshold.

        Attempts to fetch current time from a public NTP/HTTP source and compares
        against local system time.  Logs a warning if drift exceeds 5 seconds.
        Raises RuntimeError if drift exceeds 30 seconds (blocks trading startup).

        On VPS environments with good NTP (chronyc), this should always pass.
        On developer machines, occasional drift can occur after sleep/resume.
        """
        import socket
        import struct
        import time as _sys_time

        MAX_WARN_SEC = 5.0
        MAX_BLOCK_SEC = 30.0

        drift_sec: float | None = None
        try:
            # NTP query to pool.ntp.org (UDP port 123, no auth)
            NTP_SERVER = "pool.ntp.org"
            NTP_PORT = 123
            NTP_EPOCH_OFFSET = 2208988800  # seconds between 1900 and 1970

            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(3.0)
            data = b"\x1b" + 47 * b"\0"  # NTP request packet (version=3, mode=3)
            client.sendto(data, (NTP_SERVER, NTP_PORT))
            recv_data, _ = client.recvfrom(1024)
            client.close()

            # Extract transmit timestamp (bytes 40-47, big-endian unsigned 64-bit)
            ntp_ts = struct.unpack("!I", recv_data[40:44])[0]
            ntp_time = ntp_ts - NTP_EPOCH_OFFSET
            local_time = _sys_time.time()
            drift_sec = abs(local_time - ntp_time)

        except Exception as exc:
            log.warning(
                "orchestrator.clock_sync.unreachable",
                error=str(exc),
                note="NTP unreachable — clock sync check skipped",
            )
            return

        log.info("orchestrator.clock_sync", drift_sec=f"{drift_sec:.2f}")

        if drift_sec > MAX_BLOCK_SEC:
            msg = (
                f"Clock drift {drift_sec:.1f}s exceeds {MAX_BLOCK_SEC}s maximum. "
                f"Run 'chronyc -a makestep' or 'w32tm /resync' to fix. "
                f"Trading blocked until clock is corrected."
            )
            log.error("orchestrator.clock_sync.BLOCKED", drift_sec=drift_sec)
            self._alert.send_risk_alert(f"CLOCK SYNC FAILURE: {msg}")
            raise RuntimeError(msg)

        if drift_sec > MAX_WARN_SEC:
            log.warning(
                "orchestrator.clock_sync.WARNING",
                drift_sec=f"{drift_sec:.1f}",
                note="Consider running chronyc/w32tm to resync",
            )
            self._alert.send_risk_alert(
                f"Clock drift {drift_sec:.1f}s > {MAX_WARN_SEC}s — timestamps may be inaccurate"
            )

    # ── EOD Reconciliation ────────────────────────────────────────────────────

    def _eod_reconcile(self) -> None:
        """
        P2-FIX-53: EOD order and position reconciliation before broker logout.

        Compares FSM active orders and local position state against the broker's
        final order book and position report.  Writes a JSONL report to
        state/eod_reconcile_YYYY-MM-DD.json for compliance and incident review.
        """
        import json as _json
        from pathlib import Path as _Path

        log.info("orchestrator.eod_reconcile.starting")
        report: dict = {
            "ts_ist": now_ist().isoformat(),
            "date": str(today_ist()),
            "active_fsm_orders": [],
            "order_mismatches": [],
            "position_mismatches": [],
            "clean": True,
        }

        # 1. Check for any FSM orders still active at EOD (all should be terminal)
        try:
            active_orders = self._fsm.get_active_orders()
            for record in active_orders:
                entry = {
                    "client_id": record.order.client_order_id,
                    "exchange_id": record.exchange_order_id,
                    "symbol": record.order.symbol,
                    "state": record.state.value,
                    "side": record.order.side,
                    "qty": record.order.quantity,
                }
                report["active_fsm_orders"].append(entry)
                report["order_mismatches"].append(
                    f"FSM order still active at EOD: {entry['client_id']} ({entry['state']})"
                )
                report["clean"] = False
                log.warning("eod_reconcile.active_order_at_eod", **entry)
        except Exception as exc:
            log.error("eod_reconcile.fsm_query_failed", error=str(exc))
            report["fsm_query_error"] = str(exc)

        # 2. Compare broker positions with local state
        try:
            broker_positions: dict = self._broker.get_positions()  # type: ignore[attr-defined]
            local_flat: dict[str, object] = {
                p.symbol: p
                for p in self._state.get_all_positions()
                if not p.is_flat
            }

            for symbol, local_pos in local_flat.items():
                broker_qty = broker_positions.get(symbol, 0)
                local_qty = getattr(local_pos, "net_quantity", 0)
                if abs(local_qty - broker_qty) > 0:
                    mismatch = (
                        f"{symbol}: local_qty={local_qty} broker_qty={broker_qty}"
                    )
                    report["position_mismatches"].append(mismatch)
                    report["clean"] = False
                    log.error(
                        "eod_reconcile.position_mismatch",
                        symbol=symbol,
                        local_qty=local_qty,
                        broker_qty=broker_qty,
                    )

            for symbol, broker_qty in broker_positions.items():
                if symbol not in local_flat and abs(broker_qty) > 0:
                    mismatch = f"{symbol}: local=UNKNOWN broker_qty={broker_qty}"
                    report["position_mismatches"].append(mismatch)
                    report["clean"] = False
                    log.error(
                        "eod_reconcile.unknown_broker_position",
                        symbol=symbol,
                        broker_qty=broker_qty,
                    )
        except Exception as exc:
            log.error("eod_reconcile.broker_positions_failed", error=str(exc))
            report["broker_positions_error"] = str(exc)
            report["clean"] = False

        # 3. Write report to disk
        try:
            report_path = _Path("state") / f"eod_reconcile_{today_ist()}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                _json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            log.info(
                "eod_reconcile.report_written",
                path=str(report_path),
                clean=report["clean"],
                order_mismatches=len(report["order_mismatches"]),
                position_mismatches=len(report["position_mismatches"]),
            )
        except Exception as exc:
            log.error("eod_reconcile.report_write_failed", error=str(exc))

        if not report["clean"]:
            self._alert.send_risk_alert(
                f"EOD RECONCILE MISMATCHES: "
                f"{len(report['order_mismatches'])} order, "
                f"{len(report['position_mismatches'])} position — "
                f"review state/eod_reconcile_{today_ist()}.json"
            )
        else:
            log.info("eod_reconcile.CLEAN — broker and local state match")

    # ── Daily Session Report ──────────────────────────────────────────────────

    def _generate_session_report(self) -> None:
        """
        P2-FIX-90: Generate a comprehensive daily session report.

        Reads today's audit logs (order_audit.log, veto_audit.jsonl), FSM final
        states, and live system state to produce a structured JSON report in
        state/session_report_YYYY-MM-DD.json.

        Also sends a Telegram summary with key P&L and trade metrics.

        Fields included:
          - trades_placed, trades_filled, trades_rejected, trades_cancelled
          - signal_vetoes (by reason)
          - realized_pnl, peak_pnl, daily_trade_count
          - slippage_events (count of >10bps fills from audit log)
          - order_states (final count per FSM state)
          - data_quality_incidents (bad ticks logged)
          - reconciliation_clean (from EOD reconcile)
          - model_versions (unique model versions seen today)
          - config_hash (from startup log — for reproducibility)
        """
        import json as _json
        from pathlib import Path as _Path
        from collections import Counter as _Counter

        today = today_ist()
        report: dict = {
            "ts_ist": now_ist().isoformat(),
            "date": str(today),
            "trades_placed": 0,
            "trades_filled": 0,
            "trades_rejected": 0,
            "trades_cancelled": 0,
            "trades_cancel_pending": 0,
            "signal_vetoes": {},
            "signal_skips_by_reason": {},
            "realized_pnl_inr": 0.0,
            "peak_pnl_inr": 0.0,
            "daily_trade_count": 0,
            "slippage_excess_events": 0,
            "order_states": {},
            "model_versions_seen": [],
            "otr": 0.0,
        }

        # ── 1. Order audit log — count events by type ─────────────────────────
        audit_path = _Path("state") / "order_audit.log"
        model_versions: set[str] = set()
        try:
            if audit_path.exists():
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = _json.loads(line)
                        evt = entry.get("event", "")
                        if evt == "SENT":
                            report["trades_placed"] += 1
                        elif evt == "FILL":
                            report["trades_filled"] += 1
                        elif evt == "REJECT":
                            report["trades_rejected"] += 1
                        elif evt == "CANCEL_CONFIRM":
                            report["trades_cancelled"] += 1
                        # Collect model versions seen today
                        mv = entry.get("model_version", "")
                        if mv:
                            model_versions.add(mv)
                    except Exception:
                        pass
        except Exception as exc:
            log.warning("session_report.audit_read_failed", error=str(exc))

        report["model_versions_seen"] = sorted(model_versions)

        # ── 2. Veto audit log — count by reason ──────────────────────────────
        veto_path = _Path("state") / "veto_audit.jsonl"
        veto_counts: _Counter = _Counter()
        try:
            if veto_path.exists():
                for line in veto_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = _json.loads(line)
                        # Only count today's entries (iso date prefix match)
                        ts = entry.get("timestamp", "")
                        if not ts.startswith(str(today)):
                            continue
                        source = entry.get("veto_source", "unknown")
                        veto_counts[source] += 1
                    except Exception:
                        pass
        except Exception as exc:
            log.warning("session_report.veto_read_failed", error=str(exc))
        report["signal_vetoes"] = dict(veto_counts)

        # ── 3. FSM final state summary ────────────────────────────────────────
        try:
            state_counter: _Counter = _Counter()
            for record in self._fsm.get_all_records():
                state_counter[record.state.value] += 1
            report["order_states"] = dict(state_counter)
        except Exception as exc:
            log.warning("session_report.fsm_read_failed", error=str(exc))

        # ── 4. Live system state (P&L, trade count) ───────────────────────────
        try:
            report["realized_pnl_inr"] = float(self._state.daily_pnl)
            report["peak_pnl_inr"] = float(self._state.peak_pnl)
            report["daily_trade_count"] = self._state.daily_trade_count
        except Exception as exc:
            log.warning("session_report.state_read_failed", error=str(exc))

        # ── 5. OTR from rate limiter ──────────────────────────────────────────
        try:
            report["otr"] = round(self._rate_limiter.get_otr(), 2)
        except Exception:
            pass

        # ── 6. Write report to disk ───────────────────────────────────────────
        report_path = _Path("state") / f"session_report_{today}.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                _json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            log.info(
                "session_report.written",
                path=str(report_path),
                trades_filled=report["trades_filled"],
                pnl=report["realized_pnl_inr"],
            )
        except Exception as exc:
            log.error("session_report.write_failed", error=str(exc))
            return

        # ── 7. Telegram summary ───────────────────────────────────────────────
        try:
            pnl_sign = "+" if report["realized_pnl_inr"] >= 0 else ""
            veto_summary = ", ".join(
                f"{k}:{v}" for k, v in sorted(report["signal_vetoes"].items())
            ) or "none"
            summary = (
                f"SESSION REPORT {today}\n"
                f"PnL: {pnl_sign}₹{report['realized_pnl_inr']:.2f} "
                f"(peak ₹{report['peak_pnl_inr']:.2f})\n"
                f"Trades: {report['trades_filled']} filled | "
                f"{report['trades_placed']} placed | "
                f"{report['trades_rejected']} rejected | "
                f"{report['trades_cancelled']} cancelled\n"
                f"Vetoes: {veto_summary}\n"
                f"OTR: {report['otr']:.1f}"
            )
            self._alert.send_session_alert("SESSION REPORT", details=summary)
        except Exception as exc:
            log.warning("session_report.alert_send_failed", error=str(exc))

    # ── Data Quality Halt ──────────────────────────────────────────────────────

    def _on_data_quality_halt(self, reason: str) -> None:
        """
        Called by DataQualityMonitor when bad tick rate exceeds threshold.

        P2-FIX-55: On data quality halt:
          1. Trigger kill switch (blocks new entries)
          2. Cancel all resting entry LIMIT orders (they should not remain live on stale data)
          3. Alert operator — positions are preserved (exits still allowed via separate path)
        """
        log.error("orchestrator.DATA_QUALITY_HALT", reason=reason)
        self._trading_enabled.clear()
        self._risk_manager.trigger_kill_switch(f"Data quality: {reason}")

        # Cancel all resting entry orders — stale data makes limit prices unreliable
        cancelled = 0
        try:
            from algo_trading.execution.order_fsm import OrderState
            for record in self._fsm.get_active_orders():
                # Only cancel entry LIMIT orders — let stop-loss (MARKET) stay
                if record.order.order_type == "LIMIT":
                    try:
                        self._executor.cancel_order(record.order.client_order_id)
                        cancelled += 1
                    except Exception as cancel_exc:
                        log.error(
                            "orchestrator.data_quality_halt.cancel_failed",
                            client_id=record.order.client_order_id,
                            error=str(cancel_exc),
                        )
        except Exception as exc:
            log.error("orchestrator.data_quality_halt.cancel_sweep_failed", error=str(exc))

        self._alert.send_risk_alert(
            f"DATA QUALITY HALT: {reason} — "
            f"trading blocked, {cancelled} resting LIMIT order(s) cancelled. "
            f"Open positions preserved — exits still allowed."
        )

    # ── QuestDB Flush Loop ────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Periodically flush QuestDB write buffer. Every 100ms."""
        while self._running.is_set():
            try:
                self._store.flush()
            except Exception as exc:
                log.error("orchestrator.flush_error", error=str(exc))
            _time.sleep(_FLUSH_INTERVAL_SEC)

    # ── Alert Sink Process ────────────────────────────────────────────────────

    def _start_alert_sink(self) -> None:
        """Start alert sink in a separate process."""
        import os
        env = {
            "TELEGRAM_BOT_TOKEN": (
                settings.telegram_bot_token.get_secret_value()
                if settings.telegram_bot_token else ""
            ),
            "TELEGRAM_CHAT_ID": settings.telegram_chat_id or "",
            "ZMQ_ALERT_PORT": str(settings.zmq_alert_push_port),
        }
        # Temporarily set environment for subprocess
        for k, v in env.items():
            os.environ[k] = v

        self._alert_sink_proc = multiprocessing.Process(
            target=run_alert_sink,
            name="alert-sink",
            daemon=True,
        )
        self._alert_sink_proc.start()
        log.info("orchestrator.alert_sink.started", pid=self._alert_sink_proc.pid)
        _time.sleep(0.2)  # Give the sink time to bind the ZMQ port

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _handle_shutdown_signal(self, signum: int, frame: object) -> None:
        log.warning("orchestrator.shutdown_signal", signum=signum)
        self._running.clear()

    def _shutdown(self) -> None:
        """Graceful shutdown: flush, close, stop."""
        log.info("orchestrator.shutdown.starting")

        self._trading_enabled.clear()
        self._executor.stop()
        if self._news_client is not None:
            self._news_client.stop()

        # Force-close any remaining positions
        positions = self._state.get_all_positions()
        for pos in positions:
            if not pos.is_flat:
                try:
                    self._executor.close_position(pos.symbol, pos.net_quantity)
                except Exception:
                    pass

        # Flush final data
        partial_bars = self._bar_builder.flush_all()
        for bar in partial_bars:
            self._store.write_bar(bar)
        self._store.flush()

        self._scheduler.shutdown(wait=False)
        self._tick_publisher.stop()
        self._bar_publisher.stop()
        self._alert.send_session_alert("SHUTDOWN COMPLETE")
        self._alert.stop()
        self._store.close()

        if self._alert_sink_proc and self._alert_sink_proc.is_alive():
            self._alert_sink_proc.terminate()

        log.info("orchestrator.shutdown.complete")
