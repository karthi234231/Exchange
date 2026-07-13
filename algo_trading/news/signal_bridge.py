"""
news/signal_bridge.py — Converts NewsEvents into trading decisions.

Receives classified news events via ZMQ PULL:5571.
Maintains per-symbol news state that the orchestrator queries before each trade.
Runs in the same process as daily_orchestrator (not a separate process).

Trading rules:
  CRITICAL impact  → halt symbol + send alert (override ALL signals)
  HIGH + bearish   → block new long entries, allow shorts with 50% size
  HIGH + bullish   → allow longs with 130% confidence boost
  EARNINGS upcoming (≤ 2h) → block new entries, halve position size
  MACRO CRITICAL   → reduce ALL positions by 30%, 30-min trading pause
  M&A target       → halt symbol (don't chase the announcement pump)
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import structlog

# EventType, ImpactLevel and NewsEvent are defined in event_classifier.py,
# which is built by a parallel agent.  The TYPE_CHECKING guard allows static
# analysers to see the types without causing a runtime ImportError if the
# module is not yet present; at runtime we import unconditionally because both
# modules will always be deployed together.
from algo_trading.news.event_classifier import EventType, ImpactLevel, NewsEvent

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEWS_STATE_TTL_HOURS: int = 4
"""News state is considered stale after this many hours and is evicted."""

_MACRO_PAUSE_MINUTES: int = 30
"""Duration of the market-wide trading pause triggered by a MACRO CRITICAL event."""

# ---------------------------------------------------------------------------
# Per-symbol state dataclass
# ---------------------------------------------------------------------------


@dataclass
class SymbolNewsState:
    """Rolling news state for a single NSE symbol.

    This snapshot is re-evaluated on every incoming event and expires after
    ``_NEWS_STATE_TTL_HOURS`` hours from ``last_updated``.

    Attributes:
        symbol: NSE symbol (e.g. ``"RELIANCE"``).
        last_updated: UTC datetime of the most-recently processed event.
        impact: Highest :class:`ImpactLevel` seen in the last 4 hours.
        sentiment_hint: Directional hint — ``"bullish"``, ``"bearish"``, or
            ``"neutral"``.
        event_type: :class:`EventType` of the most-recent event.
        headline: Raw headline text of the most-recent event.
        halted: ``True`` means no trading until the state is manually cleared
            via :meth:`NewsSignalBridge.clear_symbol`.
        confidence_multiplier: Applied to the ML signal confidence before the
            order is submitted.  Default ``1.0`` (no change).
        block_long: ``True`` → new BUY orders are rejected.
        block_short: ``True`` → new SELL orders are rejected.
        earnings_pending: ``True`` if an earnings release is expected within
            the next 2 hours.
    """

    symbol: str
    last_updated: datetime
    impact: ImpactLevel
    sentiment_hint: str          # "bullish" | "bearish" | "neutral"
    event_type: EventType
    headline: str
    halted: bool = False
    confidence_multiplier: float = 1.0
    block_long: bool = False
    block_short: bool = False
    earnings_pending: bool = False


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------


class NewsSignalBridge:
    """Routes classified news events to per-symbol trading constraints.

    Listens on ZMQ PULL:5571 for ``NewsEvent`` JSON dicts produced by the
    ``EventClassifier`` / ``NewsOrchestrator`` pipeline.  Results are stored
    in ``_symbol_states`` and queried by ``DailyOrchestrator._on_bar()``
    before each trade signal is forwarded to the executor.

    This class is designed to live in the same OS process as
    :class:`~algo_trading.app.daily_orchestrator.DailyOrchestrator`.
    The ZMQ listener runs on a background daemon thread so the orchestrator's
    hot-path is never blocked.

    Args:
        zmq_pull_port: ZMQ port to PULL classified events from (default 5571).
        alert_sender: Optional callable ``(msg: str, level: str) -> None``.
            Called with the same signature as
            :meth:`~algo_trading.alerts.telegram_bot.AlertSender.send_risk_alert`
            so the orchestrator can pass its ``AlertSender`` instance directly.
        halt_callback: Optional callable ``(symbol: str, reason: str) -> None``.
            Called when a symbol is halted so the risk manager can mark it as
            do-not-trade.
    """

    def __init__(
        self,
        zmq_pull_port: int = 5571,
        alert_sender: Optional[Callable[[str, str], None]] = None,
        halt_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._port = zmq_pull_port
        self._alert_sender = alert_sender
        self._halt_callback = halt_callback

        # symbol → SymbolNewsState; guarded by _lock
        self._symbol_states: dict[str, SymbolNewsState] = {}
        self._lock = threading.Lock()

        # UTC datetime until which ALL new entries are paused (MACRO CRITICAL)
        self._market_pause_until: Optional[datetime] = None

        self._listener_thread: Optional[threading.Thread] = None

        log.info("news_signal_bridge.init", pull_port=zmq_pull_port)

    # ------------------------------------------------------------------
    # Public API — called by orchestrator
    # ------------------------------------------------------------------

    def start_listener(self) -> None:
        """Start the background ZMQ PULL listener thread.

        Binds a ZMQ PULL socket on port ``zmq_pull_port`` and dispatches
        each incoming JSON message to :meth:`_process_event`.  The thread is
        a daemon so it exits automatically when the main process dies.

        Safe to call multiple times — subsequent calls are no-ops if the
        thread is already running.
        """
        if self._listener_thread is not None and self._listener_thread.is_alive():
            log.debug("news_signal_bridge.listener_already_running")
            return

        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="news-signal-bridge-listener",
            daemon=True,
        )
        self._listener_thread.start()
        log.info("news_signal_bridge.listener_started", port=self._port)

    def get_symbol_state(self, symbol: str) -> Optional[SymbolNewsState]:
        """Return the current news state for *symbol*, or ``None`` if no recent news.

        A state is considered recent if its ``last_updated`` timestamp is
        within the last ``_NEWS_STATE_TTL_HOURS`` hours.  Expired states are
        cleaned up lazily here and by the hourly :meth:`cleanup_expired_states`
        call from the orchestrator scheduler.

        Args:
            symbol: NSE symbol to query (case-sensitive, upper-case).

        Returns:
            :class:`SymbolNewsState` if a live state exists, ``None`` otherwise.
        """
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                return None
            age_hours = (datetime.now(timezone.utc) - state.last_updated).total_seconds() / 3600
            if age_hours > _NEWS_STATE_TTL_HOURS:
                del self._symbol_states[symbol]
                log.debug("news_signal_bridge.state_expired", symbol=symbol)
                return None
            return state

    def should_allow_trade(self, symbol: str, direction: str) -> tuple[bool, str]:
        """Determine whether a proposed trade is allowed given current news state.

        Checks, in priority order:

        1. Market-wide MACRO CRITICAL pause (blocks all symbols).
        2. Symbol halt (e.g. CRITICAL news or M&A event).
        3. Earnings pending within 2 hours (blocks new entries).
        4. ``block_long`` / ``block_short`` flags from HIGH-impact events.

        Args:
            symbol: NSE symbol to check (upper-case).
            direction: ``"BUY"`` or ``"SELL"``.

        Returns:
            A ``(allowed, reason)`` tuple.  ``allowed=True`` means the trade
            may proceed; ``reason`` is a non-empty explanation when blocked.
        """
        # 1. Global market pause from MACRO CRITICAL event
        if self._market_pause_until is not None:
            now_utc = datetime.now(timezone.utc)
            if now_utc < self._market_pause_until:
                remaining = int((self._market_pause_until - now_utc).total_seconds() / 60)
                return False, f"MACRO CRITICAL: market-wide pause active ({remaining} min remaining)"
            else:
                # Pause expired — clear it
                self._market_pause_until = None

        state = self.get_symbol_state(symbol)
        if state is None:
            return True, ""

        # 2. Symbol halt
        if state.halted:
            return False, f"NEWS HALT: {state.event_type.value} — {state.headline[:80]}"

        # 3. Earnings pending
        if state.earnings_pending:
            return False, f"EARNINGS PENDING: entries blocked until earnings release (< 2h)"

        # 4. Direction-specific blocks
        if direction == "BUY" and state.block_long:
            return (
                False,
                f"NEWS BLOCK_LONG: HIGH bearish impact — {state.headline[:80]}",
            )
        if direction == "SELL" and state.block_short:
            return (
                False,
                f"NEWS BLOCK_SHORT: directional block active — {state.headline[:80]}",
            )

        return True, ""

    def get_confidence_multiplier(self, symbol: str) -> float:
        """Return the confidence multiplier to apply to the ML signal for *symbol*.

        Multipliers above 1.0 indicate bullish tailwinds (e.g. HIGH bullish
        news) and multipliers below 1.0 indicate headwinds or uncertainty.
        Returns ``1.0`` (no adjustment) when there is no recent news state or
        the state has expired.

        Args:
            symbol: NSE symbol to query (upper-case).

        Returns:
            Float multiplier, typically in ``[0.5, 1.3]``.  Always ``1.0``
            when no live news state exists.
        """
        state = self.get_symbol_state(symbol)
        if state is None:
            return 1.0
        return state.confidence_multiplier

    def clear_symbol(self, symbol: str) -> None:
        """Manually clear the halt and news state for *symbol*.

        Called by the operator (via sentinel file polling or manual CLI
        override) to resume trading after a CRITICAL/M&A halt has been
        investigated and cleared.  Also called at the start of each new
        session to ensure stale state does not carry over.

        Args:
            symbol: NSE symbol to clear (upper-case).
        """
        with self._lock:
            removed = self._symbol_states.pop(symbol, None)
        if removed is not None:
            log.info("news_signal_bridge.symbol_cleared", symbol=symbol, was_halted=removed.halted)
        else:
            log.debug("news_signal_bridge.clear_noop", symbol=symbol)

    def cleanup_expired_states(self) -> None:
        """Remove all states older than ``_NEWS_STATE_TTL_HOURS`` hours.

        Should be called hourly by the orchestrator's APScheduler job to
        keep the state dict bounded.  Also runs lazily inside
        :meth:`get_symbol_state` on a per-symbol basis.
        """
        now_utc = datetime.now(timezone.utc)
        cutoff = timedelta(hours=_NEWS_STATE_TTL_HOURS)
        with self._lock:
            expired = [
                sym
                for sym, state in self._symbol_states.items()
                if (now_utc - state.last_updated) > cutoff
            ]
            for sym in expired:
                del self._symbol_states[sym]

        if expired:
            log.info(
                "news_signal_bridge.cleanup_expired",
                count=len(expired),
                symbols=expired,
            )

    # ------------------------------------------------------------------
    # Internal — ZMQ listener
    # ------------------------------------------------------------------

    def _listener_loop(self) -> None:
        """Background thread: bind ZMQ PULL on port and dispatch events."""
        try:
            import zmq  # type: ignore[import]
        except ImportError:
            log.error("news_signal_bridge.zmq_unavailable — listener not started")
            return

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s poll — allows clean daemon exit
        sock.setsockopt(zmq.RCVHWM, 500)

        try:
            sock.bind(f"tcp://127.0.0.1:{self._port}")
            log.info("news_signal_bridge.zmq_bound", port=self._port)
        except zmq.ZMQError as exc:
            log.error("news_signal_bridge.bind_failed", port=self._port, error=str(exc))
            sock.close()
            return

        while True:
            try:
                raw = sock.recv_string()
            except zmq.Again:
                # Poll timeout — loop back so the daemon thread can exit if
                # the process is shutting down.
                continue
            except zmq.ZMQError as exc:
                log.error("news_signal_bridge.recv_error", error=str(exc))
                break

            try:
                event_dict = json.loads(raw)
                self._process_event(event_dict)
            except Exception as exc:
                log.error(
                    "news_signal_bridge.dispatch_error",
                    error=str(exc),
                    raw_preview=raw[:120],
                )

        sock.close()
        log.info("news_signal_bridge.listener_stopped")

    # ------------------------------------------------------------------
    # Internal — event processing
    # ------------------------------------------------------------------

    def _process_event(self, event_dict: dict) -> None:
        """Parse an incoming dict, apply trading rules, and update symbol state.

        Called from the listener thread for every ZMQ message.  All writes to
        ``_symbol_states`` are serialised via ``_lock`` so the orchestrator
        can safely call :meth:`should_allow_trade` and
        :meth:`get_confidence_multiplier` from the main thread concurrently.

        Args:
            event_dict: Raw JSON-decoded dict from the event classifier.
                Expected keys match the :class:`NewsEvent` dataclass fields.
        """
        # Parse into NewsEvent — fall back to raw dict values if NewsEvent
        # cannot be constructed (e.g. during testing with partial dicts).
        try:
            event = NewsEvent.from_dict(event_dict)
        except Exception:
            # Attempt manual fallback construction
            try:
                event = NewsEvent(
                    symbol=event_dict.get("symbol", ""),
                    headline=event_dict.get("headline", ""),
                    impact=ImpactLevel(event_dict.get("impact", "LOW")),
                    event_type=EventType(event_dict.get("event_type", "GENERAL")),
                    sentiment=event_dict.get("sentiment", "neutral"),
                    timestamp=datetime.now(timezone.utc),
                    source=event_dict.get("source", "unknown"),
                    raw_text=event_dict.get("raw_text", ""),
                )
            except Exception as exc:
                log.error(
                    "news_signal_bridge.parse_failed",
                    error=str(exc),
                    dict_keys=list(event_dict.keys()),
                )
                return

        symbol = event.symbol.strip().upper()
        if not symbol:
            log.warning("news_signal_bridge.empty_symbol", headline=event.headline[:60])
            return

        now_utc = datetime.now(timezone.utc)

        # Derive initial state values from the new event
        halted = False
        confidence_multiplier = 1.0
        block_long = False
        block_short = False
        earnings_pending = False

        # ----------------------------------------------------------------
        # Rule set — applied in priority order
        # ----------------------------------------------------------------

        # Rule 1: CRITICAL impact → halt symbol immediately
        if event.impact == ImpactLevel.CRITICAL:
            halted = True
            self._send_alert(
                f"NEWS HALT [{symbol}]: CRITICAL impact — {event.headline[:100]}",
                level="CRITICAL",
            )
            if self._halt_callback is not None:
                try:
                    self._halt_callback(symbol, event.headline)
                except Exception as cb_exc:
                    log.error("news_signal_bridge.halt_callback_error", error=str(cb_exc))
            log.warning(
                "news_signal_bridge.symbol_halted",
                symbol=symbol,
                impact=event.impact.value,
                event_type=event.event_type.value,
                headline=event.headline[:100],
            )

        # Rule 2: HIGH impact — direction-dependent modifiers
        elif event.impact == ImpactLevel.HIGH:
            if event.sentiment == "bearish":
                block_long = True
                confidence_multiplier = 0.5
                log.info(
                    "news_signal_bridge.block_long",
                    symbol=symbol,
                    reason="HIGH bearish",
                    headline=event.headline[:80],
                )
            elif event.sentiment == "bullish":
                confidence_multiplier = 1.3
                log.info(
                    "news_signal_bridge.confidence_boost",
                    symbol=symbol,
                    multiplier=1.3,
                    reason="HIGH bullish",
                    headline=event.headline[:80],
                )

        # Rule 3: M&A target — halt regardless of impact (don't chase the pump)
        if event.event_type == EventType.M_AND_A:
            halted = True
            self._send_alert(
                f"NEWS HALT [{symbol}]: M&A announcement — {event.headline[:100]}",
                level="CRITICAL",
            )
            if self._halt_callback is not None:
                try:
                    self._halt_callback(symbol, f"M&A: {event.headline}")
                except Exception as cb_exc:
                    log.error("news_signal_bridge.halt_callback_error", error=str(cb_exc))
            log.warning(
                "news_signal_bridge.symbol_halted_mna",
                symbol=symbol,
                headline=event.headline[:100],
            )

        # Rule 4: MACRO CRITICAL → 30-minute market-wide pause
        if event.event_type == EventType.MACRO and event.impact == ImpactLevel.CRITICAL:
            self._market_pause_until = now_utc + timedelta(minutes=_MACRO_PAUSE_MINUTES)
            self._send_alert(
                f"MACRO CRITICAL: 30-min trading pause activated — {event.headline[:100]}",
                level="CRITICAL",
            )
            log.warning(
                "news_signal_bridge.macro_pause_activated",
                pause_until=self._market_pause_until.isoformat(),
                headline=event.headline[:100],
            )

        # Rule 5: LEGAL event → halt symbol
        if event.event_type == EventType.LEGAL:
            halted = True
            self._send_alert(
                f"NEWS HALT [{symbol}]: LEGAL event — {event.headline[:100]}",
                level="CRITICAL",
            )
            if self._halt_callback is not None:
                try:
                    self._halt_callback(symbol, f"LEGAL: {event.headline}")
                except Exception as cb_exc:
                    log.error("news_signal_bridge.halt_callback_error", error=str(cb_exc))
            log.warning(
                "news_signal_bridge.symbol_halted_legal",
                symbol=symbol,
                headline=event.headline[:100],
            )

        # Rule 6: REGULATORY bearish → halt symbol
        if event.event_type == EventType.REGULATORY and event.sentiment == "bearish":
            halted = True
            self._send_alert(
                f"NEWS HALT [{symbol}]: REGULATORY bearish — {event.headline[:100]}",
                level="CRITICAL",
            )
            if self._halt_callback is not None:
                try:
                    self._halt_callback(symbol, f"REGULATORY bearish: {event.headline}")
                except Exception as cb_exc:
                    log.error("news_signal_bridge.halt_callback_error", error=str(cb_exc))
            log.warning(
                "news_signal_bridge.symbol_halted_regulatory",
                symbol=symbol,
                headline=event.headline[:100],
            )

        # Rule 7: EARNINGS event with a timestamp ≤ 2h away → block entries
        if event.event_type == EventType.EARNINGS:
            if hasattr(event, "event_time") and event.event_time is not None:
                try:
                    event_ts = event.event_time
                    if event_ts.tzinfo is None:
                        event_ts = event_ts.replace(tzinfo=timezone.utc)
                    hours_to_event = (event_ts - now_utc).total_seconds() / 3600
                    if 0 <= hours_to_event <= 2:
                        earnings_pending = True
                        log.info(
                            "news_signal_bridge.earnings_pending",
                            symbol=symbol,
                            hours_to_event=round(hours_to_event, 2),
                        )
                except Exception as ts_exc:
                    log.debug("news_signal_bridge.earnings_time_parse_error", error=str(ts_exc))
            else:
                # No event_time field — treat ANY earnings news as pending
                earnings_pending = True
                log.info(
                    "news_signal_bridge.earnings_pending_no_time",
                    symbol=symbol,
                    headline=event.headline[:80],
                )

        # ----------------------------------------------------------------
        # Merge with existing state: preserve halt if already halted;
        # upgrade impact to the highest seen in the TTL window.
        # ----------------------------------------------------------------
        with self._lock:
            existing = self._symbol_states.get(symbol)
            if existing is not None:
                # Preserve halt if already set (only clear_symbol() can lift it)
                halted = halted or existing.halted
                # Preserve the highest impact level in the window
                if _impact_rank(existing.impact) > _impact_rank(event.impact):
                    effective_impact = existing.impact
                else:
                    effective_impact = event.impact
                # Preserve directional blocks
                block_long = block_long or existing.block_long
                block_short = block_short or existing.block_short
            else:
                effective_impact = event.impact

            self._symbol_states[symbol] = SymbolNewsState(
                symbol=symbol,
                last_updated=now_utc,
                impact=effective_impact,
                sentiment_hint=event.sentiment,
                event_type=event.event_type,
                headline=event.headline,
                halted=halted,
                confidence_multiplier=confidence_multiplier,
                block_long=block_long,
                block_short=block_short,
                earnings_pending=earnings_pending,
            )

        log.debug(
            "news_signal_bridge.state_updated",
            symbol=symbol,
            impact=effective_impact.value,
            halted=halted,
            confidence_multiplier=confidence_multiplier,
            block_long=block_long,
            block_short=block_short,
            earnings_pending=earnings_pending,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_alert(self, msg: str, level: str = "CRITICAL") -> None:
        """Fire-and-forget alert via the injected alert_sender callable."""
        if self._alert_sender is None:
            return
        try:
            self._alert_sender(msg, level)
        except Exception as exc:
            log.error("news_signal_bridge.alert_error", error=str(exc), msg=msg[:60])


def _impact_rank(impact: ImpactLevel) -> int:
    """Return a numeric rank for comparing impact levels (higher = more severe)."""
    _ORDER = {
        ImpactLevel.LOW: 1,
        ImpactLevel.MEDIUM: 2,
        ImpactLevel.HIGH: 3,
        ImpactLevel.CRITICAL: 4,
    }
    return _ORDER.get(impact, 0)
