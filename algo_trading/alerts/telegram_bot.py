"""
alerts/telegram_bot.py — Non-blocking alert sender. [ARCH-FIX-3]

Sends alerts via ZMQ PUSH to the AlertSink process.
The call returns in < 1µs. Telegram I/O (200-2000ms) is isolated in alert_sink.py.

NEVER call the Telegram API from this module.
If the ZMQ queue is full (> SNDHWM), messages are dropped silently.
Alerts must NEVER crash or block the execution thread.

Alert types:
  - trade_alert: order entry, exit, fill
  - risk_alert: kill switch, circuit halt, drawdown
  - exception_alert: uncaught exceptions, broker errors
  - session_alert: market open, close, reconnect events
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Optional

import structlog
import zmq

from algo_trading.config import settings

log = structlog.get_logger(__name__)

_SNDHWM = 200   # Drop messages if queue exceeds this — execution must not block

# ---------------------------------------------------------------------------
# Module-level singleton + convenience function (used by engine.py and others
# that do: from algo_trading.alerts.telegram_bot import send_alert)
# ---------------------------------------------------------------------------

_default_sender: "AlertSender | None" = None


def send_alert(message: str, level: str = "INFO") -> None:
    """Fire-and-forget alert via the singleton AlertSender.

    Creates and starts the sender on first call (lazy init).
    Never raises — alert failures must not affect the calling thread.

    Args:
        message: Human-readable alert text.
        level:   Severity string — "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    global _default_sender
    try:
        if _default_sender is None:
            _default_sender = AlertSender()
            _default_sender.start()
        _default_sender._push({"type": "alert", "msg": message, "level": level})
    except Exception:
        pass


class AlertSender:
    """
    Non-blocking alert sender using ZMQ PUSH socket.

    Usage:
        sender = AlertSender()
        sender.start()
        sender.send_trade_alert("RELIANCE", "BUY LIMIT", price=2850, quantity=10)
        sender.stop()

    All send methods are fire-and-forget. They never raise.
    """

    # Minimum seconds between sends for each alert level.
    # RISK and CRITICAL are 0.0 — they always bypass deduplication.
    _COOLDOWN_SECONDS: dict[str, float] = {
        "RISK": 0.0,
        "CRITICAL": 0.0,
        "SESSION": 30.0,
        "INFO": 60.0,
        "WARN": 30.0,
        "WARNING": 30.0,  # alias used by send_risk_alert
        "ERROR": 0.0,     # exceptions always go through
    }

    def __init__(self, port: Optional[int] = None) -> None:
        self._port = port or settings.zmq_alert_push_port
        self._ctx = zmq.Context.instance()
        self._sock: Optional[zmq.Socket] = None
        # Maps level/type key → last-sent monotonic timestamp
        self._last_sent: dict[str, float] = {}

    def start(self) -> None:
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, _SNDHWM)
        self._sock.connect(f"tcp://127.0.0.1:{self._port}")
        log.info("alert_sender.started", port=self._port)

    def stop(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def send_trade_alert(
        self,
        symbol: str,
        action: str,
        price: Decimal,
        quantity: int,
        reason: str = "",
    ) -> None:
        """Alert on order entry, fill, skip, or exit."""
        msg = f"TRADE | {symbol} {action} qty={quantity} price=₹{price:.2f}"
        if reason:
            msg += f" | {reason}"
        self._push({"type": "trade", "msg": msg, "level": "INFO"})

    def send_risk_alert(self, message: str) -> None:
        """Alert on risk events: kill switch, circuit halt, drawdown breach."""
        self._push({"type": "risk", "msg": f"RISK | {message}", "level": "WARNING"})

    def send_exception_alert(self, exc: Exception, context: str = "") -> None:
        """Alert on uncaught exceptions or broker errors."""
        msg = f"ERROR | {type(exc).__name__}: {exc}"
        if context:
            msg = f"{context} | {msg}"
        self._push({"type": "exception", "msg": msg, "level": "ERROR"})

    def send_session_alert(self, event: str, details: str = "") -> None:
        """Alert on session lifecycle events: open, close, reconnect."""
        msg = f"SESSION | {event}"
        if details:
            msg += f": {details}"
        self._push({"type": "session", "msg": msg, "level": "INFO"})

    def _push(self, payload: dict) -> None:
        """Send payload via ZMQ NOBLOCK with per-level deduplication.

        RISK and CRITICAL levels always bypass deduplication (cooldown=0).
        All other levels are suppressed if the cooldown period has not elapsed
        since the last send of the same level.
        """
        if self._sock is None:
            return

        level = payload.get("level", "INFO").upper()
        cooldown = self._COOLDOWN_SECONDS.get(level, 60.0)

        if cooldown > 0.0:
            now = time.monotonic()
            last = self._last_sent.get(level, 0.0)
            if (now - last) < cooldown:
                log.debug("alert_deduped", level=level, seconds_remaining=cooldown - (now - last))
                return
            self._last_sent[level] = now

        try:
            self._sock.send_json(payload, zmq.NOBLOCK)
        except zmq.Again:
            # Queue full — drop the alert, execution continues unaffected
            log.debug("alert_sender.queue_full — message dropped", type=payload.get("type"))
        except Exception:
            # Never raise from alert path
            pass


# Backwards-compatible public name used by older imports/tests.
TelegramAlertSender = AlertSender
