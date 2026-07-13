"""
alerts/alert_sink.py — Dedicated ZMQ PULL → Telegram sink process. [ARCH-FIX-3]

This module runs as a SEPARATE OS process (spawned by main.py).
It is the ONLY code in the system that calls the Telegram API.

If Telegram is slow (200-2000ms) or dead — only alerts die. Trading continues.
If this process crashes — main.py respawns it. Trading continues.

Rate limiting: Telegram allows ~30 messages/minute per bot to a chat.
We batch alerts if they arrive faster than _MIN_SEND_INTERVAL_SEC.

Design:
  - ZMQ PULL on port 5560 (bound here, connected from AlertSender)
  - Telegram via python-telegram-bot sync API (telegram.Bot.send_message)
  - Exponential backoff on Telegram failure (don't spam on network issues)
  - Runs in a tight recv loop — exits cleanly on SIGTERM/SIGINT

Run as subprocess (from main.py):
    import multiprocessing
    p = multiprocessing.Process(target=run_alert_sink, daemon=True)
    p.start()
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import signal
import time
from typing import Optional

import requests
import zmq

log = logging.getLogger(__name__)

_MIN_SEND_INTERVAL_SEC = 2.0    # Don't send faster than 1 per 2 seconds
_MAX_BACKOFF_SEC = 60.0         # Max retry wait on Telegram failure
_RECV_TIMEOUT_MS = 1000         # Poll timeout — allows clean shutdown

# Durable audit log: CRITICAL/RISK events are written here FIRST (before
# Telegram delivery) so they survive even if the alert channel is unavailable.
# P1-FIX-21: audit log is the primary durable record; Telegram is secondary.
_AUDIT_LOG_PATH = pathlib.Path("state/alert_audit.log")

# Fallback log written when Telegram is unavailable or a send fails permanently
_FALLBACK_LOG_PATH = pathlib.Path("state/alert_fallback.log")

# Alert levels that must be durably written before attempting Telegram delivery
_AUDIT_LEVELS = frozenset({"CRITICAL", "RISK", "ERROR"})

_STOP_REQUESTED = False


def _priority(level: str) -> int:
    """Return sort priority for a given alert level (lower = higher priority)."""
    return {"RISK": 0, "CRITICAL": 0, "WARN": 1, "WARNING": 1, "SESSION": 2}.get(
        level.upper(), 3
    )


def _write_fallback(payload: dict) -> None:
    """Append a failed/undeliverable alert to the local fallback log."""
    try:
        _FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps({"ts": ts, **payload}) + "\n"
        with _FALLBACK_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        log.error("alert_sink.fallback_write_failed: %s", exc)


def _write_audit(payload: dict) -> None:
    """P1-FIX-21: Write CRITICAL/RISK/ERROR events to durable audit log FIRST.

    This is the primary record. Telegram delivery is secondary.
    Even if Telegram is down, these events are preserved locally.
    """
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps({"ts": ts, "audit": True, **payload}) + "\n"
        with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
    except Exception as exc:
        log.error("alert_sink.audit_write_failed: %s", exc)


def _handle_signal(signum: int, frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def run_alert_sink() -> None:
    """
    Entry point for the alert sink process.
    Blocks until SIGTERM/SIGINT received.
    """
    global _STOP_REQUESTED

    # Set up logging for this subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ALERT-SINK] %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Fix C — use settings for port (single shared constant) rather than a
    # hard-coded fallback.  The env var is kept as an override escape hatch.
    from algo_trading.config import settings as _settings
    port = int(os.environ.get("ZMQ_ALERT_PORT", str(_settings.zmq_alert_push_port)))

    if not bot_token or not chat_id:
        log.warning("alert_sink: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — alerts disabled")
        # Still receive from ZMQ (drain the queue) to prevent PUSH backpressure
        _drain_only_mode(port)
        return

    # Fix C — validate the Telegram token before binding the ZMQ socket so we
    # know immediately whether Telegram is reachable.
    _token_valid = _validate_telegram_token(bot_token)

    _sender = _TelegramSender(bot_token, chat_id)
    _run_loop(port, _sender, token_valid=_token_valid)


def _validate_telegram_token(bot_token: str) -> bool:
    """Call getMe to confirm the token is valid. Returns True if OK."""
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if resp.ok and data.get("ok"):
            log.info("alert_sink.telegram_token_valid: bot=%s", data.get("result", {}).get("username"))
            return True
        log.critical(
            "alert_sink.telegram_token_invalid: ok=%s description=%s — "
            "alerts will be written to %s",
            data.get("ok"),
            data.get("description", "unknown"),
            _FALLBACK_LOG_PATH,
        )
        return False
    except Exception as exc:
        log.critical(
            "alert_sink.telegram_token_check_failed: %s — alerts will be written to %s",
            exc,
            _FALLBACK_LOG_PATH,
        )
        return False


def _run_loop(port: int, sender: "_TelegramSender", *, token_valid: bool = True) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, 500)
    # We manage blocking ourselves; set socket to non-blocking for the drain phase.
    sock.setsockopt(zmq.RCVTIMEO, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")
    log.info("alert_sink.started on port %d (token_valid=%s)", port, token_valid)

    last_send_time = 0.0

    while not _STOP_REQUESTED:
        # Fix D — drain all available messages into a list, then sort by priority.
        pending: list[dict] = []
        while True:
            try:
                payload = sock.recv_json(zmq.NOBLOCK)
                pending.append(payload)
            except zmq.Again:
                break  # No more messages currently queued
            except zmq.ZMQError as exc:
                log.error("alert_sink.recv_error: %s", exc)
                break

        if not pending:
            # Nothing available — block for up to 100 ms then retry.
            time.sleep(0.1)
            continue

        # Sort by priority: RISK/CRITICAL=0, WARN=1, SESSION=2, INFO=3
        pending.sort(key=lambda p: _priority(p.get("level", "INFO")))

        for payload in pending:
            try:
                msg = payload.get("msg", "")
                level = payload.get("level", "INFO").upper()

                # P1-FIX-21: Write CRITICAL/RISK/ERROR to durable audit log FIRST.
                # This ensures the event is recorded even if Telegram delivery fails.
                if level in _AUDIT_LEVELS:
                    _write_audit(payload)

                # Fix C — if token is invalid, write to fallback log instead of Telegram
                if not token_valid:
                    if level not in _AUDIT_LEVELS:
                        _write_fallback(payload)  # audit already written above
                    continue

                # Rate limit: don't send faster than _MIN_SEND_INTERVAL_SEC
                elapsed = time.monotonic() - last_send_time
                if elapsed < _MIN_SEND_INTERVAL_SEC:
                    time.sleep(_MIN_SEND_INTERVAL_SEC - elapsed)

                formatted = f"[{level}] {msg}"
                sent_ok = sender.send(formatted)
                last_send_time = time.monotonic()

                # Fix E — write to fallback log if Telegram delivery ultimately failed.
                # Audit-level events are already in the audit log; write fallback for
                # non-audit events or to record that Telegram delivery failed.
                if not sent_ok:
                    _write_fallback(payload)

            except Exception as exc:
                log.error("alert_sink.process_error: %s", exc)

    sock.close()
    ctx.term()
    log.info("alert_sink.stopped")


def _drain_only_mode(port: int) -> None:
    """Receive and log alerts but don't send to Telegram."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, 500)
    sock.setsockopt(zmq.RCVTIMEO, _RECV_TIMEOUT_MS)
    sock.bind(f"tcp://127.0.0.1:{port}")
    log.info("alert_sink.drain_only_mode (no Telegram credentials)")

    while not _STOP_REQUESTED:
        try:
            payload = sock.recv_json()
            level = payload.get("level", "INFO").upper()
            # P1-FIX-21: Even in drain-only mode, persist critical events to audit log
            if level in _AUDIT_LEVELS:
                _write_audit(payload)
            log.info("alert_sink.dropped: %s", payload.get("msg", ""))
        except zmq.Again:
            continue
        except Exception:
            break

    sock.close()
    ctx.term()


class _TelegramSender:
    """Sends messages to Telegram with exponential backoff on failure."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._backoff = 1.0
        self._bot: Optional[object] = None
        self._init_bot()

    def _init_bot(self) -> None:
        try:
            import telegram  # type: ignore[import]
            self._bot = telegram.Bot(token=self._token)
            log.info("alert_sink.telegram_bot_initialized")
        except ImportError:
            log.warning("alert_sink: python-telegram-bot not installed — using requests fallback")
            self._bot = None

    def send(self, text: str) -> bool:
        """Send message to Telegram. Returns True on success, False on failure."""
        if self._bot is not None:
            return self._send_via_bot(text)
        else:
            return self._send_via_requests(text)

    def _send_via_bot(self, text: str) -> bool:
        try:
            import asyncio
            import telegram  # type: ignore[import]
            bot: telegram.Bot = self._bot  # type: ignore[assignment]
            # python-telegram-bot v20+ is async — run in new event loop
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    bot.send_message(
                        chat_id=self._chat_id,
                        text=text[:4096],  # Telegram message limit
                        parse_mode=None,
                    )
                )
            finally:
                loop.close()
            self._backoff = 1.0  # Reset on success
            log.debug("alert_sink.sent: %.60s", text)
            return True
        except Exception as exc:
            log.warning("alert_sink.send_failed: %s — backoff %.1fs", exc, self._backoff)
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SEC)
            return False

    def _send_via_requests(self, text: str) -> bool:
        """Fallback: send via raw HTTP if python-telegram-bot not installed."""
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text[:4096]},
                timeout=10,
            )
            resp.raise_for_status()
            self._backoff = 1.0
            return True
        except Exception as exc:
            log.warning("alert_sink.requests_send_failed: %s — backoff %.1fs", exc, self._backoff)
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SEC)
            return False


if __name__ == "__main__":
    # Can also run as: python -m algo_trading.alerts.alert_sink
    run_alert_sink()
