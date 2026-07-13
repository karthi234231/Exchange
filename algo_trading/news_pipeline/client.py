"""
news_pipeline/client.py — Lightweight ZMQ subscriber for the DailyOrchestrator.

The Prefect fusion_flow publishes FusedSignal JSON on port 5572 (ZMQ PUB) each
time it runs (every 5 minutes).  This client opens a ZMQ SUB socket on that port
in a daemon thread and maintains a thread-safe dict of the latest FusedSignal per
symbol.

Why ZMQ instead of reading PostgreSQL directly?
  - Prefect runs in a separate OS process; its in-process signal_cache is invisible
    to the orchestrator.
  - asyncpg (the PGStore client) is async-only and cannot be called from the
    synchronous _on_bar() hot-path without spinning up an event loop.
  - ZMQ PUB/SUB already used throughout the project; adds no new dependency.
  - Latency: a PUB message arrives within microseconds vs. a DB round-trip.

Fail-open contract:
  - If ZMQ is not installed, start() logs a warning and returns immediately.
  - If no signal has been received for a symbol, evaluate_trade() returns
    (True, 1.0, "") — no blocking, no boost, trading proceeds normally.
  - The background thread never raises into the calling thread.

Trade evaluation rules
----------------------
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ News signal     │ BUY (+1) trade          │ SELL/SHORT (-1) trade          │
├─────────────────┼─────────────────────────┼────────────────────────────────┤
│ STRONG_BUY      │ confidence × 1.30       │ BLOCKED                        │
│ BUY             │ confidence × 1.15       │ confidence × 0.70              │
│ NEUTRAL         │ no change               │ no change                      │
│ SELL            │ confidence × 0.70       │ confidence × 1.15              │
│ STRONG_SELL     │ BLOCKED                 │ confidence × 1.30              │
└─────────────────┴─────────────────────────┴────────────────────────────────┘

Confidence is clamped to [0.0, 1.0] after the multiplier is applied.

Usage (DailyOrchestrator):
    self._news_client = NewsPipelineClient()
    # _startup():
    self._news_client.start()
    # _on_bar(), after MTF scaling, before _on_signal():
    allow, multiplier, reason = self._news_client.evaluate_trade(symbol, direction)
    # _shutdown():
    self._news_client.stop()
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ZMQ port on which fusion_flow publishes FusedSignal JSON
_SIGNAL_PUB_PORT: int = 5572


class NewsPipelineClient:
    """
    Thread-safe ZMQ SUB client for receiving FusedSignal updates.

    A single daemon thread maintains a dict of the latest (non-expired)
    FusedSignal per symbol.  ``evaluate_trade()`` is called from the
    ``_on_bar()`` hot-path and must be fast (dict lookup + arithmetic, no I/O).
    """

    _BUY  = 1
    _SELL = -1

    def __init__(self, port: int = _SIGNAL_PUB_PORT) -> None:
        self._port = port
        # symbol → {"signal_type": str, "alpha_score": float, "ttl_expires_at": datetime}
        # Stored as a plain dict (not FusedSignal) to avoid the models import in the
        # hot-path and to keep reconstruction cost at subscriber thread time only.
        self._signals: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._zmq_available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect ZMQ SUB and start background subscriber thread."""
        try:
            import zmq  # noqa: F401
            self._zmq_available = True
        except ImportError:
            log.warning(
                "news_pipeline_client.zmq_unavailable",
                detail="news signals disabled — install pyzmq to enable",
            )
            return

        self._thread = threading.Thread(
            target=self._subscriber_loop,
            name="news-pipeline-client",
            daemon=True,
        )
        self._thread.start()
        log.info("news_pipeline_client.started", port=self._port)

    def stop(self) -> None:
        """Stop the subscriber thread gracefully (max 2 s wait)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("news_pipeline_client.stopped")

    # ------------------------------------------------------------------
    # Background subscriber loop
    # ------------------------------------------------------------------

    def _subscriber_loop(self) -> None:
        """Daemon thread: receive FusedSignal JSON from fusion_flow and cache it."""
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1_000)   # 1 s poll timeout
        sock.setsockopt(zmq.RCVHWM, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")  # no topic filter — receive all
        sock.connect(f"tcp://127.0.0.1:{self._port}")

        log.info("news_pipeline_client.subscriber_connected", port=self._port)

        while not self._stop_event.is_set():
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue  # poll timeout — check stop_event and retry
            except zmq.ZMQError as exc:
                log.error("news_pipeline_client.recv_error", error=str(exc))
                break

            try:
                data = json.loads(raw)
                entry = self._parse_signal(data)
                if entry is not None:
                    with self._lock:
                        self._signals[entry["symbol"]] = entry
                    log.debug(
                        "news_pipeline_client.signal_received",
                        symbol=entry["symbol"],
                        signal_type=entry["signal_type"],
                        alpha=round(entry["alpha_score"], 4),
                    )
            except Exception as exc:
                log.warning("news_pipeline_client.parse_error", error=str(exc))

        sock.close()
        log.info("news_pipeline_client.subscriber_stopped")

    # ------------------------------------------------------------------
    # Parse helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_signal(data: dict) -> dict | None:
        """
        Extract the fields we need from a FusedSignal.to_dict() payload.

        Returns a slim dict with just what evaluate_trade() needs, plus a
        pre-computed UTC expiry timestamp so get_signal() can skip expired entries
        without re-parsing ISO strings on every bar.
        """
        try:
            gen_at_raw = data.get("generated_at", "")
            if isinstance(gen_at_raw, str):
                gen_at = datetime.fromisoformat(gen_at_raw)
            else:
                gen_at = datetime.now(tz=timezone.utc)
            if gen_at.tzinfo is None:
                gen_at = gen_at.replace(tzinfo=timezone.utc)

            from datetime import timedelta
            ttl_minutes = int(data.get("ttl_minutes", 60))
            expires_at = gen_at + timedelta(minutes=ttl_minutes)

            return {
                "symbol":       data["symbol"],
                "signal_type":  data["signal_type"],
                "alpha_score":  float(data["alpha_score"]),
                "expires_at":   expires_at,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Signal access (called from _on_bar hot-path — must be fast)
    # ------------------------------------------------------------------

    def _get_live_signal(self, symbol: str) -> dict | None:
        """
        Return the cached signal entry for *symbol* if it has not expired.

        Expired entries are pruned lazily on access so the dict never grows
        unboundedly (max one entry per symbol, ~50-100 symbols in practice).
        """
        with self._lock:
            entry = self._signals.get(symbol)
            if entry is None:
                return None
            if datetime.now(tz=timezone.utc) >= entry["expires_at"]:
                del self._signals[symbol]
                return None
            return entry

    # ------------------------------------------------------------------
    # Trade evaluation (public API for orchestrator)
    # ------------------------------------------------------------------

    def evaluate_trade(
        self,
        symbol: str,
        direction: int,
    ) -> tuple[bool, float, str]:
        """
        Return ``(allow, confidence_multiplier, reason)`` for a pending trade.

        Parameters
        ----------
        symbol:
            NSE symbol (e.g. "RELIANCE").
        direction:
            +1 for a BUY/long entry, -1 for a SELL/short entry.

        Returns
        -------
        allow:
            False → block the trade entirely (log and return from _on_bar).
        confidence_multiplier:
            Multiply TradeSignal.confidence by this before submission.
            1.0 means no change.  Always in (0, 1.3].
        reason:
            Human-readable string for logging (empty when no news signal).
        """
        entry = self._get_live_signal(symbol)
        if entry is None:
            return True, 1.0, ""

        st    = entry["signal_type"]
        alpha = entry["alpha_score"]

        if st == "STRONG_BUY":
            if direction == self._BUY:
                return True, 1.30, f"STRONG_BUY news confirms BUY (α={alpha:.3f})"
            else:
                return False, 1.0, f"STRONG_BUY news blocks SHORT (α={alpha:.3f})"

        if st == "BUY":
            if direction == self._BUY:
                return True, 1.15, f"BUY news boosts BUY (α={alpha:.3f})"
            else:
                return True, 0.70, f"BUY news reduces SHORT size (α={alpha:.3f})"

        if st == "NEUTRAL":
            return True, 1.0, ""

        if st == "SELL":
            if direction == self._BUY:
                return True, 0.70, f"SELL news reduces BUY size (α={alpha:.3f})"
            else:
                return True, 1.15, f"SELL news boosts SHORT (α={alpha:.3f})"

        if st == "STRONG_SELL":
            if direction == self._BUY:
                return False, 1.0, f"STRONG_SELL news blocks BUY (α={alpha:.3f})"
            else:
                return True, 1.30, f"STRONG_SELL news confirms SHORT (α={alpha:.3f})"

        # Unknown signal_type — fail open
        return True, 1.0, ""
