"""
orchestrator/health_check.py — Prometheus metrics definitions, scrape server,
and HTTP health endpoints. [P1-FIX-34]

All metrics are defined at module level so they can be imported and updated
from any module (orchestrator, order_executor, tick aggregator, etc.) without
circular-import risk.

Health endpoint split [P1-FIX-34]:
  GET /health         → liveness  (200 if process alive; no secrets needed)
  GET /ready          → readiness (200 if dependencies reachable)
  GET /trading-ready  → trading readiness (200 if all gates pass)

Prometheus metrics endpoint: port 9090 (via start_metrics_server).
Health HTTP endpoints: port 8765 (via start_health_server).

Usage:
    from algo_trading.app.health_check import (
        TICKS_PER_SECOND, BAR_LATENCY_MS, INFERENCE_LATENCY_MS,
        QUESTDB_WRITE_ERRORS, ORDERS_PLACED, ORDERS_REJECTED,
        KILL_SWITCH_ACTIVE, start_metrics_server, start_health_server,
        set_readiness, set_trading_ready,
    )

    start_metrics_server()          # call once at startup (port 9090)
    start_health_server()           # call once at startup (port 8765)
    BAR_LATENCY_MS.observe(12.4)    # update from wherever the bar processing runs
    ORDERS_PLACED.labels(symbol="RELIANCE").inc()
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = structlog.get_logger(__name__)

# ── Metric Definitions ────────────────────────────────────────────────────────

TICKS_PER_SECOND = Gauge(
    "algo_ticks_per_second",
    "Live ticks per second",
    ["symbol"],
)

BAR_LATENCY_MS = Histogram(
    "algo_bar_processing_latency_ms",
    "Bar processing latency in ms",
    buckets=[1, 5, 10, 25, 50, 100, 250],
)

INFERENCE_LATENCY_MS = Histogram(
    "algo_inference_latency_ms",
    "Inference latency in ms",
    buckets=[1, 2, 5, 10, 20, 50],
)

QUESTDB_WRITE_ERRORS = Counter(
    "algo_questdb_write_errors_total",
    "QuestDB write errors",
)

ORDERS_PLACED = Counter(
    "algo_orders_placed_total",
    "Orders placed",
    ["symbol"],
)

ORDERS_REJECTED = Counter(
    "algo_orders_rejected_total",
    "Orders rejected",
    ["reason"],
)

KILL_SWITCH_ACTIVE = Gauge(
    "algo_kill_switch_active",
    "Kill switch status: 1=active",
)

# P2-FIX-58: Structured signal skip/rejection counters — diagnose why strategy is inactive
SIGNALS_SKIPPED = Counter(
    "algo_signals_skipped_total",
    "Signal skip/block events by reason",
    ["reason", "symbol"],
)

# ── Server Startup ────────────────────────────────────────────────────────────

_METRICS_PORT = 9090
# P0-FIX-15: Bind to loopback only. Metrics may expose P&L, positions, and
# kill-switch state — never expose on 0.0.0.0. If remote Prometheus scraping
# is required, use an SSH tunnel or authenticated reverse proxy.
_METRICS_ADDR = "127.0.0.1"
_server_started = False

# ── Health HTTP Endpoint (P1-FIX-34) ─────────────────────────────────────────
_HEALTH_PORT = 8765
_HEALTH_ADDR = "127.0.0.1"
_health_server_started = False

# Thread-safe readiness state (set by orchestrator)
_readiness_lock = threading.Lock()
_is_ready: bool = False
_ready_details: str = "not yet initialized"
_is_trading_ready: bool = False
_trading_ready_details: str = "not yet evaluated"


def set_readiness(ready: bool, details: str = "") -> None:
    """Set readiness state (dependencies reachable). Called by orchestrator after startup."""
    global _is_ready, _ready_details
    with _readiness_lock:
        _is_ready = ready
        _ready_details = details or ("dependencies_ok" if ready else "dependencies_unavailable")


def set_trading_ready(ready: bool, details: str = "") -> None:
    """Set trading readiness (all gates passed). Called by orchestrator pre-open checks."""
    global _is_trading_ready, _trading_ready_details
    with _readiness_lock:
        _is_trading_ready = ready
        _trading_ready_details = details or ("all_gates_passed" if ready else "gates_failed")


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for three-tier health checks. [P1-FIX-34]"""

    def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
        pass  # Suppress request logs — would pollute structlog output

    def do_GET(self) -> None:
        if self.path == "/health":
            # Liveness: process is running. Always 200 if this code executes.
            self._respond(200, {"status": "alive"})

        elif self.path == "/ready":
            # Readiness: dependencies (QuestDB, ZMQ) are reachable.
            with _readiness_lock:
                ok = _is_ready
                details = _ready_details
            self._respond(200 if ok else 503, {"status": "ready" if ok else "not_ready", "details": details})

        elif self.path == "/trading-ready":
            # Trading readiness: broker session, model, reconciliation, risk gates.
            with _readiness_lock:
                ok = _is_trading_ready
                details = _trading_ready_details
            self._respond(200 if ok else 503, {"status": "trading_ready" if ok else "not_trading_ready", "details": details})

        else:
            self._respond(404, {"status": "not_found", "path": self.path})

    def _respond(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_health_server(port: int = _HEALTH_PORT, addr: str = _HEALTH_ADDR) -> None:
    """Start the liveness/readiness/trading-readiness HTTP endpoint in a daemon thread.

    Endpoints:
      GET /health         → 200 always (liveness)
      GET /ready          → 200 if set_readiness(True) was called
      GET /trading-ready  → 200 if set_trading_ready(True) was called

    Safe to call multiple times — subsequent calls are no-ops.
    Runs in a daemon thread; exits when the main process exits.
    """
    global _health_server_started
    if _health_server_started:
        return
    server = HTTPServer((addr, port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-http")
    thread.start()
    _health_server_started = True
    log.info("health_check.health_server_started", port=port, addr=addr)


def start_metrics_server(port: int = _METRICS_PORT, addr: str = _METRICS_ADDR) -> None:
    """
    Start the Prometheus HTTP scrape endpoint.

    Safe to call multiple times — subsequent calls are no-ops.
    Call once during orchestrator startup, before market open.

    Args:
        port: TCP port for Prometheus scraping.
        addr: Bind address. Defaults to 127.0.0.1 (loopback only). [P0-FIX-15]
    """
    global _server_started
    if _server_started:
        return
    start_http_server(port, addr=addr)
    _server_started = True
    log.info("health_check.prometheus_server_started", port=port, addr=addr)
