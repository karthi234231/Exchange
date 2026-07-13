"""
health_check.py — Lightweight HTTP health check for the trading system.

Exposes GET /health returning JSON with system vitals.
Run as a subprocess alongside main.py:
  python health_check.py --port 8080

Response schema:
{
  "status": "ok" | "degraded" | "critical",
  "timestamp_ist": "2026-06-18T09:30:00+05:30",
  "checks": {
    "questdb": {"status": "ok", "latency_ms": 2.1},
    "last_tick_age_sec": 8,
    "zmq_heartbeat": {"status": "ok"},
    "daily_pnl_inr": -1250.0,
    "risk_status": "ok",
    "positions_count": 2,
    "kill_switch": false
  }
}

Status logic:
  - "ok": all checks pass
  - "degraded": any check warning (last tick age > 30s, QuestDB latency > 100ms)
  - "critical": kill switch active, QuestDB down, last tick age > 120s

Usage:
  --once: print JSON and exit (for monitoring scripts / make health)
  --port PORT: run HTTP server (default 8080)
  --state-file PATH: path to live_state.json (default state/live_state.json)
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any

# ── IST timezone (no external deps) ─────────────────────────────────────────

_IST_OFFSET = timedelta(hours=5, minutes=30)
_IST = timezone(_IST_OFFSET, name="IST")


def _now_ist() -> datetime:
    return datetime.now(tz=_IST)


def _ist_isoformat() -> str:
    return _now_ist().isoformat(timespec="seconds")


# ── Thresholds ────────────────────────────────────────────────────────────────

_TICK_AGE_WARN_SEC: float = 30.0
_TICK_AGE_CRITICAL_SEC: float = 120.0
_QUESTDB_LATENCY_WARN_MS: float = 100.0
_ZMQ_HEARTBEAT_STALE_SEC: float = 30.0

# QuestDB connection details (used for TCP probe only)
_QUESTDB_HOST: str = os.environ.get("QUESTDB_HOST", "localhost")
_QUESTDB_PG_PORT: int = int(os.environ.get("QUESTDB_PG_PORT", "8812"))
_TCP_PROBE_TIMEOUT_SEC: float = 2.0

# ZMQ heartbeat file written by main process every 10 seconds
_ZMQ_HEARTBEAT_FILE: str = os.environ.get(
    "ZMQ_HEARTBEAT_FILE", "state/zmq_heartbeat"
)


# ── Individual probe functions ────────────────────────────────────────────────

def _probe_questdb() -> dict[str, Any]:
    """
    TCP socket check against QuestDB PostgreSQL port (8812).
    Does NOT require psycopg2 — pure stdlib socket connect.

    Returns:
        {"status": "ok"|"error", "latency_ms": float}
    """
    start = time.monotonic()
    try:
        with socket.create_connection(
            (_QUESTDB_HOST, _QUESTDB_PG_PORT),
            timeout=_TCP_PROBE_TIMEOUT_SEC,
        ):
            latency_ms = (time.monotonic() - start) * 1000.0
        status = "ok"
        if latency_ms > _QUESTDB_LATENCY_WARN_MS:
            status = "slow"
        return {"status": status, "latency_ms": round(latency_ms, 2)}
    except OSError as exc:
        return {"status": "error", "error": str(exc), "latency_ms": -1}


def _probe_zmq_heartbeat(heartbeat_path: str) -> dict[str, Any]:
    """
    File-based ZMQ heartbeat check.
    main.py must touch the heartbeat file every ~10 seconds.
    If file mtime is older than _ZMQ_HEARTBEAT_STALE_SEC seconds → stale.

    Returns:
        {"status": "ok"|"stale"|"missing"}
    """
    hb = Path(heartbeat_path)
    if not hb.exists():
        return {"status": "missing"}
    try:
        mtime = hb.stat().st_mtime
        age_sec = time.time() - mtime
        if age_sec > _ZMQ_HEARTBEAT_STALE_SEC:
            return {"status": "stale", "age_sec": round(age_sec, 1)}
        return {"status": "ok", "age_sec": round(age_sec, 1)}
    except OSError as exc:
        return {"status": "error", "error": str(exc)}


def _read_state(state_file: str) -> dict[str, Any]:
    """
    Read live_state.json written by main.py.
    Returns empty dict if file is missing or malformed.
    """
    p = Path(state_file)
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return {}


# ── Main health computation ───────────────────────────────────────────────────

def compute_health(state_file: str) -> dict[str, Any]:
    """
    Run all checks and return the full health payload dict.

    Status logic:
      - "critical" : kill_switch active, QuestDB down, or tick_age > 120s
      - "degraded"  : QuestDB slow (>100ms), tick_age > 30s, ZMQ stale
      - "ok"        : all checks pass
    """
    state = _read_state(state_file)

    # ── QuestDB probe ─────────────────────────────────────────────────────────
    questdb_check = _probe_questdb()

    # ── ZMQ heartbeat ─────────────────────────────────────────────────────────
    zmq_check = _probe_zmq_heartbeat(_ZMQ_HEARTBEAT_FILE)

    # ── Tick age ──────────────────────────────────────────────────────────────
    last_tick_ts = state.get("last_tick_timestamp")
    last_tick_age_sec: float
    if last_tick_ts is None:
        last_tick_age_sec = -1.0  # Unknown — treat as degraded
    else:
        try:
            # Expect Unix timestamp (float) or ISO string
            if isinstance(last_tick_ts, (int, float)):
                last_tick_age_sec = time.time() - float(last_tick_ts)
            else:
                dt = datetime.fromisoformat(str(last_tick_ts))
                last_tick_age_sec = (
                    datetime.now(tz=timezone.utc) - dt.astimezone(timezone.utc)
                ).total_seconds()
        except Exception:
            last_tick_age_sec = -1.0

    # ── Values from state ─────────────────────────────────────────────────────
    daily_pnl_inr: float = float(state.get("daily_pnl", 0.0))
    positions_count: int = int(state.get("positions_count", 0))
    kill_switch: bool = bool(state.get("kill_switch", False))
    risk_status: str = state.get("risk_status", "ok")

    # ── Aggregate status ──────────────────────────────────────────────────────
    is_critical = (
        kill_switch
        or questdb_check["status"] == "error"
        or (last_tick_age_sec > _TICK_AGE_CRITICAL_SEC and last_tick_age_sec >= 0)
    )

    is_degraded = (
        not is_critical
        and (
            questdb_check["status"] == "slow"
            or zmq_check["status"] in ("stale", "missing")
            or (last_tick_age_sec > _TICK_AGE_WARN_SEC and last_tick_age_sec >= 0)
            or last_tick_age_sec < 0  # Unknown tick age
            or risk_status not in ("ok", "")
        )
    )

    if is_critical:
        overall_status = "critical"
    elif is_degraded:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "timestamp_ist": _ist_isoformat(),
        "checks": {
            "questdb": questdb_check,
            "last_tick_age_sec": round(last_tick_age_sec, 1),
            "zmq_heartbeat": zmq_check,
            "daily_pnl_inr": round(daily_pnl_inr, 2),
            "risk_status": risk_status,
            "positions_count": positions_count,
            "kill_switch": kill_switch,
        },
    }


# ── HTTP server ───────────────────────────────────────────────────────────────

def _build_handler(state_file: str):
    """Factory: returns an HTTP request handler class closed over state_file."""

    class _HealthHandler(http.server.BaseHTTPRequestHandler):
        """Minimal HTTP handler — serves only GET /health."""

        # Suppress default request logging to stdout (we handle it ourselves)
        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/health", "/health/"):
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return

            payload = compute_health(state_file)
            body = json.dumps(payload, indent=2).encode("utf-8")

            status_code = {
                "ok": HTTPStatus.OK,
                "degraded": HTTPStatus.OK,        # 200 — let monitoring decide
                "critical": HTTPStatus.SERVICE_UNAVAILABLE,
            }.get(payload["status"], HTTPStatus.OK)

            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _HealthHandler


class _ReusableServer(http.server.HTTPServer):
    """HTTPServer with SO_REUSEADDR set before bind."""
    allow_reuse_address = True


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE AlgoTrader health check endpoint",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--state-file",
        default="state/live_state.json",
        metavar="PATH",
        help="Path to live_state.json (default: state/live_state.json)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Print JSON health payload to stdout and exit.\n"
            "Exit codes: 0=ok, 1=degraded, 2=critical"
        ),
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        metavar="HOST",
        help="Interface to bind to (default: 127.0.0.1; use 0.0.0.0 to expose on all interfaces)",
    )
    return parser.parse_args(argv)


def _exit_code(status: str) -> int:
    return {"ok": 0, "degraded": 1, "critical": 2}.get(status, 1)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.once:
        payload = compute_health(args.state_file)
        print(json.dumps(payload, indent=2))
        sys.exit(_exit_code(payload["status"]))

    handler_class = _build_handler(args.state_file)

    server = _ReusableServer((args.bind, args.port), handler_class)

    # ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────────────
    def _shutdown(signum: int, frame: object) -> None:
        print(f"\nhealth_check: received signal {signum}, shutting down...", flush=True)
        # server.shutdown() must be called from a different thread than serve_forever(),
        # but here we call it in the signal handler which runs in the main thread
        # while serve_forever() also runs in the main thread. Use a thread:
        import threading
        t = threading.Thread(target=server.shutdown, daemon=True)
        t.start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"health_check: listening on http://{args.bind}:{args.port}/health "
        f"(state={args.state_file!r})",
        flush=True,
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()
        print("health_check: stopped.", flush=True)


if __name__ == "__main__":
    main()
