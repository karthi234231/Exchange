"""
Paper trading runner for NSE AlgoTrader.

Runs the full live pipeline in simulation mode — no real money at risk.
Paper fills are simulated at the signal bar's close price.

Usage:
    python -m algo_trading.scripts.paper_trade
    python -m algo_trading.scripts.paper_trade --symbols RELIANCE TCS --capital 500000
    python -m algo_trading.scripts.paper_trade --duration-days 30

Paper trade log is written to: logs/paper_trades_{date}.jsonl
Summary written to: logs/paper_summary_{date}.json
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

# ── FillSimulator: deferred import ────────────────────────────────────────────
try:
    from algo_trading.research.fill_simulator import FillMode, FillSimulator  # noqa: PLC0415
    _FILL_SIMULATOR_AVAILABLE = True
except ImportError:
    _FILL_SIMULATOR_AVAILABLE = False
    FillMode = None  # type: ignore[assignment,misc]
    FillSimulator = None  # type: ignore[assignment,misc]

# ── ZMQ: deferred import ───────────────────────────────────────────────────────
try:
    import zmq  # type: ignore[import]
    _ZMQ_AVAILABLE = True
except ImportError:
    _ZMQ_AVAILABLE = False

# ── IST timezone ───────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

# Default Nifty 50 top-10 symbols
_DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BAJFINANCE", "KOTAKBANK", "LT",
]

# NSE session boundaries (HH:MM)
_SESSION_OPEN = "09:15"
_SESSION_CLOSE = "15:30"
_FORCE_FLAT = "15:15"
_DAILY_SUMMARY_TIME = "15:30"

# ZMQ ports matching config defaults
_PUB_PORT_DATA = 5555       # DataIngestor PUB
_PUB_PORT_FEATURES = 5556   # FeatureComputer PUB
_PULL_PORT_SIGNALS = 5557   # MLInference PUSH → paper trader PULL

# Signal int → direction string (matches inference_engine._serialise_signal wire format)
_SIG_TO_DIR: dict[int, str] = {1: "LONG", -1: "SHORT", 0: "FLAT"}


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PaperFill:
    """Represents a simulated order fill in paper trading mode."""

    symbol: str
    direction: str          # "LONG" or "SHORT"
    qty: int
    fill_price: float
    timestamp: str          # ISO-8601 string
    order_id: str
    signal_confidence: float  # [0.0, 1.0] from ML inference


@dataclass
class _OpenPosition:
    """Tracks an open paper position."""

    fill: PaperFill
    cost_inr: float


@dataclass
class _ClosedTrade:
    """Represents a completed round-trip paper trade."""

    symbol: str
    direction: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    gross_pnl_inr: float
    costs_inr: float
    pnl_inr: float           # net P&L after costs
    order_id: str
    signal_confidence: float


# ══════════════════════════════════════════════════════════════════════════════
# PaperTradeTracker
# ══════════════════════════════════════════════════════════════════════════════

class PaperTradeTracker:
    """
    Tracks open and closed paper positions; computes running metrics.

    Thread-safety note: this class is single-threaded (main event loop only).
    All monetary calculations use plain float — paper mode only, no real risk.
    """

    def __init__(self, capital_inr: float) -> None:
        self._capital = capital_inr
        self._open: dict[str, _OpenPosition] = {}    # symbol → open position
        self._closed: list[_ClosedTrade] = []
        self._peak_equity = capital_inr
        self._equity = capital_inr

        # Cost model for realistic P&L simulation
        try:
            from algo_trading.risk.cost_model import IndiaCostModel  # noqa: PLC0415
            from decimal import Decimal  # noqa: PLC0415
            self._cost_model: object = IndiaCostModel()
            self._Decimal = Decimal
        except ImportError:
            self._cost_model = None
            self._Decimal = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def open_position(self, fill: PaperFill) -> None:
        """Record a new simulated fill as an open position."""
        if fill.symbol in self._open:
            log.warning(
                "paper_tracker.already_open",
                symbol=fill.symbol,
                action="replacing_existing",
            )

        cost_inr = fill.fill_price * fill.qty
        self._open[fill.symbol] = _OpenPosition(fill=fill, cost_inr=cost_inr)

        log.info(
            "paper_tracker.position_opened",
            symbol=fill.symbol,
            direction=fill.direction,
            qty=fill.qty,
            fill_price=fill.fill_price,
            cost_inr=round(cost_inr, 2),
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: str,
    ) -> Optional[dict]:
        """
        Close an open paper position and compute P&L.

        Returns:
            dict with pnl_inr, entry_price, exit_price, direction, qty
            or None if no open position exists for symbol.
        """
        if symbol not in self._open:
            log.warning("paper_tracker.no_open_position", symbol=symbol)
            return None

        pos = self._open.pop(symbol)
        fill = pos.fill

        if fill.direction == "LONG":
            gross_pnl = (exit_price - fill.fill_price) * fill.qty
        else:  # SHORT
            gross_pnl = (fill.fill_price - exit_price) * fill.qty

        # Apply realistic round-trip costs via IndiaCostModel
        costs_inr = 0.0
        if self._cost_model is not None and self._Decimal is not None:
            try:
                D = self._Decimal
                direction_int = 1 if fill.direction == "LONG" else -1
                cost = self._cost_model.compute_round_trip(  # type: ignore[attr-defined]
                    symbol=symbol,
                    quantity=fill.qty,
                    entry_price=D(str(fill.fill_price)),
                    exit_price=D(str(exit_price)),
                    direction=direction_int,
                )
                costs_inr = float(cost.total_inr)
            except Exception as exc:
                log.warning("paper_tracker.cost_model_error", symbol=symbol, error=str(exc))

        pnl_inr = gross_pnl - costs_inr

        trade = _ClosedTrade(
            symbol=symbol,
            direction=fill.direction,
            qty=fill.qty,
            entry_price=fill.fill_price,
            exit_price=exit_price,
            entry_time=fill.timestamp,
            exit_time=timestamp,
            gross_pnl_inr=round(gross_pnl, 2),
            costs_inr=round(costs_inr, 2),
            pnl_inr=round(pnl_inr, 2),
            order_id=fill.order_id,
            signal_confidence=fill.signal_confidence,
        )
        self._closed.append(trade)

        # Update running equity
        self._equity += pnl_inr
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        log.info(
            "paper_tracker.position_closed",
            symbol=symbol,
            pnl_inr=round(pnl_inr, 2),
            equity=round(self._equity, 2),
        )

        return {
            "pnl_inr": round(pnl_inr, 2),
            "gross_pnl_inr": round(gross_pnl, 2),
            "costs_inr": round(costs_inr, 2),
            "entry_price": fill.fill_price,
            "exit_price": exit_price,
            "direction": fill.direction,
            "qty": fill.qty,
        }

    def daily_summary(self) -> dict:
        """
        Compute and return daily performance summary.

        Returns:
            dict with keys: pnl_inr, win_rate, n_trades, sharpe_rolling,
            max_drawdown, equity, open_positions
        """
        n_trades = len(self._closed)
        total_pnl = sum(t.pnl_inr for t in self._closed)
        winners = [t for t in self._closed if t.pnl_inr > 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0.0

        # Rolling Sharpe from all closed trade PnLs
        sharpe_rolling = self._compute_sharpe([t.pnl_inr for t in self._closed])

        # Max drawdown
        max_drawdown = self._compute_max_drawdown()

        return {
            "date": _ist_now().date().isoformat(),
            "pnl_inr": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "n_trades": n_trades,
            "sharpe_rolling": round(sharpe_rolling, 4),
            "max_drawdown": round(max_drawdown, 4),
            "equity": round(self._equity, 2),
            "open_positions": list(self._open.keys()),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(pnl_series: list[float]) -> float:
        """Compute Sharpe ratio (annualised proxy) from a P&L series."""
        if len(pnl_series) < 2:
            return 0.0
        mean_pnl = sum(pnl_series) / len(pnl_series)
        variance = sum((x - mean_pnl) ** 2 for x in pnl_series) / (len(pnl_series) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0
        if std_pnl < 1e-8:
            return 0.0
        # Scale by sqrt(252) — annualised
        return (mean_pnl / std_pnl) * math.sqrt(252)

    def _compute_max_drawdown(self) -> float:
        """Compute max drawdown fraction from peak equity."""
        if self._peak_equity <= 0:
            return 0.0
        drawdown = (self._peak_equity - self._equity) / self._peak_equity
        return max(0.0, drawdown)


# ══════════════════════════════════════════════════════════════════════════════
# JSONL logger
# ══════════════════════════════════════════════════════════════════════════════

class _PaperTradeLogger:
    """Writes paper fills and events to a JSONL file."""

    def __init__(self, log_dir: Path, session_date: date) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_path = log_dir / f"paper_trades_{session_date.isoformat()}.jsonl"
        self._summary_path = log_dir / f"paper_summary_{session_date.isoformat()}.json"
        self._fh = self._trade_path.open("a", encoding="utf-8")
        log.info("paper_logger.opened", path=str(self._trade_path))

    def log_fill(self, fill: PaperFill, event: str = "open") -> None:
        record = {"event": event, **asdict(fill)}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_close(self, symbol: str, exit_price: float, pnl_dict: dict) -> None:
        record = {
            "event": "close",
            "symbol": symbol,
            "exit_price": exit_price,
            "timestamp": _ist_now_str(),
            **pnl_dict,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def write_summary(self, summary: dict) -> None:
        self._summary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        log.info("paper_logger.summary_written", path=str(self._summary_path))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ist_now() -> datetime:
    """Return timezone-aware datetime representing current IST wall-clock time."""
    return datetime.now(tz=IST)


def _ist_now_str() -> str:
    return _ist_now().isoformat(timespec="seconds")


def _hhmm_to_time(hhmm: str):
    """Parse 'HH:MM' string to datetime.time."""
    from datetime import time
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def _should_force_flat(ist_now: datetime) -> bool:
    """Return True if we are at or past 15:15 IST."""
    flat_time = _hhmm_to_time(_FORCE_FLAT)
    return ist_now.time() >= flat_time


def _is_daily_summary_time(ist_now: datetime) -> bool:
    """Return True if wall clock has crossed 15:30 IST (summary window)."""
    summary_time = _hhmm_to_time(_DAILY_SUMMARY_TIME)
    return ist_now.time() >= summary_time


def _simulate_fill(
    symbol: str,
    signal: dict,
    capital: float,
    max_position_pct: float = 0.05,
    fill_mode: str = "REALISTIC",
) -> PaperFill:
    """
    Simulate an order fill at the signal bar's close price.

    When FillSimulator is available, uses it with the requested fill_mode.
    For paper mode without bid/ask, a synthetic spread is estimated from
    ATR (if present in signal) or a fixed 5bps approximation.
    Falls back to naive signal_price when FillSimulator import failed.
    """
    import uuid  # noqa: PLC0415

    base_price = float(signal.get("signal_price", 0.0))
    direction = str(signal.get("direction", "LONG"))
    confidence = float(signal.get("confidence", 0.5))

    # Compute qty from capital allocation
    position_budget = capital * max_position_pct
    qty = max(1, int(position_budget // base_price)) if base_price > 0 else 1

    fill_price = base_price

    if _FILL_SIMULATOR_AVAILABLE and base_price > 0 and fill_mode != "NAIVE":
        try:
            # Map CLI fill_mode string to FillMode enum
            try:
                mode_enum = FillMode(fill_mode)
            except (ValueError, KeyError):
                mode_enum = FillMode.REALISTIC

            # Estimate spread from ATR when bid/ask are unavailable.
            # ATR is in price units; 5bps of base_price is the fixed fallback.
            atr = float(signal.get("atr", signal.get("feat_atr", 0.0)))
            if atr > 0:
                estimated_spread = atr * 0.10  # 10% of ATR ≈ typical bid-ask spread
            else:
                estimated_spread = base_price * 0.0005  # fixed 5bps fallback

            half_spread = estimated_spread / 2.0
            synthetic_bid = base_price - half_spread
            synthetic_ask = base_price + half_spread

            # Build a minimal quote dict that FillSimulator._get_field() understands
            synthetic_quote = {
                "bid": synthetic_bid,
                "ask": synthetic_ask,
                "ltp": base_price,
                "oi": 0,
            }

            side = "BUY" if direction == "LONG" else "SELL"
            order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

            sim = FillSimulator(mode=mode_enum)
            result = sim.simulate_fill(
                order_id=order_id,
                symbol=symbol,
                expiry=None,
                strike=base_price,
                option_type="CE",  # equity — option_type is unused by REALISTIC/PESSIMISTIC
                side=side,
                qty_lots=qty,
                quote=synthetic_quote,
                event_time=_ist_now(),
            )

            if result is not None and not result.cancelled and result.fill_price > 0:
                fill_price = result.fill_price
                log.debug(
                    "paper_fill.simulator_used",
                    symbol=symbol,
                    fill_mode=fill_mode,
                    base_price=base_price,
                    fill_price=round(fill_price, 4),
                    slippage_bps=round(result.slippage_bps, 2),
                    notes=result.notes,
                )
                return PaperFill(
                    symbol=symbol,
                    direction=direction,
                    qty=qty,
                    fill_price=fill_price,
                    timestamp=_ist_now_str(),
                    order_id=result.order_id,
                    signal_confidence=confidence,
                )
        except Exception as exc:
            log.warning(
                "paper_fill.simulator_error",
                symbol=symbol,
                fill_mode=fill_mode,
                error=str(exc),
                fallback="naive_price",
            )

    # Naive fallback: fill at signal_price
    return PaperFill(
        symbol=symbol,
        direction=direction,
        qty=qty,
        fill_price=fill_price,
        timestamp=_ist_now_str(),
        order_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
        signal_confidence=confidence,
    )


def _decode_signal(raw: bytes) -> dict:
    """Decode binary signal frame from inference_engine._serialise_signal.

    Wire format: <H(sym_len) | sym_bytes | <b(signal_int) | <f(confidence)
    signal_int: 1=LONG, -1=SHORT, 0=FLAT

    Returns:
        dict with keys: symbol, signal, confidence, signal_price, direction
    Raises:
        ValueError: if payload is too short or malformed
        struct.error: if struct.unpack fails
    """
    if len(raw) < 2:
        raise ValueError(f"Signal payload too short: {len(raw)} bytes")
    sym_len = struct.unpack_from("<H", raw, 0)[0]
    offset = 2
    if len(raw) < offset + sym_len + 1 + 4:
        raise ValueError(f"Signal payload truncated: expected {offset + sym_len + 5}, got {len(raw)}")
    sym = raw[offset:offset + sym_len].decode("utf-8")
    offset += sym_len
    signal_int = struct.unpack_from("<b", raw, offset)[0]
    offset += 1
    confidence = struct.unpack_from("<f", raw, offset)[0]
    return {
        "symbol": sym,
        "signal": signal_int,
        "confidence": float(confidence),
        "signal_price": 0.0,
        "direction": _SIG_TO_DIR.get(signal_int, "FLAT"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ZMQ subprocess launchers
# ══════════════════════════════════════════════════════════════════════════════

def _launch_zmq_processes(symbols: list[str]) -> list[subprocess.Popen]:
    """
    Launch DataIngestor, FeatureComputer, and MLInference as subprocesses.

    In paper mode the processes run exactly as in live mode — only order
    placement is stubbed out. Returns a list of Popen handles.
    """
    procs: list[subprocess.Popen] = []

    symbol_csv = ",".join(symbols)
    common_env = {
        "TRADING_MODE": "paper",
        "ENABLE_LIVE_TRADING": "false",
        "SYMBOLS": symbol_csv,
    }

    import os
    env = {**os.environ, **common_env}

    # DataIngestor — PUB on 5555
    procs.append(
        subprocess.Popen(
            [sys.executable, "-m", "algo_trading.data.websocket_feed"],
            env=env,
        )
    )

    # FeatureComputer — SUB:5555 / PUB:5556
    procs.append(
        subprocess.Popen(
            [sys.executable, "-m", "algo_trading.features.engine"],
            env=env,
        )
    )

    # MLInference — SUB:5556 / PUSH:5557
    procs.append(
        subprocess.Popen(
            [sys.executable, "-m", "algo_trading.models.inference_engine"],
            env=env,
        )
    )

    log.info(
        "paper_trade.processes_launched",
        n=len(procs),
        pids=[p.pid for p in procs],
    )
    return procs


def _terminate_zmq_processes(procs: list[subprocess.Popen]) -> None:
    """Gracefully terminate all ZMQ subprocess handles."""
    for proc in procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log.warning("paper_trade.proc_terminate_error", pid=proc.pid, error=str(exc))
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    log.info("paper_trade.processes_terminated", n=len(procs))


# ══════════════════════════════════════════════════════════════════════════════
# Main session loop
# ══════════════════════════════════════════════════════════════════════════════

def run_paper_session(
    symbols: list[str],
    capital: float,
    duration_days: int,
    log_dir: Path,
    fill_mode: str = "REALISTIC",
) -> None:
    """
    Main paper trading session loop.

    Steps:
    1. StartupReconciler check (paper mode: always OK — no open positions to reconcile).
    2. Launch ZMQ pipeline subprocesses.
    3. Read signals from ZMQ PULL:5557.
    4. Simulate fills at signal_price.
    5. Log to JSONL.
    6. Force-flat at 15:15 IST.
    7. Print and save daily summary at 15:30 IST.

    Args:
        symbols:       List of NSE equity symbols to trade.
        capital:       Simulated capital in INR.
        duration_days: Max number of calendar days to run (wall-clock).
        log_dir:       Directory for JSONL + JSON summary files.
        fill_mode:     Fill simulation mode ("NAIVE", "REALISTIC", "PESSIMISTIC").
    """
    if not _ZMQ_AVAILABLE:
        log.error(
            "paper_trade.zmq_not_installed",
            msg="pyzmq not found. Install with: pip install pyzmq",
        )
        raise SystemExit("pip install pyzmq")

    session_date = _ist_now().date()
    tracker = PaperTradeTracker(capital_inr=capital)
    paper_logger = _PaperTradeLogger(log_dir=log_dir, session_date=session_date)
    end_date = session_date + timedelta(days=duration_days)

    procs: list[subprocess.Popen] = []
    zmq_context = None
    pull_socket = None
    _shutdown_requested = [False]

    def _signal_handler(signum, frame):  # noqa: ANN001
        log.info("paper_trade.shutdown_requested", signal=signum)
        _shutdown_requested[0] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Step 1: StartupReconciler — paper mode: no real state to reconcile.
        log.info("paper_trade.startup_reconcile", mode="paper", status="ok_no_open_positions")

        # Step 2: Launch ZMQ pipeline
        procs = _launch_zmq_processes(symbols)

        # Step 3: Set up PULL socket to receive signals from MLInference
        import zmq  # type: ignore[import]
        zmq_context = zmq.Context()
        pull_socket = zmq_context.socket(zmq.PULL)
        pull_socket.setsockopt(zmq.RCVTIMEO, 500)  # 500 ms timeout for non-blocking poll
        pull_socket.setsockopt(zmq.RCVHWM, 1000)
        pull_socket.connect(f"tcp://127.0.0.1:{_PULL_PORT_SIGNALS}")
        log.info("paper_trade.zmq_pull_connected", port=_PULL_PORT_SIGNALS)

        summary_printed_today = False

        # Main event loop
        while not _shutdown_requested[0]:
            ist_now = _ist_now()

            # Stop when duration reached
            if ist_now.date() > end_date:
                log.info("paper_trade.duration_reached", end_date=str(end_date))
                break

            # Step 6: Force-flat at 15:15
            if _should_force_flat(ist_now):
                for sym in list(tracker._open.keys()):
                    # Use last known fill price as proxy exit — in real use would be LTP
                    pos = tracker._open.get(sym)
                    if pos:
                        exit_px = pos.fill.fill_price  # conservative: exit at entry
                        result = tracker.close_position(sym, exit_px, _ist_now_str())
                        if result:
                            paper_logger.log_close(sym, exit_px, result)

            # Step 7: Print daily summary at 15:30
            if _is_daily_summary_time(ist_now) and not summary_printed_today:
                summary = tracker.daily_summary()
                _print_daily_summary(summary)
                paper_logger.write_summary(summary)
                summary_printed_today = True

            # Reset summary flag after midnight
            if ist_now.hour < 9:
                summary_printed_today = False

            # Step 3+4: Poll for signals (RCVTIMEO=500ms set on socket — do NOT use NOBLOCK)
            try:
                raw = pull_socket.recv()  # blocks up to RCVTIMEO ms, then raises zmq.Again
            except zmq.Again:
                # No signal within timeout window — loop back to check session state
                continue
            except zmq.ZMQError as exc:
                log.warning("paper_trade.zmq_recv_error", error=str(exc))
                continue

            # Parse signal — binary struct from inference_engine._serialise_signal
            try:
                signal_dict: dict = _decode_signal(raw)
            except (ValueError, struct.error) as exc:
                log.warning("paper_trade.signal_parse_error", error=str(exc))
                continue

            sym = signal_dict.get("symbol", "")
            if sym not in symbols:
                continue

            direction = signal_dict.get("direction", "FLAT")
            if direction == "FLAT":
                # Close existing position if any
                if sym in tracker._open:
                    pos = tracker._open.get(sym)
                    if pos:
                        exit_px = float(signal_dict.get("signal_price", pos.fill.fill_price))
                        result = tracker.close_position(sym, exit_px, _ist_now_str())
                        if result:
                            paper_logger.log_close(sym, exit_px, result)
                continue

            # Step 4: Simulate fill
            paper_fill = _simulate_fill(sym, signal_dict, capital, fill_mode=fill_mode)

            # Close existing opposite position first
            if sym in tracker._open:
                existing = tracker._open[sym]
                if existing.fill.direction != direction:
                    exit_px = paper_fill.fill_price
                    result = tracker.close_position(sym, exit_px, _ist_now_str())
                    if result:
                        paper_logger.log_close(sym, exit_px, result)

            # Open new position
            tracker.open_position(paper_fill)

            # Step 5: Log to JSONL
            paper_logger.log_fill(paper_fill, event="open")

    except KeyboardInterrupt:
        log.info("paper_trade.keyboard_interrupt")

    finally:
        # Graceful shutdown: flush all paper trades, print final summary
        log.info("paper_trade.shutdown_flush_start")
        ist_now = _ist_now()
        for sym in list(tracker._open.keys()):
            pos = tracker._open.get(sym)
            if pos:
                result = tracker.close_position(sym, pos.fill.fill_price, _ist_now_str())
                if result:
                    paper_logger.log_close(sym, pos.fill.fill_price, result)

        final_summary = tracker.daily_summary()
        _print_final_summary(final_summary)
        paper_logger.write_summary(final_summary)
        paper_logger.close()

        if pull_socket:
            pull_socket.close()
        if zmq_context:
            zmq_context.term()

        _terminate_zmq_processes(procs)
        log.info("paper_trade.session_complete")


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_daily_summary(summary: dict) -> None:
    """Print daily summary table to stdout at 15:30 IST."""
    print("\n" + "=" * 60)
    print(f"  PAPER TRADE DAILY SUMMARY — {summary.get('date', 'N/A')}")
    print("=" * 60)
    print(f"  Total P&L (INR):    {summary['pnl_inr']:>12,.2f}")
    print(f"  Equity (INR):       {summary['equity']:>12,.2f}")
    print(f"  Trades completed:   {summary['n_trades']:>12}")
    print(f"  Win rate:           {summary['win_rate']:>11.1%}")
    print(f"  Sharpe (rolling):   {summary['sharpe_rolling']:>12.4f}")
    print(f"  Max drawdown:       {summary['max_drawdown']:>11.1%}")
    open_pos = summary.get("open_positions", [])
    print(f"  Open positions:     {', '.join(open_pos) if open_pos else 'None'}")
    print("=" * 60 + "\n")


def _print_final_summary(summary: dict) -> None:
    """Print final session summary at shutdown."""
    print("\n" + "#" * 60)
    print("  PAPER TRADE SESSION COMPLETE — FINAL SUMMARY")
    print("#" * 60)
    _print_daily_summary(summary)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE AlgoTrader — Paper Trading Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        metavar="SYMBOL",
        help="NSE equity symbols to paper trade",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=500_000.0,
        help="Simulated capital in INR",
    )
    parser.add_argument(
        "--duration-days",
        type=int,
        default=90,
        help="Max calendar days to run the paper session",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to .env config file (overrides default)",
    )
    parser.add_argument(
        "--fill-mode",
        type=str,
        default="REALISTIC",
        choices=["NAIVE", "REALISTIC", "PESSIMISTIC"],
        help="Fill simulation mode: NAIVE=signal_price, REALISTIC=spread-adjusted (default), PESSIMISTIC=worst-case",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Configure logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # If a custom config path is provided, set the env var before importing settings
    if args.config_path:
        import os
        os.environ["ENV_FILE"] = args.config_path

    # ── Safety gate: paper mode ONLY ─────────────────────────────────────────
    try:
        from algo_trading.config import settings
        if settings.enable_live_trading:
            raise RuntimeError(
                "SAFETY GATE: ENABLE_LIVE_TRADING=true detected in config. "
                "Paper trading requires ENABLE_LIVE_TRADING=false. "
                "Set ENABLE_LIVE_TRADING=false in your .env and retry."
            )
        log.info(
            "paper_trade.config_ok",
            mode=settings.trading_mode.value,
            live_trading_enabled=settings.enable_live_trading,
        )
    except ImportError:
        log.warning(
            "paper_trade.config_not_available",
            msg="Could not import settings. Proceeding with CLI args only.",
        )

    log.info(
        "paper_trade.starting",
        symbols=args.symbols,
        capital_inr=args.capital,
        duration_days=args.duration_days,
        fill_mode=args.fill_mode,
        fill_simulator_available=_FILL_SIMULATOR_AVAILABLE,
    )

    log_dir = Path("logs")
    run_paper_session(
        symbols=args.symbols,
        capital=args.capital,
        duration_days=args.duration_days,
        log_dir=log_dir,
        fill_mode=args.fill_mode,
    )


if __name__ == "__main__":
    main()
