"""
paper_trading_analysis.py — Comprehensive analysis of a paper trading JSONL log.

Computes Sharpe ratio, Profit Factor, Win Rate, Max Drawdown, fill rate, and
time-of-day / symbol breakdowns. Uses IndiaCostModel for accurate net P&L.

Usage:
    python scripts/paper_trading_analysis.py --log paper_trading.jsonl
    python scripts/paper_trading_analysis.py --log paper_trading.jsonl --output paper_analysis.json
    python scripts/paper_trading_analysis.py --log paper_trading.jsonl --output paper_analysis.json --plot

Expected JSONL events (one JSON object per line):
    {"event": "fill",           "timestamp": "...", "symbol": "...", "side": "BUY"|"SELL",
                                "quantity": 10, "fill_price": 2845.50, "signal_price": 2843.00,
                                "order_type": "LIMIT"|"MARKET", "strategy_id": "...",
                                "slippage_bps": 8.8}
    {"event": "signal_skipped", "timestamp": "...", "symbol": "...", "reason": "price_drift"}
    {"event": "pnl_update",     "timestamp": "...", "realized_pnl": 165.80,
                                "unrealized_pnl": 0.0, "daily_pnl": 165.80}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

# ── IndiaCostModel import ─────────────────────────────────────────────────────
try:
    from algo_trading.risk.cost_model import IndiaCostModel
    _COST_MODEL_AVAILABLE = True
except ImportError:
    _COST_MODEL_AVAILABLE = False

# ── matplotlib: optional ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for scripts
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

_TRADING_DAYS_PER_YEAR = 252
_IST_OFFSET = timedelta(hours=5, minutes=30)

# Model's floor slippage when no ATR/spread supplied (from cost_model.py: 0.05% = 5 bps)
_MODEL_FLOOR_SLIPPAGE_BPS = Decimal("5.0")


# ══════════════════════════════════════════════════════════════════════════════
# JSONL loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load a JSONL file. Returns (records, errors).
    Malformed lines are collected in errors and skipped.
    """
    records: list[dict] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        errors.append(f"line {lineno}: not a JSON object")
                except json.JSONDecodeError as exc:
                    errors.append(f"line {lineno}: {exc}")
    except OSError as exc:
        print(f"ERROR: cannot read '{path}': {exc}", file=sys.stderr)
        sys.exit(1)
    return records, errors


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamp. Returns None on failure."""
    if not ts_str:
        return None
    # datetime.fromisoformat in Python 3.11+ handles +05:30 directly;
    # for 3.9/3.10 we use a fallback strip.
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        try:
            # Try stripping trailing Z and retrying
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None


def _to_ist_hour(ts: Optional[datetime]) -> Optional[int]:
    """Convert a timezone-aware or naive datetime to IST hour (0–23)."""
    if ts is None:
        return None
    if ts.tzinfo is not None:
        # Convert to IST
        ist_dt = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
    else:
        # Assume already IST if naive
        ist_dt = ts
    return ist_dt.hour


def _to_ist_date(ts: Optional[datetime]) -> Optional[str]:
    """Return the IST date as YYYY-MM-DD string."""
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ist_dt = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
    else:
        ist_dt = ts
    return ist_dt.date().isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Trade reconstruction (FIFO matching)
# ══════════════════════════════════════════════════════════════════════════════

def _get_required(rec: dict, key: str, lineno: int, errors: list[str]) -> Any:
    val = rec.get(key)
    if val is None:
        errors.append(f"fill record (lineno≈{lineno}): missing '{key}' field")
    return val


def _reconstruct_trades(fills: list[dict]) -> tuple[list[dict], list[str]]:
    """
    FIFO-match BUY and SELL fills by symbol to form round-trip trades.

    Returns (trades, warnings).
    Each trade dict:
        symbol, entry_side, quantity, entry_price, exit_price,
        entry_time, exit_time, entry_hour, strategy_id,
        gross_pnl, cost_inr, net_pnl
    """
    # Queue of open BUY lots per symbol: list of (qty, price, timestamp, strategy_id)
    open_buys: dict[str, list[dict]] = defaultdict(list)
    # Queue of open SELL lots per symbol (short trades)
    open_sells: dict[str, list[dict]] = defaultdict(list)
    trades: list[dict] = []
    warnings: list[str] = []

    cost_model = IndiaCostModel() if _COST_MODEL_AVAILABLE else None

    for i, fill in enumerate(fills):
        side = str(fill.get("side", "")).upper()
        symbol = str(fill.get("symbol", "UNKNOWN"))
        try:
            qty = int(fill.get("quantity", 0))
            fill_price = float(fill.get("fill_price", 0.0))
        except (TypeError, ValueError):
            warnings.append(f"fill #{i}: invalid quantity or fill_price — skipped")
            continue

        if qty <= 0 or fill_price <= 0:
            warnings.append(f"fill #{i} ({symbol}): non-positive qty/price — skipped")
            continue

        ts_str = fill.get("timestamp", "")
        ts = _parse_timestamp(ts_str)
        strategy_id = fill.get("strategy_id", "")

        lot = {
            "qty": qty,
            "price": fill_price,
            "timestamp": ts,
            "strategy_id": strategy_id,
        }

        if side == "BUY":
            # Check if there are open short positions to close
            remaining_qty = qty
            while remaining_qty > 0 and open_sells[symbol]:
                head = open_sells[symbol][0]
                matched = min(remaining_qty, head["qty"])
                # Short trade: entry SELL → exit BUY
                gross_pnl = (head["price"] - fill_price) * matched
                cost_inr = _compute_cost(cost_model, symbol, matched,
                                         entry_price=head["price"],
                                         exit_price=fill_price,
                                         direction=-1)
                net_pnl = gross_pnl - cost_inr
                trades.append({
                    "symbol": symbol,
                    "entry_side": "SELL",
                    "quantity": matched,
                    "entry_price": head["price"],
                    "exit_price": fill_price,
                    "entry_time": head["timestamp"],
                    "exit_time": ts,
                    "entry_hour": _to_ist_hour(head["timestamp"]),
                    "strategy_id": head.get("strategy_id", strategy_id),
                    "gross_pnl": round(gross_pnl, 2),
                    "cost_inr": round(cost_inr, 2),
                    "net_pnl": round(net_pnl, 2),
                })
                remaining_qty -= matched
                head["qty"] -= matched
                if head["qty"] <= 0:
                    open_sells[symbol].pop(0)

            if remaining_qty > 0:
                open_buys[symbol].append({
                    "qty": remaining_qty,
                    "price": fill_price,
                    "timestamp": ts,
                    "strategy_id": strategy_id,
                })

        elif side == "SELL":
            # Check if there are open long positions to close
            remaining_qty = qty
            while remaining_qty > 0 and open_buys[symbol]:
                head = open_buys[symbol][0]
                matched = min(remaining_qty, head["qty"])
                # Long trade: entry BUY → exit SELL
                gross_pnl = (fill_price - head["price"]) * matched
                cost_inr = _compute_cost(cost_model, symbol, matched,
                                         entry_price=head["price"],
                                         exit_price=fill_price,
                                         direction=1)
                net_pnl = gross_pnl - cost_inr
                trades.append({
                    "symbol": symbol,
                    "entry_side": "BUY",
                    "quantity": matched,
                    "entry_price": head["price"],
                    "exit_price": fill_price,
                    "entry_time": head["timestamp"],
                    "exit_time": ts,
                    "entry_hour": _to_ist_hour(head["timestamp"]),
                    "strategy_id": head.get("strategy_id", strategy_id),
                    "gross_pnl": round(gross_pnl, 2),
                    "cost_inr": round(cost_inr, 2),
                    "net_pnl": round(net_pnl, 2),
                })
                remaining_qty -= matched
                head["qty"] -= matched
                if head["qty"] <= 0:
                    open_buys[symbol].pop(0)

            if remaining_qty > 0:
                open_sells[symbol].append({
                    "qty": remaining_qty,
                    "price": fill_price,
                    "timestamp": ts,
                    "strategy_id": strategy_id,
                })
        else:
            warnings.append(f"fill #{i} ({symbol}): unknown side '{side}' — skipped")

    # Report unclosed positions as warnings (not errors — session may be mid-run)
    for symbol, lots in open_buys.items():
        total_open = sum(l["qty"] for l in lots)
        if total_open > 0:
            warnings.append(
                f"symbol {symbol}: {total_open} BUY shares remain open (no matching SELL)"
            )
    for symbol, lots in open_sells.items():
        total_open = sum(l["qty"] for l in lots)
        if total_open > 0:
            warnings.append(
                f"symbol {symbol}: {total_open} SELL shares remain open (no matching BUY)"
            )

    return trades, warnings


def _compute_cost(
    cost_model: Optional["IndiaCostModel"],
    symbol: str,
    quantity: int,
    entry_price: float,
    exit_price: float,
    direction: int,
) -> float:
    """Compute round-trip transaction cost using IndiaCostModel (or zero if unavailable)."""
    if cost_model is None:
        return 0.0
    try:
        rt = cost_model.compute_round_trip(
            symbol=symbol,
            quantity=quantity,
            entry_price=Decimal(str(entry_price)),
            exit_price=Decimal(str(exit_price)),
            direction=direction,
        )
        return float(rt.total_inr)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) if variance > 0 else 0.0


def _compute_sharpe(daily_pnl: list[float]) -> float:
    """Annualized Sharpe ratio from daily P&L series."""
    if len(daily_pnl) < 2:
        return 0.0
    mean = _safe_mean(daily_pnl)
    std = _safe_std(daily_pnl)
    if std < 1e-9:
        return 0.0
    return (mean / std) * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _compute_max_drawdown(cumulative_pnl: list[float]) -> float:
    """
    Peak-to-trough max drawdown on cumulative P&L curve.
    Returns the absolute drawdown amount (positive = drawdown).
    """
    if not cumulative_pnl:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for val in cumulative_pnl:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_profit_factor(winning_pnl: list[float], losing_pnl: list[float]) -> Optional[float]:
    """Profit factor = sum(wins) / |sum(losses)|. None if no losses."""
    total_wins = sum(winning_pnl)
    total_losses = abs(sum(losing_pnl))
    if total_losses < 1e-9:
        return None  # infinite (no losing trades)
    return total_wins / total_losses


def _build_daily_pnl(trades: list[dict]) -> dict[str, float]:
    """Aggregate net P&L per IST calendar date from closed trades."""
    daily: dict[str, float] = defaultdict(float)
    for trade in trades:
        date_str = _to_ist_date(trade.get("exit_time"))
        if date_str:
            daily[date_str] += trade["net_pnl"]
    return dict(sorted(daily.items()))


def _compute_statistics(trades: list[dict]) -> dict:
    """Compute full trading statistics from a list of round-trip trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_trades": 0,
            "lose_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "net_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_inr": 0.0,
            "avg_trade_net_pnl": 0.0,
            "avg_trade_gross_pnl": 0.0,
        }

    winning = [t for t in trades if t["net_pnl"] > 0]
    losing  = [t for t in trades if t["net_pnl"] < 0]
    even    = [t for t in trades if t["net_pnl"] == 0]

    winning_pnl = [t["net_pnl"] for t in winning]
    losing_pnl  = [t["net_pnl"] for t in losing]
    all_net_pnl = [t["net_pnl"] for t in trades]

    gross_pnl  = sum(t["gross_pnl"] for t in trades)
    total_cost = sum(t["cost_inr"] for t in trades)
    net_pnl    = sum(t["net_pnl"] for t in trades)

    cumulative = []
    running = 0.0
    for p in all_net_pnl:
        running += p
        cumulative.append(running)

    # Daily P&L for Sharpe
    daily_map = _build_daily_pnl(trades)
    daily_series = list(daily_map.values())

    return {
        "total_trades":         len(trades),
        "win_trades":           len(winning),
        "lose_trades":          len(losing),
        "breakeven_trades":     len(even),
        "win_rate_pct":         round(len(winning) / len(trades) * 100, 2),
        "profit_factor":        _compute_profit_factor(winning_pnl, losing_pnl),
        "gross_pnl":            round(gross_pnl, 2),
        "total_costs":          round(total_cost, 2),
        "net_pnl":              round(net_pnl, 2),
        "sharpe_ratio":         round(_compute_sharpe(daily_series), 4),
        "max_drawdown_inr":     round(_compute_max_drawdown(cumulative), 2),
        "avg_trade_net_pnl":    round(_safe_mean(all_net_pnl), 2),
        "avg_trade_gross_pnl":  round(_safe_mean([t["gross_pnl"] for t in trades]), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Breakdowns
# ══════════════════════════════════════════════════════════════════════════════

def _breakdown_by_hour(trades: list[dict]) -> dict:
    """Group trades by entry hour, return win_rate and avg_pnl per hour."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        hour = t.get("entry_hour")
        if hour is not None:
            buckets[hour].append(t["net_pnl"])

    result = {}
    for hour in sorted(buckets):
        pnl_list = buckets[hour]
        wins = sum(1 for p in pnl_list if p > 0)
        result[str(hour)] = {
            "trades": len(pnl_list),
            "win_rate_pct": round(wins / len(pnl_list) * 100, 2) if pnl_list else 0.0,
            "avg_net_pnl": round(_safe_mean(pnl_list), 2),
            "total_net_pnl": round(sum(pnl_list), 2),
        }
    return result


def _breakdown_by_symbol(trades: list[dict]) -> dict:
    """Group trades by symbol, return per-symbol stats."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        buckets[t["symbol"]].append(t)

    result = {}
    for symbol in sorted(buckets):
        symbol_trades = buckets[symbol]
        pnl_list = [t["net_pnl"] for t in symbol_trades]
        wins = sum(1 for p in pnl_list if p > 0)
        losses = sum(1 for p in pnl_list if p < 0)
        result[symbol] = {
            "trades": len(symbol_trades),
            "win_trades": wins,
            "lose_trades": losses,
            "win_rate_pct": round(wins / len(symbol_trades) * 100, 2),
            "total_net_pnl": round(sum(pnl_list), 2),
            "avg_net_pnl": round(_safe_mean(pnl_list), 2),
            "total_costs": round(sum(t["cost_inr"] for t in symbol_trades), 2),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Slippage & signal skip stats
# ══════════════════════════════════════════════════════════════════════════════

def _slippage_stats(fills: list[dict]) -> dict:
    """Compute average slippage in bps from fill events that include slippage_bps."""
    bps_list = []
    for f in fills:
        raw = f.get("slippage_bps")
        if raw is not None:
            try:
                bps_list.append(float(raw))
            except (TypeError, ValueError):
                pass
    if not bps_list:
        return {"count": 0, "mean_bps": None, "median_bps": None, "std_bps": None}
    bps_sorted = sorted(bps_list)
    n = len(bps_sorted)
    median = bps_sorted[n // 2] if n % 2 else (bps_sorted[n // 2 - 1] + bps_sorted[n // 2]) / 2
    return {
        "count": n,
        "mean_bps": round(_safe_mean(bps_list), 4),
        "median_bps": round(median, 4),
        "std_bps": round(_safe_std(bps_list), 4),
    }


def _signal_skip_breakdown(skipped: list[dict]) -> dict:
    """Count signal_skipped events by reason."""
    reason_counts: dict[str, int] = defaultdict(int)
    for s in skipped:
        reason = str(s.get("reason", "unknown"))
        reason_counts[reason] += 1
    return dict(sorted(reason_counts.items()))


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_pnl(value: float) -> str:
    prefix = "+" if value >= 0 else ""
    return f"{prefix}₹{value:,.2f}"


def _fmt_opt_float(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "∞ (no losing trades)"
    return f"{value:.{decimals}f}"


def _print_report(
    stats: dict,
    fill_rate: dict,
    slippage: dict,
    skip_breakdown: dict,
    hourly: dict,
    symbol_breakdown: dict,
    log_errors: list[str],
    trade_warnings: list[str],
) -> None:
    """Print formatted analysis report to stdout."""
    W = 65

    print("\n" + "=" * W)
    print("  NSE ALGO TRADER — PAPER TRADING ANALYSIS REPORT")
    print("=" * W)

    # ── Core stats ────────────────────────────────────────────────────────────
    print("\n[TRADE STATISTICS]")
    print(f"  Total Round-Trip Trades  : {stats['total_trades']:>10}")
    print(f"  Win Trades               : {stats['win_trades']:>10}")
    print(f"  Lose Trades              : {stats['lose_trades']:>10}")
    print(f"  Breakeven Trades         : {stats['breakeven_trades']:>10}")
    print(f"  Win Rate                 : {stats['win_rate_pct']:>9.2f}%")
    pf = stats["profit_factor"]
    print(f"  Profit Factor            : {_fmt_opt_float(pf, 3):>10}")

    print("\n[P&L SUMMARY]")
    print(f"  Gross P&L                : {_fmt_pnl(stats['gross_pnl']):>14}")
    print(f"  Total Transaction Costs  : {_fmt_pnl(-stats['total_costs']):>14}")
    print(f"  Net P&L (after costs)    : {_fmt_pnl(stats['net_pnl']):>14}")
    print(f"  Avg Net P&L per Trade    : {_fmt_pnl(stats['avg_trade_net_pnl']):>14}")
    print(f"  Max Drawdown (INR)       : {_fmt_pnl(-stats['max_drawdown_inr']):>14}")

    print("\n[RISK-ADJUSTED RETURNS]")
    print(f"  Sharpe Ratio (annualised): {stats['sharpe_ratio']:>10.4f}")

    print("\n[FILL & SIGNAL STATS]")
    print(f"  Total Fills              : {fill_rate['fills']:>10}")
    print(f"  Total Signals Skipped    : {fill_rate['skipped']:>10}")
    fill_pct = fill_rate.get("fill_rate_pct", 0.0)
    print(f"  Fill Rate                : {fill_pct:>9.2f}%")
    if skip_breakdown:
        print("  Skip Reasons:")
        for reason, count in skip_breakdown.items():
            print(f"    {reason:<28}: {count:>6}")

    print("\n[SLIPPAGE]")
    if slippage["count"] == 0:
        print("  No slippage_bps data in fill records.")
    else:
        print(f"  Fills with slippage data : {slippage['count']:>10}")
        print(f"  Mean Slippage            : {slippage['mean_bps']:>9.4f} bps")
        print(f"  Median Slippage          : {slippage['median_bps']:>9.4f} bps")
        print(f"  Std Dev Slippage         : {slippage['std_bps']:>9.4f} bps")

    # ── Hourly breakdown ──────────────────────────────────────────────────────
    if hourly:
        print("\n[TIME-OF-DAY BREAKDOWN]")
        print(f"  {'Hour':<6} {'Trades':>7} {'Win%':>8} {'Avg Net P&L':>14} {'Total Net P&L':>15}")
        print("  " + "-" * 54)
        for hour_str, row in sorted(hourly.items(), key=lambda x: int(x[0])):
            print(
                f"  {hour_str:0>2}:00  "
                f"{row['trades']:>7}  "
                f"{row['win_rate_pct']:>7.1f}%  "
                f"{_fmt_pnl(row['avg_net_pnl']):>13}  "
                f"{_fmt_pnl(row['total_net_pnl']):>14}"
            )

    # ── Symbol breakdown ──────────────────────────────────────────────────────
    if symbol_breakdown:
        print("\n[SYMBOL BREAKDOWN]")
        print(f"  {'Symbol':<16} {'Trades':>7} {'Win%':>8} {'Costs':>10} {'Net P&L':>12}")
        print("  " + "-" * 58)
        for sym, row in sorted(
            symbol_breakdown.items(), key=lambda x: -x[1]["total_net_pnl"]
        ):
            print(
                f"  {sym:<16} "
                f"{row['trades']:>7}  "
                f"{row['win_rate_pct']:>7.1f}%  "
                f"{_fmt_pnl(-row['total_costs']):>9}  "
                f"{_fmt_pnl(row['total_net_pnl']):>11}"
            )

    # ── Warnings / errors ─────────────────────────────────────────────────────
    if log_errors:
        print(f"\n[JSONL PARSE ERRORS] ({len(log_errors)} errors)")
        for e in log_errors[:10]:
            print(f"  • {e}")
        if len(log_errors) > 10:
            print(f"  ... and {len(log_errors) - 10} more")

    if trade_warnings:
        print(f"\n[TRADE RECONSTRUCTION WARNINGS] ({len(trade_warnings)} warnings)")
        for w in trade_warnings[:10]:
            print(f"  • {w}")
        if len(trade_warnings) > 10:
            print(f"  ... and {len(trade_warnings) - 10} more")

    if not _COST_MODEL_AVAILABLE:
        print("\n  NOTE: algo_trading.risk.cost_model not importable — costs shown as ₹0.00")

    print("\n" + "=" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _save_plots(
    trades: list[dict],
    fills: list[dict],
    hourly: dict,
    log_path: Path,
) -> list[str]:
    """
    Save matplotlib charts alongside the log file. Returns list of saved paths.
    """
    if not _MPL_AVAILABLE:
        print(
            "WARNING: matplotlib not installed — skipping plots. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return []

    saved: list[str] = []
    base = log_path.parent / log_path.stem

    # ── 1. Cumulative P&L curve ───────────────────────────────────────────────
    if trades:
        cum_pnl = []
        running = 0.0
        for t in trades:
            running += t["net_pnl"]
            cum_pnl.append(running)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(cum_pnl)), cum_pnl, linewidth=1.5, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=[v >= 0 for v in cum_pnl],
                        alpha=0.2, color="green", label="profit")
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=[v < 0 for v in cum_pnl],
                        alpha=0.2, color="red", label="loss")
        ax.set_title("Cumulative Net P&L (INR)", fontsize=13)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("P&L (₹)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = f"{base}_cumulative_pnl.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)

    # ── 2. Win rate by hour of day ────────────────────────────────────────────
    if hourly:
        hours = sorted(int(h) for h in hourly)
        win_rates = [hourly[str(h)]["win_rate_pct"] for h in hours]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(hours, win_rates, color="teal", alpha=0.75)
        ax.axhline(50, color="red", linewidth=0.8, linestyle="--", label="50% baseline")
        ax.set_title("Win Rate by Hour of Day (IST)", fontsize=13)
        ax.set_xlabel("Hour (IST)")
        ax.set_ylabel("Win Rate (%)")
        ax.set_xticks(hours)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        # Label bars
        for bar, wr in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{wr:.0f}%", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out_path = f"{base}_win_rate_by_hour.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)

    # ── 3. Slippage distribution histogram ────────────────────────────────────
    bps_list = []
    for f in fills:
        raw = f.get("slippage_bps")
        if raw is not None:
            try:
                bps_list.append(float(raw))
            except (TypeError, ValueError):
                pass

    if bps_list:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(bps_list, bins=30, color="darkorange", alpha=0.75, edgecolor="white")
        mean_bps = _safe_mean(bps_list)
        ax.axvline(mean_bps, color="red", linewidth=1.2, linestyle="--",
                   label=f"mean={mean_bps:.2f} bps")
        ax.set_title("Slippage Distribution (bps)", fontsize=13)
        ax.set_xlabel("Slippage (bps)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = f"{base}_slippage_hist.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE AlgoTrader — Paper Trading Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log",
        required=True,
        metavar="FILE",
        help="Path to paper trading JSONL log file",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Optional path to save full JSON analysis output",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save matplotlib charts (cumulative P&L, win rate by hour, slippage histogram)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log)

    if not log_path.exists():
        print(f"ERROR: log file not found: '{log_path}'", file=sys.stderr)
        sys.exit(1)

    # ── Load JSONL ────────────────────────────────────────────────────────────
    records, parse_errors = _load_jsonl(log_path)

    if not records:
        print(
            f"WARNING: '{log_path}' is empty or contains no valid JSON objects.",
            file=sys.stderr,
        )
        # Still produce a blank report rather than crash
        records = []

    # ── Partition event types ─────────────────────────────────────────────────
    fills: list[dict]   = [r for r in records if r.get("event") == "fill"]
    skipped: list[dict] = [r for r in records if r.get("event") == "signal_skipped"]
    pnl_events: list[dict] = [r for r in records if r.get("event") == "pnl_update"]

    # ── Reconstruct trades ────────────────────────────────────────────────────
    trades, trade_warnings = _reconstruct_trades(fills)

    # ── Compute statistics ────────────────────────────────────────────────────
    stats = _compute_statistics(trades)

    fill_count = len(fills)
    skip_count = len(skipped)
    total_signals = fill_count + skip_count
    fill_rate = {
        "fills": fill_count,
        "skipped": skip_count,
        "fill_rate_pct": round(fill_count / total_signals * 100, 2) if total_signals > 0 else 0.0,
    }

    slippage = _slippage_stats(fills)
    skip_breakdown = _signal_skip_breakdown(skipped)
    hourly = _breakdown_by_hour(trades)
    symbol_breakdown = _breakdown_by_symbol(trades)

    # ── Build daily pnl for output ────────────────────────────────────────────
    daily_pnl_map = _build_daily_pnl(trades)

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(
        stats=stats,
        fill_rate=fill_rate,
        slippage=slippage,
        skip_breakdown=skip_breakdown,
        hourly=hourly,
        symbol_breakdown=symbol_breakdown,
        log_errors=parse_errors,
        trade_warnings=trade_warnings,
    )

    # ── Save JSON output ──────────────────────────────────────────────────────
    if args.output:
        output: dict[str, Any] = {
            "source_log": str(log_path.resolve()),
            "total_records": len(records),
            "parse_errors": len(parse_errors),
            "trade_warnings": trade_warnings,
            "fill_rate": fill_rate,
            "signal_skip_breakdown": skip_breakdown,
            "slippage": slippage,
            "statistics": stats,
            "daily_pnl": daily_pnl_map,
            "hourly_breakdown": hourly,
            "symbol_breakdown": symbol_breakdown,
            "cost_model_used": _COST_MODEL_AVAILABLE,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"Analysis saved to: {out_path.resolve()}")

    # ── Save plots ────────────────────────────────────────────────────────────
    if args.plot:
        saved = _save_plots(trades, fills, hourly, log_path)
        if saved:
            print("Charts saved:")
            for p in saved:
                print(f"  {p}")
        else:
            print("No charts saved (no data or matplotlib unavailable).")


if __name__ == "__main__":
    main()
