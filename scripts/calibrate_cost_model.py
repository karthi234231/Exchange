"""
calibrate_cost_model.py — Compares intended fill prices vs actual fill prices
in a paper trading JSONL log, calibrates dynamic slippage model coefficients.

Usage:
    python scripts/calibrate_cost_model.py --log paper_trading.jsonl
    python scripts/calibrate_cost_model.py --log paper_trading.jsonl --output slippage_calibration.json

Expected fill event format in JSONL:
    {"event": "fill", "timestamp": "2026-06-10T09:45:00+05:30",
     "symbol": "RELIANCE", "side": "BUY", "quantity": 10,
     "fill_price": 2845.50, "signal_price": 2843.00,
     "order_type": "LIMIT", "strategy_id": "...", "slippage_bps": 8.8}

Slippage convention:
    Positive bps  = fill was worse than signal (paid more to BUY, received less to SELL).
    Negative bps  = fill was better than signal (positive slippage / price improvement).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# ── IndiaCostModel — inspect model floor slippage assumption ──────────────────
try:
    from algo_trading.risk.cost_model import (
        IndiaCostModel,
        _ATR_SLIPPAGE_BASE_PCT,
        _OPEN_CLOSE_MULTIPLIER,
        _SPREAD_SLIPPAGE_HALF,
    )
    _COST_MODEL_AVAILABLE = True
except ImportError:
    _COST_MODEL_AVAILABLE = False
    _ATR_SLIPPAGE_BASE_PCT = None
    _OPEN_CLOSE_MULTIPLIER = None
    _SPREAD_SLIPPAGE_HALF = None

# Model floor slippage in bps (0.05% = 5 bps): used when no ATR/spread supplied
_MODEL_FLOOR_SLIPPAGE_BPS = 5.0

# Threshold for recommending an upward adjustment (20% above model assumption)
_ADJUSTMENT_THRESHOLD_PCT = 0.20

# Keys used for segmentation
_SEGMENT_KEYS = ["LIMIT_BUY", "LIMIT_SELL", "MARKET_BUY", "MARKET_SELL"]

# Open/close window in IST hours (used for time-of-day analysis)
_OPEN_HOUR = 9   # 09:00–09:59 IST
_CLOSE_HOUR = 15  # 15:00–15:59 IST


# ══════════════════════════════════════════════════════════════════════════════
# JSONL loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_fills(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load fill events from a JSONL file. Returns (fills, errors).
    Skips non-fill events and malformed lines.
    """
    fills: list[dict] = []
    errors: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {lineno}: JSON parse error — {exc}")
                    continue

                if not isinstance(obj, dict):
                    continue
                if obj.get("event") != "fill":
                    continue
                fills.append(obj)

    except OSError as exc:
        print(f"ERROR: cannot read '{path}': {exc}", file=sys.stderr)
        sys.exit(1)

    return fills, errors


# ══════════════════════════════════════════════════════════════════════════════
# Slippage computation
# ══════════════════════════════════════════════════════════════════════════════

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None


def _to_ist_hour(ts: Optional[datetime]) -> Optional[int]:
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ist_dt = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
    else:
        ist_dt = ts
    return ist_dt.hour


def _compute_actual_slippage_bps(fill: dict) -> Optional[float]:
    """
    Compute sign-aware actual slippage for a fill.

    Convention:
        BUY:  positive = fill_price > signal_price (paid more — bad)
              negative = fill_price < signal_price (paid less — good)
        SELL: positive = signal_price > fill_price (received less — bad)
              negative = signal_price < fill_price (received more — good)

    Formula for BUY:  (fill_price - signal_price) / signal_price * 10_000
    Formula for SELL: (signal_price - fill_price) / signal_price * 10_000

    Returns None if required fields are missing or invalid.
    """
    side = str(fill.get("side", "")).upper()
    try:
        fill_price = float(fill.get("fill_price", 0.0))
        signal_price = float(fill.get("signal_price", 0.0))
    except (TypeError, ValueError):
        return None

    if signal_price <= 0 or fill_price <= 0:
        return None

    if side == "BUY":
        return (fill_price - signal_price) / signal_price * 10_000
    elif side == "SELL":
        return (signal_price - fill_price) / signal_price * 10_000
    else:
        return None


def _segment_key(fill: dict) -> Optional[str]:
    """Return segment key like 'LIMIT_BUY', 'MARKET_SELL', etc."""
    order_type = str(fill.get("order_type", "")).upper().strip()
    side = str(fill.get("side", "")).upper().strip()
    if not order_type or not side:
        return None
    # Normalize to LIMIT / MARKET
    if "LIMIT" in order_type:
        ot = "LIMIT"
    elif "MARKET" in order_type or "MKT" in order_type:
        ot = "MARKET"
    else:
        ot = order_type  # Keep as-is for unknown types
    return f"{ot}_{side}"


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile from a sorted list (linear interpolation)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) if variance > 0 else 0.0


def _slippage_stats_for_series(bps_list: list[float]) -> dict:
    """Return mean/median/std/p5/p95/min/max for a list of slippage bps values."""
    if not bps_list:
        return {
            "count": 0,
            "mean_bps": None,
            "median_bps": None,
            "std_bps": None,
            "p5_bps": None,
            "p95_bps": None,
            "min_bps": None,
            "max_bps": None,
        }
    s = sorted(bps_list)
    return {
        "count": len(s),
        "mean_bps": round(_safe_mean(bps_list), 4),
        "median_bps": round(_percentile(s, 50), 4),
        "std_bps": round(_safe_std(bps_list), 4),
        "p5_bps": round(_percentile(s, 5), 4),
        "p95_bps": round(_percentile(s, 95), 4),
        "min_bps": round(s[0], 4),
        "max_bps": round(s[-1], 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model assumption retrieval
# ══════════════════════════════════════════════════════════════════════════════

def _get_model_assumptions() -> dict:
    """
    Return the cost model's assumed slippage rates as bps.

    IndiaCostModel's dynamic slippage formula (from cost_model.py):
        slippage_per_share = 0.30 * ATR * tod_multiplier  (if ATR available)
                           + 0.50 * spread                (if spread available)
        floor (no ATR/spread): 0.05% of price = 5 bps

    For a fair comparison we report the floor (5 bps) as the "assumed minimum"
    for both LIMIT and MARKET. The model does not distinguish between LIMIT/MARKET
    by default — slippage is purely ATR/spread-driven with the 5 bps floor.
    """
    notes = "Dynamic model (ATR+spread-based). Values shown are the floor (no ATR/spread)."

    if _COST_MODEL_AVAILABLE:
        # Express model params as text descriptions
        atr_pct = float(_ATR_SLIPPAGE_BASE_PCT) * 100  # type: ignore[arg-type]
        tod_mult = float(_OPEN_CLOSE_MULTIPLIER)  # type: ignore[arg-type]
        spread_half = float(_SPREAD_SLIPPAGE_HALF)  # type: ignore[arg-type]
        return {
            "notes": notes,
            "floor_slippage_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "atr_component_pct_of_atr": atr_pct,
            "open_close_multiplier": tod_mult,
            "spread_component_fraction": spread_half,
            "LIMIT_BUY_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "LIMIT_SELL_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "MARKET_BUY_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "MARKET_SELL_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
        }
    else:
        return {
            "notes": "algo_trading.risk.cost_model not importable — using hardcoded floor",
            "floor_slippage_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "LIMIT_BUY_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "LIMIT_SELL_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "MARKET_BUY_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
            "MARKET_SELL_floor_bps": _MODEL_FLOOR_SLIPPAGE_BPS,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Calibration logic
# ══════════════════════════════════════════════════════════════════════════════

def _calibration_recommendation(
    segment: str,
    actual_stats: dict,
    model_floor: float,
) -> dict:
    """
    Recommend a new slippage value if actual exceeds model assumption by >20%.

    We compare actual mean_bps against the model floor.
    Returns a recommendation dict for the segment.
    """
    mean_bps = actual_stats.get("mean_bps")
    if mean_bps is None or actual_stats["count"] == 0:
        return {
            "segment": segment,
            "current_model_floor_bps": model_floor,
            "actual_mean_bps": None,
            "recommended_bps": None,
            "recommended_adjustment": "insufficient data",
            "action": "no_change",
        }

    threshold = model_floor * (1 + _ADJUSTMENT_THRESHOLD_PCT)
    delta = mean_bps - model_floor

    if mean_bps > threshold:
        recommended = round(mean_bps * 1.10, 2)  # add 10% buffer above actual mean
        action = "increase"
        adj_str = f"increase by +{delta:.2f} bps (model floor {model_floor:.1f} → recommended {recommended:.2f})"
    elif mean_bps < model_floor * (1 - _ADJUSTMENT_THRESHOLD_PCT):
        recommended = round(max(mean_bps * 1.05, 1.0), 2)  # never go below 1 bps
        action = "decrease"
        adj_str = f"decrease by {delta:.2f} bps (model floor {model_floor:.1f} → recommended {recommended:.2f})"
    else:
        recommended = model_floor
        action = "no_change"
        adj_str = f"within tolerance (actual mean {mean_bps:.2f} bps ≈ model {model_floor:.1f} bps)"

    return {
        "segment": segment,
        "current_model_floor_bps": model_floor,
        "actual_mean_bps": round(mean_bps, 4),
        "recommended_bps": recommended,
        "recommended_adjustment": adj_str,
        "action": action,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Time-of-day slippage analysis
# ══════════════════════════════════════════════════════════════════════════════

def _time_of_day_analysis(
    fills: list[dict],
    slippage_by_fill: list[Optional[float]],
) -> dict:
    """
    Bucket fills by IST hour and compute mean/p95 slippage per bucket.
    """
    hour_buckets: dict[int, list[float]] = defaultdict(list)
    for fill, bps in zip(fills, slippage_by_fill):
        if bps is None:
            continue
        ts = _parse_timestamp(fill.get("timestamp", ""))
        hour = _to_ist_hour(ts)
        if hour is not None:
            hour_buckets[hour].append(bps)

    result = {}
    for hour in sorted(hour_buckets):
        bps_list = hour_buckets[hour]
        s = sorted(bps_list)
        result[str(hour)] = {
            "count": len(bps_list),
            "mean_bps": round(_safe_mean(bps_list), 4),
            "p95_bps": round(_percentile(s, 95), 4),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(
    fills_analyzed: int,
    skipped_fills: int,
    actual_slippage: dict,
    model_assumptions: dict,
    calibration: dict,
    tod_analysis: dict,
    parse_errors: list[str],
) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  NSE ALGO TRADER — COST MODEL CALIBRATION REPORT")
    print("=" * W)

    print(f"\n  Fills analysed  : {fills_analyzed}")
    if skipped_fills > 0:
        print(f"  Fills skipped   : {skipped_fills} (missing side/price data)")

    # ── Model assumptions ─────────────────────────────────────────────────────
    print(f"\n  Model floor slippage (no ATR/spread): "
          f"{model_assumptions['floor_slippage_bps']:.1f} bps")
    if _COST_MODEL_AVAILABLE:
        print(f"  ATR component   : {model_assumptions.get('atr_component_pct_of_atr', '?')}% of ATR")
        print(f"  Open/close mult.: {model_assumptions.get('open_close_multiplier', '?')}×")

    # ── Actual slippage table ─────────────────────────────────────────────────
    print("\n[ACTUAL SLIPPAGE BY SEGMENT]")
    print(f"  {'Segment':<16} {'N':>6} {'Mean':>9} {'Median':>9} {'Std':>9} {'P95':>9} {'Min':>9} {'Max':>9}")
    print("  " + "-" * 75)
    for seg in _SEGMENT_KEYS:
        stats = actual_slippage.get(seg, {})
        if stats.get("count", 0) == 0:
            print(f"  {seg:<16} {'—':>6} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9}")
        else:
            def _f(v: Optional[float]) -> str:
                return f"{v:.2f}" if v is not None else "—"
            print(
                f"  {seg:<16}"
                f"  {stats['count']:>5}"
                f"  {_f(stats['mean_bps']):>8}"
                f"  {_f(stats['median_bps']):>8}"
                f"  {_f(stats['std_bps']):>8}"
                f"  {_f(stats['p95_bps']):>8}"
                f"  {_f(stats['min_bps']):>8}"
                f"  {_f(stats['max_bps']):>8}"
            )

    # ── Calibration recommendations ───────────────────────────────────────────
    print("\n[CALIBRATION RECOMMENDATIONS]")
    for seg in _SEGMENT_KEYS:
        rec = calibration.get(seg, {})
        action = rec.get("action", "no_change")
        adj = rec.get("recommended_adjustment", "no data")
        symbol = {"increase": "▲", "decrease": "▼", "no_change": "✓"}.get(action, " ")
        print(f"  {symbol} {seg:<16}: {adj}")

    # ── Suggested config values ───────────────────────────────────────────────
    adjustments_needed = [
        (seg, calibration[seg])
        for seg in _SEGMENT_KEYS
        if calibration.get(seg, {}).get("action") != "no_change"
           and calibration.get(seg, {}).get("recommended_bps") is not None
    ]
    if adjustments_needed:
        print("\n[SUGGESTED CONFIG VALUES]")
        print("  Add these to your .env or config:")
        for seg, rec in adjustments_needed:
            key = seg + "_SLIPPAGE_BPS"
            val = rec["recommended_bps"]
            print(f"    {key} = {val}")
    else:
        print("\n  All segments within tolerance — no config changes needed.")

    # ── Time-of-day analysis ──────────────────────────────────────────────────
    if tod_analysis:
        print("\n[TIME-OF-DAY SLIPPAGE (IST HOURS)]")
        print(f"  {'Hour':<6} {'Fills':>7} {'Mean bps':>10} {'P95 bps':>10}  Regime")
        print("  " + "-" * 44)
        for hour_str, row in sorted(tod_analysis.items(), key=lambda x: int(x[0])):
            h = int(hour_str)
            regime = ""
            if h == _OPEN_HOUR:
                regime = "<-- OPEN (higher slippage expected)"
            elif h == _CLOSE_HOUR:
                regime = "<-- CLOSE (higher slippage expected)"
            print(
                f"  {hour_str:0>2}:00  "
                f"{row['count']:>7}  "
                f"{row['mean_bps']:>9.3f}  "
                f"{row['p95_bps']:>9.3f}  "
                f"{regime}"
            )

    if parse_errors:
        print(f"\n[PARSE ERRORS] ({len(parse_errors)} total)")
        for e in parse_errors[:10]:
            print(f"  • {e}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")

    if not _COST_MODEL_AVAILABLE:
        print(
            "\n  NOTE: algo_trading.risk.cost_model not importable — "
            "model assumptions use hardcoded floor (5 bps)"
        )

    print("\n" + "=" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE AlgoTrader — Cost Model Slippage Calibration",
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
        help="Optional path to save calibration JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log)

    if not log_path.exists():
        print(f"ERROR: log file not found: '{log_path}'", file=sys.stderr)
        sys.exit(1)

    # ── Load fills ────────────────────────────────────────────────────────────
    fills, parse_errors = _load_fills(log_path)

    if not fills:
        print(
            f"WARNING: '{log_path}' contains no 'fill' events. "
            "Calibration requires at least one fill record.",
            file=sys.stderr,
        )

    # ── Compute actual slippage per fill ──────────────────────────────────────
    slippage_by_fill: list[Optional[float]] = []
    skipped_fills = 0

    # Grouped by segment
    segment_bps: dict[str, list[float]] = defaultdict(list)

    for fill in fills:
        bps = _compute_actual_slippage_bps(fill)
        slippage_by_fill.append(bps)

        if bps is None:
            skipped_fills += 1
            continue

        key = _segment_key(fill)
        if key:
            segment_bps[key].append(bps)

    # ── Aggregate stats per segment ───────────────────────────────────────────
    actual_slippage: dict[str, dict] = {}
    for seg in _SEGMENT_KEYS:
        actual_slippage[seg] = _slippage_stats_for_series(segment_bps.get(seg, []))

    # Also compute an "ALL" aggregate
    all_bps = [bps for bps in slippage_by_fill if bps is not None]
    actual_slippage["ALL"] = _slippage_stats_for_series(all_bps)

    # ── Model assumptions ─────────────────────────────────────────────────────
    model_assumptions = _get_model_assumptions()

    # ── Calibration recommendations ───────────────────────────────────────────
    calibration: dict[str, Any] = {}
    for seg in _SEGMENT_KEYS:
        floor = model_assumptions.get(f"{seg}_floor_bps", _MODEL_FLOOR_SLIPPAGE_BPS)
        calibration[seg] = _calibration_recommendation(
            segment=seg,
            actual_stats=actual_slippage[seg],
            model_floor=float(floor),
        )

    # ── Time-of-day analysis ──────────────────────────────────────────────────
    tod_analysis = _time_of_day_analysis(fills, slippage_by_fill)

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(
        fills_analyzed=len(fills) - skipped_fills,
        skipped_fills=skipped_fills,
        actual_slippage=actual_slippage,
        model_assumptions=model_assumptions,
        calibration=calibration,
        tod_analysis=tod_analysis,
        parse_errors=parse_errors,
    )

    # ── Save JSON output ──────────────────────────────────────────────────────
    if args.output:
        output: dict[str, Any] = {
            "source_log": str(log_path.resolve()),
            "fills_analyzed": len(fills) - skipped_fills,
            "fills_skipped": skipped_fills,
            "parse_errors": len(parse_errors),
            "actual_slippage": actual_slippage,
            "model_assumptions": {
                k: (float(v) if hasattr(v, "__float__") else v)
                for k, v in model_assumptions.items()
            },
            "calibration": calibration,
            "time_of_day_slippage": tod_analysis,
            "cost_model_available": _COST_MODEL_AVAILABLE,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"Calibration results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
