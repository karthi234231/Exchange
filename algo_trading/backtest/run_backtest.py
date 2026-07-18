"""
run_backtest.py — Run EMA cross backtest on real intraday NSE data.
Loads .txt tick files from nifty-banknifty-intraday-data-main, 
aggregates to 15-min bars, runs walk-forward backtest, prints results.
"""
from __future__ import annotations

import os
import sys
import math
import statistics
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=False)],
    wrapper_class=structlog.make_filtering_bound_logger(40),  # WARNING level
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = r"c:\Exchange\data\nifty-banknifty-intraday-data-main"
NSE_SESSION_START = (9, 15)
NSE_SESSION_END = (15, 30)
NO_TRADE_BEFORE = (9, 30)
SQUARE_OFF_TIME = (15, 15)
BARS_PER_DAY = 25  # 15-min bars in NSE session
TRADING_DAYS_PER_YEAR = 250

# Backtest params
FAST_EMA = 9
SLOW_EMA = 21
INITIAL_CAPITAL = Decimal("500000")
MAX_POSITION_PCT = 0.05
STOP_LOSS_ATR_MULT = 1.5
MAX_HOLDING_BARS = 20
MAX_PRICE_CHASE_BPS = 50.0


def parse_txt_file(filepath: str) -> pl.DataFrame | None:
    """
    Parse a .txt intraday file into a Polars DataFrame.
    
    Two formats supported:
    - Stock files (8 cols): SYMBOL,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME
    - Index files (7 cols): SYMBOL,DATE,TIME,OPEN,HIGH,LOW,CLOSE (no volume)
    """
    try:
        # Read first line to determine column count
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        n_cols = len(first_line.split(','))
        
        if n_cols == 8:
            # Stock format with volume
            df = pl.read_csv(
                filepath,
                has_header=False,
                separator=",",
                schema_overrides={
                    "column_1": pl.Utf8,
                    "column_2": pl.Int64,
                    "column_3": pl.Utf8,
                    "column_4": pl.Float64,
                    "column_5": pl.Float64,
                    "column_6": pl.Float64,
                    "column_7": pl.Float64,
                    "column_8": pl.Int64,
                },
                infer_schema_length=10000,
            )
            df = df.rename({
                "column_1": "symbol", "column_2": "date", "column_3": "time",
                "column_4": "open", "column_5": "high", "column_6": "low",
                "column_7": "close", "column_8": "volume",
            })
        elif n_cols == 7:
            # Index format without volume
            df = pl.read_csv(
                filepath,
                has_header=False,
                separator=",",
                schema_overrides={
                    "column_1": pl.Utf8,
                    "column_2": pl.Int64,
                    "column_3": pl.Utf8,
                    "column_4": pl.Float64,
                    "column_5": pl.Float64,
                    "column_6": pl.Float64,
                    "column_7": pl.Float64,
                },
                infer_schema_length=10000,
            )
            df = df.rename({
                "column_1": "symbol", "column_2": "date", "column_3": "time",
                "column_4": "open", "column_5": "high", "column_6": "low",
                "column_7": "close",
            })
            df = df.with_columns(pl.lit(0).alias("volume"))
        else:
            return None
            
        if df.is_empty():
            return None
        
        # Build datetime: YYYYMMDD + HH:MM → datetime
        df = df.with_columns(
            pl.datetime(
                pl.col("date").str.slice(0, 4).cast(pl.Int32),
                pl.col("date").str.slice(4, 2).cast(pl.Int32),
                pl.col("date").str.slice(6, 2).cast(pl.Int32),
                pl.col("time").str.split(":").list.get(0).cast(pl.Int32),
                pl.col("time").str.split(":").list.get(1).cast(pl.Int32),
                0,
            ).alias("datetime")
        )
        
        # Filter to NSE session hours only
        df = df.filter(
            (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute())
            >= (NSE_SESSION_START[0] * 60 + NSE_SESSION_START[1])
        ).filter(
            (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute())
            < (NSE_SESSION_END[0] * 60 + NSE_SESSION_END[1])
        )
        
        if df.is_empty():
            return None
            
        return df.sort("datetime")
    except Exception as e:
        log.warning("parse_error", file=os.path.basename(filepath), error=str(e))
        return None


def resample_to_15min(df: pl.DataFrame) -> pl.DataFrame:
    """Resample 1-min tick data to 15-min OHLCV bars matching NSE session."""
    if df is None or df.is_empty():
        return None
    
    # Create 15-min bin: floor datetime to nearest 15-min boundary
    df = df.with_columns(
        (pl.col("datetime").dt.timestamp("ms") / (15 * 60 * 1000)).floor().cast(pl.Int64)
        .alias("bin_id")
    )
    
    # Group by bin and symbol, compute OHLCV
    resampled = df.group_by(["symbol", "bin_id"]).agg([
        pl.col("datetime").first().alias("timestamp"),
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]).sort("timestamp")
    
    if resampled.is_empty():
        return None
    
    # Convert bin_id back to proper datetime
    resampled = resampled.with_columns(
        pl.from_epoch(pl.col("bin_id") * (15 * 60), time_unit="s").alias("timestamp")
    )
    
    return resampled.select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])


def compute_ema_signals(closes: list[float], fast: int, slow: int) -> list[int]:
    """Compute EMA cross signals: +1 bullish cross, -1 bearish cross, 0 otherwise."""
    n = len(closes)
    sigs = [0] * n
    alpha_f = 2.0 / (fast + 1)
    alpha_s = 2.0 / (slow + 1)
    ema_f = closes[0]
    ema_s = closes[0]
    
    for i in range(n):
        prev_f = ema_f
        prev_s = ema_s
        ema_f = alpha_f * closes[i] + (1 - alpha_f) * ema_f
        ema_s = alpha_s * closes[i] + (1 - alpha_s) * ema_s
        if i < slow:
            continue
        if prev_f <= prev_s and ema_f > ema_s:
            sigs[i] = 1
        elif prev_f >= prev_s and ema_f < ema_s:
            sigs[i] = -1
    return sigs


def compute_atr(closes: list[float], highs: list[float], lows: list[float], period: int = 14) -> list[float]:
    """Compute Average True Range."""
    n = len(closes)
    atrs = [closes[0] * 0.005] * n
    if n < 2:
        return atrs
    true_ranges = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
        if len(true_ranges) < period:
            atrs[i] = sum(true_ranges) / len(true_ranges)
        else:
            atrs[i] = (atrs[i-1] * (period - 1) + tr) / period
    return atrs


def run_backtest(df: pl.DataFrame) -> dict:
    """Run EMA cross backtest on 15-min data with full cost model."""
    symbol = df["symbol"][0]
    closes = df["close"].to_list()
    opens = df["open"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    volumes = df["volume"].to_list()
    timestamps = df["timestamp"].to_list()
    n = len(df)
    
    signals = compute_ema_signals(closes, FAST_EMA, SLOW_EMA)
    atrs = compute_atr(closes, highs, lows)
    
    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = [1.0]
    no_trade_min = NO_TRADE_BEFORE[0] * 60 + NO_TRADE_BEFORE[1]
    sqoff_min = SQUARE_OFF_TIME[0] * 60 + SQUARE_OFF_TIME[1]
    
    i = 0
    while i < n - 1:
        sig = signals[i]
        if sig == 0:
            i += 1
            continue
        
        # Check session time
        ts = timestamps[i]
        bar_min = ts.hour * 60 + ts.minute
        if bar_min < no_trade_min or bar_min >= sqoff_min:
            i += 1
            continue
        
        signal_close = closes[i]
        atr_i = atrs[i] if not math.isnan(atrs[i]) else signal_close * 0.005
        direction = sig
        
        # 1-bar fill delay
        fill_bar = i + 1
        fill_open = opens[fill_bar]
        
        # Stale signal check
        if signal_close > 0:
            move_bps = abs(fill_open - signal_close) / signal_close * 10000
            if move_bps > MAX_PRICE_CHASE_BPS:
                i += 1
                continue
        
        # Position sizing
        entry_price = Decimal(str(fill_open))
        max_invest = capital * Decimal(str(MAX_POSITION_PCT))
        qty = max(1, int(max_invest / entry_price))
        
        # Stop loss
        atr_dec = Decimal(str(round(atr_i, 4)))
        stop_mult = Decimal(str(STOP_LOSS_ATR_MULT))
        if direction > 0:
            stop_price = entry_price - atr_dec * stop_mult
        else:
            stop_price = entry_price + atr_dec * stop_mult
        
        # Seek exit
        exit_bar = fill_bar + 1
        exit_reason = "force_close"
        exit_price = Decimal(str(closes[min(fill_bar, n - 1)]))
        
        while exit_bar < n:
            ts_exit = timestamps[exit_bar]
            exit_min = ts_exit.hour * 60 + ts_exit.minute
            
            if exit_min >= sqoff_min:
                exit_price = Decimal(str(opens[exit_bar]))
                exit_reason = "time_exit"
                break
            
            cur_high = highs[exit_bar]
            cur_low = lows[exit_bar]
            cur_open = opens[exit_bar]
            
            if direction > 0 and cur_low <= float(stop_price):
                exit_price = Decimal(str(cur_open))
                exit_reason = "stop"
                break
            if direction < 0 and cur_high >= float(stop_price):
                exit_price = Decimal(str(cur_open))
                exit_reason = "stop"
                break
            
            bars_held = exit_bar - fill_bar
            if bars_held >= MAX_HOLDING_BARS:
                exit_price = Decimal(str(opens[exit_bar]))
                exit_reason = "force_close"
                break
            
            exit_bar += 1
        else:
            exit_price = Decimal(str(closes[n - 1]))
            exit_reason = "force_close"
            exit_bar = n - 1
        
        # P&L
        qty_dec = Decimal(str(qty))
        gross_pnl = direction * (exit_price - entry_price) * qty_dec
        
        # Estimate costs (simplified: ~0.1% round-trip for intraday)
        turnover = entry_price * qty_dec + exit_price * qty_dec
        cost_pct = Decimal("0.001")  # ~10 bps round trip
        total_cost = turnover * cost_pct
        net_pnl = gross_pnl - total_cost
        
        trade = {
            "symbol": symbol,
            "entry_bar": fill_bar,
            "exit_bar": exit_bar,
            "direction": direction,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "quantity": qty,
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "exit_reason": exit_reason,
            "bars_held": exit_bar - fill_bar,
            "entry_ts": str(timestamps[fill_bar]),
            "exit_ts": str(timestamps[exit_bar]) if exit_bar < n else str(timestamps[-1]),
        }
        trades.append(trade)
        equity_curve.append(equity_curve[-1] * (1 + float(net_pnl) / float(capital)))
        
        i = exit_bar + 1
    
    # Compute metrics
    if not trades:
        return {"symbol": symbol, "total_trades": 0, "error": "no trades"}
    
    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    n_trades = len(trades)
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))
    profit_factor = net_profit / net_loss if net_loss > 0 else float("inf")
    
    returns = [p / float(capital) for p in net_pnls]
    mu = statistics.mean(returns) if len(returns) > 1 else 0.0
    sigma = statistics.stdev(returns) if len(returns) > 1 else 0.0
    trades_per_year = BARS_PER_DAY * TRADING_DAYS_PER_YEAR
    annual_factor = math.sqrt(trades_per_year)
    sharpe = (mu / sigma) * annual_factor if sigma > 0 else 0.0
    
    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    
    total_net = sum(net_pnls)
    avg_bars_held = statistics.mean([t["bars_held"] for t in trades])
    
    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1
    
    return {
        "symbol": symbol,
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "total_net_pnl": round(total_net, 2),
        "avg_bars_held": round(avg_bars_held, 1),
        "exit_reasons": dict(exit_reasons),
        "passes_gate": all([
            sharpe >= 1.0,
            profit_factor >= 1.2,
            win_rate >= 0.40,
            n_trades >= 200
        ]),
    }


def find_data_files(base_dir: str, max_files: int = 50) -> list[str]:
    """Find .txt intraday data files, preferring major indices and stocks."""
    targets = ["NIFTY.txt", "BANKNIFTY.txt", "NIFTY_F1.txt", "BANKNIFTY_F1.txt"]
    priority_files = []
    other_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".txt"):
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp)
                # Skip very small files (< 5KB) - likely incomplete data
                if sz < 5000:
                    continue
                if f in targets:
                    priority_files.append(fp)
                elif sz > 100000:  # Larger files have more data
                    other_files.append(fp)
    
    # Sort by file size descending (get the richest datasets first)
    other_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    
    # Return priority files first, then up to max_files total
    result = priority_files + other_files
    return result[:max_files]


def main():
    print("=" * 80)
    print("  NSE AlgoTrader — Real Intraday Data Backtest")
    print("  EMA(9/21) Cross Strategy | Walk-Forward Simulation")
    print("=" * 80)
    print()
    
    # Find data files
    print("Scanning for intraday .txt files...")
    data_files = find_data_files(DATA_DIR, max_files=100)
    print(f"Found {len(data_files)} usable data files")
    print()
    
    # Process NIFTY and BANKNIFTY first (priority)
    all_results = []
    
    for fp in data_files:
        fname = os.path.basename(fp)
        sz_mb = os.path.getsize(fp) / 1024 / 1024
        
        print(f"  Processing {fname} ({sz_mb:.1f} MB)...", end=" ")
        sys.stdout.flush()
        
        # Parse
        df_raw = parse_txt_file(fp)
        if df_raw is None:
            print("SKIP (parse failed)")
            continue
        
        # Print sample of raw data
        n_raw = len(df_raw)
        
        # Resample to 15-min
        df_15min = resample_to_15min(df_raw)
        if df_15min is None or len(df_15min) < 100:
            print(f"SKIP (only {len(df_15min) if df_15min is not None else 0} bars after resample)")
            continue
        
        n_bars = len(df_15min)
        date_range = f"{df_15min['timestamp'][0]} to {df_15min['timestamp'][-1]}"
        
        # Run backtest
        result = run_backtest(df_15min)
        
        if result.get("total_trades", 0) == 0:
            print(f"SKIP (0 trades — {n_bars} bars, {date_range})")
            continue
        
        print(f"{n_bars} bars, {result['total_trades']} trades")
        all_results.append(result)
    
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    if not all_results:
        print("  No backtest results generated.")
        print()
        print("  Possible issues:")
        print("    1. The .txt files may use a different format than expected")
        print("    2. EMA cross may not fire with current params")
        print("    3. Session time filters may be too restrictive")
        print()
        print("  Sample raw data from first file:")
        for fp in data_files[:3]:
            with open(fp, 'r') as f:
                print(f"    {os.path.basename(fp)}: {f.readline().strip()}")
        return
    
    # Aggregate results
    total_trades_all = sum(r["total_trades"] for r in all_results)
    weighted_win_rate = sum(r["win_rate"] * r["total_trades"] for r in all_results) / total_trades_all if total_trades_all > 0 else 0
    weighted_sharpe = sum(r["sharpe_ratio"] * r["total_trades"] for r in all_results) / total_trades_all if total_trades_all > 0 else 0
    total_pnl = sum(r["total_net_pnl"] for r in all_results)
    passing = sum(1 for r in all_results if r.get("passes_gate"))
    
    # Print detailed per-symbol results
    print(f"{'Symbol':<25} {'Trades':<8} {'Win%':<8} {'Sharpe':<10} {'PF':<10} {'PnL':<12} {'MaxDD':<8} {'Gate':<6}")
    print("-" * 87)
    for r in sorted(all_results, key=lambda x: -x["total_trades"]):
        gate = "PASS" if r.get("passes_gate") else "FAIL"
        print(f"{r['symbol']:<25} {r['total_trades']:<8} {r['win_rate']:.1%}  {r['sharpe_ratio']:<8.2f}  {r['profit_factor']:<8.2f}  ₹{r['total_net_pnl']:<+9,.0f}  {r['max_drawdown']:<7.1%}  {gate:<6}")
    
    print("-" * 87)
    print(f"{'TOTAL / WEIGHTED':<25} {total_trades_all:<8} {weighted_win_rate:.1%}  {weighted_sharpe:<8.2f}  {'':<10}  ₹{total_pnl:<+9,.0f}  {'':<8}  {passing}/{len(all_results)}")
    print()
    
    # Edge gate summary
    print("  EDGE GATE ASSESSMENT")
    print(f"    Total symbols processed: {len(all_results)}")
    print(f"    Symbols passing all gates: {passing}")
    print(f"    Sharpe ≥ 1.0: {sum(1 for r in all_results if r['sharpe_ratio'] >= 1.0)}/{len(all_results)}")
    print(f"    Profit Factor ≥ 1.2: {sum(1 for r in all_results if r['profit_factor'] >= 1.2)}/{len(all_results)}")
    print(f"    Win Rate ≥ 40%: {sum(1 for r in all_results if r['win_rate'] >= 0.40)}/{len(all_results)}")
    print(f"    Trades ≥ 200: {sum(1 for r in all_results if r['total_trades'] >= 200)}/{len(all_results)}")
    print()
    
    # Best performers
    print("  TOP 5 PERFORMERS (by Sharpe)")
    for r in sorted(all_results, key=lambda x: -x["sharpe_ratio"])[:5]:
        print(f"    {r['symbol']:<20} Sharpe={r['sharpe_ratio']:.2f}  PF={r['profit_factor']:.2f}  Win={r['win_rate']:.1%}  Trades={r['total_trades']}")
    
    print()
    
    # Exit reason breakdown across all trades
    exit_counts = defaultdict(int)
    for r in all_results:
        for reason, count in r.get("exit_reasons", {}).items():
            exit_counts[reason] += count
    total_exits = sum(exit_counts.values())
    if total_exits > 0:
        print("  EXIT REASON BREAKDOWN (all trades)")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<15}: {count:<6} ({count/total_exits:.1%})")
    
    print()
    
    # Save results
    import json
    output = {
        "generated_at": datetime.now().isoformat(),
        "data_source": "Real intraday .txt files from nifty-banknifty-intraday-data-main",
        "bar_type": "15-minute time bars (resampled from 1-min tick data)",
        "strategy": f"EMA({FAST_EMA}/{SLOW_EMA}) cross",
        "capital_inr": float(INITIAL_CAPITAL),
        "results": all_results,
        "summary": {
            "total_symbols": len(all_results),
            "total_trades": total_trades_all,
            "weighted_win_rate": round(weighted_win_rate, 4),
            "weighted_sharpe": round(weighted_sharpe, 4),
            "total_net_pnl": round(total_pnl, 2),
            "symbols_passing_gate": passing,
        }
    }
    
    output_path = "results/backtest_real_intraday_results.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  Full results saved to: {output_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()