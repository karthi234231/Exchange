r"""
run_backtest_daily.py – Daily‑bar walk‑forward backtest on all Parquet data.
Quiet mode: only prints final summary.
"""

import sys
from pathlib import Path

# --- Suppress verbose backtest logs ---
import logging
logging.getLogger("algo_trading").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime

import polars as pl
from algo_trading.backtest.runner import BacktestRunner, BacktestConfig

DATA_ROOT = PROJECT_ROOT / "data" / "indices"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "daily_backtest_results.json"

MIN_BARS = 100


def find_symbol_folders(category: str | None = None) -> list[Path]:
    if category:
        base = DATA_ROOT / category
        if not base.exists():
            print(f"Category folder not found: {base}")
            return []
        return [d for d in base.iterdir() if d.is_dir()]
    else:
        folders = []
        for cat_dir in DATA_ROOT.iterdir():
            if cat_dir.is_dir():
                folders.extend([d for d in cat_dir.iterdir() if d.is_dir()])
        return folders


def load_ohlcv(symbol_path: Path) -> pl.DataFrame | None:
    """Load daily.parquet and set timestamp to 09:30 IST."""
    p = symbol_path / "daily.parquet"
    if not p.exists():
        print(f"  No daily.parquet for {symbol_path.name}")
        return None

    df = pl.read_parquet(p)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        print(f"  Missing columns in {symbol_path.name}: {missing}")
        return None

    for col in ["open", "high", "low", "close", "volume"]:
        df = df.with_columns(pl.col(col).cast(pl.Float64))

    df = df.filter(pl.col("volume") > 0)

    if len(df) < MIN_BARS:
        print(f"  Only {len(df)} bars for {symbol_path.name} – need {MIN_BARS}")
        return None

    # ---- FIX: set time to 09:30 IST ----
    df = df.with_columns(
        (pl.col("timestamp").dt.date().cast(pl.Datetime) + pl.duration(hours=9, minutes=30)).alias("timestamp")
    )

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="Limit to a category (e.g., nifty50)")
    parser.add_argument("--symbol", help="Process a single symbol")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    args = parser.parse_args()

    if args.symbol:
        sym_path = None
        for root in DATA_ROOT.iterdir():
            if root.is_dir():
                candidate = root / args.symbol
                if candidate.exists() and candidate.is_dir():
                    sym_path = candidate
                    break
        if not sym_path:
            print(f"Symbol {args.symbol} not found.")
            return
        symbol_folders = [sym_path]
    else:
        symbol_folders = find_symbol_folders(args.category)
        if args.limit:
            symbol_folders = symbol_folders[:args.limit]

    print(f"Found {len(symbol_folders)} symbol folders to process (daily data).")

    config = BacktestConfig(
        initial_capital=500000,
        max_position_pct=0.10,
        stop_loss_atr_mult=2.0,
        max_holding_bars=40,
        no_trade_before="09:30",
        square_off_time="15:15",
        adverse_selection_bps=5.0,
        max_price_chase_bps=50.0,
    )
    runner = BacktestRunner(config)
    all_results = []

    for idx, sym_dir in enumerate(symbol_folders, 1):
        symbol = sym_dir.name
        print(f"[{idx}/{len(symbol_folders)}] Processing {symbol} ...", end=" ")
        sys.stdout.flush()

        df = load_ohlcv(sym_dir)
        if df is None:
            print("SKIP")
            continue

        try:
            report = runner.run_ema_baseline(df, fast=9, slow=21)
            result = {
                "symbol": symbol,
                "total_trades": report.total_trades,
                "win_rate": round(report.win_rate, 4),
                "profit_factor": round(report.profit_factor, 4),
                "sharpe_ratio": round(report.sharpe_ratio, 4),
                "max_drawdown": round(report.max_drawdown, 4),
                "total_net_pnl": float(report.total_net_pnl),
                "passes_edge_gate": report.passes_edge_gate,
                "gate_failures": report.gate_failures,
            }
            all_results.append(result)
            gate = "PASS" if report.passes_edge_gate else "FAIL"
            print(f"{report.total_trades} trades, Sharpe={report.sharpe_ratio:.2f}, PF={report.profit_factor:.2f}, {gate}")
        except Exception as e:
            print(f"ERROR: {e}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "data_source": str(DATA_ROOT),
        "strategy": "EMA(9/21) cross baseline (daily bars)",
        "backtest_engine": "BacktestRunner (walk‑forward, cost model, edge gates)",
        "total_symbols": len(all_results),
        "total_trades": sum(r["total_trades"] for r in all_results),
        "total_net_pnl": sum(r["total_net_pnl"] for r in all_results),
        "symbols_passing_gate": sum(1 for r in all_results if r["passes_edge_gate"]),
        "results": all_results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_JSON}")

    print("\n" + "=" * 100)
    print("  DAILY BACKTEST RESULTS (walk-forward, cost-adjusted)")
    print("=" * 100)
    if not all_results:
        print("  No results.")
        return

    print(f"\n{'Symbol':<20} {'Trades':>8} {'Win%':>8} {'Sharpe':>10} {'PF':>10} {'MaxDD':>10} {'PnL':>14} {'Gate':>8}")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: -x["sharpe_ratio"]):
        print(f"{r['symbol']:<20} {r['total_trades']:>8} {r['win_rate']:>7.1%} {r['sharpe_ratio']:>9.2f} {r['profit_factor']:>9.2f} {r['max_drawdown']:>9.1%} {r['total_net_pnl']:>13,.0f} {'PASS' if r['passes_edge_gate'] else 'FAIL':>8}")

    print("-" * 100)
    print(f"{'TOTAL / AVG':<20} {summary['total_trades']:>8} {'':>8} {'':>10} {'':>10} {'':>10} {summary['total_net_pnl']:>13,.0f} {summary['symbols_passing_gate']}/{len(all_results)}")
    print()


if __name__ == "__main__":
    main()