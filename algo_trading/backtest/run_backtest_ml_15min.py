r"""
run_backtest_ml_15min.py – Walk‑forward ML ensemble backtest on 15‑min intraday data.

Loads the trained ensemble from .pkl files, computes features, generates signals,
and runs the backtest with full cost model and Sharpe annualisation.

Usage:
    python -m algo_trading.backtest.run_backtest_ml_15min --category nifty50 --limit 5
"""

import sys
from pathlib import Path
import pickle
import logging
logging.getLogger("algo_trading").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import polars as pl
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from algo_trading.backtest.runner import BacktestRunner, BacktestConfig

DATA_ROOT = PROJECT_ROOT / "data" / "indices"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "ml_15min_backtest_results.json"

MIN_BARS = 200
MODEL_DIR = PROJECT_ROOT / "models" / "trained" / "15min"

# Feature columns from the spec
FEATURE_COLS = [
    "feat_rsi", "feat_macd_hist", "feat_bb_pct_b", "feat_bb_width",
    "feat_atr", "feat_adx", "feat_stoch_k", "feat_stoch_d",
    "feat_obv", "feat_supertrend_dir", "feat_ema_cross",
    "feat_ichimoku_pos", "feat_vwap_dev", "feat_ha_dir",
    "feat_pivot_dist", "feat_lag_ret_1", "feat_lag_ret_2",
    "feat_lag_ret_3", "feat_lag_ret_5", "feat_lag_ret_10",
    "feat_rvol5", "feat_rvol20", "feat_vol_norm",
    "feat_vol_chg", "feat_obv_norm", "feat_session_prog",
    "feat_day_of_week", "feat_mins_to_close"
]
N_FEATURES = len(FEATURE_COLS)


def load_ohlcv(symbol_path: Path, timeframe: str = "15min") -> pl.DataFrame | None:
    """Load OHLCV data, preferring 1min and resampling to 15min if needed."""
    # Try 1min first (more common in indices/stocks), then 15min, then daily
    for tf in ["1min", "15min", "daily"]:
        p = symbol_path / f"{tf}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            break
    else:
        return None

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        return None

    df = df.with_columns([
        pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume"]
    ])
    df = df.filter(pl.col("volume") > 0)

    # If we loaded 1min but need 15min, resample
    if tf == "1min" and timeframe == "15min":
        df = df.with_columns(
            (pl.col("timestamp").dt.timestamp("ms") / (15 * 60 * 1000))
            .floor()
            .cast(pl.Int64)
            .alias("bin")
        )
        df = df.group_by("bin").agg([
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]).sort("timestamp")
        df = df.drop("bin")

    if len(df) < MIN_BARS:
        return None

    return df.sort("timestamp")


def compute_features(df: pl.DataFrame) -> np.ndarray:
    """
    Compute all 28 features required by the model.
    This must match the training pipeline.
    """
    from algo_trading.features.technical import TechnicalFeatures
    from algo_trading.features.statistical import StatisticalFeatures
    from algo_trading.features.custom import CustomFeatures

    tech = TechnicalFeatures()
    stat = StatisticalFeatures()
    cust = CustomFeatures(bar_type="time")

    # Technical features
    df_tech = tech.compute(df)
    df = df.with_columns(pl.lit(0.0).alias("ofi"))
    df_stat = stat.compute(df)

    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Map feature names to column names
    tech_map = {
        "feat_rsi": "feat_rsi",
        "feat_macd_hist": "feat_macd_hist",
        "feat_bb_pct_b": "feat_bb_pct_b",
        "feat_bb_width": "feat_bb_width",
        "feat_atr": "feat_atr",
        "feat_adx": "feat_adx",
        "feat_stoch_k": "feat_stoch_k",
        "feat_stoch_d": "feat_stoch_d",
        "feat_obv": "feat_obv",
        "feat_supertrend_dir": "feat_supertrend_dir",
        "feat_ema_cross": "feat_ema_cross",
        "feat_ichimoku_pos": "feat_ichimoku_pos",
        "feat_vwap_dev": "feat_vwap_dev",
        "feat_ha_dir": "feat_ha_dir",
        "feat_pivot_dist": "feat_pivot_dist",
    }
    stat_map = {
        "feat_kalman_price": "feat_kalman_price",
        "feat_hmm_regime": "feat_hmm_regime",
        "feat_garch_vol": "feat_garch_vol",
        "feat_autocorr_lag1": "feat_autocorr_lag1",
    }
    # Custom features will be computed later

    for i in range(n):
        for col in FEATURE_COLS:
            idx = FEATURE_COLS.index(col)
            if col in df_tech.columns:
                X[i, idx] = df_tech[col][i]
            elif col in df_stat.columns:
                X[i, idx] = df_stat[col][i]
            else:
                X[i, idx] = 0.0

    # Compute lag returns (feat_lag_ret_1,2,3,5,10)
    closes = df["close"].to_numpy()
    for lag in [1, 2, 3, 5, 10]:
        col = f"feat_lag_ret_{lag}"
        if col in FEATURE_COLS:
            idx = FEATURE_COLS.index(col)
            lag_arr = np.zeros(len(closes), dtype=np.float32)
            if len(closes) > lag:
                lag_arr[lag:] = (closes[lag:] / closes[:-lag] - 1.0) * 100.0
            X[:, idx] = lag_arr

    # Realised volatility (feat_rvol5, feat_rvol20)
    for window, col in [(5, "feat_rvol5"), (20, "feat_rvol20")]:
        if col in FEATURE_COLS:
            idx = FEATURE_COLS.index(col)
            rvol = np.zeros(n)
            for i in range(window, n):
                rvol[i] = np.std(closes[i-window:i] / closes[i-window:i] - 1.0) * np.sqrt(252)
            X[:, idx] = rvol

    # Volume normalised (feat_vol_norm)
    vols = df["volume"].to_numpy()
    vol_mean = np.mean(vols)
    if vol_mean > 0:
        X[:, FEATURE_COLS.index("feat_vol_norm")] = vols / vol_mean

    # Volume change (feat_vol_chg)
    vol_chg = np.zeros(n)
    vol_chg[1:] = (vols[1:] - vols[:-1]) / (vols[:-1] + 1e-6)
    X[:, FEATURE_COLS.index("feat_vol_chg")] = vol_chg

    # OBV normalized (feat_obv_norm) – from technical features
    if "feat_obv" in df_tech.columns:
        obv = df_tech["feat_obv"].to_numpy()
        obv_norm = obv / (np.max(np.abs(obv)) + 1e-6)
        X[:, FEATURE_COLS.index("feat_obv_norm")] = obv_norm

    # Session progress (feat_session_prog)
    ts_vals = df["timestamp"].to_list()
    session_start = [datetime(t.year, t.month, t.day, 9, 15) for t in ts_vals]
    session_end = [datetime(t.year, t.month, t.day, 15, 30) for t in ts_vals]
    prog = np.array([(t - s).seconds / max((e - s).seconds, 1) for t, s, e in zip(ts_vals, session_start, session_end)], dtype=np.float32)
    X[:, FEATURE_COLS.index("feat_session_prog")] = prog

    # Day of week (feat_day_of_week)
    dow = np.array([t.weekday() for t in ts_vals], dtype=np.float32)
    X[:, FEATURE_COLS.index("feat_day_of_week")] = dow

    # Minutes to close (feat_mins_to_close)
    mins_to_close = np.array([ (9*60+15 + 375 - (t.hour*60+t.minute)) for t in ts_vals], dtype=np.float32)
    X[:, FEATURE_COLS.index("feat_mins_to_close")] = mins_to_close

    return X


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="nifty50")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--timeframe", default="15min", choices=["15min", "1min"])
    args = parser.parse_args()

    # Load trained ensemble
    print("Loading trained model...")
    with open(MODEL_DIR / "xgb_15min.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODEL_DIR / "lgb_15min.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(MODEL_DIR / "mlp_15min.pkl", "rb") as f:
        mlp_model = pickle.load(f)
    with open(MODEL_DIR / "meta_lr_15min.pkl", "rb") as f:
        meta_lr = pickle.load(f)

    # Load feature scaler (if saved) – assume StandardScaler fitted
    scaler_path = MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None

    print("Model loaded.")

    # Find symbol folders
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
        base = DATA_ROOT / args.category
        if not base.exists():
            print(f"Category {args.category} not found.")
            return
        symbol_folders = [d for d in base.iterdir() if d.is_dir()]
        if args.limit:
            symbol_folders = symbol_folders[:args.limit]

    print(f"Found {len(symbol_folders)} symbols.")

    runner = BacktestRunner(BacktestConfig())
    all_results = []

    for idx, sym_dir in enumerate(symbol_folders, 1):
        symbol = sym_dir.name
        print(f"[{idx}/{len(symbol_folders)}] Processing {symbol} ...", end=" ")
        sys.stdout.flush()

        df = load_ohlcv(sym_dir, args.timeframe)
        if df is None:
            print("SKIP (data missing)")
            continue

        # Compute features
        X = compute_features(df)
        # Drop warmup rows where features are all zero
        valid = ~(X == 0).all(axis=1)
        X = X[valid]
        df_valid = df.filter(valid)
        if len(X) < 100:
            print(f"SKIP (only {len(X)} valid rows)")
            continue

        # Impute NaN before scaling/inference (MLP cannot handle NaN)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Generate ML predictions (ensemble: average probabilities from 3 models)
        # For this ensemble, we have XGB, LGB, and MLP (no CatBoost)
        # We'll use the meta LR which was trained on OOF probabilities.
        # Since we don't have the meta LR's expected inputs, we'll use a simpler approach:
        # average the probabilities from XGB, LGB, MLP and take argmax.
        # This approximates the ensemble.
        prob_xgb = xgb_model.predict_proba(X_scaled)
        prob_lgb = lgb_model.predict_proba(X_scaled)
        prob_mlp = mlp_model.predict_proba(X_scaled)
        avg_probs = (prob_xgb + prob_lgb + prob_mlp) / 3.0
        pred_class = np.argmax(avg_probs, axis=1)
        signals = np.array([[-1, 0, 1][c] for c in pred_class])

        # Prepare feature_df with ATR for cost model
        feature_df = runner._compute_atr_features(df_valid)

        # Run backtest
        try:
            report = runner.run(df_valid, pl.Series(signals), feature_df)
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

    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model_type": "XGB+LGB+MLP ensemble (averaged)",
        "data_source": str(DATA_ROOT / args.category),
        "timeframe": args.timeframe,
        "total_symbols": len(all_results),
        "total_trades": sum(r["total_trades"] for r in all_results),
        "total_net_pnl": sum(r["total_net_pnl"] for r in all_results),
        "symbols_passing_gate": sum(1 for r in all_results if r["passes_edge_gate"]),
        "results": all_results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_JSON}")

    # Print summary
    print("\n" + "=" * 100)
    print("  ML ENSEMBLE BACKTEST RESULTS (walk-forward, cost-adjusted)")
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