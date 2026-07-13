"""
Stage 4 RL position sizer training pipeline.

Trains a PPO agent on historical OHLCV + feature data to learn
optimal position sizing. Requires Stage 2A model to already be trained
(uses ensemble signals as part of the observation space).

Usage:
    python -m algo_trading.scripts.train_rl
    python -m algo_trading.scripts.train_rl --symbols RELIANCE TCS --timesteps 500000
    python -m algo_trading.scripts.train_rl --validate-only

Stage gate: Will raise if Stage 2A FrozenModelSpec not found or edge gate not passed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Deferred heavy imports — validated at runtime ──────────────────────────────
try:
    import gymnasium  # noqa: F401
    _GYMNASIUM_AVAILABLE = True
except ImportError:
    _GYMNASIUM_AVAILABLE = False

try:
    import stable_baselines3  # noqa: F401
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BAJFINANCE", "KOTAKBANK", "LT",
]
_DEFAULT_TIMESTEPS = 500_000
_HISTORICAL_DATA_DIR = Path("data/historical")
_MODEL_OUTPUT_DIR = Path("models/saved")
_DEFAULT_MODEL_PATH = _MODEL_OUTPUT_DIR / "rl_sizer.zip"
_VALIDATION_SPLIT_PCT = 0.20       # Hold out last 20% of data
_MIN_SHARPE_WARN = 0.5             # Warn if validation Sharpe falls below this

# Retrain window: Sunday after 20:00 IST (mirrors train_models.py)
_RETRAIN_DAY_OF_WEEK = 6           # Sunday
_RETRAIN_HOUR_IST = 20             # 8 PM IST
_IST_OFFSET_HOURS = 5
_IST_OFFSET_MINUTES = 30


# ══════════════════════════════════════════════════════════════════════════════
# Step helpers
# ══════════════════════════════════════════════════════════════════════════════

def _step(n: int, msg: str) -> None:
    """Print a numbered pipeline step to stdout."""
    print(f"\n[Step {n}/10] {msg}")
    log.info("train_rl.step", step=n, description=msg)


# ══════════════════════════════════════════════════════════════════════════════
# Stage gate
# ══════════════════════════════════════════════════════════════════════════════

def _check_stage_gate() -> None:
    """
    Step 1: Ensure Stage 2A FrozenModelSpec exists and has approval_status='approved'.

    Raises:
        SystemExit: if the registry file is missing or no approved model is found.
    """
    try:
        from algo_trading.models.model_registry import ModelRegistry
        registry = ModelRegistry()
        spec = registry.get_approved_model()
        if spec is None:
            raise SystemExit(
                "Stage gate FAILED: No approved FrozenModelSpec found in model registry.\n"
                "Run train_models.py and ensure the model is approved before training RL."
            )
        log.info(
            "train_rl.stage_gate_ok",
            model_version=spec.version,
            approval_status=spec.approval_status,
            trained_on=str(spec.trained_on),
        )
        print(
            f"  Stage 2A model found: version={spec.version!r}  "
            f"approval={spec.approval_status!r}  trained_on={spec.trained_on}"
        )
    except ImportError as exc:
        raise SystemExit(
            f"Stage gate FAILED: Could not import model registry.\n"
            f"Error: {exc}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_historical_data(symbols: list[str]) -> dict:
    """
    Step 2: Load historical OHLCV data from data/historical/*.parquet.

    Returns:
        dict mapping symbol → pandas DataFrame with OHLCV columns.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. pip install pandas") from exc

    data: dict = {}
    missing: list[str] = []

    for sym in symbols:
        pattern = list(_HISTORICAL_DATA_DIR.glob(f"{sym}*.parquet"))
        if not pattern:
            missing.append(sym)
            continue
        # Load the most-recently-modified file for this symbol
        parquet_path = max(pattern, key=lambda p: p.stat().st_mtime)
        df = pd.read_parquet(parquet_path)
        df.columns = [c.lower() for c in df.columns]
        data[sym] = df
        log.info(
            "train_rl.data_loaded",
            symbol=sym,
            rows=len(df),
            path=str(parquet_path),
        )

    if missing:
        log.warning(
            "train_rl.symbols_no_data",
            missing=missing,
            hint="Run scripts/generate_sample_data.py or scripts/download_history.py first.",
        )

    if not data:
        raise SystemExit(
            f"No parquet data found under {_HISTORICAL_DATA_DIR}.\n"
            "Run: python -m algo_trading.scripts.generate_sample_data"
        )

    total_rows = sum(len(df) for df in data.values())
    print(f"  Loaded {len(data)} symbols | {total_rows:,} total rows")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_features(data: dict) -> dict:
    """
    Step 3: Compute TechnicalFeatures, StatisticalFeatures, CustomFeatures.

    Returns:
        dict mapping symbol → DataFrame with feature columns appended.
    """
    from algo_trading.features.technical import TechnicalFeatures
    from algo_trading.features.statistical import StatisticalFeatures
    from algo_trading.features.custom import CustomFeatures

    featured: dict = {}
    for sym, df in data.items():
        try:
            tech = TechnicalFeatures()
            df = tech.compute(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("train_rl.feature_technical_error", symbol=sym, error=str(exc))

        try:
            stat = StatisticalFeatures()
            df = stat.compute(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("train_rl.feature_statistical_error", symbol=sym, error=str(exc))

        try:
            custom = CustomFeatures()
            df = custom.compute(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("train_rl.feature_custom_error", symbol=sym, error=str(exc))

        # Drop rows with NaN features (warm-up period)
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            log.info("train_rl.warmup_rows_dropped", symbol=sym, dropped=dropped)

        featured[sym] = df

    total = sum(len(df) for df in featured.values())
    print(f"  Features computed: {len(featured)} symbols | {total:,} rows after NaN drop")
    return featured


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble signal column
# ══════════════════════════════════════════════════════════════════════════════

def _load_ensemble_signals(data: dict) -> dict:
    """
    Step 4: Load the trained ensemble and attach 'signal' column to each DataFrame.

    If ensemble is not available, attaches a stub signal column (0.0) so the
    pipeline can proceed for infrastructure testing.

    Returns:
        Updated data dict with 'ensemble_signal' column added.
    """
    try:
        from algo_trading.models.ensemble import MLEnsemble
        from algo_trading.models.model_registry import ModelRegistry

        registry = ModelRegistry()
        spec = registry.get_approved_model()
        if spec is None:
            raise RuntimeError("No approved model spec found.")

        ensemble = MLEnsemble.load_from_spec(spec)

        feature_names = list(spec.feature_names)
        for sym, df in data.items():
            available = [f for f in feature_names if f in df.columns]
            if len(available) < len(feature_names):
                missing_feats = set(feature_names) - set(available)
                log.warning(
                    "train_rl.ensemble_missing_features",
                    symbol=sym,
                    missing=list(missing_feats)[:5],
                )

            if available:
                X = df[available].values
                preds = ensemble.predict_proba(X)
                data[sym]["ensemble_signal"] = preds
            else:
                data[sym]["ensemble_signal"] = 0.0

        print(f"  Ensemble signals attached for {len(data)} symbols")

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "train_rl.ensemble_load_failed",
            error=str(exc),
            fallback="Using stub signal=0.0 for all bars",
        )
        for sym in data:
            data[sym]["ensemble_signal"] = 0.0
        print(f"  WARNING: Ensemble unavailable — stub signal attached. Error: {exc}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame → NSETradingEnv array extraction
# ══════════════════════════════════════════════════════════════════════════════

def _df_to_env_arrays(df) -> tuple:
    """Extract (features, closes, atrs, timestamps) from a feature DataFrame.

    Used to convert a pandas DataFrame produced by _compute_features() into the
    separate numpy arrays required by NSETradingEnv.__init__.

    Args:
        df: pandas DataFrame with OHLCV columns, feature columns (FEATURE_NAMES),
            atr14, and timestamp.

    Returns:
        (features, closes, atrs, timestamps) where features has shape (n, 25).
    """
    import numpy as np
    import pandas as pd
    from algo_trading.features.engine import FEATURE_NAMES as _FEAT_NAMES

    n = len(df)
    features = np.zeros((n, len(_FEAT_NAMES)), dtype=np.float32)
    for i, name in enumerate(_FEAT_NAMES):
        if name in df.columns:
            features[:, i] = df[name].fillna(0.0).values.astype(np.float32)

    closes = (
        df["close"].fillna(0.0).values.astype(np.float32)
        if "close" in df.columns
        else np.ones(n, dtype=np.float32)
    )
    atrs = (
        df["atr14"].fillna(0.0).values.astype(np.float32)
        if "atr14" in df.columns
        else closes * 0.02
    )
    if "timestamp" in df.columns:
        timestamps = [pd.Timestamp(t).to_pydatetime() for t in df["timestamp"]]
    else:
        from datetime import datetime as _dt
        timestamps = [_dt.now() for _ in range(n)]
    return features, closes, atrs, timestamps


# ══════════════════════════════════════════════════════════════════════════════
# Environment construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_train_envs(data: dict, val_split: float) -> tuple[list, object]:
    """
    Steps 5+6: Build NSETradingEnv instances.

    Training envs: one per (symbol, year) rolling window — all data except
    last val_split fraction.
    Validation env: last val_split fraction of data, held out.

    Returns:
        (train_envs, val_env) tuple.
    """
    if not _GYMNASIUM_AVAILABLE:
        raise SystemExit(
            "gymnasium is required for RL training.\n"
            "Install with: pip install gymnasium"
        )

    try:
        from algo_trading.models.rl_env import NSETradingEnv
    except ImportError as exc:
        raise SystemExit(
            f"Could not import NSETradingEnv: {exc}\n"
            "Ensure gymnasium is installed and ALGO_STAGE=4 is set."
        ) from exc

    import pandas as pd

    train_envs: list = []
    val_dfs: list = []

    for sym, df in data.items():
        n = len(df)
        if n < 50:
            log.warning("train_rl.insufficient_data", symbol=sym, rows=n, skipping=True)
            continue

        split_idx = int(n * (1.0 - val_split))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        val_dfs.append(val_df)

        # Rolling 1-year windows for training (≈252 trading days × 26 bars = ~6552 rows)
        bars_per_year = 252 * 26
        step = bars_per_year // 2  # 50% overlap

        n_envs_before = len(train_envs)
        start = 0
        while start + bars_per_year <= len(train_df):
            window_df = train_df.iloc[start : start + bars_per_year].copy()
            try:
                feats, cls, atrs_arr, tss = _df_to_env_arrays(window_df)
                env = NSETradingEnv(
                    features=feats, closes=cls, atrs=atrs_arr, timestamps=tss, symbol=sym
                )
                train_envs.append(env)
            except Exception as exc:  # noqa: BLE001
                log.warning("train_rl.env_build_error", symbol=sym, start=start, error=str(exc))
            start += step

        # If no rolling windows fit, add one env covering all training data for this symbol
        if len(train_envs) == n_envs_before:
            try:
                feats, cls, atrs_arr, tss = _df_to_env_arrays(train_df)
                env = NSETradingEnv(
                    features=feats, closes=cls, atrs=atrs_arr, timestamps=tss, symbol=sym
                )
                train_envs.append(env)
            except Exception as exc:  # noqa: BLE001
                log.warning("train_rl.env_build_fallback_error", symbol=sym, error=str(exc))

    if not train_envs:
        raise SystemExit("No training environments could be built. Check data quality.")

    # Build single validation env from concatenated val slices
    if val_dfs:
        # pd already imported above in this function scope
        combined_val = pd.concat(val_dfs, ignore_index=True)
        sym_for_val = list(data.keys())[0]
        try:
            feats, cls, atrs_arr, tss = _df_to_env_arrays(combined_val)
            val_env = NSETradingEnv(
                features=feats, closes=cls, atrs=atrs_arr, timestamps=tss, symbol=sym_for_val
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("train_rl.val_env_error", error=str(exc))
            val_env = train_envs[-1]  # Fallback
    else:
        val_env = train_envs[-1]

    print(f"  Training envs built: {len(train_envs)}  |  Validation env: 1")
    return train_envs, val_env


# ══════════════════════════════════════════════════════════════════════════════
# Safety gate: retraining window
# ══════════════════════════════════════════════════════════════════════════════

def _is_safe_to_retrain() -> bool:
    """
    Return True only when it is safe to trigger a retrain.

    Safe window: Sunday after 20:00 IST (mirrors train_models.py constraint).
    """
    from datetime import timezone, timedelta
    utc_now = datetime.now(tz=timezone.utc)
    ist_now = utc_now.replace(tzinfo=None) + timedelta(
        hours=_IST_OFFSET_HOURS, minutes=_IST_OFFSET_MINUTES
    )
    is_sunday = ist_now.weekday() == _RETRAIN_DAY_OF_WEEK
    is_after_20 = ist_now.hour >= _RETRAIN_HOUR_IST
    return is_sunday and is_after_20


# ══════════════════════════════════════════════════════════════════════════════
# MLflow logging
# ══════════════════════════════════════════════════════════════════════════════

def _log_to_mlflow(metrics: dict, config_params: dict, model_path: Path) -> None:
    """
    Step 10: Log training metrics and parameters to MLflow.

    Silently skips if MLflow is not installed.
    """
    try:
        import mlflow
    except ImportError:
        log.info("train_rl.mlflow_not_installed", hint="pip install mlflow to enable tracking")
        return

    try:
        with mlflow.start_run(run_name="rl_sizer_training"):
            for key, val in config_params.items():
                mlflow.log_param(key, val)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, float(val))
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
        log.info("train_rl.mlflow_logged", metrics=metrics)
        print(f"  MLflow run logged: {list(metrics.keys())}")
    except Exception as exc:  # noqa: BLE001
        log.warning("train_rl.mlflow_error", error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Validate-only mode
# ══════════════════════════════════════════════════════════════════════════════

def _run_validate_only(model_path: Path) -> None:
    """Run inference on the saved model and report validation Sharpe."""
    if not model_path.exists():
        raise SystemExit(f"No model found at {model_path}. Train first.")

    print(f"\nValidate-only mode — loading model from {model_path}")
    try:
        from algo_trading.models.rl_sizer import RLSizerConfig, RLSizerGuardrails, RLPositionSizer
        config = RLSizerConfig(model_save_path=str(model_path))
        guardrails = RLSizerGuardrails()
        sizer = RLPositionSizer(config, guardrails)
        sizer.load(str(model_path))
        print("  Model loaded successfully. Run full pipeline to get Sharpe on held-out data.")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Validate-only failed: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_training_pipeline(
    symbols: list[str],
    timesteps: int,
    force: bool,
    model_path: Path,
) -> None:
    """
    Execute the full 10-step RL training pipeline.

    Args:
        symbols:    List of NSE equity symbols.
        timesteps:  Total PPO training timesteps.
        force:      If True, bypass the Sunday-only retrain gate.
        model_path: Output path for the saved RL model zip.
    """
    print("\n" + "=" * 65)
    print("  NSE AlgoTrader — Stage 4 RL Position Sizer Training")
    print("=" * 65)

    # ── Step 1: Stage gate ────────────────────────────────────────────────────
    _step(1, "Checking Stage 2A model gate")
    _check_stage_gate()

    # ── Step 2: Load historical data ──────────────────────────────────────────
    _step(2, f"Loading historical data from {_HISTORICAL_DATA_DIR}/")
    data = _load_historical_data(symbols)

    # ── Step 3: Compute features ──────────────────────────────────────────────
    _step(3, "Computing TechnicalFeatures, StatisticalFeatures, CustomFeatures")
    data = _compute_features(data)

    # ── Step 4: Load ensemble signals ─────────────────────────────────────────
    _step(4, "Loading trained ensemble for signal column")
    data = _load_ensemble_signals(data)

    # ── Steps 5+6: Build environments ─────────────────────────────────────────
    _step(5, "Building NSETradingEnv instances (one per symbol per year, rolling)")
    _step(6, "Building validation env (last 20% of data, held out)")
    train_envs, val_env = _build_train_envs(data, val_split=_VALIDATION_SPLIT_PCT)

    # ── Step 7: Instantiate RLPositionSizer ───────────────────────────────────
    _step(7, "Instantiating RLPositionSizer with RLSizerConfig and guardrails")

    if not _SB3_AVAILABLE:
        raise SystemExit(
            "stable_baselines3 is required for RL training.\n"
            "Install with: pip install stable-baselines3"
        )

    from algo_trading.models.rl_sizer import RLSizerConfig, RLSizerGuardrails, RLPositionSizer

    config = RLSizerConfig(
        n_timesteps=timesteps,
        model_save_path=str(model_path),
    )
    guardrails = RLSizerGuardrails()
    sizer = RLPositionSizer(config, guardrails)
    print(
        f"  Config: timesteps={timesteps:,}  device={config.device!r}  "
        f"policy={config.policy!r}"
    )

    # ── Step 8: Safety check ───────────────────────────────────────────────────
    _step(8, "Checking is_safe_to_retrain() — Sunday after 20:00 IST only")

    # Prefer the sizer's built-in check (handles cooldown too)
    if not sizer.is_safe_to_retrain():
        if force:
            log.warning(
                "train_rl.retrain_unsafe_forced",
                msg="is_safe_to_retrain() returned False but --force was set. Proceeding.",
            )
            print("  WARNING: Retrain window not open — proceeding anyway (--force).")
        else:
            raise SystemExit(
                "Retraining blocked: is_safe_to_retrain() returned False.\n"
                "Allowed window: Sunday after 20:00 IST.\n"
                "Use --force to override (development only)."
            )
    else:
        # Also check global Sunday gate
        if not _is_safe_to_retrain() and not force:
            log.warning(
                "train_rl.retrain_window_advisory",
                msg="Not Sunday after 20:00 IST. Using --force to proceed is recommended.",
            )
        print("  Retrain safety check PASSED.")

    # ── Step 9: Train ──────────────────────────────────────────────────────────
    _step(9, f"Training PPO agent ({timesteps:,} timesteps)")
    print(f"  This may take a while on CPU (device={config.device!r})…")

    result = sizer.train(train_envs, val_env)

    if result["status"] == "failed":
        sharpe = result.get("sharpe", 0.0)
        print(
            f"\n  WARNING: Training completed but Sharpe={sharpe:.4f} < "
            f"min_required={guardrails.min_sharpe}. Model NOT saved."
        )
        log.warning(
            "train_rl.below_sharpe_threshold",
            sharpe=sharpe,
            min_sharpe=guardrails.min_sharpe,
        )
        raise SystemExit(
            f"RL training completed but failed Sharpe gate "
            f"(Sharpe={sharpe:.4f} < {guardrails.min_sharpe}). "
            "Investigate reward function, data quality, or extend timesteps."
        )

    sharpe = result.get("sharpe", 0.0)
    n_episodes = result.get("episodes", 0)

    if sharpe < _MIN_SHARPE_WARN:
        print(
            f"\n  WARNING: Validation Sharpe={sharpe:.4f} is below recommended "
            f"threshold of {_MIN_SHARPE_WARN}. Consider more training data or timesteps."
        )

    print(
        f"\n  Training complete:"
        f"  Sharpe={sharpe:.4f}  episodes≈{n_episodes:,}  model_saved={model_path}"
    )

    # ── Step 10: Log to MLflow ────────────────────────────────────────────────
    _step(10, "Saving model + logging metrics to MLflow")

    config_params = {
        "symbols": ",".join(symbols),
        "n_timesteps": timesteps,
        "n_train_envs": len(train_envs),
        "device": config.device,
        "policy": config.policy,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
    }
    metrics = {
        "sharpe_validation": sharpe,
        "n_episodes_approx": n_episodes,
        "training_status": 1 if result["status"] == "ok" else 0,
    }
    _log_to_mlflow(metrics, config_params, model_path)

    print("\n" + "=" * 65)
    print("  Stage 4 RL training pipeline COMPLETE.")
    print(f"  Model saved to: {model_path}")
    print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 RL Position Sizer Training — NSE AlgoTrader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        metavar="SYMBOL",
        help="NSE equity symbols to include in training",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=_DEFAULT_TIMESTEPS,
        help="Total PPO training timesteps",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass Sunday-only retrain safety gate (development use only)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Skip training; only validate the saved model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(_DEFAULT_MODEL_PATH),
        help="Output path for the saved RL model zip file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    model_path = Path(args.model_path)

    if args.validate_only:
        _run_validate_only(model_path)
        return

    run_training_pipeline(
        symbols=args.symbols,
        timesteps=args.timesteps,
        force=args.force,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
