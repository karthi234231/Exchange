"""
Import smoke-test — verifies all algo_trading modules import without crashing.

Run before any other script to confirm the codebase is intact:
    python scripts/check_imports.py

Exit code 0 = all required imports OK.
Exit code 1 = one or more required imports failed.

Optional dependencies (gymnasium, stable_baselines3, transformers, hmmlearn,
arch, filterpy, numba) are tested separately and reported as warnings, not errors.
"""
from __future__ import annotations

import importlib
import os
import sys

# ── Set ALGO_STAGE=4 so Stage 4 modules don't raise on import ────────────────
os.environ.setdefault("ALGO_STAGE", "4")

# ── Required modules (must import cleanly) ────────────────────────────────────
REQUIRED = [
    "algo_trading.config",
    "algo_trading.market_calendar",
    "algo_trading.logging_setup",
    # Data
    "algo_trading.data.broker_base",
    "algo_trading.data.broker_shoonya",
    "algo_trading.data.candle_store",
    "algo_trading.data.ring_buffer",
    "algo_trading.data.integrity_guard",
    "algo_trading.data.system_state",
    "algo_trading.data.websocket_feed",
    "algo_trading.data.tick_aggregator",
    # Risk
    "algo_trading.risk.cost_model",
    "algo_trading.risk.manager",
    "algo_trading.risk.position_sizer",
    # Execution
    "algo_trading.execution.order_fsm",
    "algo_trading.execution.order_executor",
    "algo_trading.execution.rate_limiter",
    "algo_trading.execution.signal_hierarchy",
    "algo_trading.execution.latency_profiler",
    "algo_trading.execution.trade_logger",
    # Signals
    "algo_trading.signals.ema_cross",
    # Alerts
    "algo_trading.alerts.telegram_bot",
    "algo_trading.alerts.alert_sink",
    # Features
    "algo_trading.features.technical",
    "algo_trading.features.engine",
    "algo_trading.features.microstructure",
    "algo_trading.features.hawkes",
    "algo_trading.features.tfi",
    "algo_trading.features.stage_router",
    # Labeling
    "algo_trading.labeling.triple_barrier",
    "algo_trading.labeling.walk_forward",
    # Models
    "algo_trading.models.model_registry",
    "algo_trading.models.ensemble",
    "algo_trading.models.inference_engine",
    "algo_trading.models.online_model",
    "algo_trading.models.tuner",
    "algo_trading.models.tracker",
    "algo_trading.models.explainer",
    "algo_trading.models.sentiment",
    # Backtest
    "algo_trading.backtest.runner",
    # Orchestrator
    "algo_trading.app.daily_orchestrator",
    # Scripts
    "algo_trading.scripts.download_history",
    "algo_trading.scripts.validate_edge",
    "algo_trading.scripts.train_models",
    "algo_trading.scripts.generate_sample_data",
    "algo_trading.scripts.paper_trade",
    "algo_trading.scripts.download_fo_history",
]

# ── Optional modules (missing = warning only) ─────────────────────────────────
OPTIONAL = [
    ("algo_trading.data.truedata_feed",       "truedata-ws not installed — Stage 2B feed disabled"),
    ("algo_trading.models.rl_env",            "gymnasium not installed — Stage 4 RL env disabled"),
    ("algo_trading.models.rl_sizer",          "stable-baselines3 not installed — Stage 4 RL sizer disabled"),
    ("algo_trading.data.fo_chain",            "scipy not installed — F&O chain/BS Greeks disabled"),
    ("algo_trading.features.fo_features",     "scipy not installed — F&O features disabled"),
    ("algo_trading.features.statistical",     "hmmlearn/arch/filterpy not installed — statistical features limited"),
    ("algo_trading.features.custom",          "numba not installed — OBI will use pure-Python fallback"),
    ("algo_trading.scripts.train_rl",         "gymnasium/SB3 not installed — RL training disabled"),
]


def _check(module: str) -> tuple[bool, str]:
    """Try to import module. Returns (success, error_message)."""
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 60)
    print("NSE AlgoTrader — Import Smoke Test")
    print("=" * 60)

    failures: list[tuple[str, str]] = []
    warnings: list[tuple[str, str, str]] = []

    # ── Required ──────────────────────────────────────────────────────────────
    print("\n[REQUIRED MODULES]")
    for mod in REQUIRED:
        ok, err = _check(mod)
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {mod}" + (f"\n        → {err}" if not ok else ""))
        if not ok:
            failures.append((mod, err))

    # ── Optional ──────────────────────────────────────────────────────────────
    print("\n[OPTIONAL MODULES]")
    for mod, hint in OPTIONAL:
        ok, err = _check(mod)
        if ok:
            print(f"  OK    {mod}")
        else:
            print(f"  WARN  {mod}")
            print(f"        → {hint}")
            warnings.append((mod, err, hint))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Required:  {len(REQUIRED) - len(failures)}/{len(REQUIRED)} OK  "
          f"({len(failures)} failed)")
    print(f"Optional:  {len(OPTIONAL) - len(warnings)}/{len(OPTIONAL)} OK  "
          f"({len(warnings)} missing — not errors)")

    if failures:
        print("\n[FAILED IMPORTS — fix before running any script]")
        for mod, err in failures:
            print(f"  • {mod}")
            print(f"    {err}")
        print("\nExit code: 1")
        return 1

    print("\nAll required imports OK.")
    if warnings:
        print(f"{len(warnings)} optional deps missing — install as needed (see requirements.txt).")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
