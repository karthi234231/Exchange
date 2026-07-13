# Stubs Register — NSE AlgoTrader

All stub and placeholder functions that must be replaced before live trading.
A stub is any function that logs `.stub_ok`, raises `NotImplementedError` in a
production path, returns a placeholder structure, or has a comment indicating
incomplete implementation.

Last audited: 2026-06-22

---

## Summary

| Status | Count |
|--------|-------|
| Blocks live trading (P0 / P1) | 6 |
| Stage-gated (intentional, correct) | 5 |
| Pending implementation (P2) | 3 |

---

## Stub Table

| File | Function / Location | Status | Blocks Live Trading? | Notes |
|------|--------------------|---------|-----------------------|-------|
| `scripts/train_models.py:612` | `_train_ensemble()` → returns `ensemble_stub` dict | **INCOMPLETE** | **YES** — model artifact is fake | Returns a descriptor dict, not a trained MLEnsemble. SHAP and Treelite compile are also stubs. Must be replaced before model registration is valid. |
| `scripts/train_models.py:631` | `_compile_treelite()` | **INCOMPLETE** | **YES** — `.so` files not produced | Logs `treelite.compile.stub_ok`. Treelite `.so` files are not created. Inference engine will fail to load without real `.so` files. |
| `scripts/train_models.py:651` | `_generate_shap_report()` | **INCOMPLETE** | **YES** — SEBI compliance gap | Writes stub JSON with `status: "STUB"`. Model registry (`validate_artifacts`) checks for this status and should block registration. |
| `features/engine.py:263` | `VolumeBarAggregator` | **STUB** | **YES** for Stage 2+ | Defined as "lightweight stub". Volume bars required for Stage 2A ML. Not needed for Stage 1 time bars. |
| `scripts/train_models.py:693` | `_track_experiment()` | **INCOMPLETE** | NO — monitoring only | MLflow and DVC tracking not yet connected. Affects reproducibility and experiment audit trail but does not block paper trading. |
| `fo/portfolio_risk.py:73-77` | `StrategyResult` (stub type) | **STUB TYPE** | YES for F&O | Minimal stub class standing in for real `strategies.py`. F&O trading cannot function correctly without the real type. |

---

## Stage-Gated Stubs (Intentional — Correct Behavior)

These modules raise `NotImplementedError` by design. They are NOT bugs and should NOT
be implemented until the corresponding stage is active.

| File | Class / Function | Stage Gate | Notes |
|------|-----------------|-----------|-------|
| `models/rl_sizer.py` | `RLPositionSizer` | Stage 4 only | Raises `NotImplementedError` if `ALGO_STAGE < 4`. Correct behavior. |
| `models/rl_env.py` | `NSETradingEnv` | Stage 4 only | Raises `NotImplementedError` if `ALGO_STAGE < 4`. Correct behavior. |
| `features/microstructure.py:482` | `compute_hawkes_intensity()` | Stage 3+ only | `NotImplementedError` raised. Requires sub-second tick data. Correct. |
| `features/microstructure.py:504` | `compute_trade_flow_imbalance()` | Stage 3+ only | `NotImplementedError` raised. Requires tick-level data. Correct. |
| `execution/order_executor.py:666` | Experimental modify path | N/A | `NotImplementedError` — no order modification supported. Policy: cancel + resubmit only. |

---

## How to Identify New Stubs

Search for these patterns in the codebase before any release:

```bash
# Find stub_ok log calls
grep -rn "stub_ok" algo_trading/ scripts/

# Find NotImplementedError raises outside stage-gated modules
grep -rn "NotImplementedError" algo_trading/ scripts/ | grep -v "rl_sizer\|rl_env\|microstructure\|order_executor"

# Find STUB markers in JSON/comments
grep -rn '"STUB"' algo_trading/ models/ state/

# Find functions that only return None with a comment
grep -rn "pass  # stub\|return None  # stub\|# TODO.*live" algo_trading/
```

---

## Blocking Stub Resolution Plan

### Priority 1 — Before Model Registration

**Target:** `scripts/train_models.py` — `_train_ensemble`, `_compile_treelite`, `_generate_shap_report`

These three functions are the critical path for producing real model artifacts.
They must be replaced with real implementations in this order:

1. Build `models/ensemble.py` — real XGBoost + LightGBM + CatBoost stacking
2. Call `treelite.Model.from_xgboost(model).export_lib(...)` after training
3. Call `shap.TreeExplainer(model).shap_values(X_test)` and write real SHAP report
4. Update `sebi_report.json` to remove `"STUB"` status

Only after all three are real can `model_registry.validate_artifacts()` pass.

### Priority 2 — Before Stage 2

**Target:** `features/engine.py` — `VolumeBarAggregator` stub

Replace with the full `VolumeBarAggregator` class described in `MD_Files/_part2_infrastructure.md`.
This is only required when activating Stage 2A (volume bars + ML ensemble).

### Priority 3 — Before F&O

**Target:** `fo/portfolio_risk.py` — `StrategyResult` stub type

Replace with the real `StrategyResult` class from `fo/strategies.py` once that module
is built. Required before any F&O strategy execution.
