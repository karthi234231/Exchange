# Module Ownership — NSE AlgoTrader

Defines which package area owns which concern. Changes to live execution code must not
bleed into research code and vice versa. Review this document before adding or moving
any module.

---

## Package Areas and Their Concerns

### `algo_trading/market_data/` — Live Data Ingest

**Owns:** Everything from broker WebSocket to ZMQ PUB:5555. This is the hot path.

| File | Concern |
|------|---------|
| `system_state.py` | LiveSystemState persistence, StartupReconciler |
| `broker_shoonya.py` | Shoonya WebSocket + REST adapter (EXECUTION ONLY — no research) |
| `tick_validator.py` [ASSUMED] | Tick validation: outlier rejection, staleness, gap detection |
| `candle_store.py` | QuestDB ILP writes for OHLCV bars |
| `ring_buffer.py` | Numpy ring buffer for hot-path tick storage |

**Does NOT own:** Feature computation, signal generation, order placement, research code.

**Rules:**
- No pandas on the hot path. Use Polars or numpy.
- No blocking I/O in the tick callback. Buffer and flush asynchronously.
- All ticks pass through `TickValidator` before use.
- This package never imports from `models/`, `backtest/`, or `scripts/`.

---

### `algo_trading/features/` — Feature Engine

**Owns:** All feature computation. Receives bars from ZMQ SUB:5555, emits feature
vectors to ZMQ PUB:5556.

| File | Concern |
|------|---------|
| `engine.py` | Orchestrates feature pipeline; ZMQ SUB/PUB |
| `technical.py` | Technical indicators (RSI, MACD, ATR, EMA, etc.) |
| `statistical.py` | HMM regime, GARCH vol, Kalman filter, autocorrelation |
| `microstructure.py` | VPIN filter (Stage 2+), Avellaneda-Stoikov reservation price |
| `custom.py` | VWAP deviation, OBI, Volume Clock, time-of-day encoding |
| `stage_router.py` | Routes feature computation by active stage |
| `fo_features.py` | F&O-specific features (Stage 4+) |
| `hawkes.py` | Stage 3+ ONLY — blocked at Stage 1-2 via NotImplementedError |
| `tfi.py` | Stage 3+ ONLY — blocked at Stage 1-2 via NotImplementedError |

**Does NOT own:** Bar building (market_data), signal generation (signals/), model training.

**Rules:**
- Feature schema is frozen per `FrozenModelSpec`. Do not add features without updating the spec.
- All stateful models (HMM, GARCH, Kalman) must be fitted before market open.
- Feature vectors with NaN or Inf values must be dropped, not zero-filled.
- Never import from `execution/`, `risk/`, or `models/inference_engine.py`.

---

### `algo_trading/models/` — ML Stack

**Owns:** Model training, inference, hyperparameter tuning, experiment tracking, and explainability.

| File | Concern |
|------|---------|
| `inference_engine.py` | Treelite .so inference (live path); ZMQ SUB:5556 → PUSH:5557 |
| `model_registry.py` | FrozenModelSpec; validates artifacts before registration |
| `ensemble.py` | Training: XGBoost + LightGBM + CatBoost stacking |
| `tuner.py` | Optuna hyperparameter optimization (Sunday only, never during market hours) |
| `tracker.py` | MLflow experiment logging |
| `explainer.py` | SHAP analysis and SEBI compliance report |
| `rl_sizer.py` | Stage 4 ONLY — NotImplementedError on ALGO_STAGE < 4 |
| `rl_env.py` | Stage 4 ONLY — NotImplementedError on ALGO_STAGE < 4 |

**Does NOT own:** Feature computation, order placement, risk management.

**Rules:**
- Training code (`ensemble.py`, `tuner.py`) must NEVER run during market hours.
- `inference_engine.py` loads ONLY `.so` files (Treelite). No `.pkl` files in the live inference path.
- `model_registry.py` blocks stub or incomplete models from registration.
- ML class labels are fixed: 0 = bear, 1 = sideways, 2 = bull. Never change this mapping.

---

### `algo_trading/execution/` — Order Execution

**Owns:** The complete order lifecycle from signal receipt to broker confirmation.
This is the most critical package — bugs here directly cost or lose money.

| File | Concern |
|------|---------|
| `order_executor.py` | Main execution loop; FSM + rate limiter + broker calls |
| `order_fsm.py` | Order state machine: valid state transitions |
| `rate_limiter.py` | DualLockRateLimiter: global OPS budget + per-order mod limit |
| `signal_hierarchy.py` | Priority resolution: RiskMgr > HMM > ML > ADWIN |
| `latency_profiler.py` | Tick-to-order latency measurement |
| `exception_policy.py` | Fail-open vs fail-closed policy per component |

**Does NOT own:** Risk rules (risk/), feature computation, model training.

**Rules:**
- Entries fail closed on ANY missing safety data.
- Exits have a separate, more permissive safety path.
- Rate limiter must be released on every terminal order state (FILLED, CANCELLED, REJECTED, TIMEOUT).
- Never call `cancel_order` without a confirmed `exchange_order_id`.
- No Telegram I/O in the execution thread. Use ZMQ PUSH to AlertSink.

---

### `algo_trading/risk/` — Risk Management

**Owns:** All pre-trade checks and kill switch logic.

| File | Concern |
|------|---------|
| `manager.py` | Kill switch, daily loss limit, drawdown, correlation guard, fat-finger |
| `cost_model.py` | India transaction costs: STT, exchange, GST, stamp duty, slippage |

**Does NOT own:** Order placement, feature computation, position accounting.

**Rules:**
- Risk manager has absolute veto power over any signal (highest priority in signal hierarchy).
- Fat-finger guard fails closed on missing ADV data (entries only; exits bypass).
- Cost model slippage is dynamic (ATR-based), not static bps.
- Kill switch sentinel file takes precedence over all other state.

---

### `algo_trading/signals/` — Signal Generation

**Owns:** Strategy-level signal logic (not feature computation, not order placement).

| File | Concern |
|------|---------|
| `ema_cross.py` | Stage 1 EMA(9/21) cross signal |

**Does NOT own:** Feature values (input from features/), order placement (execution/).

**Rules:**
- Signals never access the broker directly.
- No signal is emitted during the first bar (09:15 bar) — `SKIP_FIRST_BAR=true` in config.
- No signal after 15:15 IST (`NO_NEW_TRADES_AFTER_HHMM`).

---

### `algo_trading/alerts/` — Monitoring and Alerting

**Owns:** The Telegram alert pipeline. Must be decoupled from execution.

| File | Concern |
|------|---------|
| `alert_sink.py` | Separate process; ZMQ PULL:5560 → Telegram HTTP API |
| `telegram_bot.py` | Non-blocking ZMQ PUSH sender (fire-and-forget) |

**Rules:**
- `telegram_bot.py` NEVER calls the Telegram API directly.
- `alert_sink.py` runs as a SEPARATE OS process from the execution path.
- Telegram failure must NEVER crash or block the trading processes.
- Critical events (kill switch, daily loss limit, broker disconnect) are written to audit log FIRST, then queued to AlertSink.

---

### `algo_trading/fo/` — F&O Pipeline (Stage 4+)

**Owns:** All F&O-specific concerns. Disabled at Stage 1.

| File | Concern |
|------|---------|
| `orchestrator.py` | F&O session lifecycle |
| `execution.py` | Multi-leg atomic executor |
| `margin.py` | Margin calculation and headroom check |
| `expiry.py` | Tuesday expiry rules per NSE circular FAOP68747 |
| `portfolio_risk.py` | Portfolio-level F&O risk: Greeks, vol surface |
| `chain.py` [ASSUMED] | Options chain snapshot and staleness check |

**Rules:**
- F&O strategies MUST use multi-leg executor (`fo/execution.py`), not the equity `order_executor.py`.
- Margin check is a hard precondition before any leg placement.
- Expiry rules come from `fo/expiry.py` only. No hardcoded Thursday or Tuesday weekday constants.

---

### `algo_trading/backtest/` — Backtesting Harness

**Owns:** Offline strategy validation. Never runs during market hours.

| File | Concern |
|------|---------|
| `runner.py` | Walk-forward backtest with full cost model |

**Rules:**
- Backtest uses the same cost model, feature schema, and signal code as live trading.
- All P&L metrics are computed from NET of cost (not gross). Gross P&L metrics are supplementary only.
- Walk-forward uses time-based splits with purge + embargo (minimum 51-bar purge window with MACD).
- Backtest results must be reproducible given the same data and `FrozenModelSpec`.

---

### `algo_trading/scripts/` — Utility Scripts

**Owns:** Offline data preparation, model training, and validation scripts.
These are operator tools, not library code.

| File | Concern |
|------|---------|
| `download_history.py` | Historical OHLCV data download |
| `train_models.py` | Full ML training pipeline (stubs pending MLEnsemble build) |
| `train_rl.py` | Stage 4 RL training (guarded by ALGO_STAGE check) |
| `validate_edge.py` | Pre-live edge validation gate |
| `paper_trade.py` | Paper trading mode entry point |

**Rules:**
- Scripts that hit the broker API must have a `--dry-run` flag.
- `train_models.py` must not run during market hours (enforced by `is_safe_to_retrain()` check).
- Scripts import from `algo_trading.*` but never the other way around.

---

### `algo_trading/labeling/` — Label Generation

**Owns:** Triple-barrier labeling and walk-forward split logic.
Research-time only; never called during live trading.

| File | Concern |
|------|---------|
| `triple_barrier.py` | Triple-barrier method per Lopez de Prado |
| `walk_forward.py` | Time-series split with purge and embargo |

---

## Cross-Package Dependencies (allowed)

```
market_data/ --> (no imports from algo_trading sub-packages)
features/    --> market_data/ (for bar types)
signals/     --> features/ (for feature values)
models/      --> features/ (for feature schema only)
execution/   --> models/inference_engine (for signals)
execution/   --> risk/ (for pre-trade checks)
execution/   --> market_data/ (for LiveSystemState)
risk/        --> market_data/ (for position state)
alerts/      --> (no imports from algo_trading sub-packages)
backtest/    --> features/, risk/, cost_model, labeling/
scripts/     --> any package (offline use only)
```

**Prohibited imports:**
- `market_data/` must not import from `execution/`, `models/`, or `risk/`
- `features/` must not import from `execution/` or `risk/`
- `alerts/` must not import from `execution/` or `risk/`
- `execution/` must not import from `backtest/` or `scripts/`
