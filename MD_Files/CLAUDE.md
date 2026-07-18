# CLAUDE.md — AI Algo Trading System (India)

## Current Implementation Status

> **As of 2026-06-06 — Read this before doing anything else**

| Item | Status |
|---|---|
| Python package / runnable code | NOT STARTED |
| requirements.txt / pyproject.toml | NOT STARTED |
| main.py / paper trading loop | NOT STARTED |
| Order FSM | NOT STARTED |
| Risk manager | NOT STARTED |
| Cost model | NOT STARTED |
| Broker adapter (Shoonya) | NOT STARTED |
| QuestDB schema | NOT STARTED |
| Backtests | NOT STARTED |
| Tests | NOT STARTED |
| Live trading | FORBIDDEN until all gates pass |

**This file is the canonical AI instruction spec and build plan. The repository currently contains architecture docs only. No code exists yet.**

---

## Files Claude Must Read at the Start of Every Session

> These files were created as part of the project audit. They contain decisions, rules, and constraints that override anything else in this repo. Read them before writing any code.

| File | When to Read | What It Contains |
|---|---|---|
| `arch-spec.json` | Every session | Machine-readable architecture contract — broker, stack, constraints, stage definitions |
| `MD_Files/DECISIONS.md` | Every session | Single source of truth for all active decisions (broker, bar type, ML policy, go-live gate) |
| `MD_Files/RISK_REGISTER.md` | When touching any module | All open risks with status — check if the module you are building has an open risk |
| `MD_Files/ACCEPTANCE_CRITERIA.md` | Before marking any module done | Per-module definition of done — tests that must pass |
| `MD_Files/compliance_matrix.md` | Before any live/broker code | SEBI + Shoonya compliance requirements |
| `MD_Files/verified_facts.md` | Before using any external fact | Broker fees, regulatory rules, data limits — all marked ASSUMED until verified |
| `MD_Files/RUNBOOK.md` | Before paper/live testing | Emergency procedures and daily checklists |
| `MD_Files/DATA_COLLECTION_GUIDE.md` | Before any data download | What data sources exist, costs, and what Claude can run automatically |

### Key rules from those files (summary — always read originals for full detail)

- **Primary broker:** Shoonya only. AngelOne = dev reference, never production.
- **Storage:** QuestDB only. No SQLite for ticks, bars, signals, orders, fills, or PnL.
- **Bar type:** 15-min time bars for Stage 1 only. Volume bars from Stage 2A ML onward. `TRAIN_BAR_TYPE == LIVE_BAR_TYPE` always.
- **TabPFN:** Banned from production. Research notebooks only.
- **RL / deep models / sentiment:** NOT NOW list — frozen until deterministic baseline is profitable.
- **Live trading:** Hard-disabled until all gates in `DECISIONS.md` and `README.md` pass.
- **Free data (yFinance/Jugaad):** Enough for plumbing only. Not enough for 3-year ML validation.
- **Slippage:** No static 0.05% — dynamic model required.
- **Tests:** No module is complete until its acceptance criteria in `ACCEPTANCE_CRITERIA.md` pass.

---

## Project Overview
A production-target algorithmic trading system for Indian equity markets (NSE).
Python-based. Runs locally or on an Indian VPS. Uses ML ensemble for signals,
optional RL for position sizing, and connects to Indian broker APIs for execution.

## Hardware
- **Machine:** Intel i5-1345U (10 cores), 16GB RAM, NO dedicated GPU (Intel Iris Xe only)
- **GPU:** None — all code must run CPU-only
- **IDE:** VS Code (no Jupyter/Colab needed — everything in .py files)
- **Heavy training:** Use Kaggle Notebooks (free T4 GPU, 30 hrs/week) for Optuna runs

## Tech Stack
- **Language:** Python 3.11+
- **Execution Broker:** Shoonya/Finvasia (PRIMARY — zero brokerage, no per-request TOTP refresh); session tokens expire daily — handle deterministically [SURV-10]
- **Data Feed (Stage 1):** Shoonya WebSocket — throttled 1–3 snaps/sec, sufficient for EMA cross only
- **Data Feed (Stage 2+):** TrueData WebSocket (`truedata-ws`) — unthrottled L2 tick-by-tick, **REQUIRED** for Hawkes/VPIN/TFI features
- **Data Storage:** QuestDB (tick + candle storage, 150x faster than SQLite for time-series)
- **Data Processing:** Polars (feature engineering, 5-10x faster than pandas)
- **ML:** XGBoost, LightGBM, CatBoost (ensemble) — CPU mode only (tree_method='hist')
- **Inference Compilation:** Treelite (compile tree ensembles to .so shared lib) — reduces predict latency from ~40ms → ~2ms
- **RL (optional):** Stable-Baselines3 (PPO) — CPU mode, train on Kaggle if slow
- **Features:** pandas-ta, hmmlearn, arch, filterpy
- **Tuning:** Optuna — n_jobs=-1 (use all 10 cores); heavy runs overnight or on Kaggle
- **Tracking:** MLflow + DVC (experiment tracking + data/model version control)
- **Explainability:** SHAP (feature importance + SEBI compliance documentation)
- **Alerts:** Telegram via ZeroMQ PUSH/PULL sink — **execution thread never blocks on Telegram**
- **Testing:** pytest

## Project Structure
```
algo_trading/
├── CLAUDE.md                  # This file
├── .env                       # API keys (NEVER commit)
├── .env.example               # Template for .env
├── requirements.txt
├── setup.py
├── README.md
├── main.py                    # Orchestrator — main entry point
├── config.py                  # All configuration dataclasses
│
├── data/                      # Module 1: Data Pipeline
│   ├── __init__.py
│   ├── broker_base.py         # Abstract broker interface
│   ├── broker_shoonya.py      # Shoonya/Finvasia — EXECUTION ONLY (orders, positions) [CDR-3B-C24]
│   ├── broker_angelone.py     # Angel One SmartAPI (dev reference only)
│   ├── ring_buffer.py         # Numpy structured ring buffer — hot-path tick store [P2I-2.2]
│   ├── integrity_guard.py     # TickValidator + DataQualityMonitor — halt on bad data [SURV-2]
│   ├── system_state.py        # LiveSystemState persistence + StartupReconciler [SURV-6]
│   ├── websocket_feed.py      # ZeroMQ DataIngestor PUB:5555 — wraps TrueData/Shoonya feed [P2I-2.1]
│   ├── truedata_feed.py       # TrueData unthrottled L2 WebSocket — REQUIRED for microstructure [ARCH-FIX-1]
│   └── candle_store.py        # QuestDB candle storage + VolumeBarAggregator [P2I-2.3]
│
├── features/                  # Module 2: Feature Engine (23 features)
│   ├── __init__.py
│   ├── engine.py              # FeatureComputer ZeroMQ SUB/PUB:5556 [P2I-2.1]
│   ├── technical.py           # pandas-ta indicators (features 1-12, 18-19)
│   ├── statistical.py         # HMM, Kalman, GARCH, autocorrelation (13-15, 20)
│   ├── custom.py              # Volume Clock, numba OBI, VWAP dev, ToD, spread (8, 16-17, 21-23) [P2I-2.4]
│   └── microstructure.py      # Hawkes, VPIN, TFI, Avellaneda-Stoikov [CDR-P1]
│
├── labeling/                  # Module 3: Labeling Engine
│   ├── __init__.py
│   ├── triple_barrier.py      # Triple-barrier method (López de Prado)
│   └── walk_forward.py        # Walk-forward split with purge + embargo
│
├── models/                    # Module 4: ML Ensemble + RL
│   ├── __init__.py
│   ├── ensemble.py            # Stacking: XGB+LGB+CatBoost OOF → LR meta-learner [CDR-P5-STACK]
│   ├── inference_engine.py    # Treelite compiled inference ZeroMQ SUB→PUSH:5557 [ARCH-FIX-2]
│   ├── online_model.py        # River ADWIN online learning (regime adaptation) [P6A-6.4]
│   ├── tuner.py               # Optuna hyperparameter optimization [CDR-3B-C23]
│   ├── tracker.py             # MLflow experiment tracking wrapper
│   ├── explainer.py           # SHAP explainability + SEBI compliance report generation
│   ├── rl_sizer.py            # RL agent for position sizing — Phase 4 OPTIONAL [CDR-3B-C18]
│   └── sentiment.py           # Async SentimentWorker via ZeroMQ PUSH/PULL [CDR-3B-C20]
│
├── risk/                      # Module 5: Risk Management
│   ├── __init__.py
│   ├── manager.py             # Kill switches, daily loss limit, drawdown breaker
│   ├── position_sizer.py      # Volatility-adjusted (ATR-based) position sizing
│   └── cost_model.py          # India cost model: brokerage + STT + GST + slippage
│
├── execution/                 # Module 6: Execution Layer
│   ├── __init__.py
│   ├── order_manager.py       # LIMIT/MARKET + DualLockRateLimiter + partial fill handler [ARCH-FIX-5]
│   ├── signal_hierarchy.py    # Explicit priority: RiskMgr→HMM→ML→ADWIN [SURV-4]
│   ├── latency_profiler.py    # Measure tick→feature→signal→order→ack latency chain [SURV-3]
│   └── trade_logger.py        # QuestDB trade logging + model performance tracking
│
├── alerts/                    # Module 7: Monitoring
│   ├── __init__.py
│   ├── telegram_bot.py        # Non-blocking ZMQ PUSH sender — fire-and-forget [ARCH-FIX-3]
│   └── alert_sink.py          # Dedicated ZMQ PULL worker — sends to Telegram in separate process
│
├── backtest/                  # Backtesting harness
│   ├── __init__.py
│   └── runner.py              # Walk-forward backtest with realistic cost model
│
├── scripts/                   # Utility scripts
│   ├── download_history.py    # Download historical data from broker/yfinance
│   ├── train_models.py        # Run full training pipeline
│   └── paper_trade.py         # Paper trading mode entry point
│
└── tests/                     # pytest test suite
    ├── __init__.py
    ├── test_features.py
    ├── test_labeling.py
    ├── test_risk.py
    ├── test_cost_model.py
    └── test_ensemble.py
```

## Coding Conventions
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- f-strings for string formatting
- `logging` module (not print) for all output
- All config via `config.py` dataclasses, overridable by `.env`
- No hardcoded API keys anywhere — always from environment variables
- `.env` file must have `chmod 600` permissions (owner read-only); verify `.gitignore` contains `.env` before first commit
- Every module must work standalone (importable + testable independently)
- Use `pathlib.Path` not `os.path`
- QuestDB: batch writes — accumulate ticks in ring buffer, flush to QuestDB every 100ms or 50 rows (whichever first); never write per-event on the hot path (causes latency spikes)
- Use Polars DataFrames internally in feature engine; convert to pandas only where libraries require it
- All model training runs logged to MLflow; all data/model files versioned with DVC
- SHAP values computed and saved after every training run (SEBI explainability requirement)

## Key Design Decisions

### Labeling (CRITICAL — get this right or nothing works)
- Use triple-barrier method from López de Prado's AFML → [CDR-3A-C12]
- **Default: symmetric ±1.5 ATR** (upper_atr_mult = lower_atr_mult = 1.5)
  - Asymmetric 2.0/1.0 causes class imbalance → model learns to predict nearer (lower) barrier → long-side bias
  - Only use asymmetric AFTER verifying class balance with actual NSE data shows >15% in every class
- Time barrier = max_holding_bars = 20 (15-min bars = 5 hours)
- Label: +1 (hit upper first), -1 (hit lower first), 0 (timed out)
- Run `compute_label_distribution()` after labeling — if any class < 15%, fix immediately (see [IRF-Q1])
- NEVER use naive "next candle direction" as label

### Validation (CRITICAL — prevents fake profits)
- Walk-forward time-series split ONLY. Never shuffle.
- **Purge window = holding_period + max_feature_lookback + embargo** → [CDR-3B-C17]
  - With MACD(26) as longest feature: 20 + 26 + 5 = **51 bars minimum**
  - With Ichimoku(52) in feature set: 20 + 52 + 5 = **77 bars minimum**
  - Embargo = 5 bars (not 2 — 2 is insufficient for NSE autocorrelation)
- Minimum 5 walk-forward folds for any reported metric

### Features (23 total)
1. RSI(14)  2. MACD(12,26,9)  3. BB %B  4. BB Width  5. ATR(14)
6. ADX(14)  7. Stoch %K/%D  8. VWAP deviation  9. OBV  10. Supertrend
11. EMA cross(9/21)  12. Ichimoku cloud position  13. Kalman smoothed price
14. HMM regime  15. GARCH vol forecast  16. Order book imbalance
17. Price-volume divergence  18. Heikin-Ashi direction  19. Pivot distance
20. Returns autocorrelation  21. Relative strength vs Nifty
22. Time-of-day (sin/cos encoding)  23. Bid-ask spread

### Risk Management (non-negotiable — code these as hard stops)
- Max daily loss: 2% of capital → kill switch, stop all trading
- Max single position: 5% of capital
- Max drawdown: 10% from peak → pause system, alert via Telegram
- Position sizing: **ATR-based (primary)** = approximates Half-Kelly → [CDR-3B-C18]
  - Formula: `position_size = (capital × risk_pct) / (ATR14 × atr_multiplier)`
  - Kelly formula: `f* = (p×b − q)/b`, use Half-Kelly = f*×0.5 for live trading
  - RL-based sizing (PPO) = Phase 4 OPTIONAL ONLY — unstable on financial data
- Every trade has a stop-loss (ATR-based, no exceptions)
- Full India cost model on every backtest and live trade

### Execution
- Prefer LIMIT orders; price set by `get_limit_price()` (signal-strength-driven) → [SURV-11]
- **NEVER convert stale LIMIT to MARKET** — if not filled in 1 candle, cancel entirely
- **NEVER enter if price moved >30bps from signal price** — missing is free, chasing is not
- Stop-loss: always MARKET, no price cap, no exceptions
- **Never trade 9:15–9:30 bar** (opening auction artifacts) → [SURV-8]
- **Square off ALL intraday positions by 3:15 PM IST** (not 3:30 — auto-square penalty) → [SURV-8]
- Rate limit: **DualLockRateLimiter** → [ARCH-FIX-5]
  - Global: max 8 orders/second (SEBI limit is 10, keep 2-order buffer)
  - Per-order: max 5 modifications per Order ID before forced cancel + symbol halt
  - Call `release_order(order_id)` on every FILLED/CANCELLED/REJECTED event
- Partial fill threshold: if order_qty > 20% of visible depth → cancel remainder [SURV-3]
- All orders must carry broker-assigned Strategy ID (SEBI algo framework — register with your broker before live trading; exact implementation timelines vary by exchange/broker, confirm current status before going live)i 
- Daily session auto-logout before market pre-open
- On startup: run StartupReconciler BEFORE placing any order [SURV-6]

### Broker Abstraction
- `BrokerBase` abstract class with methods: login(), place_order(), cancel_order(),
  get_positions(), get_order_book(), subscribe_ws(), get_historical_data()
- Concrete implementations: `ShoonyaBroker` (primary), `AngelOneBroker` (dev reference only)
- Swap broker by changing one config line

## ⚠️ STAGE 1 GATE — Paper Trade First, ML Later

> **Read before building anything past step 8.**
> The #1 failure mode in solo algo trading: spending months building ML, then discovering on day 1 that the WebSocket drops or the order placement silently fails. Stage 1 eliminates this risk.

**Stage 1 (Steps 1–8): Build infrastructure + paper trade EMA cross ONLY**
- Strategy: `ema_cross_signal()` — EMA(9) crosses EMA(21) → LONG/SHORT signal (See [CDR-P4-S1])
- Broker: Shoonya paper trading (simulated orders, zero capital at risk)
- **Exit gate — do NOT proceed to Step 9 until ALL pass:**
  - [ ] Shoonya WebSocket connects + auto-reconnects after disconnect
  - [ ] Candles stored in QuestDB with correct IST timestamps
  - [ ] Paper order placed via `ShoonyaBroker.place_order()` returns valid order ID
  - [ ] Paper order cancelled successfully
  - [ ] Telegram alert fires on trade entry, exit, and any exception
  - [ ] 5 complete EMA cross paper trade cycles with zero crashes logged

**Stage 2 (Steps 9–22): Build ML stack on PROVEN infrastructure only**

---

## ⚠️ CRITICAL ARCHITECTURE FIXES — 5 Fatal Cracks

> These 5 issues will bleed money in live markets if not implemented.
> Each has an `[ARCH-FIX-N]` ID — use it when building the corresponding module.

---

### [ARCH-FIX-1] Data Feed Contradiction — L2 Microstructure Requires TrueData

**The crack:** `features/microstructure.py` uses Hawkes Process, VPIN, and TFI. These require tick-by-tick arrival times and multi-level order book depth. Shoonya/Angel One WebSockets deliver snapped quotes at 1–3 times per second. Running Hawkes on snapped data produces ghost liquidity readings — the inter-arrival time distribution is wrong, VPIN buckets accumulate incorrectly, and TFI signals fire on phantom order flow.

**The rule:**
```
Shoonya WebSocket  → order_manager.py ONLY (place/cancel/status)
TrueData WebSocket → websocket_feed.py → ALL feature computation
```

**Stage gate:** microstructure.py features are **Stage 2 only**. Do NOT import or compute them in Stage 1.
- Stage 1: `config.DATA_FEED = 'shoonya'` — throttled, EMA cross only
- Stage 2+: `config.DATA_FEED = 'truedata'` — unthrottled L2, microstructure enabled

**Implementation — `data/truedata_feed.py`:**
```python
# pip install truedata-ws
# Subscription: ~Rs. 300–500/month (NSE equities). Register at truedata.in
import truedata_ws

class TrueDataFeed:
    """Unthrottled L2 tick-by-tick feed. REQUIRED for Hawkes/VPIN/TFI.
    Wraps truedata_ws and publishes ticks to ZMQ PUB:5555 identically
    to Shoonya feed so downstream FeatureComputer needs no changes.
    """
    def __init__(self, user: str, password: str, symbols: list[str]):
        self.td = truedata_ws.TD_hist(user, password)
        self.realtime = truedata_ws.TD_socket(user, password)

    def start(self, zmq_pub_socket):
        self.realtime.start_live_data(self._on_tick)

    def _on_tick(self, tick):
        # tick has: ltp, bid, ask, bid_qty, ask_qty, volume, timestamp
        zmq_pub_socket.send(msgpack.packb(tick))
```

**Config guard — `config.py`:**
```python
@dataclass
class FeatureConfig:
    use_microstructure: bool = False  # True only when DATA_FEED == 'truedata'

    def __post_init__(self):
        if self.use_microstructure and DATA_FEED != 'truedata':
            raise ConfigError(
                "Hawkes/VPIN/TFI require TrueData unthrottled feed. "
                "Set DATA_FEED='truedata' or set use_microstructure=False."
            )
```

**`.env` addition (Stage 2+):**
```
TRUEDATA_USER=your_truedata_username
TRUEDATA_PASSWORD=your_truedata_password
DATA_FEED=shoonya   # Change to 'truedata' at Stage 2
```

---

### [ARCH-FIX-2] Inference Latency Trap — Compile Ensemble with Treelite

**The crack:** Running `xgb.predict()` + `lgb.predict()` + `cat.predict()` + `lr.predict()` natively on every Volume Bar takes ~40ms on i5 CPU. Volume bars can arrive every 5–30 seconds in liquid NSE stocks. The latency budget for the full pipeline (data → feature → inference → order) is 50ms. Native prediction alone blows this budget.

**The fix:** After training, compile the ensemble using **Treelite** (decision tree compiler). Treelite AOT-compiles the model into a platform-native shared library (`.so`). Inference drops to ~2ms.

**Implementation — add to `models/ensemble.py` after training:**
```python
import treelite
import treelite_runtime

def compile_for_production(
    xgb_model, lgb_model, cat_model, output_dir: str = "models/saved/"
) -> None:
    """Compile trained tree models to native shared libraries.
    Run ONCE after training. inference_engine.py loads .so files.
    """
    # XGBoost
    tl_xgb = treelite.Model.from_xgboost(xgb_model)
    tl_xgb.export_lib(toolchain='gcc', libpath=f'{output_dir}/xgb.so', verbose=False)

    # LightGBM
    tl_lgb = treelite.Model.from_lightgbm(lgb_model)
    tl_lgb.export_lib(toolchain='gcc', libpath=f'{output_dir}/lgb.so', verbose=False)

    # CatBoost → export to JSON first, then load
    cat_model.save_model(f'{output_dir}/cat.json', format='json')
    tl_cat = treelite.Model.load(f'{output_dir}/cat.json', model_format='catboost')
    tl_cat.export_lib(toolchain='gcc', libpath=f'{output_dir}/cat.so', verbose=False)

    logging.info("Compiled ensemble to .so — inference latency: ~2ms vs ~40ms native")
```

**Implementation — `models/inference_engine.py` (live inference):**
```python
import treelite_runtime

class EnsembleInferenceEngine:
    """Loads compiled .so models. NEVER loads .pkl in production."""

    def __init__(self, model_dir: str = "models/saved/"):
        self.xgb = treelite_runtime.Predictor(f'{model_dir}/xgb.so')
        self.lgb = treelite_runtime.Predictor(f'{model_dir}/lgb.so')
        self.cat = treelite_runtime.Predictor(f'{model_dir}/cat.so')
        # Meta-learner (LR) is tiny — sklearn native is fine (~0.1ms)
        self.meta = joblib.load(f'{model_dir}/meta_lr.pkl')

    def predict(self, X: np.ndarray) -> tuple[int, float]:
        """Returns (signal, confidence). Target: <5ms total."""
        dmat = treelite_runtime.DMatrix(X.astype(np.float32))
        p_xgb = self.xgb.predict(dmat)
        p_lgb = self.lgb.predict(dmat)
        p_cat = self.cat.predict(dmat)
        meta_input = np.column_stack([p_xgb, p_lgb, p_cat])
        proba = self.meta.predict_proba(meta_input)[0]
        signal = int(np.argmax(proba)) - 1  # {0,1,2} → {-1,0,+1}
        return signal, float(np.max(proba))
```

**Training pipeline rule:** `ensemble.py` trains → calls `compile_for_production()` → writes `.so` files → `inference_engine.py` loads `.so`. Never load `.pkl` files in the live inference path.

---

### [ARCH-FIX-3] Telegram Synchronous I/O — Decouple via ZeroMQ PUSH/PULL

**The crack:** The Telegram HTTP API routinely takes 200–2000ms (network jitter, Telegram rate limits, server load). If `alerts/telegram_bot.py` is called synchronously in the execution loop — even in the stop-loss placement path — a Telegram hang delays your stop-loss by up to 2 seconds in a fast-moving market.

**The fix:** ZeroMQ PUSH/PULL pattern. The execution thread fires a non-blocking ZMQ PUSH message (returns in <1µs). A dedicated `alert_sink.py` process holds the ZMQ PULL socket and is the only code that ever calls the Telegram API. If Telegram dies, only alerts die — the trade continues.

**Implementation — `alerts/telegram_bot.py` (non-blocking sender):**
```python
import zmq, logging

_ctx = zmq.Context.instance()
_push = _ctx.socket(zmq.PUSH)
_push.connect("tcp://127.0.0.1:5560")
_push.setsockopt(zmq.LINGER, 0)          # Don't block on close
_push.setsockopt(zmq.SNDHWM, 100)        # Drop if queue > 100 messages

def send_alert(msg: str, level: str = "INFO") -> None:
    """Non-blocking. Returns immediately. Never raises in the execution path."""
    try:
        _push.send_json({"msg": msg, "level": level}, zmq.NOBLOCK)
    except zmq.Again:
        logging.warning("Alert queue full — message dropped: %.80s", msg)
    except Exception:
        pass  # Alerts must NEVER crash the execution thread
```

**Implementation — `alerts/alert_sink.py` (dedicated worker process):**
```python
import zmq, os, logging, requests

TELEGRAM_URL = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def run_alert_sink():
    """Run as separate process. This is the ONLY code that talks to Telegram."""
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind("tcp://127.0.0.1:5560")
    logging.info("Alert sink started — listening on port 5560")
    while True:
        try:
            alert = pull.recv_json()
            requests.post(TELEGRAM_URL, json={
                "chat_id": CHAT_ID,
                "text": f"[{alert['level']}] {alert['msg']}"
            }, timeout=10)
        except Exception as e:
            logging.error("Telegram send failed: %s", e)
            # Continue — next alert will still be processed
```

**`main.py` startup:** Launch `alert_sink.py` as a subprocess before starting the trading processes. If it dies, restart it — but the trading processes continue regardless.

---

### [ARCH-FIX-4] TabPFN Removed — O(N²) Inference Unusable in Production

**The crack:** TabPFN's attention mechanism scales as O(N²) in the number of training examples. With 5,000+ labeled Volume Bars, a single inference call takes seconds, not milliseconds. It was listed as "Stage 2 optional" but this is a category error — it cannot be used in any live-bar environment regardless of stage.

**The fix:** Removed from architecture entirely.
- `models/tabpfn_model.py` — **DO NOT CREATE**. If you see this file in the codebase, delete it.
- For low-label scenarios (<500 examples): use the base GBDT ensemble with more aggressive Optuna regularization (`min_child_weight`, `max_depth ≤ 4`).
- For drift adaptation: `models/online_model.py` (River ADWIN) handles this correctly at O(1) per sample.

---

### [ARCH-FIX-5] SEBI OPS Dual-Lock — Per-Order Modification Limit

**The crack:** `TokenBucketRateLimiter` enforces 8 orders/second globally. But it has no per-order memory. A logic bug (rejected order that keeps retrying, or an order that partially fills and gets repeatedly re-modified) can loop indefinitely at 8 modifications/second on the same order ID. SEBI's surveillance system will flag the static IP for ban.

**The fix:** Dual-lock rate limiter — global token bucket (existing) + per-order modification counter.

**Implementation — update `execution/order_manager.py`:**
```python
class DualLockRateLimiter:
    """Global 8 OPS budget + max 5 modifications per single Order ID.
    If an order needs >5 modifications: it's a bad trade. Cancel + halt symbol.
    """
    MAX_GLOBAL_PER_SEC = 8
    MAX_MODS_PER_ORDER = 5

    def __init__(self):
        self._bucket = TokenBucket(rate=self.MAX_GLOBAL_PER_SEC, capacity=8)
        self._order_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, order_id: str) -> bool:
        """Returns True if allowed. False = reject and handle upstream."""
        with self._lock:
            if not self._bucket.consume(1):
                logging.warning("Global rate limit hit (8 OPS)")
                return False

            count = self._order_counts.get(order_id, 0)
            if count >= self.MAX_MODS_PER_ORDER:
                logging.error(
                    "Order %s exceeded %d modifications — cancelling + halting symbol",
                    order_id, self.MAX_MODS_PER_ORDER
                )
                return False   # Caller must: cancel_order(order_id) + halt_symbol()

            self._order_counts[order_id] = count + 1
            return True

    def release_order(self, order_id: str) -> None:
        """Call when order is FILLED, CANCELLED, or REJECTED."""
        with self._lock:
            self._order_counts.pop(order_id, None)
```

**Order lifecycle rule:** Every filled/cancelled/rejected order must call `rate_limiter.release_order(order_id)` to free its slot. Orphaned order IDs accumulate counts forever otherwise.

---

## ⚠️ REAL-WORLD SURVIVABILITY FIXES — 9 Fragility Cracks

> The system scores ~9/10 on research depth and ~8.5/10 on architecture.
> It scores ~3/10 on real-world profitability because trading success depends on
> **simple proven edge + execution realism + robustness under failure** — not sophistication.
> Each fix has a `[SURV-N]` ID. These are not optional improvements. They are the difference
> between a complex unprofitable system and a simple profitable one.

---

### [SURV-1] No Proven Edge — Profitability Gate Before ALL Complexity

**The fragility:** Every module in this system is additive complexity on top of an unvalidated signal. If the base signal (EMA cross, or the ML ensemble on real NSE data) doesn't have positive expectancy after costs, all the microstructure, stacking, GARCH, and ZeroMQ infrastructure produces a highly sophisticated way to lose money faster.

**The rule: zero complexity until the simple loop is proven profitable.**

**Profitability Gate — must pass before building steps 9–30:**
```python
# scripts/validate_edge.py — run this BEFORE building any ML
# Requirements: 3+ years 15-min data, 5+ Nifty 50 stocks, realistic cost model

def validate_ema_edge(symbols, years=3) -> EdgeReport:
    """
    Prove simple EMA(9/21) cross is profitable before building anything else.
    If this fails: fix the signal. Don't add complexity.
    """
    results = []
    for symbol in symbols:
        df = load_15min_candles(symbol, years=years)
        df['signal'] = ema_cross_signal(df['close'], fast=9, slow=21)
        df['cost'] = calculate_trade_cost(df['close'], per_side_bps=8)  # realistic 8bps/side
        pnl = compute_walk_forward_pnl(df, n_folds=5)
        results.append(pnl)

    sharpe = annualized_sharpe(results)
    profit_factor = total_gross_profit / total_gross_loss
    win_rate = sum(r > 0 for r in results) / len(results)

    # GATE: if ANY of these fail, STOP. Don't build ML.
    assert sharpe >= 1.0,    f"Sharpe {sharpe:.2f} < 1.0 — signal has no edge"
    assert profit_factor >= 1.2, f"PF {profit_factor:.2f} < 1.2 — costs exceed edge"
    assert win_rate >= 0.40, f"Win rate {win_rate:.1%} < 40% — too many losers"
    return EdgeReport(sharpe=sharpe, profit_factor=profit_factor, win_rate=win_rate)
```

**What "proven edge" means in practice — all 5 must pass:**
- Sharpe ≥ 1.0 on 5-fold walk-forward (not in-sample) with **minimum 200 trades per fold**
- Profit factor ≥ 1.2 after full India cost model (STT + exchange + brokerage + 8bps slippage)
- Consistent across 5+ Nifty 50 stocks, not just one cherry-picked name
- **Regime-stratified:** report Sharpe separately for bull/bear/sideways periods — must be positive in at least 2 of 3 regimes (a strategy that only works in bull markets is beta, not edge)
- **Stress windows:** does NOT blow up during opening hour (9:15–10:00), expiry weeks (last Thu of month), high-ATR days (ATR > 2x median). If it fails stress windows: the edge is fragile.

**If gate fails:** tune the signal parameters. Try different fast/slow lengths, add ADX filter (only trade when ADX > 25), add volume filter (only trade when volume > 1.5x 20-bar average). The signal must earn its keep before getting expensive machinery attached to it.

---

### [SURV-2] Data Integrity — Every Tick Must Be Validated Before Use

**The fragility:** Retail feeds including TrueData can produce: timestamp monotonicity violations (tick arrives with past timestamp), price outliers (erroneous 5% single-tick moves from fat-finger prints), zero-volume bars (stale data), duplicate ticks (same timestamp twice), sequence gaps. Running Hawkes, GARCH, or HMM on corrupted input produces confidently wrong outputs. The system will trade on ghost signals.

**Implementation — `data/integrity_guard.py`:**
```python
@dataclass
class TickValidator:
    max_price_move_pct: float = 0.03   # >3% single tick = outlier, reject
    max_gap_seconds: float = 120.0     # >2 min gap during market hours = stale
    min_volume: int = 1                # volume=0 for liquid stock = stale

    def validate(self, tick: Tick, prev_tick: Tick | None) -> ValidationResult:
        if prev_tick is None:
            return ValidationResult.OK

        # 1. Timestamp monotonicity
        if tick.timestamp <= prev_tick.timestamp:
            return ValidationResult.DUPLICATE

        # 2. Price outlier detection
        price_move = abs(tick.ltp - prev_tick.ltp) / prev_tick.ltp
        if price_move > self.max_price_move_pct:
            logging.warning("Price outlier: %.2f%% move, rejecting tick", price_move * 100)
            return ValidationResult.OUTLIER   # Use prev_tick.ltp in ring buffer

        # 3. Staleness during market hours (9:15–15:30 IST)
        if is_market_hours(tick.timestamp):
            gap = (tick.timestamp - prev_tick.timestamp).total_seconds()
            if gap > self.max_gap_seconds:
                return ValidationResult.STALE_GAP  # HALT trading

        # 4. Volume check
        if tick.volume < self.min_volume:
            return ValidationResult.ZERO_VOLUME

        return ValidationResult.OK

class DataQualityMonitor:
    """Tracks rolling data quality. Halts trading if quality degrades."""
    HALT_THRESHOLD = 0.05  # >5% bad ticks in last 100 = halt

    def should_halt_trading(self) -> bool:
        bad_rate = self.bad_ticks_last_100 / 100
        if bad_rate > self.HALT_THRESHOLD:
            logging.error("Data quality %.1f%% bad ticks — halting trading", bad_rate * 100)
            return True
        return False
```

**RULE: Bad data → halt trading, never degrade gracefully.** A degraded signal is worse than no signal because it fires with false confidence.

---

### [SURV-3] Execution Realism — Model What Actually Happens

**The fragility:** The system models execution as: signal → order placed → filled at intended price. Reality has four additional costs that are not modeled: queue position, partial fills, latency chain, and slippage bursts.

**Queue Position Model** (add to `risk/cost_model.py`):
```python
# LIMIT order at best bid — you are NOT first in queue.
# For NSE liquid stocks, best-bid queue depth = 5,000–50,000 shares.
# Conservative assumption: you are at BACK of queue.
# Expected fill probability in one bar = min_visible_depth / queue_depth
# If fill probability < 0.6: use MARKET order model for cost estimate

def estimate_limit_fill_probability(order_qty: int, visible_depth: int) -> float:
    """Conservative: assume we're at back of queue."""
    queue_turnover_estimate = visible_depth * 3  # 3x turnover per 15-min bar typical
    fill_prob = min(queue_turnover_estimate / (visible_depth + order_qty), 1.0)
    return fill_prob * 0.7  # 0.7 = adverse selection discount (good fills happen less than random)
```

**Partial Fill Model** (add to `execution/order_manager.py`):
```python
# If order_qty > 20% of visible bid/ask depth → expect partial fill
# Partial fills are NOT free: you pay brokerage on each partial fill
# AND you have an open position you didn't intend to have

PARTIAL_FILL_THRESHOLD = 0.20  # order_qty / visible_depth

def handle_partial_fill(order_id: str, filled_qty: int, total_qty: int):
    fill_ratio = filled_qty / total_qty
    if fill_ratio < 0.95:  # Not substantially complete
        # Three options — pick based on urgency:
        # 1. Cancel remainder (safest — use this by default)
        # 2. Re-submit at market (if urgency)
        # 3. Leave as-is (only if fill_ratio > 0.8 and position small)
        cancel_order_remainder(order_id)
        log_partial_fill(order_id, filled_qty, total_qty)
```

**Latency Chain Measurement** (add to `execution/latency_profiler.py`):
```python
# The ACTUAL latency chain on a home connection in India:
# Component              Typical    Worst Case
# Broker API ping        80ms       300ms
# ZMQ internal hops      1ms        5ms
# Feature compute        5ms        20ms
# Treelite inference     2ms        5ms
# Order placement        80ms       300ms
# Exchange ack           20ms       100ms
# TOTAL                  188ms      730ms
#
# Implication: obsessing over 40ms→2ms inference is optimizing 1% of the problem.
# REAL win: use an Indian VPS (reduces broker ping from ~200ms to ~20ms = 10x improvement)

@dataclass
class LatencySnapshot:
    tick_received_ns: int
    feature_computed_ns: int
    signal_generated_ns: int
    order_placed_ns: int
    exchange_ack_ns: int

    def log_to_questdb(self, symbol: str):
        # Track every component. Alert if any exceeds 2x 30-day median.
        total_ms = (self.exchange_ack_ns - self.tick_received_ns) / 1e6
        if total_ms > self.p95_threshold_ms:
            send_alert(f"Latency spike: {total_ms:.0f}ms on {symbol}", level="WARN")
```

**Slippage Burst Rule** (add to `risk/cost_model.py`):
```python
def get_dynamic_slippage_bps(atr_current: float, atr_20day: float) -> float:
    """Slippage scales with volatility. Don't use static 8bps in all conditions."""
    vol_ratio = atr_current / atr_20day
    if vol_ratio > 2.0:   return 25.0  # High vol event: use market slippage model
    if vol_ratio > 1.5:   return 15.0  # Elevated vol
    if vol_ratio > 1.0:   return 8.0   # Normal
    return 5.0                          # Low vol: tight spreads
```

---

### [SURV-4] Module Conflict Resolution — Explicit Signal Hierarchy

**The fragility:** HMM Regime + ML Ensemble + ADWIN drift flag + Risk Manager can simultaneously produce contradictory directives. Without an explicit hierarchy, the system's behavior under conflict is undefined. A sideways HMM regime + bullish ML signal + high ADWIN drift + near drawdown limit has no documented resolution. The system either silently ignores some signals or crashes.

**The fix — `execution/signal_hierarchy.py`:**
```python
class SignalPriority(IntEnum):
    """Higher number = higher authority. Lower authority defers to higher."""
    ADWIN_DRIFT   = 1   # Advisory: reduce size only, never veto
    ML_ENSEMBLE   = 2   # Primary signal source
    HMM_REGIME    = 3   # Blocks trades in sideways regime
    RISK_MANAGER  = 4   # Absolute veto on any signal

@dataclass
class TradingDecision:
    action: Literal['BUY', 'SELL', 'HOLD', 'HALT']
    size_multiplier: float = 1.0   # ADWIN can reduce: 0.25, 0.5, 1.0
    veto_reason: str = ""

def resolve_signal(
    ml_signal: int,            # +1 / 0 / -1
    regime: HMMRegime,         # bull / bear / sideways
    drift_flag: DriftSeverity, # none / moderate / severe
    risk_status: RiskStatus,   # ok / warning / halt
) -> TradingDecision:
    # Priority 4 — Risk Manager is absolute veto
    if risk_status == RiskStatus.HALT:
        return TradingDecision('HALT', veto_reason="Risk manager kill switch")

    # Priority 3 — HMM blocks in sideways
    if regime == HMMRegime.SIDEWAYS and ml_signal != 0:
        return TradingDecision('HOLD', veto_reason="HMM sideways regime")

    # Priority 2 — ML signal is the actual decision
    if ml_signal == 0:
        return TradingDecision('HOLD')

    # Priority 1 — ADWIN is advisory: reduces size, never vetoes
    size_mult = 1.0
    if drift_flag == DriftSeverity.MODERATE: size_mult = 0.5
    if drift_flag == DriftSeverity.SEVERE:   size_mult = 0.25

    action = 'BUY' if ml_signal == 1 else 'SELL'
    return TradingDecision(action, size_multiplier=size_mult)

# RL IS NOT IN THIS HIERARCHY — Phase 4 only, after 12+ months live profitability
```

**Rule:** Log every conflict and its resolution to QuestDB. Review weekly. If HMM is vetoing >40% of ML signals, the regime detector is miscalibrated.

---

### [SURV-5] Feature and Model Freeze — No Research in Production

**The fragility:** During live trading, any of these can silently corrupt results: adding a new feature (changes the feature vector → model mismatch), retraining on same-day data (lookahead), changing a hyperparameter, or running Optuna during market hours (changes the model mid-trade).

**The freeze protocol:**
```python
# models/model_registry.py
@dataclass(frozen=True)
class FrozenModelSpec:
    """Production model is immutable. Retrain only on Sundays after market close."""
    version: str                   # "v2.3.1"
    feature_names: tuple[str, ...]  # ordered tuple — any change = new version
    feature_schema_hash: str       # sha256 of feature_names
    trained_on: date               # training cutoff date
    treelite_so_path: str          # path to compiled .so

    def validate_input(self, X_columns: list[str]) -> None:
        incoming_hash = sha256(str(X_columns).encode()).hexdigest()[:8]
        if incoming_hash != self.feature_schema_hash:
            raise ModelMismatchError(
                f"Feature schema changed. Expected {self.feature_schema_hash}, "
                f"got {incoming_hash}. Load the correct model version."
            )

# RETRAIN SCHEDULE — written in stone
# Sunday 8:00 PM IST: download new week's data
# Sunday 8:30 PM IST: run walk-forward retrain
# Sunday 10:00 PM IST: run validate_edge() — must still pass gate
# Sunday 10:30 PM IST: if gate passes → deploy new FrozenModelSpec
# Monday 9:00 AM IST: production uses new model
# ANY DAY DURING MARKET HOURS: NO retraining, NO Optuna, NO feature changes

MARKET_HOURS = (time(9, 15), time(15, 30))

def is_safe_to_retrain() -> bool:
    now = datetime.now(IST)
    if now.weekday() < 5:  # Monday–Friday
        if MARKET_HOURS[0] <= now.time() <= MARKET_HOURS[1]:
            return False
    return True
```

---

### [SURV-6] State Persistence and Restart Reconciliation

**The fragility:** If the system crashes at 11:32 AM with 2 open positions and 1 pending order, restarting it will have NO memory of these. Without reconciliation: it may open duplicate positions, ignore existing stops, or trade as if flat when it's not. This is how retail algos blow up after a crash.

**Implementation — `data/system_state.py`:**
```python
@dataclass
class LiveSystemState:
    """Persisted to disk every 5 seconds. Loaded on startup."""
    positions: dict[str, Position]        # symbol → Position(qty, entry, stop)
    pending_orders: dict[str, Order]      # order_id → Order
    daily_pnl: float                      # for kill switch check on restart
    daily_trade_count: int
    last_heartbeat: datetime
    model_version: str                    # must match FrozenModelSpec.version

    def save(self, path: str = "state/live_state.json"):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, default=str)

    @classmethod
    def load(cls, path: str = "state/live_state.json") -> "LiveSystemState | None":
        if not Path(path).exists():
            return None
        with open(path) as f:
            return cls(**json.load(f))

class StartupReconciler:
    """Run BEFORE any trading after restart. Broker is the source of truth."""

    def reconcile(self, saved: LiveSystemState, broker: BrokerBase) -> ReconcileResult:
        broker_positions = broker.get_positions()
        mismatches = []

        for symbol, saved_pos in saved.positions.items():
            broker_qty = broker_positions.get(symbol, 0)
            if abs(saved_pos.quantity - broker_qty) > 0:
                mismatches.append(f"{symbol}: saved={saved_pos.quantity}, broker={broker_qty}")

        if mismatches:
            send_alert(
                f"STATE MISMATCH on restart — HALTED. Resolve manually:\n" +
                "\n".join(mismatches),
                level="CRITICAL"
            )
            return ReconcileResult.HALT_UNTIL_OPERATOR_CONFIRMS
            # NEVER auto-trade until operator explicitly confirms state

        if saved.daily_pnl <= -DAILY_LOSS_LIMIT:
            return ReconcileResult.DAILY_LIMIT_ALREADY_HIT

        return ReconcileResult.OK_TO_TRADE
```

**State persistence rule:** `system_state.save()` is called after every order event (placed, filled, cancelled, rejected) and on a 5-second heartbeat timer. The `last_heartbeat` field detects if the process hung silently.

---

### [SURV-7] ZeroMQ Message Integrity — Silent Drops Are Not Silent

**The fragility:** ZMQ PUSH/PULL and PUB/SUB can silently drop messages when: SNDHWM is exceeded (default 1000), subscriber is slow (PUB will drop oldest messages), network blip, or process restart. The FeatureComputer will compute features on a 10-message-old tick and not know. The inference engine will predict on stale features. The order manager will trade on a ghost signal.

**Implementation — add to `data/websocket_feed.py` and `features/engine.py`:**
```python
@dataclass
class SequencedTick:
    """Every tick gets a sequence number. Gaps = halt."""
    seq: int
    symbol: str
    timestamp_ns: int
    ltp: float
    volume: int
    # ... other fields

class SequenceGuard:
    """Detects ZMQ message drops via monotonic sequence numbers."""
    MAX_ALLOWED_GAP = 3   # >3 consecutive drops = halt trading

    def __init__(self):
        self._expected: dict[str, int] = {}  # symbol → next expected seq

    def check(self, tick: SequencedTick) -> bool:
        expected = self._expected.get(tick.symbol, tick.seq)

        if tick.seq < expected:
            logging.warning("%s: old message (seq %d, expected %d) — ignoring",
                           tick.symbol, tick.seq, expected)
            return False  # Drop silently — this is an old/duplicate message

        if tick.seq > expected:
            gap = tick.seq - expected
            logging.error("%s: ZMQ drop — %d messages lost (seq %d→%d)",
                         tick.symbol, gap, expected, tick.seq)
            if gap > self.MAX_ALLOWED_GAP:
                send_alert(f"ZMQ data gap: {gap} ticks dropped on {tick.symbol}", level="CRITICAL")
                return False  # Halt feature computation for this symbol

        self._expected[tick.symbol] = tick.seq + 1
        return True
```

**ZMQ socket settings — set these on ALL sockets:**
```python
# Publisher (DataIngestor)
pub_socket.setsockopt(zmq.SNDHWM, 1000)   # drop oldest if queue > 1000
pub_socket.setsockopt(zmq.LINGER, 0)

# Subscriber (FeatureComputer)
sub_socket.setsockopt(zmq.RCVHWM, 1000)
sub_socket.setsockopt(zmq.RCVTIMEO, 5000) # 5 sec timeout — detect dead publisher

# PUSH (InferenceEngine)
push_socket.setsockopt(zmq.SNDTIMEO, 100) # 100ms timeout — don't block execution
```

---

### [SURV-8] NSE-Specific Realism — Four Rules That Cost Real Money

**The fragility:** The architecture models Indian costs (STT, GST, brokerage) correctly but misses four operational realities specific to NSE intraday trading that will cause repeated losses.

**Rule 1 — Never trade the opening auction candle (9:15–9:30):**
```python
# The 9:15–9:30 bar contains pre-open call auction clearing prices (9:00–9:08)
# mixed with continuous trading. Volume, spreads, and volatility are 3–5x normal.
# EMA signals in this candle have NEGATIVE expectancy after costs.
SKIP_CANDLES_BEFORE = time(9, 30)  # First tradeable bar = 9:30 bar (second candle)

def is_tradeable_time(ts: datetime) -> bool:
    t = ts.time()
    return time(9, 30) <= t <= time(15, 15)  # Also skip last 15min (squared off by exchange)
```

**Rule 2 — NSE circuit breaker risk is a position risk:**
```python
# NSE freezes individual stocks at ±5%, ±10%, ±20% intraday.
# If you're long and stock freezes at lower circuit: you CANNOT exit.
# Your stop-loss order will be rejected. You hold to end of day.
# Model this as: any position has a 0.5% daily probability of being locked in a circuit.
# Risk management: max single position = 5% of capital even more critical (not just preference)

def adjust_for_circuit_risk(position_size: int, stock_price: float, capital: float) -> int:
    max_position_value = capital * 0.05  # Hard cap: 5% of capital
    max_shares = int(max_position_value / stock_price)
    return min(position_size, max_shares)
```

**Rule 3 — Adverse selection cost is real and must be in the cost model:**
```python
# LIMIT orders that fill are adversely selected:
# When your BUY limit fills, it usually means price was moving DOWN to you.
# Expected adverse selection for NSE retail intraday LIMIT: 3–8 bps per side.
# Add to cost model. If your edge < 16bps per round trip, it's probably adverse selection.

ADVERSE_SELECTION_BPS = 5.0   # Add to each side of LIMIT order cost

def total_round_trip_cost_bps(price, qty, segment='intraday') -> float:
    base = calculate_trade_cost(price, qty, segment)  # existing function
    adverse_selection = 2 * ADVERSE_SELECTION_BPS * price * qty / 10000
    return base + adverse_selection
```

**Rule 4 — Square off by 3:15 PM, not 3:30 PM:**
```python
# NSE auto-squares open positions at 3:20–3:25 PM at MARKET price.
# Broker auto-square charges are HIGHER than manual square (extra brokerage + penalty).
# Auto-square often happens at worst price of the day.
# Hard rule: close ALL intraday positions by 3:15 PM latest.
INTRADAY_SQUARE_OFF_TIME = time(15, 15)  # 15 min buffer before exchange closes
```

---

### [SURV-9] Microstructure Scope Reduction — Right Tool, Right Timeframe

**The fragility:** Hawkes Process models inter-arrival times of orders at millisecond scale. Trade Flow Imbalance measures order flow at tick level. At a 15-minute strategy timeframe, both of these measure phenomena that resolve 3,000× faster than your decision interval. The output is mathematical noise dressed as signal.

**The fix — microstructure scope at 15-min:**

| Model | Minimum Useful Timeframe | Keep at 15-min? | Reason |
|---|---|---|---|
| Hawkes Process | < 1 second bars | **NO — REMOVE** | Inter-arrival times are meaningless at 15-min aggregation |
| Trade Flow Imbalance (TFI) | < 10 second bars | **NO — REMOVE** | Multi-level OFI signal decays in seconds, not minutes |
| VPIN | 5–30 min volume buckets | **YES — keep as filter** | 15-min volume bars align with VPIN bucket logic |
| Avellaneda-Stoikov | Any | **YES — keep** | Reservation price useful for LIMIT order placement at any timeframe |

**Updated `features/microstructure.py` scope:**
```python
# AT 15-MINUTE TIMEFRAME — these are the ONLY valid microstructure features:

def compute_vpin(volume_bars: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
    """
    VPIN > 0.35 = toxic order flow = high adverse selection risk.
    Use as a FILTER: if VPIN > 0.35, reduce position size by 50% or skip trade.
    This is valid at 15-min because VPIN aggregates over volume buckets (~15 min each).
    """
    ...

def compute_reservation_price(mid: float, inventory: int, gamma: float,
                               sigma: float, T_remaining: float) -> float:
    """
    Avellaneda-Stoikov: r = mid - inventory * gamma * sigma^2 * T_remaining
    Valid for LIMIT order price adjustment at any timeframe.
    Use when placing LIMIT orders: adjust limit price by reservation price offset.
    """
    return mid - inventory * gamma * (sigma ** 2) * T_remaining

# Hawkes and TFI: moved to Stage 3+ only, when using sub-minute bars
# See [CDR-P1] for their implementation if/when needed
```

**Replace Hawkes/TFI features with:**
- `VPIN_bucket_filter` — adds directly to feature set as binary flag
- `reservation_price_offset` — used in order placement, not ML features
- Freed feature slots: add `bid_ask_spread_percentile` (rolling 20-bar) and `volume_vs_20bar_avg` as cleaner alternatives

---

### [SURV-10] Order State Machine — Prevent Zombie Orders

**The fragility:** Rate limiting, partial fills, and restart reconciliation are documented, but there is no single authoritative state machine governing order lifecycle. Without one: a timeout retry creates a duplicate order while the original is still live (zombie); an ACK received after a cancel creates an "open" position with no stop; a partial fill transitions to FILLED in internal state while the broker still has an open remainder. These are the #1 way retail algos blow up.

**Explicit FSM — implement in `execution/order_manager.py`:**
```python
class OrderState(str, Enum):
    NEW       = "NEW"       # Created locally, not yet sent
    SENT      = "SENT"      # Submitted to broker, awaiting ACK
    ACK       = "ACK"       # Broker confirmed receipt (not filled yet)
    PARTIAL   = "PARTIAL"   # Partially filled, remainder open
    FILLED    = "FILLED"    # Completely filled — terminal
    CANCELLED = "CANCELLED" # Cancelled by us or broker — terminal
    REJECTED  = "REJECTED"  # Broker rejected (bad params, limits, etc.) — terminal
    TIMEOUT   = "TIMEOUT"   # No ACK within timeout window — terminal (treat as CANCELLED)

# Valid transitions — any other transition is a bug
VALID_TRANSITIONS = {
    OrderState.NEW:     {OrderState.SENT},
    OrderState.SENT:    {OrderState.ACK, OrderState.REJECTED, OrderState.TIMEOUT},
    OrderState.ACK:     {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.PARTIAL: {OrderState.FILLED, OrderState.CANCELLED},
    # FILLED, CANCELLED, REJECTED, TIMEOUT have no outgoing transitions
}

def transition(order: Order, new_state: OrderState) -> Order:
    if new_state not in VALID_TRANSITIONS.get(order.state, set()):
        raise InvalidStateTransitionError(
            f"Order {order.id}: {order.state} → {new_state} is not a valid transition"
        )
    order.state = new_state
    order.state_updated_at = time.time_ns()
    system_state.save()        # Persist every transition
    rate_limiter.release_order(order.id) if new_state in TERMINAL_STATES else None
    return order

TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELLED,
                   OrderState.REJECTED, OrderState.TIMEOUT}
```

**Idempotency — use `client_order_id`:**
```python
# Every order gets a unique client_order_id = f"{symbol}_{strategy_id}_{timestamp_ns}"
# Before placing: check if an order with this client_order_id is already SENT or ACK
# If yes: broker already has it — do NOT re-send. This prevents duplicate orders on retry.
# Shoonya and most NSE brokers support client_order_id for deduplication.

def place_order_idempotent(order: Order, broker: BrokerBase) -> Order:
    existing = system_state.find_by_client_id(order.client_order_id)
    if existing and existing.state in {OrderState.SENT, OrderState.ACK}:
        logging.warning("Duplicate detected: %s already in state %s",
                        order.client_order_id, existing.state)
        return existing   # Return existing, do NOT re-send
    return broker.place_order(order)
```

**Timeout policy:**
```python
ACK_TIMEOUT_SECONDS = 10   # If no ACK in 10s → TIMEOUT → treat as CANCELLED
                            # Do NOT assume filled. Do NOT retry. Halt symbol.
PARTIAL_TIMEOUT_SECONDS = 300  # 5 min partial fill still open → cancel remainder
```

---

### [SURV-11] Smart Execution Policy — Replace Naive LIMIT→MARKET Rule

**The fragility:** The current rule `"MARKET only if LIMIT not filled within 2 candles"` has one fatal flaw at 15-min bars: 2 candles = 30 minutes. If a momentum signal triggers at 9:45, the LIMIT doesn't fill (price moved away), and at 10:15 you chase with a MARKET order — you're now entering a move that's 30 minutes old, at or near the top, with maximum slippage. You've engineered a way to buy highs and sell lows.

**The fix: price-anchored execution with a stale-signal abort.**
```python
@dataclass
class ExecutionPolicy:
    """
    Execution aggressiveness is driven by signal strength and price movement,
    NOT by a fixed candle countdown.
    """
    # How far price can move from signal price before we abort entirely
    MAX_PRICE_CHASE_BPS: float = 30.0   # Never enter if price moved >30bps from signal

    # How long a LIMIT order lives before being cancelled (NOT converted to MARKET)
    LIMIT_LIFETIME_BARS: int = 1        # If not filled in 1 candle (15 min) → cancel

    # Stop-losses: always MARKET, no exceptions, no price cap
    STOP_LOSS_ORDER_TYPE = "MARKET"

def get_limit_price(signal_price: float, direction: str,
                    signal_strength: float, atr: float) -> float:
    """
    Place LIMIT at a price that has a realistic fill probability.
    More aggressive (closer to market) when signal is stronger.
    """
    # signal_strength: 0.5 (weak) to 1.0 (strong)
    # tick_offset: how many ticks inside the spread to post
    tick_size = get_tick_size(signal_price)
    aggressiveness = (signal_strength - 0.5) * 2   # maps [0.5, 1.0] → [0.0, 1.0]
    ticks_inside = int(aggressiveness * 3)           # 0–3 ticks inside best bid/ask

    if direction == "BUY":
        return signal_price - (3 - ticks_inside) * tick_size  # passive=3 ticks below, aggressive=at best ask
    else:
        return signal_price + (3 - ticks_inside) * tick_size

def should_abort_entry(signal_price: float, current_price: float,
                       direction: str, max_chase_bps: float) -> bool:
    """
    If price has already moved too far from signal price: abort.
    NEVER chase with MARKET. Missing a trade is free. Chasing costs money.
    """
    move_bps = abs(current_price - signal_price) / signal_price * 10000
    if direction == "BUY" and current_price > signal_price:   # price ran up
        return move_bps > max_chase_bps
    if direction == "SELL" and current_price < signal_price:  # price ran down
        return move_bps > max_chase_bps
    return False   # Price moved favorably — still worth entering

# THE RULE: if LIMIT not filled in 1 candle → cancel → do NOT convert to MARKET
# Missing the trade is free. Chasing with MARKET is a guaranteed loss on momentum moves.
```

**Stop-loss exception:** Stop-losses are ALWAYS MARKET with no price cap. The asymmetry is intentional — entering is optional, protecting capital is not.

---

### [SURV-12] ADWIN Guardrails — Drift Adaptation Is a Double-Edged Sword

**The fragility:** ADWIN (Adaptive Windowing) detects distribution shifts by monitoring prediction error rate. In volatile markets (NSE expiry week, budget day, geopolitical events), ADWIN will fire repeatedly. Each firing could degrade your model by "adapting" to transient noise. After the volatile event passes, you're left with a model that has learned the anomaly as normal — and you keep trading with it until someone notices.

**Guardrails — add to `models/online_model.py`:**
```python
@dataclass
class ADWINGuardrails:
    # Minimum samples before any adaptation is permitted
    MIN_BARS_BEFORE_ADAPT: int = 200        # ~10 trading days of 15-min bars

    # Cooldown: minimum days between adaptations (prevents rapid-fire retraining)
    MIN_DAYS_BETWEEN_ADAPT: int = 5         # max ~1 adaptation per week

    # Hard freeze: no adaptation when market is abnormal
    FREEZE_WHEN_ATR_RATIO: float = 2.0      # ATR > 2x 20-day baseline = frozen

    # Validation gate: after adaptation, the model must still pass edge gate
    REQUIRE_EDGE_REVALIDATION: bool = True  # validates on last 60 bars before deploying

class SafeADWINAdapter:
    def __init__(self, base_model, guardrails: ADWINGuardrails):
        self.model = base_model
        self.g = guardrails
        self.adwin = ADWIN()
        self.last_adapt_date: date | None = None
        self.bars_since_last_adapt: int = 0

    def update(self, prediction_error: float, current_atr: float,
               baseline_atr: float) -> AdaptationResult:
        self.adwin.update(prediction_error)
        self.bars_since_last_adapt += 1

        if not self.adwin.detected_change:
            return AdaptationResult.NO_CHANGE

        # Guardrail 1: not enough data
        if self.bars_since_last_adapt < self.g.MIN_BARS_BEFORE_ADAPT:
            return AdaptationResult.BLOCKED_INSUFFICIENT_DATA

        # Guardrail 2: cooldown period
        if self.last_adapt_date:
            days_since = (date.today() - self.last_adapt_date).days
            if days_since < self.g.MIN_DAYS_BETWEEN_ADAPT:
                return AdaptationResult.BLOCKED_COOLDOWN

        # Guardrail 3: high-volatility freeze
        if current_atr / baseline_atr > self.g.FREEZE_WHEN_ATR_RATIO:
            logging.warning("ADWIN drift detected but ATR=%.1fx baseline — FROZEN",
                           current_atr / baseline_atr)
            return AdaptationResult.BLOCKED_HIGH_VOL

        # Guardrail 4: validate before deploying adapted model
        if self.g.REQUIRE_EDGE_REVALIDATION:
            if not passes_edge_gate_on_recent_data(self.model, lookback_bars=60):
                logging.error("Post-adaptation edge validation failed — reverting")
                return AdaptationResult.BLOCKED_EDGE_VALIDATION

        # All guardrails passed — allow adaptation
        self.last_adapt_date = date.today()
        self.bars_since_last_adapt = 0
        return AdaptationResult.ADAPTED
```

**Rule:** If ADWIN is firing more than once per week: the model has fundamental problems that adaptation cannot fix. Investigate signal quality, not adaptation parameters.

---

## Build Order (follow this sequence)

> **Stage 0 (step 0):** Prove simple edge exists before building anything. Gate: Sharpe ≥ 1.0 [SURV-1]
> **Stage 1 (steps 1–10):** Shoonya feed, EMA cross paper trade. Do NOT build ML or microstructure.
> **Stage 2 (steps 11–30):** TrueData feed required before enabling microstructure features.

0. **`scripts/validate_edge.py`** → prove EMA cross Sharpe ≥ 1.0 on 3yr walk-forward **BEFORE ANY OTHER STEP** [SURV-1]
1. `config.py` + `.env.example` + `requirements.txt` — `DATA_FEED=shoonya`, `SKIP_FIRST_CANDLE=true`
2. `data/broker_base.py` → `data/broker_shoonya.py` (EXECUTION only) → test connection
3. `data/candle_store.py` → QuestDB (docker run -p 9000:9000 questdb/questdb)
4. `data/ring_buffer.py` → numpy ring buffer [P2I-2.2]
5. `data/integrity_guard.py` → TickValidator + DataQualityMonitor [SURV-2]
6. `data/system_state.py` → LiveSystemState persistence + StartupReconciler [SURV-6]
7. `data/websocket_feed.py` → Stage 1: wrap Shoonya WebSocket, ZMQ PUB:5555 + SequenceGuard [SURV-7]
8. `features/technical.py` → pandas-ta indicators (skip 9:15–9:30 bar [SURV-8])
9. `features/statistical.py` → HMM, Kalman, GARCH
10. `features/custom.py` → Volume Clock, numba OBI, relative strength, ToD, spread
11. `features/engine.py` → orchestrate all features, ZMQ SUB/PUB:5556
12. `labeling/triple_barrier.py` → symmetric ±1.5 ATR, sample weights
13. `labeling/walk_forward.py` → purge+embargo (51 bars minimum)
14. `risk/cost_model.py` → India costs + adverse selection [SURV-8] + dynamic slippage [SURV-3]
15. `risk/position_sizer.py` → Half-Kelly + ATR-based + circuit breaker cap [SURV-8]
16. `risk/manager.py` → kill switches + drawdown breaker
17. `execution/signal_hierarchy.py` → RiskMgr→HMM→ML→ADWIN priority [SURV-4]
18. `execution/order_manager.py` → FSM state machine [SURV-10] + smart execution policy [SURV-11] + DualLockRateLimiter [ARCH-FIX-5] + square-off-by-3:15 [SURV-8]
19. `execution/latency_profiler.py` → measure all 5 latency components [SURV-3]
20. `execution/trade_logger.py` → QuestDB trade log + P&L attribution fields [SURV-11]
21. `models/ensemble.py` → train XGB+LGB+CatBoost; `compile_for_production()` after training [ARCH-FIX-2]
22. `models/inference_engine.py` → Treelite .so loader, ZMQ SUB→PUSH:5557 [ARCH-FIX-2]
23. `models/model_registry.py` → FrozenModelSpec + feature schema hash + retrain schedule [SURV-5]
24. `models/online_model.py` → SafeADWINAdapter with guardrails [SURV-12]
25. `models/tuner.py` → Optuna (50 trials max, fold[-2] rule) — Sunday only [SURV-5]
26. `models/tracker.py` → MLflow + DVC
27. `models/explainer.py` → SHAP after training (SEBI requirement)
28. `alerts/alert_sink.py` → ZMQ PULL worker subprocess [ARCH-FIX-3]
29. `alerts/telegram_bot.py` → ZMQ PUSH sender (non-blocking) [ARCH-FIX-3]
30. `backtest/runner.py` → walk-forward backtest with full cost model
31. `main.py` → StartupReconciler → 3-process launcher → alert_sink subprocess
32. `tests/` → pytest for each module
33. `scripts/` → utility scripts
34. **Stage 2 upgrade:** `data/truedata_feed.py` → update websocket_feed.py, set `DATA_FEED=truedata` [ARCH-FIX-1]
35. **Stage 2 upgrade:** `features/microstructure.py` → **VPIN filter + Avellaneda-Stoikov ONLY** — Hawkes/TFI removed at 15-min [SURV-9]

## Common Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Start QuestDB (run once, keeps running)
docker run -d -p 9000:9000 -p 9009:9009 --name questdb questdb/questdb

# Initialize DVC (run once in project root)
dvc init && dvc add data/

# Run paper trading
python main.py --mode paper

# Run backtest
python scripts/train_models.py --symbol RELIANCE --timeframe 15min

# Run tests
pytest tests/ -v

# Start MLflow UI
mlflow ui --port 5000

# Generate SHAP explainability report (SEBI compliance)
python models/explainer.py --model models/saved/xgb_latest.pkl
```

## Environment Variables (.env)
```
# ── Execution Broker (PRIMARY) — Shoonya/Finvasia ──────────────────────────
# Role: order placement, cancellation, position queries ONLY
SHOONYA_API_KEY=your_key
SHOONYA_USER_ID=your_id
SHOONYA_PASSWORD=your_pass
SHOONYA_TOTP_SECRET=your_totp
SHOONYA_VENDOR_CODE=your_vendor_code
SHOONYA_IMEI=your_imei

# ── Data Feed (Stage 1: Shoonya throttled | Stage 2+: TrueData unthrottled) ─
# Stage 1: DATA_FEED=shoonya — EMA cross only, no microstructure
# Stage 2+: DATA_FEED=truedata — enables Hawkes/VPIN/TFI [ARCH-FIX-1]
DATA_FEED=shoonya

# TrueData credentials (activate at Stage 2 — ~Rs. 300-500/month for NSE)
TRUEDATA_USER=your_truedata_username
TRUEDATA_PASSWORD=your_truedata_password

# ── Dev Reference Only — Angel One SmartAPI ────────────────────────────────
ANGEL_API_KEY=your_key
ANGEL_CLIENT_ID=your_id
ANGEL_PASSWORD=your_pass
ANGEL_TOTP_SECRET=your_totp

# ── Alerts ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ALERT_SINK_PORT=5560           # ZMQ PUSH/PULL port for async alert delivery [ARCH-FIX-3]

# ── Tracking ───────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=./mlruns
```

---

## Reference Index — Required Reading by Module

> Before building any module, find its IDs below and read those sections first.
> Format: [FILE-PARTNUM] = file > section to read.
> **Start every session:** `@workspace Read arch-spec.json. Read MD_Files/CLAUDE.md.`

### Knowledge Base Files

| ID | File (`MD_Files/`) | Lines | What It Contains |
|---|---|---|---|
| **CDR** | `CONVO_DEEP_RESEARCH.md` | 1,500+ | §Part1: microstructure math (Hawkes/VPIN/TFI/AS code). §Part2: infra summary. §Part3A-B: all 24 critical bug fixes with code. §Part4: 4-stage deployment + exit gates. §Part5: stacking ensemble, HMM fix, Kelly sizing. §Part6: arch-spec overview |
| **MRD** | `MASTER_RESEARCH_DATABASE_v3.md` | 8,300+ | 30+ sections: GitHub repos by stars, 300+ academic papers (ML/RL/DL/NLP/microstructure/India), books, all Indian broker API comparisons, NSE data sources, SEBI 2025–2026 compliance, backtesting frameworks deep comparison, RL papers (100+) |
| **P2I** | `_part2_infrastructure.md` | 551 | ZeroMQ 3-process pipeline code, numpy RingBuffer class, VolumeBarAggregator class, Volume Clock encoding, numba OBI, daily VPS restart systemd |
| **P6A** | `_part6_aidev.md` | 445 | Full `arch-spec.json`, 5 AI prompt templates (feature/bug/refactor/test/perf), ML model selection matrix by stage (River/TSMixer/DART — TabPFN removed [ARCH-FIX-4]), Elite Research Prompt |
| **IRF** | `IMPROVEMENT_RESEARCH_FEEDBACK.md` | 701 | 15 critical research questions + answers, gap analysis, 5 high-value additions (fractional diff, circuit breaker, adaptive Kalman, GARCH open re-estimation, dynamic ensemble weighting) |
| **SCG** | `SETUP_GUIDE_COMPLETE.md` | 1,152 | Full env setup, 25-module Claude Code prompt playbook, Jupyter research workflow, India cost model (STT+GST+exchange+slippage), platform comparison (NautilusTrader/Qlib/FinRL/VectorBT), SEBI checklist |
| *(PPL)* | *(Merged into CDR Part 5B)* | — | Project plan sessions 56–94 (post-core upgrades) are now in CDR §"Optional Upgrades" |

### File Abbreviations (for reference IDs below)
- **CDR** = CONVO_DEEP_RESEARCH.md
- **MRD** = MASTER_RESEARCH_DATABASE_v3.md
- **P2I** = _part2_infrastructure.md
- **P6A** = _part6_aidev.md
- **SCG** = SETUP_GUIDE_COMPLETE.md
- **IRF** = IMPROVEMENT_RESEARCH_FEEDBACK.md
- **PPL** (merged) = Project plan content is in CDR §"Optional Upgrades After Core System Is Live"

### Module → Reference Map

| Module File | Reference IDs | Topic |
|---|---|---|
| `data/broker_shoonya.py` | [CDR-3B-C24], [P2I-2.1] | **EXECUTION ONLY** — SessionManager TOTP heartbeat, place/cancel/status |
| `data/broker_angelone.py` | [CDR-3A-C1] | Angel One SmartAPI (dev reference only, NOT production) |
| `data/ring_buffer.py` | [P2I-2.2], [CDR-3A-C6] | Numpy structured ring buffer — full implementation in P2I §2.2 |
| `data/integrity_guard.py` | [SURV-2] | TickValidator (outlier/gap/stale/dup detection) + DataQualityMonitor (halt if >5% bad) |
| `data/system_state.py` | [SURV-6] | LiveSystemState persistence (5s heartbeat) + StartupReconciler (halt if broker mismatch) |
| `data/truedata_feed.py` | [ARCH-FIX-1] | **Stage 2+ only** — TrueData unthrottled L2 WebSocket, required for microstructure features |
| `data/websocket_feed.py` | [P2I-2.1], [CDR-3A-C6], [ARCH-FIX-1], [SURV-7] | DataIngestor PUB:5555 + SequenceGuard (ZMQ drop detection) |
| `data/candle_store.py` | [P2I-2.3], [CDR-3A-C2] | VolumeBarAggregator — full class with calibrate_from_history() in P2I §2.3 |
| `features/microstructure.py` | [CDR-P1], [MRD-S06], [SURV-9] | **VPIN filter + Avellaneda-Stoikov ONLY at 15-min** — Hawkes/TFI removed (wrong timescale) |
| `features/technical.py` | [MRD-S01-S12], [SCG-FEAT] | 23 technical indicators via pandas-ta |
| `features/statistical.py` | [CDR-P1-AS], [CDR-3B-C16], [CDR-3B-C21], [IRF-Q5] | HMM regime (min-duration fix), GARCH (open re-estimation), OFI Kalman |
| `features/custom.py` | [P2I-2.4], [P2I-2.5], [CDR-3A-C3] | Volume Clock encoding + numba OBI — code in P2I §2.4–2.5 |
| `features/engine.py` | [P2I-2.1], [CDR-P2-ZMQ] | FeatureComputer SUB/PUB:5556 — ZeroMQ code in P2I §2.1 |
| `labeling/triple_barrier.py` | [CDR-3A-C12], [CDR-P5-TB] | Symmetric ±1.5 ATR, sample weights |
| `labeling/walk_forward.py` | [CDR-3B-C17], [CDR-P5-PW] | Purge window 51 bars (MACD) / 77 bars (Ichimoku), TimeSeriesSplit |
| `models/model_registry.py` | [SURV-5] | FrozenModelSpec + feature schema hash + retrain schedule (Sunday 8PM only) |
| `models/ensemble.py` | [CDR-P5-STACK], [CDR-3B-C15], [MRD-S14-S18] | Stacking: XGB+LGB+CatBoost OOF → LR meta-learner; calls compile_for_production() after training |
| `models/online_model.py` | [P6A-6.4] | River ADWIN online learning — drift adaptation, O(1)/sample |
| `models/inference_engine.py` | [P2I-2.1], [CDR-P2-ZMQ], [ARCH-FIX-2] | **Treelite compiled .so inference** — ~2ms vs ~40ms native; ZeroMQ SUB→PUSH:5557 |
| `models/tuner.py` | [CDR-3B-C23], [MRD-S19] | Optuna fold[-2] rule, complexity penalty (not fold[-1]) |
| `models/rl_sizer.py` | [CDR-3B-C18] | NotImplementedError — Phase 4 OPTIONAL only |
| `models/sentiment.py` | [CDR-3B-C20] | Async SentimentWorker via ZeroMQ PUSH/PULL |
| `risk/position_sizer.py` | [CDR-3B-C18] | Kelly f*, Half-Kelly, ATR-based, target-vol sizing |
| `risk/manager.py` | [CDR-P4-S1], [IRF-Q15] | Kill switches, 2% daily loss, 10% DD, circuit breaker handler |
| `risk/cost_model.py` | [SCG-COST], [CDR-3A-C9] | Dynamic slippage + India STT + exchange + GST + slippage |
| `execution/signal_hierarchy.py` | [SURV-4] | SignalPriority enum — RiskMgr(4)→HMM(3)→ML(2)→ADWIN(1) — resolve_signal() |
| `execution/latency_profiler.py` | [SURV-3] | LatencySnapshot (5 components), logs to QuestDB, alert if >2x p95 |
| `execution/order_manager.py` | [CDR-3A-C4], [CDR-3B-C19], [ARCH-FIX-5], [SURV-3], [SURV-8], [SURV-10], [SURV-11] | FSM state machine + DualLockRateLimiter + smart execution policy + 3:15 PM square-off |
| `execution/trade_logger.py` | [SURV-11] | QuestDB trade log + P&L attribution: slippage_bps, spread_cost_bps, latency_ms, partial_fill_ratio, abort_reason |
| `alerts/telegram_bot.py` | [ARCH-FIX-3] | **Non-blocking ZMQ PUSH** — fire-and-forget, execution never waits for Telegram |
| `alerts/alert_sink.py` | [ARCH-FIX-3] | Dedicated ZMQ PULL worker process — only code that calls Telegram HTTP API |
| `models/explainer.py` | [MRD-S22], [MRD-S14] | SHAP values, SEBI explainability requirement |
| `backtest/runner.py` | [CDR-P4], [MRD-S27], [SCG-COST] | Walk-forward backtest with full cost model |
| `main.py` | [P2I-2.6], [CDR-P4-S1], [ARCH-FIX-3] | 3-process launcher + alert_sink subprocess, SIGUSR1 handler, memory trim |
| `arch-spec.json` | [P6A-6.2] | **Full machine-readable spec — copy from P6A §6.2** |

### Algorithm Quick-Reference

| Algorithm | Reference | One-Line Description |
|---|---|---|
| Hawkes Process | [CDR-P1-HP] | Self-exciting order arrival — **Stage 3+ only, sub-minute bars required** [SURV-9] |
| VPIN | [CDR-P1-VP] | Volume-sync PIN, VPIN>0.35 = toxic flow — **use as filter at 15-min, valid** [SURV-9] |
| TFI | [CDR-P1-TF] | Trade flow imbalance — **Stage 3+ only, tick-level data required** [SURV-9] |
| Avellaneda-Stoikov | [CDR-P1-AS] | r(s,t)=s−q·γ·σ²·(T−t) reservation price — **valid at all timeframes** [SURV-9] |
| Volume Bars | [CDR-P2-VB] | Dynamic threshold = ADV/target_bars |
| Volume Clock | [CDR-3B-C13] | Cumulative volume percentile, not sin/cos time |
| Triple Barrier | [CDR-3A-C10] | ±1.5 ATR symmetric, sample_weight from label dist |
| Stacking Ensemble | [CDR-P5-ST] | OOF + LogisticRegression meta, TimeSeriesSplit |
| HMM Regime | [CDR-3B-C16] | min_duration post-proc, sticky prior=0.95 |
| Kelly Sizing | [CDR-3B-C18] | f*=(p×b−q)/b, Half-Kelly for live trading |
| Purge Window | [CDR-3B-C17] | Min 51 bars = holding(20)+MACD(26)+embargo(5); 77 bars with Ichimoku(52) |
| ZeroMQ Pipeline | [CDR-P2-ZMQ] | 3-process: Ingestor→Feature→Inference, bypasses GIL |
| OFI Kalman | [CDR-3B-C22] | State=[price,OFI] replaces Newtonian [price,vel] |

### Research Papers Referenced

| Paper | Reference | Relevance |
|---|---|---|
| López de Prado: AFML (2018) | [MRD-BOOK1] | Triple-barrier, CPCV, purge/embargo, fractional diff — MUST READ |
| López de Prado: MLFM (2020) | [MRD-BOOK1] | Feature importance, clustering, portfolio construction |
| Robert Carver: Systematic Trading | [MRD-BOOK2] | Position sizing, diversification, systematic approach |
| Easley, LdP, O'Hara: VPIN (2012) | [CDR-P1-VP] | Volume-Synchronized PIN — order flow toxicity |
| Avellaneda & Stoikov (2008) | [CDR-P1-AS] | Market making model — reservation price formula |
| Hawkes (1971) | [CDR-P1-HP] | Self-exciting point processes — trade burst detection |
| Cont (2001): Price dynamics | [MRD-S06] | Order flow and stock return distributions |
| Kyle (1985): Lambda | [CDR-P1-KL] | Price impact model — lambda = market depth |
| XGBoost (Chen & Guestrin 2016) | [MRD-S14] | Primary ML model — hist method for CPU |
| LightGBM (Ke et al. 2017) | [MRD-S15] | Secondary ensemble — DART variant for noisy regimes |
| CatBoost (Prokhorenkova et al. 2018) | [MRD-S16] | Tertiary ensemble — handles categoricals natively |
| Kelly Criterion (Kelly 1956) | [CDR-3B-C18] | Optimal fraction formula: f*=(p×b−q)/b |
| Stable-Baselines3 (Raffin et al. 2021) | [MRD-S03] | RL algorithms (PPO) — Phase 4 OPTIONAL only |
| FinRL (Liu et al. NeurIPS 2020) | [MRD-S03] | RL for financial trading — reference architecture |
| Microsoft Qlib (Yang et al. 2020) | [MRD-S01] | Full AI quant pipeline — architecture reference |
| TSMixer (Chen et al. 2023) | [P6A-6.4] | Linear mixer for time series, O(N) complexity — Stage 3+ |
| SEBI Circular Feb 4 2025 | [SCG-SEBI], [MRD-S14] | Strategy ID mandatory, 10 OPS limit |
| NSE Operational Modalities Jul 2025 | [MRD-S14] | Static IP, OPS limits implementation |

### Open Source Tools Referenced

| Tool | Reference | Purpose |
|---|---|---|
| Shoonya ShoonyaApi-py | [CDR-3B-C24] | **PRIMARY execution broker** — zero brokerage; session expires daily, handle with deterministic re-login [SURV-10] |
| TrueData (`truedata-ws`) | [ARCH-FIX-1], [MRD-S13] | **Stage 2+ data feed** — unthrottled L2 tick-by-tick, REQUIRED for Hawkes/VPIN/TFI |
| Angel One SmartAPI | [CDR-3A-C1] | Development reference only (best docs) |
| QuestDB | [P2I-2.2], [MRD-S27] | Time-series storage, ILP protocol, write-only on hot path |
| ZeroMQ (pyzmq) | [P2I-2.1] | Inter-process communication — GIL bypass — code in P2I §2.1 |
| msgpack | [P2I-2.1] | Zero-copy tick serialization between ZMQ processes |
| Polars | [MRD-S08] | Feature engineering — NO pandas on hot path |
| numba | [P2I-2.5] | JIT-accelerated OBI computation (50-200x over pure Python) |
| hmmlearn | [CDR-3B-C16] | HMM regime detection with min-duration fix |
| arch | [MRD-S13], [IRF-Q5] | GARCH(1,1) volatility forecasting |
| filterpy | [CDR-3B-C21] | Kalman filter — OFI-augmented state |
| pandas-ta | [MRD-S09] | Technical indicators (pandas-based) |
| Optuna | [CDR-3B-C23], [MRD-S19] | Bayesian HPO — max 50 trials, complexity penalty |
| MLflow | [MRD-S20] | Experiment tracking + model versioning |
| DVC | [MRD-S20] | Data and model version control |
| SHAP | [MRD-S22] | Feature importance + SEBI explainability compliance |
| Treelite | [ARCH-FIX-2] | **Production inference compiler** — AOT-compiles XGB/LGB/CatBoost to .so, ~2ms vs ~40ms |
| River | [P6A-6.4] | Online learning (ADWIN) for regime drift adaptation — O(1)/sample, replaces TabPFN |
| neuralforecast (TSMixer) | [P6A-6.4] | Stage 3+ deep temporal model — code in P6A §6.4 |
| APScheduler | [P2I-2.6] | Session scheduling (pre-open restart, daily cron) |
| structlog | [P6A-6.2] | Structured logging with timestamp_ns + symbol |
| NautilusTrader | [MRD-S01], [SCG-PLAT] | End-goal production platform (Rust+Python) |
| VectorBT | [MRD-S27], [SCG-PLAT] | Fast vectorized backtesting for research |
| PyBroker | [MRD-S27] | Walk-forward ML backtesting built-in |

### Improvement Research Topics (IRF)

Open questions with research answers — see `IMPROVEMENT_RESEARCH_FEEDBACK.md` for full answers.

| Topic | Reference | One-Line Answer |
|---|---|---|
| Triple-barrier label distribution check | [IRF-Q1] | Run diagnostic: if any class < 15% or > 55%, fix immediately |
| HMM retraining frequency | [IRF-Q2] | Weekly refit + online ADWIN adaptation; daily on expiry |
| Walk-forward minimum sample count | [IRF-Q8] | ≥ 3 years 15-min data; ~10K effective observations after purge |
| Ensemble weighting strategy | [IRF-Q6] | Stacking > soft voting; dynamic by recent accuracy if > 3 months data |
| GARCH re-estimation at open | [IRF-Q5] | Refit daily at 9:10 AM IST on overnight gap-adjusted returns |
| Fractional differentiation (d-value) | [IRF-Q11] | d ≈ 0.35–0.45; test ADF stationarity on NSE intraday data |
| Slippage distribution (NSE empirical) | [IRF-Q10] | LIMIT 1-tick inside: 3–8 bps avg; MARKET: 8–20 bps |
| Kalman Q/R calibration | [IRF-Q13] | Adaptive preferred; fixed Q=0.01, R=1.0 as starting point |
| Circuit breaker / NSE halt handler | [IRF-Q15] | Price velocity > 3σ → reduce size; halt → cancel all pending + notify |
| WebSocket disconnect recovery | [IRF-Q4] | State reconcile: QuestDB candle state vs broker position report |
| Session token expiry handler | [IRF-Q12] | Shoonya primary has no TOTP issues; fallback = cancel + flatten |
| SEBI Strategy ID tagging | [IRF-Q9] | Angel One: `strategy_id` field in order payload; Shoonya: `remarks` |

---

## References
- López de Prado: "Advances in Financial Machine Learning" (triple-barrier, CPCV)
- Robert Carver: "Systematic Trading" (position sizing, diversification)
- FinRL: github.com/AI4Finance-Foundation/FinRL
- Microsoft Qlib: github.com/microsoft/qlib
- SEBI Algo Circular: Feb 4, 2025 (SEBI/HO/MIRSD/MIRSD-PoD/P/2025/0000013)

---

## Appendix: Architecture Notes (from Blueprint V2)

### Why Not 5 LLMs — The Reality Check

The original idea of 5 generative AIs voting on 20 algo signals is the wrong primary brain for trading:

- LLMs predict the next token. They do not understand market microstructure, order flow, or probability distributions.
- They hallucinate. A model that outputs `{"trade": "BUY", "confidence": 92}` with no mathematical basis is more dangerous than no AI at all.
- Running 5 local LLMs (7B–13B each) on every candle is too slow. A 1-minute candle gives ~55 seconds of compute time; inference on a consumer GPU for 5 models takes 30–120 seconds.
- LLMs have no feedback loop. They'll make the same mistake forever unless manually re-prompted.
- Local 7B models (Llama, Mistral, Qwen) are significantly dumber than GPT-4/Claude at reasoning.

| Approach | Speed | Learns from P&L | Math-Native |
|---|---|---|---|
| Generative AI (LLM voting) | Very Slow | No | No |
| Reinforcement Learning (RL) | Fast after training | Yes | Yes |
| Ensemble ML (XGBoost/LightGBM) | Very Fast | Yes (retrain) | Yes |
| Classical Quant (stat arb, mean reversion) | Instant | No (rules-based) | Yes |
| Hybrid: RL + Ensemble + LLM for sentiment only | Fast | Partially | Mostly |

**LLMs have one valid role:** daily news sentiment scoring (once per day, not per candle). The sentiment score becomes feature #24 in the ML model. Use FinGPT (`github.com/AI4Finance-Foundation/FinGPT`) if adding this layer.

---

### Full Pipeline Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                    │
│  Indian Broker API (WebSocket) → Live Candles + L2 Order Book        │
│  + News Feed (RSS/API) → Sentiment pipeline                         │
│  Latency target: <200ms tick-to-feature on 1-min candles             │
│  Timeframe: 15-min primary (swing) │ 1-min secondary (scalp)        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     FEATURE ENGINE (23 features)                      │
│  TA-Lib / pandas-ta → technical indicators                           │
│  statsmodels / hmmlearn → statistical features (regime, volatility)  │
│  Custom → order book imbalance, VWAP dev, relative strength vs Nifty │
│  Meta → time-of-day encoding, bid-ask spread (liquidity)             │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     LABELING ENGINE (critical)                        │
│  Triple-barrier method (López de Prado):                             │
│    y=+1 if price hits +1.5 ATR (upper) FIRST within N bars          │
│    y=-1 if price hits -1.5 ATR (lower) FIRST within N bars          │
│    y=0  if neither barrier hit within N bars (timeout)              │
│  Walk-forward split with purge + embargo gaps                        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     DECISION ENGINE                                   │
│  ML Ensemble (XGBoost + LightGBM + CatBoost) → WHAT to trade        │
│  Hyperparameter tuning: Optuna                                       │
│  Experiment tracking: MLflow                                         │
│  RL Agent (PPO via Stable-Baselines3) → HOW MUCH (position size)    │
│  Optional: LLM sentiment (news only, once/day, not per candle)       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     RISK MANAGEMENT (non-negotiable)                  │
│  Max drawdown kill switch │ Volatility-adjusted position sizing      │
│  Daily loss limit │ Correlation check │ Regime filter (HMM)          │
│  Slippage model (0.05% per side) │ Full brokerage + STT cost model  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     EXECUTION LAYER                                   │
│  Order types: LIMIT preferred (reduce slippage), MARKET for urgency  │
│  Ping latency budget: measure broker API round-trip (<100ms target)  │
│  Broker API → Place/Modify/Cancel with retry logic                   │
│  QuestDB → Trade log, model performance, feature snapshots           │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Hybrid Decision Engine Flow (Option D — Recommended)

```
Features (23 signals)
    │
    ├──→ XGBoost → signal + probability     ─┐
    ├──→ LightGBM → signal + probability      ├─→ Weighted Ensemble Vote
    ├──→ CatBoost → signal + probability     ─┘
    │         │
    │    "BUY with 78% confidence"
    │         │
    ├──→ RL Agent (PPO) → position size (% of capital)
    │    (volatility-adjusted: smaller size when ATR is high)
    │         │
    ├──→ Regime Filter (HMM) → allow/block
    │    (block trades in "sideways/crab" regime)
    │         │
    ├──→ Risk Manager → approve/reject/reduce
    │         │
    └──→ EXECUTE via LIMIT order (reduce slippage)
```

The ML ensemble decides WHAT to trade. The RL agent decides HOW MUCH. The regime filter blocks garbage conditions. The risk manager has veto power.

---

### Broker API Comparison — Ranked for Algo Trading (2026)

SEBI retail algo framework became fully mandatory April 1, 2026: every algo order must carry an exchange-assigned Strategy ID, static IP required, mandatory 2FA + daily session logout, max 10 orders/second per exchange.

#### Tier 1 — Best for Self-Built Algo Systems

| Broker | API Cost | Brokerage | WebSocket | Python SDK | Why Pick It |
|---|---|---|---|---|---|
| **Shoonya (Finvasia)** | Free | Zero | Yes | Official | Zero brokerage + free API = lowest cost for high-frequency strategies |
| **Flattrade** | Free | Zero | Yes | Community | Same zero-brokerage model as Shoonya |
| **Angel One SmartAPI** | Free | ₹20/order | Yes | Official | Best documentation, largest retail algo community, easiest onboarding |
| **Dhan (DhanHQ)** | Free | ₹20/order | Yes | Official | Developer-first design, fast infrastructure |
| **Fyers** | Free | ₹20/order | Yes | Official | Flexible integration, strong SDK |

#### Tier 2 — Established but Paid/Limited

| Broker | API Cost | Brokerage | Notes |
|---|---|---|---|
| **Zerodha Kite Connect** | ~₹2,000/month | ₹20/order | Most mature ecosystem, huge community, but API cost adds up |
| **Upstox** | Free (basic) | ₹20/order | Good manual-to-algo bridge |
| **Kotak Neo** | Free | Zero (Trade Free plan) | Sub-50ms claimed latency |

**Recommendation:** Use Shoonya/Finvasia as primary (zero brokerage + free API = lowest cost for all strategies). Keep Angel One SmartAPI as dev reference (best docs, easiest onboarding for learning the API pattern). Keep Dhan as backup.

#### SEBI Compliance Checklist

- [ ] Register a static public IPv4 with your broker. Whether it must be an Indian IP is broker policy — confirm before buying VPS. SEBI circulars require static IP registration, not explicitly Indian hosting.
- [ ] Enable 2FA on all API access
- [ ] Ensure daily automatic session logout before pre-open (9:00 AM IST re-login cron)
- [ ] Stay under 10 OPS per exchange — above requires exchange registration
- [ ] Maintain audit trail logs for 5 years (QuestDB trade log satisfies this if timestamps are correct IST)
- [ ] Register your algorithm with broker and get a Strategy ID before going live
- [ ] Get broker approval for API connectivity before going live — submit SOP, pass UAT

#### Deployment Security Checklist

**Secrets:**
- [ ] `chmod 600 .env` — readable by owner only
- [ ] Confirm: `git check-ignore -v .env` returns `.env` before first commit
- [ ] Never log credential values — log `"SHOONYA_KEY present: True"`, never the key itself
- [ ] Rotate all API keys immediately if VPS is ever accessed by unknown party

**VPS hardening (when moving off local machine):**
- [ ] Disable password SSH auth: `sshd_config → PasswordAuthentication no` — key-only login
- [ ] Firewall (UFW): `ufw default deny incoming`, then allow port 22 from your IP only
- [ ] Run trading processes as non-root: `adduser trader` — never `sudo python main.py`
- [ ] Outbound allowlist: broker API endpoints + ZMQ ports (5555–5560) only

**Process resilience:**
- [ ] Each ZMQ process as a separate systemd service with `Restart=on-failure`, `RestartSec=5`
- [ ] `alert_sink.py` in its own service — Telegram failure must not take down trading
- [ ] Mid-session broker disconnect → reconnect → **always run `StartupReconciler.reconcile()` before resuming trading** (not only on cold start)

---

### Library Reference — Why These Tools

**pandas-ta vs TA-Lib:** Use pandas-ta if you want zero install headaches (pure pandas, no C dependency). Use TA-Lib if you need maximum speed (C-based backend).

**Why Optuna matters:** XGBoost has ~15 hyperparameters. Manual tuning is guesswork. Optuna uses Tree-Structured Parzen Estimators to find optimal settings in 100–500 trials. It can 2–5x out-of-sample performance versus default params.

**Why MLflow matters:** After 50 experiments, you will not remember which settings produced which results. MLflow logs everything: parameters, metrics, trained model files, feature importance plots. It's a lab notebook.

#### Statistical / Quant Libraries

| Library | What It Does | Relevant Models |
|---|---|---|
| **statsmodels** | Statistical modeling | Cointegration (Engle-Granger, Johansen), ARIMA/GARCH, regime detection |
| **hmmlearn** | Hidden Markov Models | Detect bull/bear/sideways regimes from price data |
| **filterpy** | Kalman Filters | Noise smoothing, dynamic hedging ratios, pairs trading |
| **arch** | ARCH/GARCH models | Volatility forecasting, risk modeling |

#### Deep Learning Libraries (for later phases)

| Library | What It Does |
|---|---|
| **PyTorch + PyTorch Forecasting** | LSTM, Transformer-based (TFT, N-BEATS, N-HiTS) |
| **Darts** (unit8co) | Unified API for 20+ time series models |
| **tsai** | State-of-the-art deep learning for time series (fastai-based) |

---

### Why Features 21–23 Are Critical

- **Relative Strength vs Nifty (#21):** A stock up 2% means nothing if Nifty is up 3%. Without it, the model just learns "buy when market goes up" — that's beta, not edge. Calculated as: stock return − Nifty return.
- **Time-of-Day (#22):** Indian market has distinct behavioral regimes. 9:15–9:45 = volatile (opening auction spillover). 11:00–13:00 = dead zone (low volume). 14:30–15:20 = institutional flow. Encode as sin/cos of minutes since 9:15.
- **Bid-Ask Spread (#23):** Wide spread = illiquid = higher slippage = backtest lies. A stock showing a perfect signal with a 0.5% spread will eat the entire expected profit on entry+exit.

---

### Triple-Barrier Labeling — Detailed Specification

**Bad labels (do NOT use):**
- `y = 1 if next_candle_close > current_close` — noise; +0.01% and +5% get same label
- `y = return_next_5_bars` — regression on raw returns, R² ≈ 0.01
- `y = 1 if next_bar_high > threshold` — lookahead bias

**Triple-barrier pseudocode (CORRECTED — symmetric ±1.5 ATR):**
```
For each candle at time t:
  Upper barrier = entry_price + 1.5 × ATR(14)   ← symmetric ±1.5 default
  Lower barrier = entry_price - 1.5 × ATR(14)   ← asymmetric 2.0/1.0 causes class imbalance
  Time barrier  = N bars (e.g., 20 bars for 15-min = 5 hours)

  y = 1  if price hits upper barrier FIRST (profitable trade)
  y = -1 if price hits lower barrier FIRST (stopped out)
  y = 0  if neither hit within N bars (timeout)
```

ALWAYS run `compute_label_distribution()` after labeling. If any class < 15% of total: either tune multipliers or add `compute_sample_weights()`. See [CDR-3A-C12] and [IRF-Q1] for full NSE calibration guide.

**Walk-forward diagram:**
```
[=== TRAIN (2022-01 to 2023-12) ===][PURGE 5 days][EMBARGO 2 days][== TEST (2024-01 to 2024-06) ==]
                                     ↑ delete      ↑ skip
                                     overlapping   autocorrelation
                                     labels        spillover

Then roll forward:
[=== TRAIN (2022-07 to 2024-06) ===][PURGE][EMBARGO][== TEST (2024-07 to 2024-12) ==]
```

`PyBroker` has walk-forward ML backtesting built in. `scikit-learn`'s `TimeSeriesSplit` is a starting point but requires manual purge/embargo.

---

### India Cost Model — Full Breakdown

Every backtest must subtract these from every trade or profits are fictional:

| Cost Component | Equity Intraday | Equity Delivery | F&O |
|---|---|---|---|
| Brokerage | ₹20 flat or ₹0 | ₹20 flat or ₹0 | ₹20 flat or ₹0 |
| STT | 0.025% sell side | 0.1% both sides | varies |
| Exchange charges | ~0.00345% | ~0.00345% | ~0.05% |
| GST (on brokerage + exchange) | 18% | 18% | 18% |
| SEBI charges | 0.0001% | 0.0001% | 0.0001% |
| Stamp duty | ~0.003% buy side | ~0.015% buy side | varies by state |
| **Slippage** | **0.03–0.10%** | **0.03–0.05%** | **0.05–0.15%** |

**Slippage is the silent killer.** By the time a MARKET order reaches the exchange, the price has moved. Budget 0.05% per side minimum; for illiquid stocks, 0.10–0.20%.

**LIMIT vs MARKET trade-off:** A strategy with 70% fill rate on LIMIT orders but near-zero slippage often beats 100% fill on MARKET orders with 0.1% slippage.

### Volatility-Adjusted Position Sizing — Worked Example

```
position_size = (risk_per_trade × capital) / (ATR × ATR_multiplier)

Example:
  capital        = ₹5,00,000
  risk_per_trade = 1% = ₹5,000
  ATR(14)        = ₹15
  ATR_multiplier = 2 (stop-loss distance in ATR units)

  position_size  = 5000 / (15 × 2) = 166 shares

If ATR doubles to ₹30 (high volatility day):
  position_size  = 5000 / (30 × 2) = 83 shares  ← auto-reduces
```

This ensures the same rupee risk regardless of market conditions.

---

### Five Ways You Die (Top Failure Modes)

#### Death #1: Bad Labels
Training on `y = 1 if next_candle_close > current_close` teaches the model noise. Fix: triple-barrier method.

#### Death #2: Data Leakage
Random train/test split on time series lets the model see future patterns. Backtest shows 80% accuracy; live shows 50%. Fix: walk-forward with purge + embargo (non-negotiable).

#### Death #3: Execution Fantasy
Backtest assumes buying at exact close price with zero delay. Reality: network latency (50–200ms), order queue, bid-ask spread, market impact. Fix: 0.05% slippage model per side, LIMIT orders, measure broker API round-trip. If edge is <0.1% per trade, it's probably not real after costs.

#### Death #4: India Cost Structure
STT + exchange charges + GST + stamp duty + SEBI charges = ~0.05–0.08% per round trip on top of brokerage. High-frequency strategies (many small trades) get eaten alive. Fix: calculate average profit per trade AFTER all costs. If expected edge <0.15% per trade, use fewer, higher-conviction trades. Swing trading (15-min to daily) gives retail a better chance.

#### Death #5: Regime Change
ML trained on 2024 bull market data drops from 60% to 45% accuracy in 2025 consolidation and bleeds money. Fix:
1. HMM regime filter (feature #14): if regime = "sideways/crab," reduce sizes or pause trading.
2. Retrain models monthly or quarterly on a rolling window.
3. Use diverse features (relative strength, regime state, volatility forecasts) — pure price-pattern features decay fastest.

---

### Trade Logging — SQL Schema

```sql
-- Core trade log
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    symbol TEXT,
    direction TEXT,           -- BUY / SELL
    order_type TEXT,          -- LIMIT / MARKET
    quantity INTEGER,
    entry_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    slippage_entry REAL,      -- actual vs intended entry price
    slippage_exit REAL,
    pnl_gross REAL,
    brokerage REAL,
    stt REAL,
    other_charges REAL,
    pnl_net REAL,
    model_signal TEXT,        -- JSON of all model outputs
    features_snapshot TEXT,   -- JSON of 23 features at entry
    regime_at_entry TEXT,     -- bull / bear / sideways
    holding_period_sec INTEGER,
    status TEXT               -- OPEN / CLOSED / STOPPED_OUT / TIMED_OUT
);

-- Model accuracy tracking
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY,
    date TEXT,
    model_name TEXT,          -- xgboost / lightgbm / catboost / rl_agent
    model_version TEXT,       -- link to MLflow run ID
    total_signals INTEGER,
    correct_signals INTEGER,
    accuracy REAL,
    avg_confidence REAL,
    sharpe_ratio REAL,
    profit_factor REAL,
    max_drawdown REAL
);

-- Feature importance drift tracking
CREATE TABLE feature_drift (
    id INTEGER PRIMARY KEY,
    date TEXT,
    model_name TEXT,
    feature_name TEXT,
    importance_score REAL,    -- track if feature importance shifts over time
    correlation_with_target REAL
);
```

**Key metrics to track:** Win rate (meaningless without R:R), Profit Factor (target >1.5), Sharpe Ratio (target >1.5 annualized), Max Drawdown (target <15%), average holding period, per-model accuracy for ensemble reweighting, slippage tracking (intended vs actual fill).

---

### Timeframe Comparison — Why Start at 15-min

| Factor | 1-min Intraday | 15-min Swing | Daily Positional |
|---|---|---|---|
| Signal-to-noise | Very low | Moderate | High |
| Cost impact | Huge (many trades) | Moderate | Small |
| Latency sensitivity | High (needs <100ms) | Low | None |
| Slippage impact | High | Moderate | Low |
| Data needed for training | Millions of bars | Hundreds of K bars | Thousands of bars |
| Retail edge probability | Low | Moderate | Highest |

**Start with 15-min candles on Nifty 50 stocks.** It gives 26 candles/day × 250 days = 6,500 bars/year per stock, enough time to process, and enough profit-per-trade to survive India's cost structure. Once profitable on 15-min for 6+ months, then consider adding 5-min or 1-min.

---

### Platform Learning Path

```
Learning:       Backtrader → VectorBT (fast iteration)
Building:       PyBroker + pandas-ta + XGBoost + Optuna + MLflow
Scaling:        FinRL or Microsoft Qlib (ML/RL native)
Production:     NautilusTrader (when serious and profitable)
```

| Platform | Stars | What Makes It Special |
|---|---|---|
| **Microsoft Qlib** | 16K+ | Full ML pipeline: data → alpha → risk → portfolio → execution. SOTA research papers built-in |
| **NautilusTrader** | 17K+ | Rust core, nanosecond precision. Same code for backtest and live. End-goal platform |
| **QuantConnect Lean** | 10K+ | Cloud + local backtesting, multi-asset, broker integrations |
| **FinRL** | 10K+ | Best for RL-based trading. NeurIPS paper. Active development |
| **pysystemtrade** | 2.5K+ | Robert Carver's own implementation. Real-world proven. Best for futures/systematic |
| **VectorBT** | — | Blazing fast vectorized backtesting. Test thousands of parameter combinations in seconds |
| **PyBroker** | — | Walk-forward ML backtesting built in. Write strategies using scikit-learn/XGBoost directly |

---

### Indian Market Data Sources

| Source | Data Type | Cost | Notes |
|---|---|---|---|
| **Broker WebSocket** | Live ticks, candles, order book | Free with account | Primary source |
| **Jugaad Data** | NSE/BSE historical bhavcopy | Free | `pip install jugaad-data` |
| **NSE India** website | EOD data, F&O chain | Free | `nsetools` or `nsepython` |
| **AkShare** | Multi-market including India | Free | `pip install akshare` |
| **Yahoo Finance** (yfinance) | Historical daily/weekly | Free | Good for backtesting, not live |
| **TrueData / Global Datafeeds** | Professional tick data India | ₹500–2000/month | For serious backtesting with clean data |

---

### Phased Build Timeline

#### Phase 1: Foundation (Weeks 1–3)
- Set up Shoonya/Finvasia account + static IP + SEBI compliance (Angel One: dev reference only)
- Build WebSocket pipeline → store 15-min candles in QuestDB
- Compute 23 features on live data
- Paper trade manually using signals in terminal
- Measure broker API latency (ping test every 15 min, log results)

#### Phase 2: Labeling + ML Engine (Weeks 4–7)
- Download 2+ years historical data
- Implement triple-barrier labeling + walk-forward with purge/embargo
- Train XGBoost + LightGBM + CatBoost ensemble
- Tune with Optuna (300 trials per model), track with MLflow
- Backtest with VectorBT or PyBroker (walk-forward)
- **Gate:** Walk-forward Sharpe < 1.0 → fix features/labels before proceeding

#### Phase 3: Risk + Execution (Weeks 8–10)
- Build volatility-adjusted position sizing
- Build realistic cost model (brokerage + STT + slippage)
- Build risk management kill switches
- Implement LIMIT order logic with fill-rate tracking
- Run paper trading in full automated mode for 4+ weeks

#### Phase 4: Enhancements (Weeks 11+)
- Add HMM regime filter (block trades in crab market)
- Add RL agent (FinRL/SB3) for position sizing
- Add LLM sentiment layer (single model, news only, once/day)
- Build performance dashboard
- Go live on Shoonya (already primary — zero-brokerage live trading from Phase 1)
- Weekly model retraining + monthly walk-forward re-validation

---

### Extended Reading List

#### Papers

| Paper | Why Read It |
|---|---|
| XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, KDD 2016) | Understand the primary model's internals |
| LightGBM: A Highly Efficient GBDT (Ke et al., NeurIPS 2017) | Speed/accuracy tradeoffs |
| FinRL: Deep RL Library for Automated Stock Trading (Liu et al., NeurIPS 2020 Workshop) | The RL framework in use |
| Stable-Baselines3: Reliable RL Implementations (Raffin et al., JMLR 2021) | The RL algorithms in use |
| Trading-R1: Financial Trading with LLM Reasoning via RL (2025) | Cutting-edge RL+LLM hybrid |

#### Books

| Book | Author | Why Read It |
|---|---|---|
| Advances in Financial Machine Learning | Marcos López de Prado | THE bible. Triple-barrier, purge/embargo, feature importance. Read before coding. |
| Systematic Trading | Robert Carver | Rule-based systems, position sizing, realistic cost modeling |
| Machine Learning for Algorithmic Trading | Stefan Jansen | 700+ pages, full Python code, ML + alternative data + backtesting |
| Quantitative Trading | Ernest Chan | Getting started with quant systems at retail scale |

#### SEBI Regulations

| Document | Date | What It Covers |
|---|---|---|
| SEBI Circular: Safer Participation of Retail Investors in Algo Trading | Feb 4, 2025 | Foundational algo regulation |
| NSE Operational Modalities for Algo Framework | Jul 22, 2025 | Static IP, OPS limits, Strategy ID implementation |
| SEBI Extension Circular | Sep 30, 2025 | Glide path to April 2026 mandatory compliance |

---

### GitHub Repository Bookmarks

| Repository | URL | Purpose |
|---|---|---|
| awesome-quant | `github.com/wilsonfreitas/awesome-quant` | Master list of every quant library |
| best-of-algorithmic-trading | `github.com/merovinh/best-of-algorithmic-trading` | Ranked list of 110+ algo trading projects |
| FinRL | `github.com/AI4Finance-Foundation/FinRL` | Financial reinforcement learning |
| FinRL-Trading | `github.com/AI4Finance-Foundation/FinRL-Trading` | Production-grade RL trading |
| FinGPT | `github.com/AI4Finance-Foundation/FinGPT` | Financial LLM for sentiment |
| Microsoft Qlib | `github.com/microsoft/qlib` | AI-oriented quant platform |
| NautilusTrader | `github.com/nautechsystems/nautilus_trader` | Rust+Python production engine |
| pysystemtrade | `github.com/robcarver17/pysystemtrade` | Robert Carver's systematic trading |
| Stable-Baselines3 | `github.com/DLR-RM/stable-baselines3` | RL algorithms (PPO, SAC, etc.) |
| VectorBT | `vectorbt.dev` | Fast vectorized backtesting |
| PyBroker | `github.com/edtechre/pybroker` | Walk-forward ML backtesting |
| Optuna | `github.com/optuna/optuna` | Bayesian hyperparameter optimization |
| MLflow | `github.com/mlflow/mlflow` | Experiment tracking |
| SmartAPI Python | `github.com/angel-one/smartapi-python` | Angel One API client |
| Shoonya API | `github.com/Shoonya-Dev/ShoonyaApi-py` | Finvasia/Shoonya API client |
| DhanHQ Python | `github.com/dhan-oss/DhanHQ-py` | Dhan API client |

---

### Honest Scoring — Retail vs Institutional

| Dimension | Solo Retail | World Best (Citadel, Two Sigma, Jump) |
|---|---|---|
| Data quality | Free broker data, ~100ms delay | Colocated servers, nanosecond data, alternative data |
| Compute | 1 GPU, 32GB RAM | Thousands of GPUs |
| Team | 1 person | 100+ PhDs |
| Engineering | Python scripts | Custom C++/FPGA hardware |
| Alpha (edge) | 2–6% annual if good | 0.5–2.5% on billions |

**Realistic target:** After 6 months of live profitable trading → 6.5/10 (elite for solo retail in India). At retail scale (₹5–50 lakh capital), 15–25% annual return with controlled drawdown is excellent. Institutions can't deploy at this small scale — their minimum trade sizes are larger than the entire portfolio. Small size is an advantage in illiquid corners of the market.
