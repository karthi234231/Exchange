# Complete Setup Guide: AI Algo Trading System (India/NSE)

> **PLANNED BUILD GUIDE — Code does not exist yet**
>
> The commands in the Quick Reference below (`python main.py`, `pytest tests/`, etc.) will fail because the corresponding Python files have not been created yet.
> This guide describes the intended build sequence, not the current state.
>
> - **Current repo state:** Architecture docs only — no `.py`, `requirements.txt`, `tests/`, or `Dockerfile` exist.
> - **First milestone:** Build a Stage 1 paper-trading scaffold with mock broker that runs without real credentials.
> - **Do not run `--mode live`** until all gates in `MD_Files/DECISIONS.md` are satisfied.
>
> See `README.md` for current project status.

> **⚠️ BROKER CORRECTION (May 2026):** This guide was originally written with Angel One SmartAPI as the primary broker. The architecture decision has been updated: **Shoonya (Finvasia) is now the PRIMARY execution broker** (zero brokerage, no TOTP expiry). Angel One = dev reference only for documentation.
>
> When following the prompt playbook:
> - Prompt 4: Build `broker_shoonya.py` FIRST (PRIMARY); Angel One = dev reference only
> - Prompt 6 (WebSocket): Use Shoonya's WebSocket + ZeroMQ PUB:5555, not Angel One SmartStream
> - Prompt 11 (Triple-barrier): Defaults are **now ±1.5 ATR symmetric** (not 2.0/1.0 asymmetric)
> - Prompt 16 (Cost model): Default broker = `'shoonya'` (zero brokerage), not `'angelone'`
> - Prompt 24 (History download): Use Shoonya API + QuestDB, not Angel One + SQLite
>
> See `CLAUDE.md` for authoritative architecture decisions.

---

## Introduction

This guide covers the full setup journey for building a production-grade AI-powered algorithmic trading system targeting Indian equity markets (NSE). It consolidates two complementary approaches into one reference:

1. **Environment Setup** — How to install Claude Code, set up the project from scratch, and use a prompt-by-prompt playbook to build every production module (25 prompts covering data pipelines, feature engineering, ML ensemble, risk management, execution, and backtesting).

2. **Jupyter / Kaggle Notebook Setup** — A research-first workflow where you explore, visualize, and validate every component in Jupyter notebooks before converting to production scripts. Recommended if you want to understand what you are building before deploying capital.

The two approaches are complementary: use the Jupyter path for Phase 1 research and validation, then use the Claude Code prompt playbook for Phase 2 production conversion.

---

## Quick Reference

Most common commands from both guides:

```bash
# --- Environment / Claude Code ---
claude --version                                      # Check Claude Code installation
claude                                                # Launch Claude Code in project dir
source venv/bin/activate                              # Activate Python virtual environment (Linux/Mac)
# venv\Scripts\activate                              # Windows equivalent

# --- Jupyter ---
jupyter lab                                           # Launch JupyterLab

# --- Python dependencies ---
pip install -r requirements.txt                       # Install all dependencies

# --- QuestDB (time-series database) ---
docker run -d -p 9000:9000 -p 9009:9009 --name questdb questdb/questdb

# --- DVC (data + model versioning) ---
dvc init && dvc add data/

# --- Trading system ---
python main.py --mode paper                           # Paper trading
python main.py --mode live                            # Live trading
python scripts/train_models.py --symbol RELIANCE --timeframe 15min

# --- Testing & monitoring ---
pytest tests/ -v
mlflow ui --port 5000                                 # Open http://localhost:5000
python models/explainer.py --model models/saved/xgb_latest.pkl  # SHAP / SEBI report

# --- Historical data download ---
python scripts/download_history.py --symbols RELIANCE TCS INFY --years 2
```

---

## Section 1: Environment Setup

*Content from SETUP_GUIDE.md — Building the Algo Trading System with Claude Code*

---

### Step 1: Install Claude Code

You need two things: Node.js 18+ and a Claude subscription.

#### Install Node.js (if you don't have it)

**Windows (use WSL2 — strongly recommended):**
```bash
# Open PowerShell as Admin, install WSL2
wsl --install

# Restart PC, then open Ubuntu from Start Menu
# Inside WSL:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
node --version  # Should show v22.x.x
```

**Linux (Ubuntu/Debian):**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
```

**macOS:**
```bash
brew install node
```

#### Install Claude Code

**Option A: Native installer (recommended — auto-updates, no Node.js needed at runtime)**
```bash
# Linux/macOS
curl -fsSL https://downloads.claude.ai/claude-code/install.sh | sh

# Verify
claude --version
```

**Option B: npm**
```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

#### Authenticate
```bash
claude
# First launch opens your browser → log in with your Anthropic account
# Session token is stored locally in ~/.claude/
```

#### Cost
- Claude Pro: $20/month (approx. Rs. 1,700) — works for this project
- Claude Max: $100/month — higher limits for heavy sessions
- API pay-as-you-go: ~$0.50-3.00 per coding session
- The tool itself is free to install

---

### Step 2: Set Up the Project

```bash
# Create project directory
mkdir algo_trading
cd algo_trading

# Initialize git (Claude Code works best with git)
git init

# Copy the CLAUDE.md file into this directory
# (Download it from the file I gave you, or copy-paste it)

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Create .env file for your API keys
cp .env.example .env
# Edit .env with your actual keys
```

---

### Step 3: Launch Claude Code and Start Building

```bash
cd algo_trading
claude
```

Claude Code will automatically read your CLAUDE.md file and understand the entire project structure, conventions, and build order.

---

### Step 4: Prompt Playbook (Feed These to Claude Code in Order)

Below are the exact prompts to give Claude Code, one at a time.
Wait for each module to be complete and tested before moving to the next.

---

#### Module 0: Project Setup

**Prompt 1:**
```
Create the project scaffolding: config.py with all dataclasses (BrokerConfig,
TradingConfig, FeatureConfig, LabelConfig, ModelConfig, RiskConfig, ExecutionConfig),
.env.example with all environment variable placeholders, and requirements.txt with
all dependencies. Follow the CLAUDE.md conventions. Use python-dotenv to load .env.
```

**Prompt 2:**
```
Create the directory structure from CLAUDE.md with all __init__.py files.
Create setup.py for the project. Create a basic README.md.
```

---

#### Module 1: Data Pipeline (Broker Connection + Storage)

**Prompt 3:**
```
Build data/broker_base.py: an abstract base class BrokerBase with these methods:
login(), place_order(symbol, side, qty, order_type, price), cancel_order(order_id),
get_positions(), get_order_book(), get_ltp(symbol), get_historical_data(symbol,
interval, from_date, to_date), subscribe_ws(symbols, on_tick_callback). All methods
must have type hints and docstrings. Use abc.ABC.
```

**Prompt 4:**
```
Build data/broker_shoonya.py: concrete implementation of BrokerBase for Shoonya/Finvasia
(PRIMARY execution broker — zero brokerage, no TOTP expiry issues). Use the ShoonyaApi-py
library (pip install ShoonyaApi-py). Implement all BrokerBase methods: login() using
SHOONYA_USER_ID/PASSWORD/VENDOR_CODE/IMEI from .env, place_order(), cancel_order(),
get_positions(), get_order_book(), get_ltp(), get_historical_data(), subscribe_ws().
Handle API errors with retries (3 attempts, exponential backoff). Log with logging module.
Include a SessionManager that auto-refreshes the session token before market open (see
[CDR-3B-C24] for full implementation). Load all credentials from config/env.

(Optional — dev reference only) Also build data/broker_angelone.py using smartapi-python
for Angel One SmartAPI. This is for documentation/learning purposes only — NOT for live
execution. Note: Angel One has TOTP expiry issues [CDR-3A-C1] that make it unsuitable
for unattended live trading.
```

**Prompt 5:**
```
Build data/candle_store.py: QuestDB storage for OHLCV candles (NOT SQLite — QuestDB is canonical per arch-spec.json). Methods:
store_candle(symbol, timeframe, timestamp, open, high, low, close, volume),
get_candles(symbol, timeframe, from_dt, to_dt) → pl.DataFrame,
get_latest_candles(symbol, timeframe, n) → pl.DataFrame.
Use InfluxDB Line Protocol (port 9009) for writes. Use PostgreSQL wire protocol (port 8812) for queries.
Ensure (symbol, timeframe, timestamp) uniqueness via QuestDB dedup. Use Polars, not pandas.
Write a test in tests/test_candle_store.py.
```

**Prompt 6:**
```
Build data/websocket_feed.py: ZeroMQ DataIngestor process that connects to Shoonya's
WebSocket API (subscribe_ws() from broker_shoonya.py), subscribes to symbols from
config watchlist, aggregates ticks into 1-min candles, resamples to 5-min and 15-min
candles, stores them via candle_store.py (QuestDB), and publishes tick data via ZeroMQ
PUB socket on port 5555 for downstream FeatureComputer process.
Handle disconnections with auto-reconnect (exponential backoff). Log all events.
See [P2I-2.1] for full ZeroMQ DataIngestor implementation with msgpack serialization.
Must work as a standalone process launchable from main.py.
```

---

#### Module 2: Feature Engine (23 Features)

**Prompt 7:**
```
Build features/technical.py: a function compute_technical_features(df: pd.DataFrame)
→ pd.DataFrame that takes OHLCV candles and adds these columns using pandas-ta:
rsi_14, macd_signal, bb_percent_b, bb_width, atr_14, adx_14, stoch_k, stoch_d,
obv, supertrend, ema_cross (1 if ema9 > ema21 else 0), ichimoku_cloud
(1 if price above cloud, -1 if below, 0 if inside), heikin_ashi_direction,
pivot_distance. Handle NaN rows at the start (from lookback periods) by marking
them, not dropping. Write tests.
```

**Prompt 8:**
```
Build features/statistical.py with these functions:
1. compute_hmm_regime(returns: pd.Series, n_states=3) → pd.Series of regime labels
   using hmmlearn GaussianHMM. Fit on training data, predict on new data.
2. compute_kalman_smooth(prices: pd.Series) → pd.Series using filterpy KalmanFilter.
3. compute_garch_forecast(returns: pd.Series, p=1, q=1) → pd.Series of next-bar
   volatility forecasts using arch library.
4. compute_autocorrelation(returns: pd.Series, lags=[1,2,3,4,5]) → pd.DataFrame
   with autocorrelation at each lag.
Handle edge cases (insufficient data, convergence failures). Write tests.
```

**Prompt 9:**
```
Build features/custom.py with these functions:
1. compute_vwap_deviation(df) → Series: (close - vwap) / vwap
2. compute_order_book_imbalance(bid_volume, ask_volume) → Series: (bid-ask)/(bid+ask)
3. compute_price_volume_divergence(close, volume, window=14) → Series:
   correlation between price change and volume change over rolling window
4. compute_relative_strength_vs_index(stock_returns, index_returns, window=20) → Series
5. compute_time_of_day(timestamps) → DataFrame with sin_tod and cos_tod columns
   (minutes since 9:15 AM, encoded as sin/cos for cyclical nature)
6. compute_bid_ask_spread(bid_price, ask_price) → Series: (ask-bid)/mid
Write tests for each function.
```

**Prompt 10:**
```
Build features/engine.py: the master feature pipeline. Function:
compute_all_features(ohlcv_df, index_df=None, order_book_df=None) → pd.DataFrame
that calls technical.py, statistical.py, and custom.py, assembles all 23 features
into one DataFrame aligned by timestamp. Drop rows where any feature is NaN
(warmup period). Return clean DataFrame ready for ML. Log feature stats (mean, std,
% missing) for monitoring. Write integration test.
```

---

#### Module 3: Labeling Engine

**Prompt 11:**
```
Build labeling/triple_barrier.py: function apply_triple_barrier(
    df: pd.DataFrame,  # must have 'close' and 'atr' columns
    upper_atr_mult: float = 1.5,   # SYMMETRIC default — do NOT use 2.0/1.0 (causes class imbalance)
    lower_atr_mult: float = 1.5,   # see [CDR-3A-C12]: asymmetric causes long-side bias
    max_holding_bars: int = 20,
) → pd.Series of labels (+1, -1, 0).

For each bar i:
  upper = close[i] + upper_atr_mult * atr[i]   # +1.5 ATR symmetric
  lower = close[i] - lower_atr_mult * atr[i]   # -1.5 ATR symmetric
  Look forward up to max_holding_bars:
    If high hits upper first → +1
    If low hits lower first → -1
    If neither within max_holding_bars → 0

IMPORTANT: After labeling, always call compute_label_distribution() and verify
every class is 15–55% of total. If any class < 15%, the multipliers are wrong — adjust
before training. See [CDR-3A-C12] for NSE calibration guide.

Also return a DataFrame with columns: label, barrier_hit_bar, actual_return, sample_weight.
This is the MOST IMPORTANT function in the project. Test it thoroughly with known
price sequences where you can manually verify the correct label.
```

**Prompt 12:**
```
Build labeling/walk_forward.py: function walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    purge_bars: int = 20,  # must be >= max_holding_bars
    embargo_bars: int = 5,
) → generator of (train_idx, test_idx) tuples.

This is a time-series walk-forward split that:
1. Divides data chronologically into n_splits expanding or rolling windows
2. For each fold, training data is ALWAYS before test data
3. Purge: remove training samples whose label window (max_holding_bars) overlaps
   with the first bar of the test set
4. Embargo: after purge, skip embargo_bars additional samples
5. NEVER shuffles data

Write extensive tests: verify no future leakage, verify purge removes correct rows,
verify embargo gap exists, verify train is always before test.
```

---

#### Module 4: ML Ensemble

**Prompt 13:**
```
Build models/ensemble.py:
- Class EnsembleModel with methods:
  train(X_train, y_train): trains XGBoost, LightGBM, CatBoost
  predict(X) → dict with keys 'signal' (+1/-1/0) and 'confidence' (0-1)
  Each sub-model's prediction is weighted. Default weights: equal (1/3 each).
  Weights are adjustable based on recent performance.
  predict_proba(X) → dict with per-model probabilities
  save(path), load(path): persist/load trained models

- Ensemble signal logic:
  Get probability of +1 from each model.
  Weighted average → final probability.
  If avg_prob > 0.6 → signal = BUY
  If avg_prob < 0.4 → signal = SELL
  Else → HOLD (no trade)
  Confidence = abs(avg_prob - 0.5) * 2  (0 to 1 scale)

Write tests with synthetic data.
```

**Prompt 14:**
```
Build models/tuner.py: function tune_model(
    model_type: str,  # 'xgboost' | 'lightgbm' | 'catboost'
    X_train, y_train,
    X_val, y_val,
    n_trials: int = 300,
) → dict of best hyperparameters.

Use Optuna with TPE sampler. Define sensible search spaces for each model type:
- max_depth: 3-10
- learning_rate: 0.01-0.3 (log)
- n_estimators: 100-1000
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- min_child_weight: 1-10
- reg_alpha/reg_lambda: 1e-8 to 10 (log)

Objective: maximize walk-forward F1-score (not accuracy, because classes are imbalanced).
Use Optuna pruning (MedianPruner) to stop bad trials early.
Log each trial to MLflow via tracker.py.
```

**Prompt 15:**
```
Build models/tracker.py: a wrapper around MLflow for experiment tracking.
- start_run(name, params) → context manager
- log_metrics(metrics_dict)
- log_model(model, name)
- log_artifact(filepath)
- get_best_run(metric_name) → run info

Configure MLflow to store in local ./mlruns directory by default.
Each training run should log: all hyperparameters, walk-forward metrics per fold,
average metrics, feature importance plot (saved as artifact), trained model.
```

---

#### Module 5: Risk Management

**Prompt 16:**
```
Build risk/cost_model.py: function calculate_trade_cost(
    price: float, quantity: int, side: str, segment: str = 'equity_intraday',
    broker: str = 'shoonya'  # default = Shoonya (zero brokerage); use 'angelone' for dev reference
) → dict with breakdown: brokerage, stt, exchange_charges, gst, sebi_charges,
stamp_duty, total_cost, cost_per_share, cost_percentage.

Model real Indian costs:
- Brokerage: Rs. 0 (Shoonya — PRIMARY) or Rs. 20/order (Angel One — dev ref only)
- STT: 0.025% sell-side (intraday), 0.1% both sides (delivery)
- Exchange: ~0.00345%
- GST: 18% on (brokerage + exchange charges)
- SEBI: 0.0001%
- Stamp: ~0.003% buy-side (varies by state, use Maharashtra default)
- Slippage model: configurable, default 0.05% per side

Write tests against manually calculated expected values.
```

**Prompt 17:**
```
Build risk/position_sizer.py: function calculate_position_size(
    capital: float,
    risk_per_trade: float,  # as fraction, e.g., 0.01 = 1%
    atr: float,
    atr_multiplier: float = 2.0,
    price: float = None,
    max_position_pct: float = 0.05,  # max 5% of capital in one trade
) → dict with quantity, risk_amount, stop_distance, position_value.

Volatility-adjusted: position_size = (capital * risk_per_trade) / (atr * atr_multiplier)
Cap at max_position_pct of capital.
Round down to nearest lot size (default 1 for equity).
Write tests.
```

**Prompt 18:**
```
Build risk/manager.py: class RiskManager that enforces ALL these rules:
1. Daily P&L tracking. If daily_loss > max_daily_loss_pct → kill_switch = True,
   reject all new orders, send Telegram alert.
2. Max drawdown from peak equity. If drawdown > max_drawdown_pct → pause system.
3. Max open positions limit.
4. Max single position size (from position_sizer).
5. No new trades after config.no_new_trades_after time.
6. Force close all at config.force_close_time.
7. Method: can_trade() → bool (checks all rules).
8. Method: pre_trade_check(symbol, side, qty, price) → (bool, reason_string).
9. State persisted in QuestDB (risk_state table) so it survives restarts. No SQLite.
Write comprehensive tests for every kill condition.
```

---

#### Module 6: Execution + Logging

**Prompt 19:**
```
Build execution/order_manager.py: class OrderManager that:
1. Places LIMIT orders with configurable offset from LTP (e.g., 0.05% better).
2. If LIMIT not filled within wait_bars (default 2), converts to MARKET.
3. Rate limiting: max 8 orders/second via token bucket algorithm.
4. Retry logic: 3 attempts with exponential backoff on API errors.
5. Always attaches stop-loss order immediately after entry fill.
6. Methods: enter_trade(symbol, side, qty, price, sl, tp) → order_id,
   exit_trade(order_id) → fill_info, cancel_pending().
7. Tracks intended vs actual fill price for slippage measurement.
8. Paper trade mode: simulates fills at LTP + slippage model (no actual API calls).
```

**Prompt 20:**
```
Build execution/trade_logger.py: QuestDB trade logger (NOT SQLite — use QuestDB orders/fills/pnl_daily tables per arch-spec.json) with these tables:
1. trades: id, timestamp, symbol, direction, order_type, quantity, entry_price,
   intended_entry, exit_price, stop_loss, take_profit, slippage_entry, slippage_exit,
   pnl_gross, brokerage, stt, other_charges, pnl_net, model_signal (JSON),
   features_snapshot (JSON), regime_at_entry, holding_period_sec, status.
2. model_performance: id, date, model_name, model_version, total_signals,
   correct_signals, accuracy, profit_factor, sharpe_ratio, max_drawdown.
3. daily_summary: date, total_trades, winning_trades, gross_pnl, total_costs,
   net_pnl, max_drawdown_today, equity_at_close.
Methods: log_trade(), update_trade(), get_daily_summary(), get_model_performance().
Write tests.
```

---

#### Module 7: Alerts + Orchestrator

**Prompt 21:**
```
Build alerts/telegram_bot.py: async Telegram bot using python-telegram-bot or
simple httpx/requests to Telegram Bot API. Functions:
- send_trade_alert(trade_info): "BUY RELIANCE @ Rs. 2,850 | Qty: 166 | SL: Rs. 2,820 | Confidence: 78%"
- send_error_alert(error_msg): "ERROR: WebSocket disconnected. Reconnecting..."
- send_daily_summary(summary): "Daily P&L: +Rs. 3,200 | Trades: 5 | Win Rate: 60%"
- send_kill_switch_alert(reason): "KILL SWITCH: Daily loss limit hit (-2.1%). All trading stopped."
Load TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env.
```

**Prompt 22:**
```
Build main.py: the orchestrator that ties everything together.
Main loop:
1. Login to broker API
2. Start WebSocket feed in background thread
3. Every 15 minutes (on candle close):
   a. Get latest candles from candle_store
   b. Compute all 23 features via feature engine
   c. Run ML ensemble prediction
   d. Check risk manager: can_trade()?
   e. If signal + risk approved → calculate position size → place order
   f. Log everything to trade_logger
   g. Send Telegram alert if trade placed
4. Every 1 minute: check open positions, update trailing stops, check for exits
5. At force_close_time: close all open positions
6. At end of day: log daily summary, send Telegram daily report

CLI arguments: --mode paper|live, --config path/to/config, --symbols override
Graceful shutdown on Ctrl+C (close positions if in live mode, or just exit if paper).
```

---

#### Module 8: Backtesting + Training Scripts

**Prompt 23:**
```
Build backtest/runner.py: walk-forward backtesting engine.
1. Load historical candles from candle_store or CSV
2. Compute features
3. Apply triple-barrier labels
4. For each walk-forward fold:
   a. Train ensemble on training set (with Optuna tuning)
   b. Predict on test set
   c. Simulate trades with realistic cost model
   d. Track equity curve, trades, metrics
5. Output: aggregated metrics across all folds (Sharpe, profit factor,
   max drawdown, win rate), equity curve plot, per-model accuracy breakdown.
6. Log everything to MLflow.
Reject any strategy with walk-forward Sharpe < 1.0.
```

**Prompt 24:**
```
Build scripts/download_history.py: download 2+ years of historical data.
Use Shoonya get_historical_data API for 15-min candles (PRIMARY broker — see
broker_shoonya.py). Also download Nifty 50 index data (for relative strength feature).
Store in QuestDB via candle_store.py (NOT SQLite — QuestDB is the time-series store).
Handle API rate limits (Shoonya: max ~3 requests/second).
Show progress bar with tqdm.
```

**Prompt 25:**
```
Write all tests in tests/. Use pytest. Each test file should:
- Test happy path
- Test edge cases (empty data, NaN values, insufficient bars)
- Test the labeling thoroughly (create known price patterns, verify labels manually)
- Test risk manager kill switches trigger correctly
- Test cost model against hand-calculated values
- Test walk-forward split has no data leakage
Run: pytest tests/ -v --tb=short
```

---

### Step 5: After Building — The Real Work Begins

Once Claude Code has built all modules:

```bash
# 1. Download historical data
python scripts/download_history.py --symbols RELIANCE TCS INFY --years 2

# 2. Run backtest to validate
python scripts/train_models.py --timeframe 15min

# 3. Check MLflow for results
mlflow ui --port 5000
# Open http://localhost:5000 in browser

# 4. If Sharpe > 1.0 on walk-forward → proceed to paper trading
python main.py --mode paper

# 5. Paper trade for minimum 3 months / 40 sessions. Track in QuestDB (not SQLite).
#    Gate: Sharpe >= 1.0 after costs, zero unreconciled events, kill switch tested.

# 6. Only after 4+ weeks of paper profit → go live
python main.py --mode live
```

---

### Tips for Working with Claude Code

1. **One prompt = one module.** Don't ask for everything at once.
2. **Review every file** Claude Code creates. Read the code. Understand it.
3. **Run tests after every module.** Don't move on until tests pass.
4. **Use `/compact`** if the conversation gets long (frees up context window).
5. **Git commit after each working module.** `git add . && git commit -m "Module 1: data pipeline"`
6. **If Claude Code makes a mistake,** describe the error and let it fix it.
7. **The CLAUDE.md file is your contract.** Claude Code reads it every session.

---

### Total Cost Summary (Claude Code / Production Path)

| Item | Cost | Required? |
|---|---|---|
| Claude Code (Claude Pro subscription) | $20/month (approx. Rs. 1,700) | Yes |
| Indian VPS with static IP | Rs. 500-1500/month | Yes (SEBI compliance) |
| TrueData (clean historical data) | Rs. 1000/month | Recommended, not required |
| Broker account (Angel One / Shoonya) | Free | Yes |
| All Python libraries | Free | Yes |
| Ollama + LLMs (sentiment layer) | Free | Optional |
| **Total minimum** | **~Rs. 2,200/month** | |
| **Total recommended** | **~Rs. 3,200/month** | |

All libraries, frameworks, and the ML/RL stack are 100% free and open-source.

---

---

## Section 2: Jupyter / Kaggle Notebook Setup

*Content from SETUP_GUIDE_JUPYTER.md — Jupyter-First Algo Trading Development*

---

### The Approach: Research in Jupyter → Production in Scripts

```
Phase 1 (Jupyter):  Explore → Build → Prove it works → See charts
Phase 2 (Scripts):  Convert proven notebooks → Production modules → Live trade
```

| Task | Use Jupyter | Use Python Scripts |
|---|---|---|
| Download & explore data | Yes | |
| Compute & visualize features | Yes | |
| Triple-barrier labeling | Yes | |
| Train ML models | Yes | |
| Optuna tuning | Yes | |
| Backtesting with equity curves | Yes | |
| Walk-forward validation | Yes | |
| Live WebSocket feed | | Yes |
| Order execution | | Yes |
| Risk management (kill switches) | | Yes |
| Main trading loop | | Yes |
| Telegram alerts | | Yes |

---

### Step 1: Install Everything (Jupyter-First Stack)

#### Python + Jupyter
```bash
# Create project folder
mkdir algo_trading
cd algo_trading

# Python virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install core dependencies
pip install jupyter jupyterlab notebook

# Install trading & ML stack
pip install pandas numpy pandas-ta
pip install xgboost lightgbm catboost scikit-learn
pip install optuna mlflow
pip install hmmlearn filterpy arch statsmodels
pip install matplotlib plotly seaborn
pip install yfinance            # Free historical data (start here)
pip install smartapi-python     # Angel One (when ready for live)
pip install python-dotenv tqdm httpx pyotp
pip install pytest

# Optional (Phase 2+)
pip install stable-baselines3 gymnasium   # RL
pip install ollama                        # LLM sentiment

# Launch Jupyter
jupyter lab
```

#### Claude Code (for when you convert to production)
```bash
# Native installer (recommended)
curl -fsSL https://downloads.claude.ai/claude-code/install.sh | sh

# Or npm
npm install -g @anthropic-ai/claude-code

# Requires Claude Pro subscription ($20/month)
```

---

### Step 2: Project Structure

```
algo_trading/
├── CLAUDE.md                      # Project spec (for Claude Code later)
├── .env                           # API keys
├── requirements.txt
│
├── notebooks/                     # RESEARCH (Jupyter - you work here first)
│   ├── 01_data_download.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_labeling.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_optuna_tuning.ipynb
│   ├── 06_backtest.ipynb
│   ├── 07_rl_position_sizing.ipynb     # Optional
│   └── 08_sentiment_analysis.ipynb     # Optional
│
├── src/                           # PRODUCTION (Claude Code builds this)
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── labeling/
│   ├── models/
│   ├── risk/
│   ├── execution/
│   └── alerts/
│
├── main.py                        # Live trading entry point
├── data/                          # CSVs + historical downloads (ticks/bars stored in QuestDB, not here)
└── tests/
```

---

### Step 3: Jupyter Notebook Playbook

#### Notebook 1: Data Download & Exploration

**What you'll learn:** How Indian market data looks, gaps, volumes, OHLCV structure.

**If using Claude Code to generate the notebook:**
```
Create notebooks/01_data_download.ipynb that:

1. Uses yfinance to download 2 years of 15-min data for these NSE stocks:
   RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS,
   BHARTIARTL.NS, ITC.NS, LT.NS, KOTAKBANK.NS
   Also download ^NSEI (Nifty 50 index).

2. For each stock, show:
   - df.head(), df.tail(), df.shape, df.info()
   - Check for missing values and gaps
   - Daily volume distribution (histogram)
   - Price chart with volume bars below (use plotly for interactive)

3. Save all data to data/historical/ as parquet files (one per symbol).

4. Print summary: date range, total bars per symbol, any gaps found.

Use markdown cells to explain each step. This is a learning notebook.
```

**If writing yourself, the key code:**
```python
import yfinance as yf
import pandas as pd

# Download 2 years of 15-min data
symbol = "RELIANCE.NS"
df = yf.download(symbol, period="2y", interval="15m")
df.to_parquet(f"data/historical/{symbol.replace('.NS','')}_15min.parquet")
```

> **Note:** yfinance gives ~60 days of 15-min data for free. For 2 years of
> intraday data, you'll need your broker's historical API or TrueData.
> Start with what yfinance gives — it's enough to build and test the pipeline.
> Daily data goes back much further (2+ years available free).

---

#### Notebook 2: Feature Engineering (23 Features)

**Claude Code prompt:**
```
Create notebooks/02_feature_engineering.ipynb that:

1. Load RELIANCE 15-min parquet from data/historical/.

2. Compute ALL 23 features, one section per feature with explanation:

   TECHNICAL (using pandas_ta):
   - RSI(14): what it means, plot it, show overbought/oversold zones
   - MACD(12,26,9): plot MACD line, signal line, histogram
   - Bollinger %B and Bandwidth: plot bands on price chart
   - ATR(14): plot volatility over time
   - ADX(14): plot trend strength
   - Stochastic %K/%D: plot with overbought/oversold
   - OBV: plot cumulative volume
   - Supertrend: plot on price chart with buy/sell signals
   - EMA Cross (9/21): plot both EMAs, mark crossover points
   - Ichimoku Cloud: plot cloud, mark price position relative to cloud
   - Heikin-Ashi direction: show HA candles
   - Pivot Point distance: calculate and plot

   STATISTICAL:
   - Kalman Filter smoothed price: plot raw vs smoothed
   - HMM Regime detection (3 states): color price chart by regime
   - GARCH volatility forecast: plot predicted vs realized volatility
   - Returns autocorrelation (lags 1-5): plot autocorrelation function

   CUSTOM:
   - VWAP deviation: plot VWAP and deviation from it
   - Order book imbalance: explain that this needs live L2 data,
     create placeholder column filled with 0 for now
   - Price-Volume divergence: rolling correlation between returns and volume
   - Relative Strength vs Nifty: stock return minus Nifty return, rolling 20-bar
   - Time-of-Day encoding: sin/cos of minutes since 9:15 AM
   - Bid-Ask spread: explain needs live L2 data, placeholder for now

3. After computing all features, show:
   - Correlation heatmap of all 23 features (use seaborn)
   - Distribution of each feature (histograms)
   - Feature statistics table: mean, std, min, max, % NaN

4. Save the feature DataFrame to data/features/RELIANCE_features.parquet

Use lots of markdown explanations. The person reading this is learning
what each feature means and why it matters for trading.
```

---

#### Notebook 3: Triple-Barrier Labeling

**Claude Code prompt:**
```
Create notebooks/03_labeling.ipynb that:

1. Load the feature DataFrame from notebook 02.

2. EXPLAIN the triple-barrier method visually:
   - Draw a diagram showing: entry price, upper barrier (+2 ATR),
     lower barrier (-1 ATR), time barrier (20 bars)
   - Use plotly to show an interactive price chart with barriers drawn
     for a specific example trade

3. Implement triple_barrier_label(df, upper_mult=2.0, lower_mult=1.0, max_bars=20):
   - For each bar, compute upper and lower barriers using ATR
   - Look forward: which barrier gets hit first?
   - Label: +1 (upper hit = profit), -1 (lower hit = loss), 0 (timeout)
   - Return labels + metadata (which bar the barrier was hit, actual return)

4. Apply to the full dataset. Show:
   - Class distribution: how many +1, -1, 0 (bar chart)
   - Average return for each label class (verify +1 has positive, -1 negative)
   - Example trades: plot 5 random +1 labels and 5 random -1 labels
     on price charts with barriers visible

5. Experiment with different barrier settings:
   - upper_mult: [1.5, 2.0, 2.5, 3.0]
   - lower_mult: [0.5, 1.0, 1.5]
   - max_bars: [10, 20, 30]
   Show how class distribution changes. Pick the best combination.

6. EXPLAIN walk-forward validation:
   - Why random train/test split is WRONG for time series
   - What purge and embargo mean, with a visual timeline diagram
   - Implement walk_forward_split with purge and embargo
   - Show the splits visually (colored bars on timeline)

7. Save labeled DataFrame to data/labeled/RELIANCE_labeled.parquet
```

---

#### Notebook 4: Model Training (ML Ensemble)

**Claude Code prompt:**
```
Create notebooks/04_model_training.ipynb that:

1. Load labeled data from notebook 03.

2. Split features (X) and labels (y). Drop label=0 (timeout) OR keep as
   a 3-class problem. Start with binary: +1 vs -1 only.

3. Train three models on the FIRST walk-forward fold:
   a. XGBoost classifier with default params
   b. LightGBM classifier with default params
   c. CatBoost classifier with default params

   For each model, show:
   - Classification report (precision, recall, F1 per class)
   - Confusion matrix (plotted as heatmap)
   - Feature importance (top 15 features, horizontal bar chart)
   - ROC curve and AUC score

4. Build ensemble:
   - Get predicted probabilities from each model
   - Weighted average (equal weights to start)
   - Convert to signal: prob > 0.6 → BUY, prob < 0.4 → SELL, else HOLD
   - Show ensemble accuracy vs individual model accuracy

5. Run FULL walk-forward evaluation (all folds):
   - For each fold: train on training set, predict on test set
   - Collect all test predictions across folds
   - Show OVERALL walk-forward metrics:
     * Accuracy, F1, Precision, Recall
     * Confusion matrix
     * Equity curve (assuming Rs. 1,00,000 starting capital, 1% risk per trade,
       volatility-adjusted sizing, full India cost model)
     * Monthly returns table
     * Sharpe ratio, max drawdown, profit factor, win rate

6. CRITICAL CHECK at the end:
   - Print "WALK-FORWARD SHARPE RATIO: X.XX"
   - If Sharpe > 1.0: "Strategy shows potential. Proceed to tuning."
   - If Sharpe < 1.0: "Strategy does NOT have edge. Do NOT proceed.
     Go back to features and labeling. Try different settings."

This is the most important notebook. Take time to understand every output.
```

---

#### Notebook 5: Optuna Hyperparameter Tuning

**Claude Code prompt:**
```
Create notebooks/05_optuna_tuning.ipynb that:

1. Load labeled data.

2. Define Optuna objective function for each model:
   - Use walk-forward cross-validation (not random split!) as the evaluation
   - Objective: maximize average walk-forward F1-score
   - Search space per model:
     * max_depth: 3-10
     * learning_rate: 0.01-0.3 (log scale)
     * n_estimators: 100-1000
     * subsample: 0.6-1.0
     * colsample_bytree: 0.6-1.0
     * min_child_weight: 1-10
     * reg_alpha: 1e-8 to 10 (log)
     * reg_lambda: 1e-8 to 10 (log)

3. Run 200 trials per model (takes time — show progress).
   Use Optuna's MedianPruner to stop bad trials early.

4. Show results:
   - Best params for each model
   - Optimization history plot (Optuna built-in visualization)
   - Parameter importance plot
   - Compare: default params vs tuned params (table of metrics)

5. Retrain ensemble with tuned params. Run walk-forward again.
   Show improvement in Sharpe, profit factor, drawdown.

6. Log everything to MLflow:
   - Best params, metrics, trained models as artifacts
   - Show how to view results: mlflow ui --port 5000

7. Save tuned models to data/models/ using joblib.
```

---

#### Notebook 6: Full Backtest with Equity Curve

**Claude Code prompt:**
```
Create notebooks/06_backtest.ipynb that:

1. Load tuned ensemble models from notebook 05.

2. Implement realistic backtest engine:
   - Walk-forward: train on past, predict on future, roll forward
   - On each signal:
     * Calculate position size (volatility-adjusted, ATR-based)
     * Apply FULL India cost model (brokerage, STT, GST, stamp, slippage)
     * Entry at next-bar open + 0.05% slippage
     * Stop-loss at entry - (lower_mult x ATR)
     * Take-profit at entry + (upper_mult x ATR)
     * Time exit at max_holding_bars if neither SL nor TP hit
   - Track equity curve bar-by-bar

3. Visualize:
   - Equity curve (Rs.) with drawdown shading (plotly interactive)
   - Monthly returns heatmap (calendar style)
   - Trade distribution: histogram of P&L per trade
   - Win rate by hour of day (does the model work better at certain times?)
   - Win rate by regime (bull vs bear vs sideways)
   - Cumulative return vs buy-and-hold Nifty (benchmark comparison)
   - Rolling Sharpe ratio (63-bar window)

4. Print final report card:
   - Total return, Annualized return, Sharpe ratio
   - Max drawdown (% and Rs.), Max drawdown duration
   - Profit factor, Win rate, Avg win / Avg loss
   - Total trades, Avg holding period
   - Total costs paid (brokerage + taxes + slippage)
   - Best month, Worst month
   - Expectancy per trade (Rs.)

5. Stress test:
   - What happens if slippage is 2x the model? (0.10% instead of 0.05%)
   - What happens if brokerage is Rs. 20 instead of Rs. 0?
   - Show how metrics change under worse assumptions.

6. FINAL VERDICT:
   - If walk-forward Sharpe > 1.5 and Profit Factor > 1.5 and
     Max Drawdown < 15%: "READY FOR PAPER TRADING"
   - Otherwise: "NOT READY. Iterate on features/labels/tuning."
```

---

#### Notebook 7 (Optional): RL Position Sizing

**Claude Code prompt:**
```
Create notebooks/07_rl_position_sizing.ipynb that:

1. Explain: ML ensemble decides WHAT to trade. RL agent decides HOW MUCH.

2. Set up a custom Gymnasium environment:
   - State: current feature values + current position + portfolio equity +
     recent P&L + current ATR (volatility)
   - Action: continuous [0, 1] = fraction of max allowed position size
   - Reward: risk-adjusted P&L (penalize drawdowns)
   - Episode: one walk-forward test period

3. Train PPO agent using Stable-Baselines3:
   - Train on same walk-forward folds as ML ensemble
   - 100K-500K timesteps per training run

4. Compare: fixed position sizing vs ATR-based vs RL-based
   - Equity curves side by side
   - Sharpe ratio comparison table
   - Max drawdown comparison

5. Save trained RL model if it improves metrics.

Note: This notebook requires stable-baselines3 and a GPU helps but isn't required.
Skip this notebook initially — come back to it after the base system works.
```

---

#### Notebook 8 (Optional): News Sentiment

**Claude Code prompt:**
```
Create notebooks/08_sentiment_analysis.ipynb that:

1. Explain: LLM used ONLY for daily news sentiment, NOT for trade decisions.

2. Collect sample news headlines (use RSS feeds or hardcoded examples):
   - MoneyControl, Economic Times, Livemint for Indian market news
   - Show 20-30 example headlines

3. Use Ollama (local LLM) to score sentiment:
   - Install: curl -fsSL https://ollama.com/install.sh | sh
   - Pull model: ollama pull qwen2.5:7b
   - For each headline, prompt: "Rate the sentiment of this financial news
     headline for Indian stock markets. Reply with ONLY a number from
     -1 (very bearish) to +1 (very bullish). Headline: {headline}"
   - Parse response to float

4. Show sentiment distribution, correlation with next-day Nifty returns.

5. If correlation is meaningful (>0.05), add as feature #24 in the ML pipeline.
   If not, don't use it — adding noise features hurts the model.

Note: Requires Ollama installed and ~4GB RAM for the 7B model.
Skip this notebook initially — it's an enhancement, not core.
```

---

### Step 4: Convert Notebooks → Production Code (Claude Code)

Once notebooks 01-06 are working and your walk-forward backtest shows Sharpe > 1.0:

```bash
cd algo_trading
claude
```

**The conversion prompt:**
```
I have 6 working Jupyter notebooks in notebooks/ that implement an algo trading
system for Indian markets. The research is done and validated. Now I need to convert
this into production Python modules that can run as a live trading bot.

Read the CLAUDE.md for the full project spec and module structure.
Read each notebook to understand the working code.

Convert the proven notebook code into the production module structure defined in
CLAUDE.md, following the build order specified there. Start with config.py and
work through each module. Each module must:
- Work standalone (importable and testable)
- Use logging instead of print
- Handle errors gracefully
- Have type hints and docstrings
- Have a corresponding test file in tests/

Start with Module 0 (config + setup), then ask me before proceeding to each next module.
```

---

### Summary: The Full Journey

```
Week 1-2:   Notebooks 01-03 (data, features, labeling)
            → You understand the data and features visually
            → You understand WHY triple-barrier works

Week 3-4:   Notebooks 04-05 (training, tuning)
            → You have a trained, tuned ML ensemble
            → You know your walk-forward Sharpe ratio

Week 5:     Notebook 06 (backtest)
            → You have proof (or disproof) that the system works
            → STOP HERE if Sharpe < 1.0. Go back and iterate.

Week 6-7:   Claude Code converts notebooks → production code
            → Broker API connection (Angel One / Shoonya)
            → Risk manager, execution layer, Telegram alerts

Week 8-11+: Paper trading (main.py --mode paper)
            → Minimum 3 calendar months / 40 sessions (single consolidated gate — see DECISIONS.md).
            → Track everything in QuestDB (not SQLite).
            → Weekly review: is live performance matching backtest?

Week 12+:   Live trading (main.py --mode live) — only if paper was profitable
```

---

### Cost: Jupyter-First Approach

| Item | Cost | When Needed |
|---|---|---|
| Python + Jupyter + all ML libraries | Free | Now |
| yfinance historical data | Free | Now |
| Claude Pro (for Claude Code, Phase 2) | Rs. 1,700/month | Week 6+ |
| Indian VPS with static IP | Rs. 500-1,500/month | Week 8+ (live trading) |
| Broker account (Angel One / Shoonya) | Free | Week 6+ |
| TrueData (better historical data) | Rs. 1,000/month | Optional |
| **Phase 1 cost (notebooks)** | **Rs. 0** | **Weeks 1-5** |
| **Phase 2 cost (production)** | **~Rs. 2,200/month** | **Week 6+** |

**You can build and validate the entire research pipeline for free.**
Claude Code is only needed when you're ready to convert to production.
And even then — you can use this chat (Claude.ai) to help write the
production code if you prefer. Claude Code just makes it faster.

---

## Section 3: Open-Source Tools Practical Guide
### Source: OpenSource_Free_Tools_Practical_Guide.txt (extracted May 2026)
### Focus: "DIG" tips, install script, deprecated tools, free alternatives — not duplicated in MASTER_RESEARCH_DATABASE_v3.md

---

### Quick Install Script — Full Trading Stack

```bash
# Core ML stack:
pip install xgboost lightgbm catboost scikit-learn pandas numpy polars

# Backtesting:
pip install nautilus_trader lib-pybroker vectorbt zipline-reloaded

# Technical indicators + statistics:
pip install pandas_ta hmmlearn arch filterpy ruptures statsmodels

# RL:
pip install stable-baselines3[extra] gymnasium finrl

# LLM / NLP:
pip install transformers torch sentencepiece accelerate

# Time-series foundation models:
pip install uni2ts timesfm
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Emerging architectures:
pip install mamba-ssm causal-conv1d pykan ncps

# Infrastructure:
pip install mlflow dvc redis kafka-python questdb chromadb

# Data (India):
pip install jugaad-data nsepy yfinance akshare kiteconnect dhanhq NorenRestApiPy

# Portfolio + optimization:
pip install PyPortfolioOpt Riskfolio-Lib optuna

# Explainability (SEBI compliance):
pip install shap dowhy openbb

# Analysis + visualization:
pip install matplotlib seaborn plotly scipy
```

---

### What's Not Free and Its Free Replacement

| Paid Tool | Free Alternative |
|-----------|-----------------|
| Bloomberg Terminal ($25K/yr) | OpenBB Terminal (free) + free data providers |
| Bloomberg BQuant | Qlib + Jupyter (free) |
| NVIDIA RAPIDS (needs NVIDIA GPU) | Polars + Numba (CPU-optimized) |
| VectorBT PRO ($20/mo) | PyBroker (free, better for ML anyway) |
| QuantConnect Cloud | Lean self-hosted (free) |
| Kite API (Rs.2,000/mo) | Shoonya API (free, zero brokerage) or Angel One SmartAPI (free) |
| TimeGPT API (paid after 500 calls) | Chronos / Moirai (run locally, unlimited) |
| Kensho / Bloomberg BNLP | FinBERT + FinLLaMA (local, free) |
| Refinitiv/Reuters data | BSE announcements + jugaad-data (free) |

---

### Deprecated Tools — Do Not Use

| Tool | Status | Better Alternative |
|------|--------|--------------------|
| AWS FinSpace | SHUTTING DOWN October 2026 | QuestDB |
| Backtrader | Last update May 2019, effectively abandoned | NautilusTrader (production) or PyBroker (ML) |
| AutoGPT for trading | Never worked reliably, obsolete | TradingAgents or CrewAI flows |
| BloombergGPT | Not publicly accessible | FinBERT or FinGPT |
| VectorBT open-source for ML | Stagnant (active dev moved to PRO commercial) | PyBroker for ML strategies |

---

### Key DIG Tips — What to Mine From Each Repo

**NautilusTrader** — Use `InteractiveBrokersAdapter` in `/adapters/interactive_brokers/` as a template for building a ShoonyaAdapter for NSE. The `/examples/backtest/` folder has complete working equity/futures examples.

**QuantConnect Lean** — The `BrokerageModel` class is how you plug in custom cost models. Extend `DefaultBrokerageModel` with NSE STT + GST + stamp duty calculations. Check `/Brokerages/` for broker implementation patterns.

**PyBroker** — `/examples/walk_forward.ipynb` shows the correct way to avoid look-ahead bias in ML backtesting: model training happens inside the temporal loop, not before it. BCa bootstrap Sharpe gives statistically meaningful confidence intervals.

**Jugaad-Data** (9/10 for India) — The bhavcopy downloader gives full market daily data for all ~2000 NSE stocks. Use for universe screening:
```python
from jugaad_data.nse import stock_df
df = stock_df(symbol="RELIANCE", from_date=date(2020,1,1),
              to_date=date(2024,12,31), series="EQ")
```

**NSEpy** — The option chain history endpoint is rare; most free tools don't have F&O historical data. NSE website changes sometimes break it — use jugaad-data as fallback for OHLCV.

**Optuna** — After running a study, use `optuna.visualization.plot_param_importances(study)` to identify which hyperparameters actually matter. Usually only 2-3 params significantly affect walk-forward results. Run 200+ trials overnight for meaningful improvement.

**Microsoft Qlib** — `/qlib/contrib/data/handler.py` contains Alpha158 — 158 pre-built factor definitions directly applicable to NSE data. These are production-quality feature engineering patterns from Microsoft Research.

**QuestDB** — Key SQL patterns:
```sql
-- 15-minute candles from tick data (SAMPLE BY):
SELECT timestamp, symbol,
       first(price) AS open, max(price) AS high,
       min(price) AS low, last(price) AS close,
       sum(volume) AS volume
FROM nse_ticks
WHERE symbol = 'RELIANCE'
SAMPLE BY 15m

-- Latest tick per symbol (LATEST ON):
SELECT * FROM nse_ticks LATEST ON timestamp PARTITION BY symbol
```

**ChromaDB** — Regime analog search (underused technique, no model training needed):
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("market_states")
for date, state_vector in historical_states:
    collection.add(embeddings=[state_vector.tolist()], ids=[str(date)],
                   metadatas=[{"return_next_week": 0.023}])
results = collection.query(query_embeddings=[current_state.tolist()], n_results=10)
analog_signal = np.mean([m["return_next_week"] for m in results["metadatas"][0]])
```

**SHAP** — Mandatory for SEBI compliance. Include beeswarm plot in algo registration docs:
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # Beeswarm plot for SEBI documentation
```

---

## Section 4: External Requirements and Capital Checklist
### Source: EXTERNAL_REQUIREMENTS.txt (extracted May 2026)
### Note: Original file used Angel One as primary broker — corrected to Shoonya per CLAUDE.md

---

### Pre-Session-1 Checklist

**Computer (verify):**
- Intel i5-1345U or better, 16GB RAM, 50GB free disk space, Windows 10/11+, stable internet during 9:15am-3:30pm IST market hours

**Software to install (all free):**
- Python 3.11 or 3.12 — tick "Add Python to PATH" during install
- VS Code with Python + Pylance extensions
- Git
- Docker Desktop — needs virtualization enabled in BIOS; confirm "Running" status; then run `docker run -d -p 9000:9000 -p 9009:9009 --name questdb questdb/questdb`

**Broker accounts:**
- **Shoonya/Finvasia** (primary — zero brokerage) at shoonya.com. Documents: Aadhaar, PAN, bank account. Time: 2-3 working days.
- Credentials to provide at Session 1: User ID, Password, API Key (from Shoonya developer portal)
- Angel One SmartAPI (dev reference only — not required before Session 1 for this project)
- Kaggle account with phone verified (for free T4 GPU — 30 hrs/week for Optuna hyperparameter runs)

**Telegram bot (needed before Session 43):**
1. Telegram → @BotFather → Start → `/newbot` → give name + username (must end in `bot`)
2. Save bot token: format `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
3. Get chat ID: @userinfobot → Start → note the number

---

### Capital Requirements by Stage

| Phase | Capital Needed | Notes |
|-------|---------------|-------|
| Development (Sessions 1-50) | Rs. 0 | Paper trading only |
| Live order testing (Sessions 36-37) | Rs. 5,000-10,000 | Small test orders to verify fills |
| Paper trading (Sessions 51-53) | Rs. 0 | Simulated, no real money |
| Live trading start | Rs. 50,000-1,00,000 | Start small, validate first |
| Minimum for equity intraday | Rs. 2,00,000 | Below this, position sizes become impractical |
| Comfortable multi-stock system | Rs. 5,00,000 | Good position sizing flexibility |
| Nifty/BankNifty Futures (Stage 2) | Rs. 3,00,000-5,00,000 | Lot margin + drawdown buffer |
| Index Option Buying (Stage 3A) | Rs. 1,00,000-3,00,000 | Buy premium upfront |

---

### Key Decisions You Must Make

| Decision | Needed Before | Recommended Default |
|----------|--------------|---------------------|
| Stock watchlist | Session 21 | RELIANCE, HDFCBANK, ICICIBANK, TCS, INFY |
| Primary timeframe | Session 21 | 15-minute |
| Capital amount | Session 33 | Rs. 2,00,000 minimum |
| Risk per trade | Session 33 | 1% of capital |
| Stop loss multiplier | Session 33 | 1.5x ATR14 |
| Risk tolerance | Session 34 | Moderate |
| Trading type | Session 29 | Equity Intraday (MIS) |

**Risk tolerance tiers (Session 34):**

| Level | Daily Loss Limit | Max Drawdown | Max Position Size |
|-------|-----------------|--------------|-------------------|
| Conservative | 1% of capital | 5% from peak | 3% of capital |
| Moderate (default) | 2% of capital | 10% from peak | 5% of capital |
| Aggressive | 3% of capital | 15% from peak | 7% of capital |

At Rs. 2,00,000 with Moderate: daily loss stop = Rs. 4,000; drawdown stop = Rs. 20,000; max position = Rs. 10,000.

---

### Sessions Requiring Live Market Hours (NSE 9:15am-3:30pm IST, weekdays only)

| Session | Activity |
|---------|----------|
| Session 5 | WebSocket live feed testing |
| Session 6 | Candle aggregation from live ticks |
| Session 36 | LIMIT order testing (Rs. 5,000+ in account required) |
| Session 37 | MARKET order + retry logic testing |
| Session 41 | Signal pipeline live integration (optional) |
| Session 42 | Paper trading live run |
| Sessions 51-53 | Full paper trading (every market day for 2 weeks) |

---

### SEBI Strategy Registration (Session 49)

**Cannot go live without this.** August 2025 mandate, mandatory from April 2026.

Steps:
1. Login to broker's algo registration portal
2. Fill strategy name, description, risk parameters, asset class
3. Attach SHAP explainability report (generated at Session 27)
4. Attach 3-month paper trading P&L report (from Sessions 51-53)
5. Attach personal declaration form (signed digitally)
6. Submit → broker forwards to NSE/BSE for Strategy ID assignment
7. Approval time: 5-15 working days
8. Strategy registration fee: Rs. 500-2,000 (confirm with your broker)

**Strategy ID** goes into every order via the `tag` field. Without it, orders may be rejected by the exchange.

---

### VPS vs Laptop for Live Trading (Session 56 Decision)

| Option | Monthly Cost | Latency to NSE | Notes |
|--------|-------------|----------------|-------|
| Your laptop | Rs. 0 | Variable | Power cuts and WiFi drops stop trading. OK for testing phase only. |
| DigitalOcean Mumbai | ~Rs. 800 (2GB RAM) | 9ms | Recommended first VPS choice |
| AWS Lightsail Mumbai | ~Rs. 1,200 | ~10ms | More expensive, comparable performance |
| Hostinger VPS India | ~Rs. 500 | Variable | Cheapest; check SLA carefully |
| Hetzner Cloud India | ~Rs. 600 | ~10ms | Best performance/price ratio |

Setup on Linux VPS: Ubuntu 22.04 → install Python + Docker + Git → copy code → configure systemd auto-start at 9:00am IST.

---

### Monthly Recurring Costs Summary

| Item | Cost |
|------|------|
| Shoonya broker + API | Rs. 0 |
| QuestDB (local Docker or VPS) | Rs. 0 |
| Python, VS Code, Git | Rs. 0 |
| Kaggle GPU (Optuna runs) | Rs. 0 (30 hrs/week) |
| VPS for live trading (optional, recommended) | Rs. 500-2,000 |
| Cloud backup for MLflow/DVC | Rs. 0 (Google Drive 15GB free) to Rs. 150 (AWS S3) |
| SEBI Strategy Registration | Rs. 500-2,000 ONE-TIME |
