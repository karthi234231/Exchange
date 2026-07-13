# Architecture Overview — NSE AlgoTrader

Single authoritative architecture document. Supersedes any conflicting descriptions
in `MD_Files/` or `arch-spec.json` for the current deployed codebase.

Last updated: 2026-06-22 | Current stage: Stage 1 (paper trading)

---

## 1. System Overview

NSE AlgoTrader is a three-process ZeroMQ pipeline that separates market data ingestion,
feature computation, and ML inference/execution into independent OS processes, bypassing
Python's GIL for true parallelism.

**Three core processes:**

1. **DataIngestor** — Shoonya WebSocket → bar builder → ZMQ PUB:5555
2. **FeatureComputer** — ZMQ SUB:5555 → 25 features → ZMQ PUB:5556
3. **MLInference + OrderExecutor** — ZMQ SUB:5556 → signal → risk check → broker order

A fourth process, **AlertSink**, runs independently and handles Telegram delivery
without ever blocking the execution path.

**Target latency (Mumbai VPS):** 200ms tick-to-order. From home connection: 300–500ms.

---

## 2. Data Flow

```
Shoonya WebSocket
    |
    | (1-3 snapshots/sec, Stage 1)
    v
DataIngestor (daily_orchestrator.py)
    |--- builds 15-min time bars
    |--- validates ticks (TickValidator)
    |--- writes candles to QuestDB:9000 (ILP)
    |
    | ZMQ PUB:5555 (msgpack-encoded bars)
    v
FeatureComputer (features/engine.py)
    |--- 25 features: technical, statistical, microstructure
    |--- validates feature vector (no NaN/Inf)
    |
    | ZMQ PUB:5556 (feature vectors)
    v
MLInference (models/inference_engine.py)
    |--- Treelite .so inference (~2ms vs ~40ms native)
    |--- produces signal: -1 (bear) / 0 (sideways) / +1 (bull)
    |--- ML label map: class 0=bear, 1=sideways, 2=bull
    |
    | ZMQ PUSH:5557 (signal + confidence)
    v
RiskManager (risk/manager.py)
    |--- kill switch check
    |--- daily loss limit (2% of capital)
    |--- max drawdown (10% from peak)
    |--- position size limit (5% per position)
    |--- correlation guard
    |--- fat-finger check
    |
    v
OrderExecutor (execution/order_executor.py)
    |--- order FSM: NEW → SENT → ACK → PARTIAL → FILLED/CANCELLED/REJECTED
    |--- DualLockRateLimiter: global 8 OPS + per-order max 5 modifications
    |--- builds broker payload (client_order_id in remarks field)
    |
    | Shoonya REST API
    v
Broker (execution)
    |
    | WebSocket order update callbacks
    v
OrderFSM (execution/order_fsm.py)
    |--- ACK / FILL / CANCEL_CONFIRM / REJECT transitions
    |--- rate limiter released on all terminal states
    |
    v
LiveSystemState (market_data/system_state.py)
    |--- position ledger updated on every fill
    |--- persisted to state/live_state.json every 5 seconds
    |--- writes to QuestDB trade/fill tables
    |
    v
AlertSink (alerts/alert_sink.py)   <--- ZMQ PULL:5560
    |--- receives alerts from execution path (non-blocking PUSH)
    |--- sends to Telegram (decoupled; failure does not affect trading)
```

---

## 3. Key Components Table

| Module | Purpose | ZMQ Role | Threading Model |
|--------|---------|----------|-----------------|
| `orchestrator/daily_orchestrator.py` | Session lifecycle: login → market open → EOD → logout | — | Main thread + APScheduler |
| `market_data/` | Shoonya WebSocket, bar builder, tick validation | PUB:5555 | Dedicated thread per symbol [ASSUMED] |
| `features/engine.py` | Orchestrates 25 features on each bar | SUB:5555 → PUB:5556 | Single event loop |
| `features/technical.py` | RSI, MACD, BB, ATR, ADX, Stoch, EMA, etc. | — | Called synchronously by engine |
| `features/statistical.py` | HMM regime, GARCH vol, Kalman, autocorrelation | — | Called synchronously by engine |
| `features/microstructure.py` | VPIN filter, Avellaneda-Stoikov reservation price | — | Stage 2+ only; Hawkes/TFI blocked |
| `models/inference_engine.py` | Treelite .so ensemble inference | SUB:5556 → PUSH:5557 | Single event loop |
| `models/model_registry.py` | FrozenModelSpec validation; blocks stub models | — | Startup only |
| `execution/order_executor.py` | Order FSM + rate limiter + broker calls | PULL:5557 | Single execution thread |
| `execution/order_fsm.py` | Order state machine (NEW/SENT/ACK/PARTIAL/FILLED/etc.) | — | Called by executor |
| `execution/rate_limiter.py` | DualLockRateLimiter: 8 global OPS + 5 mods per order | — | Thread-safe |
| `execution/signal_hierarchy.py` | Priority resolution: RiskMgr(4) > HMM(3) > ML(2) > ADWIN(1) | — | Called synchronously |
| `risk/manager.py` | Kill switch, daily loss limit, drawdown, correlation guard | — | Called synchronously in execution thread |
| `risk/cost_model.py` | India costs: STT + exchange + GST + stamp duty + slippage | — | Pure computation |
| `alerts/alert_sink.py` | ZMQ PULL worker → Telegram (separate process) | PULL:5560 | Separate OS process |
| `alerts/telegram_bot.py` | Non-blocking ZMQ PUSH sender | PUSH:5560 | Fire-and-forget |
| `backtest/runner.py` | Walk-forward backtest with realistic cost model | — | Offline / research |

---

## 4. F&O Pipeline

F&O is **not active until Stage 4** (after the equity strategy is profitable for 3+ months
live). The F&O modules exist in `algo_trading/fo/` but are disabled at Stage 1.

When active (Stage 4+), the F&O pipeline differs from equity:

- **Separate orchestrator**: `fo/orchestrator.py` handles F&O session lifecycle
- **Multi-leg executor**: `fo/execution.py` handles spread/hedge leg atomicity
- **Margin service**: margin check is a hard precondition before any leg placement
- **Instrument master**: F&O symbols resolved via master file (not string suffixes)
- **Expiry calendar**: centralized Tuesday-expiry rules per NSE circular FAOP68747
- **Segment timing**: F&O close extended to 15:40 IST from August 3, 2026 [ASSUMED — secondary source only; verify with official NSE circular before coding production rules]

Known F&O issues that block live F&O trading (from audit):
- Multi-leg executor not wired into orchestrator path (P0 #4)
- `get_ltp` missing from `LiveSystemState` (P0 #5)
- Expiry rules inconsistent between modules (P0 #6)

---

## 5. Current Stage: Stage 1

**What is active in Stage 1:**
- EMA cross signal (EMA 9/21) on 15-min time bars
- Shoonya WebSocket (1–3 snapshots/second — throttled)
- Paper trading mode (`TRADING_MODE=paper`)
- Technical and statistical features (no Hawkes/TFI — requires Stage 2B+ with TrueData)
- ZMQ 3-process pipeline (plumbing complete)

**What must happen before Stage 2:**
- 30-day paper run with Sharpe >= 1.0, profit_factor >= 1.2
- Kill switch tested and working
- 5+ complete trade cycles without crash
- QuestDB storing candles with correct IST timestamps
- Telegram alerts firing on trade events and exceptions

**What Stage 2 adds:**
- TrueData feed (unthrottled L2 tick-by-tick) — required for volume bars
- Volume bars (replaces 15-min time bars)
- ML ensemble (XGBoost + LightGBM + CatBoost → Treelite .so)
- VPIN filter and Avellaneda-Stoikov reservation price

---

## 6. NOT in Scope (Yet)

The following exist in the codebase as stubs or Phase 4+ components.
Do NOT attempt to use them in Stage 1.

| Feature | Status | Notes |
|---------|--------|-------|
| Hawkes Process (`features/hawkes.py`) | Blocked | Requires sub-second tick data. `NotImplementedError` raised at Stage 1 |
| Trade Flow Imbalance / TFI (`features/tfi.py`) | Blocked | Same as Hawkes — tick-level data required |
| RL Position Sizer (`models/rl_sizer.py`) | Stage 4 only | `NotImplementedError` raised if `ALGO_STAGE < 4` |
| RL Trading Environment (`models/rl_env.py`) | Stage 4 only | `NotImplementedError` raised if `ALGO_STAGE < 4` |
| F&O orchestrator | Stage 4 only | Equity profitability required first |
| TabPFN | Banned permanently | O(N²) inference — cannot run in production at any stage |
| Sentiment / LLM features | NOT NOW list | Frozen until deterministic baseline is profitable |
| Online model / ADWIN (`models/online_model.py`) | Stage 3+ | Guardrails not yet implemented |

---

## 7. ZMQ Port Map

| Port | Direction | From → To | Message Format |
|------|-----------|-----------|----------------|
| 5555 | PUB/SUB | DataIngestor → FeatureComputer | msgpack-encoded bar |
| 5556 | PUB/SUB | FeatureComputer → MLInference | msgpack-encoded feature vector |
| 5557 | PUSH/PULL | MLInference → OrderExecutor | msgpack-encoded signal |
| 5558 | PUB/SUB | BarBuilder → FeatureEngine/SignalRouter | msgpack-encoded bar |
| 5560 | PUSH/PULL | Execution → AlertSink | JSON-encoded alert |

---

## 8. ASCII Architecture Diagram

```
+-------------------------------------------------------------+
|                    NSE AlgoTrader                            |
|                                                              |
|  [TrueData/Shoonya WS] --> [DataIngestor:PUB:5555]         |
|           |                                                  |
|  [QuestDB:9000] <-- candles -- [FeatureComputer:SUB/PUB:5556] |
|                                        |                     |
|                           [MLInference:SUB/PUSH:5557]       |
|                                        |                     |
|                              [RiskManager] <-- [KillSwitch] |
|                                        |                     |
|                           [OrderExecutor] --> [Broker REST] |
|                                        |                     |
|                    [WebSocket callbacks] --> [FSM]           |
|                                        |                     |
|                         [LiveSystemState] --> [QuestDB]     |
|                                        |                     |
|                           [AlertSink:PULL:5560] --> Telegram |
+-------------------------------------------------------------+
```

```
+----------------------------- Stage 1 Only -------------------+
|                                                              |
|  Shoonya WS (throttled 1-3/sec)                             |
|      --> 15-min time bars                                    |
|      --> EMA cross signal                                    |
|      --> Paper orders (simulated fills)                      |
|                                                              |
+-------------------------------------------------------------+
```

```
+----------------------- Stage 2+ (not active) ---------------+
|                                                              |
|  TrueData WS (unthrottled L2) <-- REQUIRED for:            |
|      --> Volume bars                                         |
|      --> VPIN filter                                         |
|      --> ML ensemble (XGB + LGB + CatBoost)                 |
|                                                              |
+-------------------------------------------------------------+
```
