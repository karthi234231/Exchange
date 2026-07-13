# NSE AlgoTrader

> CPU-Optimized Algorithmic Trading Framework for NSE India  
> Python • Shoonya/Finvasia • QuestDB • ZeroMQ • Treelite

⚠️ Current Status: **Paper Trading**

---

## Architecture

```text
Shoonya WebSocket
        │
        ▼
Market Data Engine
        │
        ▼
Feature Engine
(25+ Features)
        │
        ▼
ML Inference Engine
(Treelite)
        │
        ▼
Risk Manager
        │
        ▼
Order Executor
        │
        ▼
Shoonya API
        │
        ▼
Telegram Alerts
```

---

## Core Stack

```text
Broker        : Shoonya / Finvasia
Language      : Python 3.11+
Database      : QuestDB
Messaging     : ZeroMQ
ML Models     : XGBoost / LightGBM / CatBoost
Inference     : Treelite
Deployment    : Docker + Systemd
Monitoring    : Structlog + Telegram
```

---

## Methods Used

### Market Data

- Shoonya WebSocket Streaming
- OHLC Aggregation
- Time-Based Bar Construction
- Tick Processing Pipeline

### Feature Engineering

#### Trend

- EMA
- SMA
- EMA Crossovers

#### Momentum

- RSI
- ROC
- Momentum Score

#### Volatility

- ATR
- Rolling Volatility
- Range Expansion

#### Statistical

- Z-Score
- Mean Reversion Signals
- Return Distribution Metrics

#### Volume

- Relative Volume
- Volume Spike Detection
- Participation Metrics

---

## Machine Learning

### Supported Models

```text
XGBoost
LightGBM
CatBoost
```

### Inference Optimization

```text
Native Inference    ~40ms
Treelite            ~2ms
```

---

## Risk Engine

```text
Max Daily Loss      : 2%
Max Position Size   : 5%
Max Exposure        : 20%
Max Drawdown        : 10%
```

Rules:

- No entries after 15:15 IST
- Auto square-off protection
- Limit entries only
- Market stop-loss execution
- Kill Switch Protection
- Exposure Checks
- Correlation Controls

---

## Project Structure

```text
algo_trading/

├── alerts/
├── execution/
├── features/
├── models/
├── orchestrator/
├── risk/
├── scripts/
├── storage/

├── config.py
└── main.py
```

---

## Quick Start

```bash
git clone https://github.com/KarthikJogii/MyProj.git

cd MyProj

cp .env.example .env

pip install -r requirements.txt

pip install \
"git+https://github.com/Shoonya-Dev/ShoonyaApi-py.git#egg=NorenRestApiPy"

docker compose up -d questdb

python -m algo_trading.orchestrator.daily_orchestrator --mode paper
```

---

## Validation Before Live Trading

- WebSocket reconnect test
- Order placement test
- Order cancellation test
- Kill switch verification
- QuestDB data validation
- Telegram alert validation
- Walk-forward Sharpe ≥ 1.0
- All broker compliance checks completed

---

## Roadmap

```text
Stage 1 ✅ Paper Trading
Stage 2 ⏳ Advanced Features
Stage 3 ⏳ ML Ensemble
Stage 4 ⏳ Position Sizing Models
Stage 5 ⏳ Live Deployment
```

---

## Author

**Karthik Jogi**

GitHub: https://github.com/KarthikJogii

---
⭐ Star the repository if you find it useful.
