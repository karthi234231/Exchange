# 📈 NSE AlgoTrader

**Production-Grade Algorithmic Trading System for Indian Equity Markets**

---

## 📋 Overview

NSE AlgoTrader is a production-targeted algorithmic trading system for the Indian National Stock Exchange (NSE) combining ML ensemble signals, real-time data processing, risk management, and SEBI-compliant execution.

### Key Highlights

- **ML Ensemble**: XGBoost + LightGBM + CatBoost → LogisticRegression meta
- **Real-time Pipeline**: 3-process ZeroMQ architecture
- **Risk Management**: Kill switches, daily loss limits, drawdown protection
- **SEBI Compliance**: Strategy ID tagging, SHAP explainability, 5-year audit trail
- **Zero-Brokerage**: Shoonya/Finvasia primary broker

### Current Status: Stage 1 — Paper Trading Ready

> ⚠️ **Live Trading is DISABLED by default.** All 13+ safety gates must pass before live mode can be enabled.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12+ works (pandas-ta beta) |
| Docker | Latest | Required for QuestDB |
| Shoonya Account | Active | Required for WebSocket data |

### Installation

```bash
# Clone and setup
git clone https://github.com/karthi234231/Exchange.git
cd Exchange

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Start QuestDB

```bash
docker run -d \
  --name questdb \
  -p 9000:9000 \
  -p 8812:8812 \
  -v questdb_data:/var/lib/questdb \
  questdb/questdb:latest
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your Shoonya credentials
# TRADING_MODE=paper
# ENABLE_LIVE_TRADING=false
```

### Run Commands

```bash
# Validate EMA cross edge (Stage 0 gate)
python -m algo_trading.scripts.validate_edge --symbols RELIANCE

# Run paper trading (Stage 1)
python -m algo_trading.scripts.paper_trade --symbols RELIANCE --capital 500000

# Run backtest
python -m algo_trading.scripts.run_backtest --symbol RELIANCE

# Train ML models (Stage 2A)
python -m algo_trading.scripts.train_models --symbols RELIANCE --timeframe 15min

# Run full system
python -m algo_trading.main --mode paper

# Health check
python -m algo_trading.app.health_check --once
```

---

## 🏗️ Architecture

### 3-Process ZeroMQ Pipeline

```
[DataIngestor] → ZMQ PUB:5555 → [FeatureComputer] → ZMQ PUB:5556 → [MLInference]
      ↓                                           ↓                     ↓
Shoonya WS                              23 Features Engine        Treelite .so
(1-3 ticks/sec)                         (Technical + Statistical)  (~2ms inference)
                                                                        ↓
                                                              [OrderExecutor]
                                                              → Shoonya API
```

### Process Details

| Process | Port | Role |
|---------|------|------|
| DataIngestor | PUB:5555 | Shoonya WebSocket → bar builder |
| FeatureComputer | SUB:5555 → PUB:5556 | 25 features → ZMQ |
| MLInference | SUB:5556 → PUSH:5557 | Treelite .so inference |
| OrderExecutor | PULL:5557 | FSM + rate limiter + broker calls |
| AlertSink | PULL:5560 | Telegram delivery (decoupled) |

---

## 📊 Features

### ML Pipeline

- **Triple-Barrier Labeling**: López de Prado method, symmetric ±1.5 ATR
- **Stacking Ensemble**: XGBoost + LightGBM + CatBoost → LR meta-learner
- **Treelite Compilation**: ~2ms inference vs ~40ms native
- **SHAP Explainability**: SEBI compliance requirement
- **Optuna Hyperparameter Tuning**: Sunday only, never during market hours

### Feature Engineering (23+ Features)

| Category | Features |
|----------|----------|
| Technical | RSI, MACD, BB, ATR, ADX, Stoch, Supertrend, EMA cross, Ichimoku |
| Statistical | HMM regime, GARCH vol, Kalman filter, autocorrelation |
| Custom | VWAP deviation, OBI, Volume Clock, time-of-day encoding, relative strength |
| Microstructure (Stage 2B+) | VPIN, Avellaneda-Stoikov reservation price |

### Risk Management (Non-Negotiable)

| Rule | Threshold | Action |
|------|-----------|--------|
| Daily Loss Limit | 2% of capital | Kill switch → halt all trading |
| Max Drawdown | 10% from peak | Pause system, require manual reset |
| Single Position | 5% of capital | Hard cap, never exceeded |
| Gross Exposure | 20% of capital | Portfolio-level cap |
| Price Drift | 30bps from signal | Abort entry, never chase |
| Rate Limit | 8 orders/sec | SEBI limit (10 OPS) with 2-order buffer |
| Order Modifications | 5 per order | Cancel + resubmit, no infinite loops |
| Force Close | 15:15 IST | All positions closed, no overnight |
| Circuit Breaker | 20% price velocity | Halt symbol immediately |

### Compliance

- **SEBI Strategy ID**: Tagged on every order via `remarks` field
- **SHAP Explainability Report**: Generated after every training run
- **5-Year Audit Trail**: QuestDB + immutable audit logs
- **Order FSM**: Explicit state machine prevents zombie orders
- **Kill Switch**: Tested and verified in paper mode
- **Static IP Registration**: VPS deployment guide included

---

## 📂 Project Structure

```
Exchange/
├── algo_trading/              # Main package (164+ Python files)
│   ├── alerts/                # Telegram alert system
│   ├── app/                   # Daily orchestrator, health checks
│   ├── backtest/              # Backtesting engine
│   ├── core/                  # Calendar, logging
│   ├── execution/             # Order FSM, rate limiter, execution
│   ├── features/              # Technical, statistical, microstructure
│   ├── fo/                    # F&O strategies (Stage 4)
│   ├── labeling/              # Triple-barrier, walk-forward
│   ├── market_data/           # Broker adapters, data ingestion
│   ├── models/                # ML ensemble, inference, RL
│   ├── monitoring/            # Reconciliation, alert deduplication
│   ├── news/                  # News sentiment pipeline
│   ├── news_pipeline/         # Full Prefect-based news pipeline
│   ├── research/              # Research tools
│   ├── risk/                  # Risk manager, cost model
│   ├── scripts/               # Utility scripts
│   └── signals/               # EMA cross, mean reversion, etc.
├── config/                    # nse_holidays.yaml, tax_rates.yaml
├── docs/                      # Documentation
├── models/trained/            # Trained model artifacts
│   ├── 15min/                 # 15-min models (XGB, LGB, MLP)
│   ├── frozen_model_spec.json # Model version/feature schema
│   └── sebi_report.json       # SHAP explainability report
├── results/                   # Backtest reports, validation results
├── tests/                     # 60+ test files
├── arch-spec.json             # Machine-readable architecture spec
├── docker-compose.yml         # Docker setup
├── Makefile                   # Common commands
├── pyproject.toml             # Package metadata
└── requirements.txt           # Dependencies
```

---

## 🎯 Stage-Gated Deployment

| Stage | Description | Gate |
|-------|-------------|------|
| **Stage 0** | Prove edge exists | EMA cross Sharpe ≥ 1.0, PF ≥ 1.2, WR ≥ 40% |
| **Stage 1** | Paper trading (no ML) | 30-day run, Sharpe ≥ 1.0, kill switch tested |
| **Stage 2A** | ML ensemble | 20 days zero crashes, AUC > 0.52, DD < 10% |
| **Stage 2B** | Microstructure features | Requires TrueData L2 feed (₹300-500/month) |
| **Stage 3** | Hawkes/TFI | Requires sub-minute data (TrueData unthrottled) |
| **Stage 4** | F&O strategies | 3+ profitable equity months, capital ≥ ₹15L |

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| Language | Python | 3.11+ |
| Data Storage | QuestDB | 7.x |
| Data Processing | Polars | 0.20+ |
| ML Models | XGBoost, LightGBM, CatBoost | 2.0+, 4.0+, 1.2+ |
| Inference | Treelite | Latest |
| IPC | ZeroMQ (pyzmq) | 26.0+ |
| Hyperparameter Tuning | Optuna | 3.6+ |
| Experiment Tracking | MLflow | 2.9+ |
| Explainability | SHAP | 0.41+ |
| Regime Detection | hmmlearn, arch, filterpy | Latest |
| Technical Indicators | pandas-ta | 0.3+ |
| Broker | Shoonya (Finvasia) | NorenAPI |
| Testing | pytest | 7.4+ |

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| CLAUDE.md | The definitive build guide |
| DECISIONS.md | Single source of truth for all active decisions |
| ARCHITECTURE_OVERVIEW.md | 3-process pipeline, stage definitions |
| RISK_REGISTER.md | All open risks (P0 items block live trading) |
| WHAT_ACTUALLY_OUTPERFORMS.md | Research synthesis: live alpha vs hype |
| LOCAL_DEV_QUICKSTART.md | Step-by-step local setup guide |
| VPS_DEPLOYMENT.md | Production deployment guide |
| RUNBOOK.md | Emergency procedures and daily checklists |
| INCIDENT_RUNBOOK.md | Specific incident responses |
| ROLLBACK_PROCEDURE.md | How to revert to a known-good version |
| RELEASE_CHECKLIST.md | 8 gates that must pass before any release |
| compliance_matrix.md | SEBI + Shoonya compliance requirements |
| RETENTION_POLICY.md | 5-year audit trail retention |
| BACKUP_POLICY.md | What to back up, when, where, and how |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ -v --cov=algo_trading

# Run in parallel
pytest tests/ -v -n auto
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

### Code Style

```bash
# Run code quality checks
make quality

# Or individually:
ruff check algo_trading/ tests/
mypy algo_trading/
```

---

## ⚠️ Important Warnings

### Live Trading is FORBIDDEN by Default

Live trading is disabled until ALL of these pass:

- [ ] 3 calendar months paper trading completed
- [ ] Paper Sharpe ≥ 1.0 after full India cost model
- [ ] Zero unreconciled broker state events
- [ ] Zero duplicate orders in paper log
- [ ] Kill switch tested manually
- [ ] Manual close runbook documented and tested
- [ ] Startup reconciler tested after simulated restart
- [ ] Order FSM tested for every state transition
- [ ] SEBI Strategy ID registered and configured
- [ ] Static IP registered if required by broker
- [ ] `ENABLE_LIVE_TRADING=true` set explicitly in .env

### Risk Disclaimers

- This is **educational/research software**
- Past performance does not guarantee future results
- You are **fully responsible** for your own trading decisions
- No warranty or guarantee of profitability is provided
- **Never trade with money you cannot afford to lose**

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚡ Quick Commands Reference

| Action | Command |
|--------|---------|
| Install | `pip install -r requirements.txt` |
| Run tests | `pytest tests/ -v` |
| Validate edge | `python -m algo_trading.scripts.validate_edge --symbols RELIANCE` |
| Paper trade | `python -m algo_trading.scripts.paper_trade --symbols RELIANCE --capital 500000` |
| Backtest | `python -m algo_trading.scripts.run_backtest --symbol RELIANCE` |
| Train models | `python -m algo_trading.scripts.train_models --symbols RELIANCE --timeframe 15min` |
| Run system | `python -m algo_trading.main --mode paper` |
| Health check | `python -m algo_trading.app.health_check --once` |
| Start QuestDB | `docker run -d -p 9000:9000 -p 8812:8812 --name questdb questdb/questdb:latest` |

---

<div align="center">

**Built with ❤️ for the Indian Markets**

[Report Bug](https://github.com/karthi234231/Exchange/issues) · [Request Feature](https://github.com/karthi234231/Exchange/issues)

</div>
