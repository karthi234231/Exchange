# Local Development Quickstart — NSE AlgoTrader

Step-by-step guide to get a local development environment running in paper trading mode.
This is for development and testing only. Do NOT use this guide to set up a live trading
system — see `MD_Files/VPS_DEPLOYMENT.md` for production deployment.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | **3.11** (required) | Python 3.12 may work. Python 3.13 is NOT supported — several ML packages are incompatible. Use `python --version` to check. |
| Docker | Latest stable | Required for QuestDB. Docker Desktop on Windows works fine. |
| git | Any recent version | — |
| Windows | Windows 10/11 | Linux/macOS also work with minor path differences |

**Python version note:** The system is developed and tested on Python 3.11. The Docker
image targets Python 3.11. If your system Python is 3.13, install 3.11 using
[pyenv](https://github.com/pyenv-win/pyenv-win) (Windows) or your OS package manager.

---

## Step 1 — Clone and Enter the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EXCHANGE.git
cd EXCHANGE
```

---

## Step 2 — Create Virtual Environment

**Windows (PowerShell or Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash / bash
# OR:
.venv\Scripts\Activate.ps1       # PowerShell
```

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Confirm the correct Python is active:
```bash
python --version   # Must show 3.11.x
which python       # Must point to .venv
```

---

## Step 3 — Install Dependencies

For paper trading development (no live broker required):
```bash
pip install -e ".[paper]"
```

For the full ML stack (required for model training):
```bash
pip install -e ".[paper,ml]"
```

For F&O development (Stage 4+, not needed for Stage 1):
```bash
pip install -e ".[paper,ml,fo]"
```

The Shoonya API package must be installed separately (not on PyPI):
```bash
pip install "git+https://github.com/Shoonya-Dev/ShoonyaApi-py.git#egg=NorenRestApiPy"
```

---

## Step 4 — Start QuestDB

QuestDB stores candles, features, and trades. It runs in Docker.

```bash
docker run -d \
  --name questdb \
  -p 9000:9000 \
  -p 8812:8812 \
  -v questdb_data:/var/lib/questdb \
  questdb/questdb:latest
```

Verify QuestDB is running:
```bash
curl -s http://localhost:9000/status
# Should return: {"status":"OK"}
```

QuestDB web console: http://localhost:9000

To stop QuestDB:
```bash
docker stop questdb
```

To start it again later:
```bash
docker start questdb
```

---

## Step 5 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
TRADING_MODE=paper
ENABLE_LIVE_TRADING=false
CAPITAL_INR=500000

# Shoonya credentials (needed even for paper mode to connect WebSocket)
SHOONYA_USER=YOUR_USER_ID
SHOONYA_PASSWORD=YOUR_PASSWORD
SHOONYA_TOTP_SECRET=YOUR_TOTP_SECRET
SHOONYA_VENDOR_CODE=YOUR_VENDOR_CODE
SHOONYA_API_SECRET=YOUR_API_SECRET
SHOONYA_IMEI=YOUR_IMEI

# Telegram (optional but recommended for alerts)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID

# QuestDB (default Docker settings)
QUESTDB_HOST=localhost
QUESTDB_ILP_PORT=9000
QUESTDB_PG_PORT=8812
QUESTDB_PG_USER=admin
QUESTDB_PG_PASSWORD=quest
```

For a safer development config, copy the pre-made safe paper config:
```bash
cp docs/safe_paper_config.env .env
# Then fill in your Shoonya credentials
```

**Security note on Windows:** If this repository is inside OneDrive, be aware that
OneDrive may sync `.env` to the cloud. Move the repository outside OneDrive if possible,
or add `.env` to OneDrive exclusions. The `.gitignore` prevents committing `.env`.

---

## Step 6 — Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run fast unit tests only
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ -v --cov=algo_trading --cov-report=term-missing
```

Run code quality checks (requires `[paper]` or `[ml]` install):
```bash
make quality
# OR individually:
ruff check algo_trading/ tests/
mypy algo_trading/
```

Expected result: all tests pass, no ruff errors, mypy clean.

---

## Step 7 — Start Paper Trading

```bash
python -m algo_trading.orchestrator.daily_orchestrator --mode paper
```

Or with the alternative entry point:
```bash
python -m algo_trading.scripts.paper_trade
```

Watch live logs:
```bash
# In a separate terminal
tail -f logs/algo_trading.jsonl | python -m json.tool
```

The system will:
1. Login to Shoonya (even in paper mode, a real account is needed for WebSocket data)
2. Subscribe to symbols from `SYMBOLS` config
3. Build 15-minute bars
4. Compute features
5. Generate EMA cross signals
6. Simulate paper orders (no real orders placed)
7. Log everything to `logs/` and `state/`

---

## Step 8 — Validate Edge Before Model Training

Before training any ML models, validate that the simple EMA cross strategy has edge:

```bash
# Download historical data first (requires Shoonya credentials)
python -m algo_trading.scripts.download_history \
    --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK \
    --interval 15m \
    --years 3

# Run edge validation
python -m algo_trading.scripts.validate_edge \
    --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK
```

Expected output: Sharpe >= 1.0, profit_factor >= 1.2, win_rate >= 40%.
If any gate fails, do NOT proceed to ML training.

---

## Gotchas

### Windows Path Separators

The codebase uses `pathlib.Path` throughout, which handles Windows/Linux path separators
automatically. If you encounter `FileNotFoundError`, check that the path starts from the
repository root, not from a hardcoded Unix path.

### Python 3.13 Not Supported

`hmmlearn`, `arch`, and some ML packages do not have stable Python 3.13 builds as of
mid-2026. If you see `ImportError` for these packages, downgrade to Python 3.11.

### Shoonya Account Required Even for Paper Mode

Shoonya WebSocket requires a valid logged-in session to receive tick data, even in paper
mode. You cannot run the full system without a Shoonya account. For pure unit testing
(no live data), all tests use mocks and do not require credentials.

### QuestDB Default Password

The default QuestDB password is `"quest"`. This is fine for local development. For any
deployment that has network access, change it and update `QUESTDB_PG_PASSWORD` in `.env`.

### Windows Firewall and ZMQ

ZeroMQ ports (5555–5560) may be blocked by Windows Defender Firewall. If you see ZMQ
connection errors, add firewall exceptions for localhost traffic on these ports, or
temporarily disable the firewall for local testing.

### OneDrive Sync Interference

OneDrive can lock files while syncing, causing `PermissionError` on log files or QuestDB
data files. Move the repository outside OneDrive sync scope if you encounter this issue.

---

## Directory Layout After Setup

```
EXCHANGE/
  .env                     # Your local config (never commit)
  .venv/                   # Python virtual environment
  algo_trading/            # Main package
  data/                    # Downloaded historical data (gitignored)
  logs/                    # Runtime logs (gitignored)
  models/trained/          # Model artifacts (gitignored until trained)
  state/                   # Live state, audit log, config snapshot
  tests/                   # pytest test suite
  docs/                    # This documentation
```
