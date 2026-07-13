# Data Collection Guide

Everything you need to know about collecting data for backtesting and model training.

---

## Overview

| Source | Data Type | Cost | Who Handles It |
|---|---|---|---|
| yFinance | Historical OHLCV (daily + 15-min) | Free | Claude runs it |
| Jugaad Data | NSE historical bhavcopy (EOD, all stocks) | Free | Claude runs it |
| NSE India (nsepython) | EOD data, F&O chain, index data | Free | Claude runs it |
| AkShare | Multi-market historical data | Free | Claude runs it |
| TrueData | Tick-by-tick L2 unthrottled feed | Rs. 300–500/month | You subscribe, give credentials |
| Shoonya WebSocket | Live ticks + historical candles via API | Free (with account) | You give API credentials |

---

## What Data Is Actually Needed

### Phase 1–2: Foundation + ML Model Training
- **15-min OHLCV candles** — minimum 3 years, for 5+ Nifty 50 stocks
- **Daily OHLCV** — 5+ years for regime detection and long-range features
- **NSE bhavcopy (EOD)** — full market daily snapshot for universe filtering

> **Data availability reality check:**
> - yFinance provides only ~60 days of 15-min intraday history — NOT 3 years.
> - Jugaad Data provides EOD bhavcopy only — NO intraday candles.
> - Shoonya historical API provides ~400 days of 15-min candles — still short of 3 years.
> - **Free data is sufficient for pipeline scaffolding and smoke testing only.**
> - **For the Stage 0 edge-validation gate (3-year walk-forward), paid intraday data is required.** Options: TrueData historical, broker historical API if verified, or record live data for 12+ months before running serious ML.
>
> Two-tier data plan:
> - **Tier 1 (free, scaffold):** yFinance daily + 60-day 15-min, Jugaad EOD — enough to build and test the code pipeline.
> - **Tier 2 (paid, ML validation):** TrueData or equivalent 2-3 year intraday history — required before reporting any Sharpe number as meaningful.

### Phase 3–4: Microstructure Features (Hawkes, VPIN, TFI)
- **Tick-by-tick L2 data** — unthrottled, multi-level order book depth
- **Only TrueData provides this** for Indian markets at a reasonable cost

> yFinance and Jugaad cannot provide tick data. TrueData is mandatory here.

---

## Free Data Sources — Claude Handles These

### 1. yFinance
**What:** Historical OHLCV for NSE stocks (daily + intraday 15-min up to ~60 days back)

**Used for:**
- Daily data going back 5+ years — ideal for backtesting EMA signals
- 15-min intraday for recent 60 days (limited, but useful for validation)

**Limitation:** Only ~60 days of 15-min intraday history. For 3 years of 15-min data, combine with Jugaad or TrueData historical.

**Claude will run:**
```python
import yfinance as yf

symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
           "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS"]

for sym in symbols:
    df = yf.download(sym, period="5y", interval="1d")       # 5 years daily
    df.to_csv(f"data/historical/{sym}_daily.csv")

    df_15m = yf.download(sym, period="60d", interval="15m") # 60 days 15-min
    df_15m.to_csv(f"data/historical/{sym}_15min.csv")
```

---

### 2. Jugaad Data
**What:** NSE/BSE historical bhavcopy — complete EOD data for all listed stocks

**Used for:**
- Full market EOD snapshots going back years
- Universe filtering (which stocks to trade on a given day)
- Sector/index membership data

**Limitation:** EOD only. No intraday candles.

**Claude will run:**
```python
from jugaad_data.nse import bhavcopy_save, bhavcopy_fo_save
import datetime

# Download NSE equity bhavcopy for last 3 years
start = datetime.date(2021, 1, 1)
end   = datetime.date.today()

bhavcopy_save(start, end, "data/bhavcopy/equity/")
bhavcopy_fo_save(start, end, "data/bhavcopy/fo/")  # F&O bhavcopy
```

---

### 3. NSEPython (NSE India)
**What:** Live and historical NSE data — index data, F&O chain, option chain, VIX

**Used for:**
- India VIX historical data (needed for volatility regime filter)
- Nifty 50 constituent list (which stocks are in index)
- F&O expiry calendar

**Limitation:** Scrapes NSE website — can be rate-limited. Not for bulk downloads.

**Claude will run:**
```python
from nsepython import *

# Nifty 50 list
nifty50 = nse_get_index_list()

# India VIX historical
vix = nse_get_index_quote("INDIA VIX")

# F&O expiry dates
expiry = nse_get_expiry_dates()
```

---

### 4. AkShare
**What:** Multi-market data aggregator — includes Indian market data

**Used for:**
- Backup/cross-validation source when yFinance has gaps
- Macro data (FII/DII flows, index futures data)

**Claude will run:**
```python
import akshare as ak

# NSE stock historical
df = ak.stock_in_se_daily(symbol="RELIANCE", start_date="20210101", end_date="20241231")
df.to_csv("data/historical/RELIANCE_akshare.csv")
```

---

## Paid Data Source — You Handle Subscription

### TrueData
**What:** Professional-grade tick-by-tick L2 data, unthrottled WebSocket feed

**Why it's mandatory for Stage 2+:**
The microstructure features (Hawkes Process, VPIN, Trade Flow Imbalance) require:
- Tick arrival timestamps accurate to milliseconds
- Multi-level order book (bid/ask at each level)
- Unthrottled feed (not snapped at 1–3 sec like Shoonya free feed)

Running these features on a snapped/throttled feed produces **wrong signals** — the inter-arrival time distribution is corrupted, VPIN buckets accumulate incorrectly, and TFI fires on phantom order flow.

**Cost:** Rs. 300–500/month (NSE equities segment)

**How to subscribe:**
1. Go to [truedata.in](https://truedata.in)
2. Register for a live data plan (NSE equities)
3. Get your `username` and `password`

**What you give Claude:**
```
TRUEDATA_USER=your_username
TRUEDATA_PASSWORD=your_password
```

Claude will wire these into the `.env` file and `data/truedata_feed.py` — no manual setup needed from you.

**What TrueData also provides for backtesting:**
- Historical tick data (purchasable separately — ask their support for pricing)
- Clean historical 1-min/5-min/15-min candles going back years (cleaner than yFinance)

---

## Shoonya API (Historical Candles)
**What:** Shoonya broker API provides historical candles for free (with account)

**Used for:**
- Backfilling 15-min candles for the past 400 days (API limit)
- Cross-validating yFinance data quality

**What you give Claude:** Shoonya API credentials (you already have the account)

**Claude will run:**
```python
from api_helper import ShoonyaApiPy

api = ShoonyaApiPy()
api.login(userid=USER, password=PASS, twoFA=TOTP, vendor_code=VC,
          api_secret=SECRET, imei=IMEI)

# Pull 15-min candles — up to 400 days back
ret = api.get_time_price_series(
    exchange='NSE', token='22', starttime='01-01-2024 00:00:00',
    endtime='01-01-2025 00:00:00', interval=15
)
```

---

## Data Storage Structure

Once Claude downloads everything, it will be organized as:

```
EXCHANGE/
└── data/
    ├── historical/
    │   ├── RELIANCE.NS_daily.csv
    │   ├── RELIANCE.NS_15min.csv
    │   ├── TCS.NS_daily.csv
    │   └── ... (all Nifty 50 stocks)
    ├── bhavcopy/
    │   ├── equity/       ← Jugaad EOD bhavcopy
    │   └── fo/           ← F&O bhavcopy
    ├── vix/
    │   └── india_vix_historical.csv
    └── live/             ← QuestDB stores live ticks here (Stage 2+)
```

---

## Download Sequence (What Claude Will Do, In Order)

| Step | Action | Source | Time Estimate |
|---|---|---|---|
| 1 | Install required libraries | pip | 2 min |
| 2 | Download 5yr daily OHLCV — Nifty 50 stocks | yFinance | 5 min |
| 3 | Download 60-day 15-min OHLCV — Nifty 50 stocks | yFinance | 5 min |
| 4 | Download 3yr EOD bhavcopy — all NSE stocks | Jugaad Data | 10–20 min |
| 5 | Download India VIX + Nifty 50 constituent list | NSEPython | 2 min |
| 6 | Cross-validate data gaps between sources | Claude script | 5 min |
| 7 | Wire TrueData credentials (when you provide them) | You → Claude | 1 min |
| 8 | Wire Shoonya historical pull (when you provide creds) | You → Claude | 1 min |

**Total for free data:** ~30–40 minutes, zero effort from you.

---

## What You Need to Do

1. **Confirm Python is installed** (Claude will check)
2. **Tell Claude where to save data** (default: `EXCHANGE/data/`)
3. **Subscribe to TrueData** when ready for Stage 2 — share credentials via `.env`
4. **Share Shoonya API credentials** for historical candle backfill

Everything else — scripts, downloads, folder creation, data validation — Claude handles.

---

## Data Quality Check (Claude Runs This After Download)

After all downloads complete, Claude will run a validation script that checks:

- No missing dates (trading day gaps)
- No price outliers (>5% single-bar moves flagged for review)
- OHLC consistency (High >= Low, Close within High/Low range)
- Volume > 0 on all bars
- Overlap/consistency between yFinance and Jugaad for same stocks

Any issues found are reported before model training begins.
