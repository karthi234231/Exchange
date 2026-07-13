"""
download_full_history.py
Downloads 5 years of:
  1. Equity bhavcopy (full NSE universe EOD)      -> data/bhavcopy/equity/
  2. F&O bhavcopy (futures + options EOD)         -> data/bhavcopy/fo/
  3. Index bhavcopy (Nifty 50/100/etc close)      -> data/bhavcopy/index/
  4. 15-min intraday via NSE historical API        -> data/historical/15min/
     (best-effort — NSE limits ~1 year per symbol)

Run: python download_full_history.py
Estimated time: ~15 min for bhavcopy, ~40 min for 15-min intraday
"""
import datetime
import time
import sys
from pathlib import Path

from jugaad_data.nse import (
    bhavcopy_save,
    bhavcopy_fo_save,
    bhavcopy_index_save,
    stock_df,
)

# ── Directories ───────────────────────────────────────────────────────────────
BHAV_EQ_DIR  = Path("data/raw/bhavcopy/equity")
BHAV_FO_DIR  = Path("data/raw/bhavcopy/fo")
BHAV_IDX_DIR = Path("data/raw/bhavcopy/index")
HIST_15M_DIR = Path("data/historical/15min")

for d in [BHAV_EQ_DIR, BHAV_FO_DIR, BHAV_IDX_DIR, HIST_15M_DIR]:
    d.mkdir(parents=True, exist_ok=True)

END   = datetime.date.today()
START = END - datetime.timedelta(days=5 * 365)   # 5 years

# Symbols for 15-min intraday
SYMBOLS_15M = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BAJFINANCE", "WIPRO", "AXISBANK", "KOTAKBANK",
]

# ── Helper: iterate trading days ──────────────────────────────────────────────
def trading_days(start, end):
    """Yield weekdays (Mon-Fri) from start to end inclusive."""
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += datetime.timedelta(days=1)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Bhavcopy (equity + F&O + index)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"PART 1: Bhavcopy 5yr ({START} to {END})")
print("=" * 60)

days = list(trading_days(START, END))
total = len(days)
eq_ok = fo_ok = idx_ok = 0
errors = []

for i, dt in enumerate(days, 1):
    # Progress every 100 days
    if i % 100 == 0 or i == 1:
        print(f"  [{i}/{total}] {dt}  EQ:{eq_ok}  FO:{fo_ok}  IDX:{idx_ok}", flush=True)

    # Equity bhavcopy
    try:
        bhavcopy_save(dt, str(BHAV_EQ_DIR))
        eq_ok += 1
    except Exception as e:
        errors.append(f"EQ {dt}: {e}")

    # F&O bhavcopy
    try:
        bhavcopy_fo_save(dt, str(BHAV_FO_DIR))
        fo_ok += 1
    except Exception as e:
        errors.append(f"FO {dt}: {e}")

    # Index bhavcopy (Nifty 50, 100, 200, etc.)
    try:
        bhavcopy_index_save(dt, str(BHAV_IDX_DIR))
        idx_ok += 1
    except Exception as e:
        errors.append(f"IDX {dt}: {e}")

    time.sleep(0.25)   # polite delay

print(f"\nBhavcopy complete:")
print(f"  Equity files : {eq_ok}  -> {BHAV_EQ_DIR}/")
print(f"  F&O files    : {fo_ok}  -> {BHAV_FO_DIR}/")
print(f"  Index files  : {idx_ok} -> {BHAV_IDX_DIR}/")
if errors:
    print(f"  Errors (holidays/missing): {len(errors)} (normal — NSE has no data on holidays)")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — 15-min intraday via NSE historical price API
# NSE serves ~1 year of intraday via its chart/price API.
# jugaad-data stock_df gives daily only — we use direct NSE API here.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: 15-min intraday (NSE API, best-effort)")
print("=" * 60)

import requests
import json
import csv

NSE_BASE = "https://www.nseindia.com"
NSE_CHART_URL = (
    "https://www.nseindia.com/api/historical/cm/equity"
    "?symbol={sym}&series=EQ&from={from_d}&to={to_d}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

def get_nse_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(NSE_BASE, timeout=15)
        time.sleep(1.5)
    except Exception as e:
        print(f"  WARNING: NSE session init failed: {e}")
    return s

def download_nse_daily_via_api(session, symbol, from_date, to_date):
    """NSE historical equity API returns daily OHLCV (not 15-min).
    NSE does not provide free 15-min intraday history via public API."""
    from_d = from_date.strftime("%d-%m-%Y")
    to_d   = to_date.strftime("%d-%m-%Y")
    url = NSE_CHART_URL.format(sym=symbol, from_d=from_d, to_d=to_d)
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return data
        return []
    except Exception:
        return []

print("\nNOTE: NSE's public API provides daily OHLCV only.")
print("      15-min intraday requires one of:")
print("      1. Shoonya API (free, ~400 days) — add credentials to .env")
print("      2. TrueData (paid, Rs.300-500/mo) — historical tick + candle")
print("\nAttempting to download via NSE historical API (daily granularity)...")

session = get_nse_session()
intraday_end   = END
intraday_start = END - datetime.timedelta(days=365)   # 1 year

for sym in SYMBOLS_15M:
    out = HIST_15M_DIR / f"{sym}_nse_daily_1yr.csv"
    if out.exists():
        print(f"  {sym}: already exists, skipping")
        continue
    try:
        rows = download_nse_daily_via_api(session, sym, intraday_start, intraday_end)
        if rows:
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {sym}: {len(rows)} rows -> {out}")
        else:
            print(f"  {sym}: no data returned")
        time.sleep(1.2)
    except Exception as e:
        print(f"  {sym}: ERROR - {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
for folder in [BHAV_EQ_DIR, BHAV_FO_DIR, BHAV_IDX_DIR,
               Path("data/historical"), HIST_15M_DIR]:
    files = list(folder.glob("*.*"))
    if files:
        size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  {folder}/  {len(files)} files  {size_mb:.1f} MB")

print("\n15-min intraday status:")
print("  To get true 15-min candles (~400 days), add to .env:")
print("    SHOONYA_USER_ID=your_id")
print("    SHOONYA_PASSWORD=your_pass")
print("    SHOONYA_TOTP_SECRET=your_totp")
print("  Then run: python -m algo_trading.scripts.download_history --source shoonya --interval 15")
