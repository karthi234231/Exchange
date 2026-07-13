"""
download_nse_data.py
Downloads 5-year daily OHLCV for top NSE stocks using jugaad-data (NSE direct).
Also downloads NSE bhavcopy (full market EOD snapshot).
Run: python download_nse_data.py
"""
import datetime
import time
from pathlib import Path

from jugaad_data.nse import stock_df, bhavcopy_save

OUTPUT_DIR = Path("data/historical")
BHAV_DIR   = Path("data/raw/bhavcopy/equity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BHAV_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "HINDUNILVR", "BAJFINANCE", "WIPRO", "AXISBANK",
    "KOTAKBANK", "LT", "MARUTI", "ASIANPAINT", "TITAN",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "ADANIPORTS", "ONGC",
    "NIFTY_50",  # special — handled separately
]

# 5 years back from today
END   = datetime.date.today()
START = END - datetime.timedelta(days=5 * 365)

print(f"Downloading daily data: {START} to {END}")
print(f"Output: {OUTPUT_DIR}/\n")

errors = []

for sym in SYMBOLS:
    if sym == "NIFTY_50":
        # Nifty 50 index via jugaad-data index_df
        try:
            from jugaad_data.nse import index_df
            df = index_df("NIFTY 50", START, END)
            out = OUTPUT_DIR / "NIFTY50_1d.csv"
            df.to_csv(out, index=False)
            print(f"  NIFTY50: {len(df)} rows -> {out}")
        except Exception as e:
            # NSE API format changed for index — download Nifty ETF (NIFTYBEES) as proxy
            try:
                df = stock_df(symbol="NIFTYBEES", from_date=START, to_date=END, series="EQ")
                out = OUTPUT_DIR / "NIFTY50_proxy_1d.csv"
                df.to_csv(out, index=False)
                print(f"  NIFTY50 (via NIFTYBEES ETF proxy): {len(df)} rows -> {out}")
            except Exception as e2:
                print(f"  NIFTY50: SKIP - {e2}")
        continue

    try:
        df = stock_df(
            symbol=sym,
            from_date=START,
            to_date=END,
            series="EQ",
        )
        if df is None or len(df) == 0:
            print(f"  {sym}: empty")
            errors.append(sym)
            continue

        out = OUTPUT_DIR / f"{sym}_1d.csv"
        df.to_csv(out, index=False)
        print(f"  {sym}: {len(df)} rows -> {out}")
        time.sleep(0.8)   # be polite to NSE servers

    except Exception as e:
        print(f"  {sym}: ERROR - {e}")
        errors.append(sym)
        time.sleep(1)

# ── Bhavcopy (full market EOD snapshot) ──────────────────────────────────────
# bhavcopy_save takes ONE date at a time — loop through last 30 trading days
print(f"\nDownloading bhavcopy (last 30 days)...")
bhav_start = END - datetime.timedelta(days=45)  # extra days to cover weekends/holidays
saved_count = 0
cursor = bhav_start
while cursor <= END:
    if cursor.weekday() < 5:  # Mon-Fri only
        try:
            bhavcopy_save(cursor, str(BHAV_DIR))
            saved_count += 1
        except Exception:
            pass  # holidays silently skip
    cursor += datetime.timedelta(days=1)
    time.sleep(0.3)
print(f"  Bhavcopy: {saved_count} files -> {BHAV_DIR}/")

# ── Summary ───────────────────────────────────────────────────────────────────
all_files = list(OUTPUT_DIR.glob("*.csv"))
print(f"\nDone. {len(all_files)} CSV files in {OUTPUT_DIR}/")
if errors:
    print(f"Failed: {errors}")
