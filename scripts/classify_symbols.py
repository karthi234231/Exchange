"""
classify_symbols.py

Reads symbols_list.txt and classifies each symbol into categories:
- nifty50
- banknifty
- nifty_indices   (all other NIFTY indices)
- etfs
- currency
- fo_remnants     (F&O option/contract leftovers)
- commodity
- bonds
- stocks          (all other equities, including mid/small caps)

Outputs a CSV: C:\Exchange\data\index_mapping_detailed.csv
with columns: symbol, category

Usage:
    python -m scripts.classify_symbols
"""

import re
from pathlib import Path

SYMBOLS_FILE = Path(r"C:\Exchange\symbols_list.txt")
OUTPUT_MAPPING = Path(r"C:\Exchange\data\index_mapping_detailed.csv")

# ------------------------------------------------------------------
# Official Nifty 50 and Bank Nifty constituents (as of 2024)
# You can update these lists if needed.
# ------------------------------------------------------------------
NIFTY50 = {
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC", "AXISBANK", "LT",
    "HCLTECH", "ASIANPAINT", "M&M", "MARUTI", "SUNPHARMA", "TITAN",
    "WIPRO", "ADANIPORTS", "NTPC", "ULTRACEMCO", "POWERGRID",
    "INDUSINDBK", "BAJFINANCE", "COALINDIA", "GRASIM", "BRITANNIA",
    "JSWSTEEL", "TATASTEEL", "ADANIENT", "APOLLOHOSP", "HDFCLIFE",
    "SBILIFE", "DRREDDY", "CIPLA", "DIVISLAB", "ONGC", "BPCL",
    "HEROMOTOCO", "NESTLEIND", "TATACONSUM", "EICHERMOT", "BAJAJ-AUTO",
    "MUTHOOTFIN", "SHREECEM", "UPL", "TECHM", "LTTS"
}

BANKNIFTY = {
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCBANK",
    "BANDHANBNK", "RBLBANK", "YESBANK", "CANBK", "UNIONBANK",
    "BANKINDIA", "IOB", "UCOBANK", "CENTRALBK", "INDIANB"
}

# ------------------------------------------------------------------
# Classification rules (order matters – first match wins)
# ------------------------------------------------------------------
def classify_symbol(symbol: str) -> str:
    # 1. Bank Nifty constituents
    if symbol in BANKNIFTY:
        return "banknifty"

    # 2. Nifty 50 constituents (except those already in BankNifty)
    if symbol in NIFTY50:
        return "nifty50"

    # 3. Nifty indices (all symbols starting with "NIFTY" or "BANKNIFTY")
    if symbol.startswith("NIFTY") or symbol.startswith("BANKNIFTY"):
        return "nifty_indices"

    # 4. ETFs – look for common suffixes
    if (symbol.endswith("BEES") or "ETF" in symbol or "IETF" in symbol or
        "BETF" in symbol or symbol.endswith("BND")):
        return "etfs"

    # 5. Currency pairs – end with "INR" and not a stock name
    if symbol.endswith("INR") and not any(c.isalpha() and c.islower() for c in symbol):
        return "currency"

    # 6. F&O leftovers – contain option/contract patterns
    if re.search(r'\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2,4}', symbol):
        return "fo_remnants"
    if "CE" in symbol or "PE" in symbol or "FUT" in symbol:
        return "fo_remnants"

    # 7. Commodity indices
    if symbol.startswith("MCX") or "COMDEX" in symbol or "METAL" in symbol or "AGRI" in symbol:
        return "commodity"

    # 8. Bonds / G-Sec
    if "G-SEC" in symbol or "GS" in symbol or "BOND" in symbol:
        return "bonds"

    # 9. Default: stocks (equities)
    return "stocks"

# ------------------------------------------------------------------
def main():
    if not SYMBOLS_FILE.exists():
        print(f"Error: {SYMBOLS_FILE} not found.")
        return

    symbols = [s.strip() for s in SYMBOLS_FILE.read_text().splitlines() if s.strip()]
    print(f"Loaded {len(symbols)} symbols.")

    # Classify each
    classified = {}
    for sym in symbols:
        category = classify_symbol(sym)
        classified[sym] = category

    # Write mapping CSV
    OUTPUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPING, "w") as f:
        f.write("symbol,category\n")
        for sym in sorted(classified):
            f.write(f"{sym},{classified[sym]}\n")

    # Show summary
    from collections import Counter
    counts = Counter(classified.values())
    print("Category counts:")
    for cat, cnt in counts.most_common():
        print(f"  {cat}: {cnt}")
    print(f"\nMapping saved to {OUTPUT_MAPPING}")

if __name__ == "__main__":
    main()