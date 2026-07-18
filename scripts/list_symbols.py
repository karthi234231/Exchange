"""
list_symbols.py

Scans C:\Exchange\data\consolidated_v2 for all symbol folders,
and writes a list of symbols to a single file in C:\Exchange.

Usage:
    python -m scripts.list_symbols                 # outputs symbols_list.txt
    python -m scripts.list_symbols --csv           # outputs symbols_list.csv
"""

import sys
from pathlib import Path

CONSOLIDATED_DIR = Path(r"C:\Exchange\data\consolidated_v2")
OUTPUT_TXT = Path(r"C:\Exchange\symbols_list.txt")
OUTPUT_CSV = Path(r"C:\Exchange\symbols_list.csv")

def main(output_csv: bool = False):
    if not CONSOLIDATED_DIR.exists():
        print(f"Error: {CONSOLIDATED_DIR} not found.")
        sys.exit(1)

    # Get all subdirectories (symbol folders)
    symbols = [d.name for d in CONSOLIDATED_DIR.iterdir() if d.is_dir()]
    symbols.sort()  # alphabetical order

    if output_csv:
        # Write CSV with header "symbol"
        with open(OUTPUT_CSV, "w") as f:
            f.write("symbol\n")
            for sym in symbols:
                f.write(f"{sym}\n")
        print(f"Symbols saved to {OUTPUT_CSV} ({len(symbols)} symbols)")
    else:
        # Write plain text, one per line
        with open(OUTPUT_TXT, "w") as f:
            for sym in symbols:
                f.write(sym + "\n")
        print(f"Symbols saved to {OUTPUT_TXT} ({len(symbols)} symbols)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="Output as CSV instead of plain text")
    args = parser.parse_args()
    main(output_csv=args.csv)