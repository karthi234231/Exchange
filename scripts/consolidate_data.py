"""STEP 1: Consolidate 338,992 tiny files into ~5,000 symbol files.

What this does:
- Walks the data directory ONCE
- Groups all files by the SYMBOL in the file header (not filename)
- Merges all 1-min bars per symbol, sorted by datetime
- Writes one CSV per symbol to data/consolidated/

Why: 338,992 files × 57ms avg I/O = 5+ hours to scan.
      ~5,000 symbol files × 50ms = 4 minutes.

Usage: python scripts/consolidate_data.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(r"c:\Exchange\data\nifty-banknifty-intraday-data-main")
OUT_DIR = Path(r"c:\Exchange\data\consolidated")

MIN_ROWS = 100  # Skip symbols with fewer than 100 rows


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Walk and group by symbol (from file header)
    print("Phase 1: Walking directory tree and grouping by symbol...")
    by_symbol: dict[str, list[Path]] = defaultdict(list)
    total_files = 0
    for root, dirs, fnames in os.walk(DATA_DIR):
        for fn in fnames:
            if not fn.lower().endswith(('.txt', '.csv')):
                continue
            fp = Path(root) / fn
            # Read first line to get symbol
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                    first = fh.readline().strip()
                sym = first.split(',')[0].upper()
            except Exception:
                continue
            by_symbol[sym].append(fp)
            total_files += 1
            if total_files % 50000 == 0:
                print(f"  ... {total_files} files scanned, {len(by_symbol)} symbols found")

    print(f"  Scanned {total_files} files, found {len(by_symbol)} symbols in {time.perf_counter()-t0:.1f}s")

    # Phase 2: Merge and write per-symbol files
    print("\nPhase 2: Merging and writing consolidated files...")
    written = 0
    skipped_small = 0
    skipped_error = 0

    for sym in sorted(by_symbol):
        files = by_symbol[sym]
        all_rows: list[list[str]] = []

        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        parts = line.strip().split(',')
                        if len(parts) >= 7:
                            all_rows.append(parts[:8])  # Keep up to 8 cols
            except Exception:
                skipped_error += 1
                continue

        if len(all_rows) < MIN_ROWS:
            skipped_small += 1
            continue

        # Sort by date+time
        all_rows.sort(key=lambda r: (r[1], r[2]))

        # Write consolidated file
        out_path = OUT_DIR / f"{sym}.csv"
        try:
            with open(out_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume'])
                for row in all_rows:
                    if len(row) == 7:
                        row = row + ['0']  # Add volume=0 for 7-col files
                    writer.writerow(row)
            written += 1
        except Exception:
            skipped_error += 1

        if written % 500 == 0:
            print(f"  ... {written} symbols written ({time.perf_counter()-t0:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Files written: {written}")
    print(f"  Skipped (small): {skipped_small}")
    print(f"  Skipped (error): {skipped_error}")
    print(f"  Output: {OUT_DIR}")


if __name__ == '__main__':
    main()