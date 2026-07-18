"""
Incremental update: process only new/modified files since last run.

Compares mtime with last scan timestamp in metadata.parquet,
identifies affected groups, and re-consolidates only those.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from consolidate_groups import consolidate_group
from build_inventory import process_file, IGNORE_DIRS

INPUT_DIR = Path(r"C:\Exchange\data")
OUTPUT_DIR = Path(r"C:\Exchange\data\consolidated_v2")
METADATA_PATH = OUTPUT_DIR / "metadata.parquet"
INCREMENTAL_STATE_PATH = OUTPUT_DIR / "incremental_state.json"


def load_last_scan_time() -> float:
    """Load last scan timestamp from state file."""
    if INCREMENTAL_STATE_PATH.exists():
        try:
            import json
            data = json.loads(INCREMENTAL_STATE_PATH.read_text())
            return data.get("last_scan_time", 0)
        except Exception:
            return 0
    return 0


def save_scan_state() -> None:
    """Save current scan timestamp."""
    import json
    state = {"last_scan_time": datetime.now().timestamp()}
    INCREMENTAL_STATE_PATH.write_text(json.dumps(state))


def main() -> None:
    t0 = datetime.now()
    last_scan = load_last_scan_time()
    
    if not METADATA_PATH.exists():
        print("ERROR: metadata.parquet not found. Run build_inventory.py first.")
        return
    
    print(f"[Incremental] Last scan: {datetime.fromtimestamp(last_scan)}")
    
    # Load existing metadata
    meta = pl.read_parquet(METADATA_PATH)
    
    # Find new/modified files
    new_files = []
    for root, dirs, files in INPUT_DIR.walk():
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f.lower().endswith((".csv", ".txt", ".parquet")):
                fp = root / f
                try:
                    mtime = fp.stat().st_mtime
                    if mtime > last_scan:
                        new_files.append(fp)
                except Exception:
                    continue
    
    print(f"  New/modified files: {len(new_files)}")
    
    if not new_files:
        print("  Nothing to update.")
        return
    
    # Process new files
    new_records = []
    for fp in new_files:
        record = process_file(fp)
        new_records.append(record)
    
    # Append to metadata
    new_df = pl.DataFrame(new_records)
    meta = pl.concat([meta, new_df])
    meta.write_parquet(METADATA_PATH)
    
    # Consolidate affected groups
    affected = new_df.filter(~pl.col("is_corrupt")).group_by(["symbol", "timeframe"]).agg(pl.all().first())
    
    print(f"  Affected groups: {len(affected)}")
    
    processed = 0
    for row in affected.iter_rows(named=True):
        symbol = row["symbol"]
        timeframe = row["timeframe"]
        
        group_files = meta.filter(
            (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)
        ).to_dicts()
        
        if not group_files:
            continue
        
        print(f"  Re-consolidating {symbol}/{timeframe}...")
        result = consolidate_group(symbol, timeframe, [Path(f["path"]) for f in group_files], group_files)
        
        if result["status"] == "ok":
            print(f"    OK: {result['rows']} rows")
        else:
            print(f"    FAILED: {result.get('error', 'unknown')}")
        
        processed += 1
    
    # Update state
    save_scan_state()
    
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n[Incremental] Done in {elapsed:.1f}s")
    print(f"  Processed {processed} groups")


if __name__ == "__main__":
    main()