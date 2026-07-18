"""
Phase 1: Build file inventory for NSE data consolidation v2.

Incremental: first run scans all 340k+ files (~75 min). Subsequent runs
only process new/modified files (mtime-based), taking seconds.

State tracked in: last_scan_state.json
Metadata stored in: consolidated_v2/metadata.parquet

Enhanced with:
- Structured logging (logging module)
- Robust error handling (per-file, per-directory)
- Performance optimizations (pruning without re-walk)
- Schema-safe operations (how="diagonal" for concat)
- Detailed progress logging every 1,000 files
"""

from __future__ import annotations

import json
import logging
import os
import re
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

# ---------- Configuration ----------
INPUT_DIR = Path(r"C:\Exchange\data")
OUTPUT_DIR = Path(r"C:\Exchange\data\consolidated_v2")
METADATA_PATH = OUTPUT_DIR / "metadata.parquet"
STATE_FILE = OUTPUT_DIR / "last_scan_state.json"

IGNORE_DIRS = {
    "consolidated",
    "consolidated_v2",
    "temp",
    ".git",
    "__pycache__",
    "test_data",
    "test_consolidated",
    "backtest_results",
}

SOURCE_PRIORITY = {
    "truedata": 1,
    "shoonya": 2,
    "yfinance": 3,
    "eod2_data": 4,
}

# Regex patterns for F&O without underscores (e.g. NIFTY20DEC10000CE)
_FO_NO_UNDERSCORE_RE = re.compile(
    r'^([A-Z]+?)'           # Symbol (letters only, non-greedy)
    r'(\d{2}[A-Z]{3})'      # Date code: 2-digit year + 3-letter month
    r'(\d+)'                # Strike price
    r'(CE|PE)?$'            # Optional option type
)

_FO_NUMERIC_DATE_RE = re.compile(
    r'^([A-Z]+?)'           # Symbol (letters only, non-greedy)
    r'(\d{5,6})'            # Numeric date (5-6 digits)
    r'(\d+)'                # Strike price
    r'(CE|PE)?$'            # Optional option type
)

# ---------- Logging Setup ----------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler(OUTPUT_DIR / "build_inventory.log", mode="a", encoding="utf-8"),
    ],
    force=True,  # Ensure reconfiguration
)
logger = logging.getLogger(__name__)

# ---------- Helpers ----------

def _is_fo(stem: str) -> bool:
    """Return True if filename stem looks like a F&O contract (skip these)."""
    s = stem.upper()
    # Under-score patterns
    if re.search(r'_(F|FUT|CE|PE|OPT)\d*$', s):
        return True
    if re.search(r'_F\d+$', s):
        return True
    if '_CE' in s or '_PE' in s:
        return True
    # No-underscore patterns
    if _FO_NO_UNDERSCORE_RE.match(s):
        return True
    if _FO_NUMERIC_DATE_RE.match(s):
        return True
    return False

def _detect_source_priority(path_str: str) -> int:
    p = path_str.lower()
    for key, prio in SOURCE_PRIORITY.items():
        if key in p:
            return prio
    return 4

def extract_symbol_info(stem: str) -> tuple[str, str, Any, Any]:
    """Extract (symbol, asset_type, expiry, strike) from filename stem."""
    s = stem.upper()

    # Options with underscores: ..._CE<n> or ..._PE<n>
    if "_CE" in s or "_PE" in s:
        parts = s.split("_")
        strike = None
        try:
            strike = int(parts[-1])
            sym_parts = parts[:-1]
        except ValueError:
            sym_parts = parts

        opt_type = None
        expiry = None
        symbol_parts = []
        for p in sym_parts:
            if p in ("CE", "PE"):
                opt_type = p
            elif (len(p) in (5, 6, 8)) and p.isalnum() and any(ch.isdigit() for ch in p):
                expiry = p
            else:
                symbol_parts.append(p)

        symbol = "_".join(symbol_parts) if symbol_parts else s
        return symbol, "options", expiry, strike

    # Futures with underscores: ..._F<n> or ..._FUT
    if "_FUT" in s or re.search(r"_F\d+$", s):
        parts = s.split("_")
        if parts[-1].startswith("F") and parts[-1][1:].isdigit():
            parts = parts[:-1]
        elif parts[-1] == "FUT":
            parts = parts[:-1]

        expiry = None
        symbol_parts = []
        for p in parts:
            if (len(p) in (5, 6, 8)) and p.isalnum() and any(ch.isdigit() for ch in p):
                expiry = p
            else:
                symbol_parts.append(p)

        symbol = "_".join(symbol_parts) if symbol_parts else s
        return symbol, "futures", expiry, None

    # F&O without underscores (handled by _is_fo guard, but keep as fallback)
    m = _FO_NO_UNDERSCORE_RE.match(s)
    if m:
        return m.group(1), ("options" if m.group(4) else "futures"), m.group(2), int(m.group(3))

    m = _FO_NUMERIC_DATE_RE.match(s)
    if m:
        return m.group(1), ("options" if m.group(4) else "futures"), m.group(2), int(m.group(3))

    # Index
    if s in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX") or s.startswith(("NIFTY_", "BANKNIFTY_", "FINNIFTY_", "SENSEX_")):
        return s.split("_")[0], "index", None, None

    # Equity (default)
    return s.split("_")[0], "equity", None, None

def detect_timeframe(path: Path, n_rows: int = 100) -> str:
    """Detect timeframe by reading first n_rows and computing median time delta.
    Supports date-only (YYYY-MM-DD), datetime (YYYY-MM-DD HH:MM:SS), time-only (HH:MM).
    """
    try:
        if path.suffix.lower() == ".parquet":
            try:
                lf = pl.read_parquet(path, n_rows=n_rows)
            except Exception as e:
                logger.warning(f"Parquet read failed for {path}: {e}; falling back to 1min")
                return "1min"
            if "timestamp" in lf.columns:
                ts = lf["timestamp"]
            elif "time" in lf.columns:
                ts = lf["time"]
            else:
                return "1min"
            if ts.dtype == pl.Datetime:
                diffs = ts.diff().drop_nulls().dt.total_minutes().to_list()
            else:
                try:
                    ts = ts.cast(pl.Datetime)
                    diffs = ts.diff().drop_nulls().dt.total_minutes().to_list()
                except Exception:
                    return "1min"
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline().strip() for _ in range(n_rows + 1)]
                lines = [l for l in lines if l]

            if len(lines) < 2:
                return "1min"

            header = lines[0].split(",")
            time_idx = None
            for i, col in enumerate(header):
                col_lower = col.lower()
                if col_lower in ("time", "date", "datetime", "timestamp"):
                    time_idx = i
                    break

            if time_idx is None:
                return "1min"

            times = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > time_idx:
                    times.append(parts[time_idx])
                else:
                    break

            if len(times) < 2:
                return "1min"

            parsed = []
            for t in times:
                t = t.strip()
                if not t:
                    continue
                try:
                    if "-" in t and ":" in t:
                        if len(t.split()) == 2:
                            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                        else:
                            dt = datetime.strptime(t, "%Y-%m-%d %H:%M")
                    elif "-" in t:
                        dt = datetime.strptime(t, "%Y-%m-%d")
                    elif ":" in t:
                        if len(t) == 5:
                            dt = datetime.strptime(t, "%H:%M")
                        else:
                            dt = datetime.strptime(t, "%H:%M:%S")
                    else:
                        if len(t) == 4 and t.isdigit():
                            dt = datetime.strptime(t, "%H%M")
                        elif len(t) == 8 and t.isdigit():
                            dt = datetime.strptime(t, "%Y%m%d")
                        else:
                            continue
                    parsed.append(dt)
                except Exception:
                    continue

            if len(parsed) < 2:
                return "1min"

            diffs = []
            for i in range(1, len(parsed)):
                diff = (parsed[i] - parsed[i-1]).total_seconds() / 60
                if diff > 0:
                    diffs.append(diff)

            if not diffs:
                return "1min"

            median_diff = sorted(diffs)[len(diffs) // 2]

            if median_diff <= 1:
                return "1min"
            elif median_diff <= 15:
                return "15min"
            elif median_diff <= 60:
                return "60min"
            else:
                return "daily"

    except Exception as e:
        logger.warning(f"Error detecting timeframe for {path}: {e}; defaulting to 1min")
        return "1min"

def is_corrupt(path: Path) -> bool:
    """Quick integrity check: file readable, has content."""
    try:
        size = path.stat().st_size
        if size == 0:
            return True
        if path.suffix.lower() in (".csv", ".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
                if not first.strip():
                    return True
        elif path.suffix.lower() == ".parquet":
            try:
                pl.read_parquet(path, n_rows=1)
            except Exception:
                return True
        return False
    except Exception:
        return True

def process_file(path: Path) -> dict[str, Any] | None:
    """Extract metadata for a single file. Returns None if file should be skipped (F&O or corrupt)."""
    stem = path.stem.upper()

    # Skip F&O contracts
    if _is_fo(stem):
        return None

    try:
        stat = path.stat()
    except OSError as e:
        logger.error(f"stat() failed for {path}: {e}")
        return None

    symbol, asset_type, expiry, strike = extract_symbol_info(stem)
    timeframe = detect_timeframe(path)
    corrupt = is_corrupt(path)

    return {
        "path": str(path),
        "stem": stem,
        "extension": path.suffix.lower(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "symbol": symbol,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "source_priority": _detect_source_priority(str(path)),
        "is_corrupt": corrupt,
        "flags": json.dumps([]),
        "expiry": str(expiry) if expiry is not None else None,
        "strike": int(strike) if strike is not None else None,
    }

def load_state() -> float:
    """Return last_scan_time from state file, or 0 if none."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f).get("last_scan_time", 0)
        except Exception as e:
            logger.warning(f"Error reading state file: {e}; resetting last_scan_time=0")
            return 0
    return 0

def save_state(last_scan_time: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_scan_time": last_scan_time}, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def load_existing_metadata() -> tuple[pl.DataFrame | None, set[str]]:
    """Return (existing_df, set_of_known_paths) or (None, empty_set)."""
    if METADATA_PATH.exists():
        try:
            df = pl.read_parquet(METADATA_PATH)
            known_paths = set(df["path"].to_list())
            return df, known_paths
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}; starting fresh.")
            return None, set()
    return None, set()

# ---------- Main (incremental) ----------

def main(force_full_scan: bool = False, max_files: int | None = None) -> None:
    t0 = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load state
    last_scan = 0.0 if force_full_scan else load_state()
    existing_df, known_paths = load_existing_metadata()

    logger.info("Starting Phase 1 (inventory)")
    if existing_df is not None:
        logger.info(f"Existing metadata records: {len(existing_df)}")
        logger.info(f"Last scan timestamp: {datetime.fromtimestamp(last_scan) if last_scan else 'NEVER'}")
    else:
        logger.info("No existing metadata found; will create new.")

    # Walk files
    new_records: list[dict[str, Any]] = []
    total_checked = 0
    total_skipped_fo = 0
    total_skipped_unchanged = 0
    total_errors = 0

    valid_paths = set()

    logger.info("Starting directory walk...")
    try:
        for root, dirs, files in os.walk(INPUT_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for f in files:
                if not f.lower().endswith((".csv", ".txt", ".parquet")):
                    continue

                fp = Path(root) / f
                str_path = str(fp)
                valid_paths.add(str_path)
                total_checked += 1

                # Log progress every 1000 files
                if total_checked % 1000 == 0:
                    logger.info(f"Processed {total_checked} files so far...")

                try:
                    mtime = fp.stat().st_mtime
                except OSError as e:
                    logger.error(f"stat() failed for {fp}: {e}")
                    total_errors += 1
                    continue

                # Skip if unchanged and already in metadata
                if last_scan and str_path in known_paths:
                    if mtime <= last_scan:
                        total_skipped_unchanged += 1
                        continue

                # Process (may return None for F&O)
                try:
                    rec = process_file(fp)
                except Exception as e:
                    logger.error(f"process_file failed for {fp}: {e}")
                    total_errors += 1
                    continue

                if rec is None:
                    total_skipped_fo += 1
                    continue

                new_records.append(rec)

                if max_files and len(new_records) >= max_files:
                    logger.info(f"Reached max_files limit ({max_files})")
                    break
            if max_files and len(new_records) >= max_files:
                break
    except PermissionError as e:
        logger.error(f"Permission error during walk: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during walk: {e}")

    logger.info(f"Walk complete. Checked {total_checked} files, found {len(new_records)} new/modified.")

    # Merge with existing metadata
    combined = None
    if new_records:
        try:
            new_df = pl.DataFrame(new_records)
            if existing_df is not None:
                existing_df = existing_df.filter(~pl.col("path").is_in([r["path"] for r in new_records]))
                combined = pl.concat([existing_df, new_df], how="diagonal")
            else:
                combined = new_df
        except Exception as e:
            logger.error(f"Failed to merge new records: {e}")
            combined = None

    if combined is None and existing_df is not None:
        combined = existing_df

    # Prune: remove entries for files that no longer exist
    if combined is not None:
        def file_exists(path_str: str) -> bool:
            try:
                return Path(path_str).exists()
            except Exception:
                return False

        existing_paths = set(combined["path"].to_list())
        valid_paths_exist = {p for p in existing_paths if file_exists(p)}
        combined = combined.filter(pl.col("path").is_in(list(valid_paths_exist)))

        try:
            combined.write_parquet(METADATA_PATH)
            logger.info(f"Metadata written to {METADATA_PATH}")
            logger.info(f"Total records: {len(combined)}")
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")
    else:
        logger.warning("No metadata to write (no existing and no new records).")

    # Save new scan time (even if no changes, update time)
    save_state(time_module.time())

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase 1 completed in {elapsed:.1f} seconds")
    logger.info(f"  Checked: {total_checked} files")
    logger.info(f"  Skipped (F&O): {total_skipped_fo}")
    logger.info(f"  Skipped (unchanged): {total_skipped_unchanged}")
    logger.info(f"  Errors: {total_errors}")

    if combined is not None:
        logger.info(f"  Symbols: {combined['symbol'].n_unique()}")
        logger.info(f"  Corrupt files: {combined['is_corrupt'].sum()}")
        logger.info("  By asset_type:")
        for row in combined.group_by("asset_type").len().sort("len", descending=True).iter_rows(named=True):
            logger.info(f"    {row['asset_type']}: {row['len']} files")
        logger.info("  By timeframe:")
        for row in combined.group_by("timeframe").len().sort("len", descending=True).iter_rows(named=True):
            logger.info(f"    {row['timeframe']}: {row['len']} files")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    main(force_full_scan=force)