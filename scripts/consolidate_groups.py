"""
Phase 2: Group files by (symbol, timeframe) and merge into Parquet.

Robust, resumable, and memory-efficient consolidation.
Processes one group at a time, writes output, clears memory, and logs progress.
Supports checkpoint/resume, batch processing, and detailed error logging.

Usage:
    python -m scripts.consolidate_groups
    python -m scripts.consolidate_groups --limit 10
    python -m scripts.consolidate_groups --resume
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl

# ---------- Configuration ----------
INPUT_DIR = Path(r"C:\Exchange\data")
OUTPUT_DIR = Path(r"C:\Exchange\data\consolidated_v2")
METADATA_PATH = OUTPUT_DIR / "metadata.parquet"
CHECKPOINT_PATH = OUTPUT_DIR / "processed_groups.txt"
FAILED_PATH = OUTPUT_DIR / "failed_groups.txt"
LOG_PATH = OUTPUT_DIR / "consolidate.log"

# ---------- Logging Setup ----------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(file_format)
logger.addHandler(console_handler)


# ---------- Helper Functions ----------

def read_file(path: Path, source_priority: int = 4) -> pl.DataFrame | None:
    """
    Read a single file into a Polars DataFrame with robust column detection.
    Handles CSV (with/without header), different delimiters, and Parquet.
    Derives symbol from filename if missing.
    """
    path = Path(path)
    try:
        # Detect if first row looks like a header
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        header_lower = first_line.lower()
        has_header = any(
            col in header_lower
            for col in ["date", "time", "open", "high", "low", "close", "volume", "symbol"]
        )

        # Determine delimiter
        delimiters = [",", ";", "\t", " "]
        delimiter = ","
        for sep in delimiters:
            if sep in first_line:
                n_first = len(first_line.split(sep))
                n_second = len(second_line.split(sep)) if second_line else 0
                if n_first > 1 and (n_second == 0 or abs(n_first - n_second) <= 1):
                    delimiter = sep
                    break

        # Read file
        if path.suffix.lower() == ".parquet":
            df = pl.scan_parquet(path).collect()
        else:
            df = pl.scan_csv(
                path,
                has_header=has_header,
                separator=delimiter,
                infer_schema_length=100,
                ignore_errors=True,
            ).collect()

        if df.is_empty():
            return None

        # Rename columns to standard names
        rename_map = {}
        for col in df.columns:
            c_lower = col.lower()
            if c_lower in ("symbol", "ticker"):
                rename_map[col] = "symbol"
            elif c_lower in ("date", "datetime", "timestamp"):
                rename_map[col] = "date"
            elif c_lower == "time":
                rename_map[col] = "time"
            elif c_lower in ("open", "op"):
                rename_map[col] = "open"
            elif c_lower in ("high", "hi"):
                rename_map[col] = "high"
            elif c_lower in ("low", "lo"):
                rename_map[col] = "low"
            elif c_lower in ("close", "cls", "last"):
                rename_map[col] = "close"
            elif c_lower in ("volume", "vol", "traded_qty"):
                rename_map[col] = "volume"
        if rename_map:
            df = df.rename(rename_map)

        # If symbol column is missing, try to derive from filename
        if "symbol" not in df.columns:
            symbol_candidate = path.stem.upper()
            for suffix in ["_EQ", "_NS", "_NSE", "_BSE"]:
                if symbol_candidate.endswith(suffix):
                    symbol_candidate = symbol_candidate[:-len(suffix)]
                    break
            df = df.with_columns(pl.lit(symbol_candidate).alias("symbol"))

        # If we still lack required columns, try reading as no‑header
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing and path.suffix.lower() != ".parquet":
            # Try 8-column no-header (symbol, date, time, open, high, low, close, volume)
            try:
                df_no_header = pl.read_csv(
                    path,
                    has_header=False,
                    separator=delimiter,
                    infer_schema_length=100,
                    ignore_errors=True,
                    new_columns=["symbol", "date", "time", "open", "high", "low", "close", "volume"]
                )
                if all(c in df_no_header.columns for c in required):
                    df = df_no_header
                else:
                    # Try 6-column no-header (date, time, open, high, low, close)
                    df_no_header_6 = pl.read_csv(
                        path,
                        has_header=False,
                        separator=delimiter,
                        infer_schema_length=100,
                        ignore_errors=True,
                        new_columns=["date", "time", "open", "high", "low", "close"]
                    )
                    if all(c in df_no_header_6.columns for c in required):
                        df = df_no_header_6
                        # Add volume = 0
                        df = df.with_columns(pl.lit(0.0).alias("volume"))
                    else:
                        logger.warning(f"Could not parse {path} with standard schemas")
                        return None
            except Exception as e:
                logger.warning(f"Failed to parse {path} as no-header: {e}")
                return None

        # Final check
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(f"Missing columns in {path}: {missing}")
            return None

        # Cast numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Add source priority
        df = df.with_columns(pl.lit(source_priority).alias("_source_prio"))

        return df

    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def parse_timestamp(df: pl.DataFrame) -> pl.DataFrame | None:
    """Parse timestamp from date and/or time columns."""
    if "date" in df.columns and "time" in df.columns:
        df = df.with_columns(
            (pl.col("date").cast(pl.Utf8) + " " + pl.col("time").cast(pl.Utf8)).alias("_ts")
        )
        df = df.with_columns(
            pl.when(pl.col("_ts").str.contains("-"))
            .then(pl.col("_ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
            .otherwise(pl.col("_ts").str.strptime(pl.Datetime, "%Y%m%d %H:%M", strict=False))
            .alias("timestamp")
        )
        df = df.drop(["_ts", "date", "time"])
    elif "date" in df.columns:
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("timestamp")
        )
        df = df.drop("date")
    elif "time" in df.columns:
        df = df.with_columns(
            pl.col("time").str.strptime(pl.Datetime, "%H:%M:%S", strict=False).alias("timestamp")
        )
        df = df.drop("time")
    else:
        logger.warning("No date or time column found; dropping group")
        return None

    df = df.drop_nulls(subset=["timestamp"])
    return df


def process_group(symbol: str, timeframe: str, paths: list[Path], source_priorities: list[int]) -> dict[str, Any]:
    """
    Process one group: read all files, merge, deduplicate, write Parquet.
    Returns result dict with status and metadata.
    """
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "ok",
        "rows": 0,
        "files_processed": 0,
        "files_failed": 0,
        "error": None,
    }

    frames = []
    for p, priority in zip(paths, source_priorities):
        df = read_file(p, priority)
        if df is not None:
            frames.append(df)
            result["files_processed"] += 1
        else:
            result["files_failed"] += 1

    if not frames:
        result["status"] = "no_data"
        return result

    # ---------- SCHEMA NORMALISATION (fixes width mismatches) ----------
    standard_cols = ["symbol", "date", "time", "open", "high", "low", "close", "volume", "_source_prio"]
    normalised = []
    for df in frames:
        # Keep only columns that exist in standard_cols
        existing = [c for c in df.columns if c in standard_cols]
        df = df.select(existing)
        # Add missing columns with null
        for col in standard_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        # Enforce order
        df = df.select(standard_cols)
        normalised.append(df)
    frames = normalised
    # ----------------------------------------------------------------

    try:
        combined = pl.concat(frames)
    except Exception as e:
        result["status"] = "concat_failed"
        result["error"] = str(e)[:200]
        return result

    # Parse timestamp
    try:
        combined = parse_timestamp(combined)
        if combined is None:
            result["status"] = "timestamp_parse_failed"
            return result
    except Exception as e:
        result["status"] = "timestamp_parse_failed"
        result["error"] = str(e)[:200]
        return result

    # Ensure required columns exist
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(c in combined.columns for c in required_cols):
        result["status"] = "missing_columns"
        missing = [c for c in required_cols if c not in combined.columns]
        result["error"] = f"Missing columns: {missing}"
        return result

    # Deduplicate: sort by timestamp and source_priority, keep first
    combined = combined.sort(["timestamp", "_source_prio"]).unique(
        subset=["timestamp"], keep="first", maintain_order=True
    )

    # Drop auxiliary columns (strict=False for safety)
    combined = combined.drop(["_source_prio", "symbol"], strict=False)

    # Select final columns
    combined = combined.select(required_cols)

    if len(combined) == 0:
        result["status"] = "empty_after_dedup"
        return result

    try:
        out_dir = OUTPUT_DIR / symbol.upper()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{timeframe}.parquet"
        combined.write_parquet(out_path)
        result["status"] = "ok"
        result["rows"] = len(combined)
        result["output_path"] = str(out_path)
    except Exception as e:
        result["status"] = "write_failed"
        result["error"] = str(e)[:200]

    # Free memory
    del frames, combined
    gc.collect()
    return result


def load_checkpoint() -> set[str]:
    """Load already processed group keys from checkpoint file."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_checkpoint(key: str):
    """Append a processed group key to checkpoint file."""
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(key + "\n")


def log_failed(symbol: str, timeframe: str, error: str):
    """Log a failed group to the failure file."""
    with open(FAILED_PATH, "a") as f:
        f.write(f"{symbol},{timeframe},{error}\n")


def main(limit_groups: int | None = None, resume: bool = False) -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        logger.error(f"Metadata not found: {METADATA_PATH}")
        return

    logger.info("Loading metadata...")
    try:
        meta = pl.read_parquet(METADATA_PATH)
        meta = meta.filter(~pl.col("is_corrupt"))
        meta = meta.filter(pl.col("asset_type") == "equity")
        groups = meta.group_by(["symbol", "timeframe"]).agg([
            pl.col("path"),
            pl.col("source_priority"),
        ])
        total = len(groups)
        logger.info(f"Found {total} equity groups.")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return

    processed = load_checkpoint() if resume else set()
    if processed:
        logger.info(f"Resuming from checkpoint: {len(processed)} groups already processed.")

    results: list[dict] = []
    failed: list[tuple[str, str, str]] = []
    processed_count = 0

    for row in groups.iter_rows(named=True):
        symbol, timeframe = row["symbol"], row["timeframe"]
        key = f"{symbol}_{timeframe}"
        if key in processed:
            continue

        paths = row["path"]
        priorities = row["source_priority"]
        if not paths or len(paths) == 0:
            continue

        logger.info(f"Processing {symbol}/{timeframe} ({len(paths)} files)")
        res = process_group(symbol, timeframe, paths, priorities)
        results.append(res)
        processed_count += 1

        if res["status"] == "ok":
            save_checkpoint(key)
        else:
            failed.append((symbol, timeframe, res.get("error", "unknown")))
            log_failed(symbol, timeframe, res.get("error", "unknown"))

        if processed_count % 10 == 0:
            ok_count = sum(1 for r in results if r["status"] == "ok")
            logger.info(f"Progress: {processed_count}/{total} groups, OK: {ok_count}, Failed: {len(failed)}")

        if limit_groups and processed_count >= limit_groups:
            logger.info(f"Reached limit {limit_groups}")
            break

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Phase 2 completed in {elapsed:.1f} seconds")
    logger.info(f"  Total groups: {len(results)}")
    logger.info(f"  OK: {ok_count}")
    logger.info(f"  Failed: {len(failed)}")
    if failed:
        logger.info(f"  Failed groups (first 20):")
        for sym, tf, err in failed[:20]:
            logger.info(f"    {sym}/{tf}: {err}")
        logger.info(f"  See {FAILED_PATH} for full list.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Process only first N groups (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    main(limit_groups=args.limit, resume=args.resume)