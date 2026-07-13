"""
scripts/build_data_manifest.py — Generate data/MANIFEST.json documenting all data files.

Scans --data-dir recursively for .parquet and .csv files. For each file extracts:
symbol, timeframe, date range, row count, expected bars, missing bars, whether
corporate actions are applied, and file size.

Usage:
    python scripts/build_data_manifest.py
    python scripts/build_data_manifest.py --data-dir data/ --output data/MANIFEST.json

Output: MANIFEST.json with file inventory and summary statistics.

Exit codes:
    0 — manifest written successfully
    1 — one or more files had errors (manifest still written for good files)
    2 — argument / filesystem error
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

# ── Logging (standalone — does not call configure_logging to avoid .env requirement) ──
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)

# ── Date column names to try, in priority order ───────────────────────────────
_DATE_COLUMNS = ("date", "timestamp", "datetime", "Date", "Timestamp", "DateTime")

# ── Timeframe patterns (keyword → bars per trading day) ──────────────────────
# For NSE: session = 09:15–15:30 = 375 minutes
# 1-min: 375 bars, 3-min: 125, 5-min: 75, 10-min: 37, 15-min: 25, 30-min: 13
# 60-min: 7, hourly: 7, daily: 1
_TIMEFRAME_BARS_PER_DAY: dict[str, int] = {
    "daily": 1,
    "1d": 1,
    "day": 1,
    "1min": 375,
    "1m": 375,
    "3min": 125,
    "3m": 125,
    "5min": 75,
    "5m": 75,
    "10min": 37,
    "10m": 37,
    "15min": 25,
    "15m": 25,
    "30min": 13,
    "30m": 13,
    "60min": 7,
    "60m": 7,
    "1h": 7,
    "1hour": 7,
    "hourly": 7,
}

# Approximate NSE trading days per calendar year (252 is the global standard;
# NSE averages ~248–252 after holidays)
_TRADING_DAYS_PER_YEAR: float = 250.0
_TRADING_DAYS_PER_CALENDAR_DAY: float = _TRADING_DAYS_PER_YEAR / 365.0


def _count_trading_days(start: date, end: date) -> int:
    """Estimate NSE trading days between start and end (inclusive).

    Uses weekday count as a proxy — excludes weekends but not holidays.
    Holidays account for ~10–12 days/year; this estimate is acceptable for
    computing expected bars (not for precise calendar checks).

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        Approximate number of NSE trading days.
    """
    if end < start:
        return 0
    total_days = (end - start).days + 1
    # Count weekends
    weekends = sum(
        1
        for i in range(total_days)
        if (start + pd.Timedelta(days=i)).weekday() >= 5
    )
    # Approximate holidays: 10 per year
    calendar_years = total_days / 365.0
    holidays = round(calendar_years * 10)
    return max(0, total_days - weekends - holidays)


def _parse_timeframe_from_filename(stem: str) -> str:
    """Extract timeframe from filename stem.

    Filename convention: SYMBOL_TIMEFRAME[_adj|_ca|_extra].parquet
    Examples:
        RELIANCE_daily → daily
        INFY_15min → 15min
        TCS_1d_adj → 1d
        HDFCBANK_5m → 5m
        SBIN_daily_adj → daily

    Falls back to "unknown" if no recognised timeframe token is found.

    Args:
        stem: Filename without extension (e.g. "RELIANCE_daily_adj").

    Returns:
        Timeframe string (e.g. "daily", "15min", "5m") or "unknown".
    """
    parts = stem.lower().split("_")
    # Skip the first part (symbol) and check remaining parts for known timeframes
    for part in parts[1:]:
        if part in _TIMEFRAME_BARS_PER_DAY:
            return part
    return "unknown"


def _parse_symbol_from_filename(stem: str) -> str:
    """Extract symbol from filename stem (everything before the first underscore).

    Args:
        stem: Filename without extension (e.g. "RELIANCE_daily_adj").

    Returns:
        Uppercase symbol string (e.g. "RELIANCE").
    """
    return stem.split("_")[0].upper()


def _has_ca_applied(stem: str) -> bool:
    """Return True if the filename contains an _adj or _ca suffix."""
    lower = stem.lower()
    parts = lower.split("_")
    return "adj" in parts or "ca" in parts


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Return the first recognised date/timestamp column name in df."""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def _coerce_to_date(series: pd.Series) -> pd.Series:
    """Convert a date/timestamp series to timezone-naive date values."""
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None) if ts.dt.tz is None else ts.dt.tz_convert(None)
    return ts


def _process_file(path: Path, data_dir: Path) -> dict[str, Any] | None:
    """Process a single data file and return its manifest entry.

    Args:
        path: Absolute path to the data file.
        data_dir: Base data directory (for relative path in manifest).

    Returns:
        Manifest entry dict, or None if the file cannot be processed.
    """
    stem = path.stem
    symbol = _parse_symbol_from_filename(stem)
    timeframe = _parse_timeframe_from_filename(stem)
    ca_applied = _has_ca_applied(stem)
    file_size_kb = round(path.stat().st_size / 1024, 1)

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        log.warning("file_read_error", path=str(path), error=str(exc))
        return None

    total_bars = len(df)

    # Resolve date range
    date_col = _find_date_column(df)
    start_date_str: str | None = None
    end_date_str: str | None = None

    if date_col is not None and total_bars > 0:
        try:
            ts_series = _coerce_to_date(df[date_col])
            ts_series = ts_series.dropna()
            if not ts_series.empty:
                start_date_str = ts_series.min().strftime("%Y-%m-%d")
                end_date_str = ts_series.max().strftime("%Y-%m-%d")
        except Exception as exc:
            log.debug(
                "date_parse_failed",
                path=str(path),
                column=date_col,
                error=str(exc),
            )
    else:
        if date_col is None:
            log.debug(
                "no_date_column",
                path=str(path),
                available=list(df.columns[:10]),
            )

    # Compute expected bars
    expected_bars: int | None = None
    missing_bars: int | None = None
    missing_pct: float | None = None

    bars_per_day = _TIMEFRAME_BARS_PER_DAY.get(timeframe)

    if (
        bars_per_day is not None
        and start_date_str is not None
        and end_date_str is not None
    ):
        try:
            start_d = date.fromisoformat(start_date_str)
            end_d = date.fromisoformat(end_date_str)
            trading_days = _count_trading_days(start_d, end_d)
            expected_bars = trading_days * bars_per_day
            missing_bars = max(0, expected_bars - total_bars)
            missing_pct = (
                round(missing_bars / expected_bars * 100, 1)
                if expected_bars > 0
                else 0.0
            )
        except Exception as exc:
            log.debug("expected_bars_calc_failed", path=str(path), error=str(exc))

    try:
        rel_path = str(path.relative_to(data_dir))
    except ValueError:
        rel_path = str(path)

    entry: dict[str, Any] = {
        "path": rel_path,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "total_bars": total_bars,
        "expected_bars": expected_bars,
        "missing_bars": missing_bars,
        "missing_pct": missing_pct,
        "ca_applied": ca_applied,
        "file_size_kb": file_size_kb,
    }

    log.info(
        "file_processed",
        symbol=symbol,
        timeframe=timeframe,
        total_bars=total_bars,
        missing_bars=missing_bars,
        ca_applied=ca_applied,
    )
    return entry


def _print_summary_table(files: list[dict[str, Any]]) -> None:
    """Print a formatted summary table of all manifest entries to stdout."""
    if not files:
        print("No files in manifest.")
        return

    col_path = 40
    col_sym = 12
    col_tf = 8
    col_bars = 8
    col_exp = 8
    col_miss = 7
    col_miss_pct = 8
    col_ca = 5

    header = (
        f"{'Path':<{col_path}} | "
        f"{'Symbol':<{col_sym}} | "
        f"{'TF':<{col_tf}} | "
        f"{'Bars':>{col_bars}} | "
        f"{'Exp':>{col_exp}} | "
        f"{'Miss':>{col_miss}} | "
        f"{'Miss%':>{col_miss_pct}} | "
        f"{'CA':>{col_ca}}"
    )
    sep = "-" * len(header)

    print()
    print("Data Manifest")
    print(sep)
    print(header)
    print(sep)

    for entry in files:
        path_str = entry["path"]
        # Truncate long paths from the left
        if len(path_str) > col_path:
            path_str = "..." + path_str[-(col_path - 3):]

        miss_bars = entry["missing_bars"]
        miss_pct = entry["missing_pct"]
        expected = entry["expected_bars"]

        miss_str = str(miss_bars) if miss_bars is not None else "N/A"
        miss_pct_str = f"{miss_pct:.1f}%" if miss_pct is not None else "N/A"
        exp_str = str(expected) if expected is not None else "N/A"

        print(
            f"{path_str:<{col_path}} | "
            f"{entry['symbol']:<{col_sym}} | "
            f"{entry['timeframe']:<{col_tf}} | "
            f"{entry['total_bars']:>{col_bars},} | "
            f"{exp_str:>{col_exp}} | "
            f"{miss_str:>{col_miss}} | "
            f"{miss_pct_str:>{col_miss_pct}} | "
            f"{'Yes' if entry['ca_applied'] else 'No':>{col_ca}}"
        )

    print(sep)
    print()


def _build_manifest(
    files: list[dict[str, Any]],
    data_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build the complete MANIFEST structure."""
    total_files = len(files)
    unique_symbols = len({f["symbol"] for f in files})
    files_with_gaps = sum(
        1 for f in files if f["missing_bars"] is not None and f["missing_bars"] > 0
    )
    files_ca_applied = sum(1 for f in files if f["ca_applied"])

    return {
        "generated_at": generated_at,
        "data_dir": str(data_dir),
        "files": files,
        "summary": {
            "total_files": total_files,
            "total_symbols": unique_symbols,
            "files_with_gaps": files_with_gaps,
            "files_ca_applied": files_ca_applied,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MANIFEST.json documenting all data files: "
            "date range, bar counts, missing bars, and corporate action status."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/build_data_manifest.py\n"
            "  python scripts/build_data_manifest.py --data-dir data/ "
            "--output data/MANIFEST.json\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        metavar="DIR",
        help="Root data directory to scan recursively (default: data/).",
    )
    parser.add_argument(
        "--output",
        default="data/MANIFEST.json",
        metavar="FILE",
        help="Output path for MANIFEST.json (default: data/MANIFEST.json).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("data_dir_not_found", path=str(data_dir))
        print(f"ERROR: --data-dir {data_dir!r} does not exist.", file=sys.stderr)
        return 2

    if not data_dir.is_dir():
        print(f"ERROR: --data-dir {data_dir!r} is not a directory.", file=sys.stderr)
        return 2

    # Discover all parquet and csv files recursively
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    csv_files = sorted(data_dir.rglob("*.csv"))
    all_files = parquet_files + csv_files

    if not all_files:
        print(
            f"WARNING: No .parquet or .csv files found under {data_dir!r}.",
            file=sys.stderr,
        )
        # Write empty manifest anyway
        all_files = []

    log.info(
        "scanning_data_dir",
        data_dir=str(data_dir),
        parquet_count=len(parquet_files),
        csv_count=len(csv_files),
    )

    generated_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    entries: list[dict[str, Any]] = []
    errors = 0

    for path in all_files:
        entry = _process_file(path, data_dir)
        if entry is None:
            errors += 1
        else:
            entries.append(entry)

    # Sort: by symbol, then timeframe
    entries.sort(key=lambda e: (e["symbol"], e["timeframe"], e["path"]))

    _print_summary_table(entries)

    manifest = _build_manifest(entries, data_dir, generated_at)

    # Print summary to stdout
    summary = manifest["summary"]
    print(
        f"Summary: {summary['total_files']} files, "
        f"{summary['total_symbols']} symbols, "
        f"{summary['files_with_gaps']} with gaps, "
        f"{summary['files_ca_applied']} CA-adjusted"
    )

    # Write manifest JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log.info(
        "manifest_written",
        path=str(out_path),
        total_files=summary["total_files"],
        total_symbols=summary["total_symbols"],
    )
    print(f"\nManifest written to: {out_path}")

    if errors:
        log.warning("files_with_errors", count=errors)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
