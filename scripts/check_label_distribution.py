"""
scripts/check_label_distribution.py — Audit triple-barrier label distribution per symbol.

Reads all .parquet (or .csv) files in --data-dir, computes per-class counts and
percentages for the label column (-1, 0, 1), and warns when any class falls below
the --min-pct threshold (default 15%).

Usage:
    python scripts/check_label_distribution.py --data-dir data/features/
    python scripts/check_label_distribution.py --data-dir data/features/ --min-pct 20
    python scripts/check_label_distribution.py --data-dir data/features/ --output label_dist.json

Exit codes:
    0 — all symbols have balanced classes (all >= min-pct)
    1 — at least one symbol has an imbalanced class (< min-pct)
    2 — argument / filesystem error
"""
from __future__ import annotations

import argparse
import json
import sys
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

# Label column names to try, in priority order
_LABEL_COLUMNS = ("label", "target")

# Class names for display
_CLASS_NAMES: dict[int, str] = {1: "Buy", 0: "Hold", -1: "Sell"}
_EXPECTED_CLASSES = (-1, 0, 1)


def _find_data_files(data_dir: Path) -> list[Path]:
    """Return all .parquet files in data_dir; falls back to .csv if none found."""
    parquets = sorted(data_dir.glob("*.parquet"))
    if parquets:
        log.debug("found_parquet_files", count=len(parquets), dir=str(data_dir))
        return parquets

    csvs = sorted(data_dir.glob("*.csv"))
    if csvs:
        log.debug("falling_back_to_csv", count=len(csvs), dir=str(data_dir))
        return csvs

    return []


def _load_labels(path: Path) -> pd.Series | None:
    """Load just the label column from a parquet or csv file.

    Returns a pandas Series of label values, or None if the file cannot be loaded
    or does not contain a recognised label column.
    """
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        log.warning("file_read_error", path=str(path), error=str(exc))
        return None

    # Find label column
    for col in _LABEL_COLUMNS:
        if col in df.columns:
            return df[col]

    log.warning(
        "no_label_column",
        path=str(path),
        available_columns=list(df.columns),
        tried=list(_LABEL_COLUMNS),
    )
    return None


def _symbol_from_path(path: Path) -> str:
    """Extract symbol name from filename.

    Strips the suffix and takes the part before the first underscore.
    E.g. RELIANCE_features.parquet → RELIANCE
         INFY.parquet → INFY
    """
    stem = path.stem  # filename without extension
    return stem.split("_")[0].upper()


def _compute_distribution(
    labels: pd.Series,
    min_pct: float,
) -> dict[str, Any]:
    """Compute per-class counts, percentages, and imbalance status.

    Args:
        labels: Series of label values (-1, 0, 1).
        min_pct: Minimum percentage threshold below which a class is flagged WARN.

    Returns:
        Dict with keys: counts (-1/0/1), total, status, warnings.
    """
    total = len(labels)
    if total == 0:
        return {
            "counts": {-1: 0, 0: 0, 1: 0},
            "pcts": {-1: 0.0, 0: 0.0, 1: 0.0},
            "total": 0,
            "status": "WARN",
            "warnings": ["empty dataset"],
        }

    counts: dict[int, int] = {}
    for cls in _EXPECTED_CLASSES:
        counts[cls] = int((labels == cls).sum())

    pcts: dict[int, float] = {cls: (cnt / total * 100) for cls, cnt in counts.items()}

    warn_msgs: list[str] = []
    for cls in _EXPECTED_CLASSES:
        if pcts[cls] < min_pct:
            warn_msgs.append(f"{_CLASS_NAMES[cls]} < {min_pct:.0f}%")

    return {
        "counts": counts,
        "pcts": pcts,
        "total": total,
        "status": "WARN" if warn_msgs else "OK",
        "warnings": warn_msgs,
    }


def _print_table(
    rows: list[dict[str, Any]],
    min_pct: float,
) -> None:
    """Print a formatted distribution table to stdout."""
    col_sym = 14
    col_total = 7
    col_pct = 8

    header = (
        f"{'Symbol':<{col_sym}} | "
        f"{'Total':>{col_total}} | "
        f"{'Buy%':>{col_pct}} | "
        f"{'Hold%':>{col_pct}} | "
        f"{'Sell%':>{col_pct}} | "
        f"Status"
    )
    sep = "-" * len(header)

    print()
    print(f"Label Distribution Audit  (threshold: {min_pct:.0f}%)")
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        sym = row["symbol"]
        dist = row["dist"]
        counts = dist["counts"]
        pcts = dist["pcts"]
        total = dist["total"]
        status = dist["status"]
        warnings = dist["warnings"]

        status_str = "OK" if status == "OK" else f"WARN: {', '.join(warnings)}"

        print(
            f"{sym:<{col_sym}} | "
            f"{total:>{col_total},} | "
            f"{pcts[1]:>{col_pct - 1}.1f}% | "
            f"{pcts[0]:>{col_pct - 1}.1f}% | "
            f"{pcts[-1]:>{col_pct - 1}.1f}% | "
            f"{status_str}"
        )

    print(sep)
    print()


def _build_json_output(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON output structure from per-symbol rows."""
    result: dict[str, Any] = {}
    for row in rows:
        sym = row["symbol"]
        dist = row["dist"]
        result[sym] = {
            "-1": dist["counts"][-1],
            "0": dist["counts"][0],
            "1": dist["counts"][1],
            "total": dist["total"],
            "status": dist["status"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit triple-barrier label distribution per symbol. "
            "Warns if any class < min-pct threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/check_label_distribution.py --data-dir data/features/\n"
            "  python scripts/check_label_distribution.py --data-dir data/features/ "
            "--min-pct 20 --output label_dist.json\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="DIR",
        help="Directory containing labeled .parquet or .csv files.",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=15.0,
        metavar="PCT",
        help="Minimum class percentage before flagging WARN (default: 15.0).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Optional path to write JSON summary (e.g. label_dist.json).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("data_dir_not_found", path=str(data_dir))
        print(f"ERROR: --data-dir {data_dir!r} does not exist.", file=sys.stderr)
        return 2

    if not data_dir.is_dir():
        log.error("data_dir_not_a_directory", path=str(data_dir))
        print(f"ERROR: --data-dir {data_dir!r} is not a directory.", file=sys.stderr)
        return 2

    min_pct: float = args.min_pct
    if not (0.0 < min_pct < 100.0):
        print(f"ERROR: --min-pct must be between 0 and 100, got {min_pct}.", file=sys.stderr)
        return 2

    files = _find_data_files(data_dir)
    if not files:
        print(
            f"ERROR: No .parquet or .csv files found in {data_dir!r}.",
            file=sys.stderr,
        )
        return 2

    log.info("scanning_files", count=len(files), data_dir=str(data_dir), min_pct=min_pct)

    rows: list[dict[str, Any]] = []
    skipped = 0

    for path in files:
        symbol = _symbol_from_path(path)
        labels = _load_labels(path)

        if labels is None:
            skipped += 1
            log.warning("skipping_file", path=str(path), reason="no label column found")
            continue

        dist = _compute_distribution(labels, min_pct)
        rows.append({"symbol": symbol, "path": str(path), "dist": dist})

        log.info(
            "symbol_processed",
            symbol=symbol,
            total=dist["total"],
            status=dist["status"],
            buy_pct=round(dist["pcts"][1], 1),
            hold_pct=round(dist["pcts"][0], 1),
            sell_pct=round(dist["pcts"][-1], 1),
        )

    if not rows:
        print(
            "ERROR: No files could be processed (all files missing label column).",
            file=sys.stderr,
        )
        return 2

    if skipped:
        log.warning("files_skipped", count=skipped)

    # Sort by symbol name for consistent output
    rows.sort(key=lambda r: r["symbol"])

    _print_table(rows, min_pct)

    # Summary counts
    warn_symbols = [r["symbol"] for r in rows if r["dist"]["status"] == "WARN"]
    ok_count = len(rows) - len(warn_symbols)

    print(
        f"Summary: {ok_count}/{len(rows)} symbols OK, "
        f"{len(warn_symbols)} WARN"
        + (f": {', '.join(warn_symbols)}" if warn_symbols else "")
    )

    # JSON output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = _build_json_output(rows)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(json_data, fh, indent=2)
        log.info("json_output_written", path=str(out_path), symbols=len(json_data))
        print(f"JSON output written to: {out_path}")

    if warn_symbols:
        log.warning(
            "imbalanced_symbols_found",
            count=len(warn_symbols),
            symbols=warn_symbols,
            advice=(
                "Check ATR multipliers in BarrierSpec, widen training date range, "
                "or apply SMOTE/class-weighting before model training."
            ),
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
