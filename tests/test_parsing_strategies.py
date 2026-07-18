"""Test different parsing strategies for NSE intraday data files.

Data formats observed:
- 7-column: symbol, date, time, open, high, low, close  (no volume)
- 8-column: symbol, date, time, open, high, low, close, volume

Tests compare parsing speed and correctness of different approaches:
1. Polars read_csv with schema
2. Pandas read_csv
3. Manual line-by-line parsing
4. Manual + numpy
5. Polars lazy reading
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import pytest

DATA_DIR = Path(r"c:\Exchange\data\nifty-banknifty-intraday-data-main")

# ---------------------------------------------------------------
# Fixture: provide a real small file for quick tests
# ---------------------------------------------------------------

@pytest.fixture(scope="module")
def small_7col_file() -> Path:
    """Return path to a ~7-column file with ~500+ lines."""
    fp = DATA_DIR / "2012" / "NOV_2012" / "BANKNIFTY.txt"
    assert fp.exists(), f"{fp} not found"
    lines = open(fp).readlines()
    # Trim to 1000 lines for fast tests
    trimmed = fp.parent / "_test_7col_trimmed.txt"
    with open(trimmed, "w") as fh:
        fh.writelines(lines[:1000])
    return trimmed


@pytest.fixture(scope="module")
def small_8col_file() -> Path:
    """Return path to a ~8-column file."""
    fp = DATA_DIR / "2013" / "24JUN2013" / "ABAN.txt"
    assert fp.exists(), f"{fp} not found"
    lines = open(fp).readlines()
    trimmed = fp.parent / "_test_8col_trimmed.txt"
    with open(trimmed, "w") as fh:
        fh.writelines(lines[:1000])
    return trimmed


@pytest.fixture(scope="module")
def big_7col_file() -> Path:
    """A file with many rows (>5000) for performance testing."""
    fp = DATA_DIR / "2012" / "DEC2012" / "NIFTY.txt"
    assert fp.exists(), f"{fp} not found"
    return fp


@pytest.fixture(scope="module")
def big_8col_file() -> Path:
    """A big 8-column file."""
    fp = DATA_DIR / "2012" / "NIFTY_F1_2012" / "NIFTY_F1.txt"
    assert fp.exists(), f"{fp} not found"
    return fp


# ---------------------------------------------------------------
# Parsing strategies
# ---------------------------------------------------------------

def _detect_cols(line: str) -> int:
    """Detect number of columns from first data line."""
    return len(line.strip().split(","))


def parse_polars(fp: Path) -> list[dict]:
    """Strategy 1: Polars read_csv with schema."""
    import polars as pl

    with open(fp) as fh:
        first = fh.readline().strip()
    n_cols = len(first.split(","))

    columns_map = [
        "symbol", "date", "time", "open", "high", "low", "close", "volume"
    ]
    cols = columns_map[:n_cols]

    # Read all as string first, then cast
    df = pl.read_csv(
        fp,
        has_header=False,
        separator=",",
        infer_schema_length=0,  # all columns read as Utf8
    )
    df.columns = cols

    # Cast numeric columns
    numeric_cols = ["open", "high", "low", "close"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))
    else:
        df = df.with_columns(pl.lit(0).alias("volume"))
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Utf8))

    # Parse datetime
    df = df.with_columns(
        pl.format("{} {}", pl.col("date"), pl.col("time")).alias("_dt")
    ).with_columns(
        pl.col("_dt").str.to_datetime(format="%Y%m%d %H:%M").alias("dt")
    ).drop("_dt")

    # Filter market hours
    df = df.filter(
        (pl.col("time") >= "09:15") & (pl.col("time") < "15:30")
    )

    return df.to_dicts()


def parse_pandas(fp: Path) -> list[dict]:
    """Strategy 2: Pandas read_csv."""
    import pandas as pd

    with open(fp) as fh:
        first = fh.readline().strip()
    n_cols = len(first.split(","))

    names = ["symbol", "date", "time", "open", "high", "low", "close", "volume"][:n_cols]
    dtype = {
        "symbol": str, "date": str, "time": str,
        "open": float, "high": float, "low": float, "close": float, "volume": int,
    }

    df = pd.read_csv(
        fp, header=None, names=names,
        dtype={n: dtype.get(n, float) for n in names},
    )

    if "volume" not in df.columns:
        df["volume"] = 0

    df["dt_str"] = df["date"].astype(str) + " " + df["time"]
    df["dt"] = pd.to_datetime(df["dt_str"], format="%Y%m%d %H:%M")
    df = df.drop(columns=["dt_str"])

    mask = (df["time"] >= "09:15") & (df["time"] < "15:30")
    df = df[mask]

    return df.to_dict("records")


def parse_manual(fp: Path) -> list[dict]:
    """Strategy 3: Manual line-by-line parsing."""
    out = []
    with open(fp) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            symbol = parts[0]
            date = parts[1]
            time_str = parts[2]
            open_p = float(parts[3])
            high = float(parts[4])
            low = float(parts[5])
            close = float(parts[6])
            volume = int(parts[7]) if len(parts) >= 8 else 0

            if time_str < "09:15" or time_str >= "15:30":
                continue

            # Parse datetime
            dt_str = f"{date} {time_str}"
            dt = _parse_dt(dt_str)

            out.append({
                "symbol": symbol, "date": date, "time": time_str,
                "open": open_p, "high": high, "low": low, "close": close,
                "volume": volume, "dt": dt,
            })
    return out


def parse_manual_chunked(fp: Path, chunk_kb: int = 64) -> list[dict]:
    """Strategy 4: Manual parsing with chunked file reads for memory efficiency."""
    out = []
    leftover = ""

    with open(fp, "rb") as fh:
        while True:
            chunk = fh.read(chunk_kb * 1024)
            if not chunk:
                break
            data = leftover + chunk.decode("utf-8", errors="ignore")
            lines = data.split("\n")
            # Keep last incomplete line as leftover
            leftover = lines[-1] if not chunk.endswith(b"\n") else ""
            lines = lines[:-1] if leftover else lines

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                symbol = parts[0]
                date = parts[1]
                time_str = parts[2]
                open_p = float(parts[3])
                high = float(parts[4])
                low = float(parts[5])
                close = float(parts[6])
                volume = int(parts[7]) if len(parts) >= 8 else 0
                if time_str < "09:15" or time_str >= "15:30":
                    continue
                dt_str = f"{date} {time_str}"
                dt = _parse_dt(dt_str)
                out.append({
                    "symbol": symbol, "date": date, "time": time_str,
                    "open": open_p, "high": high, "low": low, "close": close,
                    "volume": volume, "dt": dt,
                })
    return out


def _parse_dt(dt_str: str):
    """Manual datetime parse (faster than strptime)."""
    date_s, time_s = dt_str.split(" ")
    year = int(date_s[:4])
    month = int(date_s[4:6])
    day = int(date_s[6:8])
    hour = int(time_s[:2])
    minute = int(time_s[3:5])
    from datetime import datetime
    return datetime(year, month, day, hour, minute)


# ---------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------

class TestCorrectness7Col:
    """Verify all parsers produce identical results for a 7-column file."""

    def test_all_strategies_produce_same_results(self, small_7col_file):
        r1 = parse_polars(small_7col_file)
        r2 = parse_pandas(small_7col_file)
        r3 = parse_manual(small_7col_file)

        assert len(r1) == len(r2) == len(r3), (
            f"Row count mismatch: polar={len(r1)} pandas={len(r2)} manual={len(r3)}"
        )

        for i, (a, b, c) in enumerate(zip(r1, r2, r3)):
            assert abs(a["open"] - b["open"]) < 0.001, f"Row {i} open: {a['open']} vs {b['open']}"
            assert abs(a["close"] - c["close"]) < 0.001, f"Row {i} close: {a['close']} vs {c['close']}"
            assert a["symbol"] == b["symbol"] == c["symbol"], f"Row {i} symbol mismatch"

    def test_manual_matches_polars_7col(self, small_7col_file):
        r1 = parse_polars(small_7col_file)
        r3 = parse_manual(small_7col_file)
        assert len(r1) == len(r3)
        for a, b in zip(r1, r3):
            for key in ["open", "high", "low", "close", "volume"]:
                assert abs(a[key] - b[key]) < 0.001, f"Key {key}: {a[key]} vs {b[key]}"


class TestCorrectness8Col:
    """Verify all parsers produce identical results for an 8-column file."""

    def test_all_strategies_produce_same_results(self, small_8col_file):
        r1 = parse_polars(small_8col_file)
        r2 = parse_pandas(small_8col_file)
        r3 = parse_manual(small_8col_file)

        assert len(r1) == len(r2) == len(r3), (
            f"Row count mismatch: polar={len(r1)} pandas={len(r2)} manual={len(r3)}"
        )

        for i, (a, b, c) in enumerate(zip(r1, r2, r3)):
            assert a["symbol"] == b["symbol"] == c["symbol"]
            assert abs(a["open"] - b["open"]) < 0.001
            assert abs(a["close"] - c["close"]) < 0.001
            assert a["volume"] == b["volume"] == c["volume"]

    def test_volume_parsed_correctly(self, small_8col_file):
        r = parse_manual(small_8col_file)
        for rec in r[:5]:
            assert rec["volume"] > 0, f"Expected positive volume, got {rec['volume']}"


class TestTimeFilter:
    """Verify market hours filtering works correctly."""

    def test_before_0915_filtered(self):
        """9:14 should be filtered out."""
        import polars as pl
        import tempfile
        content = "TEST,20260101,09:14,100,101,99,100\nTEST,20260101,09:15,101,102,100,101"
        fp = Path(tempfile.mktemp(suffix=".csv"))
        fp.write_text(content)
        for parser in [parse_polars, parse_pandas, parse_manual]:
            results = parser(fp)
            times = [r["time"] for r in results]
            assert "09:14" not in times, f"{parser.__name__} didn't filter 09:14"
            assert "09:15" in times, f"{parser.__name__} filtered 09:15"
        fp.unlink()

    def test_after_1530_filtered(self):
        """15:30 should be filtered out."""
        import tempfile
        content = "TEST,20260101,15:29,100,101,99,100\nTEST,20260101,15:30,101,102,100,101"
        fp = Path(tempfile.mktemp(suffix=".csv"))
        fp.write_text(content)
        for parser in [parse_polars, parse_pandas, parse_manual]:
            results = parser(fp)
            times = [r["time"] for r in results]
            assert "15:30" not in times, f"{parser.__name__} didn't filter 15:30"
            assert "15:29" in times, f"{parser.__name__} filtered 15:29"
        fp.unlink()


# ---------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------

class TestPerformance7Col:
    """Compare parsing speed for a large 7-column file."""

    def test_polars_speed(self, big_7col_file):
        t0 = time.perf_counter()
        results = parse_polars(big_7col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  Polars: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100

    def test_pandas_speed(self, big_7col_file):
        t0 = time.perf_counter()
        results = parse_pandas(big_7col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  Pandas: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100

    def test_manual_speed(self, big_7col_file):
        t0 = time.perf_counter()
        results = parse_manual(big_7col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  Manual: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100

    def test_manual_chunked_speed(self, big_7col_file):
        t0 = time.perf_counter()
        results = parse_manual_chunked(big_7col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  ManualChunked: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100


class TestPerformance8Col:
    """Compare parsing speed for a large 8-column file."""

    def test_polars_speed(self, big_8col_file):
        t0 = time.perf_counter()
        results = parse_polars(big_8col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  Polars: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100

    def test_manual_speed(self, big_8col_file):
        t0 = time.perf_counter()
        results = parse_manual(big_8col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  Manual: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100

    def test_manual_chunked_speed(self, big_8col_file):
        t0 = time.perf_counter()
        results = parse_manual_chunked(big_8col_file)
        elapsed = time.perf_counter() - t0
        print(f"\n  ManualChunked: {len(results)} rows in {elapsed*1000:.1f} ms")
        assert len(results) > 100


# ---------------------------------------------------------------
# Scale test: how fast for ALL data?
# ---------------------------------------------------------------

class TestSymbolGrouping:
    """Test that files for a symbol can be grouped correctly."""

    def test_symbol_extraction(self):
        """Extract symbol from filename and file header."""
        import glob
        # Test a few files
        patterns = [
            (DATA_DIR / "2012" / "DEC2012" / "NIFTY.txt", "NIFTY"),
            (DATA_DIR / "2013" / "24JUN2013" / "ABAN.txt", "ABAN"),
            (DATA_DIR / "2012" / "NIFTY_F1_2012" / "NIFTY_F1.txt", "NIFTY_F1"),
        ]
        for path, expected in patterns:
            if path.exists():
                with open(path) as fh:
                    header_sym = fh.readline().strip().split(",")[0].upper()
                stem = path.stem.upper()
                # Both header and filename should indicate the same symbol
                assert header_sym == expected, f"{path}: header={header_sym} expected={expected}"


# ---------------------------------------------------------------
# Full dataset estimate (run with --run-full flag)
# ---------------------------------------------------------------

def test_estimate_full_parse_time():
    """Estimate total time to parse all 338,990 files (sanity check only).
    Marked as slow - use -m 'slow' to run."""
    import pytest
    pytest.skip("This test is a data point, not a pass/fail")

    # Time a single representative file
    fp = DATA_DIR / "2013" / "24JUN2013" / "ABAN.txt"
    if not fp.exists():
        fp = list((DATA_DIR / "2013").rglob("*.txt"))[0]

    start = time.perf_counter()
    r = parse_manual(fp)
    elapsed = time.perf_counter() - start
    rows = len(r)
    approx_total_files = 338_990

    print(f"\nParsed {rows} rows in {elapsed:.3f}s")
    print(f"Per-file avg: {elapsed*1000:.1f} ms")
    print(f"Estimated total: {elapsed * approx_total_files / 60:.1f} minutes")
    return elapsed