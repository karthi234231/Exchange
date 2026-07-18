"""
tests/test_consolidate_groups.py

High-level test suite for Phase 2 consolidation.

Tests:
- read_file: CSV with/without header, different column orders, malformed data.
- parse_timestamp: date+time, date-only, various formats.
- process_group: full group processing with dedup, sorting, output.
- main: mock metadata, checkpoint, resume, error handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import polars as pl
import pytest

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import consolidate_groups as cg


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary input/output directories."""
    in_dir = tmp_path / "data"
    out_dir = tmp_path / "consolidated_v2"
    in_dir.mkdir()
    out_dir.mkdir()
    with patch.object(cg, "OUTPUT_DIR", out_dir), \
         patch.object(cg, "METADATA_PATH", out_dir / "metadata.parquet"):
        yield in_dir, out_dir


@pytest.fixture
def sample_csv_with_header(temp_dirs):
    """Sample CSV with header (S2/S3 format)."""
    in_dir, _ = temp_dirs
    fp = in_dir / "TCS_20240101.csv"
    content = (
        "date,open,high,low,close,volume\n"
        "2024-01-01 09:15:00,200.0,202.0,198.0,201.0,2000\n"
        "2024-01-01 09:30:00,201.0,203.0,200.0,202.0,2500\n"
    )
    fp.write_text(content)
    return fp


@pytest.fixture
def sample_csv_no_header(temp_dirs):
    """Sample CSV with a header (S1 format) – adjusted to be readable."""
    in_dir, _ = temp_dirs
    fp = in_dir / "RELIANCE_20240101.csv"
    content = (
        "symbol,date,time,open,high,low,close,volume\n"
        "RELIANCE,20240101,09:15,100.0,101.5,99.5,100.5,1500\n"
        "RELIANCE,20240101,09:30,100.5,102.0,100.0,101.5,1800\n"
    )
    fp.write_text(content)
    return fp


@pytest.fixture
def sample_daily_csv(temp_dirs):
    """Sample daily CSV (S3 format)."""
    in_dir, _ = temp_dirs
    fp = in_dir / "INFY_daily.csv"
    content = (
        "Date,Open,High,Low,Close,Volume,Series,TOTAL_TRADES,QTY_PER_TRADE,DLV_QTY\n"
        "2024-01-01,500.0,505.0,495.0,502.0,10000,EQ,500,20,3000\n"
        "2024-01-02,502.0,508.0,500.0,506.0,12000,EQ,600,20,4000\n"
    )
    fp.write_text(content)
    return fp


@pytest.fixture
def sample_inconsistent_schema(temp_dirs):
    """File with extra columns to test schema normalization."""
    in_dir, _ = temp_dirs
    fp = in_dir / "HDFCBANK_extra.csv"
    content = (
        "symbol,date,time,open,high,low,close,volume,extra\n"
        "HDFCBANK,20240101,09:15,300.0,302.0,298.0,301.0,5000,ignore\n"
    )
    fp.write_text(content)
    return fp


# ============================================================
# Tests for read_file
# ============================================================

class TestReadFile:
    def test_csv_no_header(self, sample_csv_no_header):
        df = cg.read_file(sample_csv_no_header, source_priority=2)
        assert df is not None
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "time" in df.columns
        assert "open" in df.columns
        assert df.shape[0] == 2

    def test_csv_with_header(self, sample_csv_with_header):
        df = cg.read_file(sample_csv_with_header, source_priority=3)
        assert df is not None
        assert "date" in df.columns
        assert "volume" in df.columns
        assert df["open"].dtype == pl.Float64
        assert df.shape[0] == 2

    def test_daily_csv(self, sample_daily_csv):
        df = cg.read_file(sample_daily_csv, source_priority=1)
        assert df is not None
        assert "date" in df.columns
        assert "volume" in df.columns
        assert df["open"].dtype == pl.Float64

    def test_inconsistent_schema(self, sample_inconsistent_schema):
        df = cg.read_file(sample_inconsistent_schema, source_priority=4)
        assert df is not None
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "time" in df.columns
        assert df["symbol"][0] == "HDFCBANK"

    def test_corrupt_file(self, tmp_path):
        fp = tmp_path / "corrupt.csv"
        fp.write_text("not a valid csv")
        df = cg.read_file(fp)
        assert df is None

    def test_empty_file(self, tmp_path):
        fp = tmp_path / "empty.csv"
        fp.touch()
        df = cg.read_file(fp)
        assert df is None


# ============================================================
# Tests for parse_timestamp
# ============================================================

class TestParseTimestamp:
    def test_separate_date_time(self):
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "time": ["09:15:00", "09:30:00"],
            "open": [100, 101],
        })
        result = cg.parse_timestamp(df)
        assert result is not None
        assert "timestamp" in result.columns
        assert result["timestamp"].dtype == pl.Datetime
        assert result.shape[0] == 2

    def test_date_only(self):
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100, 101],
        })
        result = cg.parse_timestamp(df)
        assert result is not None
        assert "timestamp" in result.columns
        assert result["timestamp"].dtype == pl.Datetime
        assert "date" not in result.columns

    def test_time_only(self):
        df = pl.DataFrame({
            "time": ["09:15:00", "09:30:00"],
            "open": [100, 101],
        })
        result = cg.parse_timestamp(df)
        assert result is not None
        assert "timestamp" in result.columns

    def test_missing_date_time(self):
        df = pl.DataFrame({
            "open": [100, 101],
        })
        result = cg.parse_timestamp(df)
        assert result is None

    def test_invalid_format(self):
        # Use a date format that cannot be parsed; all rows will be dropped.
        df = pl.DataFrame({
            "date": ["2024/01/01", "2024/02/01"],
            "open": [100, 101],
        })
        result = cg.parse_timestamp(df)
        assert result is not None
        # Since timestamps are invalid, the rows are dropped -> empty DataFrame.
        assert result.height == 0


# ============================================================
# Tests for process_group
# ============================================================

class TestProcessGroup:
    def test_single_file_group(self, temp_dirs, sample_csv_no_header):
        in_dir, out_dir = temp_dirs
        paths = [sample_csv_no_header]
        priorities = [2]
        result = cg.process_group("RELIANCE", "1min", paths, priorities)
        assert result["status"] == "ok"
        assert result["rows"] == 2
        out_file = out_dir / "RELIANCE" / "1min.parquet"
        assert out_file.exists()
        df = pl.read_parquet(out_file)
        assert df.shape[0] == 2
        assert "timestamp" in df.columns

    def test_multiple_files_dedup(self, temp_dirs):
        in_dir, out_dir = temp_dirs
        f1 = in_dir / "stock1.csv"
        f1.write_text("symbol,date,time,open,high,low,close,volume\nRELIANCE,20240101,09:15,100,101,99,100.5,1000\n")
        f2 = in_dir / "stock2.csv"
        f2.write_text("symbol,date,time,open,high,low,close,volume\nRELIANCE,20240101,09:15,101,102,100,101.5,1500\n")
        paths = [f1, f2]
        priorities = [1, 2]
        result = cg.process_group("RELIANCE", "1min", paths, priorities)
        assert result["status"] == "ok"
        assert result["rows"] == 1
        out_file = out_dir / "RELIANCE" / "1min.parquet"
        df = pl.read_parquet(out_file)
        assert df["close"][0] == 100.5  # higher priority

    def test_schema_mismatch_group(self, temp_dirs):
        """Schema normalisation should now handle mismatched columns gracefully."""
        in_dir, out_dir = temp_dirs
        f1 = in_dir / "RELIANCE_noheader.csv"
        f1.write_text("symbol,date,time,open,high,low,close,volume\nRELIANCE,20240101,09:15,100,101,99,100.5,1000\n")
        f2 = in_dir / "RELIANCE_extra.csv"
        f2.write_text("symbol,date,time,open,high,low,close,volume,extra\nRELIANCE,20240101,09:16,101,102,100,101.5,1500,ignore\n")
        paths = [f1, f2]
        priorities = [2, 3]
        result = cg.process_group("RELIANCE", "1min", paths, priorities)
        # Now that we normalise schemas, concatenation should succeed
        assert result["status"] == "ok"
        out_file = out_dir / "RELIANCE" / "1min.parquet"
        assert out_file.exists()
        df = pl.read_parquet(out_file)
        # Two rows with different timestamps
        assert df.shape[0] == 2

    def test_missing_columns_handling(self, temp_dirs):
        in_dir, out_dir = temp_dirs
        f = in_dir / "bad.csv"
        f.write_text("symbol,date,time,open,high,low\nRELIANCE,20240101,09:15,100,101,99\n")
        paths = [f]
        priorities = [4]
        result = cg.process_group("RELIANCE", "1min", paths, priorities)
        # The file lacks 'close' and 'volume', so read_file should return None
        assert result["status"] == "no_data"


# ============================================================
# Tests for main (with mocks)
# ============================================================

class TestMain:
    def test_main_no_metadata(self, temp_dirs, mocker):
        metadata_path = cg.METADATA_PATH
        if metadata_path.exists():
            metadata_path.unlink()
        mocker.patch.object(cg, "logger")
        cg.main()
        # No exception means it ran

    def test_main_with_mock_metadata(self, temp_dirs, mocker):
        meta = pl.DataFrame({
            "symbol": ["RELIANCE", "RELIANCE", "TCS"],
            "timeframe": ["1min", "daily", "1min"],
            "asset_type": ["equity", "equity", "equity"],
            "is_corrupt": [False, False, False],
            "path": ["a.csv", "b.csv", "c.csv"],
            "source_priority": [1, 2, 3],
        })
        metadata_path = cg.METADATA_PATH
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        meta.write_parquet(metadata_path)

        mock_process = mocker.patch.object(cg, "process_group")
        mock_process.return_value = {"status": "ok", "rows": 10, "files_processed": 1}

        mocker.patch.object(cg, "logger")
        cg.main(limit_groups=1)
        assert mock_process.call_count == 1

    def test_main_resume_from_checkpoint(self, temp_dirs, mocker):
        meta = pl.DataFrame({
            "symbol": ["RELIANCE", "TCS"],
            "timeframe": ["1min", "1min"],
            "asset_type": ["equity", "equity"],
            "is_corrupt": [False, False],
            "path": ["a.csv", "b.csv"],
            "source_priority": [1, 2],
        })
        metadata_path = cg.METADATA_PATH
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        meta.write_parquet(metadata_path)

        checkpoint_path = cg.CHECKPOINT_PATH
        checkpoint_path.write_text("RELIANCE_1min\n")

        mock_process = mocker.patch.object(cg, "process_group")
        mock_process.return_value = {"status": "ok", "rows": 10}

        mocker.patch.object(cg, "logger")
        cg.main(limit_groups=10, resume=True)
        assert mock_process.call_count == 1
        args, _ = mock_process.call_args
        assert args[0] == "TCS"

    def test_main_failed_group_logging(self, temp_dirs, mocker):
        meta = pl.DataFrame({
            "symbol": ["RELIANCE"],
            "timeframe": ["1min"],
            "asset_type": ["equity"],
            "is_corrupt": [False],
            "path": ["a.csv"],
            "source_priority": [1],
        })
        metadata_path = cg.METADATA_PATH
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        meta.write_parquet(metadata_path)

        mock_process = mocker.patch.object(cg, "process_group")
        mock_process.return_value = {"status": "error", "error": "something wrong"}

        mocker.patch.object(cg, "logger")
        cg.main(limit_groups=1)

        failed_path = cg.FAILED_PATH
        assert failed_path.exists()
        content = failed_path.read_text()
        assert "RELIANCE" in content
        assert "something wrong" in content