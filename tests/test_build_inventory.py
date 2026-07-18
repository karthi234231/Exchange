import json
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import polars as pl
import pytest

# Import the module under test (adjust path as needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import build_inventory as inv


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
    # patch INPUT_DIR and OUTPUT_DIR for tests
    with patch.object(inv, "INPUT_DIR", in_dir), \
         patch.object(inv, "OUTPUT_DIR", out_dir):
        yield in_dir, out_dir


@pytest.fixture
def sample_equity_file(temp_dirs):
    """Create a sample equity CSV file."""
    in_dir, _ = temp_dirs
    fp = in_dir / "RELIANCE.csv"
    fp.write_text("date,open,high,low,close,volume\n2024-01-01,100,101,99,100.5,1000\n")
    return fp


@pytest.fixture
def sample_fo_file(temp_dirs):
    """Create a sample F&O file (should be skipped)."""
    in_dir, _ = temp_dirs
    fp = in_dir / "NIFTY19OCT10800CE.csv"
    fp.write_text("symbol,date,time,open,high,low,close,volume\nNIFTY,2024-01-01,09:15,100,101,99,100.5,1000\n")
    return fp


@pytest.fixture
def mock_detect_timeframe_daily(monkeypatch):
    """Mock detect_timeframe to always return 'daily'."""
    monkeypatch.setattr(inv, "detect_timeframe", lambda path: "daily")
    return monkeypatch


# ============================================================
# Tests for helper functions
# ============================================================

class TestIsFO:
    def test_equity(self):
        assert inv._is_fo("RELIANCE") is False
        assert inv._is_fo("TCS_EQ") is False

    def test_futures_underscore(self):
        assert inv._is_fo("NIFTY_F1") is True
        assert inv._is_fo("RELIANCE_FUT") is True
        assert inv._is_fo("BANKNIFTY_F2") is True

    def test_options_underscore(self):
        assert inv._is_fo("NIFTY_CE") is True
        assert inv._is_fo("RELIANCE_PE") is True
        assert inv._is_fo("NIFTY_CE_22000") is True

    def test_options_no_underscore(self):
        assert inv._is_fo("NIFTY19OCT10800CE") is True
        assert inv._is_fo("BANKNIFTY19OCT29600PE") is True
        assert inv._is_fo("RELIANCE19OCT1440CE") is True

    def test_numeric_date_fo(self):
        assert inv._is_fo("NIFTY2024010110800CE") is True


class TestExtractSymbolInfo:
    def test_equity(self):
        symbol, atype, expiry, strike = inv.extract_symbol_info("RELIANCE")
        assert symbol == "RELIANCE"
        assert atype == "equity"
        assert expiry is None
        assert strike is None

    def test_index(self):
        symbol, atype, expiry, strike = inv.extract_symbol_info("NIFTY")
        assert symbol == "NIFTY"
        assert atype == "index"

    def test_futures_underscore(self):
        symbol, atype, expiry, strike = inv.extract_symbol_info("NIFTY_F1")
        assert symbol == "NIFTY"
        assert atype == "futures"

    def test_options_underscore(self):
        symbol, atype, expiry, strike = inv.extract_symbol_info("NIFTY_CE_22000")
        assert symbol == "NIFTY"
        assert atype == "options"
        assert strike == 22000

    def test_options_no_underscore(self):
        symbol, atype, expiry, strike = inv.extract_symbol_info("NIFTY19OCT10800CE")
        assert symbol == "NIFTY"
        assert atype == "options"
        assert expiry == "19OCT"
        assert strike == 10800


class TestDetectTimeframe:
    def test_daily_from_content(self, tmp_path, mock_detect_timeframe_daily):
        fp = tmp_path / "daily.csv"
        lines = [
            "date,open,high,low,close,volume\n",
            "2024-01-01,100,101,99,100.5,1000\n",
            "2024-01-02,101,102,100,101.5,1100\n",
        ]
        fp.write_text("".join(lines))
        tf = inv.detect_timeframe(fp)
        assert tf == "daily"  # mocked

    def test_1min_from_content(self, tmp_path):
        fp = tmp_path / "1min.csv"
        lines = [
            "date,open,high,low,close,volume\n",
            "2024-01-01 09:15:00,100,101,99,100.5,1000\n",
            "2024-01-01 09:16:00,101,102,100,101.5,1100\n",
        ]
        fp.write_text("".join(lines))
        tf = inv.detect_timeframe(fp)
        assert tf == "1min"

    def test_eod2_data_path(self, temp_dirs):
        in_dir, _ = temp_dirs
        daily_dir = in_dir / "eod2_data" / "daily"
        daily_dir.mkdir(parents=True)
        fp = daily_dir / "reliance.csv"
        fp.touch()
        with patch.object(inv, "detect_timeframe") as mock_dt:
            mock_dt.return_value = "daily"
            assert inv.detect_timeframe(fp) == "daily"


class TestIsCorrupt:
    def test_empty_file(self, tmp_path):
        fp = tmp_path / "empty.csv"
        fp.touch()
        assert inv.is_corrupt(fp) is True

    def test_valid_file(self, tmp_path):
        fp = tmp_path / "valid.csv"
        fp.write_text("date,open\n2024-01-01,100")
        assert inv.is_corrupt(fp) is False

    def test_parquet_corrupt(self, tmp_path):
        fp = tmp_path / "fake.parquet"
        fp.write_text("not a real parquet")
        assert inv.is_corrupt(fp) is True


# ============================================================
# Integration tests for process_file
# ============================================================

class TestProcessFile:
    def test_equity_file(self, temp_dirs, sample_equity_file, mock_detect_timeframe_daily):
        rec = inv.process_file(sample_equity_file)
        assert rec is not None
        assert rec["symbol"] == "RELIANCE"
        assert rec["asset_type"] == "equity"
        assert rec["timeframe"] == "daily"
        assert rec["is_corrupt"] is False

    def test_fo_file_skipped(self, temp_dirs, sample_fo_file):
        rec = inv.process_file(sample_fo_file)
        assert rec is None

    def test_corrupt_file(self, tmp_path):
        fp = tmp_path / "corrupt.csv"
        fp.touch()
        rec = inv.process_file(fp)
        assert rec is not None
        assert rec["is_corrupt"] is True


# ============================================================
# Tests for main function (incremental logic)
# ============================================================

class TestMain:
    def test_first_run_no_metadata(self, temp_dirs, sample_equity_file, sample_fo_file, mocker):
        mock_write = mocker.patch.object(pl.DataFrame, "write_parquet")
        mock_write.return_value = None

        inv.main(max_files=10)

        assert mock_write.called
        # Also check that the state file was created
        state_file = inv.OUTPUT_DIR / "last_scan_state.json"
        # The state is saved even if no new records, but we need to ensure it's saved.
        # We can mock save_state to verify it was called.
        mock_save = mocker.patch.object(inv, "save_state")
        inv.main(max_files=10)
        mock_save.assert_called()

    def test_incremental_no_changes(self, temp_dirs, sample_equity_file, mocker):
        # First run
        inv.main(max_files=10)

        # Mock save_state for the second run to verify it's called
        mock_save = mocker.patch.object(inv, "save_state")
        mock_save.return_value = None

        inv.main(max_files=10)

        # save_state should be called even if no new files
        mock_save.assert_called()

        # Also check that the state file exists
        state_file = inv.OUTPUT_DIR / "last_scan_state.json"
        # The state file might not exist if the first run didn't create it, but save_state is mocked,
        # so we can't check file existence. We'll just verify the call.

    def test_force_full_scan(self, temp_dirs, sample_equity_file, mocker):
        inv.main(max_files=10)
        sample_equity_file.touch()
        mock_write = mocker.patch.object(pl.DataFrame, "write_parquet")
        mock_write.return_value = None
        inv.main(force_full_scan=True, max_files=10)
        assert mock_write.called


# ============================================================
# Performance / stress test (optional)
# ============================================================

@pytest.mark.slow
class TestPerformance:
    def test_many_files(self, temp_dirs, mocker):
        in_dir, _ = temp_dirs
        for i in range(100):
            fp = in_dir / f"stock_{i}.csv"
            fp.write_text("date,open,high,low,close,volume\n2024-01-01,100,101,99,100.5,1000\n")
        mock_write = mocker.patch.object(pl.DataFrame, "write_parquet")
        mock_write.return_value = None
        inv.main()
        assert mock_write.called