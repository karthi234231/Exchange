"""
algo_trading/market_data/bhav_copy_ingestor.py — NSE F&O Bhav Copy ingestion.

NSE changed the bhav copy format on July 8, 2024 to the CM-UDiFF
(Common Market-Unified Data Interface for Financial Feeds) format.
The old format silently breaks after this date.

This module handles BOTH formats:
  Pre-July 8 2024: legacy CSV format (fo{DD}{MON}{YYYY}bhav.csv.zip)
  Post-July 8 2024: CM-UDiFF format (BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip)

Format detection: automatic based on date threshold (July 8 2024).

NSE F&O Bhav Copy — Column Mapping
=====================================

Legacy format (pre-July 8 2024):
  URL: https://archives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MON}/
       fo{DD}{MON}{YYYY}bhav.csv.zip
  Key columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
               OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
               OPEN_INT, CHG_IN_OI, TIMESTAMP

CM-UDiFF format (post-July 8 2024):
  URL: https://nsearchives.nseindia.com/content/fo/
       BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
  Key columns (34 total): TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId,
               TckrSymb, ISIN, SctySrs, XpryDt, FinInstrmNm, OptnTp, StrkPric,
               OpnPric, HghPric, LwPric, ClsPric, LastPric, PrvClsPric, UndrlyPric,
               SttlmPric, OpnIntrst, ChngInOpnIntrst, TtlTradgVol, TtlNbOfTxsExctd,
               TtlValOfTxsExctd, TtlBuyQty, TtlSellQty, NtlVal, MktLotSz, ...
  FinInstrmTp discriminates instrument type: FUTIDX/FUTSTK/OPTIDX/OPTSTK

Sources:
  [P] NSE Website: https://www.nseindia.com/market-data/historical-data-fo
  [S] NSE CM-UDiFF format change circular (July 2024)
  [T] Community observations: format changed July 8 2024

WARNING: NSE frequently changes URLs and file naming conventions.
Verify actual URL patterns against the NSE website before production use.
The URLs documented here are best-effort based on observed patterns.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

import structlog

log = structlog.get_logger(__name__)

# ── Format cutover date ───────────────────────────────────────────────────────

# NSE changed bhav copy format on July 8 2024.
# Any date from this day forward uses CM-UDiFF format.
# [S] Community-reported; verify against official NSE circular if available.
CMUDIFF_CUTOVER_DATE = date(2024, 7, 8)

_MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


class BhavCopyFormat(str, Enum):
    LEGACY  = "LEGACY"    # pre-July 8 2024
    CMUDIFF = "CMUDIFF"   # post-July 8 2024


@dataclass
class FOBhavRow:
    """
    Single row of F&O bhav copy data, normalized across both formats.

    All fields mapped to a consistent internal schema regardless of
    whether the source was legacy or CM-UDiFF format.
    """
    trade_date: date
    symbol: str           # e.g. "NIFTY", "BANKNIFTY", "RELIANCE"
    instrument: str       # "OPTIDX", "FUTIDX", "OPTSTK", "FUTSTK"
    expiry_date: date
    strike_price: float   # 0.0 for futures
    option_type: str      # "CE", "PE", or "" for futures
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    settle_price: float
    volume_contracts: int  # number of contracts
    value_lakhs: float     # total value in lakhs INR
    open_interest: int     # OI in contracts
    change_in_oi: int      # CHG_IN_OI
    source_format: BhavCopyFormat

    @property
    def is_futures(self) -> bool:
        return "FUT" in self.instrument

    @property
    def is_options(self) -> bool:
        return "OPT" in self.instrument

    @property
    def is_index(self) -> bool:
        return "IDX" in self.instrument


def detect_format(trade_date: date) -> BhavCopyFormat:
    """
    Determine which format to use based on trade date.

    Parameters
    ----------
    trade_date : date
        Date of the bhav copy.

    Returns
    -------
    BhavCopyFormat.CMUDIFF if date >= July 8 2024, else BhavCopyFormat.LEGACY
    """
    return (
        BhavCopyFormat.CMUDIFF
        if trade_date >= CMUDIFF_CUTOVER_DATE
        else BhavCopyFormat.LEGACY
    )


def build_url(trade_date: date) -> str:
    """
    Construct the NSE F&O bhav copy download URL for a given date.

    WARNING: NSE changes URLs periodically. Verify against NSE website
    before production use. These URLs are best-effort based on observed patterns.

    Parameters
    ----------
    trade_date : date
        Trading date (must be a trading day).

    Returns
    -------
    str
        Full URL for the bhav copy zip file.
    """
    fmt = detect_format(trade_date)

    if fmt == BhavCopyFormat.LEGACY:
        # Example: .../DERIVATIVES/2024/JAN/fo01JAN2024bhav.csv.zip
        year = trade_date.year
        mon = _MONTH_ABBR[trade_date.month]
        day = f"{trade_date.day:02d}"
        return (
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
            f"{year}/{mon}/fo{day}{mon}{year}bhav.csv.zip"
        )
    else:
        # CM-UDiFF format: BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
        # Example: BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip
        date_str = trade_date.strftime("%Y%m%d")
        return (
            f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
        )


def parse_legacy_csv(csv_content: str, trade_date: date) -> list[FOBhavRow]:
    """
    Parse legacy bhav copy format (pre-July 8 2024).

    Legacy columns (comma-separated):
      INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH,
      LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP

    Date formats: EXPIRY_DT = "DD-Mon-YYYY" (e.g. "30-JAN-2025")
    """
    rows: list[FOBhavRow] = []
    lines = csv_content.strip().split("\n")

    if not lines:
        return rows

    # Parse header
    header = [h.strip().upper() for h in lines[0].split(",")]

    def col(name: str) -> int:
        try:
            return header.index(name)
        except ValueError:
            return -1

    idx = {
        "instrument":  col("INSTRUMENT"),
        "symbol":      col("SYMBOL"),
        "expiry":      col("EXPIRY_DT"),
        "strike":      col("STRIKE_PR"),
        "option_type": col("OPTION_TYP"),
        "open":        col("OPEN"),
        "high":        col("HIGH"),
        "low":         col("LOW"),
        "close":       col("CLOSE"),
        "settle":      col("SETTLE_PR"),
        "contracts":   col("CONTRACTS"),
        "value":       col("VAL_INLAKH"),
        "oi":          col("OPEN_INT"),
        "chg_oi":      col("CHG_IN_OI"),
    }

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]

        def get(field: str, default: str = "0") -> str:
            i = idx.get(field, -1)
            if i < 0 or i >= len(parts):
                return default
            return parts[i]

        try:
            expiry_str = get("expiry", "")
            expiry_date = _parse_legacy_date(expiry_str)

            option_type = get("option_type", "").strip().upper()
            if option_type == "XX":
                option_type = ""  # futures have "XX" in legacy format

            rows.append(FOBhavRow(
                trade_date=trade_date,
                symbol=get("symbol").upper(),
                instrument=get("instrument").upper(),
                expiry_date=expiry_date,
                strike_price=_float(get("strike")),
                option_type=option_type,
                open_price=_float(get("open")),
                high_price=_float(get("high")),
                low_price=_float(get("low")),
                close_price=_float(get("close")),
                settle_price=_float(get("settle")),
                volume_contracts=_int(get("contracts")),
                value_lakhs=_float(get("value")),
                open_interest=_int(get("oi")),
                change_in_oi=_int(get("chg_oi")),
                source_format=BhavCopyFormat.LEGACY,
            ))
        except Exception as e:
            log.debug("bhav.legacy_parse_skip", line=line[:80], error=str(e))
            continue

    log.info("bhav.legacy_parsed", date=str(trade_date), rows=len(rows))
    return rows


def parse_cmudiff_csv(csv_content: str, trade_date: date) -> list[FOBhavRow]:
    """
    Parse CM-UDiFF format (post-July 8 2024).

    CM-UDiFF columns (comma-separated):
      TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, TckrSymb,
      ISIN, SctySrs, XpryDt, FinInstrmNm, OptnTp, StrkPric, OpnPric,
      HghPric, LwPric, ClsPric, LastPric, PrvsClsgPric, UndrlygPric,
      SttlmPric, OpnIntrst, ChngInOpnIntrst, TtlTradgVol, TtlNbOfTxsExctd,
      TtlTrfVal, TtlBuyQty, TtlSellQty, ...

    Column names confirmed against the NSE CM-UDiFF specification (July 2024 cutover).
    Key fix: total transaction value column is TtlValOfTxsExctd (not TtlTrfVal).
    FinInstrmTp is the authoritative discriminator for FUTIDX/FUTSTK/OPTIDX/OPTSTK.
    """
    rows: list[FOBhavRow] = []
    lines = csv_content.strip().split("\n")

    if not lines:
        return rows

    header = [h.strip() for h in lines[0].split(",")]

    def col(name: str) -> int:
        try:
            return header.index(name)
        except ValueError:
            # Try case-insensitive
            name_lower = name.lower()
            for i, h in enumerate(header):
                if h.lower() == name_lower:
                    return i
            return -1

    # CM-UDiFF column mapping (confirmed against NSE July 2024 specification)
    idx = {
        "trade_date":  col("TradDt"),
        "symbol":      col("TckrSymb"),
        "fin_instr":   col("FinInstrmTp"),  # FUTIDX/FUTSTK/OPTIDX/OPTSTK discriminator
        "fin_id":      col("FinInstrmId"),
        "expiry":      col("XpryDt"),
        "option_type": col("OptnTp"),
        "strike":      col("StrkPric"),
        "open":        col("OpnPric"),
        "high":        col("HghPric"),
        "low":         col("LwPric"),
        "close":       col("ClsPric"),
        "settle":      col("SttlmPric"),
        "contracts":   col("TtlTradgVol"),
        "value":       col("TtlValOfTxsExctd"),  # total value of transactions executed
        "oi":          col("OpnIntrst"),
        "chg_oi":      col("ChngInOpnIntrst"),
    }

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]

        def get(field: str, default: str = "0") -> str:
            i = idx.get(field, -1)
            if i < 0 or i >= len(parts):
                return default
            return parts[i]

        try:
            # CM-UDiFF date format: YYYY-MM-DD
            expiry_str = get("expiry", "")
            expiry_date = _parse_cmudiff_date(expiry_str)

            instrument = get("fin_instr", "").strip().upper()
            option_type = get("option_type", "").strip().upper()
            # CM-UDiFF uses "-" for non-applicable fields
            if option_type in ("-", "NA", ""):
                option_type = ""

            rows.append(FOBhavRow(
                trade_date=trade_date,
                symbol=get("symbol").upper(),
                instrument=instrument,
                expiry_date=expiry_date,
                strike_price=_float(get("strike")),
                option_type=option_type,
                open_price=_float(get("open")),
                high_price=_float(get("high")),
                low_price=_float(get("low")),
                close_price=_float(get("close")),
                settle_price=_float(get("settle")),
                volume_contracts=_int(get("contracts")),
                value_lakhs=_float(get("value")),
                open_interest=_int(get("oi")),
                change_in_oi=_int(get("chg_oi")),
                source_format=BhavCopyFormat.CMUDIFF,
            ))
        except Exception as e:
            log.debug("bhav.cmudiff_parse_skip", line=line[:80], error=str(e))
            continue

    log.info("bhav.cmudiff_parsed", date=str(trade_date), rows=len(rows))
    return rows


def parse_bhav_zip_bytes(
    zip_bytes: bytes,
    trade_date: date,
    fmt: BhavCopyFormat | None = None,
) -> list[FOBhavRow]:
    """
    Parse a bhav copy zip file from its bytes content.

    Parameters
    ----------
    zip_bytes : bytes
        Raw bytes of the .zip file.
    trade_date : date
        The trading date (used for format detection if fmt is None).
    fmt : BhavCopyFormat | None
        Force a specific format. If None, auto-detects from trade_date.

    Returns
    -------
    list[FOBhavRow]
        Parsed rows. May be empty if the file is empty or unparseable.
    """
    detected_fmt = fmt or detect_format(trade_date)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find the CSV file inside the zip
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                log.error("bhav.no_csv_in_zip", date=str(trade_date))
                return []

            # Use first CSV (there should be exactly one)
            csv_content = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    except zipfile.BadZipFile as e:
        log.error("bhav.bad_zip", date=str(trade_date), error=str(e))
        return []

    if detected_fmt == BhavCopyFormat.LEGACY:
        return parse_legacy_csv(csv_content, trade_date)
    else:
        return parse_cmudiff_csv(csv_content, trade_date)


def parse_bhav_file(file_path: Path, trade_date: date) -> list[FOBhavRow]:
    """
    Parse a bhav copy from a local file path.

    Parameters
    ----------
    file_path : Path
        Path to the .zip file (or .csv file — detected by extension).
    trade_date : date
        The trading date.
    """
    path = Path(file_path)
    if not path.exists():
        log.error("bhav.file_not_found", path=str(path))
        return []

    if path.suffix.lower() == ".zip":
        return parse_bhav_zip_bytes(path.read_bytes(), trade_date)
    elif path.suffix.lower() == ".csv":
        fmt = detect_format(trade_date)
        content = path.read_text(encoding="utf-8", errors="replace")
        if fmt == BhavCopyFormat.LEGACY:
            return parse_legacy_csv(content, trade_date)
        else:
            return parse_cmudiff_csv(content, trade_date)
    else:
        log.error("bhav.unknown_extension", path=str(path))
        return []


def filter_index_options(
    rows: list[FOBhavRow],
    symbol: str | None = None,
) -> list[FOBhavRow]:
    """
    Filter bhav rows to index options only (OPTIDX instrument type).

    Parameters
    ----------
    rows : list[FOBhavRow]
        All bhav rows.
    symbol : str | None
        If provided, filter to this symbol (e.g. "NIFTY", "BANKNIFTY").

    Returns
    -------
    list[FOBhavRow]
        Filtered rows.
    """
    result = [r for r in rows if "OPTIDX" in r.instrument]
    if symbol:
        result = [r for r in result if r.symbol == symbol.upper()]
    return result


def filter_index_futures(
    rows: list[FOBhavRow],
    symbol: str | None = None,
) -> list[FOBhavRow]:
    """Filter to index futures (FUTIDX instrument type)."""
    result = [r for r in rows if "FUTIDX" in r.instrument]
    if symbol:
        result = [r for r in result if r.symbol == symbol.upper()]
    return result


def bhav_to_chain_data(
    rows: list[FOBhavRow],
    expiry_date: date,
) -> dict:
    """
    Convert bhav rows for one expiry into a simplified chain_data dict.

    This format is compatible with the SnapshotStore interface.
    Note: bhav copy is EOD only (no intraday). Use for historical research only.

    Returns
    -------
    dict with structure:
    {
        "trade_date": str,
        "expiry_date": str,
        "symbol": str,
        "data_quality": "COMPLETE" | "PARTIAL",
        "quotes": {
            (expiry, strike, opt_type): {
                "settle_price": float, "oi": int, "volume": int,
                "close": float, "open": float, ...
            }
        }
    }
    """
    exp_rows = [r for r in rows if r.expiry_date == expiry_date]
    if not exp_rows:
        return {}

    symbol = exp_rows[0].symbol
    trade_date = exp_rows[0].trade_date

    quotes = {}
    for row in exp_rows:
        if not row.is_options:
            continue
        key = (row.expiry_date, row.strike_price, row.option_type)
        quotes[key] = {
            "settle_price": row.settle_price,
            "close_price": row.close_price,
            "open_price": row.open_price,
            "high_price": row.high_price,
            "low_price": row.low_price,
            "open_interest": row.open_interest,
            "change_in_oi": row.change_in_oi,
            "volume_contracts": row.volume_contracts,
            "source_format": row.source_format.value,
        }

    strikes = set(k[1] for k in quotes.keys())
    quality = "COMPLETE" if len(strikes) >= 10 else "PARTIAL"

    return {
        "trade_date": str(trade_date),
        "expiry_date": str(expiry_date),
        "symbol": symbol,
        "data_quality": quality,
        "schema_version": 1,
        "source": f"NSE_BHAV_{exp_rows[0].source_format.value}",
        "quotes": quotes,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_legacy_date(s: str) -> date:
    """Parse legacy bhav date format: DD-Mon-YYYY (e.g. 30-JAN-2025)."""
    s = s.strip().upper()
    if not s or s == "-":
        return date(1900, 1, 1)
    # Try DD-MON-YYYY
    try:
        return datetime.strptime(s, "%d-%b-%Y").date()
    except ValueError:
        pass
    # Try DD-MM-YYYY
    try:
        return datetime.strptime(s, "%d-%m-%Y").date()
    except ValueError:
        pass
    raise ValueError(f"Cannot parse legacy date: {s!r}")


def _parse_cmudiff_date(s: str) -> date:
    """Parse CM-UDiFF date format: YYYY-MM-DD (ISO format)."""
    s = s.strip()
    if not s or s == "-":
        return date(1900, 1, 1)
    # Try ISO format
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    # Try DD-Mon-YYYY as fallback
    return _parse_legacy_date(s)


def _float(s: str, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        v = float(s.replace(",", "").strip())
        return v if v == v else default  # NaN check
    except (ValueError, AttributeError):
        return default


def _int(s: str, default: int = 0) -> int:
    """Safe int conversion."""
    try:
        return int(float(s.replace(",", "").strip()))
    except (ValueError, AttributeError):
        return default
