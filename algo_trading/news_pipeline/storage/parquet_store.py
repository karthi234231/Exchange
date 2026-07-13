"""algo_trading.news_pipeline.storage.parquet_store — Parquet-based historical storage.

Daily snapshots are written to::

    {PARQUET_DIR}/{table}/{YYYY-MM-DD}.parquet

Compression: Snappy.  All files are self-describing via Arrow schema.

Usage::

    store = ParquetStore()
    store.write_news_events(events)
    df = store.read_signals(start_date, end_date)
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.models import (
    BulkDealRecord,
    FusedSignal,
    ProcessedNewsEvent,
)

if TYPE_CHECKING:
    import pandas as pd

log = structlog.get_logger(__name__)


class ParquetStore:
    """Parquet-based historical storage for news events, signals, deals, and OHLCV data.

    Files are partitioned by date::

        {base_dir}/{table}/{YYYY-MM-DD}.parquet

    When a file already exists for a given date the new records are appended
    (read-modify-write) so multiple writes within the same trading day are safe.
    """

    def __init__(self, base_dir: str = cfg.PARQUET_DIR) -> None:
        self._base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, table: str, date: datetime.date) -> Path:
        """Return the absolute file path for *table* on *date*."""
        return self._base_dir / table / f"{date.isoformat()}.parquet"

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _append_or_write(self, path: Path, new_table: pa.Table) -> None:
        """Write *new_table* to *path*, appending to any existing file."""
        if path.exists():
            existing = pq.read_table(path)
            # Reconcile schemas: cast new_table to existing schema where possible
            try:
                new_table = new_table.cast(existing.schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # Schema mismatch — let combined write infer a union schema
                pass
            combined = pa.concat_tables([existing, new_table], promote_options="default")
        else:
            combined = new_table

        pq.write_table(combined, path, compression="snappy")

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def write_news_events(
        self,
        events: list[ProcessedNewsEvent],
        date: datetime.date | None = None,
    ) -> Path:
        """Append processed news events to today's (or *date*'s) parquet file.

        Schema mirrors :class:`ProcessedNewsEvent` with list fields stored as
        JSON strings for maximum Arrow compatibility.
        """
        if not events:
            log.warning("parquet_store.write_news_events.empty")
            return self._path("news_events", date or datetime.date.today())

        target_date = date or datetime.date.today()
        path = self._path("news_events", target_date)
        self._ensure_dir(path)

        import json as _json

        rows = []
        for e in events:
            d = e.to_dict()
            # Serialise nested structures as JSON strings for Arrow compatibility
            d["symbols"] = _json.dumps(d["symbols"])
            d["entities"] = _json.dumps(d["entities"])
            d["tags"] = _json.dumps(d["tags"])
            rows.append(d)

        table = pa.Table.from_pylist(rows)
        self._append_or_write(path, table)
        log.info("parquet_store.wrote_news_events", path=str(path), count=len(events))
        return path

    def write_bulk_deals(
        self,
        deals: list[BulkDealRecord],
        date: datetime.date | None = None,
    ) -> Path:
        """Append bulk/block deal records to the parquet file for *date*."""
        target_date = date or datetime.date.today()
        path = self._path("bulk_deals", target_date)
        self._ensure_dir(path)

        if not deals:
            log.warning("parquet_store.write_bulk_deals.empty")
            return path

        rows = [d.to_dict() for d in deals]
        table = pa.Table.from_pylist(rows)
        self._append_or_write(path, table)
        log.info("parquet_store.wrote_bulk_deals", path=str(path), count=len(deals))
        return path

    def write_signals(
        self,
        signals: list[FusedSignal],
        date: datetime.date | None = None,
    ) -> Path:
        """Append fused signal records to the parquet file for *date*."""
        target_date = date or datetime.date.today()
        path = self._path("fused_signals", target_date)
        self._ensure_dir(path)

        if not signals:
            log.warning("parquet_store.write_signals.empty")
            return path

        import json as _json

        rows = []
        for s in signals:
            d = s.to_dict()
            d["contributing_events"] = _json.dumps(d["contributing_events"])
            rows.append(d)

        table = pa.Table.from_pylist(rows)
        self._append_or_write(path, table)
        log.info("parquet_store.wrote_signals", path=str(path), count=len(signals))
        return path

    def write_ohlcv(
        self,
        symbol: str,
        df: "pd.DataFrame",
        date: datetime.date | None = None,
    ) -> Path:
        """Write an OHLCV DataFrame (from jugaad-data) for *symbol* on *date*.

        Expected DataFrame columns: date, open, high, low, close, volume.
        The *symbol* column is injected automatically.
        """
        target_date = date or datetime.date.today()
        # OHLCV is partitioned by symbol then date
        path = self._base_dir / "ohlcv" / symbol / f"{target_date.isoformat()}.parquet"
        self._ensure_dir(path)

        # Normalise column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"write_ohlcv: DataFrame missing columns {missing}")

        # Inject symbol column
        df["symbol"] = symbol

        # Coerce date column to string for Arrow compatibility
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        log.info("parquet_store.wrote_ohlcv", path=str(path), symbol=symbol, rows=len(df))
        return path

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def _read_date_range(self, table: str, start: datetime.date, end: datetime.date) -> "pd.DataFrame":
        """Read all parquet files for *table* between *start* and *end* inclusive.

        Returns an empty DataFrame if no files are found.
        """
        import pandas as _pd

        files: list[Path] = []
        delta = (end - start).days
        for i in range(delta + 1):
            p = self._path(table, start + datetime.timedelta(days=i))
            if p.exists():
                files.append(p)

        if not files:
            log.warning(
                "parquet_store.read_date_range.no_files",
                table=table,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            return _pd.DataFrame()

        frames = [pq.read_table(f).to_pandas() for f in files]
        return _pd.concat(frames, ignore_index=True)

    def read_signals(self, start: datetime.date, end: datetime.date) -> "pd.DataFrame":
        """Return a DataFrame of fused signals between *start* and *end* inclusive."""
        return self._read_date_range("fused_signals", start, end)

    def read_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> "pd.DataFrame":
        """Return OHLCV DataFrame for *symbol* between *start* and *end* inclusive."""
        import pandas as _pd

        files: list[Path] = []
        delta = (end - start).days
        for i in range(delta + 1):
            d = start + datetime.timedelta(days=i)
            p = self._base_dir / "ohlcv" / symbol / f"{d.isoformat()}.parquet"
            if p.exists():
                files.append(p)

        if not files:
            log.warning(
                "parquet_store.read_ohlcv.no_files",
                symbol=symbol,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            return _pd.DataFrame()

        frames = [pq.read_table(f).to_pandas() for f in files]
        return _pd.concat(frames, ignore_index=True)

    def read_news_events(self, start: datetime.date, end: datetime.date) -> "pd.DataFrame":
        """Return a DataFrame of processed news events between *start* and *end* inclusive."""
        return self._read_date_range("news_events", start, end)
