"""
scripts/download_history.py — Download historical OHLCV data from Shoonya/yFinance.

Downloads NSE equity OHLCV data and caches it as Parquet files.  Two sources
are supported:

  - yFinance: ~60 days of 15-min intraday; multiple years of daily data.
    Symbol mapping via NSE_SYMBOL_MAP (.NS suffix for NSE equities).
  - Shoonya: ~400 days of 15-min candles via get_time_price_series().
    Requires SHOONYA_* credentials in .env.  Handles API pagination.

For 3-year training data at 15-min resolution use the Shoonya source and
combine multiple 400-day chunks if needed.  For daily-resolution ML work,
yFinance 3-year download is adequate.

Output: data/historical/{symbol}_{timeframe}_{from}_{to}.parquet
        OR data/historical/{symbol}_{timeframe}.parquet (latest cache)

Usage:
    python -m algo_trading.scripts.download_history --symbols RELIANCE,TCS,NIFTY
    python -m algo_trading.scripts.download_history --source yfinance --years 3
    python -m algo_trading.scripts.download_history --source shoonya --days 400
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Symbol mappings ───────────────────────────────────────────────────────────
# yFinance uses .NS suffix for NSE stocks; indices use ^ prefix.
NSE_SYMBOL_MAP: dict[str, str] = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "WIPRO": "WIPRO.NS",
    "AXISBANK": "AXISBANK.NS",
    "SBIN": "SBIN.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# Shoonya candle interval strings supported by get_time_price_series
_SHOONYA_VALID_INTERVALS: frozenset[str] = frozenset({"1", "3", "5", "10", "15", "30", "60"})

# yFinance interval strings
_YFINANCE_VALID_INTERVALS: frozenset[str] = frozenset({
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo",
})

# Maximum days of 15-min history from Shoonya (ASSUMED — verify before relying on this)
_SHOONYA_MAX_HISTORY_DAYS: int = 400

# yFinance 15-min intraday limit (ASSUMED)
_YFINANCE_15MIN_MAX_DAYS: int = 60


class HistoricalDataDownloader:
    """Downloads and caches OHLCV data for NSE symbols.

    Supports two data sources:
      - ``yfinance``: Free, no credentials. Limited to ~60 days intraday.
      - ``shoonya``: ~400 days 15-min data. Requires .env credentials.

    All returned DataFrames have columns: timestamp, open, high, low, close,
    volume (all with correct types).  Timestamps are IST (UTC+5:30) timezone-aware.

    Args:
        output_dir: Directory for Parquet cache files.  Defaults to
            ``data/historical`` relative to the current working directory.
    """

    def __init__(self, output_dir: Path = Path("data/historical")) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log = log.bind(component="HistoricalDataDownloader")

    # ── yFinance ──────────────────────────────────────────────────────────────

    def download_yfinance(
        self,
        symbol: str,
        years: int = 3,
        interval: str = "15m",
    ) -> pl.DataFrame:
        """Download historical OHLCV from yFinance.

        Note: 15-min interval is limited to approximately 60 days by the
        yFinance API.  For 3-year training data use interval="1d" (daily bars).
        For 15-min data with longer history use the Shoonya source.

        All timestamps are converted to IST (Asia/Kolkata, UTC+5:30).

        Args:
            symbol: NSE symbol (e.g. "RELIANCE").  Automatically mapped to
                yFinance ticker using NSE_SYMBOL_MAP.
            years: Number of years of history to request.  Ignored for
                intraday intervals — yFinance caps at 60 days regardless.
            interval: yFinance interval string.  Must be one of the
                _YFINANCE_VALID_INTERVALS set.  Defaults to "15m".

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close,
            volume.

        Raises:
            ValueError: If the interval is not supported.
            ImportError: If yfinance is not installed.
            RuntimeError: If the download returns empty data.
        """
        try:
            import yfinance as yf  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "yfinance is not installed.  Run: pip install yfinance"
            ) from exc

        if interval not in _YFINANCE_VALID_INTERVALS:
            raise ValueError(
                f"Unsupported yFinance interval {interval!r}.  "
                f"Valid: {sorted(_YFINANCE_VALID_INTERVALS)}"
            )

        ticker_sym = NSE_SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}.NS")
        end_dt = datetime.now(tz=timezone.utc)

        # yFinance caps intraday at 60 days; for daily use full years range
        if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            if years > 0:
                self._log.warning(
                    "yfinance.intraday_limit",
                    note=f"yFinance caps 15m at ~{_YFINANCE_15MIN_MAX_DAYS} days; "
                         "years parameter ignored for intraday intervals",
                )
            start_dt = end_dt - timedelta(days=_YFINANCE_15MIN_MAX_DAYS)
        else:
            start_dt = end_dt - timedelta(days=years * 365)

        self._log.info(
            "yfinance.download.start",
            ticker=ticker_sym,
            interval=interval,
            start=start_dt.date().isoformat(),
            end=end_dt.date().isoformat(),
        )

        ticker = yf.Ticker(ticker_sym)
        raw = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            prepost=False,
        )

        if raw is None or raw.empty:
            raise RuntimeError(
                f"yFinance returned no data for {ticker_sym!r} "
                f"(interval={interval}, start={start_dt.date()}).  "
                "Check that the symbol is valid and the date range has trading data."
            )

        df = self._pandas_to_polars_ohlcv(raw, symbol)
        self._log.info(
            "yfinance.download.complete",
            symbol=symbol,
            rows=len(df),
            interval=interval,
        )
        return df

    # ── Shoonya ───────────────────────────────────────────────────────────────

    def download_shoonya(
        self,
        symbol: str,
        days: int = 400,
        interval: str = "15",
    ) -> pl.DataFrame:
        """Download historical OHLCV from Shoonya get_time_price_series().

        Handles pagination — Shoonya returns a limited number of candles per
        call, so this method issues multiple requests to cover the full range.

        Requires Shoonya credentials in .env (SHOONYA_USER, SHOONYA_PASSWORD,
        etc.) loaded via algo_trading.config.settings.

        Args:
            symbol: NSE equity symbol (e.g. "RELIANCE", "TCS").  Will have
                "-EQ" appended for Shoonya lookup if not already present.
            days: Number of calendar days of history to fetch.  Shoonya is
                assumed to support up to 400 days (verify before relying).
            interval: Candle interval in minutes as a string.  Must be one of
                _SHOONYA_VALID_INTERVALS.  Defaults to "15".

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close,
            volume.  Sorted chronologically.  Timestamps in IST.

        Raises:
            ValueError: If the interval is not supported.
            ImportError: If NorenRestApiPy is not installed.
            RuntimeError: If login or data fetch fails.
        """
        if interval not in _SHOONYA_VALID_INTERVALS:
            raise ValueError(
                f"Unsupported Shoonya interval {interval!r}.  "
                f"Valid: {sorted(_SHOONYA_VALID_INTERVALS)}"
            )

        # Import settings here — avoids circular import at module level
        from algo_trading.config import settings  # noqa: PLC0415
        from algo_trading.data.broker_shoonya import ShoonyaBroker  # noqa: PLC0415
        from algo_trading.data.broker_base import Bar  # noqa: PLC0415

        broker = ShoonyaBroker()
        try:
            broker.login()
        except Exception as exc:
            raise RuntimeError(f"Shoonya login failed: {exc}") from exc

        end_dt = datetime.now(tz=IST)
        start_dt = end_dt - timedelta(days=min(days, _SHOONYA_MAX_HISTORY_DAYS))

        self._log.info(
            "shoonya.download.start",
            symbol=symbol,
            interval=interval,
            days=days,
            start=start_dt.date().isoformat(),
            end=end_dt.date().isoformat(),
        )

        # Shoonya may cap candles per request; chunk into 30-day windows
        all_bars: list[Bar] = []
        chunk_days = 30
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
            try:
                chunk_bars = broker.get_historical_bars(
                    symbol=symbol if symbol.endswith("-EQ") else f"{symbol}-EQ",
                    interval_minutes=int(interval),
                    start=cursor,
                    end=chunk_end,
                )
                all_bars.extend(chunk_bars)
                self._log.debug(
                    "shoonya.download.chunk",
                    start=cursor.date().isoformat(),
                    end=chunk_end.date().isoformat(),
                    bars_received=len(chunk_bars),
                )
            except Exception as exc:
                self._log.warning(
                    "shoonya.download.chunk_failed",
                    start=cursor.date().isoformat(),
                    end=chunk_end.date().isoformat(),
                    error=str(exc),
                )
            cursor = chunk_end

        broker.logout()

        if not all_bars:
            raise RuntimeError(
                f"Shoonya returned no bars for {symbol!r} over the last {days} days."
            )

        # Deduplicate and sort
        seen: set[datetime] = set()
        deduped = []
        for bar in sorted(all_bars, key=lambda b: b.timestamp):
            if bar.timestamp not in seen:
                seen.add(bar.timestamp)
                deduped.append(bar)

        df = pl.DataFrame({
            "timestamp": [b.timestamp for b in deduped],
            "open": [float(b.open) for b in deduped],
            "high": [float(b.high) for b in deduped],
            "low": [float(b.low) for b in deduped],
            "close": [float(b.close) for b in deduped],
            "volume": [b.volume for b in deduped],
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "Asia/Kolkata")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ])

        self._log.info(
            "shoonya.download.complete",
            symbol=symbol,
            rows=len(df),
            interval=interval,
        )
        return df

    # ── Parquet I/O ───────────────────────────────────────────────────────────

    def save_parquet(
        self, df: pl.DataFrame, symbol: str, timeframe: str
    ) -> Path:
        """Save OHLCV DataFrame to a Parquet file.

        File is named: ``{output_dir}/{symbol}_{timeframe}.parquet``.
        Any existing file is overwritten.

        Args:
            df: OHLCV DataFrame to persist.
            symbol: Symbol name used in the filename.
            timeframe: Timeframe label (e.g. "15m", "1d") used in filename.

        Returns:
            Path to the written Parquet file.
        """
        filename = f"{symbol.upper()}_{timeframe}.parquet"
        path = self._output_dir / filename
        df.write_parquet(path, compression="snappy")
        self._log.info(
            "parquet.saved",
            path=str(path),
            rows=len(df),
            size_kb=path.stat().st_size // 1024,
        )
        return path

    def load_parquet(self, symbol: str, timeframe: str) -> pl.DataFrame:
        """Load a cached OHLCV Parquet file.

        Args:
            symbol: NSE symbol (case-insensitive).
            timeframe: Timeframe label matching the saved file.

        Returns:
            Polars DataFrame loaded from disk.

        Raises:
            FileNotFoundError: If no cached file exists for the symbol/timeframe.
            ValueError: If the file is more than 7 days stale (warn only —
                does not raise; staleness check is informational).
        """
        filename = f"{symbol.upper()}_{timeframe}.parquet"
        path = self._output_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No cached data found for {symbol} {timeframe}.  "
                f"Expected path: {path}.  Run download first."
            )

        # Staleness warning (informational only)
        age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age_days > 7:
            self._log.warning(
                "parquet.stale",
                path=str(path),
                age_days=age_days,
                note="Consider re-downloading for fresher training data",
            )

        df = pl.read_parquet(path)
        self._log.info(
            "parquet.loaded",
            path=str(path),
            rows=len(df),
            age_days=age_days,
        )
        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pandas_to_polars_ohlcv(
        self,
        raw_df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
        symbol: str,
    ) -> pl.DataFrame:
        """Convert a yFinance pandas OHLCV DataFrame to the standard Polars schema.

        Args:
            raw_df: pandas DataFrame returned by yf.Ticker.history().
            symbol: Symbol name for logging.

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close,
            volume.  Timestamps converted to IST.
        """
        import pandas as pd  # noqa: PLC0415

        df_reset = raw_df.reset_index()

        # yFinance timestamp column is "Datetime" for intraday, "Date" for daily
        ts_col = "Datetime" if "Datetime" in df_reset.columns else "Date"
        df_reset = df_reset.rename(
            columns={
                ts_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Keep only required columns
        keep = ["timestamp", "open", "high", "low", "close", "volume"]
        df_reset = df_reset[[c for c in keep if c in df_reset.columns]]

        # Ensure timestamps are timezone-aware IST
        ts = pd.to_datetime(df_reset["timestamp"])
        if ts.dt.tz is None:
            # Daily bars from yFinance are tz-naive UTC midnight; localise to IST
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        else:
            ts = ts.dt.tz_convert("Asia/Kolkata")
        df_reset["timestamp"] = ts

        pl_df = pl.from_pandas(df_reset).with_columns([
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ])

        # Drop rows with null OHLCV (yFinance sometimes returns NaN for non-trading days)
        pl_df = pl_df.drop_nulls(subset=["open", "high", "low", "close"])
        return pl_df.sort("timestamp")


def main() -> None:
    """CLI entry point for download_history.

    Examples:
        python -m algo_trading.scripts.download_history \\
            --symbols RELIANCE,TCS,HDFCBANK --source yfinance --interval 1d --years 3
        python -m algo_trading.scripts.download_history \\
            --symbols NIFTY --source shoonya --interval 15 --days 400
    """
    parser = argparse.ArgumentParser(
        description="Download historical NSE OHLCV data (yFinance or Shoonya).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        default="RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK",
        help="Comma-separated NSE symbols (default: RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK)",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "shoonya"],
        default="yfinance",
        help="Data source (default: yfinance)",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        help="Candle interval: '15m' for yfinance, '15' for shoonya (default: 15m)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of history for yFinance daily download (default: 3)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=400,
        help="Days of history for Shoonya download (default: 400)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/historical",
        help="Output directory for Parquet files (default: data/historical)",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    downloader = HistoricalDataDownloader(output_dir=Path(args.output_dir))

    errors: list[str] = []
    for symbol in symbols:
        try:
            if args.source == "yfinance":
                df = downloader.download_yfinance(
                    symbol=symbol,
                    years=args.years,
                    interval=args.interval,
                )
            else:
                df = downloader.download_shoonya(
                    symbol=symbol,
                    days=args.days,
                    interval=args.interval,
                )

            # Use full interval string as timeframe label to avoid ambiguity
            # (e.g. "1h" and "1d" would both become "1" with rstrip, corrupting filenames)
            timeframe = args.interval
            out_path = downloader.save_parquet(df, symbol=symbol, timeframe=timeframe)
            print(f"  {symbol}: {len(df)} bars → {out_path}")

        except Exception as exc:
            log.error("download.failed", symbol=symbol, error=str(exc))
            errors.append(f"{symbol}: {exc}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
