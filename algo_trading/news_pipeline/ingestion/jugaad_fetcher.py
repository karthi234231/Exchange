"""algo_trading.news_pipeline.ingestion.jugaad_fetcher — jugaad-data wrapper.

Wraps the ``jugaad-data`` library to fetch:
- OHLCV history for individual symbols (``NSEHistory.stock_df``)
- Index OHLCV history (``NSEHistory.index_df``)
- Full bhavcopy (``NSEHistory.full_bhavcopy_df``)
- Corporate actions (dividends, splits, bonuses) via the NSE corporate actions
  endpoint exposed by ``jugaad_data.nse``

All jugaad-data calls are synchronous; they run in a ``ThreadPoolExecutor``
so the event loop stays unblocked.

The ``_fetch_internal()`` method (called by the base retry loop) fetches
today's corporate actions and converts each one to a ``RawNewsItem`` so the
rest of the pipeline can score and route it like any other news event.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog

from algo_trading.news_pipeline.ingestion.base import BaseFetcher
from algo_trading.news_pipeline.models import RawNewsItem, make_id

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401

log = structlog.get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jugaad")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _today() -> date:
    return datetime.now(tz=timezone.utc).date()


def _nse_history() -> Any | None:
    """Import and return an NSEHistory instance; None if jugaad-data is absent."""
    try:
        from jugaad_data.nse import NSEHistory  # type: ignore[import]
        return NSEHistory()
    except ImportError:
        log.warning(
            "jugaad_data_not_installed",
            msg="jugaad-data is not installed; skipping jugaad fetch",
        )
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("jugaad_nse_history_init_failed", error=str(exc))
        return None


def _normalise_df(df: Any) -> Any:
    """Normalise DataFrame column names to lowercase and return it."""
    if df is None:
        return None
    try:
        df.columns = [str(c).lower().strip() for c in df.columns]
    except Exception:  # noqa: BLE001
        pass
    return df


# ---------------------------------------------------------------------------
# JugaadFetcher
# ---------------------------------------------------------------------------

class JugaadFetcher(BaseFetcher):
    """Fetches historical OHLCV and corporate-action data via jugaad-data.

    ``_fetch_internal()`` is the entry point used by the base class retry
    loop.  It fetches today's corporate actions and returns them as
    ``list[RawNewsItem]``.

    Additional public helpers (``fetch_ohlcv``, ``fetch_bhavcopy``,
    ``fetch_corporate_actions``) can be called directly by orchestrators or
    backtesting utilities.
    """

    source_name = "jugaad"

    # ------------------------------------------------------------------
    # BaseFetcher implementation
    # ------------------------------------------------------------------

    async def _fetch_internal(self) -> list[RawNewsItem]:
        """Fetch today's corporate actions.  Returns list[RawNewsItem]."""
        today = _today()
        yesterday = today - timedelta(days=1)
        return await self.fetch_corporate_actions(
            symbol=None,
            from_date=yesterday,
            to_date=today,
        )

    # ------------------------------------------------------------------
    # OHLCV history
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
    ) -> "pd.DataFrame | None":
        """Fetch OHLCV history for a single NSE equity symbol.

        Uses ``NSEHistory.stock_df(symbol, from_date, to_date)``.
        Column names are normalised to lowercase.
        Returns None on any error.

        Parameters
        ----------
        symbol:
            NSE ticker, e.g. "RELIANCE".
        from_date / to_date:
            Inclusive date range.
        """
        nse = _nse_history()
        if nse is None:
            return None
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                _EXECUTOR, nse.stock_df, symbol, from_date, to_date
            )
            if df is None or (hasattr(df, "empty") and df.empty):
                log.debug("jugaad_ohlcv_empty", symbol=symbol)
                return None
            df = _normalise_df(df)
            log.debug("jugaad_ohlcv_fetched", symbol=symbol, rows=len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning("jugaad_ohlcv_failed", symbol=symbol, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Index OHLCV
    # ------------------------------------------------------------------

    async def fetch_index_ohlcv(
        self,
        index: str,
        from_date: date,
        to_date: date,
    ) -> "pd.DataFrame | None":
        """Fetch OHLCV history for an NSE index (e.g. "NIFTY 50").

        Uses ``NSEHistory.index_df(index, from_date, to_date)``.
        Column names are normalised to lowercase.
        Returns None on any error.
        """
        nse = _nse_history()
        if nse is None:
            return None
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                _EXECUTOR, nse.index_df, index, from_date, to_date
            )
            if df is None or (hasattr(df, "empty") and df.empty):
                log.debug("jugaad_index_ohlcv_empty", index=index)
                return None
            df = _normalise_df(df)
            log.debug("jugaad_index_ohlcv_fetched", index=index, rows=len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning("jugaad_index_ohlcv_failed", index=index, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Bhavcopy
    # ------------------------------------------------------------------

    async def fetch_bhavcopy(self, dt: date) -> "pd.DataFrame | None":
        """Download the full NSE bhavcopy (EOD data for all symbols) for *dt*.

        Uses ``NSEHistory.full_bhavcopy_df(dt)``.
        Column names are normalised to lowercase.
        Returns None on any error (e.g. holiday, data not yet published).

        Parameters
        ----------
        dt:
            Trading date.  Bhavcopy for today is usually available after ~18:00 IST.
        """
        nse = _nse_history()
        if nse is None:
            return None
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                _EXECUTOR, nse.full_bhavcopy_df, dt
            )
            if df is None or (hasattr(df, "empty") and df.empty):
                log.debug("jugaad_bhavcopy_empty", date=str(dt))
                return None
            df = _normalise_df(df)
            log.debug("jugaad_bhavcopy_fetched", date=str(dt), rows=len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning("jugaad_bhavcopy_failed", date=str(dt), error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------

    async def fetch_corporate_actions(
        self,
        symbol: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[RawNewsItem]:
        """Fetch corporate actions and convert each one to a RawNewsItem.

        Corporate actions (dividends, splits, bonuses, rights issues) are
        fetched from NSE via jugaad-data and returned as ``RawNewsItem``
        objects with ``source="jugaad_corp_action"``.  This allows the NLP
        and signal-fusion layers to treat them like news events.

        Parameters
        ----------
        symbol:
            Filter to a specific NSE ticker.  None fetches all symbols.
        from_date / to_date:
            Date range.  Defaults to yesterday–today.
        """
        today = _today()
        if from_date is None:
            from_date = today - timedelta(days=1)
        if to_date is None:
            to_date = today

        nse = _nse_history()
        if nse is None:
            return []

        df = await self._fetch_corp_actions_df(nse, symbol, from_date, to_date)
        if df is None or (hasattr(df, "empty") and df.empty):
            return []

        return self._corp_actions_df_to_news(df)

    async def _fetch_corp_actions_df(
        self,
        nse: Any,
        symbol: str | None,
        from_date: date,
        to_date: date,
    ) -> Any:
        """Try known jugaad-data method signatures for corporate actions."""
        # jugaad-data doesn't have a single stable API for corporate actions.
        # We try several candidate methods in order.
        candidates = [
            ("corp_actions", (symbol, from_date, to_date)),
            ("corporate_actions", (symbol, from_date, to_date)),
            ("stock_corp_action", (symbol, from_date, to_date)),
        ]

        for method_name, args in candidates:
            fn = getattr(nse, method_name, None)
            if fn is None:
                continue
            try:
                clean_args = tuple(a for a in args if a is not None)
                df = await asyncio.get_event_loop().run_in_executor(
                    _EXECUTOR, fn, *clean_args
                )
                if df is not None and not (hasattr(df, "empty") and df.empty):
                    log.debug(
                        "jugaad_corp_actions_fetched",
                        method=method_name,
                        rows=len(df),
                    )
                    return df
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "jugaad_corp_action_method_failed",
                    method=method_name,
                    error=str(exc),
                )

        # Last resort: try the NSE corporate-actions URL directly via
        # jugaad's session (some versions expose raw_request).
        try:
            from jugaad_data.nse import NSEHistory  # type: ignore[import]
            import requests  # type: ignore[import]

            url = (
                "https://www.nseindia.com/api/corporates-corporateActions"
                "?index=equities"
            )
            if symbol:
                url += f"&symbol={symbol}"
            session_fn = getattr(nse, "session", None) or getattr(nse, "_session", None)
            if session_fn is not None:
                # jugaad maintains a requests.Session with NSE cookies.
                def _get() -> Any:
                    try:
                        return session_fn.get(url, timeout=15).json()
                    except Exception:  # noqa: BLE001
                        return None

                raw = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _get)
                if raw and isinstance(raw, list):
                    import pandas as _pd  # type: ignore[import]
                    df = _pd.DataFrame(raw)
                    log.debug("jugaad_corp_actions_raw_api", rows=len(df))
                    return df
        except Exception as exc:  # noqa: BLE001
            log.debug("jugaad_corp_action_raw_api_failed", error=str(exc))

        log.warning("jugaad_corp_actions_unavailable",
                    msg="No working jugaad-data method found for corporate actions")
        return None

    def _corp_actions_df_to_news(self, df: Any) -> list[RawNewsItem]:
        """Convert a corporate-actions DataFrame to a list of RawNewsItem."""
        items: list[RawNewsItem] = []
        fetched_at = _now_utc()

        try:
            df.columns = [str(c).lower().strip() for c in df.columns]
        except Exception:  # noqa: BLE001
            pass

        for _, row in df.iterrows():
            try:
                # Column names vary across jugaad versions; try several aliases.
                symbol = (
                    str(row.get("symbol") or row.get("sym") or row.get("scrip_cd") or "").strip().upper()
                )
                if not symbol:
                    continue

                action_type = str(
                    row.get("purpose") or row.get("action") or row.get("subject") or "CORPORATE ACTION"
                ).strip()

                ex_date_raw = row.get("ex_date") or row.get("exdate") or row.get("date") or ""
                try:
                    ex_date = datetime.strptime(str(ex_date_raw), "%d-%b-%Y").replace(
                        tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    try:
                        ex_date = datetime.strptime(str(ex_date_raw), "%Y-%m-%d").replace(
                            tzinfo=timezone.utc
                        )
                    except (ValueError, TypeError):
                        ex_date = fetched_at

                remarks = str(row.get("remarks") or row.get("detail") or "").strip()
                headline = f"{symbol}: {action_type}"
                if remarks:
                    headline += f" — {remarks[:200]}"

                url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
                item_id = make_id(f"{symbol}|{ex_date_raw}|{action_type}|{remarks}")

                items.append(
                    RawNewsItem(
                        id=item_id,
                        source="jugaad_corp_action",
                        headline=headline,
                        url=url,
                        published_at=ex_date,
                        fetched_at=fetched_at,
                        body=remarks[:500],
                        symbol_hints=[symbol],
                    )
                )
            except Exception as row_exc:  # noqa: BLE001
                log.debug("jugaad_corp_action_row_error", error=str(row_exc))

        log.debug("jugaad_corp_actions_converted", items=len(items))
        return items

    # ------------------------------------------------------------------
    # Thread executor helper
    # ------------------------------------------------------------------

    async def _run_sync(self, func, *args) -> Any:
        """Run a synchronous jugaad-data call inside the shared executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, func, *args)
