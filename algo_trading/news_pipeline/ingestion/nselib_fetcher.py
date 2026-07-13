"""algo_trading.news_pipeline.ingestion.nselib_fetcher — NSE structured data with dual-source strategy.

Fetches the following from NSE:

- Bulk deals  (bulk_deals_archives / bulk_deals)
- Block deals  (block_deals_archives / block_deals)
- Deliverable positions  (deliverable_positions)
- India VIX  (india_vix / indiaVix / get_india_vix)

Source priority per data type
------------------------------
1. ``NSEDirectClient`` (direct aiohttp calls to NSE's internal JSON API) — tried first.
2. ``nselib`` library — used as fallback if the primary source fails or returns nothing.

If both fail for a given sub-type the result is an empty list and a warning is logged.
A failure in one sub-type never prevents the others from running.

All nselib calls are synchronous and run inside ``_EXECUTOR`` (a ``ThreadPoolExecutor``)
so they do not block the asyncio event loop.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any

import structlog

from algo_trading.news_pipeline.ingestion.base import BaseFetcher
from algo_trading.news_pipeline.models import BulkDealRecord, DeliveryRecord, make_id

log = structlog.get_logger(__name__)

# Thread pool for synchronous nselib calls — kept small to avoid overwhelming NSE.
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nselib")

# Module-level singleton for NSEDirectClient (lazy-initialised on first use).
_direct_client: Any | None = None
_direct_client_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _today_str(fmt: str = "%d-%b-%Y") -> str:
    """Return today's date in *fmt*.  Default "DD-Mon-YYYY" (e.g. "20-Jun-2026")."""
    return datetime.now(tz=timezone.utc).strftime(fmt)


def _today_ddmmyyyy() -> str:
    """Return today's date as "DD-MM-YYYY" for nselib functions that need it."""
    return datetime.now(tz=timezone.utc).strftime("%d-%m-%Y")


def _today_date() -> date:
    """Return today as a ``datetime.date`` in UTC."""
    return datetime.now(tz=timezone.utc).date()


def _midnight_utc(dt_date_str: str) -> datetime:
    """Parse "DD-Mon-YYYY" or "DD-MM-YYYY" into a UTC midnight datetime."""
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Safe nselib import
# ---------------------------------------------------------------------------

def _import_capital_market() -> Any | None:
    """Import nselib.capital_market, returning None if the library is absent."""
    try:
        from nselib import capital_market  # type: ignore[import]
        return capital_market
    except ImportError:
        log.warning("nselib_not_installed", msg="nselib is not installed; skipping NSE nselib fallback")
        return None


# ---------------------------------------------------------------------------
# Column-name fuzzy finder (for nselib DataFrames)
# ---------------------------------------------------------------------------

def _find_col(df: Any, candidates: list[str]) -> str | None:
    """Return the first column from *candidates* that exists in *df*.columns."""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# NSEDirectClient singleton management
# ---------------------------------------------------------------------------

async def _get_direct_client() -> Any | None:
    """Return the module-level NSEDirectClient singleton, creating it on first call.

    Returns None if the client cannot be imported or connected (e.g. network down
    at startup), so callers can fall back to nselib without crashing.
    """
    global _direct_client  # noqa: PLW0603

    # Fast path — client already initialised.
    if _direct_client is not None:
        return _direct_client

    async with _direct_client_lock:
        # Double-checked locking: another coroutine may have created it while we waited.
        if _direct_client is not None:
            return _direct_client

        try:
            from algo_trading.news_pipeline.ingestion.nse_direct_client import NSEDirectClient  # noqa: PLC0415

            client = NSEDirectClient()
            await client.connect()
            _direct_client = client
            log.info("nse_direct_client.initialised")
            return _direct_client
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "nse_direct_client.init_failed",
                error=str(exc),
                fallback="will use nselib",
            )
            return None


# ---------------------------------------------------------------------------
# NSElibFetcher
# ---------------------------------------------------------------------------

class NSElibFetcher(BaseFetcher):
    """Fetches structured market data from NSE with a dual-source fallback strategy.

    Primary source : ``NSEDirectClient`` (direct aiohttp to NSE JSON API)
    Fallback source: ``nselib`` Python library (synchronous, run in thread pool)

    Data fetched on each poll:
    - Bulk deals for today
    - Block deals for today
    - Deliverable positions for today

    The class also exposes ``get_india_vix()`` which callers (e.g. the signal
    fusion layer) can invoke directly.

    Architecture
    ------------
    ``_fetch_internal()`` fires three concurrent tasks (bulk / block / delivery).
    Each task first tries NSEDirectClient; if that returns an empty list it
    transparently falls back to nselib, logging which source ultimately provided
    the data so observability is clear.
    """

    source_name = "nselib"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def _fetch_internal(self) -> list[BulkDealRecord | DeliveryRecord]:
        """Run all sub-fetches concurrently and aggregate results."""
        today = _today_date()

        bulk_task     = asyncio.create_task(self._fetch_bulk_deals(today))
        block_task    = asyncio.create_task(self._fetch_block_deals(today))
        delivery_task = asyncio.create_task(self._fetch_delivery(today))

        results: list[BulkDealRecord | DeliveryRecord] = []
        for task in asyncio.as_completed([bulk_task, block_task, delivery_task]):
            try:
                records = await task
                results.extend(records)
            except Exception as exc:  # noqa: BLE001
                log.warning("nselib_fetcher.subtask_failed", error=str(exc))

        log.debug("nselib_fetcher.fetch_complete", total_records=len(results))
        return results

    # ------------------------------------------------------------------
    # Bulk deals
    # ------------------------------------------------------------------

    async def _fetch_bulk_deals(self, today: date) -> list[BulkDealRecord]:
        """Fetch today's bulk deals — NSEDirectClient first, nselib fallback."""
        # --- Primary: NSEDirectClient ---
        client = await _get_direct_client()
        if client is not None:
            try:
                raw_records = await client.get_bulk_deals(today)
                if raw_records:
                    log.debug(
                        "nselib_fetcher.bulk_deals.source",
                        source="direct",
                        records=len(raw_records),
                    )
                    return self._dicts_to_bulk_deal_records(raw_records)
                log.debug("nselib_fetcher.bulk_deals.direct_empty", attempting_fallback=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "nselib_fetcher.bulk_deals.direct_failed",
                    error=str(exc),
                    attempting_fallback=True,
                )

        # --- Fallback: nselib ---
        return await self._nselib_bulk_deals()

    async def _nselib_bulk_deals(self) -> list[BulkDealRecord]:
        """Fetch bulk deals via nselib (synchronous, runs in thread pool)."""
        cm = _import_capital_market()
        if cm is None:
            return []
        try:
            df = await self._run_sync(cm.bulk_deals_archives, _today_str(), _today_str())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "nselib_fetcher.nselib.bulk_deals_archives_failed",
                error=str(exc),
                attempting_fallback=True,
            )
            try:
                df = await self._run_sync(cm.bulk_deals, _today_ddmmyyyy())
            except Exception as exc2:  # noqa: BLE001
                log.warning("nselib_fetcher.nselib.bulk_deals_failed", error=str(exc2))
                return []

        records = self._df_to_bulk_deal_records(df, deal_type="bulk")
        if records:
            log.debug(
                "nselib_fetcher.bulk_deals.source",
                source="nselib",
                records=len(records),
            )
        return records

    # ------------------------------------------------------------------
    # Block deals
    # ------------------------------------------------------------------

    async def _fetch_block_deals(self, today: date) -> list[BulkDealRecord]:
        """Fetch today's block deals — NSEDirectClient first, nselib fallback."""
        # --- Primary: NSEDirectClient ---
        client = await _get_direct_client()
        if client is not None:
            try:
                raw_records = await client.get_block_deals(today)
                if raw_records:
                    log.debug(
                        "nselib_fetcher.block_deals.source",
                        source="direct",
                        records=len(raw_records),
                    )
                    return self._dicts_to_bulk_deal_records(raw_records)
                log.debug("nselib_fetcher.block_deals.direct_empty", attempting_fallback=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "nselib_fetcher.block_deals.direct_failed",
                    error=str(exc),
                    attempting_fallback=True,
                )

        # --- Fallback: nselib ---
        return await self._nselib_block_deals()

    async def _nselib_block_deals(self) -> list[BulkDealRecord]:
        """Fetch block deals via nselib (synchronous, runs in thread pool)."""
        cm = _import_capital_market()
        if cm is None:
            return []
        try:
            df = await self._run_sync(cm.block_deals_archives, _today_str(), _today_str())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "nselib_fetcher.nselib.block_deals_archives_failed",
                error=str(exc),
                attempting_fallback=True,
            )
            try:
                df = await self._run_sync(cm.block_deals, _today_ddmmyyyy())
            except Exception as exc2:  # noqa: BLE001
                log.warning("nselib_fetcher.nselib.block_deals_failed", error=str(exc2))
                return []

        records = self._df_to_bulk_deal_records(df, deal_type="block")
        if records:
            log.debug(
                "nselib_fetcher.block_deals.source",
                source="nselib",
                records=len(records),
            )
        return records

    # ------------------------------------------------------------------
    # Delivery positions
    # ------------------------------------------------------------------

    async def _fetch_delivery(self, today: date) -> list[DeliveryRecord]:
        """Fetch today's deliverable positions — NSEDirectClient first, nselib fallback."""
        # --- Primary: NSEDirectClient ---
        client = await _get_direct_client()
        if client is not None:
            try:
                raw_records = await client.get_delivery_data(today)
                if raw_records:
                    log.debug(
                        "nselib_fetcher.delivery.source",
                        source="direct",
                        records=len(raw_records),
                    )
                    return self._dicts_to_delivery_records(raw_records)
                log.debug("nselib_fetcher.delivery.direct_empty", attempting_fallback=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "nselib_fetcher.delivery.direct_failed",
                    error=str(exc),
                    attempting_fallback=True,
                )

        # --- Fallback: nselib ---
        return await self._nselib_delivery()

    async def _nselib_delivery(self) -> list[DeliveryRecord]:
        """Fetch deliverable positions via nselib (synchronous, runs in thread pool)."""
        cm = _import_capital_market()
        if cm is None:
            return []
        try:
            df = await self._run_sync(
                cm.deliverable_positions,
                _today_str(),
                _today_str(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("nselib_fetcher.nselib.delivery_failed", error=str(exc))
            return []

        records = self._df_to_delivery_records(df)
        if records:
            log.debug(
                "nselib_fetcher.delivery.source",
                source="nselib",
                records=len(records),
            )
        return records

    # ------------------------------------------------------------------
    # India VIX
    # ------------------------------------------------------------------

    async def get_india_vix(self) -> float | None:
        """Fetch the current India VIX value.

        Tries NSEDirectClient first.  Falls back to nselib if the primary
        source fails or returns None.  Returns None if both sources fail.
        """
        # --- Primary: NSEDirectClient ---
        client = await _get_direct_client()
        if client is not None:
            try:
                vix = await client.get_india_vix()
                if vix is not None:
                    log.debug("india_vix.source", source="direct", vix=vix)
                    return vix
                log.debug("india_vix.direct_returned_none", attempting_fallback=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "india_vix.direct_failed",
                    error=str(exc),
                    attempting_fallback=True,
                )

        # --- Fallback: nselib ---
        return await self._nselib_india_vix()

    async def _nselib_india_vix(self) -> float | None:
        """Fetch India VIX via nselib.

        Tries several nselib function names because the API has changed
        across library versions.  Returns None if unavailable.
        """
        cm = _import_capital_market()
        if cm is None:
            return None

        candidates = ["india_vix", "indiaVix", "get_india_vix"]
        for fn_name in candidates:
            fn = getattr(cm, fn_name, None)
            if fn is None:
                continue
            try:
                result = await self._run_sync(fn)
                if result is None:
                    continue
                # Result may be a DataFrame, dict, or scalar.
                try:
                    import pandas as pd  # type: ignore[import]  # noqa: PLC0415, F401

                    if hasattr(result, "empty"):
                        if result.empty:
                            continue
                        # Take the last row's numeric value.
                        for col in result.columns:
                            try:
                                vix_val = float(result.iloc[-1][col])
                                if 5 < vix_val < 100:
                                    log.debug(
                                        "india_vix.source",
                                        source="nselib",
                                        fn=fn_name,
                                        vix=vix_val,
                                    )
                                    return vix_val
                            except (ValueError, TypeError):
                                pass
                        continue
                except ImportError:
                    pass

                vix = float(result)
                log.debug("india_vix.source", source="nselib", fn=fn_name, vix=vix)
                return vix
            except Exception as exc:  # noqa: BLE001
                log.debug("india_vix.nselib_fn_failed", fn=fn_name, error=str(exc))

        log.warning("india_vix_unavailable", tried=candidates)
        return None

    # ------------------------------------------------------------------
    # Thread executor helper
    # ------------------------------------------------------------------

    async def _run_sync(self, func: Any, *args: Any) -> Any:
        """Run a synchronous nselib call inside the shared thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, func, *args)

    # ------------------------------------------------------------------
    # NSEDirectClient dict → dataclass converters
    # ------------------------------------------------------------------

    def _dicts_to_bulk_deal_records(self, raw: list[dict]) -> list[BulkDealRecord]:
        """Convert NSEDirectClient deal dicts to ``BulkDealRecord`` dataclasses.

        Each dict from NSEDirectClient has keys:
          symbol, client_name, direction, quantity, price, deal_value_cr,
          deal_type, trade_date_str
        """
        records: list[BulkDealRecord] = []
        fetched_at = _now_utc()

        for item in raw:
            try:
                symbol         = str(item["symbol"]).strip().upper()
                client_name    = str(item.get("client_name") or "UNKNOWN").strip()
                direction      = str(item.get("direction") or "BUY").strip().upper()
                quantity       = int(item.get("quantity") or 0)
                price          = float(item.get("price") or 0.0)
                deal_value_cr  = float(item.get("deal_value_cr") or quantity * price / 1e7)
                deal_type      = str(item.get("deal_type") or "bulk").strip().lower()
                trade_date_str = str(item.get("trade_date_str") or _today_str())
                deal_date      = _midnight_utc(trade_date_str)

                composite_key = f"{symbol}|{trade_date_str}|{client_name}|{quantity}|{direction}"
                deal_id = make_id(composite_key)

                records.append(
                    BulkDealRecord(
                        id=deal_id,
                        deal_date=deal_date,
                        symbol=symbol,
                        client_name=client_name,
                        deal_type=deal_type,
                        direction=direction,
                        quantity=quantity,
                        price=price,
                        deal_value_cr=round(deal_value_cr, 4),
                        fetched_at=fetched_at,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("nselib_fetcher.direct.deal_row_error", error=str(exc))

        return records

    def _dicts_to_delivery_records(self, raw: list[dict]) -> list[DeliveryRecord]:
        """Convert NSEDirectClient delivery dicts to ``DeliveryRecord`` dataclasses.

        Each dict from NSEDirectClient has keys:
          symbol, delivery_qty, traded_qty, delivery_pct, trade_date_str
        """
        records: list[DeliveryRecord] = []
        fetched_at = _now_utc()

        for item in raw:
            try:
                symbol         = str(item["symbol"]).strip().upper()
                delivery_qty   = int(item.get("delivery_qty") or 0)
                traded_qty     = int(item.get("traded_qty") or delivery_qty or 1)
                delivery_pct   = float(item.get("delivery_pct") or (delivery_qty / max(traded_qty, 1) * 100))
                trade_date_str = str(item.get("trade_date_str") or _today_str())
                trade_date     = _midnight_utc(trade_date_str)

                records.append(
                    DeliveryRecord(
                        trade_date=trade_date,
                        symbol=symbol,
                        delivery_qty=delivery_qty,
                        traded_qty=traded_qty,
                        delivery_pct=round(delivery_pct, 4),
                        fetched_at=fetched_at,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("nselib_fetcher.direct.delivery_row_error", error=str(exc))

        return records

    # ------------------------------------------------------------------
    # nselib DataFrame → dataclass converters
    # ------------------------------------------------------------------

    def _df_to_bulk_deal_records(self, df: Any, deal_type: str) -> list[BulkDealRecord]:
        """Convert a nselib bulk/block deals DataFrame to ``BulkDealRecord`` objects."""
        records: list[BulkDealRecord] = []
        fetched_at = _now_utc()
        today_dt   = _midnight_utc(_today_str())

        try:
            if df is None or (hasattr(df, "empty") and df.empty):
                return []

            df.columns = [str(c).upper().strip() for c in df.columns]

            symbol_col    = _find_col(df, ["SYMBOL", "SCRIP_CD", "SYM"])
            client_col    = _find_col(df, ["CLIENT_NAME", "CLIENT", "CLIENTNAME"])
            direction_col = _find_col(df, ["BUY_SELL", "BUYSELL", "TRANSACTION_TYPE", "TXN_TYPE"])
            qty_col       = _find_col(df, ["QUANTITY_TRADED", "QUANTITY", "QTY", "SHARES"])
            price_col     = _find_col(df, ["TRADE_PRICE", "PRICE", "AVG_PRICE", "AVGPRICE"])

            if not symbol_col:
                log.warning(
                    "nselib_fetcher.nselib.bulk_no_symbol_col",
                    deal_type=deal_type,
                    cols=list(df.columns),
                )
                return []

            for _, row in df.iterrows():
                try:
                    symbol        = str(row[symbol_col]).strip().upper()
                    client        = str(row.get(client_col, "UNKNOWN") or "UNKNOWN").strip()
                    direction_raw = str(row.get(direction_col, "BUY") or "BUY").strip().upper()
                    direction     = "BUY" if direction_raw.startswith("B") else "SELL"
                    qty           = int(float(str(row.get(qty_col, 0) or 0).replace(",", "")))
                    price         = float(str(row.get(price_col, 0) or 0).replace(",", ""))
                    deal_value_cr = qty * price / 1e7

                    composite_key = f"{symbol}|{_today_str()}|{client}|{qty}|{direction}"
                    deal_id = make_id(composite_key)

                    records.append(
                        BulkDealRecord(
                            id=deal_id,
                            deal_date=today_dt,
                            symbol=symbol,
                            client_name=client,
                            deal_type=deal_type,
                            direction=direction,
                            quantity=qty,
                            price=price,
                            deal_value_cr=round(deal_value_cr, 4),
                            fetched_at=fetched_at,
                        )
                    )
                except Exception as row_exc:  # noqa: BLE001
                    log.debug(
                        "nselib_fetcher.nselib.bulk_row_error",
                        deal_type=deal_type,
                        error=str(row_exc),
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "nselib_fetcher.nselib.bulk_parse_failed",
                deal_type=deal_type,
                error=str(exc),
            )

        log.debug(
            "nselib_fetcher.nselib.deals_parsed",
            deal_type=deal_type,
            records=len(records),
        )
        return records

    def _df_to_delivery_records(self, df: Any) -> list[DeliveryRecord]:
        """Convert a nselib deliverable-positions DataFrame to ``DeliveryRecord`` objects."""
        records: list[DeliveryRecord] = []
        fetched_at = _now_utc()
        today_dt   = _midnight_utc(_today_str())

        try:
            if df is None or (hasattr(df, "empty") and df.empty):
                return []

            # Normalise column names to upper-case for consistent access.
            df.columns = [str(c).upper().strip() for c in df.columns]

            symbol_col     = _find_col(df, ["SYMBOL", "SYM"])
            deliv_qty_col  = _find_col(df, ["DELIV_QTY", "DELIVERABLE_QTY", "DELIVQTY"])
            deliv_pct_col  = _find_col(df, ["DELIV_PER", "DELIVERABLE_PER", "DELIVERY_PER", "DELIVPCT"])
            traded_qty_col = _find_col(df, ["TRADED_QTY", "TOTALTRADEDQUANTITY", "TRADEDQTY"])

            if not symbol_col:
                log.warning(
                    "nselib_fetcher.nselib.delivery_no_symbol_col",
                    cols=list(df.columns),
                )
                return []

            for _, row in df.iterrows():
                try:
                    symbol       = str(row[symbol_col]).strip().upper()
                    delivery_qty = int(float(row.get(deliv_qty_col, 0) or 0))
                    traded_qty   = int(float(row.get(traded_qty_col, delivery_qty) or delivery_qty))

                    if deliv_pct_col and row.get(deliv_pct_col) not in (None, ""):
                        delivery_pct = float(row[deliv_pct_col])
                    else:
                        delivery_pct = (
                            delivery_qty / traded_qty * 100
                            if traded_qty > 0 else 0.0
                        )

                    records.append(
                        DeliveryRecord(
                            trade_date=today_dt,
                            symbol=symbol,
                            delivery_qty=delivery_qty,
                            traded_qty=traded_qty,
                            delivery_pct=round(delivery_pct, 4),
                            fetched_at=fetched_at,
                        )
                    )
                except Exception as row_exc:  # noqa: BLE001
                    log.debug(
                        "nselib_fetcher.nselib.delivery_row_error",
                        error=str(row_exc),
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("nselib_fetcher.nselib.delivery_parse_failed", error=str(exc))

        log.debug("nselib_fetcher.nselib.delivery_parsed", records=len(records))
        return records
