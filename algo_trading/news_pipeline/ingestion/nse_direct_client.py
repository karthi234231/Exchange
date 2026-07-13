"""
algo_trading.news_pipeline.ingestion.nse_direct_client — Direct NSE API client.

Replaces nselib and jugaad-data for NSE structured market data.  Uses direct
aiohttp calls to NSE's semi-public JSON API endpoints via NSESession (which
handles cookie management, UA rotation and rate limiting).

Endpoints used
--------------
  Homepage warm-up:    https://www.nseindia.com/
  Quote warm-up:       https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE
  Bulk deals:          https://www.nseindia.com/api/historical/bulk-deals?from=...&to=...
  Block deals:         https://www.nseindia.com/api/historical/block-deals?from=...&to=...
  VIX + all indices:   https://www.nseindia.com/api/allIndices
  Corporate actions:   https://www.nseindia.com/api/corporate-announcements?index=equities&period=3
  Deliverables (MTO):  https://www.nseindia.com/api/historical/deliverable?from=...&to=...

Response contract
-----------------
All methods return [] / None on failure.  They never raise — callers can depend
on this for fail-open behaviour.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional

import structlog

from algo_trading.news_pipeline.ingestion.http_session import (
    CircuitBreaker,
    CircuitOpenError,
    NSESession,
)

log = structlog.get_logger(__name__)

_NSE_BASE = "https://www.nseindia.com"

# Date format NSE APIs accept: DD-Mon-YYYY (e.g. "20-Jun-2026")
_NSE_DATE_FMT = "%d-%b-%Y"


def _fmt_date(d: date) -> str:
    """Format a date object as "DD-Mon-YYYY" for NSE API params."""
    return d.strftime(_NSE_DATE_FMT)


def _parse_date_flexible(raw: str) -> datetime:
    """
    Parse NSE date strings which can be:
      DD-Mon-YYYY, DD-MM-YYYY, YYYY-MM-DD, DD-Mon-YYYY HH:MM:SS,
      ISO-8601 with offset (2026-06-20T09:15:32+05:30 or Z suffix).
    Returns UTC datetime; falls back to utcnow() on failure.
    """
    raw = str(raw).strip()
    if not raw:
        return datetime.now(tz=timezone.utc)

    # Handle ISO-8601 with numeric TZ offset like +05:30
    # Python <3.11 fromisoformat doesn't handle these — replace manually.
    iso_try = raw.replace(" ", "T")
    try:
        return datetime.fromisoformat(iso_try.replace("Z", "+00:00"))
    except ValueError:
        pass

    for fmt in (
        "%d-%b-%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d-%b-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime.now(tz=timezone.utc)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default


class NSEDirectClient:
    """
    Async client for NSE's internal JSON APIs.

    Instantiate once and call connect() before use.  The client owns an
    NSESession (cookie management + UA rotation) and a CircuitBreaker per
    logical API group.

    All public methods return [] or None on failure.
    """

    def __init__(self, timeout_sec: float = 15.0) -> None:
        self._session     = NSESession(timeout_sec=timeout_sec)
        self._cb_deals    = CircuitBreaker(source="nse_deals")
        self._cb_vix      = CircuitBreaker(source="nse_vix")
        self._cb_corp     = CircuitBreaker(source="nse_corp")
        self._cb_delivery = CircuitBreaker(source="nse_delivery")

    async def connect(self) -> None:
        """Warm up NSE session cookies. Call once before any API requests."""
        await self._session.connect()

    async def close(self) -> None:
        await self._session.close()

    # ------------------------------------------------------------------
    # Bulk deals
    # ------------------------------------------------------------------

    async def get_bulk_deals(self, trade_date: date) -> list[dict]:
        """
        Fetch bulk deals for *trade_date*.

        Returns a list of dicts with keys:
          symbol, client_name, direction ("BUY"/"SELL"),
          quantity, price, deal_value_cr, trade_date_str
        """
        if not self._cb_deals.allow():
            log.warning("nse_direct.bulk_deals.circuit_open")
            return []

        url = f"{_NSE_BASE}/api/historical/bulk-deals"
        params = {"from": _fmt_date(trade_date), "to": _fmt_date(trade_date)}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning(
                        "nse_direct.bulk_deals.bad_status",
                        status=resp.status,
                        date=_fmt_date(trade_date),
                    )
                    self._cb_deals.record_failure()
                    return []
                raw = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("nse_direct.bulk_deals.fetch_failed", error=str(exc))
            self._cb_deals.record_failure()
            return []

        records = self._parse_deal_response(raw, "bulk", trade_date)
        self._cb_deals.record_success()
        log.debug(
            "nse_direct.bulk_deals.fetched",
            date=_fmt_date(trade_date),
            records=len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Block deals
    # ------------------------------------------------------------------

    async def get_block_deals(self, trade_date: date) -> list[dict]:
        """Fetch block deals for *trade_date*. Same schema as bulk deals."""
        if not self._cb_deals.allow():
            log.warning("nse_direct.block_deals.circuit_open")
            return []

        url = f"{_NSE_BASE}/api/historical/block-deals"
        params = {"from": _fmt_date(trade_date), "to": _fmt_date(trade_date)}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning(
                        "nse_direct.block_deals.bad_status",
                        status=resp.status,
                        date=_fmt_date(trade_date),
                    )
                    self._cb_deals.record_failure()
                    return []
                raw = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("nse_direct.block_deals.fetch_failed", error=str(exc))
            self._cb_deals.record_failure()
            return []

        records = self._parse_deal_response(raw, "block", trade_date)
        self._cb_deals.record_success()
        log.debug(
            "nse_direct.block_deals.fetched",
            date=_fmt_date(trade_date),
            records=len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Deliverable positions
    # ------------------------------------------------------------------

    async def get_delivery_data(self, trade_date: date) -> list[dict]:
        """
        Fetch deliverable position data for *trade_date*.

        Returns list of dicts with keys:
          symbol, delivery_qty, traded_qty, delivery_pct
        """
        if not self._cb_delivery.allow():
            log.warning("nse_direct.delivery.circuit_open")
            return []

        url = f"{_NSE_BASE}/api/historical/deliverable"
        params = {
            "from": _fmt_date(trade_date),
            "to":   _fmt_date(trade_date),
            "type": "equities",
        }

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 404:
                    # Data may not be available for non-trading days; not an error.
                    log.debug("nse_direct.delivery.not_found", date=_fmt_date(trade_date))
                    return []
                if resp.status != 200:
                    log.warning(
                        "nse_direct.delivery.bad_status",
                        status=resp.status,
                        date=_fmt_date(trade_date),
                    )
                    self._cb_delivery.record_failure()
                    return []
                raw = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("nse_direct.delivery.fetch_failed", error=str(exc))
            self._cb_delivery.record_failure()
            return []

        records = self._parse_delivery_response(raw, trade_date)
        self._cb_delivery.record_success()
        log.debug(
            "nse_direct.delivery.fetched",
            date=_fmt_date(trade_date),
            records=len(records),
        )
        return records

    # ------------------------------------------------------------------
    # India VIX
    # ------------------------------------------------------------------

    async def get_india_vix(self) -> Optional[float]:
        """
        Fetch India VIX from the allIndices endpoint.

        Returns the VIX float value (typically 10-50) or None on failure.
        """
        if not self._cb_vix.allow():
            log.warning("nse_direct.vix.circuit_open")
            return None

        url = f"{_NSE_BASE}/api/allIndices"

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    log.warning("nse_direct.vix.bad_status", status=resp.status)
                    self._cb_vix.record_failure()
                    return None
                raw = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("nse_direct.vix.fetch_failed", error=str(exc))
            self._cb_vix.record_failure()
            return None

        vix = self._parse_vix(raw)
        if vix is not None:
            self._cb_vix.record_success()
            log.debug("nse_direct.vix.fetched", vix=vix)
        else:
            log.warning("nse_direct.vix.not_found_in_response")
        return vix

    # ------------------------------------------------------------------
    # Corporate announcements
    # ------------------------------------------------------------------

    async def get_corporate_announcements(self, days: int = 3) -> list[dict]:
        """
        Fetch NSE corporate announcements for the last *days* days.

        Returns list of dicts with keys:
          symbol, subject, body, announcement_date
        """
        if not self._cb_corp.allow():
            log.warning("nse_direct.corp_ann.circuit_open")
            return []

        url = f"{_NSE_BASE}/api/corporate-announcements"
        params = {"index": "equities", "period": str(days)}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning("nse_direct.corp_ann.bad_status", status=resp.status)
                    self._cb_corp.record_failure()
                    return []
                raw = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("nse_direct.corp_ann.fetch_failed", error=str(exc))
            self._cb_corp.record_failure()
            return []

        records = self._parse_corp_announcements(raw)
        self._cb_corp.record_success()
        log.debug("nse_direct.corp_ann.fetched", records=len(records))
        return records

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_deal_response(raw: Any, deal_type: str, trade_date: date) -> list[dict]:
        """
        Parse bulk/block deal API response.

        NSE response shapes seen in the wild:
          - {"data": [...]}
          - [...]          (direct list)
          - {"BULK_DEALS_DATA": [...]}
        """
        data: Any = raw
        if isinstance(raw, dict):
            data = (
                raw.get("data")
                or raw.get("BULK_DEALS_DATA")
                or raw.get("BLOCK_DEALS_DATA")
                or raw.get("deals")
                or []
            )
        if not isinstance(data, list):
            log.warning(
                "nse_direct.deal_parse.unexpected_shape",
                deal_type=deal_type,
                type=type(data).__name__,
            )
            return []

        trade_date_str = _fmt_date(trade_date)
        records: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                symbol = str(
                    item.get("symbol")
                    or item.get("Symbol")
                    or item.get("scrips")
                    or item.get("SYMBOL")
                    or ""
                ).strip().upper()
                if not symbol:
                    continue

                client = str(
                    item.get("client")
                    or item.get("clientName")
                    or item.get("CLIENT_NAME")
                    or item.get("Client")
                    or "UNKNOWN"
                ).strip()

                raw_dir = str(
                    item.get("buySellType")
                    or item.get("BUY_SELL")
                    or item.get("buyOrSell")
                    or item.get("BS")
                    or "BUY"
                ).strip().upper()
                direction = "BUY" if raw_dir.startswith("B") else "SELL"

                qty = _safe_int(
                    item.get("quantity")
                    or item.get("QUANTITY")
                    or item.get("qty")
                    or item.get("QTY")
                    or 0
                )
                price = _safe_float(
                    item.get("price")
                    or item.get("PRICE")
                    or item.get("tradePrice")
                    or item.get("TRADE_PRICE")
                    or 0
                )
                deal_value_cr = qty * price / 1e7

                records.append(
                    {
                        "symbol":         symbol,
                        "client_name":    client,
                        "direction":      direction,
                        "quantity":       qty,
                        "price":          price,
                        "deal_value_cr":  round(deal_value_cr, 4),
                        "deal_type":      deal_type,
                        "trade_date_str": trade_date_str,
                    }
                )
            except Exception as exc:
                log.debug(
                    "nse_direct.deal_parse.row_error",
                    deal_type=deal_type,
                    error=str(exc),
                )

        return records

    @staticmethod
    def _parse_delivery_response(raw: Any, trade_date: date) -> list[dict]:
        """Parse deliverable positions API response."""
        data: Any = raw
        if isinstance(raw, dict):
            data = (
                raw.get("data")
                or raw.get("deliverable")
                or raw.get("deliverables")
                or raw.get("DELIVERABLE")
                or []
            )
        if not isinstance(data, list):
            log.warning(
                "nse_direct.delivery_parse.unexpected_shape",
                type=type(data).__name__,
            )
            return []

        trade_date_str = _fmt_date(trade_date)
        records: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                symbol = str(
                    item.get("symbol")
                    or item.get("SYMBOL")
                    or item.get("scrips")
                    or ""
                ).strip().upper()
                if not symbol:
                    continue

                delivery_qty = _safe_int(
                    item.get("delivQty")
                    or item.get("DELIV_QTY")
                    or item.get("deliverable_quantity")
                    or 0
                )
                traded_qty = _safe_int(
                    item.get("tradedQty")
                    or item.get("TRADED_QTY")
                    or item.get("totalTradedQuantity")
                    or delivery_qty
                    or 1
                )
                if traded_qty == 0:
                    traded_qty = 1  # guard division by zero

                raw_pct = (
                    item.get("delivPercentage")
                    or item.get("DELIV_PER")
                    or item.get("delivPer")
                    or None
                )
                if raw_pct is not None and str(raw_pct).strip():
                    delivery_pct = _safe_float(raw_pct)
                else:
                    delivery_pct = delivery_qty / traded_qty * 100

                records.append(
                    {
                        "symbol":         symbol,
                        "delivery_qty":   delivery_qty,
                        "traded_qty":     traded_qty,
                        "delivery_pct":   round(delivery_pct, 4),
                        "trade_date_str": trade_date_str,
                    }
                )
            except Exception as exc:
                log.debug("nse_direct.delivery_parse.row_error", error=str(exc))

        return records

    @staticmethod
    def _parse_vix(raw: Any) -> Optional[float]:
        """Extract India VIX float from the allIndices response."""
        data: Any = raw
        if isinstance(raw, dict):
            data = raw.get("data") or raw.get("indices") or []
        if not isinstance(data, list):
            return None

        for idx in data:
            if not isinstance(idx, dict):
                continue
            name = str(idx.get("indexSymbol") or idx.get("index") or "").upper()
            if "VIX" in name and "INDIA" in name:
                try:
                    return float(idx.get("last") or idx.get("lastPrice") or 0)
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _parse_corp_announcements(raw: Any) -> list[dict]:
        """Parse corporate announcements response."""
        data: Any = raw
        if isinstance(raw, dict):
            data = raw.get("data") or raw.get("announcements") or []
        if not isinstance(data, list):
            return []

        records: list[dict] = []
        for ann in data:
            if not isinstance(ann, dict):
                continue
            try:
                symbol = str(
                    ann.get("symbol")
                    or ann.get("sym")
                    or ann.get("SYMBOL")
                    or ""
                ).strip().upper()
                if not symbol:
                    continue

                subject = str(
                    ann.get("desc")
                    or ann.get("subject")
                    or ann.get("Subject")
                    or ""
                ).strip()

                body_raw = str(
                    ann.get("attchmntText")
                    or ann.get("attachment")
                    or ann.get("body")
                    or ""
                ).strip()
                # Truncate at sentence boundary up to 500 chars
                body = body_raw[:500]
                if len(body_raw) > 500:
                    last_period = body.rfind(".")
                    if last_period > 100:
                        body = body[: last_period + 1]

                date_raw = str(
                    ann.get("an_dt")
                    or ann.get("date")
                    or ann.get("announcementDate")
                    or ""
                ).strip()
                announcement_date = _parse_date_flexible(date_raw)

                records.append(
                    {
                        "symbol":            symbol,
                        "subject":           subject,
                        "body":              body,
                        "announcement_date": announcement_date,
                    }
                )
            except Exception as exc:
                log.debug("nse_direct.corp_ann.row_error", error=str(exc))

        return records
