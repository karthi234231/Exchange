"""
algo_trading/market_data/delivery_volume.py — NSE equity delivery volume data.

NSE publishes daily delivery quantity (shares that resulted in actual delivery
vs intraday squared off). High delivery % = conviction buying = bullish signal.

Also ingests bulk/block deals from NSE daily reports.

Delivery % formula: delivery_qty / total_traded_qty * 100
Interpretation:
  delivery_pct > 60%: high conviction, institutional accumulation likely
  delivery_pct < 20%: speculative, mostly intraday — treat with caution

Bulk deal: single trade > 0.5% of company shares in a session
Block deal: pre-arranged large trade on separate block deal window
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests
import structlog

log = structlog.get_logger(__name__)

# ── URL constants (ASSUMED — verify before production use) ────────────────────

# ASSUMED: NSE delivery data endpoint for a specific symbol.
# Actual URL may require cookie-based session; check NSE website before use.
DELIVERY_URL_BASE = (
    "https://www.nseindia.com/api/deliveryAndSurveillanceReport"
)

# ASSUMED: NSE bulk deals endpoint.
BULK_DEALS_URL_BASE = (
    "https://www.nseindia.com/api/snapshot-capital-market-largedeals-bulkdeals"
)

# ASSUMED: NSE block deals endpoint.
BLOCK_DEALS_URL_BASE = (
    "https://www.nseindia.com/api/snapshot-capital-market-largedeals-blockdeals"
)

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

_HTTP_TIMEOUT: int = 10  # seconds

# Delivery percentage thresholds
_HIGH_DELIVERY_THRESHOLD: float = 50.0   # pct >= this → is_high_delivery=True
_ACCUMULATION_THRESHOLD: float  = 60.0   # signal threshold for accumulation
_DISTRIBUTION_THRESHOLD: float  = 25.0   # signal threshold for distribution

# Minimum records required to emit a non-neutral signal
_MIN_RECORDS_FOR_SIGNAL: int = 3


@dataclass
class DeliveryRecord:
    """Daily delivery data for a single symbol.

    Attributes:
        date: Trading date.
        symbol: NSE trading symbol, e.g. ``"RELIANCE"``.
        delivery_qty: Number of shares that resulted in delivery.
        total_qty: Total traded quantity (delivery + intraday).
        delivery_pct: Delivery as percentage of total traded quantity.
        is_high_delivery: ``True`` when ``delivery_pct >= 50%``.
    """

    date: date
    symbol: str
    delivery_qty: int
    total_qty: int
    delivery_pct: float
    is_high_delivery: bool


@dataclass
class BulkDealRecord:
    """Single bulk or block deal entry.

    Attributes:
        date: Date of the deal.
        symbol: NSE trading symbol.
        client_name: Name of the client / institution.
        buy_sell: ``"BUY"`` or ``"SELL"``.
        qty: Number of shares transacted.
        price: Trade price per share in INR.
        deal_type: ``"bulk"`` or ``"block"``.
    """

    date: date
    symbol: str
    client_name: str
    buy_sell: str
    qty: int
    price: float
    deal_type: str


class DeliveryFetcher:
    """Fetches delivery volume and bulk/block deal data from NSE.

    All fetch methods return empty lists on failure rather than raising
    exceptions, consistent with the graceful-degradation pattern used
    across the market data layer.

    Example usage::

        fetcher = DeliveryFetcher()
        records = fetcher.fetch_delivery("RELIANCE", n_days=30)
        signal  = fetcher.delivery_signal(records)
        bulk    = fetcher.fetch_bulk_deals(n_days=7)
    """

    def fetch_delivery(
        self,
        symbol: str,
        n_days: int = 30,
    ) -> list[DeliveryRecord]:
        """Fetch delivery volume data for a given symbol.

        Args:
            symbol: NSE equity symbol, e.g. ``"RELIANCE"``.
            n_days: Number of recent trading days to request.

        Returns:
            List of :class:`DeliveryRecord` sorted oldest-first.
            Returns an empty list on any fetch failure.
        """
        symbol = symbol.upper().strip()
        url = f"{DELIVERY_URL_BASE}?symbol={symbol}&series=EQ"
        log.info("delivery.fetch_start", symbol=symbol, n_days=n_days, url=url)

        payload = self._get_json(url, context="delivery")
        if payload is None:
            return []

        records = self._parse_delivery_payload(payload, symbol, n_days)
        log.info("delivery.fetch_complete", symbol=symbol, records=len(records))
        return records

    def fetch_bulk_deals(self, n_days: int = 7) -> list[BulkDealRecord]:
        """Fetch recent bulk and block deal data from NSE.

        Attempts both the bulk-deals and block-deals endpoints and merges
        the results.

        Args:
            n_days: Number of recent days to filter deals by.

        Returns:
            List of :class:`BulkDealRecord` sorted oldest-first.
            Returns an empty list on any fetch failure.
        """
        log.info("delivery.bulk_deals_fetch_start", n_days=n_days)

        bulk_payload = self._get_json(BULK_DEALS_URL_BASE, context="bulk_deals")
        block_payload = self._get_json(BLOCK_DEALS_URL_BASE, context="block_deals")

        records: list[BulkDealRecord] = []

        if bulk_payload is not None:
            records.extend(
                self._parse_deal_payload(bulk_payload, deal_type="bulk", n_days=n_days)
            )
        if block_payload is not None:
            records.extend(
                self._parse_deal_payload(block_payload, deal_type="block", n_days=n_days)
            )

        records.sort(key=lambda r: r.date)
        log.info("delivery.bulk_deals_fetch_complete", records=len(records))
        return records

    def delivery_signal(self, records: list[DeliveryRecord]) -> str:
        """Classify delivery pattern as accumulation, distribution, or neutral.

        Uses the average delivery percentage across all records.

        - avg_delivery_pct >= 60% → ``"accumulation"`` (institutional buying)
        - avg_delivery_pct <= 25% → ``"distribution"`` (speculative / selling)
        - Otherwise              → ``"neutral"``

        Args:
            records: Delivery records for the symbol (any sort order).

        Returns:
            One of ``"accumulation"``, ``"distribution"``, or ``"neutral"``.
        """
        if len(records) < _MIN_RECORDS_FOR_SIGNAL:
            log.debug(
                "delivery.signal_insufficient_data",
                n=len(records),
                min_required=_MIN_RECORDS_FOR_SIGNAL,
            )
            return "neutral"

        avg_pct = sum(r.delivery_pct for r in records) / len(records)
        high_delivery_days = sum(1 for r in records if r.is_high_delivery)
        high_delivery_ratio = high_delivery_days / len(records)

        log.debug(
            "delivery.signal_computed",
            avg_delivery_pct=round(avg_pct, 2),
            high_delivery_days=high_delivery_days,
            high_delivery_ratio=round(high_delivery_ratio, 3),
        )

        if avg_pct >= _ACCUMULATION_THRESHOLD:
            return "accumulation"
        if avg_pct <= _DISTRIBUTION_THRESHOLD:
            return "distribution"
        return "neutral"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_json(self, url: str, context: str) -> Any | None:
        """Perform an HTTP GET and return parsed JSON, or None on failure.

        Args:
            url: Full URL to request.
            context: Short label used in log messages for traceability.

        Returns:
            Parsed JSON payload or ``None`` on any error.
        """
        try:
            response = requests.get(
                url,
                headers=_BROWSER_HEADERS,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as exc:
            log.warning(
                "delivery.connection_error",
                context=context,
                url=url,
                error=str(exc),
            )
            return None
        except requests.exceptions.Timeout:
            log.warning(
                "delivery.timeout",
                context=context,
                url=url,
                timeout_sec=_HTTP_TIMEOUT,
            )
            return None
        except requests.exceptions.RequestException as exc:
            log.warning("delivery.request_error", context=context, url=url, error=str(exc))
            return None

        if response.status_code == 403:
            log.warning(
                "delivery.http_403",
                context=context,
                url=url,
                note="NSE blocking scraper — browser session may be required",
            )
            return None

        if response.status_code != 200:
            log.warning(
                "delivery.http_error",
                context=context,
                url=url,
                status_code=response.status_code,
            )
            return None

        try:
            return response.json()
        except ValueError as exc:
            log.warning(
                "delivery.json_parse_error",
                context=context,
                url=url,
                error=str(exc),
            )
            return None

    def _parse_delivery_payload(
        self,
        payload: Any,
        symbol: str,
        n_days: int,
    ) -> list[DeliveryRecord]:
        """Parse NSE delivery payload into :class:`DeliveryRecord` objects.

        Args:
            payload: Raw JSON from NSE.
            symbol: Symbol name (used to fill the record).
            n_days: Limit results to the most recent ``n_days`` records.

        Returns:
            List of :class:`DeliveryRecord` sorted oldest-first.
        """
        if isinstance(payload, dict):
            # NSE often wraps data in {"data": [...]}
            items = payload.get("data") or payload.get("deliveryData") or []
        elif isinstance(payload, list):
            items = payload
        else:
            log.warning("delivery.unexpected_payload_type", type=type(payload).__name__)
            return []

        records: list[DeliveryRecord] = []
        for item in items:
            try:
                record = self._parse_delivery_item(item, symbol)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                log.debug(
                    "delivery.item_parse_skip",
                    item=str(item)[:120],
                    error=str(exc),
                )
                continue

        records.sort(key=lambda r: r.date)
        return records[-n_days:] if n_days > 0 else records

    def _parse_delivery_item(
        self, item: dict[str, Any], symbol: str
    ) -> DeliveryRecord | None:
        """Parse a single delivery data dict.

        Args:
            item: Raw dict from NSE payload.
            symbol: Fallback symbol if not present in the item.

        Returns:
            Parsed :class:`DeliveryRecord` or ``None`` if unparseable.
        """
        if not isinstance(item, dict):
            return None

        date_str = (
            item.get("date") or item.get("Date")
            or item.get("tradeDate") or item.get("CH_TIMESTAMP", "")
        )
        if not date_str:
            return None
        trade_date = _parse_date(str(date_str))
        if trade_date is None:
            return None

        sym = (
            item.get("symbol") or item.get("Symbol")
            or item.get("CH_SYMBOL") or symbol
        )

        delivery_qty = _parse_int(
            item.get("deliverableQty")
            or item.get("CH_DELIVERABLE_QTY") or 0
        )
        total_qty = _parse_int(
            item.get("totalTradedQuantity")
            or item.get("CH_TOT_TRADED_QTY") or 0
        )

        if total_qty > 0:
            delivery_pct = (delivery_qty / total_qty) * 100.0
        else:
            delivery_pct = _parse_float(
                item.get("deliveryToTradedQty")
                or item.get("CH_DELIVERABLE_QTY_TO_TRADED_QTY") or 0.0
            )

        return DeliveryRecord(
            date=trade_date,
            symbol=str(sym).upper(),
            delivery_qty=delivery_qty,
            total_qty=total_qty,
            delivery_pct=round(delivery_pct, 2),
            is_high_delivery=delivery_pct >= _HIGH_DELIVERY_THRESHOLD,
        )

    def _parse_deal_payload(
        self,
        payload: Any,
        deal_type: str,
        n_days: int,
    ) -> list[BulkDealRecord]:
        """Parse bulk or block deal payload.

        Args:
            payload: Raw JSON from NSE.
            deal_type: ``"bulk"`` or ``"block"``.
            n_days: Limit results to the most recent ``n_days`` calendar days.

        Returns:
            List of :class:`BulkDealRecord`.
        """
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("BULKDEALS") or []
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        records: list[BulkDealRecord] = []
        for item in items:
            try:
                record = self._parse_deal_item(item, deal_type)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                log.debug(
                    "delivery.deal_item_parse_skip",
                    item=str(item)[:120],
                    error=str(exc),
                )
                continue

        if not records:
            return records

        records.sort(key=lambda r: r.date)
        if n_days > 0 and records:
            cutoff = records[-1].date
            from datetime import timedelta
            window_start = cutoff - timedelta(days=n_days)
            records = [r for r in records if r.date >= window_start]

        return records

    def _parse_deal_item(
        self, item: dict[str, Any], deal_type: str
    ) -> BulkDealRecord | None:
        """Parse a single deal dict into a :class:`BulkDealRecord`.

        Args:
            item: Raw dict from NSE.
            deal_type: ``"bulk"`` or ``"block"``.

        Returns:
            Parsed :class:`BulkDealRecord` or ``None``.
        """
        if not isinstance(item, dict):
            return None

        date_str = (
            item.get("date") or item.get("Date")
            or item.get("BD_DT_DATE") or item.get("BD_DT", "")
        )
        if not date_str:
            return None
        trade_date = _parse_date(str(date_str))
        if trade_date is None:
            return None

        symbol = (
            item.get("symbol") or item.get("Symbol")
            or item.get("BD_SYMBOL") or ""
        )
        client = (
            item.get("clientName") or item.get("ClientName")
            or item.get("BD_CLIENT_NAME") or "UNKNOWN"
        )
        buy_sell = (
            item.get("buySell") or item.get("BuySell")
            or item.get("BD_BUY_SELL") or "BUY"
        )
        qty   = _parse_int(item.get("quantity") or item.get("BD_QTY_TRD") or 0)
        price = _parse_float(item.get("price") or item.get("BD_TP_WATP") or 0.0)

        return BulkDealRecord(
            date=trade_date,
            symbol=str(symbol).upper(),
            client_name=str(client).strip(),
            buy_sell=str(buy_sell).upper().strip(),
            qty=qty,
            price=price,
            deal_type=deal_type,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_date(s: str) -> date | None:
    """Parse common NSE date string formats.

    Args:
        s: Raw date string.

    Returns:
        Parsed :class:`datetime.date` or ``None``.
    """
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    log.debug("delivery.date_parse_fail", raw=s)
    return None


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion from raw NSE payload values.

    Args:
        value: Raw value (str, int, float, or None).
        default: Return value on parse failure.

    Returns:
        Parsed float or ``default``.
    """
    if value is None:
        return default
    try:
        v = float(str(value).replace(",", "").strip())
        return v if not math.isnan(v) else default
    except (ValueError, TypeError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    """Safe integer conversion from raw NSE payload values.

    Args:
        value: Raw value.
        default: Return value on parse failure.

    Returns:
        Parsed int or ``default``.
    """
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default
