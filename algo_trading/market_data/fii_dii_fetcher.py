"""
algo_trading/market_data/fii_dii_fetcher.py — FII/DII institutional flow data.

Downloads FII (Foreign Institutional Investor) and DII (Domestic Institutional
Investor) daily net buy/sell data from NSE India's publicly available reports.

Data is used as a macro sentiment filter:
  - FII net buyers for 5+ consecutive days → bullish macro regime
  - FII net sellers for 5+ consecutive days → bearish macro regime
  - Extreme FII selling (> 2 std dev) → elevated market risk

WARNING: NSE changes its data formats and URLs frequently. All URLs marked
ASSUMED — verify against https://www.nseindia.com/market-data/fii-dii-activity
before production use.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests
import structlog

log = structlog.get_logger(__name__)

# ── URL constants (ASSUMED — verify before production use) ────────────────────

# ASSUMED: NSE FII/DII participation data endpoint.
# Actual URL may differ; check https://www.nseindia.com/market-data/fii-dii-activity
FII_DII_URL_BASE = (
    "https://www.nseindia.com/api/fiidiiTradeReact"
)

# Request headers that mimic a browser — NSE blocks bare requests without
# a plausible User-Agent.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/market-data/fii-dii-activity",
}

_HTTP_TIMEOUT: int = 10  # seconds

# Thresholds for regime classification
_REGIME_LOOKBACK_DAYS: int = 5
_BULLISH_THRESHOLD_INR: float = 0.0   # FII cumulative net > 0 over lookback
_BEARISH_THRESHOLD_INR: float = 0.0   # FII cumulative net < 0 over lookback


@dataclass
class FIIDIIRecord:
    """Single day FII/DII participation record.

    All monetary values are in INR crore (1 crore = 10 million INR).

    Attributes:
        date: Trading date for this record.
        fii_net_inr_crore: FII net buy/sell (positive = net buyer).
        dii_net_inr_crore: DII net buy/sell (positive = net buyer).
        fii_gross_buy: FII total gross purchase value in INR crore.
        fii_gross_sell: FII total gross sale value in INR crore.
        dii_gross_buy: DII total gross purchase value in INR crore.
        dii_gross_sell: DII total gross sale value in INR crore.
    """

    date: date
    fii_net_inr_crore: float
    dii_net_inr_crore: float
    fii_gross_buy: float
    fii_gross_sell: float
    dii_gross_buy: float
    dii_gross_sell: float


class FIIDIIFetcher:
    """Fetches and interprets FII/DII institutional flow data from NSE.

    NSE frequently blocks automated scrapers with HTTP 403 responses.
    All fetch methods degrade gracefully — returning an empty list with a
    log warning rather than raising an exception.

    Example usage::

        fetcher = FIIDIIFetcher()
        records = fetcher.fetch_recent(n_days=30)
        regime  = fetcher.compute_regime(records)
        spiking = fetcher.is_extreme_selling(records)
    """

    def fetch_recent(self, n_days: int = 30) -> list[FIIDIIRecord]:
        """Fetch recent FII/DII participation data from NSE.

        Attempts to download from the NSE website. If the request fails
        for any reason (HTTP 403, 500, connection error, JSON parse error),
        logs a warning and returns an empty list — does NOT raise.

        Args:
            n_days: Number of recent trading days to request.

        Returns:
            List of :class:`FIIDIIRecord` sorted oldest-first.
            Returns an empty list on any fetch failure.
        """
        url = f"{FII_DII_URL_BASE}?category=fiiDii&numberOfDays={n_days}"
        log.info("fii_dii.fetch_start", url=url, n_days=n_days)

        try:
            response = requests.get(
                url,
                headers=_BROWSER_HEADERS,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as exc:
            log.warning(
                "fii_dii.connection_error",
                url=url,
                error=str(exc),
                note="NSE may be unreachable or blocking the request",
            )
            return []
        except requests.exceptions.Timeout:
            log.warning("fii_dii.timeout", url=url, timeout_sec=_HTTP_TIMEOUT)
            return []
        except requests.exceptions.RequestException as exc:
            log.warning("fii_dii.request_error", url=url, error=str(exc))
            return []

        if response.status_code == 403:
            log.warning(
                "fii_dii.http_403",
                url=url,
                note="NSE is blocking the scraper — browser session/cookie may be required",
            )
            return []

        if response.status_code == 500:
            log.warning("fii_dii.http_500", url=url)
            return []

        if response.status_code != 200:
            log.warning(
                "fii_dii.http_error",
                url=url,
                status_code=response.status_code,
            )
            return []

        try:
            payload: Any = response.json()
        except ValueError as exc:
            log.warning("fii_dii.json_parse_error", url=url, error=str(exc))
            return []

        records = self._parse_payload(payload)
        log.info("fii_dii.fetch_complete", records=len(records))
        return records

    def compute_regime(self, records: list[FIIDIIRecord]) -> str:
        """Classify the current macro regime based on FII flow.

        Uses the last :data:`_REGIME_LOOKBACK_DAYS` records to determine
        the cumulative FII net position.

        - Cumulative net > 0 over lookback window → ``"fii_bullish"``
        - Cumulative net < 0 over lookback window → ``"fii_bearish"``
        - Otherwise → ``"neutral"``

        Args:
            records: List of FII/DII records (any order accepted; sorted
                internally).

        Returns:
            One of ``"fii_bullish"``, ``"fii_bearish"``, or ``"neutral"``.
        """
        if not records:
            log.debug("fii_dii.regime_no_data")
            return "neutral"

        sorted_records = sorted(records, key=lambda r: r.date)
        recent = sorted_records[-_REGIME_LOOKBACK_DAYS:]

        cumulative_net = sum(r.fii_net_inr_crore for r in recent)
        log.debug(
            "fii_dii.regime_computed",
            lookback_days=len(recent),
            cumulative_net_crore=cumulative_net,
        )

        if cumulative_net > _BULLISH_THRESHOLD_INR:
            return "fii_bullish"
        if cumulative_net < _BEARISH_THRESHOLD_INR:
            return "fii_bearish"
        return "neutral"

    def is_extreme_selling(
        self,
        records: list[FIIDIIRecord],
        lookback: int = 60,
        z_thresh: float = 2.0,
    ) -> bool:
        """Detect statistically extreme FII selling activity.

        Computes the z-score of the most recent FII net value against the
        historical distribution over ``lookback`` days.  If the z-score is
        below ``-z_thresh`` the most recent session represents extreme
        selling relative to recent history.

        Args:
            records: FII/DII records sorted oldest-first.  Requires at least
                3 records for a meaningful standard deviation.
            lookback: Number of historical days to use for mean/std
                computation (default 60).
            z_thresh: Z-score magnitude threshold for flagging extreme
                selling (default 2.0).

        Returns:
            ``True`` if the latest session shows extreme FII selling.
            ``False`` if there is insufficient data or no extreme event.
        """
        if len(records) < 3:
            log.debug("fii_dii.extreme_selling_insufficient_data", n=len(records))
            return False

        sorted_records = sorted(records, key=lambda r: r.date)
        history = sorted_records[-lookback:]

        if len(history) < 3:
            return False

        net_values = [r.fii_net_inr_crore for r in history]
        latest = net_values[-1]
        historical = net_values[:-1]

        try:
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical)
        except statistics.StatisticsError:
            return False

        if stdev == 0.0:
            return False

        z_score = (latest - mean) / stdev
        is_extreme = z_score < -z_thresh

        log.debug(
            "fii_dii.extreme_selling_check",
            latest_net=latest,
            mean=round(mean, 2),
            stdev=round(stdev, 2),
            z_score=round(z_score, 3),
            is_extreme=is_extreme,
            z_thresh=z_thresh,
        )
        return is_extreme

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_payload(self, payload: Any) -> list[FIIDIIRecord]:
        """Parse the raw JSON payload from NSE into :class:`FIIDIIRecord` objects.

        The expected structure is a list of dicts.  Unknown formats are
        logged and skipped rather than raising exceptions.

        Args:
            payload: Parsed JSON response from NSE API.

        Returns:
            List of :class:`FIIDIIRecord` sorted oldest-first.
        """
        if not isinstance(payload, list):
            log.warning(
                "fii_dii.unexpected_payload_type",
                payload_type=type(payload).__name__,
            )
            return []

        records: list[FIIDIIRecord] = []
        for item in payload:
            try:
                record = self._parse_item(item)
                if record is not None:
                    records.append(record)
            except Exception as exc:
                log.debug("fii_dii.item_parse_skip", item=str(item)[:120], error=str(exc))
                continue

        records.sort(key=lambda r: r.date)
        return records

    def _parse_item(self, item: dict[str, Any]) -> FIIDIIRecord | None:
        """Parse a single item dict into a :class:`FIIDIIRecord`.

        Args:
            item: Raw dict from NSE JSON payload.

        Returns:
            Parsed record or ``None`` if the item cannot be parsed.
        """
        if not isinstance(item, dict):
            return None

        date_str = item.get("date") or item.get("Date") or item.get("tradeDate", "")
        if not date_str:
            return None

        trade_date = _parse_date(str(date_str))
        if trade_date is None:
            return None

        fii_buy  = _parse_crore(item.get("fiiBuy")  or item.get("fiiBuyValue",  0.0))
        fii_sell = _parse_crore(item.get("fiiSell") or item.get("fiiSellValue", 0.0))
        dii_buy  = _parse_crore(item.get("diiBuy")  or item.get("diiBuyValue",  0.0))
        dii_sell = _parse_crore(item.get("diiSell") or item.get("diiSellValue", 0.0))

        return FIIDIIRecord(
            date=trade_date,
            fii_net_inr_crore=fii_buy - fii_sell,
            dii_net_inr_crore=dii_buy - dii_sell,
            fii_gross_buy=fii_buy,
            fii_gross_sell=fii_sell,
            dii_gross_buy=dii_buy,
            dii_gross_sell=dii_sell,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_date(s: str) -> date | None:
    """Try common NSE date string formats and return a date or None.

    Args:
        s: Raw date string, e.g. ``"01-Jan-2025"`` or ``"2025-01-01"``.

    Returns:
        Parsed :class:`datetime.date` or ``None`` on failure.
    """
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    log.debug("fii_dii.date_parse_fail", raw=s)
    return None


def _parse_crore(value: Any) -> float:
    """Convert a raw value to a float INR crore amount.

    Args:
        value: String or numeric value from the payload.

    Returns:
        Float representation; 0.0 on parse failure or NaN.
    """
    if value is None:
        return 0.0
    try:
        v = float(str(value).replace(",", "").strip())
        return v if not math.isnan(v) else 0.0
    except (ValueError, TypeError):
        return 0.0
