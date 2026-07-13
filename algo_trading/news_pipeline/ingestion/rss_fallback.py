"""algo_trading.news_pipeline.ingestion.rss_fallback — RSS & NSE announcement fallback.

Used when the Pulse scraper fails or returns no items.  Polls eight RSS feeds
and the NSE corporate-announcements API concurrently.

Sources
-------
- Economic Times Markets RSS
- MoneyControl Top News RSS
- Business Standard Markets RSS
- LiveMint Markets RSS
- Hindu BusinessLine Markets RSS
- Financial Express Markets RSS
- CNBCTV18 Markets RSS
- Business Today Markets RSS
- NSE corporate announcements API (requires cookie warm-up)

Each source has a 15-second timeout and returns [] on any error so that a
single bad feed never blocks the others.  ``feedparser`` is used for RSS;
``aiohttp`` for HTTP.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp
import structlog

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.ingestion.base import BaseFetcher
from algo_trading.news_pipeline.ingestion.http_session import _rotate_ua
from algo_trading.news_pipeline.models import RawNewsItem, make_id

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_RSS_SOURCES: list[tuple[str, str]] = [
    ("rss_et",   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("rss_mc",   "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("rss_bs",   "https://www.business-standard.com/rss/markets-106.rss"),
    ("rss_mint", "https://www.livemint.com/rss/markets"),
    ("rss_hbl",  "https://www.thehindubusinessline.com/markets/?service=rss"),
    ("rss_fe",   "https://www.financialexpress.com/market/rss/"),
    ("rss_cnbc", "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"),
    ("rss_bt",   "https://www.businesstoday.in/rss/markets"),
]

_NSE_ANN_SOURCE = "nse_ann"
_NSE_HOME_URL   = "https://www.nseindia.com/"
_NSE_ANN_URL    = (
    "https://www.nseindia.com/api/corporate-announcements?index=equities"
)

_TIMEOUT_SEC = 15   # per-feed timeout (seconds)

_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Common English / finance abbreviation stop-words (not NSE tickers)
# ---------------------------------------------------------------------------

_SKIP_WORDS: frozenset[str] = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
    "HIS", "HOW", "ITS", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY",
    "WHO", "BOY", "DID", "INC", "LTD", "NSE", "BSE", "ETF", "IPO",
    "FII", "DII", "RBI", "GDP", "CEO", "CFO", "COO", "CTO", "AGM",
    "EGM", "QOQ", "YOY", "PAT", "PBT", "EBIT", "EBITDA", "NAV",
    "AUM", "SEBI", "AMFI", "IRDA", "MCA", "GOI", "USD", "EUR",
    "INR", "GBP", "JPY", "SGD", "UAE", "USA", "UK", "UPI", "EMI",
    "SIP", "NRI", "HNI", "SME", "MSE", "MSME", "CBDC", "BUY",
    "SELL", "HOLD", "LONG", "SHORT", "CALL", "PUT", "OPEN", "HIGH",
    "LOW", "CLOSE", "PREV", "NEXT", "MORE", "LESS", "THAN", "THIS",
    "THAT", "WITH", "FROM", "WILL", "BEEN", "HAVE", "THEY", "THEIR",
    "SAID", "EACH", "WHICH", "DOES", "ALSO", "INTO", "THAN", "THEN",
    "WHEN", "WHAT", "SOME", "KNOW", "JUST", "MOST", "OVER", "SUCH",
    "BOTH", "MUCH", "MANY", "WELL", "GOOD", "EVEN", "BACK", "ONLY",
    "COME", "MADE", "MAKE", "VERY", "YEAR", "LAST", "FIRST", "LIVE",
    "LOOK", "LIKE", "SAME", "WENT", "MUST", "ALSO", "CORP", "CO",
    "PVT", "LLP", "UNIT", "FUND", "DEBT", "CASH", "BOND", "RATE",
    "SHARE", "STOCK", "TRADE", "NEWS", "MARKET", "INDIA", "TOP",
    "AMID", "AFTER", "AHEAD", "NEAR", "JUMP", "RISE", "FALL", "DROP",
    "GAIN", "LOSS", "PLAN", "DEAL", "BANK", "FIRM", "CORP", "SAYS",
    "SAYS", "SEES", "HITS", "CUTS", "SETS", "ADDS", "ENDS", "WINS",
    "PAYS", "GETS", "EYES", "SIGN", "SEND", "NEED", "WANT", "MOVE",
    "WEEK", "MONTH", "TIME", "DATE", "HALF", "FULL", "EACH", "BOTH",
    "NIFTY", "SENSEX",
})

_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,14})\b")


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_date_flexible(date_str: str | None) -> datetime:
    """Parse an RFC-2822 or ISO-8601 date string (RSS pubDate).

    Handles:
    - RFC-2822 format (standard RSS pubDate)
    - ISO-8601 with Z suffix
    - ISO-8601 with "+05:30" numeric TZ offsets (Python <3.11 fromisoformat limitation)

    Falls back to utcnow() on any parse failure.
    """
    if not date_str:
        return _now_utc()
    # Try RFC-2822 first (most RSS feeds use this).
    try:
        return parsedate_to_datetime(date_str)
    except Exception:  # noqa: BLE001
        pass
    # Try ISO-8601 with Z → +00:00 replacement.
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        pass
    # Handle "+05:30" style numeric TZ offsets (Python <3.11 fromisoformat limitation).
    try:
        cleaned = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', date_str)
        cleaned = cleaned.replace("Z", "+0000")
        dt = datetime.strptime(cleaned[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        pass
    return _now_utc()


# Keep the old name as an alias so any external callers aren't broken.
_parse_rfc2822 = _parse_date_flexible


# ---------------------------------------------------------------------------
# Safe feedparser import
# ---------------------------------------------------------------------------

def _feedparse(url_or_content: str, content: str | None = None) -> Any:
    """Call feedparser.parse(), returning None if feedparser is not installed."""
    try:
        import feedparser  # type: ignore[import]
        if content is not None:
            return feedparser.parse(content)
        return feedparser.parse(url_or_content)
    except ImportError:
        log.warning(
            "feedparser_not_installed",
            msg="feedparser is not installed; RSS fallback will be limited",
        )
        return None


# ---------------------------------------------------------------------------
# RSSFallback
# ---------------------------------------------------------------------------

class RSSFallback(BaseFetcher):
    """RSS/API fallback fetcher for Indian financial news.

    Polls eight RSS feeds and the NSE corporate-announcements API
    concurrently.  Designed to be rock-solid: every feed is isolated so a
    failure in one never affects the others.

    Returns ``list[RawNewsItem]``.  Used when the Pulse scraper returns
    an empty list.
    """

    source_name = "rss_fallback"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session (lazily created, cookie-aware)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={**_BROWSER_HEADERS, "User-Agent": _rotate_ua()},
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SEC),
                cookie_jar=aiohttp.CookieJar(),
            )
        return self._session

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def _fetch_internal(self) -> list[RawNewsItem]:
        """Fetch all RSS feeds and NSE announcements concurrently."""
        tasks: list[asyncio.Task] = []

        for source_name, url in _RSS_SOURCES:
            tasks.append(asyncio.create_task(
                self._fetch_rss(source_name, url),
                name=f"rss_{source_name}",
            ))

        tasks.append(asyncio.create_task(
            self._fetch_nse_announcements(),
            name="nse_announcements",
        ))

        all_items: list[RawNewsItem] = []
        for coro in asyncio.as_completed(tasks):
            try:
                items = await coro
                all_items.extend(items)
            except Exception as exc:  # noqa: BLE001
                log.warning("rss_fallback_task_error", error=str(exc))

        log.debug("rss_fallback_complete", total_items=len(all_items))
        return all_items

    # ------------------------------------------------------------------
    # Per-feed RSS fetch
    # ------------------------------------------------------------------

    async def _fetch_rss(self, source: str, url: str) -> list[RawNewsItem]:
        """Fetch and parse a single RSS feed URL.

        Downloads the raw XML with aiohttp (to benefit from the shared
        cookie-aware session and consistent headers) then passes the content
        to feedparser.  Returns [] on any error.
        """
        fetched_at = _now_utc()
        try:
            session = await self._get_session()
            async with session.get(
                url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SEC),
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "rss_bad_status",
                        source=source,
                        status=resp.status,
                        url=url,
                    )
                    return []
                raw_xml = await resp.text(encoding="utf-8", errors="replace")
        except asyncio.TimeoutError:
            log.warning("rss_timeout", source=source, url=url)
            return []
        except aiohttp.ClientError as exc:
            log.warning("rss_client_error", source=source, error=str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("rss_unexpected_error", source=source, error=str(exc))
            return []

        feed = _feedparse(url, content=raw_xml)
        if feed is None:
            return []

        items: list[RawNewsItem] = []
        for entry in getattr(feed, "entries", []):
            try:
                headline = (
                    getattr(entry, "title", None) or ""
                ).strip()
                if not headline:
                    continue

                link = (
                    getattr(entry, "link", None)
                    or getattr(entry, "id", None)
                    or ""
                ).strip()
                # Skip entries with no valid link — using the feed URL as
                # fallback would cause all linkless entries to share the same
                # ID and collide during deduplication.
                if not link:
                    continue
                item_id = make_id(link)

                pub_date_str = (
                    getattr(entry, "published", None)
                    or getattr(entry, "updated", None)
                    or ""
                )
                published_at = _parse_date_flexible(pub_date_str)

                summary = (
                    getattr(entry, "summary", None)
                    or getattr(entry, "description", None)
                    or ""
                )
                body = re.sub(r"<[^>]+>", "", summary)[:500].strip()

                combined = f"{headline} {body}"
                symbol_hints = self._extract_symbol_hints(combined)

                items.append(
                    RawNewsItem(
                        id=item_id,
                        source=source,
                        headline=headline,
                        url=link,
                        published_at=published_at,
                        fetched_at=fetched_at,
                        body=body,
                        symbol_hints=symbol_hints,
                    )
                )
            except Exception as entry_exc:  # noqa: BLE001
                log.debug("rss_entry_parse_error",
                          source=source, error=str(entry_exc))

        log.debug("rss_feed_parsed", source=source, items=len(items))
        return items

    # ------------------------------------------------------------------
    # NSE corporate announcements
    # ------------------------------------------------------------------

    async def _fetch_nse_announcements(self) -> list[RawNewsItem]:
        """Fetch NSE corporate announcements via the official JSON API.

        NSE requires a cookie warm-up: GET the homepage first so that the
        server sets ``nseappid`` and other session cookies, then call the
        announcements API with those cookies.  Without this the API returns
        HTTP 401 or an empty response.

        On 401 from the API, cookies are re-warmed and the request is retried
        once before giving up.

        Returns [] on any error.
        """
        fetched_at = _now_utc()
        session = await self._get_session()

        # Step 1: Cookie warm-up.
        try:
            async with session.get(
                _NSE_HOME_URL,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SEC),
            ) as warmup_resp:
                await warmup_resp.read()  # consume body to release connection
        except Exception as exc:  # noqa: BLE001
            log.warning("nse_warmup_failed", error=str(exc))
            return []

        # Step 2: Call the announcements API (with one retry on 401).
        headers = {
            **_BROWSER_HEADERS,
            "User-Agent": _rotate_ua(),
            "X-Requested-With": "XMLHttpRequest",
            "Referer": _NSE_HOME_URL,
            "Accept": "application/json, text/plain, */*",
        }

        async def _do_api_request() -> tuple[int, Any]:
            """Perform the API GET and return (status, parsed_data_or_None)."""
            async with session.get(
                _NSE_ANN_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SEC),
            ) as resp:
                status = resp.status
                if status != 200:
                    return status, None
                try:
                    data = await resp.json(content_type=None)
                except Exception:  # noqa: BLE001
                    raw = await resp.text()
                    import json
                    try:
                        data = json.loads(raw)
                    except Exception as jexc:  # noqa: BLE001
                        log.warning("nse_ann_json_parse_failed", error=str(jexc))
                        return status, None
                return status, data

        try:
            status, data = await _do_api_request()
        except asyncio.TimeoutError:
            log.warning("nse_ann_timeout", url=_NSE_ANN_URL)
            return []
        except aiohttp.ClientError as exc:
            log.warning("nse_ann_client_error", error=str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("nse_ann_unexpected_error", error=str(exc))
            return []

        # On 401, re-warm cookies and retry once.
        if status == 401:
            log.warning("nse_ann_401_rewarm", url=_NSE_ANN_URL)
            try:
                async with session.get(
                    _NSE_HOME_URL,
                    timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SEC),
                ) as warmup_resp:
                    await warmup_resp.read()
            except Exception as exc:  # noqa: BLE001
                log.warning("nse_rewarm_failed", error=str(exc))
                return []
            try:
                status, data = await _do_api_request()
            except Exception as exc:  # noqa: BLE001
                log.warning("nse_ann_retry_failed", error=str(exc))
                return []

        if status != 200 or data is None:
            log.warning(
                "nse_ann_bad_status",
                status=status,
                url=_NSE_ANN_URL,
            )
            return []

        # Step 3: Parse the response.
        items: list[RawNewsItem] = []
        if not isinstance(data, list):
            # Some NSE API responses wrap the list in a dict.
            if isinstance(data, dict):
                data = data.get("data") or data.get("announcements") or []
            else:
                log.warning("nse_ann_unexpected_shape", type=type(data).__name__)
                return []

        for ann in data:
            try:
                symbol = str(ann.get("symbol") or ann.get("sym") or "").strip().upper()
                if not symbol:
                    continue

                desc = str(ann.get("desc") or ann.get("subject") or "").strip()
                an_dt_raw = str(ann.get("an_dt") or ann.get("date") or "").strip()
                attch_text = str(ann.get("attchmntText") or ann.get("attachment") or "").strip()

                headline = f"{symbol}: {desc}" if desc else symbol
                body = attch_text[:500]

                # Parse announcement date.
                try:
                    published_at = datetime.fromisoformat(
                        an_dt_raw.replace(" ", "T").replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    published_at = fetched_at

                url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
                item_id = make_id(f"nse_ann|{symbol}|{an_dt_raw}|{desc}")

                combined = f"{headline} {body}"
                symbol_hints = self._extract_symbol_hints(combined)
                # Always include the primary symbol.
                if symbol not in symbol_hints:
                    symbol_hints = [symbol] + symbol_hints[:cfg.MAX_SYMBOLS_PER_ARTICLE - 1]

                items.append(
                    RawNewsItem(
                        id=item_id,
                        source=_NSE_ANN_SOURCE,
                        headline=headline,
                        url=url,
                        published_at=published_at,
                        fetched_at=fetched_at,
                        body=body,
                        symbol_hints=symbol_hints[:cfg.MAX_SYMBOLS_PER_ARTICLE],
                    )
                )
            except Exception as ann_exc:  # noqa: BLE001
                log.debug("nse_ann_row_error", error=str(ann_exc))

        log.debug("nse_ann_parsed", items=len(items))
        return items

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _extract_symbol_hints(self, text: str) -> list[str]:
        """Return uppercase tokens from *text* that resemble NSE ticker symbols.

        Filters out common English words and finance abbreviations.
        Caps the result at ``MAX_SYMBOLS_PER_ARTICLE`` entries.
        """
        raw = _TICKER_RE.findall(text)
        return [
            tok for tok in raw
            if tok not in _SKIP_WORDS
        ][:cfg.MAX_SYMBOLS_PER_ARTICLE]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
