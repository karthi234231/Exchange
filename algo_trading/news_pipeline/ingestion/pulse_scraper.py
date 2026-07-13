"""algo_trading.news_pipeline.ingestion.pulse_scraper — Zerodha Pulse scraper.

Zerodha Pulse (https://pulse.zerodha.com/) is a React/Next.js app that
aggregates Indian financial news.  The news feed is embedded in the server-
rendered HTML as a ``__NEXT_DATA__`` JSON blob in a <script> tag.

Fetch strategy (tried in order):
1. Parse ``<script id="__NEXT_DATA__">`` from the Pulse homepage HTML.
2. Walk the Next.js page props for any list that contains "headline",
   "title", or "link" keys.
3. Fall back to scraping visible <a href> links from the rendered HTML that
   match financial-news URL patterns.
4. If nothing is found, return [] and emit a warning — the RSSFallback will
   take over.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import structlog

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.ingestion.base import BaseFetcher
from algo_trading.news_pipeline.ingestion.http_session import _rotate_ua
from algo_trading.news_pipeline.models import RawNewsItem, make_id

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PULSE_URL = cfg.PULSE_URL                # "https://pulse.zerodha.com/"
_TIMEOUT = aiohttp.ClientTimeout(total=cfg.PULSE_TIMEOUT_SEC)

_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://zerodha.com/",
}

# Common English words that look like tickers but are not NSE symbols.
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
    "COME", "MADE", "MAKE", "VERY", "YEAR", "LAST", "FIRST", "LONG",
    "LIVE", "LOOK", "LIKE", "SAME", "WENT", "MUST", "ALSO",
})

# NSE-style ticker: 2–15 uppercase ASCII letters (and digits allowed after first char).
_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,14})\b")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_symbol_hints(text: str) -> list[str]:
    """Return uppercase tokens from *text* that look like NSE ticker symbols."""
    raw = _TICKER_RE.findall(text)
    return [tok for tok in raw if tok not in _SKIP_WORDS][:cfg.MAX_SYMBOLS_PER_ARTICLE]


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _safe_url(href: str | None, base: str = _PULSE_URL) -> str | None:
    """Resolve relative URLs against *base*; return None for empty/non-http."""
    if not href:
        return None
    try:
        resolved = urljoin(base, href.strip())
        parsed = urlparse(resolved)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return resolved
    except Exception:  # noqa: BLE001
        pass
    return None


def _walk_for_news(obj: Any, depth: int = 0) -> list[dict]:
    """Recursively walk a parsed JSON object looking for news-like dicts.

    A "news-like" dict is one that has at least a "title" or "headline" key
    plus a "link" or "url" key.
    """
    if depth > 12:
        return []
    results: list[dict] = []
    if isinstance(obj, dict):
        has_text = "title" in obj or "headline" in obj or "name" in obj
        has_link = "link" in obj or "url" in obj or "href" in obj
        if has_text and has_link:
            results.append(obj)
        else:
            for v in obj.values():
                results.extend(_walk_for_news(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_walk_for_news(item, depth + 1))
    return results


def _dict_to_raw_news(record: dict, fetched_at: datetime) -> RawNewsItem | None:
    """Convert a raw news-like dict to a RawNewsItem.  Returns None if unusable."""
    headline = (
        record.get("title")
        or record.get("headline")
        or record.get("name")
        or ""
    ).strip()
    if not headline:
        return None

    url_raw = (
        record.get("link")
        or record.get("url")
        or record.get("href")
        or ""
    ).strip()
    url = _safe_url(url_raw) or _PULSE_URL
    item_id = make_id(url)

    # Parse published_at if available; fall back to fetched_at.
    pub_raw = record.get("pubDate") or record.get("publishedAt") or record.get("date") or ""
    try:
        published_at = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00"))
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        try:
            # Handle "+05:30" style offsets (Python <3.11 fromisoformat limitation)
            import re as _re
            cleaned = _re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', str(pub_raw))
            cleaned = cleaned.replace("Z", "+0000")
            from datetime import datetime as _dt
            published_at = _dt.strptime(cleaned[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            published_at = fetched_at

    body = (record.get("body") or record.get("summary") or record.get("description") or "")[:500].strip()
    combined_text = f"{headline} {body}"
    symbol_hints = _extract_symbol_hints(combined_text)

    return RawNewsItem(
        id=item_id,
        source="pulse",
        headline=headline,
        url=url,
        published_at=published_at,
        fetched_at=fetched_at,
        body=body,
        symbol_hints=symbol_hints,
    )


# ---------------------------------------------------------------------------
# PulseScraper
# ---------------------------------------------------------------------------

class PulseScraper(BaseFetcher):
    """Scrapes Zerodha Pulse for aggregated Indian financial news.

    Strategy (attempted in order):
    1. Fetch the Pulse homepage HTML.
    2. Parse ``<script id="__NEXT_DATA__">`` for React/Next.js hydration JSON.
    3. Walk the hydration JSON recursively for news-like records.
    4. If that yields nothing, fall back to scanning ``<a href>`` tags whose
       href points to known news domains.
    5. If still nothing, return [] with a warning log.

    Deduplication is in-memory (``_seen_ids`` set) for the process lifetime.
    """

    source_name = "pulse"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._seen_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared aiohttp session, creating it lazily."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={**_HEADERS, "User-Agent": _rotate_ua()},
                timeout=_TIMEOUT,
            )
        return self._session

    # ------------------------------------------------------------------
    # Internal fetch
    # ------------------------------------------------------------------

    async def _fetch_internal(self) -> list[RawNewsItem]:
        """Fetch and parse Zerodha Pulse HTML.  Returns [] on any failure."""
        fetched_at = _now_utc()
        session = await self._get_session()

        try:
            async with session.get(_PULSE_URL) as resp:
                if resp.status != 200:
                    log.warning(
                        "pulse_bad_status",
                        status=resp.status,
                        url=_PULSE_URL,
                    )
                    return []
                html = await resp.text(encoding="utf-8", errors="replace")
        except aiohttp.ClientError as exc:
            log.warning("pulse_request_failed", error=str(exc))
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("pulse_unexpected_error", error=str(exc))
            return []

        items: list[RawNewsItem] = []

        # ------------------------------------------------------------------
        # Strategy 1: Parse __NEXT_DATA__ script tag.
        # ------------------------------------------------------------------
        next_data_match = re.search(
            r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if next_data_match:
            try:
                next_json = json.loads(next_data_match.group(1))
                news_dicts = _walk_for_news(next_json)
                for nd in news_dicts:
                    item = _dict_to_raw_news(nd, fetched_at)
                    if item and item.id not in self._seen_ids:
                        self._seen_ids.add(item.id)
                        items.append(item)
                log.debug("pulse_next_data_parsed", count=len(items))
            except json.JSONDecodeError as exc:
                log.warning("pulse_next_data_json_error", error=str(exc))

        # ------------------------------------------------------------------
        # Strategy 2: Scan any other inline JSON blobs in <script> tags.
        # ------------------------------------------------------------------
        if not items:
            script_pattern = re.compile(
                r"<script[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE
            )
            for script_match in script_pattern.finditer(html):
                content = script_match.group(1).strip()
                # Skip tiny scripts and non-JSON content.
                if len(content) < 30 or not content.startswith(("{", "[")):
                    continue
                try:
                    blob = json.loads(content)
                    news_dicts = _walk_for_news(blob)
                    for nd in news_dicts:
                        item = _dict_to_raw_news(nd, fetched_at)
                        if item and item.id not in self._seen_ids:
                            self._seen_ids.add(item.id)
                            items.append(item)
                except json.JSONDecodeError:
                    pass
            if items:
                log.debug("pulse_inline_json_parsed", count=len(items))

        # ------------------------------------------------------------------
        # Strategy 3: Scrape visible <a href> anchor links.
        # ------------------------------------------------------------------
        if not items:
            items = self._scrape_anchor_links(html, fetched_at)
            if items:
                log.debug("pulse_anchor_fallback_parsed", count=len(items))

        if not items:
            log.warning(
                "pulse_parse_failed",
                msg="Could not extract any news items from Pulse HTML; "
                    "RSS fallback will handle ingestion.",
            )

        return items

    # ------------------------------------------------------------------
    # Anchor link scraping
    # ------------------------------------------------------------------

    _NEWS_DOMAIN_RE = re.compile(
        r"(economictimes|moneycontrol|business-standard|livemint|"
        r"thehindu|ndtv|cnbctv18|financialexpress|zeebiz|"
        r"reuters|bloomberg|bseindia|nseindia|"
        r"hindustantimes|tribuneindia|firstpost|thequint|"
        r"outlookindia|businesstoday|forbesindia|inc42|"
        r"yourstory|entrackr|vccircle|mint|scroll|"
        r"thewire|theprint|deccanherald|telegraphindia|tribunebiz)",
        re.IGNORECASE,
    )
    _HREF_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)

    def _scrape_anchor_links(
        self, html: str, fetched_at: datetime
    ) -> list[RawNewsItem]:
        """Extract news headline links from raw HTML anchor tags."""
        items: list[RawNewsItem] = []
        for m in self._HREF_RE.finditer(html):
            href_raw, link_text_raw = m.group(1), m.group(2)
            url = _safe_url(href_raw)
            if not url:
                continue
            if not self._NEWS_DOMAIN_RE.search(url):
                continue
            # Strip HTML tags from link text.
            headline = re.sub(r"<[^>]+>", "", link_text_raw).strip()
            if len(headline) < 10:
                continue
            item_id = make_id(url)
            if item_id in self._seen_ids:
                continue
            self._seen_ids.add(item_id)
            items.append(
                RawNewsItem(
                    id=item_id,
                    source="pulse",
                    headline=headline,
                    url=url,
                    published_at=fetched_at,
                    fetched_at=fetched_at,
                    body="",
                    symbol_hints=_extract_symbol_hints(headline),
                )
            )
        return items

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
