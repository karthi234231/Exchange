"""
news/scraper.py — Async news scraper for Indian equity markets.

Polls RSS feeds every 5 minutes and NSE corporate announcements every 2 minutes.
Deduplicates by URL hash. Publishes new articles via ZMQ PUSH:5570.
Runs as a standalone process: python -m algo_trading.news.scraper

Sources:
  - Economic Times Markets RSS
  - Moneycontrol Top News RSS
  - Business Standard Markets RSS
  - NSE Corporate Announcements API (JSON)
  - BSE Corporate Announcements (where available)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import feedparser
import structlog
import zmq

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RSS_FEEDS = [
    ("economic_times",    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("moneycontrol",      "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("business_standard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("livemint",          "https://www.livemint.com/rss/markets"),
]

_NSE_ANNOUNCEMENT_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"

_STATE_DIR = Path(__file__).parent.parent.parent / "state"
_SEEN_IDS_FILE = _STATE_DIR / "news_seen_ids.json"

# How many seconds between scrape cycles for RSS vs NSE announcements
_RSS_INTERVAL_SEC = 300   # 5 minutes
_NSE_INTERVAL_SEC = 120   # 2 minutes

_HTTP_TIMEOUT_SEC = 10

# Regex to extract possible NSE-style ticker symbols (e.g. RELIANCE, TCS)
# Looks for all-caps words of length 2–15 (NSE symbols are 2–10 chars but be lenient)
_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,15})\b")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawArticle:
    """A single news article or corporate announcement, normalised across sources."""
    id: str                          # sha256 hex of the canonical URL
    source: str                      # e.g. "economic_times", "nse_announcements"
    headline: str
    url: str
    published_at: datetime           # always UTC
    body: str                        # first 500 chars of description / summary
    symbol_hints: list[str] = field(default_factory=list)  # raw symbol hints from text


def _make_id(url: str) -> str:
    """Return sha256 hex digest of the URL, used as the dedup key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _extract_symbol_hints(text: str) -> list[str]:
    """Return candidate NSE-style tickers found in text (heuristic, uppercase words)."""
    # Skip very-common English words that happen to be all-caps acronyms
    _SKIP = frozenset(["THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                       "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET",
                       "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD",
                       "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET",
                       "PUT", "SAY", "SHE", "TOO", "USE", "NSE", "BSE", "ETF",
                       "IPO", "FII", "DII", "GDP", "CPI", "WPI", "RBI", "SEBI",
                       "GST", "PAT", "EPS", "MD", "CEO", "CFO", "CTO",
                       "ED", "FIR", "MCA", "IT", "US", "UK", "EU", "Q1",
                       "Q2", "Q3", "Q4", "FY", "YOY", "QOQ", "PTI"])
    found = _SYMBOL_PATTERN.findall(text)
    return [sym for sym in found if sym not in _SKIP]


def _feedparser_to_datetime(parsed_time: Optional[tuple]) -> datetime:
    """Convert feedparser time_struct tuple to UTC datetime. Falls back to now()."""
    if parsed_time is None:
        return datetime.now(tz=timezone.utc)
    try:
        return datetime.fromtimestamp(time.mktime(parsed_time), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class NewsDeduplicator:
    """
    Persists seen article IDs to state/news_seen_ids.json.

    Format on disk:
      { "<sha256_id>": "<iso-timestamp-when-seen>" }

    cleanup_old() is called automatically before each scrape cycle.
    """

    def __init__(self, state_file: Path = _SEEN_IDS_FILE) -> None:
        self._file = state_file
        self._data: dict[str, str] = {}   # id → ISO timestamp string
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                raw = self._file.read_text(encoding="utf-8")
                self._data = json.loads(raw)
                log.debug("deduplicator_loaded", count=len(self._data), file=str(self._file))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("deduplicator_load_failed", error=str(exc), file=str(self._file))
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("deduplicator_save_failed", error=str(exc))

    def is_new(self, article_id: str) -> bool:
        """Return True if this article_id has NOT been seen before."""
        return article_id not in self._data

    def mark_seen(self, article_id: str) -> None:
        """Record article_id as seen at the current UTC time."""
        self._data[article_id] = datetime.now(tz=timezone.utc).isoformat()
        self._save()

    def cleanup_old(self, older_than_hours: int = 48) -> None:
        """
        Remove IDs that were seen more than `older_than_hours` hours ago.
        Prevents unbounded growth of the state file.
        """
        cutoff_ts = datetime.now(tz=timezone.utc).timestamp() - (older_than_hours * 3600)
        before = len(self._data)
        to_delete = []
        for article_id, iso_str in self._data.items():
            try:
                seen_ts = datetime.fromisoformat(iso_str).timestamp()
                if seen_ts < cutoff_ts:
                    to_delete.append(article_id)
            except (ValueError, TypeError):
                to_delete.append(article_id)   # corrupt entry — remove it

        for article_id in to_delete:
            del self._data[article_id]

        if to_delete:
            self._save()
            log.info("deduplicator_cleanup", removed=len(to_delete), remaining=len(self._data),
                     before=before, older_than_hours=older_than_hours)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

async def fetch_rss(
    session: aiohttp.ClientSession,
    source: str,
    url: str,
) -> list[RawArticle]:
    """
    Fetch and parse a single RSS feed, returning a list of RawArticle objects.

    Uses feedparser for parsing. Timeout: 10 seconds.
    Returns [] on any error (logs WARNING).
    """
    try:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
        async with session.get(url, timeout=timeout) as resp:
            raw_bytes = await resp.read()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        log.warning("rss_fetch_failed", source=source, url=url, error=str(exc))
        return []

    # feedparser.parse() is synchronous but fast for typical RSS sizes (~100 entries)
    feed = feedparser.parse(raw_bytes)
    articles: list[RawArticle] = []

    for entry in feed.entries:
        link: str = getattr(entry, "link", "") or ""
        if not link:
            continue  # can't dedup without a URL

        title: str = getattr(entry, "title", "") or ""
        summary: str = getattr(entry, "summary", "") or ""
        published_parsed = getattr(entry, "published_parsed", None)

        art = RawArticle(
            id=_make_id(link),
            source=source,
            headline=title.strip(),
            url=link.strip(),
            published_at=_feedparser_to_datetime(published_parsed),
            body=summary[:500],
            symbol_hints=_extract_symbol_hints(title + " " + summary),
        )
        articles.append(art)

    log.debug("rss_fetched", source=source, count=len(articles))
    return articles


async def fetch_nse_announcements(
    session: aiohttp.ClientSession,
) -> list[RawArticle]:
    """
    Fetch NSE corporate announcements from the NSE API.

    The response is a JSON list of dicts. Relevant keys:
      symbol, desc, an_dt, attchmntText

    Returns [] on any error (logs WARNING).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
        # NSE API requires a Referer header — without it the API returns 401/403
        "Referer": "https://www.nseindia.com/",
    }
    try:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
        async with session.get(
            _NSE_ANNOUNCEMENT_URL,
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
        log.warning("nse_announcements_fetch_failed", url=_NSE_ANNOUNCEMENT_URL, error=str(exc))
        return []

    if not isinstance(data, list):
        log.warning("nse_announcements_unexpected_format", type=type(data).__name__)
        return []

    articles: list[RawArticle] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        symbol: str = str(item.get("symbol", "")).strip().upper()
        desc: str = str(item.get("desc", "")).strip()
        an_dt: str = str(item.get("an_dt", "")).strip()
        body_text: str = str(item.get("attchmntText", "")).strip()

        # Build a stable canonical URL from symbol + timestamp for dedup
        url_key = f"nse://announcements/{symbol}/{an_dt}"
        canonical_url = f"https://www.nseindia.com/companies-listing/corporate-filings-announcements?symbol={symbol}"

        # Parse announcement datetime — NSE format is typically "DD-Mon-YYYY HH:MM:SS"
        pub_dt = _parse_nse_datetime(an_dt)

        art = RawArticle(
            id=_make_id(url_key),
            source="nse_announcements",
            headline=desc,
            url=canonical_url,
            published_at=pub_dt,
            body=body_text[:500],
            symbol_hints=[symbol] if symbol else [],
        )
        articles.append(art)

    log.debug("nse_announcements_fetched", count=len(articles))
    return articles


def _parse_nse_datetime(an_dt: str) -> datetime:
    """
    Parse NSE announcement datetime string to UTC datetime.

    NSE uses formats like:
      "06-Jun-2026 09:15:32"
      "2026-06-06T09:15:32"

    Falls back to utcnow() if format is unrecognised.
    """
    if not an_dt:
        return datetime.now(tz=timezone.utc)

    _FORMATS = [
        "%d-%b-%Y %H:%M:%S",   # 06-Jun-2026 09:15:32
        "%Y-%m-%dT%H:%M:%S",   # 2026-06-06T09:15:32
        "%d-%m-%Y %H:%M:%S",   # 06-06-2026 09:15:32
        "%Y-%m-%d %H:%M:%S",   # 2026-06-06 09:15:32
        "%d-%b-%Y",            # 06-Jun-2026
        "%Y-%m-%d",            # 2026-06-06
    ]
    for fmt in _FORMATS:
        try:
            # NSE datetimes are IST (UTC+5:30); store as UTC
            dt_naive = datetime.strptime(an_dt, fmt)
            # Approximate IST → UTC by subtracting 5h30m
            from datetime import timedelta
            dt_utc = dt_naive.replace(tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)
            return dt_utc
        except ValueError:
            continue

    log.debug("nse_datetime_parse_failed", an_dt=an_dt)
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# ZMQ publisher helper
# ---------------------------------------------------------------------------

def _article_to_payload(article: RawArticle) -> bytes:
    """Serialise a RawArticle to a JSON-encoded bytes payload for ZMQ."""
    payload = {
        "id":           article.id,
        "source":       article.source,
        "headline":     article.headline,
        "url":          article.url,
        "published_iso": article.published_at.isoformat(),
        "body":         article.body,
        "symbol_hints": article.symbol_hints,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Main scraper coroutine
# ---------------------------------------------------------------------------

async def run_scraper(zmq_push_port: int = 5570) -> None:
    """
    Main async scraper loop.

    - Polls all RSS feeds every 5 minutes.
    - Polls NSE corporate announcements every 2 minutes.
    - Deduplicates by URL hash.
    - Publishes new articles via ZMQ PUSH as JSON payloads.

    Runs until cancelled (e.g. SIGINT/SIGTERM handled by __main__ wrapper).
    """
    # ── ZMQ setup ────────────────────────────────────────────────────────────
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 200)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{zmq_push_port}")
    log.info("zmq_push_bound", port=zmq_push_port)

    dedup = NewsDeduplicator()
    published_total = 0
    cycle = 0

    # Track independent poll timers for RSS vs NSE
    last_rss_poll: float = 0.0
    last_nse_poll: float = 0.0

    # aiohttp session with a cookie jar (NSE requires cookie warm-up)
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm up NSE cookie by visiting the homepage first
        try:
            await session.get(
                "https://www.nseindia.com",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
            log.info("nse_cookie_warmup_done")
        except Exception as exc:  # noqa: BLE001
            log.warning("nse_cookie_warmup_failed", error=str(exc))

        log.info("scraper_started", rss_interval_sec=_RSS_INTERVAL_SEC,
                 nse_interval_sec=_NSE_INTERVAL_SEC)

        while True:
            now = time.monotonic()
            articles: list[RawArticle] = []

            # ── RSS poll ──────────────────────────────────────────────────────
            if now - last_rss_poll >= _RSS_INTERVAL_SEC:
                dedup.cleanup_old(older_than_hours=48)   # prune before new batch
                rss_tasks = [fetch_rss(session, src, url) for src, url in _RSS_FEEDS]
                rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
                for result in rss_results:
                    if isinstance(result, list):
                        articles.extend(result)
                    else:
                        log.warning("rss_task_exception", error=str(result))
                last_rss_poll = time.monotonic()

            # ── NSE announcements poll ────────────────────────────────────────
            if now - last_nse_poll >= _NSE_INTERVAL_SEC:
                nse_articles = await fetch_nse_announcements(session)
                articles.extend(nse_articles)
                last_nse_poll = time.monotonic()

            # ── Dedup + publish ───────────────────────────────────────────────
            new_count = 0
            for article in articles:
                if not dedup.is_new(article.id):
                    continue
                payload = _article_to_payload(article)
                try:
                    sock.send(payload, flags=zmq.NOBLOCK)
                    dedup.mark_seen(article.id)
                    new_count += 1
                    published_total += 1
                except zmq.Again:
                    log.warning("zmq_push_hwm_reached", article_id=article.id)
                except zmq.ZMQError as exc:
                    log.error("zmq_push_error", error=str(exc))

            if articles or new_count:
                log.info("scraper_cycle_done", cycle=cycle, fetched=len(articles),
                         new_published=new_count, total_published=published_total)

            cycle += 1
            # Sleep for the shortest poll interval so we don't overshoot either timer
            await asyncio.sleep(min(_RSS_INTERVAL_SEC, _NSE_INTERVAL_SEC))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import signal
    import sys

    # Basic logging without importing algo_trading.logging_setup to keep the
    # scraper runnable as a standalone process with minimal dependencies.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = loop.create_task(run_scraper())

    def _shutdown(sig: int, frame: object) -> None:  # noqa: ARG001
        log.info("scraper_shutdown_requested", signal=sig)
        main_task.cancel()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        log.info("scraper_stopped")
    finally:
        loop.close()
