"""
algo_trading.news_pipeline.ingestion.http_session — Central HTTP session manager.

Provides:
  - NSESession: aiohttp session with proper NSE cookie warm-up and auto-refresh
    on 401/403, a pool of rotating browser User-Agents, Sec-Fetch-* headers,
    and per-domain minimum delay to avoid rate limits.
  - CircuitBreaker: thread-safe per-source state machine (CLOSED → OPEN → HALF_OPEN).
  - _rotate_ua(): module-level function returning the next User-Agent from the pool.

Why this module exists
----------------------
NSE India requires:
  1. A GET to the homepage to receive session cookies (nseappid, nsit, AKA*, etc.)
  2. A second GET to a quote page to receive additional cookies
  3. API calls with those cookies + Referer + X-Requested-With headers
  4. Browser-like headers (Sec-Fetch-*, Accept-Encoding: gzip)

Without this, NSE returns 401 or empty JSON.  Zerodha Pulse also rate-limits
static User-Agents.  This module handles both by:
  - Maintaining a persistent aiohttp ClientSession with cookie_jar
  - Warming up NSE cookies on connect() and re-warming on 401/403
  - Rotating User-Agents from a pool of 10 real Chrome/Firefox strings
  - Enforcing minimum 2-second gap between requests to the same domain
"""
from __future__ import annotations

import asyncio
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# User-Agent rotation pool
# ---------------------------------------------------------------------------

_UA_POOL: list[str] = [
    # Chrome 124 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 123 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome 122 macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox 125 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox 124 Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Edge 124 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Chrome 121 Android
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36",
    # Safari 17 macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    # Chrome 120 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Opera 109 Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
]

_ua_index = 0
_ua_lock = threading.Lock()


def _rotate_ua() -> str:
    """Return the next User-Agent from the pool in round-robin order."""
    global _ua_index
    with _ua_lock:
        ua = _UA_POOL[_ua_index % len(_UA_POOL)]
        _ua_index += 1
    return ua


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class _CBState(Enum):
    CLOSED    = "CLOSED"     # normal operation
    OPEN      = "OPEN"       # failures exceeded threshold; block calls
    HALF_OPEN = "HALF_OPEN"  # cooldown elapsed; allow one probe


@dataclass
class CircuitBreaker:
    """
    Thread-safe per-source circuit breaker.

    States:
      CLOSED    → normal; failure_count increments on each error
      OPEN      → tripped after failure_count ≥ failure_threshold;
                  all calls raise CircuitOpenError immediately
      HALF_OPEN → entered after recovery_timeout seconds in OPEN;
                  next call is allowed as a probe:
                    success → CLOSED (reset)
                    failure → OPEN again

    Usage:
        cb = CircuitBreaker(source="pulse")
        if cb.allow():
            try:
                result = await do_work()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            raise CircuitOpenError(cb.source)
    """

    source:             str
    failure_threshold:  int   = 5
    recovery_timeout:   float = 60.0   # seconds

    _state:          _CBState       = field(default=_CBState.CLOSED, init=False, repr=False)
    _failure_count:  int            = field(default=0,               init=False, repr=False)
    _opened_at:      float          = field(default=0.0,             init=False, repr=False)
    _lock:           threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def allow(self) -> bool:
        """Return True if a request should be attempted."""
        with self._lock:
            if self._state == _CBState.CLOSED:
                return True
            if self._state == _CBState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._state = _CBState.HALF_OPEN
                    log.info("circuit_breaker.half_open", source=self.source)
                    return True
                return False
            # HALF_OPEN: allow the probe
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state in (_CBState.HALF_OPEN, _CBState.OPEN):
                log.info("circuit_breaker.closed", source=self.source)
            self._state = _CBState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == _CBState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                self._state = _CBState.OPEN
                self._opened_at = time.monotonic()
                log.warning(
                    "circuit_breaker.opened",
                    source=self.source,
                    failures=self._failure_count,
                    recovery_in_seconds=self.recovery_timeout,
                )

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._state == _CBState.OPEN

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is in OPEN state."""

    def __init__(self, source: str) -> None:
        super().__init__(f"Circuit breaker OPEN for source '{source}'")
        self.source = source


# ---------------------------------------------------------------------------
# Per-domain rate limiter
# ---------------------------------------------------------------------------

class _DomainRateLimiter:
    """
    Enforces a minimum delay between successive requests to the same domain.

    Thread-safe via asyncio lock per domain (one per event loop).
    """

    def __init__(self, min_delay_seconds: float = 2.0) -> None:
        self._min_delay = min_delay_seconds
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def acquire(self, url: str) -> None:
        """Wait until the minimum delay since the last request to this domain has elapsed."""
        domain = urlparse(url).netloc or url
        lock = self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            elapsed = now - last
            if elapsed < self._min_delay:
                jitter = random.uniform(0.1, 0.5)
                await asyncio.sleep(self._min_delay - elapsed + jitter)
            self._last_request[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# NSESession
# ---------------------------------------------------------------------------

_NSE_HOME_URL     = "https://www.nseindia.com/"
_NSE_QUOTE_URL    = "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE"
_COOKIE_MAX_AGE_S = 300   # re-warm cookies every 5 minutes


class NSESession:
    """
    aiohttp ClientSession pre-configured for NSE API access.

    Features:
    - Cookie warm-up: two GET requests (homepage + quote page) to receive
      session cookies before any API call.
    - Auto-refresh: cookies are re-warmed when 401/403 is received or when
      the session is older than _COOKIE_MAX_AGE_S seconds.
    - Rotating User-Agent: each new session gets a fresh UA from the pool.
    - Sec-Fetch-* headers matching what Chrome sends to same-origin XHR.
    - Accept-Encoding: gzip, deflate, br for compressed responses.

    Usage:
        nse = NSESession()
        await nse.connect()
        try:
            async with nse.get("https://www.nseindia.com/api/allIndices") as resp:
                data = await resp.json(content_type=None)
        finally:
            await nse.close()
    """

    def __init__(
        self,
        timeout_sec: float = 15.0,
        min_request_delay: float = 2.0,
    ) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cookies_warmed_at: float = 0.0
        self._rate_limiter = _DomainRateLimiter(min_delay_seconds=min_request_delay)
        self._warm_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the session and perform initial cookie warm-up."""
        await self._ensure_session()
        await self._warm_cookies()

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Request interface
    # ------------------------------------------------------------------

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Perform a GET request with:
        - Rate limiting (min 2s between requests to same domain)
        - Automatic cookie refresh on 401/403
        - NSE API headers (Referer, X-Requested-With, Sec-Fetch-*)
        """
        await self._ensure_session()
        await self._ensure_cookies_fresh()
        await self._rate_limiter.acquire(url)

        headers = self._api_headers(url)
        kwargs.setdefault("headers", {})
        kwargs["headers"] = {**headers, **kwargs["headers"]}
        kwargs.setdefault("timeout", self._timeout)

        assert self._session is not None
        resp = await self._session.get(url, **kwargs)

        # Auto-refresh cookies on auth failure and retry once
        if resp.status in (401, 403):
            log.warning("nse_session.auth_failure", status=resp.status, url=url)
            resp.release()
            await self._warm_cookies(force=True)
            resp = await self._session.get(url, **kwargs)

        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            ua = _rotate_ua()
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                cookie_jar=aiohttp.CookieJar(),
                headers={
                    "User-Agent":      ua,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection":      "keep-alive",
                },
            )
            self._cookies_warmed_at = 0.0  # force warm-up
            log.debug("nse_session.created", ua=ua[:40])

    async def _ensure_cookies_fresh(self) -> None:
        elapsed = time.monotonic() - self._cookies_warmed_at
        if elapsed > _COOKIE_MAX_AGE_S:
            await self._warm_cookies()

    async def _warm_cookies(self, force: bool = False) -> None:
        """
        Perform the two-step NSE cookie warm-up.

        Step 1: GET homepage (sets nseappid, nsit, AKA_A1 cookies).
        Step 2: GET a quote page (sets additional session tokens).
        Both are fire-and-read; response bodies are discarded.
        """
        async with self._warm_lock:
            # Double-check inside lock to avoid concurrent warm-ups.
            if not force and time.monotonic() - self._cookies_warmed_at < _COOKIE_MAX_AGE_S:
                return

            assert self._session is not None
            ua = _rotate_ua()

            # Step 1: homepage
            try:
                async with self._session.get(
                    _NSE_HOME_URL,
                    headers={"User-Agent": ua, "Accept": "text/html,*/*"},
                    timeout=aiohttp.ClientTimeout(total=10.0),
                    allow_redirects=True,
                ) as resp:
                    await resp.read()
                log.debug("nse_session.warmup_step1", status=resp.status)
            except Exception as exc:
                log.warning("nse_session.warmup_step1_failed", error=str(exc))
                return

            await asyncio.sleep(random.uniform(0.5, 1.2))

            # Step 2: quote page (sets additional cookies)
            try:
                async with self._session.get(
                    _NSE_QUOTE_URL,
                    headers={"User-Agent": ua, "Referer": _NSE_HOME_URL, "Accept": "text/html,*/*"},
                    timeout=aiohttp.ClientTimeout(total=10.0),
                    allow_redirects=True,
                ) as resp:
                    await resp.read()
                log.debug("nse_session.warmup_step2", status=resp.status)
            except Exception as exc:
                log.warning("nse_session.warmup_step2_failed", error=str(exc))

            self._cookies_warmed_at = time.monotonic()
            log.info("nse_session.cookies_warmed")

    @staticmethod
    def _api_headers(url: str) -> dict[str, str]:
        """Return headers appropriate for an NSE API call."""
        is_nse = "nseindia.com" in url
        if is_nse:
            return {
                "Accept":           "application/json, text/plain, */*",
                "Referer":          _NSE_HOME_URL,
                "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Site":   "same-origin",
                "Sec-Fetch-Mode":   "cors",
                "Sec-Fetch-Dest":   "empty",
                "DNT":              "1",
            }
        return {
            "Accept":         "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        }
