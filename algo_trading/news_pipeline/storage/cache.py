"""algo_trading.news_pipeline.storage.cache — Thread-safe in-memory TTL cache.

Stdlib only (no external dependencies).

Module-level singletons
-----------------------
``news_id_cache``   — deduplication cache (TTL = 48 h)
``signal_cache``    — latest signal per symbol  (TTL = 60 min)

Usage::

    from algo_trading.news_pipeline.storage.cache import news_id_cache, signal_cache

    # Dedup check
    if not news_id_cache.exists(article_id):
        news_id_cache.set(article_id, True)
        process(article)

    # Cache a signal
    signal_cache.set(symbol, signal.to_dict(), ttl_seconds=signal.ttl_minutes * 60)
    cached = signal_cache.get(symbol)   # None if expired / missing
"""
from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe key-value cache with per-entry TTL.

    Each entry stores ``(value, expire_at)`` where ``expire_at`` is an
    absolute monotonic timestamp produced by :func:`time.monotonic`.

    ``cleanup()`` is triggered probabilistically inside ``set()`` to prevent
    unbounded memory growth: once every *cleanup_interval* calls (default 1000).
    """

    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        cleanup_interval: int = 1000,
    ) -> None:
        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._set_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store *value* under *key* with an optional per-entry TTL.

        If *ttl_seconds* is None the instance-level default is used.
        Triggers probabilistic cleanup every ``cleanup_interval`` calls.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expire_at = time.monotonic() + ttl

        with self._lock:
            self._store[key] = (value, expire_at)
            self._set_count += 1
            should_cleanup = (self._set_count % self._cleanup_interval) == 0

        if should_cleanup:
            self.cleanup()

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if missing or expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() > expire_at:
                # Lazy eviction — remove stale entry on access
                del self._store[key]
                return None
            return value

    def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present and not yet expired."""
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""
        with self._lock:
            self._store.pop(key, None)

    def cleanup(self) -> int:
        """Evict all expired entries.

        Returns the number of entries removed.  Thread-safe; safe to call
        manually at any time.
        """
        now = time.monotonic()
        expired_keys: list[str] = []

        with self._lock:
            for key, (_, expire_at) in self._store.items():
                if now > expire_at:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._store[key]

        return len(expired_keys)

    def size(self) -> int:
        """Return the number of entries currently in the cache (including expired).

        Call :meth:`cleanup` first if you need the count of *live* entries.
        """
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Dunder helpers (useful for debugging / testing)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TTLCache(size={self.size()}, default_ttl={self._default_ttl}s)"
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Deduplication cache for raw news item IDs.
#: TTL matches DEDUP_TTL_HOURS (48 h) so IDs seen within the last two days
#: are never re-processed even across process restarts if the cache is warmed
#: from PostgreSQL on startup.
news_id_cache: TTLCache = TTLCache(default_ttl_seconds=48 * 3600)

#: Per-symbol signal cache.
#: Default TTL of 60 minutes mirrors SIGNAL_TTL_MINUTES; callers may override
#: per-entry by passing signal.ttl_minutes * 60 to cache.set().
signal_cache: TTLCache = TTLCache(default_ttl_seconds=3600)
