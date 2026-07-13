"""
Two-layer deduplication for raw news items.

Layer 1 (fast):  in-memory TTL cache (48h) of seen article IDs.
                 Uses the news_id_cache singleton from storage.cache.
Layer 2 (authoritative): check PostgreSQL news_events table for existing IDs.
                 Only reached on cache misses (rare after warm-up).

Dedup key = RawNewsItem.id (SHA-256 of canonical URL, see models.make_id).

Design constraints:
- Never raises — DB errors are caught and logged; items are assumed new on DB error
  to preserve throughput (better to reprocess than to silently discard).
- Fully async: filter_new and is_new are coroutines so they compose naturally
  with the async ingestion/processing pipeline.
- Works without a DB connection (pg_store=None) — in-memory cache only.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import structlog

from algo_trading.news_pipeline.models import RawNewsItem
from algo_trading.news_pipeline.storage.cache import TTLCache
from algo_trading.news_pipeline.config import cfg

if TYPE_CHECKING:
    # PGStore is imported lazily at runtime to avoid a hard dependency
    pass

logger = structlog.get_logger(__name__)


class Deduplicator:
    """
    Checks whether a RawNewsItem has already been processed.

    Two-layer check:
      1. In-memory TTL cache (O(1), no I/O).
      2. PostgreSQL news_events table (authoritative, async, only on cache miss).

    Usage::

        dedup = Deduplicator(pg_store=pg)
        new_items = await dedup.filter_new(raw_items)
        for item in new_items:
            event = await process(item)
            dedup.mark_seen(item.id)     # eagerly mark; DB write is pg_store's job

    Args:
        pg_store: Optional PGStore instance. When None the deduplicator works
                  in cache-only mode (suitable for testing and batch offline use).
    """

    def __init__(self, pg_store: Optional[object] = None) -> None:
        self._cache: TTLCache = TTLCache(
            default_ttl_seconds=cfg.DEDUP_TTL_HOURS * 3600
        )
        self._pg = pg_store

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def filter_new(self, items: list[RawNewsItem]) -> list[RawNewsItem]:
        """
        Return only items not seen before.

        Algorithm:
          For each item:
            1. Check in-memory cache (instant).
            2. If cache hit -> duplicate, skip.
            3. If cache miss -> check DB (if pg_store configured).
            4. If DB hit -> duplicate; backfill cache.
            5. If DB miss (or no DB) -> new item; add to cache.

        Args:
            items: List of raw items from an ingestion source.

        Returns:
            Subset of items that are genuinely new (not seen in cache or DB).
        """
        new_items: list[RawNewsItem] = []
        for item in items:
            if await self.is_new(item):
                new_items.append(item)
                self.mark_seen(item.id)

        if len(items) > len(new_items):
            logger.debug(
                "deduplicator_filter_result",
                total=len(items),
                new=len(new_items),
                duplicates=len(items) - len(new_items),
            )

        return new_items

    async def is_new(self, item: RawNewsItem) -> bool:
        """
        Return True if the item has NOT been seen before.

        Checks in-memory cache first. Falls through to DB on cache miss.
        On any DB error assumes the item is new (fail-open for throughput).

        Args:
            item: RawNewsItem to check.

        Returns:
            True if the item is new; False if it is a duplicate.
        """
        article_id = item.id

        # Layer 1: in-memory cache
        if self._cache.exists(article_id):
            logger.debug("dedup_cache_hit", article_id=article_id[:16])
            return False

        # Layer 2: PostgreSQL (only if pg_store configured)
        if self._pg is not None:
            try:
                exists_in_db = await self._pg_exists(article_id)
                if exists_in_db:
                    # Backfill cache so future checks stay fast
                    self._cache.set(article_id, True)
                    logger.debug("dedup_db_hit", article_id=article_id[:16])
                    return False
            except Exception as exc:
                # Fail-open: treat as new on DB error to preserve throughput
                logger.warning(
                    "dedup_db_check_failed",
                    article_id=article_id[:16],
                    error=str(exc),
                    action="treating_as_new",
                )

        return True

    def mark_seen(self, item_id: str) -> None:
        """
        Add *item_id* to the in-memory cache.

        This should be called immediately after an item is accepted for
        processing so that concurrent pipeline invocations don't duplicate.
        The corresponding DB write is handled by pg_store after the full
        ProcessedNewsEvent is persisted.

        Args:
            item_id: SHA-256 hex ID from RawNewsItem.id.
        """
        self._cache.set(item_id, True)
        logger.debug("dedup_marked_seen", article_id=item_id[:16])

    def cache_size(self) -> int:
        """Return the number of entries currently held in the in-memory cache."""
        return self._cache.size()

    def sweep_cache(self) -> int:
        """Evict expired entries from the in-memory cache. Returns count removed."""
        removed = self._cache.cleanup()
        if removed:
            logger.debug("dedup_cache_swept", removed=removed)
        return removed

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _pg_exists(self, article_id: str) -> bool:
        """
        Check whether *article_id* exists in the news_events PostgreSQL table.

        Delegates to pg_store.article_exists(article_id) if that method is
        available, otherwise falls back to a raw SQL query via
        pg_store.fetch_one().

        Args:
            article_id: SHA-256 hex digest.

        Returns:
            True if a row with this ID exists in the DB.
        """
        pg = self._pg

        # Preferred: pg_store exposes news_id_exists()
        if hasattr(pg, "news_id_exists"):
            result = pg.news_id_exists(article_id)  # type: ignore[union-attr]
            # Support both sync and async implementations
            if hasattr(result, "__await__"):
                return bool(await result)
            return bool(result)

        # Fallback: pg_store exposes article_exists()
        if hasattr(pg, "article_exists"):
            result = pg.article_exists(article_id)  # type: ignore[union-attr]
            if hasattr(result, "__await__"):
                return bool(await result)
            return bool(result)

        # Unknown pg_store interface — assume not in DB
        logger.warning(
            "dedup_pg_store_no_compatible_method",
            available_methods=[m for m in dir(pg) if not m.startswith("_")],
        )
        return False
