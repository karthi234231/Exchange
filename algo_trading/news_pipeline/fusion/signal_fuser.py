"""
SignalFuser: orchestrates data loading and signal generation for all watched symbols.

Per-symbol workflow:
1. Load recent news events from PGStore (last cfg.NEWS_TTL_HOURS hours)
2. Load today's bulk/block deals from PGStore
3. Load delivery history (30 days) from PGStore
4. Get current India VIX from cache or NSElibFetcher
5. Call AlphaScorer to compute ComponentScores
6. Build FusedSignal dataclass
7. Write to PGStore fused_signals table
8. Update in-memory signal_cache

Exposes:
- fuse_all(symbols) → list[FusedSignal]   (called by Prefect fusion_flow)
- fuse_one(symbol) → FusedSignal | None
- get_cached_signal(symbol) → FusedSignal | None   (called by FastAPI)
"""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import structlog

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.models import FusedSignal, new_signal_id
from algo_trading.news_pipeline.fusion.alpha_scorer import AlphaScorer, ComponentScores
from algo_trading.news_pipeline.storage.cache import signal_cache

log = structlog.get_logger(__name__)

# MARKET is a pseudo-symbol used for macro/index news that affects all symbols.
_MACRO_SYMBOL = "MARKET"


class SignalFuser:
    """
    Orchestrates data loading → scoring → storage for every watched symbol.

    Intended to be instantiated once per process and reused across Prefect task
    invocations.  Thread-safety note: async methods must be called from the same
    event loop; ``get_cached_signal`` is synchronous and thread-safe via the
    underlying TTLCache lock.
    """

    def __init__(self, pg_store, vix_fetcher=None) -> None:
        """
        Parameters
        ----------
        pg_store:
            A connected ``PGStore`` instance.
        vix_fetcher:
            An optional ``NSElibFetcher`` instance used to pull live India VIX.
            When None, VIX defaults to the last successfully fetched value or
            None (no VIX adjustment applied).
        """
        self._pg = pg_store
        self._vix = vix_fetcher
        self._scorer = AlphaScorer()
        self._last_vix: float | None = None

    # ------------------------------------------------------------------
    # Public: fuse all symbols
    # ------------------------------------------------------------------

    async def fuse_all(self, symbols: list[str]) -> list[FusedSignal]:
        """
        Fuse signals for every symbol in *symbols*.

        Runs each symbol sequentially to avoid overwhelming the DB pool.
        Logs a summary counter of signal types at the end.

        Returns a list of successfully generated FusedSignal objects; symbols
        that produce no data (all scores zero, no events found) are omitted.
        """
        results: list[FusedSignal] = []
        for symbol in symbols:
            try:
                signal = await self.fuse_one(symbol)
                if signal is not None:
                    results.append(signal)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "signal_fuser.fuse_all.symbol_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        # Summary log.
        type_counts: Counter[str] = Counter(s.signal_type for s in results)
        log.info(
            "signal_fuser.fuse_all.complete",
            total=len(results),
            STRONG_BUY=type_counts.get("STRONG_BUY", 0),
            BUY=type_counts.get("BUY", 0),
            NEUTRAL=type_counts.get("NEUTRAL", 0),
            SELL=type_counts.get("SELL", 0),
            STRONG_SELL=type_counts.get("STRONG_SELL", 0),
        )
        return results

    # ------------------------------------------------------------------
    # Public: fuse one symbol
    # ------------------------------------------------------------------

    async def fuse_one(self, symbol: str) -> FusedSignal | None:
        """
        Full pipeline for a single symbol.

        Returns None if all component scores are exactly 0.0 and no
        news/deal/delivery data was found (prevents writing meaningless signals).
        """
        log.debug("signal_fuser.fuse_one.start", symbol=symbol)

        # 1. Load news events and compute news score.
        news_events = await self._load_news(symbol)
        news_score = self._scorer.compute_news_score(news_events)

        # 2. Load bulk/block deals and compute deal score.
        deals = await self._pg.get_today_bulk_deals(symbol=symbol)
        bulk_deal_score = self._scorer.compute_bulk_deal_score(deals)

        # 3. Load delivery history and compute delivery score.
        delivery_history = await self._pg.get_delivery_history(
            symbol=symbol,
            days=cfg.DELIVERY_ZSCORE_WINDOW_DAYS,
        )
        delivery_score = self._scorer.compute_delivery_score(delivery_history)

        # 4. Fetch VIX and compute adjustment.
        vix_value = await self._get_vix()
        vix_adjustment = self._scorer.compute_vix_adjustment(vix_value)

        # Guard: skip if there is genuinely nothing to score.
        has_data = bool(news_events or deals or len(delivery_history) > 1)
        if not has_data and news_score == 0.0 and bulk_deal_score == 0.0 and delivery_score == 0.0:
            log.debug("signal_fuser.fuse_one.no_data", symbol=symbol)
            return None

        # 5. Combine into ComponentScores.
        scores: ComponentScores = self._scorer.combine(
            news_score=news_score,
            bulk_deal_score=bulk_deal_score,
            delivery_score=delivery_score,
            vix_adjustment=vix_adjustment,
        )

        # 6. Collect contributing event IDs.
        contributing_ids = [
            str(ev.get("id", ""))
            for ev in news_events
            if ev.get("id")
        ]

        # 7. Build FusedSignal.
        signal = self._build_signal(symbol, scores, contributing_ids)

        # 8. Persist to DB and cache.
        await self._save_signal(signal)

        log.info(
            "signal_fuser.fuse_one.complete",
            symbol=symbol,
            signal_type=signal.signal_type,
            alpha_score=round(signal.alpha_score, 4),
            contributing_events=len(contributing_ids),
        )
        return signal

    # ------------------------------------------------------------------
    # VIX fetching with caching
    # ------------------------------------------------------------------

    async def _get_vix(self) -> float | None:
        """
        Attempt to fetch a fresh India VIX from the injected NSElibFetcher.

        On success the value is cached in ``self._last_vix`` so subsequent
        symbols in the same fuse_all() run reuse it without extra network calls.
        On failure the last known value is returned (may be None on first run).
        """
        if self._vix is None:
            return self._last_vix

        try:
            vix_value = await self._vix.get_india_vix()
            if vix_value is not None:
                self._last_vix = vix_value
                log.debug("signal_fuser.vix_fetched", vix=vix_value)
            else:
                log.debug("signal_fuser.vix_none_returned", using_cached=self._last_vix)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "signal_fuser.vix_fetch_failed",
                error=str(exc),
                using_cached=self._last_vix,
            )

        return self._last_vix

    # ------------------------------------------------------------------
    # News loading with macro merge
    # ------------------------------------------------------------------

    async def _load_news(self, symbol: str) -> list[dict]:
        """
        Load symbol-specific news + macro (MARKET) news from PGStore.

        Merges both lists, deduplicates by ``id``, preserving order
        (symbol-specific first, then macro).
        """
        symbol_events: list[dict] = await self._pg.get_recent_news(
            symbols=[symbol],
            hours=cfg.NEWS_TTL_HOURS,
        )

        macro_events: list[dict] = []
        if symbol != _MACRO_SYMBOL:
            try:
                macro_events = await self._pg.get_recent_news(
                    symbols=[_MACRO_SYMBOL],
                    hours=cfg.NEWS_TTL_HOURS,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "signal_fuser.load_news.macro_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        # Deduplicate: symbol-specific events take precedence.
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for ev in symbol_events + macro_events:
            ev_id = str(ev.get("id", ""))
            if ev_id and ev_id in seen_ids:
                continue
            seen_ids.add(ev_id)
            merged.append(ev)

        log.debug(
            "signal_fuser.load_news",
            symbol=symbol,
            symbol_events=len(symbol_events),
            macro_events=len(macro_events),
            merged=len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _save_signal(self, signal: FusedSignal) -> None:
        """
        Write the signal to PostgreSQL and update the in-memory TTLCache.

        DB failure is logged but does not raise — the signal is still cached
        in memory so the FastAPI endpoint can serve it within the same process.
        """
        try:
            await self._pg.upsert_signal(signal)
            log.debug("signal_fuser.signal_saved_db", symbol=signal.symbol, id=signal.id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "signal_fuser.signal_save_db_failed",
                symbol=signal.symbol,
                error=str(exc),
            )

        # Always update the in-memory cache regardless of DB outcome.
        # Cache as a dict so FastAPI routes can use it directly via _dict_to_signal_response.
        signal_cache.set(
            key=signal.symbol,
            value=signal.to_dict(),
            ttl_seconds=signal.ttl_minutes * 60,
        )
        log.debug("signal_fuser.signal_cached", symbol=signal.symbol)

    # ------------------------------------------------------------------
    # Cache retrieval (synchronous — called by FastAPI)
    # ------------------------------------------------------------------

    def get_cached_signal(self, symbol: str) -> FusedSignal | None:
        """
        Return the in-memory cached FusedSignal for *symbol*.

        The cache stores signal dicts (from ``signal.to_dict()``).
        This method reconstructs a FusedSignal from the cached dict.

        Returns None when:
        - No signal has ever been generated for the symbol in this process, or
        - The TTLCache entry has expired (the underlying TTL cache handles eviction).

        The explicit ``is_expired()`` check is a second safety layer in case the
        TTLCache TTL and the signal TTL drift apart.
        """
        from datetime import datetime, timedelta, timezone

        cached_dict: dict | None = signal_cache.get(symbol)
        if cached_dict is None:
            return None

        # Check signal TTL manually from the dict to avoid reconstructing the object
        # just for expiry checking.
        try:
            gen_at_raw = cached_dict.get("generated_at")
            ttl_minutes = int(cached_dict.get("ttl_minutes", 60))
            if isinstance(gen_at_raw, str):
                gen_at = datetime.fromisoformat(gen_at_raw)
            elif isinstance(gen_at_raw, datetime):
                gen_at = gen_at_raw
            else:
                gen_at = datetime.now(tz=timezone.utc)
            if gen_at.tzinfo is None:
                gen_at = gen_at.replace(tzinfo=timezone.utc)
            expiry = gen_at + timedelta(minutes=ttl_minutes)
            if datetime.now(tz=timezone.utc) >= expiry:
                signal_cache.delete(symbol)
                return None
        except Exception:
            # On any parsing error, treat as expired
            signal_cache.delete(symbol)
            return None

        # Reconstruct FusedSignal from dict
        try:
            gen_at_val = cached_dict.get("generated_at")
            if isinstance(gen_at_val, str):
                generated_at = datetime.fromisoformat(gen_at_val)
            elif isinstance(gen_at_val, datetime):
                generated_at = gen_at_val
            else:
                generated_at = datetime.now(tz=timezone.utc)
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)

            return FusedSignal(
                id=cached_dict["id"],
                generated_at=generated_at,
                symbol=cached_dict["symbol"],
                signal_type=cached_dict["signal_type"],
                alpha_score=float(cached_dict["alpha_score"]),
                news_score=float(cached_dict["news_score"]),
                bulk_deal_score=float(cached_dict["bulk_deal_score"]),
                delivery_score=float(cached_dict["delivery_score"]),
                vix_score=float(cached_dict["vix_score"]),
                contributing_events=list(cached_dict.get("contributing_events", [])),
                ttl_minutes=int(cached_dict["ttl_minutes"]),
            )
        except Exception as exc:
            log.warning(
                "signal_fuser.get_cached_signal.reconstruct_failed",
                symbol=symbol,
                error=str(exc),
            )
            signal_cache.delete(symbol)
            return None

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        scores: ComponentScores,
        contributing_ids: list[str],
    ) -> FusedSignal:
        """
        Construct a FusedSignal dataclass from AlphaScorer's ComponentScores.

        The signal ID is a fresh UUID4 so every fuse cycle produces a unique,
        auditable record in the fused_signals table.
        """
        return FusedSignal(
            id=new_signal_id(),
            generated_at=datetime.now(tz=timezone.utc),
            symbol=symbol,
            signal_type=scores.signal_type,
            alpha_score=scores.alpha_score,
            news_score=scores.news_score,
            bulk_deal_score=scores.bulk_deal_score,
            delivery_score=scores.delivery_score,
            vix_score=scores.vix_score,
            contributing_events=contributing_ids,
            ttl_minutes=cfg.SIGNAL_TTL_MINUTES,
        )
