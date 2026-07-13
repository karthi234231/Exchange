"""
Prefect 2.x orchestration layer for the NSE news intelligence pipeline.

5 flows:
    1. ingest_flow          — every 5 min — fetch Pulse + nselib + RSS → raw to PG
    2. process_flow         — every 5 min — NER + FinBERT + tagging → processed events to PG
    3. fusion_flow          — every 5 min — SignalFuser for all watched symbols → signals to PG
    4. history_flow         — daily 16:30 IST (11:00 UTC) — bhavcopy + OHLCV via jugaad → Parquet
    5. backtest_eval_flow   — daily 20:00 IST (14:30 UTC) — evaluate yesterday signals vs actual

All flows:
    - retries=3 / retry_delay_seconds=30 on tasks; retries=1 on flows
    - Use PipelineTracker.run() as context manager
    - Log start/end with structlog (outside Prefect) and get_run_logger() inside tasks
    - Never crash the whole flow on a single-item error (per-item try/except)
    - Support serve mode (no server) and deployment mode (PREFECT_API_URL set)
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import structlog
from prefect import flow, task, get_run_logger
from prefect.schedules import IntervalSchedule, CronSchedule

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.models import (
    BulkDealRecord,
    DeliveryRecord,
    FusedSignal,
    ProcessedNewsEvent,
    SignalPerformance,
    make_id,
    new_signal_id,
)
from algo_trading.news_pipeline.storage.pg_store import PGStore
from algo_trading.news_pipeline.storage.parquet_store import ParquetStore
from algo_trading.news_pipeline.storage.cache import news_id_cache
from algo_trading.news_pipeline.monitoring.tracker import PipelineTracker

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ZMQ publisher — pushes FusedSignal JSON to DailyOrchestrator subscriber
# Port 5572: fusion_flow PUB → NewsPipelineClient SUB (in orchestrator process)
# ---------------------------------------------------------------------------

_SIGNAL_PUB_PORT: int = 5572
_signal_pub_socket: object | None = None  # lazy-init on first fusion run


def _get_signal_publisher() -> object | None:
    """
    Lazily create a ZMQ PUB socket bound to port 5572.

    Returns None silently if pyzmq is not installed so that flows.py remains
    importable in test environments without ZMQ.
    """
    global _signal_pub_socket
    if _signal_pub_socket is not None:
        return _signal_pub_socket
    try:
        import zmq  # type: ignore[import]
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 500)
        sock.bind(f"tcp://127.0.0.1:{_SIGNAL_PUB_PORT}")
        _signal_pub_socket = sock
        log.info("fusion_flow.signal_publisher_bound", port=_SIGNAL_PUB_PORT)
    except ImportError:
        log.debug("fusion_flow.zmq_unavailable — signal publishing disabled")
    except Exception as exc:
        log.warning("fusion_flow.signal_publisher_bind_failed", error=str(exc))
    return _signal_pub_socket


def _publish_signal(signal: FusedSignal) -> None:
    """
    Publish a FusedSignal to the ZMQ PUB socket (fire-and-forget, NOBLOCK).

    Called inside fusion_flow after each signal is stored.  Silently drops
    the message if the send buffer is full (NOBLOCK) — the orchestrator will
    use the next cycle's signal instead.
    """
    publisher = _get_signal_publisher()
    if publisher is None:
        return
    try:
        import zmq  # type: ignore[import]
        publisher.send_string(json.dumps(signal.to_dict()), flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
    except Exception:
        pass  # EAGAIN (buffer full) or context terminated — ignore

# ── Shared state ─────────────────────────────────────────────────────────────

_pg: PGStore | None = None
_parquet: ParquetStore | None = None
_models_preloaded: bool = False   # guard: preload FinBERT + spaCy only once

# In-process queue populated by ingest_flow, consumed by process_flow.
# Each entry is a RawNewsItem.to_dict().  Protected from concurrent writes by
# asyncio's cooperative multitasking (no thread-safety needed here).
_raw_queue: list[dict] = []


async def _get_pg() -> PGStore:
    """Lazy singleton PGStore — connects once per process lifetime."""
    global _pg
    if _pg is None:
        _pg = PGStore()
        await _pg.connect()
    return _pg


def _get_parquet() -> ParquetStore:
    """Lazy singleton ParquetStore — no async needed."""
    global _parquet
    if _parquet is None:
        _parquet = ParquetStore()
    return _parquet


def _preload_models() -> None:
    """
    Eagerly load FinBERT and spaCy models on the Prefect worker thread.

    Called once at the start of process_flow so that the first real article
    does not block for 20-60 seconds waiting for model download/load.
    Safe to call multiple times (guarded by _models_preloaded flag).
    """
    global _models_preloaded
    if _models_preloaded:
        return

    log.info("flows.preloading_nlp_models")
    try:
        from algo_trading.news_pipeline.processing.finbert_adapter import FinBERTAdapter
        FinBERTAdapter.preload()
        log.info("flows.finbert_preloaded")
    except Exception as exc:
        log.warning("flows.finbert_preload_failed", error=str(exc))

    try:
        from algo_trading.news_pipeline.processing.ner_pipeline import NERPipeline
        NERPipeline.preload()
        log.info("flows.spacy_preloaded")
    except Exception as exc:
        log.warning("flows.spacy_preload_failed", error=str(exc))

    _models_preloaded = True


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 1: INGEST FLOW
# ═══════════════════════════════════════════════════════════════════════════════

@task(retries=3, retry_delay_seconds=30, name="fetch-pulse")
async def fetch_pulse_task() -> list[dict]:
    """Fetch news from Zerodha Pulse.  Returns list of RawNewsItem.to_dict()."""
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.pulse_scraper import PulseScraper

    scraper = PulseScraper()
    try:
        items, result = await scraper.fetch()
        logger.info(f"pulse_fetched count={len(items)} success={result.success}")
        log.info("pulse_fetched", count=len(items), success=result.success)
        return [i.to_dict() for i in items]
    except Exception as exc:
        logger.warning(f"fetch_pulse_task error: {exc}")
        log.warning("fetch_pulse_task.error", error=str(exc))
        return []
    finally:
        try:
            await scraper.close()
        except Exception:
            pass


@task(retries=3, retry_delay_seconds=30, name="fetch-nselib")
async def fetch_nselib_task() -> list[dict]:
    """Fetch bulk deals + delivery from nselib.  Returns list of model dicts."""
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.nselib_fetcher import NSElibFetcher

    fetcher = NSElibFetcher()
    try:
        items, result = await fetcher.fetch()
        logger.info(f"nselib_fetched count={len(items)} success={result.success}")
        log.info("nselib_fetched", count=len(items), success=result.success)
        return [i.to_dict() for i in items]
    except Exception as exc:
        logger.warning(f"fetch_nselib_task error: {exc}")
        log.warning("fetch_nselib_task.error", error=str(exc))
        return []


@task(retries=2, retry_delay_seconds=60, name="fetch-rss-fallback")
async def fetch_rss_task() -> list[dict]:
    """RSS fallback — always runs as backup source."""
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.rss_fallback import RSSFallback

    fetcher = RSSFallback()
    try:
        items, result = await fetcher.fetch()
        logger.info(f"rss_fetched count={len(items)} success={result.success}")
        log.info("rss_fetched", count=len(items), success=result.success)
        return [i.to_dict() for i in items]
    except Exception as exc:
        logger.warning(f"fetch_rss_task error: {exc}")
        log.warning("fetch_rss_task.error", error=str(exc))
        return []


@task(name="store-raw-items")
async def store_raw_items_task(
    pulse_items: list[dict],
    nselib_items: list[dict],
    rss_items: list[dict],
) -> int:
    """
    Deduplicate all sources by item id.

    - BulkDealRecord dicts  → upsert to PG bulk_deals table + _raw_queue skip
    - DeliveryRecord dicts  → upsert to PG delivery_data table
    - RawNewsItem dicts     → dedup via news_id_cache + push to _raw_queue for NLP

    Returns count of net-new items written (news items enqueued + deals stored).
    """
    logger = get_run_logger()
    pg = await _get_pg()

    stored = 0
    seen_ids: set[str] = set()

    # ── Process nselib structured records (BulkDealRecord / DeliveryRecord) ──
    for item_dict in nselib_items:
        # Distinguish by presence of model-specific keys
        if "deal_type" in item_dict:
            # BulkDealRecord
            try:
                deal_id = item_dict.get("id", "")
                if deal_id in seen_ids:
                    continue
                seen_ids.add(deal_id)

                deal = BulkDealRecord(
                    id=item_dict["id"],
                    deal_date=_parse_dt(item_dict.get("deal_date")),
                    symbol=item_dict["symbol"],
                    client_name=item_dict.get("client_name", "UNKNOWN"),
                    deal_type=item_dict["deal_type"],
                    direction=item_dict["direction"],
                    quantity=int(item_dict.get("quantity", 0)),
                    price=float(item_dict.get("price", 0.0)),
                    deal_value_cr=float(item_dict.get("deal_value_cr", 0.0)),
                    fetched_at=_parse_dt(item_dict.get("fetched_at")),
                )
                await pg.upsert_bulk_deal(deal)
                stored += 1
                logger.debug(f"stored bulk deal {deal.id} symbol={deal.symbol}")
            except Exception as exc:
                logger.warning(f"bulk_deal store failed: {exc}")
                log.warning("store_raw.bulk_deal_failed", error=str(exc))

        elif "delivery_qty" in item_dict:
            # DeliveryRecord
            try:
                rec = DeliveryRecord(
                    trade_date=_parse_dt(item_dict.get("trade_date")),
                    symbol=item_dict["symbol"],
                    delivery_qty=int(item_dict.get("delivery_qty", 0)),
                    traded_qty=int(item_dict.get("traded_qty", 0)),
                    delivery_pct=float(item_dict.get("delivery_pct", 0.0)),
                    fetched_at=_parse_dt(item_dict.get("fetched_at")),
                )
                await pg.upsert_delivery(rec)
                stored += 1
            except Exception as exc:
                logger.warning(f"delivery_record store failed: {exc}")
                log.warning("store_raw.delivery_failed", error=str(exc))

    # ── Process news items from Pulse + RSS (and any jugaad RawNewsItems) ──
    news_sources: list[list[dict]] = [pulse_items, rss_items]
    for source_items in news_sources:
        for item_dict in source_items:
            try:
                item_id = item_dict.get("id", "")
                if not item_id:
                    continue
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Fast in-memory dedup check
                if news_id_cache.exists(item_id):
                    continue

                # Mark as seen with 48-hour TTL
                news_id_cache.set(item_id, True, ttl_seconds=cfg.DEDUP_TTL_HOURS * 3600)

                # Enqueue for NLP processing
                _raw_queue.append(item_dict)
                stored += 1
            except Exception as exc:
                logger.warning(f"news_item queue failed: {exc}")
                log.warning("store_raw.news_item_failed", error=str(exc))

    logger.info(f"store_raw_items: stored/enqueued={stored} queue_depth={len(_raw_queue)}")
    log.info(
        "store_raw_items.done",
        stored=stored,
        queue_depth=len(_raw_queue),
    )
    return stored


@flow(
    name="ingest-flow",
    description="Fetch news from Pulse + nselib + RSS, store raw to PostgreSQL",
    retries=1,
)
async def ingest_flow() -> dict:
    """
    Runs every 5 minutes.

    Fetches all 3 sources in parallel.  Pulse is primary news source; RSS is
    always run as backup.  nselib provides structured bulk/block deals and
    delivery data (market hours or post-close).

    Returns summary: {pulse_count, nselib_count, rss_count, stored_count}
    """
    log.info("ingest_flow.start")
    pg = await _get_pg()

    async with PipelineTracker.run("ingest-flow", pg_store=pg) as tracker:
        # Fire all three fetches in parallel
        pulse_fut, nselib_fut, rss_fut = await asyncio.gather(
            fetch_pulse_task(),
            fetch_nselib_task(),
            fetch_rss_task(),
            return_exceptions=True,
        )

        # Unwrap results — treat exceptions as empty lists
        pulse_items: list[dict] = pulse_fut if isinstance(pulse_fut, list) else []
        nselib_items: list[dict] = nselib_fut if isinstance(nselib_fut, list) else []
        rss_items: list[dict] = rss_fut if isinstance(rss_fut, list) else []

        if isinstance(pulse_fut, Exception):
            tracker.add_error(f"pulse fetch exception: {pulse_fut}")
        if isinstance(nselib_fut, Exception):
            tracker.add_error(f"nselib fetch exception: {nselib_fut}")
        if isinstance(rss_fut, Exception):
            tracker.add_error(f"rss fetch exception: {rss_fut}")

        stored_count = await store_raw_items_task(pulse_items, nselib_items, rss_items)
        tracker.add_records(stored_count)

        summary = {
            "pulse_count": len(pulse_items),
            "nselib_count": len(nselib_items),
            "rss_count": len(rss_items),
            "stored_count": stored_count,
        }

    log.info("ingest_flow.done", **summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 2: PROCESS FLOW
# ═══════════════════════════════════════════════════════════════════════════════

@task(name="fetch-unprocessed")
async def fetch_unprocessed_task(limit: int = 100) -> list[dict]:
    """
    Drain up to *limit* items from the module-level _raw_queue.

    The queue is populated by store_raw_items_task during ingest_flow.
    Items are raw RawNewsItem dicts that haven't yet been through the NLP
    pipeline.

    Returns the drained batch (removed from _raw_queue).
    """
    logger = get_run_logger()
    batch: list[dict] = []
    # Pop from front up to limit; works under asyncio cooperative multitasking
    while _raw_queue and len(batch) < limit:
        batch.append(_raw_queue.pop(0))
    logger.info(f"fetch_unprocessed: drained {len(batch)} items; {len(_raw_queue)} remain")
    log.info("fetch_unprocessed.done", batch=len(batch), remaining=len(_raw_queue))
    return batch


@task(name="run-ner-finbert")
async def run_nlp_task(raw_items: list[dict]) -> list[dict]:
    """
    Run NER + FinBERT + event tagging on each raw item.

    Pipeline per item:
        1. NERPipeline.extract_symbols(headline + body, hints=symbol_hints)
        2. NERPipeline.extract_entities(headline + body)
        3. FinBERTAdapter.score(headline)
        4. EventTagger.tag(headline, body, sentiment_label)
        5. Build ProcessedNewsEvent
        6. Skip items with no symbols found and no macro keywords

    Wraps each item individually — a single bad item never kills the batch.

    Returns list of ProcessedNewsEvent.to_dict().
    """
    logger = get_run_logger()
    from algo_trading.news_pipeline.processing.ner_pipeline import NERPipeline
    from algo_trading.news_pipeline.processing.finbert_adapter import FinBERTAdapter
    from algo_trading.news_pipeline.processing.event_tagger import EventTagger

    ner = NERPipeline()
    finbert = FinBERTAdapter()
    tagger = EventTagger()

    processed: list[dict] = []
    skipped = 0

    for item in raw_items:
        try:
            headline: str = item.get("headline", "")
            body: str = item.get("body", "")
            hints: list[str] = item.get("symbol_hints", [])
            combined_text = f"{headline} {body}"

            # 1. NER symbol extraction
            symbols = ner.extract_symbols(combined_text, hints=hints)

            # 2. NER entity extraction
            entities = ner.extract_entities(combined_text)

            # 3. Check if macro news (allow through even without symbols)
            is_macro = ner.is_macro_news(combined_text)

            if not symbols and not is_macro:
                skipped += 1
                logger.debug(f"nlp skip (no symbols, non-macro): {headline[:60]}")
                continue

            # For macro news with no extracted symbols, assign a synthetic marker
            if not symbols and is_macro:
                symbols = ["MACRO"]

            # 4. FinBERT sentiment
            score, label, confidence = finbert.score(headline)

            # 5. Event tagging
            event_type, _impact, tags = tagger.tag(headline, body, label)

            # 6. Build ProcessedNewsEvent
            published_at_raw = item.get("published_at")
            fetched_at_raw = item.get("fetched_at")
            published_at = _parse_dt(published_at_raw) or datetime.now(tz=timezone.utc)
            fetched_at = _parse_dt(fetched_at_raw) or datetime.now(tz=timezone.utc)

            event = ProcessedNewsEvent(
                id=item["id"],
                source=item.get("source", "unknown"),
                headline=headline,
                url=item.get("url", ""),
                published_at=published_at,
                fetched_at=fetched_at,
                symbols=symbols[: cfg.MAX_SYMBOLS_PER_ARTICLE],
                event_type=event_type,
                sentiment=label,
                sentiment_score=score,
                sentiment_confidence=confidence,
                raw_body=body,
                entities=entities,
                tags=tags,
            )
            processed.append(event.to_dict())

        except Exception as exc:
            skipped += 1
            logger.warning(f"nlp_task item failed: {exc}")
            log.warning("run_nlp_task.item_failed", error=str(exc))

    logger.info(f"run_nlp: processed={len(processed)} skipped={skipped}")
    log.info("run_nlp_task.done", processed=len(processed), skipped=skipped)
    return processed


@task(name="store-processed-events")
async def store_processed_task(events: list[dict]) -> int:
    """Upsert ProcessedNewsEvent dicts to PG news_events table.  Returns count stored."""
    logger = get_run_logger()
    if not events:
        return 0

    pg = await _get_pg()
    stored = 0

    for event_dict in events:
        try:
            published_at = _parse_dt(event_dict.get("published_at"))
            fetched_at = _parse_dt(event_dict.get("fetched_at"))

            event = ProcessedNewsEvent(
                id=event_dict["id"],
                source=event_dict.get("source", "unknown"),
                headline=event_dict.get("headline", ""),
                url=event_dict.get("url", ""),
                published_at=published_at or datetime.now(tz=timezone.utc),
                fetched_at=fetched_at or datetime.now(tz=timezone.utc),
                symbols=event_dict.get("symbols", []),
                event_type=event_dict.get("event_type", "GENERIC"),
                sentiment=event_dict.get("sentiment", "neutral"),
                sentiment_score=float(event_dict.get("sentiment_score", 0.0)),
                sentiment_confidence=float(event_dict.get("sentiment_confidence", 0.0)),
                raw_body=event_dict.get("raw_body", ""),
                entities=event_dict.get("entities", {}),
                tags=event_dict.get("tags", []),
            )
            await pg.upsert_news_event(event)
            stored += 1
        except Exception as exc:
            logger.warning(f"store_processed item failed: {exc}")
            log.warning("store_processed_task.item_failed", error=str(exc))

    logger.info(f"store_processed: stored={stored}/{len(events)}")
    log.info("store_processed_task.done", stored=stored, total=len(events))
    return stored


@flow(
    name="process-flow",
    description="Run NER + FinBERT + event tagging on new raw items",
    retries=1,
)
async def process_flow() -> dict:
    """
    Runs every 5 minutes (offset 2 min after ingest_flow).

    Drains _raw_queue → NLP pipeline → upsert to PG.

    Returns {processed_count, skipped_count}.
    """
    log.info("process_flow.start")
    # Eager model load — blocks once at startup, free on every subsequent call.
    _preload_models()
    pg = await _get_pg()

    async with PipelineTracker.run("process-flow", pg_store=pg) as tracker:
        raw_items = await fetch_unprocessed_task(limit=100)

        if not raw_items:
            log.info("process_flow.no_items")
            return {"processed_count": 0, "skipped_count": 0}

        events = await run_nlp_task(raw_items)
        skipped_count = len(raw_items) - len(events)

        stored = await store_processed_task(events)
        tracker.add_records(stored)

        summary = {
            "processed_count": stored,
            "skipped_count": skipped_count,
        }

    log.info("process_flow.done", **summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 3: FUSION FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def _alpha_to_signal_type(alpha: float) -> str:
    """Map a float alpha score to a named signal type using config thresholds."""
    if alpha >= cfg.STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    if alpha >= cfg.BUY_THRESHOLD:
        return "BUY"
    if alpha <= cfg.STRONG_SELL_THRESHOLD:
        return "STRONG_SELL"
    if alpha <= cfg.SELL_THRESHOLD:
        return "SELL"
    return "NEUTRAL"


async def _fuse_one_symbol(symbol: str, pg: PGStore) -> FusedSignal | None:
    """
    Compute a FusedSignal for one symbol from PG data.

    Sub-scores:
        news_score:      weighted mean sentiment_score of recent events (last 4 h)
        bulk_deal_score: direction-weighted deal-value ratio for today's deals
        delivery_score:  today's delivery_pct normalised to [-1, 1] (z-score proxy)
        vix_score:       VIX dampening adjustment (0.0 if VIX unavailable)

    alpha = w_news*news + w_bulk*bulk + w_delivery*delivery + w_vix*vix
    Clamped to [-1, 1].
    """
    now_utc = datetime.now(tz=timezone.utc)

    # ── 1. News score ─────────────────────────────────────────────────────────
    try:
        recent_news = await pg.get_recent_news(
            symbols=[symbol],
            hours=cfg.NEWS_TTL_HOURS,
            limit=20,
        )
    except Exception as exc:
        log.warning("fusion.news_fetch_failed", symbol=symbol, error=str(exc))
        recent_news = []

    news_score = 0.0
    contributing_event_ids: list[str] = []
    if recent_news:
        scores = [float(row.get("sentiment_score", 0.0)) for row in recent_news]
        # Confidence-weighted average
        confs = [max(float(row.get("sentiment_confidence", 0.5)), 0.01) for row in recent_news]
        total_conf = sum(confs)
        news_score = sum(s * c for s, c in zip(scores, confs)) / total_conf if total_conf > 0 else 0.0
        contributing_event_ids = [row["id"] for row in recent_news if row.get("id")]

    # ── 2. Bulk deal score ────────────────────────────────────────────────────
    try:
        deals = await pg.get_today_bulk_deals(symbol=symbol)
    except Exception as exc:
        log.warning("fusion.bulk_deals_failed", symbol=symbol, error=str(exc))
        deals = []

    bulk_deal_score = 0.0
    if deals:
        buy_value = sum(float(d.get("deal_value_cr", 0.0)) for d in deals if d.get("direction") == "BUY")
        sell_value = sum(float(d.get("deal_value_cr", 0.0)) for d in deals if d.get("direction") == "SELL")
        total_value = buy_value + sell_value
        if total_value > 0:
            # Normalise to [-1, 1]: full buy -> +1, full sell -> -1
            bulk_deal_score = (buy_value - sell_value) / total_value
            # Scale by deal size significance
            high_threshold = max(cfg.BULK_DEAL_HIGH_CR, cfg.BLOCK_DEAL_HIGH_CR)
            if total_value >= high_threshold:
                bulk_deal_score = bulk_deal_score  # already in [-1, 1]
            else:
                # Partial weight proportional to deal size vs threshold
                bulk_deal_score *= min(total_value / high_threshold, 1.0)

    # ── 3. Delivery score ─────────────────────────────────────────────────────
    try:
        delivery_history = await pg.get_delivery_history(
            symbol=symbol,
            days=cfg.DELIVERY_ZSCORE_WINDOW_DAYS,
        )
    except Exception as exc:
        log.warning("fusion.delivery_failed", symbol=symbol, error=str(exc))
        delivery_history = []

    delivery_score = 0.0
    if delivery_history:
        pcts = [float(d.get("delivery_pct", 0.0)) for d in delivery_history]
        # Most recent entry is the "today" reading
        today_pct = pcts[0] if pcts else 0.0
        if len(pcts) >= 2:
            mean_pct = sum(pcts) / len(pcts)
            std_pct = (sum((p - mean_pct) ** 2 for p in pcts) / len(pcts)) ** 0.5
            if std_pct > 0.0:
                z = (today_pct - mean_pct) / std_pct
                # Clamp z-score to [-3, 3] then normalise to [-1, 1]
                delivery_score = max(-1.0, min(1.0, z / 3.0))
        else:
            # Only one data point: just use pct as a raw fraction of 100
            delivery_score = max(-1.0, min(1.0, (today_pct - 50.0) / 50.0))

    # ── 4. VIX score ──────────────────────────────────────────────────────────
    vix_score = 0.0
    try:
        from algo_trading.news_pipeline.ingestion.nselib_fetcher import NSElibFetcher
        nselib = NSElibFetcher()
        vix = await nselib.get_india_vix()
        if vix is not None and vix > cfg.VIX_HIGH_THRESHOLD:
            # High VIX dampens bullish signals: reduce alpha by dampen factor
            vix_score = -(1.0 - cfg.VIX_DAMPEN_FACTOR)  # e.g. -0.3 when VIX > 20
    except Exception as exc:
        log.debug("fusion.vix_fetch_failed", symbol=symbol, error=str(exc))

    # ── 5. Weighted alpha ─────────────────────────────────────────────────────
    alpha = (
        cfg.WEIGHT_NEWS * news_score
        + cfg.WEIGHT_BULK_DEAL * bulk_deal_score
        + cfg.WEIGHT_DELIVERY * delivery_score
        + cfg.WEIGHT_VIX * vix_score
    )
    alpha = max(-1.0, min(1.0, alpha))
    signal_type = _alpha_to_signal_type(alpha)

    return FusedSignal(
        id=new_signal_id(),
        generated_at=now_utc,
        symbol=symbol,
        signal_type=signal_type,
        alpha_score=round(alpha, 6),
        news_score=round(news_score, 6),
        bulk_deal_score=round(bulk_deal_score, 6),
        delivery_score=round(delivery_score, 6),
        vix_score=round(vix_score, 6),
        contributing_events=contributing_event_ids,
        ttl_minutes=cfg.SIGNAL_TTL_MINUTES,
    )


@task(name="fuse-signals")
async def fuse_signals_task(symbols: list[str]) -> list[dict]:
    """
    Compute a FusedSignal for each symbol via news + bulk deals + delivery + VIX.

    Each symbol is computed independently; failures are logged and skipped so
    one bad symbol never blocks the rest.

    Returns list of FusedSignal.to_dict().
    """
    logger = get_run_logger()
    pg = await _get_pg()

    results: list[dict] = []
    for symbol in symbols:
        try:
            signal = await _fuse_one_symbol(symbol, pg)
            if signal is not None:
                results.append(signal.to_dict())
                logger.debug(f"fused {symbol}: {signal.signal_type} alpha={signal.alpha_score:.4f}")
        except Exception as exc:
            logger.warning(f"fuse_signals_task symbol={symbol} failed: {exc}")
            log.warning("fuse_signals_task.symbol_failed", symbol=symbol, error=str(exc))

    logger.info(f"fuse_signals: generated {len(results)}/{len(symbols)} signals")
    log.info("fuse_signals_task.done", generated=len(results), symbols=len(symbols))
    return results


@flow(
    name="fusion-flow",
    description="Fuse news + bulk deals + delivery → alpha signals for all watched symbols",
    retries=1,
)
async def fusion_flow(symbols: list[str] | None = None) -> dict:
    """
    Runs every 5 minutes (offset 4 min after ingest_flow).

    symbols defaults to cfg.WATCHED_SYMBOLS.

    Returns {signals_generated, strong_buy_count, buy_count,
             neutral_count, sell_count, strong_sell_count}.
    """
    log.info("fusion_flow.start")
    pg = await _get_pg()
    parquet = _get_parquet()

    target_symbols = symbols or cfg.WATCHED_SYMBOLS

    async with PipelineTracker.run("fusion-flow", pg_store=pg) as tracker:
        signal_dicts = await fuse_signals_task(target_symbols)

        type_counts: dict[str, int] = defaultdict(int)
        fused_signals: list[FusedSignal] = []

        for sd in signal_dicts:
            try:
                published_at = _parse_dt(sd.get("generated_at")) or datetime.now(tz=timezone.utc)
                signal = FusedSignal(
                    id=sd["id"],
                    generated_at=published_at,
                    symbol=sd["symbol"],
                    signal_type=sd["signal_type"],
                    alpha_score=float(sd["alpha_score"]),
                    news_score=float(sd["news_score"]),
                    bulk_deal_score=float(sd["bulk_deal_score"]),
                    delivery_score=float(sd["delivery_score"]),
                    vix_score=float(sd["vix_score"]),
                    contributing_events=sd.get("contributing_events", []),
                    ttl_minutes=int(sd.get("ttl_minutes", cfg.SIGNAL_TTL_MINUTES)),
                )

                # Persist to PG
                await pg.upsert_signal(signal)

                # Update signal cache
                from algo_trading.news_pipeline.storage.cache import signal_cache
                signal_cache.set(
                    signal.symbol,
                    signal.to_dict(),
                    ttl_seconds=signal.ttl_minutes * 60,
                )

                fused_signals.append(signal)
                type_counts[signal.signal_type] += 1

                # Publish to DailyOrchestrator via ZMQ PUB (port 5572).
                # Fire-and-forget: if the orchestrator isn't running, the message
                # is silently dropped (ZMQ NOBLOCK + no subscribers = drop).
                _publish_signal(signal)

            except Exception as exc:
                log.warning("fusion_flow.signal_store_failed", error=str(exc))

        # Write today's signals to Parquet for historical record (sync I/O → executor)
        if fused_signals:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, parquet.write_signals, fused_signals, date.today()
                )
            except Exception as exc:
                log.warning("fusion_flow.parquet_write_failed", error=str(exc))

        tracker.add_records(len(fused_signals))

        summary = {
            "signals_generated": len(fused_signals),
            "strong_buy_count": type_counts.get("STRONG_BUY", 0),
            "buy_count": type_counts.get("BUY", 0),
            "neutral_count": type_counts.get("NEUTRAL", 0),
            "sell_count": type_counts.get("SELL", 0),
            "strong_sell_count": type_counts.get("STRONG_SELL", 0),
        }

    log.info("fusion_flow.done", **summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 4: HISTORY FLOW (daily EOD)
# ═══════════════════════════════════════════════════════════════════════════════

@task(name="download-bhavcopy")
async def download_bhavcopy_task(trade_date: date) -> dict:
    """
    Download full NSE bhavcopy via jugaad-data for *trade_date*.

    Writes to Parquet at {PARQUET_DIR}/bhavcopy/{trade_date}.parquet.
    Returns {rows_downloaded, file_path}.
    """
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.jugaad_fetcher import JugaadFetcher

    parquet = _get_parquet()
    fetcher = JugaadFetcher()

    try:
        df = await fetcher.fetch_bhavcopy(trade_date)
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.warning(f"bhavcopy empty for {trade_date} (holiday or data not yet available)")
            log.warning("download_bhavcopy.empty", trade_date=str(trade_date))
            return {"rows_downloaded": 0, "file_path": ""}

        # Normalise columns
        df.columns = [c.lower() for c in df.columns]

        # Write to Parquet via ParquetStore internal path mechanism (sync I/O → executor)
        import pyarrow as pa
        import pyarrow.parquet as pq
        from pathlib import Path

        base = Path(parquet._base_dir)
        out_path = base / "bhavcopy" / f"{trade_date.isoformat()}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df, preserve_index=False)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: pq.write_table(table, out_path, compression="snappy"),
        )
        rows = len(df)

        logger.info(f"bhavcopy downloaded: {rows} rows → {out_path}")
        log.info("download_bhavcopy.done", rows=rows, path=str(out_path))
        return {"rows_downloaded": rows, "file_path": str(out_path)}

    except Exception as exc:
        logger.error(f"download_bhavcopy_task failed: {exc}")
        log.error("download_bhavcopy_task.failed", trade_date=str(trade_date), error=str(exc))
        return {"rows_downloaded": 0, "file_path": ""}


@task(name="download-ohlcv-per-symbol")
async def download_ohlcv_task(symbols: list[str], trade_date: date) -> dict:
    """
    Download OHLCV for each watched symbol via jugaad-data for *trade_date*.

    Writes each symbol to {PARQUET_DIR}/ohlcv/{symbol}/{trade_date}.parquet.
    Returns {symbols_downloaded, symbols_failed}.
    """
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.jugaad_fetcher import JugaadFetcher

    parquet = _get_parquet()
    fetcher = JugaadFetcher()

    downloaded = 0
    failed = 0

    for symbol in symbols:
        # Skip index symbols — jugaad uses different API for those
        if symbol in ("NIFTY50", "BANKNIFTY", "SENSEX", "MACRO"):
            try:
                # Map to jugaad index name
                index_map = {
                    "NIFTY50": "NIFTY 50",
                    "BANKNIFTY": "NIFTY BANK",
                    "SENSEX": "SENSEX",
                }
                index_name = index_map.get(symbol, symbol)
                df = await fetcher.fetch_index_ohlcv(index_name, trade_date, trade_date)
                if df is not None and not (hasattr(df, "empty") and df.empty):
                    _sym, _df, _td = symbol, df, trade_date
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: parquet.write_ohlcv(_sym, _df, date=_td)
                    )
                    downloaded += 1
                    logger.debug(f"ohlcv downloaded: {symbol} (index)")
            except Exception as exc:
                failed += 1
                logger.warning(f"ohlcv index download failed for {symbol}: {exc}")
                log.warning("download_ohlcv.index_failed", symbol=symbol, error=str(exc))
            continue

        try:
            df = await fetcher.fetch_ohlcv(symbol, trade_date, trade_date)
            if df is None or (hasattr(df, "empty") and df.empty):
                failed += 1
                logger.warning(f"ohlcv empty for {symbol} on {trade_date}")
                continue
            _sym2, _df2, _td2 = symbol, df, trade_date
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: parquet.write_ohlcv(_sym2, _df2, date=_td2)
            )
            downloaded += 1
            logger.debug(f"ohlcv downloaded: {symbol} rows={len(df)}")
        except Exception as exc:
            failed += 1
            logger.warning(f"ohlcv download failed for {symbol}: {exc}")
            log.warning("download_ohlcv.symbol_failed", symbol=symbol, error=str(exc))

    result = {"symbols_downloaded": downloaded, "symbols_failed": failed}
    logger.info(f"download_ohlcv: {downloaded} downloaded, {failed} failed")
    log.info("download_ohlcv_task.done", **result)
    return result


@task(name="download-delivery-data")
async def download_delivery_task(trade_date: date) -> int:
    """
    Download delivery data via nselib for *trade_date*, upsert to PG.
    Returns count of rows upserted.
    """
    logger = get_run_logger()
    from algo_trading.news_pipeline.ingestion.nselib_fetcher import NSElibFetcher

    pg = await _get_pg()
    fetcher = NSElibFetcher()
    stored = 0

    try:
        items, result = await fetcher.fetch()
        logger.info(f"delivery download: {len(items)} records from nselib")

        for item in items:
            if not isinstance(item, DeliveryRecord):
                # Skip non-delivery records (BulkDealRecord)
                if hasattr(item, "delivery_qty"):
                    pass
                else:
                    continue
            try:
                if isinstance(item, DeliveryRecord):
                    await pg.upsert_delivery(item)
                    stored += 1
            except Exception as exc:
                logger.warning(f"delivery upsert failed: {exc}")
                log.warning("download_delivery.upsert_failed", error=str(exc))

    except Exception as exc:
        logger.error(f"download_delivery_task failed: {exc}")
        log.error("download_delivery_task.failed", trade_date=str(trade_date), error=str(exc))

    logger.info(f"download_delivery: upserted={stored}")
    log.info("download_delivery_task.done", stored=stored)
    return stored


@flow(
    name="history-flow",
    description="Daily EOD: download bhavcopy + OHLCV + delivery → Parquet + PostgreSQL",
    retries=2,
)
async def history_flow(trade_date: date | None = None) -> dict:
    """
    Runs daily at 16:30 IST (11:00 UTC) after market close.

    trade_date defaults to today (IST date).

    Downloads:
        - Full bhavcopy (all symbols) → Parquet
        - Per-symbol OHLCV for watched symbols → Parquet
        - Delivery data → PG delivery_data table

    Returns {bhavcopy_rows, symbols_downloaded, symbols_failed, delivery_rows}.
    """
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")

    effective_date = trade_date or datetime.now(tz=_IST).date()
    log.info("history_flow.start", trade_date=str(effective_date))

    pg = await _get_pg()

    async with PipelineTracker.run("history-flow", pg_store=pg) as tracker:
        # Run bhavcopy download and OHLCV downloads concurrently
        bhavcopy_result, ohlcv_result = await asyncio.gather(
            download_bhavcopy_task(effective_date),
            download_ohlcv_task(cfg.WATCHED_SYMBOLS, effective_date),
            return_exceptions=True,
        )

        bhavcopy_rows = 0
        if isinstance(bhavcopy_result, dict):
            bhavcopy_rows = bhavcopy_result.get("rows_downloaded", 0)
        elif isinstance(bhavcopy_result, Exception):
            tracker.add_error(f"bhavcopy exception: {bhavcopy_result}")

        ohlcv_downloaded = 0
        ohlcv_failed = 0
        if isinstance(ohlcv_result, dict):
            ohlcv_downloaded = ohlcv_result.get("symbols_downloaded", 0)
            ohlcv_failed = ohlcv_result.get("symbols_failed", 0)
        elif isinstance(ohlcv_result, Exception):
            tracker.add_error(f"ohlcv exception: {ohlcv_result}")

        delivery_rows = await download_delivery_task(effective_date)

        total = bhavcopy_rows + ohlcv_downloaded + delivery_rows
        tracker.add_records(total)

        summary = {
            "bhavcopy_rows": bhavcopy_rows,
            "symbols_downloaded": ohlcv_downloaded,
            "symbols_failed": ohlcv_failed,
            "delivery_rows": delivery_rows,
        }

    log.info("history_flow.done", **summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 5: BACKTEST EVAL FLOW (daily)
# ═══════════════════════════════════════════════════════════════════════════════

@task(name="evaluate-yesterday-signals")
async def evaluate_signals_task(eval_date: date) -> dict:
    """
    Evaluate yesterday's fused signals against actual price movement.

    For each signal generated on *eval_date - 1*:
        entry_price = yesterday's close (from Parquet OHLCV)
        exit_price  = *eval_date*'s close (or open if close unavailable)
        hit         = True if direction matched (BUY→price up, SELL→price down)
        return_pct  = (exit - entry) / entry * 100

    Writes a SignalPerformance row to PG signal_performance table.
    Returns {evaluated, hits, misses, avg_return_pct}.
    """
    logger = get_run_logger()
    pg = await _get_pg()
    parquet = _get_parquet()

    yesterday = eval_date - timedelta(days=1)
    evaluated = 0
    hits = 0
    misses = 0
    total_return = 0.0

    # Load yesterday's signals from PG
    try:
        # Fetch signals generated within yesterday's window (00:00–23:59 UTC)
        yesterday_start = datetime(yesterday.year, yesterday.month, yesterday.day,
                                   0, 0, 0, tzinfo=timezone.utc)
        yesterday_end = yesterday_start + timedelta(days=1)

        # Use get_latest_signals — filter by date in post-processing
        all_signals = await pg.get_latest_signals(limit=500)
        yesterday_signals = [
            s for s in all_signals
            if _signal_generated_in_window(s, yesterday_start, yesterday_end)
        ]
        logger.info(f"eval: found {len(yesterday_signals)} signals for {yesterday}")
    except Exception as exc:
        logger.error(f"evaluate_signals_task: failed to load signals: {exc}")
        log.error("evaluate_signals.load_failed", error=str(exc))
        return {"evaluated": 0, "hits": 0, "misses": 0, "avg_return_pct": 0.0}

    for signal_row in yesterday_signals:
        symbol: str = signal_row.get("symbol", "")
        signal_type: str = signal_row.get("signal_type", "NEUTRAL")
        generated_at_raw = signal_row.get("generated_at")
        signal_generated_at = _parse_dt(generated_at_raw) or datetime.now(tz=timezone.utc)

        if symbol in ("MACRO", "NIFTY50", "BANKNIFTY"):
            # Skip index/macro signals — no direct price comparison
            continue

        try:
            # Load yesterday's OHLCV to get entry_price (close)
            entry_df = parquet.read_ohlcv(symbol, yesterday, yesterday)
            if entry_df is None or (hasattr(entry_df, "empty") and entry_df.empty):
                logger.warning(f"eval: no OHLCV for {symbol} on {yesterday}")
                continue

            entry_close = _get_close(entry_df)
            if entry_close is None or entry_close <= 0:
                logger.warning(f"eval: zero/null close for {symbol} on {yesterday}")
                continue

            # Load eval_date's OHLCV to get exit_price (close, or open as fallback)
            exit_df = parquet.read_ohlcv(symbol, eval_date, eval_date)
            if exit_df is None or (hasattr(exit_df, "empty") and exit_df.empty):
                # Try the bhavcopy for eval_date's open price
                logger.warning(f"eval: no OHLCV for {symbol} on {eval_date}; skipping")
                continue

            exit_price = _get_close(exit_df)
            if exit_price is None or exit_price <= 0:
                exit_price = _get_open(exit_df)
            if exit_price is None or exit_price <= 0:
                logger.warning(f"eval: no usable exit price for {symbol} on {eval_date}")
                continue

            entry_price = float(entry_close)
            exit_price = float(exit_price)
            return_pct = (exit_price - entry_price) / entry_price * 100.0

            # A signal "hits" if the direction was correct
            price_up = exit_price > entry_price
            if signal_type in ("STRONG_BUY", "BUY"):
                hit = price_up
            elif signal_type in ("STRONG_SELL", "SELL"):
                hit = not price_up
            else:
                # NEUTRAL: hit if return is within ±0.5%
                hit = abs(return_pct) <= 0.5

            # Approximate holding hours: 6.25 market hours per session
            holding_hours = 6.25

            perf_id = make_id(f"{symbol}|{signal_generated_at.isoformat()}|{eval_date.isoformat()}")
            perf = SignalPerformance(
                id=perf_id,
                evaluation_date=datetime(eval_date.year, eval_date.month, eval_date.day,
                                         tzinfo=timezone.utc),
                symbol=symbol,
                signal_type=signal_type,
                signal_generated_at=signal_generated_at,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                return_pct=round(return_pct, 6),
                hit=hit,
                holding_hours=holding_hours,
            )

            await pg.insert_signal_performance(perf)

            evaluated += 1
            total_return += return_pct
            if hit:
                hits += 1
            else:
                misses += 1

            logger.debug(
                f"eval {symbol}: {signal_type} entry={entry_price:.2f} "
                f"exit={exit_price:.2f} return={return_pct:.2f}% hit={hit}"
            )

        except Exception as exc:
            logger.warning(f"evaluate_signals_task symbol={symbol} failed: {exc}")
            log.warning("evaluate_signals.symbol_failed", symbol=symbol, error=str(exc))

    avg_return_pct = round(total_return / evaluated, 6) if evaluated > 0 else 0.0
    result = {
        "evaluated": evaluated,
        "hits": hits,
        "misses": misses,
        "avg_return_pct": avg_return_pct,
    }
    logger.info(f"evaluate_signals: {result}")
    log.info("evaluate_signals_task.done", **result)
    return result


@flow(
    name="backtest-eval-flow",
    description="Daily: evaluate yesterday's signals against actual price moves",
    retries=1,
)
async def backtest_eval_flow(eval_date: date | None = None) -> dict:
    """
    Runs daily at 20:00 IST (14:30 UTC).

    eval_date defaults to today.  Evaluates signals generated yesterday.

    Returns {evaluated, hits, misses, avg_return_pct}.
    """
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")

    effective_date = eval_date or datetime.now(tz=_IST).date()
    log.info("backtest_eval_flow.start", eval_date=str(effective_date))

    pg = await _get_pg()

    async with PipelineTracker.run("backtest-eval-flow", pg_store=pg) as tracker:
        result = await evaluate_signals_task(effective_date)
        tracker.add_records(result.get("evaluated", 0))

    log.info("backtest_eval_flow.done", **result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER — entry point for manual / test runs
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline_once(symbols: list[str] | None = None) -> None:
    """
    Run ingest → process → fusion in sequence.

    Used for manual one-shot runs and integration testing.
    Does not require a running Prefect server.
    """
    log.info("run_pipeline_once.start")
    await ingest_flow()
    await process_flow()
    await fusion_flow(symbols=symbols or cfg.WATCHED_SYMBOLS)
    log.info("run_pipeline_once.done")


# ═══════════════════════════════════════════════════════════════════════════════
# SERVE / DEPLOYMENT helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_ingest_deployment():
    """Every 5 minutes, offset 0 s."""
    return ingest_flow.to_deployment(
        name="ingest-flow-scheduled",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5))],
        description="Fetch Pulse + nselib + RSS every 5 min",
    )


def _build_process_deployment():
    """Every 5 minutes, offset 2 min via cron."""
    return process_flow.to_deployment(
        name="process-flow-scheduled",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5),
                                    anchor_date=datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc))],
        description="NER + FinBERT + tagging every 5 min (offset 2 min)",
    )


def _build_fusion_deployment():
    """Every 5 minutes, offset 4 min via cron."""
    return fusion_flow.to_deployment(
        name="fusion-flow-scheduled",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5),
                                    anchor_date=datetime(2024, 1, 1, 0, 4, 0, tzinfo=timezone.utc))],
        description="Signal fusion every 5 min (offset 4 min)",
    )


def _build_history_deployment():
    """Daily at 11:00 UTC = 16:30 IST."""
    return history_flow.to_deployment(
        name="history-flow-scheduled",
        schedules=[CronSchedule(cron="0 11 * * 1-5", timezone="UTC")],
        description="Daily EOD bhavcopy + OHLCV download (Mon-Fri 16:30 IST)",
    )


def _build_backtest_deployment():
    """Daily at 14:30 UTC = 20:00 IST."""
    return backtest_eval_flow.to_deployment(
        name="backtest-eval-flow-scheduled",
        schedules=[CronSchedule(cron="30 14 * * 1-5", timezone="UTC")],
        description="Daily signal evaluation at 20:00 IST",
    )


async def _serve_all_flows_async() -> None:
    """
    Start all 5 flows using Prefect serve() — no Prefect server required.

    Serve mode runs flows locally with their schedules.  All flows share
    this single asyncio event loop.
    """
    from prefect.runner.runner import Runner

    ingest_d = ingest_flow.to_deployment(
        name="ingest-flow-serve",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5))],
    )
    process_d = process_flow.to_deployment(
        name="process-flow-serve",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5),
                                    anchor_date=datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc))],
    )
    fusion_d = fusion_flow.to_deployment(
        name="fusion-flow-serve",
        schedules=[IntervalSchedule(interval=timedelta(minutes=5),
                                    anchor_date=datetime(2024, 1, 1, 0, 4, 0, tzinfo=timezone.utc))],
    )
    history_d = history_flow.to_deployment(
        name="history-flow-serve",
        schedules=[CronSchedule(cron="0 11 * * 1-5", timezone="UTC")],
    )
    backtest_d = backtest_eval_flow.to_deployment(
        name="backtest-eval-flow-serve",
        schedules=[CronSchedule(cron="30 14 * * 1-5", timezone="UTC")],
    )

    runner = Runner(name="news-pipeline-runner")
    await runner.add(ingest_d)
    await runner.add(process_d)
    await runner.add(fusion_d)
    await runner.add(history_d)
    await runner.add(backtest_d)

    log.info("serve_all_flows.starting", mode="serve-no-server")
    await runner.start()


def serve_all_flows() -> None:
    """
    Start all 5 flows with their schedules.

    Mode selection:
        - PREFECT_API_URL is set  → deploy to Prefect server via apply()
        - PREFECT_API_URL not set → serve locally via serve() (no server)
    """
    prefect_api_url = os.environ.get("PREFECT_API_URL", "").strip()

    if prefect_api_url:
        log.info("serve_all_flows.server_mode", api_url=prefect_api_url)
        # Server mode: apply deployments to the running Prefect API
        deployments = [
            _build_ingest_deployment(),
            _build_process_deployment(),
            _build_fusion_deployment(),
            _build_history_deployment(),
            _build_backtest_deployment(),
        ]
        for dep in deployments:
            asyncio.run(dep.apply())
            log.info("serve_all_flows.deployment_applied", name=dep.name)
    else:
        log.info("serve_all_flows.serve_mode", note="no PREFECT_API_URL; running locally")
        asyncio.run(_serve_all_flows_async())


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_dt(value: str | datetime | None) -> datetime | None:
    """Parse an ISO-8601 string or pass through a datetime.  Returns None if unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        dt = datetime.fromisoformat(str(value))
        # Ensure UTC-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _signal_generated_in_window(
    signal_row: dict,
    start: datetime,
    end: datetime,
) -> bool:
    """Return True if signal's generated_at falls within [start, end)."""
    raw = signal_row.get("generated_at")
    dt = _parse_dt(raw)
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return start <= dt < end


def _get_close(df: Any) -> float | None:
    """Extract the last close price from an OHLCV DataFrame.  None if not found."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    cols = [c.lower() for c in df.columns]
    try:
        if "close" in cols:
            col_idx = cols.index("close")
            orig_col = df.columns[col_idx]
            val = df[orig_col].iloc[-1]
            return float(val)
    except (IndexError, ValueError, TypeError):
        pass
    return None


def _get_open(df: Any) -> float | None:
    """Extract the last open price from an OHLCV DataFrame.  None if not found."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    cols = [c.lower() for c in df.columns]
    try:
        if "open" in cols:
            col_idx = cols.index("open")
            orig_col = df.columns[col_idx]
            val = df[orig_col].iloc[-1]
            return float(val)
    except (IndexError, ValueError, TypeError):
        pass
    return None
