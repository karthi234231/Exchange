"""algo_trading.news_pipeline.storage.pg_store — Async PostgreSQL storage via asyncpg.

Manages six tables:
  news_events, bulk_deals, delivery_data, fused_signals,
  pipeline_runs, signal_performance

Usage::

    store = PGStore()
    await store.connect()
    await store.upsert_news_event(event)
    await store.close()
"""
from __future__ import annotations

import json
import datetime
from typing import Any

import asyncpg
import structlog

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.models import (
    BulkDealRecord,
    DeliveryRecord,
    FusedSignal,
    PipelineRun,
    ProcessedNewsEvent,
    SignalPerformance,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS news_events (
        id                    TEXT PRIMARY KEY,
        source                TEXT NOT NULL,
        headline              TEXT NOT NULL,
        url                   TEXT,
        published_at          TIMESTAMPTZ NOT NULL,
        fetched_at            TIMESTAMPTZ NOT NULL,
        symbols               TEXT[],
        event_type            TEXT,
        sentiment             TEXT,
        sentiment_score       DOUBLE PRECISION,
        sentiment_confidence  DOUBLE PRECISION,
        raw_body              TEXT,
        entities              JSONB,
        tags                  TEXT[],
        created_at            TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bulk_deals (
        id               TEXT PRIMARY KEY,
        deal_date        DATE NOT NULL,
        symbol           TEXT NOT NULL,
        client_name      TEXT,
        deal_type        TEXT NOT NULL,
        direction        TEXT NOT NULL,
        quantity         BIGINT,
        price            DOUBLE PRECISION,
        deal_value_cr    DOUBLE PRECISION,
        fetched_at       TIMESTAMPTZ NOT NULL,
        created_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS delivery_data (
        id            TEXT PRIMARY KEY,
        trade_date    DATE NOT NULL,
        symbol        TEXT NOT NULL,
        delivery_qty  BIGINT,
        traded_qty    BIGINT,
        delivery_pct  DOUBLE PRECISION,
        fetched_at    TIMESTAMPTZ NOT NULL,
        UNIQUE(trade_date, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fused_signals (
        id                   TEXT PRIMARY KEY,
        generated_at         TIMESTAMPTZ NOT NULL,
        symbol               TEXT NOT NULL,
        signal_type          TEXT NOT NULL,
        alpha_score          DOUBLE PRECISION,
        news_score           DOUBLE PRECISION,
        bulk_deal_score      DOUBLE PRECISION,
        delivery_score       DOUBLE PRECISION,
        vix_score            DOUBLE PRECISION,
        contributing_events  JSONB,
        ttl_minutes          INT,
        created_at           TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id                 TEXT PRIMARY KEY,
        run_id             TEXT,
        flow_name          TEXT NOT NULL,
        started_at         TIMESTAMPTZ NOT NULL,
        completed_at       TIMESTAMPTZ,
        status             TEXT NOT NULL,
        records_processed  INT DEFAULT 0,
        errors             JSONB,
        latency_ms         DOUBLE PRECISION,
        created_at         TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signal_performance (
        id                   TEXT PRIMARY KEY,
        evaluation_date      DATE NOT NULL,
        symbol               TEXT NOT NULL,
        signal_type          TEXT NOT NULL,
        signal_generated_at  TIMESTAMPTZ NOT NULL,
        entry_price          DOUBLE PRECISION,
        exit_price           DOUBLE PRECISION,
        return_pct           DOUBLE PRECISION,
        hit                  BOOLEAN,
        holding_hours        DOUBLE PRECISION,
        created_at           TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_news_events_symbols ON news_events USING GIN(symbols)",
    "CREATE INDEX IF NOT EXISTS idx_news_events_published ON news_events(published_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_bulk_deals_symbol ON bulk_deals(symbol, deal_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fused_signals_symbol ON fused_signals(symbol, generated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fused_signals_generated ON fused_signals(generated_at DESC)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | datetime.datetime | None) -> datetime.datetime | None:
    """Ensure a datetime value is a proper datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def _parse_date(value: str | datetime.datetime | datetime.date | None) -> datetime.date | None:
    """Coerce a value to datetime.date."""
    if value is None:
        return None
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.datetime):
        return value.date()
    return datetime.date.fromisoformat(value[:10])


def _row_to_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg Record to a plain dict with JSON-serialisable values."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (datetime.datetime, datetime.date)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PGStore:
    """Async PostgreSQL storage backed by an asyncpg connection pool."""

    def __init__(self, dsn: str = cfg.POSTGRES_DSN) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the asyncpg connection pool and run DDL migrations."""
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            await self.create_tables()
            log.info("pg_store.connected", dsn=self._dsn)
        except Exception as exc:
            log.error("pg_store.connect_failed", error=str(exc))
            raise

    async def close(self) -> None:
        """Gracefully close all pooled connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("pg_store.closed")

    async def create_tables(self) -> None:
        """Execute all CREATE TABLE / CREATE INDEX DDL statements."""
        if self._pool is None:
            raise RuntimeError("PGStore.connect() must be called before create_tables()")
        async with self._pool.acquire() as conn:
            for stmt in _DDL_STATEMENTS:
                await conn.execute(stmt)
        log.info("pg_store.tables_ready")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PGStore is not connected — call await store.connect() first.")
        return self._pool

    # ------------------------------------------------------------------
    # News Events
    # ------------------------------------------------------------------

    async def upsert_news_event(self, event: ProcessedNewsEvent) -> None:
        """Insert a processed news event; silently ignore duplicates (ON CONFLICT DO NOTHING)."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO news_events (
                        id, source, headline, url, published_at, fetched_at,
                        symbols, event_type, sentiment, sentiment_score,
                        sentiment_confidence, raw_body, entities, tags
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7, $8, $9, $10,
                        $11, $12, $13, $14
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    event.id,
                    event.source,
                    event.headline,
                    event.url,
                    _parse_dt(event.published_at),
                    _parse_dt(event.fetched_at),
                    list(event.symbols),
                    event.event_type,
                    event.sentiment,
                    float(event.sentiment_score),
                    float(event.sentiment_confidence),
                    event.raw_body,
                    json.dumps(event.entities),
                    list(event.tags),
                )
        except Exception as exc:
            log.error("pg_store.upsert_news_event.failed", news_id=event.id, error=str(exc))
            raise

    async def get_recent_news(
        self,
        symbols: list[str] | None = None,
        hours: int = 4,
        limit: int = 100,
        sentiment: str | None = None,
    ) -> list[dict]:
        """Return recent news events as a list of dicts.

        Filters by time window and optionally by symbol array overlap and sentiment.
        """
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=hours)
                params: list[Any] = [cutoff, limit]
                clauses = ["published_at >= $1"]

                if symbols:
                    params.insert(1, symbols)
                    # Renumber placeholders
                    clauses.append("symbols && $2")
                    # cutoff stays $1, symbols $2, limit shifts
                    limit_ph = f"${len(params)}"
                else:
                    limit_ph = "$2"

                if sentiment:
                    params.insert(-1, sentiment)
                    clauses.append(f"sentiment = ${len(params) - 1}")
                    limit_ph = f"${len(params)}"

                where = " AND ".join(clauses)
                query = f"""
                    SELECT * FROM news_events
                    WHERE {where}
                    ORDER BY published_at DESC
                    LIMIT {limit_ph}
                """
                rows = await conn.fetch(query, *params)
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_recent_news.failed", error=str(exc))
            return []

    async def news_id_exists(self, news_id: str) -> bool:
        """Fast deduplication check — returns True if the id is already stored."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM news_events WHERE id = $1 LIMIT 1", news_id
                )
                return row is not None
        except Exception as exc:
            log.error("pg_store.news_id_exists.failed", news_id=news_id, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Bulk Deals
    # ------------------------------------------------------------------

    async def upsert_bulk_deal(self, deal: BulkDealRecord) -> None:
        """Insert a bulk/block deal; silently ignore duplicates."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bulk_deals (
                        id, deal_date, symbol, client_name,
                        deal_type, direction, quantity, price,
                        deal_value_cr, fetched_at
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    deal.id,
                    _parse_date(deal.deal_date),
                    deal.symbol,
                    deal.client_name,
                    deal.deal_type,
                    deal.direction,
                    int(deal.quantity),
                    float(deal.price),
                    float(deal.deal_value_cr),
                    _parse_dt(deal.fetched_at),
                )
        except Exception as exc:
            log.error("pg_store.upsert_bulk_deal.failed", deal_id=deal.id, error=str(exc))
            raise

    async def get_today_bulk_deals(self, symbol: str | None = None) -> list[dict]:
        """Return today's bulk and block deals, optionally filtered by symbol."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                today = datetime.date.today()
                if symbol:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM bulk_deals
                        WHERE deal_date = $1 AND symbol = $2
                        ORDER BY deal_value_cr DESC
                        """,
                        today,
                        symbol,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM bulk_deals
                        WHERE deal_date = $1
                        ORDER BY deal_value_cr DESC
                        """,
                        today,
                    )
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_today_bulk_deals.failed", symbol=symbol, error=str(exc))
            return []

    async def get_bulk_deals_range(self, symbol: str, days: int = 30) -> list[dict]:
        """Return bulk/block deals for a symbol over the past *days* calendar days."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                cutoff = datetime.date.today() - datetime.timedelta(days=days)
                rows = await conn.fetch(
                    """
                    SELECT * FROM bulk_deals
                    WHERE symbol = $1 AND deal_date >= $2
                    ORDER BY deal_date DESC, deal_value_cr DESC
                    """,
                    symbol,
                    cutoff,
                )
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_bulk_deals_range.failed", symbol=symbol, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Delivery Data
    # ------------------------------------------------------------------

    async def upsert_delivery(self, record: DeliveryRecord) -> None:
        """Insert or update delivery record. Updates all columns on conflict."""
        pool = self._pool_or_raise()
        try:
            # Build a stable id from (trade_date, symbol)
            trade_date = _parse_date(record.trade_date)
            rec_id = f"{record.symbol}|{trade_date}"

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO delivery_data (
                        id, trade_date, symbol, delivery_qty,
                        traded_qty, delivery_pct, fetched_at
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7
                    )
                    ON CONFLICT (trade_date, symbol)
                    DO UPDATE SET
                        delivery_qty = EXCLUDED.delivery_qty,
                        traded_qty   = EXCLUDED.traded_qty,
                        delivery_pct = EXCLUDED.delivery_pct,
                        fetched_at   = EXCLUDED.fetched_at
                    """,
                    rec_id,
                    trade_date,
                    record.symbol,
                    int(record.delivery_qty),
                    int(record.traded_qty),
                    float(record.delivery_pct),
                    _parse_dt(record.fetched_at),
                )
        except Exception as exc:
            log.error(
                "pg_store.upsert_delivery.failed",
                symbol=record.symbol,
                error=str(exc),
            )
            raise

    async def get_delivery_history(self, symbol: str, days: int = 30) -> list[dict]:
        """Return delivery history for a symbol over the past *days* days."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                cutoff = datetime.date.today() - datetime.timedelta(days=days)
                rows = await conn.fetch(
                    """
                    SELECT * FROM delivery_data
                    WHERE symbol = $1 AND trade_date >= $2
                    ORDER BY trade_date DESC
                    """,
                    symbol,
                    cutoff,
                )
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_delivery_history.failed", symbol=symbol, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Fused Signals
    # ------------------------------------------------------------------

    async def upsert_signal(self, signal: FusedSignal) -> None:
        """Insert or replace a fused signal record."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO fused_signals (
                        id, generated_at, symbol, signal_type,
                        alpha_score, news_score, bulk_deal_score,
                        delivery_score, vix_score,
                        contributing_events, ttl_minutes
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7,
                        $8, $9,
                        $10, $11
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        generated_at        = EXCLUDED.generated_at,
                        signal_type         = EXCLUDED.signal_type,
                        alpha_score         = EXCLUDED.alpha_score,
                        news_score          = EXCLUDED.news_score,
                        bulk_deal_score     = EXCLUDED.bulk_deal_score,
                        delivery_score      = EXCLUDED.delivery_score,
                        vix_score           = EXCLUDED.vix_score,
                        contributing_events = EXCLUDED.contributing_events,
                        ttl_minutes         = EXCLUDED.ttl_minutes
                    """,
                    signal.id,
                    _parse_dt(signal.generated_at),
                    signal.symbol,
                    signal.signal_type,
                    float(signal.alpha_score),
                    float(signal.news_score),
                    float(signal.bulk_deal_score),
                    float(signal.delivery_score),
                    float(signal.vix_score),
                    json.dumps(list(signal.contributing_events)),
                    int(signal.ttl_minutes),
                )
        except Exception as exc:
            log.error("pg_store.upsert_signal.failed", signal_id=signal.id, error=str(exc))
            raise

    async def get_latest_signals(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return the most recent fused signals, optionally filtered to specific symbols."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                if symbols:
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT ON (symbol) *
                        FROM fused_signals
                        WHERE symbol = ANY($1)
                        ORDER BY symbol, generated_at DESC
                        LIMIT $2
                        """,
                        symbols,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT ON (symbol) *
                        FROM fused_signals
                        ORDER BY symbol, generated_at DESC
                        LIMIT $1
                        """,
                        limit,
                    )
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_latest_signals.failed", error=str(exc))
            return []

    async def get_signal(self, symbol: str) -> dict | None:
        """Return the latest non-expired signal for a symbol, or None."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
                row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM fused_signals
                    WHERE symbol = $1
                      AND generated_at + (ttl_minutes * INTERVAL '1 minute') > $2
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """,
                    symbol,
                    now_utc,
                )
                return _row_to_dict(row) if row else None
        except Exception as exc:
            log.error("pg_store.get_signal.failed", symbol=symbol, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Pipeline Runs
    # ------------------------------------------------------------------

    async def insert_pipeline_run(self, run: PipelineRun) -> None:
        """Insert a new pipeline run record."""
        pool = self._pool_or_raise()
        try:
            # Use run_id as the primary key id since PipelineRun has no separate 'id' field
            rec_id = run.run_id
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        id, run_id, flow_name, started_at, completed_at,
                        status, records_processed, errors, latency_ms
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8, $9
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    rec_id,
                    run.run_id,
                    run.flow_name,
                    _parse_dt(run.started_at),
                    _parse_dt(run.completed_at),
                    run.status,
                    int(run.records_processed),
                    json.dumps(list(run.errors)),
                    float(run.latency_ms),
                )
        except Exception as exc:
            log.error("pg_store.insert_pipeline_run.failed", run_id=run.run_id, error=str(exc))
            raise

    async def update_pipeline_run(self, run: PipelineRun) -> None:
        """Update an existing pipeline run with completion details."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE pipeline_runs SET
                        completed_at      = $1,
                        status            = $2,
                        records_processed = $3,
                        errors            = $4,
                        latency_ms        = $5
                    WHERE id = $6
                    """,
                    _parse_dt(run.completed_at),
                    run.status,
                    int(run.records_processed),
                    json.dumps(list(run.errors)),
                    float(run.latency_ms),
                    run.run_id,
                )
        except Exception as exc:
            log.error("pg_store.update_pipeline_run.failed", run_id=run.run_id, error=str(exc))
            raise

    async def get_recent_runs(
        self, flow_name: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Return recent pipeline runs, optionally filtered by flow name."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                if flow_name:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM pipeline_runs
                        WHERE flow_name = $1
                        ORDER BY started_at DESC
                        LIMIT $2
                        """,
                        flow_name,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM pipeline_runs
                        ORDER BY started_at DESC
                        LIMIT $1
                        """,
                        limit,
                    )
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:
            log.error("pg_store.get_recent_runs.failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Signal Performance
    # ------------------------------------------------------------------

    async def insert_signal_performance(self, perf: SignalPerformance) -> None:
        """Insert a signal performance record (backtest result)."""
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO signal_performance (
                        id, evaluation_date, symbol, signal_type,
                        signal_generated_at, entry_price, exit_price,
                        return_pct, hit, holding_hours
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7,
                        $8, $9, $10
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    perf.id,
                    _parse_date(perf.evaluation_date),
                    perf.symbol,
                    perf.signal_type,
                    _parse_dt(perf.signal_generated_at),
                    float(perf.entry_price),
                    float(perf.exit_price),
                    float(perf.return_pct),
                    bool(perf.hit),
                    float(perf.holding_hours),
                )
        except Exception as exc:
            log.error("pg_store.insert_signal_performance.failed", perf_id=perf.id, error=str(exc))
            raise

    async def get_signal_summary(self, days: int = 30) -> dict:
        """Return aggregated performance stats over the past *days* days.

        Returns a dict with keys:
            total_signals, win_rate, avg_return, by_symbol, by_type
        """
        pool = self._pool_or_raise()
        try:
            async with pool.acquire() as conn:
                cutoff = datetime.date.today() - datetime.timedelta(days=days)

                # Overall stats
                overall_row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*)                          AS total_signals,
                        AVG(CASE WHEN hit THEN 1.0 ELSE 0.0 END) AS win_rate,
                        AVG(return_pct)                   AS avg_return
                    FROM signal_performance
                    WHERE evaluation_date >= $1
                    """,
                    cutoff,
                )

                # By symbol
                symbol_rows = await conn.fetch(
                    """
                    SELECT
                        symbol,
                        COUNT(*)                          AS total,
                        AVG(CASE WHEN hit THEN 1.0 ELSE 0.0 END) AS win_rate,
                        AVG(return_pct)                   AS avg_return
                    FROM signal_performance
                    WHERE evaluation_date >= $1
                    GROUP BY symbol
                    ORDER BY total DESC
                    """,
                    cutoff,
                )

                # By signal type
                type_rows = await conn.fetch(
                    """
                    SELECT
                        signal_type,
                        COUNT(*)                          AS total,
                        AVG(CASE WHEN hit THEN 1.0 ELSE 0.0 END) AS win_rate,
                        AVG(return_pct)                   AS avg_return
                    FROM signal_performance
                    WHERE evaluation_date >= $1
                    GROUP BY signal_type
                    ORDER BY total DESC
                    """,
                    cutoff,
                )

                by_symbol = {
                    r["symbol"]: {
                        "total": r["total"],
                        "win_rate": round(float(r["win_rate"] or 0.0), 4),
                        "avg_return": round(float(r["avg_return"] or 0.0), 6),
                    }
                    for r in symbol_rows
                }

                by_type = {
                    r["signal_type"]: {
                        "total": r["total"],
                        "win_rate": round(float(r["win_rate"] or 0.0), 4),
                        "avg_return": round(float(r["avg_return"] or 0.0), 6),
                    }
                    for r in type_rows
                }

                return {
                    "total_signals": int(overall_row["total_signals"] or 0),
                    "win_rate": round(float(overall_row["win_rate"] or 0.0), 4),
                    "avg_return": round(float(overall_row["avg_return"] or 0.0), 6),
                    "by_symbol": by_symbol,
                    "by_type": by_type,
                }
        except Exception as exc:
            log.error("pg_store.get_signal_summary.failed", error=str(exc))
            return {
                "total_signals": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "by_symbol": {},
                "by_type": {},
            }
