"""algo_trading.news_pipeline.models — Shared data models for the news intelligence pipeline.

All models are plain stdlib dataclasses.  No external dependencies.
Every model exposes a ``to_dict()`` method suitable for JSON serialisation
(datetime objects are converted to ISO-8601 strings).

Helper
------
``make_id(data: str) -> str``
    Returns the SHA-256 hex digest of *data*.  Use it to build deterministic,
    collision-resistant IDs from canonical URLs, composite keys, etc.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def make_id(data: str) -> str:
    """Return the SHA-256 hex digest of *data* (UTF-8 encoded).

    Example::

        article_id = make_id(canonical_url)
        deal_id    = make_id(f"{symbol}|{date}|{client}|{quantity}")
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _dt_to_iso(dt: datetime | None) -> str | None:
    """Convert a datetime to an ISO-8601 string, or None if dt is None."""
    if dt is None:
        return None
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Raw ingestion layer
# ---------------------------------------------------------------------------

@dataclass
class RawNewsItem:
    """Raw news item as returned by any ingestion source before NLP processing.

    The ``id`` field is the SHA-256 hex digest of the canonical URL so
    deduplication is O(1) with a simple set/bloom filter.
    """

    id: str                   # make_id(canonical_url)
    source: str               # "pulse" | "rss_et" | "rss_mc" | "nse_announcements" | …
    headline: str
    url: str
    published_at: datetime    # timezone-aware preferred; naive treated as UTC
    fetched_at: datetime      # when the item was retrieved by the scraper
    body: str                 # first ~500 chars of article text (or full if short)
    symbol_hints: list[str]   # uppercase tokens that look like NSE ticker symbols

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "headline": self.headline,
            "url": self.url,
            "published_at": _dt_to_iso(self.published_at),
            "fetched_at": _dt_to_iso(self.fetched_at),
            "body": self.body,
            "symbol_hints": list(self.symbol_hints),
        }


# ---------------------------------------------------------------------------
# Processed / enriched layer
# ---------------------------------------------------------------------------

@dataclass
class ProcessedNewsEvent:
    """News item after NER extraction, FinBERT scoring, and event-type tagging.

    ``sentiment_score`` is in [-1.0, 1.0] where -1 is maximally bearish and
    +1 is maximally bullish.  ``sentiment_confidence`` is in [0.0, 1.0].
    """

    id: str                        # same sha256 as the originating RawNewsItem
    source: str
    headline: str
    url: str
    published_at: datetime
    fetched_at: datetime
    symbols: list[str]             # resolved, validated NSE symbols
    event_type: str                # EARNINGS | DIVIDEND | MERGER | RESULTS | REGULATORY | OTHER
    sentiment: str                 # "bullish" | "bearish" | "neutral"
    sentiment_score: float         # [-1.0, 1.0]
    sentiment_confidence: float    # [0.0, 1.0]
    raw_body: str                  # original body text (unchanged from RawNewsItem)
    entities: dict                 # spaCy NER output: {"ORG": [...], "PERSON": [...], …}
    tags: list[str]                # free-form classifier tags, e.g. ["q4-results", "beat"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "headline": self.headline,
            "url": self.url,
            "published_at": _dt_to_iso(self.published_at),
            "fetched_at": _dt_to_iso(self.fetched_at),
            "symbols": list(self.symbols),
            "event_type": self.event_type,
            "sentiment": self.sentiment,
            "sentiment_score": round(self.sentiment_score, 6),
            "sentiment_confidence": round(self.sentiment_confidence, 6),
            "raw_body": self.raw_body,
            "entities": self.entities,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Market microstructure data
# ---------------------------------------------------------------------------

@dataclass
class BulkDealRecord:
    """Single bulk or block deal as reported by NSE via nselib.

    ``deal_value_cr`` = ``quantity * price / 1e7`` (crore INR).
    ``id`` is the SHA-256 of ``f"{symbol}|{deal_date}|{client_name}|{quantity}"``.
    """

    id: str                  # make_id composite key
    deal_date: datetime      # date portion only (time is midnight UTC)
    symbol: str              # NSE ticker, e.g. "RELIANCE"
    client_name: str         # institutional or retail client name
    deal_type: str           # "bulk" | "block"
    direction: str           # "BUY" | "SELL"
    quantity: int            # number of shares
    price: float             # execution price (INR)
    deal_value_cr: float     # notional value in crore INR
    fetched_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "deal_date": _dt_to_iso(self.deal_date),
            "symbol": self.symbol,
            "client_name": self.client_name,
            "deal_type": self.deal_type,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": round(self.price, 4),
            "deal_value_cr": round(self.deal_value_cr, 4),
            "fetched_at": _dt_to_iso(self.fetched_at),
        }


@dataclass
class DeliveryRecord:
    """Delivery volume data for a symbol on a single trading day.

    ``delivery_pct`` = ``delivery_qty / traded_qty * 100``.
    A high delivery percentage (relative to its own z-score history) signals
    genuine conviction rather than intraday speculation.
    """

    trade_date: datetime    # trading day (time midnight UTC)
    symbol: str
    delivery_qty: int       # shares held overnight (delivery)
    traded_qty: int         # total traded volume for the day
    delivery_pct: float     # delivery_qty / traded_qty * 100
    fetched_at: datetime

    def to_dict(self) -> dict:
        return {
            "trade_date": _dt_to_iso(self.trade_date),
            "symbol": self.symbol,
            "delivery_qty": self.delivery_qty,
            "traded_qty": self.traded_qty,
            "delivery_pct": round(self.delivery_pct, 4),
            "fetched_at": _dt_to_iso(self.fetched_at),
        }


# ---------------------------------------------------------------------------
# Signal fusion output
# ---------------------------------------------------------------------------

@dataclass
class FusedSignal:
    """Alpha signal produced by the signal fusion layer.

    ``alpha_score`` is a weighted combination of news, bulk-deal, delivery,
    and VIX sub-scores.  It sits in [-1.0, 1.0].  The ``signal_type`` label
    is derived by comparing ``alpha_score`` against Config thresholds.

    TTL
    ---
    Signals are ephemeral.  ``is_expired()`` returns True when
    ``generated_at + timedelta(minutes=ttl_minutes) < utcnow()``.
    """

    id: str                         # str(uuid.uuid4())
    generated_at: datetime
    symbol: str
    signal_type: str                # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
    alpha_score: float              # [-1.0, 1.0]
    news_score: float               # news sub-score before weighting
    bulk_deal_score: float          # bulk/block deal sub-score before weighting
    delivery_score: float           # delivery pct z-score sub-score before weighting
    vix_score: float                # VIX adjustment applied to alpha_score
    contributing_events: list[str]  # ProcessedNewsEvent IDs that fed this signal
    ttl_minutes: int                # how long (in minutes) the signal stays valid

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generated_at": _dt_to_iso(self.generated_at),
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "alpha_score": round(self.alpha_score, 6),
            "news_score": round(self.news_score, 6),
            "bulk_deal_score": round(self.bulk_deal_score, 6),
            "delivery_score": round(self.delivery_score, 6),
            "vix_score": round(self.vix_score, 6),
            "contributing_events": list(self.contributing_events),
            "ttl_minutes": self.ttl_minutes,
        }

    def is_expired(self) -> bool:
        """Return True if the signal's TTL has elapsed.

        Uses UTC now so timezone-naive ``generated_at`` values are treated
        as UTC to avoid silent comparison errors.
        """
        now_utc = datetime.now(tz=timezone.utc)
        # Normalise generated_at: if naive, assume UTC
        gen = self.generated_at
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        expiry = gen + timedelta(minutes=self.ttl_minutes)
        return now_utc >= expiry


# ---------------------------------------------------------------------------
# Pipeline observability
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    """Metadata record for a single Prefect (or equivalent) flow run.

    Written at the end of each orchestrator cycle so we have a persistent
    audit trail of pipeline health metrics.
    """

    run_id: str
    flow_name: str
    started_at: datetime
    completed_at: datetime | None   # None while the run is still in progress
    status: str                     # "running" | "completed" | "failed"
    records_processed: int
    errors: list[str]               # human-readable error messages
    latency_ms: float               # wall-clock duration of the run in milliseconds

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "flow_name": self.flow_name,
            "started_at": _dt_to_iso(self.started_at),
            "completed_at": _dt_to_iso(self.completed_at),
            "status": self.status,
            "records_processed": self.records_processed,
            "errors": list(self.errors),
            "latency_ms": round(self.latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# Backtest / signal evaluation
# ---------------------------------------------------------------------------

@dataclass
class SignalPerformance:
    """Backtest record that tracks whether a FusedSignal predicted correctly.

    After ``holding_hours`` elapse, the system calculates ``return_pct`` and
    sets ``hit = True`` if the direction matched the signal_type.
    """

    id: str                        # make_id(f"{symbol}|{signal_generated_at}|{evaluation_date}")
    evaluation_date: datetime      # date the evaluation was run
    symbol: str
    signal_type: str               # original signal type (STRONG_BUY, BUY, …)
    signal_generated_at: datetime  # timestamp of the original FusedSignal
    entry_price: float             # price at signal generation time (INR)
    exit_price: float              # price at evaluation time (INR)
    return_pct: float              # (exit_price - entry_price) / entry_price * 100
    hit: bool                      # True if direction was correct
    holding_hours: float           # how many market hours between entry and exit

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "evaluation_date": _dt_to_iso(self.evaluation_date),
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "signal_generated_at": _dt_to_iso(self.signal_generated_at),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "return_pct": round(self.return_pct, 6),
            "hit": self.hit,
            "holding_hours": round(self.holding_hours, 4),
        }


# ---------------------------------------------------------------------------
# Convenience factory helpers (keep models self-contained)
# ---------------------------------------------------------------------------

def new_signal_id() -> str:
    """Return a fresh UUID4 string for use as FusedSignal.id."""
    return str(uuid.uuid4())
