"""News endpoints — /news."""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from algo_trading.news_pipeline.api.schemas import NewsEventResponse, NewsListResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_news_response(row: dict) -> NewsEventResponse:
    """Convert a PGStore row dict to a NewsEventResponse.

    ``symbols`` and ``tags`` may be PostgreSQL arrays (already lists) or JSON
    strings (from asyncpg JSONB fallback) — handle both.
    """
    import json as _json

    def _to_list(val) -> list[str]:
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            try:
                parsed = _json.loads(val)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except (ValueError, TypeError):
                pass
        return []

    published_at = row.get("published_at")
    if isinstance(published_at, str):
        published_at = datetime.fromisoformat(published_at)
    if published_at is not None and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return NewsEventResponse(
        id=row["id"],
        source=row.get("source", ""),
        headline=row.get("headline", ""),
        url=row.get("url"),
        published_at=published_at or datetime.now(tz=timezone.utc),
        symbols=_to_list(row.get("symbols", [])),
        event_type=row.get("event_type", "OTHER"),
        sentiment=row.get("sentiment", "neutral"),
        sentiment_score=float(row.get("sentiment_score", 0.0)),
        tags=_to_list(row.get("tags", [])),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=NewsListResponse)
async def get_news(
    request: Request,
    symbol: str | None = Query(None, description="Filter by NSE symbol"),
    hours: int = Query(4, ge=1, le=48, description="Look-back window in hours"),
    sentiment: str | None = Query(
        None,
        pattern="^(bullish|bearish|neutral)$",
        description="Filter by sentiment label",
    ),
    limit: int = Query(100, ge=1, le=500),
) -> NewsListResponse:
    """Get recent processed news events with optional filters.

    Query parameters
    ----------------
    - **symbol**: restrict to events that mention this NSE ticker.
    - **hours**: look-back window (default 4 h, max 48 h).
    - **sentiment**: ``bullish`` | ``bearish`` | ``neutral``.
    - **limit**: maximum number of events returned (default 100, max 500).
    """
    try:
        pg = request.app.state.pg
        symbols = [symbol.upper()] if symbol else None

        rows = await pg.get_recent_news(
            symbols=symbols,
            hours=hours,
            limit=limit,
            sentiment=sentiment,
        )

        events = [_dict_to_news_response(r) for r in rows]

        filters: dict = {"hours": hours}
        if symbol:
            filters["symbol"] = symbol.upper()
        if sentiment:
            filters["sentiment"] = sentiment

        return NewsListResponse(
            events=events,
            count=len(events),
            filters_applied=filters,
        )
    except Exception as exc:
        log.error("api.get_news.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to retrieve news events") from exc


@router.get("/{symbol}", response_model=NewsListResponse)
async def get_news_for_symbol(
    symbol: str,
    request: Request,
    hours: int = Query(4, ge=1, le=48, description="Look-back window in hours"),
    limit: int = Query(50, ge=1, le=200),
) -> NewsListResponse:
    """Get recent news events for a specific NSE symbol.

    Path parameter
    --------------
    - **symbol**: NSE ticker (case-insensitive; normalised to uppercase).
    """
    try:
        pg = request.app.state.pg
        sym = symbol.upper()

        rows = await pg.get_recent_news(
            symbols=[sym],
            hours=hours,
            limit=limit,
            sentiment=None,
        )

        events = [_dict_to_news_response(r) for r in rows]

        return NewsListResponse(
            events=events,
            count=len(events),
            filters_applied={"symbol": sym, "hours": hours},
        )
    except Exception as exc:
        log.error("api.get_news_for_symbol.failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to retrieve news events") from exc
