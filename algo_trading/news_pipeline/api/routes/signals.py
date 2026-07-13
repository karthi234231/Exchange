"""Signal endpoints — /signals."""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from algo_trading.news_pipeline.api.schemas import SignalResponse, SignalsListResponse
from algo_trading.news_pipeline.storage.cache import signal_cache

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/signals", tags=["signals"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_signal_response(row: dict) -> SignalResponse:
    """Convert a PGStore row dict (or cache dict) to a SignalResponse.

    Both sources return the same field names.  ``contributing_events`` may be
    stored as a JSON string (from asyncpg JSONB) or already a list — handle both.
    """
    import json as _json

    # contributing_events: asyncpg returns JSONB as a list; cache stores a list
    ce = row.get("contributing_events", [])
    if isinstance(ce, str):
        try:
            ce = _json.loads(ce)
        except (ValueError, TypeError):
            ce = []

    # generated_at: may arrive as str or datetime
    gen_at = row.get("generated_at")
    if isinstance(gen_at, str):
        gen_at = datetime.fromisoformat(gen_at)
    # Ensure timezone-aware for is_expired() parity
    if gen_at is not None and gen_at.tzinfo is None:
        gen_at = gen_at.replace(tzinfo=timezone.utc)

    ttl_minutes: int = int(row.get("ttl_minutes", 60))
    expired = False
    if gen_at is not None:
        from datetime import timedelta
        expiry = gen_at + timedelta(minutes=ttl_minutes)
        expired = datetime.now(tz=timezone.utc) >= expiry

    return SignalResponse(
        symbol=row["symbol"],
        signal_type=row.get("signal_type", "NEUTRAL"),
        alpha_score=float(row.get("alpha_score", 0.0)),
        news_score=float(row.get("news_score", 0.0)),
        bulk_deal_score=float(row.get("bulk_deal_score", 0.0)),
        delivery_score=float(row.get("delivery_score", 0.0)),
        vix_score=float(row.get("vix_score", 0.0)),
        generated_at=gen_at or datetime.now(tz=timezone.utc),
        ttl_minutes=ttl_minutes,
        contributing_events=ce if isinstance(ce, list) else [],
        expired=expired,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=SignalsListResponse)
async def get_signals(
    request: Request,
    symbol: str | None = Query(None, description="Filter by NSE symbol"),
    limit: int = Query(50, ge=1, le=200),
) -> SignalsListResponse:
    """Get latest fused signals.

    - If *symbol* is provided: return signals for that symbol only.
    - Else: return the latest signal per symbol for all watched symbols.

    Fast path: checks ``signal_cache`` (in-memory TTL cache) first.
    Falls back to ``pg_store.get_latest_signals()`` on a cache miss.
    """
    try:
        pg = request.app.state.pg
        rows: list[dict] = []

        if symbol:
            # Try in-memory cache first for a single symbol
            cached = signal_cache.get(symbol.upper())
            if cached is not None:
                rows = [cached]
            else:
                # Fall back to PG (returns latest non-expired signal)
                row = await pg.get_signal(symbol.upper())
                if row is not None:
                    rows = [row]
        else:
            # For the full list we query PG (cache doesn't hold all symbols)
            rows = await pg.get_latest_signals(limit=limit)

        signals = [_dict_to_signal_response(r) for r in rows]
        return SignalsListResponse(
            signals=signals,
            count=len(signals),
            generated_at=datetime.now(tz=timezone.utc),
        )
    except Exception as exc:
        log.error("api.get_signals.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to retrieve signals") from exc


@router.get("/{symbol}", response_model=SignalResponse)
async def get_signal_for_symbol(symbol: str, request: Request) -> SignalResponse:
    """Get the latest signal for a specific symbol.

    Returns 404 if no valid (non-expired) signal exists for the symbol.
    """
    try:
        sym = symbol.upper()
        pg = request.app.state.pg

        # Fast path: in-memory cache
        cached = signal_cache.get(sym)
        if cached is not None:
            return _dict_to_signal_response(cached)

        # Fallback: PG (only returns non-expired signals)
        row = await pg.get_signal(sym)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No active signal found for symbol '{sym}'")

        return _dict_to_signal_response(row)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("api.get_signal_for_symbol.failed", symbol=symbol, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to retrieve signal") from exc
