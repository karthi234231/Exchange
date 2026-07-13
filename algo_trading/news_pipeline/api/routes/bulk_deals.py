"""Bulk-deal endpoints — /bulk-deals."""
from __future__ import annotations

import datetime

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from algo_trading.news_pipeline.api.schemas import BulkDealResponse, BulkDealsListResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/bulk-deals", tags=["bulk-deals"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_bulk_deal_response(row: dict) -> BulkDealResponse:
    """Convert a PGStore row dict to a BulkDealResponse.

    ``deal_date`` may be a ``datetime.date``, ``datetime.datetime``, or ISO string.
    Normalise to a plain ``YYYY-MM-DD`` string for the response.
    """
    deal_date_raw = row.get("deal_date")
    if isinstance(deal_date_raw, datetime.datetime):
        deal_date_str = deal_date_raw.date().isoformat()
    elif isinstance(deal_date_raw, datetime.date):
        deal_date_str = deal_date_raw.isoformat()
    elif isinstance(deal_date_raw, str):
        # Trim to date portion in case it's a full ISO-8601 string
        deal_date_str = deal_date_raw[:10]
    else:
        deal_date_str = str(deal_date_raw)

    return BulkDealResponse(
        id=row["id"],
        deal_date=deal_date_str,
        symbol=row.get("symbol", ""),
        client_name=row.get("client_name", ""),
        deal_type=row.get("deal_type", "bulk"),
        direction=row.get("direction", "BUY"),
        quantity=int(row.get("quantity", 0)),
        price=float(row.get("price", 0.0)),
        deal_value_cr=float(row.get("deal_value_cr", 0.0)),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=BulkDealsListResponse)
async def get_bulk_deals(
    request: Request,
    symbol: str | None = Query(None, description="Filter by NSE symbol"),
) -> BulkDealsListResponse:
    """Get today's bulk and block deals.

    Returns all deals reported by NSE for the current calendar date,
    ordered by deal value (largest first).  Optionally filtered to a single
    *symbol*.
    """
    try:
        pg = request.app.state.pg
        sym = symbol.upper() if symbol else None

        rows = await pg.get_today_bulk_deals(symbol=sym)
        deals = [_dict_to_bulk_deal_response(r) for r in rows]

        return BulkDealsListResponse(
            deals=deals,
            count=len(deals),
            trade_date=datetime.date.today().isoformat(),
        )
    except Exception as exc:
        log.error("api.get_bulk_deals.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to retrieve bulk deals") from exc


@router.get("/{symbol}/history")
async def get_bulk_deals_history(
    symbol: str,
    request: Request,
    days: int = Query(30, ge=1, le=90, description="Look-back window in calendar days"),
) -> dict:
    """Get bulk/block deal history for a specific symbol.

    Returns all deals for *symbol* over the past *days* calendar days,
    plus a lightweight summary (total count, total value in crore INR,
    buy count, sell count).

    Response schema
    ---------------
    ``{ symbol, period_days, total_deals, total_value_cr, buy_count, sell_count, deals: [...] }``
    """
    try:
        pg = request.app.state.pg
        sym = symbol.upper()

        rows = await pg.get_bulk_deals_range(symbol=sym, days=days)
        deals = [_dict_to_bulk_deal_response(r) for r in rows]

        total_value_cr = round(sum(d.deal_value_cr for d in deals), 4)
        buy_count = sum(1 for d in deals if d.direction.upper() == "BUY")
        sell_count = sum(1 for d in deals if d.direction.upper() == "SELL")

        return {
            "symbol": sym,
            "period_days": days,
            "total_deals": len(deals),
            "total_value_cr": total_value_cr,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "deals": [d.model_dump() for d in deals],
        }
    except Exception as exc:
        log.error("api.get_bulk_deals_history.failed", symbol=symbol, error=str(exc))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve bulk deal history"
        ) from exc
