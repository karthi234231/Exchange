"""Pydantic v2 response models for all FastAPI endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SignalResponse(BaseModel):
    symbol: str
    signal_type: str          # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
    alpha_score: float
    news_score: float
    bulk_deal_score: float
    delivery_score: float
    vix_score: float
    generated_at: datetime
    ttl_minutes: int
    contributing_events: list[str]
    expired: bool


class NewsEventResponse(BaseModel):
    id: str
    source: str
    headline: str
    url: Optional[str]
    published_at: datetime
    symbols: list[str]
    event_type: str
    sentiment: str
    sentiment_score: float
    tags: list[str]


class BulkDealResponse(BaseModel):
    id: str
    deal_date: str
    symbol: str
    client_name: str
    deal_type: str     # bulk | block
    direction: str     # BUY | SELL
    quantity: int
    price: float
    deal_value_cr: float


class HealthResponse(BaseModel):
    status: str        # healthy | degraded | unhealthy
    flows: dict        # per-flow health from PipelineHealthSummary
    overall_healthy: bool
    checked_at: datetime


class BacktestSummaryResponse(BaseModel):
    period_days: int
    total_evaluated: int
    win_rate: float
    avg_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    by_signal_type: dict
    by_symbol: dict
    best_signal_type: str
    best_symbol: str
    recent_10: list[dict]


class SignalsListResponse(BaseModel):
    signals: list[SignalResponse]
    count: int
    generated_at: datetime


class NewsListResponse(BaseModel):
    events: list[NewsEventResponse]
    count: int
    filters_applied: dict


class BulkDealsListResponse(BaseModel):
    deals: list[BulkDealResponse]
    count: int
    trade_date: str
