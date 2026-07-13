"""
FastAPI application entry point.

Lifespan:
- startup: connect PGStore, store in app.state.pg
           create SignalBacktester, store in app.state.backtester
- shutdown: close PGStore connection pool

CORS: allow all origins (for Streamlit dashboard on same machine)

Routes:
- /signals    — signal endpoints
- /news       — news endpoints
- /health     — health endpoints
- /bulk-deals — bulk deal endpoints
- /backtest   — backtest endpoints
- GET /       — redirect to /docs

Run:
    uvicorn algo_trading.news_pipeline.api.main:app --host 0.0.0.0 --port 8765

Or:
    python -m algo_trading.news_pipeline.api.main
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from algo_trading.news_pipeline.config import cfg
from algo_trading.news_pipeline.storage.pg_store import PGStore
from algo_trading.news_pipeline.backtest.signal_backtest import SignalBacktester
from algo_trading.news_pipeline.storage.parquet_store import ParquetStore
from algo_trading.news_pipeline.api.routes import signals, news, health, bulk_deals, backtest

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of shared resources."""
    # ── Startup ──────────────────────────────────────────────────────────
    pg = PGStore(cfg.POSTGRES_DSN)
    await pg.connect()
    app.state.pg = pg

    parquet = ParquetStore(cfg.PARQUET_DIR)
    app.state.parquet = parquet

    app.state.backtester = SignalBacktester(pg, parquet)

    log.info(
        "api.startup_complete",
        dsn=cfg.POSTGRES_DSN[:30] + "...",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
    )

    yield  # application is now serving requests

    # ── Shutdown ──────────────────────────────────────────────────────────
    await pg.close()
    log.info("api.shutdown_complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NSE News Pipeline API",
    description=(
        "Real-time news signals, bulk deals, and alpha scores for NSE algo trading. "
        "Backed by FinBERT sentiment analysis, spaCy NER, and signal fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Streamlit dashboard on the same machine
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(signals.router)
app.include_router(news.router)
app.include_router(health.router)
app.include_router(bulk_deals.router)
app.include_router(backtest.router)

# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect bare root to the Swagger UI."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "algo_trading.news_pipeline.api.main:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=False,
        log_level="info",
    )
