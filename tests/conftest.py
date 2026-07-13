"""
Shared pytest fixtures for NSE AlgoTrader test suite.

Fixtures here are available to all test files without import.
"""
from __future__ import annotations

import os
os.environ.setdefault("TRADING_MODE", "mock")
os.environ.setdefault("RESEARCH_MODE", "true")
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

# ── Stage gate: set ALGO_STAGE=4 so Stage 4 modules don't raise on import ──
os.environ.setdefault("ALGO_STAGE", "4")


# ---------------------------------------------------------------------------
# Synthetic OHLCV data
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base_price: float, atr: float, seed: int, drift: float = 0.0) -> pl.DataFrame:
    """GBM-based OHLCV DataFrame with required columns for the full pipeline."""
    rng = np.random.default_rng(seed)
    dt = 1 / (252 * 26)
    sigma = atr / base_price
    log_returns = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    closes = base_price * np.cumprod(np.exp(log_returns))
    opens = np.empty(n)
    opens[0] = base_price
    opens[1:] = closes[:-1] * (1 + rng.normal(0, 0.001, n - 1))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volumes = rng.integers(100_000, 2_000_000, n)
    tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
    tr[0] = highs[0] - lows[0]
    atr14 = np.array([tr[max(0, i - 13): i + 1].mean() for i in range(n)])
    base_ts = datetime(2023, 1, 2, 9, 15)
    timestamps = [base_ts + timedelta(minutes=15 * i) for i in range(n)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": opens.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "close": closes.tolist(),
        "volume": volumes.tolist(),
        "atr14": atr14.tolist(),
        "symbol": ["RELIANCE"] * n,
    })


@pytest.fixture(scope="session")
def sample_ohlcv_small():
    """200-bar synthetic OHLCV for fast unit tests."""
    return _make_ohlcv(n=200, base_price=2800.0, atr=15.0, seed=42)


@pytest.fixture(scope="session")
def sample_ohlcv_large():
    """3000-bar synthetic OHLCV — sufficient for walk-forward splits (min 5 folds × 200 trades)."""
    return _make_ohlcv(n=3000, base_price=2800.0, atr=15.0, seed=42)


@pytest.fixture(scope="session")
def sample_ohlcv_trending_up():
    """300-bar strongly trending-up OHLCV (positive drift for labeling tests)."""
    return _make_ohlcv(n=300, base_price=2800.0, atr=15.0, seed=7, drift=5.0)


@pytest.fixture(scope="session")
def sample_ohlcv_trending_down():
    """300-bar strongly trending-down OHLCV (negative drift for labeling tests)."""
    return _make_ohlcv(n=300, base_price=2800.0, atr=15.0, seed=13, drift=-5.0)


# ---------------------------------------------------------------------------
# Tick data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_ticks_ns():
    """100 synthetic tick timestamps in nanoseconds for Hawkes tests."""
    rng = np.random.default_rng(42)
    inter_arrivals = rng.exponential(1 / 5.0, 100)  # Poisson rate=5/sec
    times_sec = np.cumsum(inter_arrivals)
    base_ns = int(datetime(2023, 1, 2, 9, 15).timestamp() * 1e9)
    return base_ns + (times_sec * 1e9).astype(np.int64)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cost_model():
    """Shared IndiaCostModel instance."""
    try:
        from algo_trading.risk.cost_model import IndiaCostModel
        return IndiaCostModel()
    except ImportError:
        pytest.skip("algo_trading.risk.cost_model not importable")


# ---------------------------------------------------------------------------
# Order book snapshot helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def default_ob_snapshot():
    """Default L2 order book snapshot for TFI tests."""
    try:
        from algo_trading.features.tfi import OrderBookSnapshot
        return OrderBookSnapshot(
            timestamp_ns=int(datetime(2023, 1, 2, 9, 30).timestamp() * 1e9),
            symbol="RELIANCE",
            bids=[(2800.0, 1000), (2799.5, 800), (2799.0, 600), (2798.5, 400), (2798.0, 200)],
            asks=[(2800.5, 900), (2801.0, 700), (2801.5, 500), (2802.0, 300), (2802.5, 100)],
        )
    except ImportError:
        pytest.skip("tfi module not available")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end pipeline tests (slower)")
    config.addinivalue_line("markers", "stage3: requires Stage 3 (sub-minute data, TrueData)")
    config.addinivalue_line("markers", "stage4: requires Stage 4 (RL, F&O, sentiment)")
    config.addinivalue_line("markers", "slow: tests that take > 5 seconds")
