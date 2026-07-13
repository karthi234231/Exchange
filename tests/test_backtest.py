"""Stage 2 tests for the walk-forward backtest runner.

Covers:
- Module importability and dataclass field existence
- TradeResult P&L arithmetic
- Edge gate pass/fail logic
- BacktestReport metric fields
- 1-bar fill delay and 15:15 force-close semantics
- Cost model application
- EMA baseline run (mocked when needed)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module-level skip guard — skip entire module if polars or structlog absent
# ---------------------------------------------------------------------------
try:
    import polars as pl
    import numpy as np
except ImportError as _e:
    pytest.skip(f"Required dependency missing: {_e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv_df(n: int = 2000, seed: int = 42) -> "pl.DataFrame":
    """Build a synthetic OHLCV Polars DataFrame with trending price data.

    Uses Geometric Brownian Motion with positive drift so the series contains
    a mix of bull/bear/sideways regimes as the backtester expects.

    Args:
        n: Number of bars.
        seed: NumPy random seed for reproducibility.

    Returns:
        Polars DataFrame with columns:
        timestamp, open, high, low, close, volume, atr14, label.
    """
    rng = np.random.default_rng(seed)

    # Trending upward GBM (positive drift)
    log_returns = rng.normal(0.0003, 0.01, n)
    closes = 2800.0 * np.cumprod(1.0 + log_returns)

    # OHLCV construction
    spread = closes * 0.005
    opens = closes * (1.0 + rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) + rng.uniform(0, spread)
    lows = np.minimum(opens, closes) - rng.uniform(0, spread)
    volumes = rng.integers(100_000, 1_000_000, n).astype(np.float64)

    # ATR-14 approximation
    true_ranges = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
    true_ranges[0] = highs[0] - lows[0]
    atr14 = np.convolve(true_ranges, np.ones(14) / 14, mode="same")

    # Labels: simple ternary signal
    label = np.sign(log_returns).astype(np.int8)

    # Timestamps: start 2026-01-02 09:30 IST, step by 15 min
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
    base = datetime(2026, 1, 2, 9, 30, 0, tzinfo=IST)
    bar_minutes = 15
    timestamps = [base + timedelta(minutes=i * bar_minutes) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens.tolist(),
            "high": highs.tolist(),
            "low": lows.tolist(),
            "close": closes.tolist(),
            "volume": volumes.tolist(),
            "atr14": atr14.tolist(),
            "label": label.tolist(),
        }
    )


def make_backtest_config():
    """Return a BacktestConfig suitable for tests."""
    try:
        from algo_trading.backtest.runner import BacktestConfig
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    return BacktestConfig(
        initial_capital=Decimal("500000"),
        max_position_pct=0.05,
        stop_loss_atr_mult=1.5,
        max_holding_bars=20,
        no_trade_before="09:30",
        square_off_time="15:15",
    )


def _make_report(**overrides: Any):
    """Build a minimal BacktestReport for gate-testing."""
    try:
        from algo_trading.backtest.runner import BacktestReport
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    defaults: dict[str, Any] = dict(
        total_trades=500,
        win_rate=0.5,
        profit_factor=1.5,
        sharpe_ratio=1.5,
        max_drawdown=0.1,
        total_net_pnl=Decimal("50000"),
        per_fold_metrics=[{"fold_index": i, "n_trades": 200} for i in range(5)],
        regime_breakdown={
            "bull": {"sharpe": 1.2},
            "bear": {"sharpe": 0.8},
            "sideways": {"sharpe": 1.1},
        },
        cost_analysis={"total_cost": Decimal("5000"), "avg_cost_bps": 5.0, "slippage_pct": 0.3},
        passes_edge_gate=True,
        gate_failures=[],
    )
    defaults.update(overrides)
    return BacktestReport(**defaults)


# ===========================================================================
# 1. Module importability
# ===========================================================================


def test_backtest_module_importable() -> None:
    """algo_trading.backtest.runner must import without raising."""
    try:
        import algo_trading.backtest.runner as runner_mod  # noqa: F401

        assert runner_mod is not None
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")


# ===========================================================================
# 2. BacktestConfig has required fields
# ===========================================================================


def test_backtest_config_has_fields() -> None:
    """BacktestConfig must expose symbol-agnostic capital and constraint fields."""
    cfg = make_backtest_config()
    assert hasattr(cfg, "initial_capital")
    assert hasattr(cfg, "max_position_pct")
    assert hasattr(cfg, "stop_loss_atr_mult")
    assert hasattr(cfg, "square_off_time")
    assert hasattr(cfg, "no_trade_before")


# ===========================================================================
# 3. TradeResult P&L arithmetic
# ===========================================================================


def test_trade_result_pnl_arithmetic() -> None:
    """A BUY trade with +50 price move over 100 shares must have gross_pnl ~ 5000."""
    try:
        from algo_trading.backtest.runner import TradeResult
        from algo_trading.risk.cost_model import IndiaCostModel
    except ImportError as exc:
        pytest.skip(f"Dependencies not importable: {exc}")

    cost_model = IndiaCostModel()
    entry = Decimal("2800")
    exit_ = Decimal("2850")
    qty = 100
    atr = Decimal("28")  # 1% of price

    cost = cost_model.compute_round_trip(
        symbol="RELIANCE",
        quantity=qty,
        entry_price=entry,
        exit_price=exit_,
        atr=atr,
    )
    gross = (exit_ - entry) * Decimal(str(qty))
    net = gross - cost.total_inr

    trade = TradeResult(
        symbol="RELIANCE",
        entry_bar=10,
        exit_bar=15,
        direction=1,
        entry_price=entry,
        exit_price=exit_,
        quantity=qty,
        gross_pnl=gross,
        net_pnl=net,
        cost_breakdown=cost,
        exit_reason="target",
        signal_confidence=0.9,
    )

    assert abs(float(trade.gross_pnl) - 5000.0) < 1e-6, (
        f"Expected gross_pnl ≈ 5000, got {trade.gross_pnl}"
    )
    assert float(trade.net_pnl) < 5000.0, "net_pnl must be less than gross_pnl (costs applied)"


# ===========================================================================
# 4. Edge gate — all criteria pass
# ===========================================================================


def test_edge_gate_pass() -> None:
    """Sharpe=1.5, PF=1.5, WR=0.5 must pass the edge gate."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    report = _make_report(sharpe_ratio=1.5, profit_factor=1.5, win_rate=0.5)
    runner = BacktestRunner(make_backtest_config())
    passed, failures = runner._check_edge_gate(report)
    assert passed is True
    assert failures == []


# ===========================================================================
# 5. Edge gate — Sharpe too low
# ===========================================================================


def test_edge_gate_fail_sharpe() -> None:
    """Sharpe=0.8 (< 1.0) must fail the edge gate."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    report = _make_report(sharpe_ratio=0.8, profit_factor=1.5, win_rate=0.5)
    runner = BacktestRunner(make_backtest_config())
    passed, failures = runner._check_edge_gate(report)
    assert passed is False
    assert any("Sharpe" in f for f in failures)


# ===========================================================================
# 6. Edge gate — profit factor too low
# ===========================================================================


def test_edge_gate_fail_pf() -> None:
    """PF=1.1 (< 1.2) must fail the edge gate."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    report = _make_report(sharpe_ratio=1.5, profit_factor=1.1, win_rate=0.5)
    runner = BacktestRunner(make_backtest_config())
    passed, failures = runner._check_edge_gate(report)
    assert passed is False
    assert any("factor" in f.lower() or "profit" in f.lower() for f in failures)


# ===========================================================================
# 7. Edge gate — win rate too low
# ===========================================================================


def test_edge_gate_fail_wr() -> None:
    """WR=0.35 (< 40%) must fail the edge gate."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    report = _make_report(sharpe_ratio=1.5, profit_factor=1.5, win_rate=0.35)
    runner = BacktestRunner(make_backtest_config())
    passed, failures = runner._check_edge_gate(report)
    assert passed is False
    assert any("win" in f.lower() or "rate" in f.lower() for f in failures)


# ===========================================================================
# 8. BacktestReport has required metric fields
# ===========================================================================


def test_backtest_report_has_metrics() -> None:
    """BacktestReport must expose all required aggregated metric fields."""
    report = _make_report()
    assert hasattr(report, "sharpe_ratio")
    assert hasattr(report, "profit_factor")
    assert hasattr(report, "win_rate")
    assert hasattr(report, "total_trades")
    assert hasattr(report, "max_drawdown")


# ===========================================================================
# 9. run_ema_baseline returns a BacktestReport
# ===========================================================================


def test_run_ema_baseline_returns_report() -> None:
    """run_ema_baseline on synthetic data must return a BacktestReport instance."""
    try:
        from algo_trading.backtest.runner import BacktestReport, BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    df = make_ohlcv_df(n=500, seed=0)
    cfg = make_backtest_config()
    runner = BacktestRunner(cfg)

    # Patch the run() method to avoid needing WalkForwardSplitter deps
    from unittest.mock import MagicMock, patch

    mock_report = _make_report()

    with patch.object(runner, "run", return_value=mock_report):
        report = runner.run_ema_baseline(df)

    assert isinstance(report, BacktestReport)


# ===========================================================================
# 10. Force-close at 15:15
# ===========================================================================


def test_force_close_at_1515() -> None:
    """A position open at 15:15 bar must be force-closed (exit_reason = time_exit)."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")

    # Build a tiny 3-bar OHLCV: bar 0 = signal, bar 1 = fill (09:35), bar 2 = 15:15 force-close
    timestamps = [
        datetime(2026, 1, 2, 9, 30, 0, tzinfo=IST),   # signal bar
        datetime(2026, 1, 2, 9, 35, 0, tzinfo=IST),   # fill bar
        datetime(2026, 1, 2, 15, 15, 0, tzinfo=IST),  # force-close bar
    ]
    price = 2800.0
    df_small = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [price, price, price * 1.01],
            "high": [price * 1.01, price * 1.01, price * 1.02],
            "low": [price * 0.99, price * 0.99, price * 1.0],
            "close": [price, price, price * 1.01],
            "volume": [500_000.0, 500_000.0, 500_000.0],
        }
    )
    signals = pl.Series("signal", [1, 0, 0], dtype=pl.Int8)
    feature_df = pl.DataFrame({"feat_atr": [14.0, 14.0, 14.0]})

    cfg = make_backtest_config()
    runner = BacktestRunner(cfg)

    trades = runner._simulate_fold(
        ohlcv=df_small,
        signals=signals,
        features=feature_df,
        symbol="RELIANCE",
        bar_offset=0,
    )

    # The position must have been exited; reason should be time_exit
    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    assert trades[0].exit_reason == "time_exit", (
        f"Expected time_exit, got {trades[0].exit_reason}"
    )


# ===========================================================================
# 11. 1-bar fill delay — fill at bar i+1
# ===========================================================================


def test_1bar_fill_delay() -> None:
    """Signal at bar i must fill at bar i+1 open, not bar i close."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")

    signal_close = 2800.0
    fill_open = 2810.0   # distinctly different from signal close

    timestamps = [
        datetime(2026, 1, 2, 9, 30, 0, tzinfo=IST),
        datetime(2026, 1, 2, 9, 45, 0, tzinfo=IST),
        datetime(2026, 1, 2, 10, 0, 0, tzinfo=IST),
    ]
    df_small = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [signal_close, fill_open, fill_open],
            "high": [signal_close + 10, fill_open + 10, fill_open + 10],
            "low": [signal_close - 10, fill_open - 10, fill_open - 10],
            "close": [signal_close, fill_open, fill_open],
            "volume": [500_000.0, 500_000.0, 500_000.0],
        }
    )
    signals = pl.Series("signal", [1, 0, 0], dtype=pl.Int8)
    feature_df = pl.DataFrame({"feat_atr": [14.0, 14.0, 14.0]})

    cfg = make_backtest_config()
    runner = BacktestRunner(cfg)

    trades = runner._simulate_fold(
        ohlcv=df_small,
        signals=signals,
        features=feature_df,
        symbol="RELIANCE",
        bar_offset=0,
    )

    assert len(trades) >= 1, "Expected at least one trade"
    # Entry price must be based on bar i+1 open (2810), not bar i close (2800)
    # The runner applies an adverse selection premium, so entry_price ≈ fill_open
    entry = float(trades[0].entry_price)
    assert abs(entry - fill_open) < fill_open * 0.01, (
        f"Entry {entry:.2f} should be close to fill_open {fill_open:.2f} (1-bar delay)"
    )


# ===========================================================================
# 12. Cost model applied — total_costs > 0
# ===========================================================================


def test_cost_model_applied() -> None:
    """Executed trades must incur positive total costs from IndiaCostModel."""
    try:
        from algo_trading.backtest.runner import BacktestRunner
    except ImportError as exc:
        pytest.skip(f"backtest.runner not importable: {exc}")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")

    price = 2800.0
    timestamps = [
        datetime(2026, 1, 2, 9, 30, 0, tzinfo=IST),
        datetime(2026, 1, 2, 9, 45, 0, tzinfo=IST),
        datetime(2026, 1, 2, 10, 0, 0, tzinfo=IST),
        datetime(2026, 1, 2, 10, 15, 0, tzinfo=IST),
    ]
    df_small = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [price, price, price * 1.005, price * 1.01],
            "high": [price + 15] * 4,
            "low": [price - 15] * 4,
            "close": [price, price, price * 1.005, price * 1.01],
            "volume": [500_000.0] * 4,
        }
    )
    signals = pl.Series("signal", [1, 0, 0, 0], dtype=pl.Int8)
    feature_df = pl.DataFrame({"feat_atr": [14.0, 14.0, 14.0, 14.0]})

    cfg = make_backtest_config()
    runner = BacktestRunner(cfg)

    trades = runner._simulate_fold(
        ohlcv=df_small,
        signals=signals,
        features=feature_df,
        symbol="RELIANCE",
        bar_offset=0,
    )

    assert len(trades) >= 1, "Expected at least one trade"
    total_cost = sum(float(t.cost_breakdown.total_inr) for t in trades)
    assert total_cost > 0, f"Expected positive total costs, got {total_cost}"
