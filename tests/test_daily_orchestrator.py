"""
tests/test_daily_orchestrator.py — Unit tests for DailyOrchestrator signal routing.

Tests the pure logic paths in _on_bar() without instantiating any real
infrastructure (QuestDB, Shoonya, ZMQ, APScheduler).  Every heavy dependency
is patched at the class level; only the attributes actually exercised by each
test are set on the stub instance.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _Bar:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


def _bar(symbol: str = "NIFTY", close: float = 19_500.0) -> _Bar:
    return _Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 15, 10, 15, tzinfo=timezone.utc),
        open=Decimal(str(close - 50)),
        high=Decimal(str(close + 100)),
        low=Decimal(str(close - 100)),
        close=Decimal(str(close)),
        volume=1000,
    )


def _signal(direction: int = 1, confidence: float = 0.8) -> object:
    """Return a minimal TradeSignal-like object."""
    from algo_trading.execution.order_executor import TradeSignal
    return TradeSignal(
        symbol="NIFTY",
        direction=direction,
        entry_price=Decimal("19500"),
        stop_loss_price=Decimal("19400"),
        atr=Decimal("100"),
        confidence=confidence,
        timestamp=None,
    )


# ---------------------------------------------------------------------------
# Fixture: patch all heavy deps, return a bare DailyOrchestrator instance
# ---------------------------------------------------------------------------

_PATCHES = {
    "algo_trading.app.daily_orchestrator.QuestDBStore": MagicMock,
    "algo_trading.app.daily_orchestrator.LiveSystemState": MagicMock,
    "algo_trading.app.daily_orchestrator.ShoonyaBroker": MagicMock,
    "algo_trading.app.daily_orchestrator.AlertSender": MagicMock,
    "algo_trading.app.daily_orchestrator.OrderFSM": MagicMock,
    "algo_trading.app.daily_orchestrator.DualLockRateLimiter": MagicMock,
    "algo_trading.app.daily_orchestrator.RiskManager": MagicMock,
    "algo_trading.app.daily_orchestrator.OrderExecutor": MagicMock,
    "algo_trading.app.daily_orchestrator.TimeBarBuilder": MagicMock,
    "algo_trading.app.daily_orchestrator.TickValidator": MagicMock,
    "algo_trading.app.daily_orchestrator.DataQualityMonitor": MagicMock,
    "algo_trading.app.daily_orchestrator.TickPublisher": MagicMock,
    "algo_trading.app.daily_orchestrator.BarPublisher": MagicMock,
    "algo_trading.app.daily_orchestrator.BackgroundScheduler": MagicMock,
    "algo_trading.app.daily_orchestrator.FilteredEMACrossStrategy": MagicMock,
    "algo_trading.app.daily_orchestrator.MeanReversionStrategy": MagicMock,
    "algo_trading.app.daily_orchestrator.MultiTimeframeConfirmation": MagicMock,
}


def _make_orch():
    """
    Build a DailyOrchestrator whose __init__ runs but all heavy deps are mocks.
    Returns the instance; call sites set additional state as needed.
    """
    with patch.multiple("algo_trading.app.daily_orchestrator", **_PATCHES):
        from algo_trading.app.daily_orchestrator import DailyOrchestrator
        orch = DailyOrchestrator()

    # Convenience: wire up real threading events (already set by __init__)
    return orch


# ---------------------------------------------------------------------------
# TestMTFAlwaysUpdated
# ---------------------------------------------------------------------------

class TestMTFAlwaysUpdated:
    """MTF accumulator must receive every bar even when trading is disabled."""

    def test_mtf_updated_when_trading_disabled(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()  # clear → disabled
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._strategy.feed_bar = MagicMock(return_value=None)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._mtf.update.assert_called_once()   # MTF must have been called
        orch._strategy.feed_bar.assert_not_called()   # but no signal generation

    def test_mtf_updated_before_kill_switch_check(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = True
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._strategy.feed_bar = MagicMock(return_value=None)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._mtf.update.assert_called_once()
        orch._strategy.feed_bar.assert_not_called()

    def test_bar_always_persisted(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()   # disabled
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._store.write_bar.assert_called_once()
        orch._bar_publisher.publish_bar.assert_called_once()


# ---------------------------------------------------------------------------
# TestSignalSelection
# ---------------------------------------------------------------------------

class TestSignalSelection:
    """Correct signal selected from the two strategy outputs."""

    def _enabled_orch(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = False
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._mtf.get_size_multiplier = MagicMock(return_value=1.0)
        orch._on_signal = MagicMock()
        return orch

    def test_trend_signal_routed(self):
        orch = self._enabled_orch()
        sig = _signal(direction=1, confidence=0.8)
        orch._strategy.feed_bar = MagicMock(return_value=sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._on_signal.assert_called_once_with(sig)

    def test_mean_rev_signal_routed(self):
        orch = self._enabled_orch()
        sig = _signal(direction=-1, confidence=0.75)
        orch._strategy.feed_bar = MagicMock(return_value=None)
        orch._mean_rev.feed_bar = MagicMock(return_value=sig)

        orch._on_bar(_bar())

        orch._on_signal.assert_called_once_with(sig)

    def test_no_signal_no_route(self):
        orch = self._enabled_orch()
        orch._strategy.feed_bar = MagicMock(return_value=None)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._on_signal.assert_not_called()

    def test_both_fire_higher_confidence_wins(self):
        orch = self._enabled_orch()
        trend_sig = _signal(direction=1, confidence=0.9)
        rev_sig = _signal(direction=-1, confidence=0.7)
        orch._strategy.feed_bar = MagicMock(return_value=trend_sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=rev_sig)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed is trend_sig

    def test_both_fire_mean_rev_wins_when_higher_confidence(self):
        orch = self._enabled_orch()
        trend_sig = _signal(direction=1, confidence=0.6)
        rev_sig = _signal(direction=-1, confidence=0.85)
        orch._strategy.feed_bar = MagicMock(return_value=trend_sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=rev_sig)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed is rev_sig

    def test_equal_confidence_trend_wins(self):
        """When confidence is equal, trend signal takes priority (index order)."""
        orch = self._enabled_orch()
        trend_sig = _signal(direction=1, confidence=0.8)
        rev_sig = _signal(direction=-1, confidence=0.8)
        orch._strategy.feed_bar = MagicMock(return_value=trend_sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=rev_sig)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed is trend_sig


# ---------------------------------------------------------------------------
# TestMTFMultiplier
# ---------------------------------------------------------------------------

class TestMTFMultiplier:
    """MTF multiplier correctly gates and scales signals."""

    def _enabled_orch(self, multiplier: float):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = False
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._mtf.get_size_multiplier = MagicMock(return_value=multiplier)
        orch._on_signal = MagicMock()
        return orch

    def test_counter_trend_signal_suppressed(self):
        orch = self._enabled_orch(0.0)
        orch._strategy.feed_bar = MagicMock(return_value=_signal())
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._on_signal.assert_not_called()

    def test_aligned_signal_confidence_unchanged(self):
        orch = self._enabled_orch(1.0)
        sig = _signal(confidence=0.8)
        orch._strategy.feed_bar = MagicMock(return_value=sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed.confidence == pytest.approx(0.8)

    def test_sideways_confidence_halved(self):
        orch = self._enabled_orch(0.5)
        sig = _signal(confidence=0.8)
        orch._strategy.feed_bar = MagicMock(return_value=sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed.confidence == pytest.approx(0.4)

    def test_sideways_preserves_other_fields(self):
        orch = self._enabled_orch(0.5)
        sig = _signal(direction=-1, confidence=1.0)
        orch._strategy.feed_bar = MagicMock(return_value=sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed.direction == -1
        assert routed.symbol == "NIFTY"
        assert routed.entry_price == Decimal("19500")
        assert routed.stop_loss_price == Decimal("19400")

    def test_multiplier_applied_to_selected_signal(self):
        """Ensure the multiplier is applied to whichever signal was selected."""
        orch = self._enabled_orch(0.5)
        orch._strategy.feed_bar = MagicMock(return_value=None)
        rev_sig = _signal(confidence=0.9)
        orch._mean_rev.feed_bar = MagicMock(return_value=rev_sig)

        orch._on_bar(_bar())

        routed = orch._on_signal.call_args[0][0]
        assert routed.confidence == pytest.approx(0.45)

    def test_multiplier_query_uses_correct_symbol_and_direction(self):
        orch = self._enabled_orch(1.0)
        sig = _signal(direction=-1)
        orch._strategy.feed_bar = MagicMock(return_value=sig)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar(symbol="BANKNIFTY"))

        orch._mtf.get_size_multiplier.assert_called_once_with("BANKNIFTY", -1)


# ---------------------------------------------------------------------------
# TestKillSwitch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    """Kill switch correctly blocks signal generation."""

    def test_kill_switch_blocks_signals(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = True
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._strategy.feed_bar = MagicMock(return_value=_signal())
        orch._mean_rev.feed_bar = MagicMock(return_value=None)
        orch._on_signal = MagicMock()

        orch._on_bar(_bar())

        orch._strategy.feed_bar.assert_not_called()
        orch._on_signal.assert_not_called()

    def test_kill_switch_still_persists_bar(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = True
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)

        orch._on_bar(_bar())

        orch._store.write_bar.assert_called_once()


# ---------------------------------------------------------------------------
# TestBothStrategiesFed
# ---------------------------------------------------------------------------

class TestBothStrategiesFed:
    """Both strategies receive every bar when trading is enabled."""

    def test_both_strategies_always_fed(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = False
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._mtf.get_size_multiplier = MagicMock(return_value=1.0)
        orch._strategy.feed_bar = MagicMock(return_value=None)
        orch._mean_rev.feed_bar = MagicMock(return_value=None)
        orch._on_signal = MagicMock()

        b = _bar()
        orch._on_bar(b)

        orch._strategy.feed_bar.assert_called_once_with(b)
        orch._mean_rev.feed_bar.assert_called_once_with(b)

    def test_both_strategies_fed_regardless_of_signal_outcome(self):
        """Even when MTF suppresses the signal, both strategies must run."""
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = False
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._mtf.get_size_multiplier = MagicMock(return_value=0.0)  # suppress
        orch._strategy.feed_bar = MagicMock(return_value=_signal())
        orch._mean_rev.feed_bar = MagicMock(return_value=None)
        orch._on_signal = MagicMock()

        orch._on_bar(_bar())

        orch._strategy.feed_bar.assert_called_once()
        orch._mean_rev.feed_bar.assert_called_once()
        orch._on_signal.assert_not_called()


# ---------------------------------------------------------------------------
# TestMultiSymbolIsolation
# ---------------------------------------------------------------------------

class TestMultiSymbolIsolation:
    """MTF multiplier query is per-symbol — two symbols do not cross-contaminate."""

    def test_multiplier_queried_per_symbol(self):
        orch = _make_orch()
        orch._trading_enabled = threading.Event()
        orch._trading_enabled.set()
        orch._risk_manager.is_kill_switch_active = False
        orch._store.write_bar = MagicMock()
        orch._bar_publisher.publish_bar = MagicMock()
        orch._mtf.update = MagicMock(return_value=None)
        orch._on_signal = MagicMock()

        calls = []
        def _get_mult(sym, direction):
            calls.append(sym)
            return 1.0

        orch._mtf.get_size_multiplier = _get_mult
        orch._strategy.feed_bar = MagicMock(return_value=_signal())
        orch._mean_rev.feed_bar = MagicMock(return_value=None)

        orch._on_bar(_bar(symbol="NIFTY"))
        orch._on_bar(_bar(symbol="BANKNIFTY"))

        assert calls == ["NIFTY", "BANKNIFTY"]
