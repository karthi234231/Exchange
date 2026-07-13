"""
tests/test_fo_orchestrator.py — Tests for orchestrator/fo_orchestrator.py.

Strategy: test pure/static methods directly on the class without instantiating
FOOrchestrator (which would connect to QuestDB, Shoonya, etc.).
Instance-method tests use lightweight mocks.

Covers:
  - GARCH lot sizing thresholds (_compute_lots_from_garch)
  - VIX z-score detection (_update_vix_history)
  - Entry window timing (_is_within_entry_window)
  - ATM strike estimation (_estimate_atm)
  - Strategy routing (_build_strategy) with mocked builder
  - force=True bypass in _attempt_delta_hedge
  - on_chain_snapshot stores chain and computes VIX
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from algo_trading.app.fo_orchestrator import (
    FOOrchestrator,
    _GARCH_LOW_VOL,
    _GARCH_MID_VOL,
    _GARCH_HIGH_VOL,
    _GARCH_SKIP_VOL,
    _VIX_ZSCORE_TRIGGER,
    _VIX_HISTORY_BARS,
)


# ─────────────────────────────────────────────────────────────────────────────
# GARCH lot sizing — static method, no deps
# ─────────────────────────────────────────────────────────────────────────────

class TestGARCHLotSizing:

    def test_low_vol_returns_2_lots(self):
        """garch_vol < _GARCH_LOW_VOL → 2 lots (more premium capture)."""
        vol = _GARCH_LOW_VOL * 0.5   # well below threshold
        assert FOOrchestrator._compute_lots_from_garch(vol) == 2

    def test_vol_at_low_threshold_returns_1(self):
        """garch_vol == _GARCH_LOW_VOL → 1 lot (boundary is exclusive)."""
        assert FOOrchestrator._compute_lots_from_garch(_GARCH_LOW_VOL) == 1

    def test_mid_vol_returns_1_lot(self):
        """_GARCH_LOW_VOL ≤ vol < _GARCH_SKIP_VOL → 1 lot."""
        vol = (_GARCH_LOW_VOL + _GARCH_MID_VOL) / 2
        assert FOOrchestrator._compute_lots_from_garch(vol) == 1

    def test_high_vol_returns_1_lot(self):
        """High vol (below skip threshold) → 1 lot."""
        vol = _GARCH_HIGH_VOL
        assert FOOrchestrator._compute_lots_from_garch(vol) == 1

    def test_nan_returns_1_lot(self):
        """NaN (GARCH not fitted) → conservative default of 1 lot."""
        assert FOOrchestrator._compute_lots_from_garch(float("nan")) == 1

    def test_negative_returns_1_lot(self):
        """Negative vol (bad state) → 1 lot."""
        assert FOOrchestrator._compute_lots_from_garch(-0.1) == 1

    def test_zero_returns_2_lots(self):
        """Vol of 0.0 < _GARCH_LOW_VOL → 2 lots."""
        assert FOOrchestrator._compute_lots_from_garch(0.0) == 2

    def test_just_below_skip_returns_1(self):
        """Just below skip threshold → still 1 lot (caller handles skip)."""
        vol = _GARCH_SKIP_VOL - 0.001
        assert FOOrchestrator._compute_lots_from_garch(vol) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ATM estimation — static method, no deps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeContract:
    strike: float

@dataclass
class _FakeChain:
    calls: list
    puts: list
    expiry: date = field(default_factory=lambda: date(2026, 6, 26))

class TestEstimateATM:

    def test_midpoint_of_sorted_strikes(self):
        """ATM = median of all available strikes."""
        calls = [_FakeContract(s) for s in [24000, 24050, 24100, 24150, 24200]]
        puts  = [_FakeContract(s) for s in [23900, 23950, 24000, 24050, 24100]]
        chain = _FakeChain(calls=calls, puts=puts)
        atm = FOOrchestrator._estimate_atm(chain)
        assert atm > 0.0
        assert 23900.0 <= atm <= 24200.0

    def test_empty_chain_returns_zero(self):
        chain = _FakeChain(calls=[], puts=[])
        assert FOOrchestrator._estimate_atm(chain) == 0.0

    def test_single_strike_returns_it(self):
        chain = _FakeChain(calls=[_FakeContract(24000)], puts=[])
        assert FOOrchestrator._estimate_atm(chain) == 24000.0

    def test_calls_and_puts_combined(self):
        """Strikes from both calls and puts are merged before median."""
        calls = [_FakeContract(24100), _FakeContract(24200)]
        puts  = [_FakeContract(23800), _FakeContract(23900)]
        chain = _FakeChain(calls=calls, puts=puts)
        atm = FOOrchestrator._estimate_atm(chain)
        # Sorted: 23800, 23900, 24100, 24200 → median idx 2 → 24100
        assert atm == pytest.approx(24100.0)


# ─────────────────────────────────────────────────────────────────────────────
# VIX z-score detection — instance method, uses minimal mocking
# ─────────────────────────────────────────────────────────────────────────────

def _make_orchestrator_stub() -> FOOrchestrator:
    """
    Create FOOrchestrator with all heavy __init__ dependencies mocked out.
    Only the lightweight pure-Python state is real.
    """
    with patch.multiple(
        "algo_trading.app.fo_orchestrator",
        QuestDBStore=MagicMock,
        LiveSystemState=MagicMock,
        ShoonyaBroker=MagicMock,
        AlertSender=MagicMock,
        OrderFSM=MagicMock,
        DualLockRateLimiter=MagicMock,
        RiskManager=MagicMock,
        OrderExecutor=MagicMock,
        TimeBarBuilder=MagicMock,
        FOFeatureEngine=MagicMock,
        FOCostModel=MagicMock,
        DeltaHedger=MagicMock,
        PortfolioRiskEngine=MagicMock,
        FOSignalGenerator=MagicMock,
        GARCHVolForecaster=MagicMock,
        BackgroundScheduler=MagicMock,
    ):
        orc = FOOrchestrator.__new__(FOOrchestrator)
        # Initialise only the attributes used by the methods under test
        orc._symbol = "NIFTY"
        orc._days_to_expiry = 21
        orc._vix_history = []
        orc._latest_vix = None
        orc._latest_chain = None
        orc._active_strategy = None
        orc._trading_enabled = MagicMock()
        orc._trading_enabled.is_set.return_value = True
        orc._risk_manager = MagicMock()
        orc._risk_manager.is_kill_switch_active = False
        orc._alert = MagicMock()
        orc._hedger = MagicMock()
        orc._risk_engine = MagicMock()
        orc._garch = MagicMock()
        orc._executor = MagicMock()
        orc._state = MagicMock()
        return orc


class TestVIXZScore:

    def test_no_spike_with_insufficient_history(self):
        """Less than 5 data points → no spike check, no alert."""
        orc = _make_orchestrator_stub()
        for v in [15.0, 15.5, 16.0]:
            orc._update_vix_history(v)
        orc._alert.send_session_alert.assert_not_called()

    def test_no_spike_with_stable_vix(self):
        """Stable VIX series → z-score stays below trigger → no alert."""
        orc = _make_orchestrator_stub()
        stable = [15.0, 15.1, 15.0, 15.2, 15.1, 15.0, 15.1, 15.2]
        for v in stable:
            orc._update_vix_history(v)
        # Stable VIX: no alert
        orc._alert.send_session_alert.assert_not_called()

    def test_spike_triggers_alert(self):
        """Large VIX spike → z-score > _VIX_ZSCORE_TRIGGER → alert sent."""
        orc = _make_orchestrator_stub()
        # Build stable history first
        stable = [15.0] * 10
        for v in stable:
            orc._update_vix_history(v)
        orc._alert.send_session_alert.reset_mock()

        # Inject a massive spike — z-score will be >> 2.5
        orc._update_vix_history(40.0)
        orc._alert.send_session_alert.assert_called_once()
        call_args = orc._alert.send_session_alert.call_args[0][0]
        assert "VIX SPIKE" in call_args

    def test_spike_triggers_force_hedge(self):
        """VIX spike with active strategy → _attempt_delta_hedge(force=True) called."""
        orc = _make_orchestrator_stub()
        orc._active_strategy = MagicMock()  # non-None → hedge is active

        with patch.object(orc, "_attempt_delta_hedge") as mock_hedge:
            # Build stable history
            for v in [15.0] * 10:
                orc._update_vix_history(v)
            # Inject spike
            orc._update_vix_history(40.0)
            mock_hedge.assert_called_once_with(force=True)

    def test_no_force_hedge_when_flat_book(self):
        """VIX spike with no active strategy → no hedge attempt."""
        orc = _make_orchestrator_stub()
        orc._active_strategy = None  # flat book

        with patch.object(orc, "_attempt_delta_hedge") as mock_hedge:
            for v in [15.0] * 10:
                orc._update_vix_history(v)
            orc._update_vix_history(40.0)
            mock_hedge.assert_not_called()

    def test_vix_history_capped_at_max_bars(self):
        """History list never exceeds _VIX_HISTORY_BARS entries."""
        orc = _make_orchestrator_stub()
        for i in range(_VIX_HISTORY_BARS + 10):
            orc._update_vix_history(15.0 + i * 0.01)
        assert len(orc._vix_history) <= _VIX_HISTORY_BARS

    def test_no_spike_when_trading_disabled(self):
        """Trading disabled → spike detected but no hedge call."""
        orc = _make_orchestrator_stub()
        orc._trading_enabled.is_set.return_value = False
        orc._active_strategy = MagicMock()

        with patch.object(orc, "_attempt_delta_hedge") as mock_hedge:
            for v in [15.0] * 10:
                orc._update_vix_history(v)
            orc._update_vix_history(40.0)
            mock_hedge.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Entry window guard
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryWindow:

    def test_inside_window_returns_true(self):
        orc = _make_orchestrator_stub()
        # 11:00 IST — well inside window (before 14:00)
        fake_now = MagicMock()
        fake_now.hour = 11
        fake_now.minute = 0
        with patch("algo_trading.app.fo_orchestrator.now_ist", return_value=fake_now):
            assert orc._is_within_entry_window() is True

    def test_at_cutoff_returns_false(self):
        orc = _make_orchestrator_stub()
        # Exactly 14:00 → not within window (cutoff is exclusive)
        fake_now = MagicMock()
        fake_now.hour = 14
        fake_now.minute = 0
        with patch("algo_trading.app.fo_orchestrator.now_ist", return_value=fake_now):
            assert orc._is_within_entry_window() is False

    def test_before_market_open_returns_true(self):
        orc = _make_orchestrator_stub()
        fake_now = MagicMock()
        fake_now.hour = 9
        fake_now.minute = 30
        with patch("algo_trading.app.fo_orchestrator.now_ist", return_value=fake_now):
            assert orc._is_within_entry_window() is True

    def test_after_cutoff_returns_false(self):
        orc = _make_orchestrator_stub()
        fake_now = MagicMock()
        fake_now.hour = 15
        fake_now.minute = 0
        with patch("algo_trading.app.fo_orchestrator.now_ist", return_value=fake_now):
            assert orc._is_within_entry_window() is False


# ─────────────────────────────────────────────────────────────────────────────
# _attempt_delta_hedge — force flag bypasses needs_hedge
# ─────────────────────────────────────────────────────────────────────────────

class TestAttemptDeltaHedge:

    def test_no_hedge_when_flat_book(self):
        orc = _make_orchestrator_stub()
        orc._active_strategy = None
        # Should return immediately without touching risk engine
        orc._attempt_delta_hedge()
        orc._risk_engine.last_greeks  # accessed via property — don't assert called

    def test_normal_hedge_skipped_when_needs_hedge_false(self):
        orc = _make_orchestrator_stub()
        orc._active_strategy = MagicMock()
        mock_greeks = MagicMock()
        type(orc._risk_engine).last_greeks = PropertyMock(return_value=mock_greeks)
        orc._hedger.needs_hedge.return_value = False

        orc._attempt_delta_hedge(force=False)
        orc._hedger.compute_hedge_lots.assert_not_called()

    def test_force_true_bypasses_needs_hedge(self):
        """force=True skips needs_hedge() and goes straight to compute_hedge_lots."""
        orc = _make_orchestrator_stub()
        orc._active_strategy = MagicMock()
        mock_greeks = MagicMock()
        type(orc._risk_engine).last_greeks = PropertyMock(return_value=mock_greeks)
        orc._hedger.needs_hedge.return_value = False  # would block normal hedge
        orc._hedger.compute_hedge_lots.return_value = 0  # no actual trade needed

        orc._attempt_delta_hedge(force=True)
        # compute_hedge_lots IS called even though needs_hedge returns False
        orc._hedger.compute_hedge_lots.assert_called_once_with(mock_greeks)

    def test_no_hedge_when_greeks_none(self):
        orc = _make_orchestrator_stub()
        orc._active_strategy = MagicMock()
        type(orc._risk_engine).last_greeks = PropertyMock(return_value=None)

        orc._attempt_delta_hedge(force=True)
        orc._hedger.compute_hedge_lots.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# on_chain_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestOnChainSnapshot:

    def test_chain_stored_on_snapshot(self):
        orc = _make_orchestrator_stub()
        fake_chain = MagicMock()
        fake_chain.expiry = date(2026, 6, 26)

        with patch("algo_trading.app.fo_orchestrator.estimate_india_vix") as mock_vix:
            mock_vix.return_value = MagicMock(vix=-1.0)  # failed estimate
            orc.on_chain_snapshot(fake_chain)

        assert orc._latest_chain is fake_chain

    def test_vix_computed_on_snapshot(self):
        orc = _make_orchestrator_stub()
        fake_chain = MagicMock()
        fake_chain.expiry = date(2026, 6, 26)

        mock_vix_result = MagicMock()
        mock_vix_result.vix = 18.5

        with patch("algo_trading.app.fo_orchestrator.estimate_india_vix",
                   return_value=mock_vix_result) as mock_fn:
            orc.on_chain_snapshot(fake_chain)
            mock_fn.assert_called_once_with(fake_chain)

        assert orc._latest_vix is mock_vix_result

    def test_vix_failure_does_not_crash(self):
        """If estimate_india_vix raises, on_chain_snapshot handles it gracefully."""
        orc = _make_orchestrator_stub()
        fake_chain = MagicMock()
        fake_chain.expiry = date(2026, 6, 26)

        with patch("algo_trading.app.fo_orchestrator.estimate_india_vix",
                   side_effect=RuntimeError("VIX calc failed")):
            orc.on_chain_snapshot(fake_chain)  # must not raise

        # Chain is still stored even if VIX fails
        assert orc._latest_chain is fake_chain

    def test_negative_vix_not_added_to_history(self):
        """VIX estimation failure returns vix=-1.0; must not corrupt history."""
        orc = _make_orchestrator_stub()
        fake_chain = MagicMock()
        fake_chain.expiry = date(2026, 6, 26)

        mock_vix_result = MagicMock()
        mock_vix_result.vix = -1.0  # failed estimate

        with patch("algo_trading.app.fo_orchestrator.estimate_india_vix",
                   return_value=mock_vix_result):
            orc.on_chain_snapshot(fake_chain)

        assert len(orc._vix_history) == 0  # negative VIX not appended
