"""
tests/test_fo_execution.py — Unit tests for algo_trading/fo/execution.py

Covers:
  - MultiLegState.is_terminal property
  - LegExecution properties and expected_qty
  - MultiLegExecution: filled_legs, unfilled_legs, all_legs_filled, _recompute_net_premium
  - MultiLegExecutor: _execution_priority, _compute_limit_price, _build_trading_symbol,
    _net_premium_acceptable
  - FOPositionManager: add_execution, update_execution, daily_pnl, max_concurrent_lots
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from algo_trading.fo.execution import (
    FOPositionManager,
    LegExecution,
    MultiLegExecution,
    MultiLegExecutor,
    MultiLegState,
)
from algo_trading.fo.strategies import StrategyLeg, StrategyResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_leg(
    symbol: str = "NIFTY",
    option_type: str = "CE",
    strike: float = 24500.0,
    expiry: date = date(2026, 6, 19),
    direction: int = -1,
    lots: int = 1,
    lot_size: int = 65,
    entry_premium: float = 150.0,
) -> StrategyLeg:
    return StrategyLeg(
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        direction=direction,
        lots=lots,
        lot_size=lot_size,
        entry_premium=entry_premium,
    )


def make_executor(**kwargs) -> MultiLegExecutor:
    broker = MagicMock()
    bsm = MagicMock()
    return MultiLegExecutor(broker=broker, bsm=bsm, **kwargs)


def make_execution(
    legs: list[LegExecution],
    state: MultiLegState = MultiLegState.LEGGING,
    net_premium_received: float = 0.0,
    expected_net_premium: float = 0.0,
) -> MultiLegExecution:
    return MultiLegExecution(
        execution_id=str(uuid.uuid4()),
        strategy_name="TestStrategy",
        state=state,
        legs=legs,
        started_at=datetime.utcnow(),
        net_premium_received=net_premium_received,
        expected_net_premium=expected_net_premium,
    )


def make_strategy(
    legs: list[StrategyLeg],
    net_premium: float = 1000.0,
) -> StrategyResult:
    return StrategyResult(
        name="TestStrategy",
        legs=legs,
        net_premium=net_premium,
        max_profit=net_premium,
        max_loss=float("inf"),
        break_even_upper=None,
        break_even_lower=None,
        margin_required=0.0,
        current_pnl=0.0,
        portfolio_delta=0.0,
        portfolio_vega=0.0,
        portfolio_theta=0.0,
    )


# ── TestMultiLegState ─────────────────────────────────────────────────────────

class TestMultiLegState:
    def test_terminal_states(self):
        assert MultiLegState.FILLED.is_terminal is True
        assert MultiLegState.ABORTED.is_terminal is True
        assert MultiLegState.PARTIAL_ABORT_FAILED.is_terminal is True

    def test_non_terminal_states(self):
        for state in (
            MultiLegState.PENDING,
            MultiLegState.LEGGING,
            MultiLegState.PARTIALLY_FILLED,
            MultiLegState.STUCK,
            MultiLegState.ABORTING,
        ):
            assert state.is_terminal is False, f"{state} should not be terminal"


# ── TestLegExecution ──────────────────────────────────────────────────────────

class TestLegExecution:
    def test_is_filled_true(self):
        le = LegExecution(leg=make_leg(), state="FILLED")
        assert le.is_filled is True

    def test_is_filled_false(self):
        le = LegExecution(leg=make_leg(), state="PENDING")
        assert le.is_filled is False

    def test_is_pending_or_sent(self):
        assert LegExecution(leg=make_leg(), state="PENDING").is_pending_or_sent is True
        assert LegExecution(leg=make_leg(), state="SENT").is_pending_or_sent is True
        assert LegExecution(leg=make_leg(), state="FILLED").is_pending_or_sent is False
        assert LegExecution(leg=make_leg(), state="CANCELLED").is_pending_or_sent is False

    def test_expected_qty(self):
        le = LegExecution(leg=make_leg(lots=2, lot_size=65))
        assert le.expected_qty() == 130


# ── TestMultiLegExecution ─────────────────────────────────────────────────────

class TestMultiLegExecution:
    def _two_legs(self) -> list[LegExecution]:
        filled = LegExecution(leg=make_leg(strike=24500.0), state="FILLED", filled_price=155.0)
        pending = LegExecution(leg=make_leg(strike=24000.0, option_type="PE"), state="PENDING")
        return [filled, pending]

    def test_filled_legs(self):
        legs = self._two_legs()
        ex = make_execution(legs)
        assert len(ex.filled_legs()) == 1
        assert ex.filled_legs()[0].filled_price == 155.0

    def test_unfilled_legs(self):
        legs = self._two_legs()
        ex = make_execution(legs)
        assert len(ex.unfilled_legs()) == 1
        assert ex.unfilled_legs()[0].state == "PENDING"

    def test_all_legs_filled_true(self):
        legs = [
            LegExecution(leg=make_leg(strike=24500.0), state="FILLED", filled_price=155.0),
            LegExecution(leg=make_leg(strike=24000.0, option_type="PE"), state="FILLED", filled_price=145.0),
        ]
        ex = make_execution(legs)
        assert ex.all_legs_filled() is True

    def test_all_legs_filled_false(self):
        legs = self._two_legs()
        ex = make_execution(legs)
        assert ex.all_legs_filled() is False

    def test_recompute_net_premium_credit_strategy(self):
        # Short CE + Short PE: both direction=-1
        ce_leg = LegExecution(
            leg=make_leg(direction=-1, lots=1, lot_size=65, entry_premium=155.0),
            state="FILLED",
            filled_price=155.0,
        )
        pe_leg = LegExecution(
            leg=make_leg(option_type="PE", direction=-1, lots=1, lot_size=65, entry_premium=145.0),
            state="FILLED",
            filled_price=145.0,
        )
        ex = make_execution([ce_leg, pe_leg])
        ex._recompute_net_premium()
        # net = (-1)*155*65 + (-1)*145*65 = -10075 + -9425 = -19500
        # net_premium_received = -(-19500) = 19500
        assert ex.net_premium_received == pytest.approx(19500.0)

    def test_recompute_net_premium_debit_strategy(self):
        # Long CE: direction=+1
        ce_leg = LegExecution(
            leg=make_leg(direction=+1, lots=1, lot_size=65, entry_premium=155.0),
            state="FILLED",
            filled_price=155.0,
        )
        ex = make_execution([ce_leg])
        ex._recompute_net_premium()
        # net = (+1)*155*65 = 10075; net_premium_received = -10075
        assert ex.net_premium_received == pytest.approx(-10075.0)


# ── TestExecutionPriority ─────────────────────────────────────────────────────

class TestExecutionPriority:
    def test_sells_before_buys(self):
        executor = make_executor()
        sell = make_leg(direction=-1, entry_premium=100.0)
        buy = make_leg(direction=+1, entry_premium=100.0)
        result = executor._execution_priority([buy, sell])
        assert result[0].direction == -1
        assert result[1].direction == +1

    def test_higher_premium_first_within_sells(self):
        executor = make_executor()
        low = make_leg(direction=-1, entry_premium=80.0, strike=25000.0)
        high = make_leg(direction=-1, entry_premium=150.0, strike=24500.0)
        result = executor._execution_priority([low, high])
        assert result[0].entry_premium == 150.0
        assert result[1].entry_premium == 80.0

    def test_buys_sorted_by_premium(self):
        executor = make_executor()
        low = make_leg(direction=+1, entry_premium=50.0, strike=23000.0)
        high = make_leg(direction=+1, entry_premium=120.0, strike=24000.0)
        result = executor._execution_priority([low, high])
        assert result[0].entry_premium == 120.0
        assert result[1].entry_premium == 50.0

    def test_iron_condor_order(self):
        executor = make_executor()
        sell_ce = make_leg(direction=-1, entry_premium=100.0, option_type="CE", strike=24500.0)
        sell_pe = make_leg(direction=-1, entry_premium=90.0, option_type="PE", strike=24000.0)
        buy_ce = make_leg(direction=+1, entry_premium=50.0, option_type="CE", strike=25000.0)
        buy_pe = make_leg(direction=+1, entry_premium=40.0, option_type="PE", strike=23500.0)
        result = executor._execution_priority([buy_ce, sell_ce, buy_pe, sell_pe])
        directions = [leg.direction for leg in result]
        # All sells (-1) must precede all buys (+1)
        assert directions[:2] == [-1, -1]
        assert directions[2:] == [+1, +1]


# ── TestComputeLimitPrice ─────────────────────────────────────────────────────

class TestComputeLimitPrice:
    def test_sell_leg_below_ltp(self):
        executor = make_executor()
        leg = make_leg(direction=-1)
        price = executor._compute_limit_price(leg, current_ltp=100.0)
        # 100 * 0.998 = 99.80; already on 0.05 grid
        assert price == pytest.approx(99.80)

    def test_buy_leg_above_ltp(self):
        executor = make_executor()
        leg = make_leg(direction=+1)
        price = executor._compute_limit_price(leg, current_ltp=100.0)
        # 100 * 1.002 = 100.20; already on 0.05 grid
        assert price == pytest.approx(100.20)

    def test_tick_rounding(self):
        executor = make_executor()
        leg = make_leg(direction=-1)
        price = executor._compute_limit_price(leg, current_ltp=75.0)
        # 75 * 0.998 = 74.85; on 0.05 grid (74.85 / 0.05 = 1497 exactly)
        assert price == pytest.approx(74.85)
        # Must be on 0.05 grid
        assert round(price / 0.05) * 0.05 == pytest.approx(price)

    def test_floor_at_0_05(self):
        executor = make_executor()
        leg = make_leg(direction=-1)
        price = executor._compute_limit_price(leg, current_ltp=0.01)
        assert price >= 0.05


# ── TestBuildTradingSymbol ─────────────────────────────────────────────────────

class TestBuildTradingSymbol:
    def test_ce_symbol(self):
        leg = make_leg(symbol="NIFTY", option_type="CE", strike=24500.0, expiry=date(2026, 6, 19))
        assert MultiLegExecutor._build_trading_symbol(leg) == "NIFTY19JUN2624500CE"

    def test_pe_symbol(self):
        leg = make_leg(symbol="NIFTY", option_type="PE", strike=24000.0, expiry=date(2026, 6, 19))
        assert MultiLegExecutor._build_trading_symbol(leg) == "NIFTY19JUN2624000PE"

    def test_fut_symbol(self):
        leg = make_leg(symbol="NIFTY", option_type="FUT", strike=0.0, expiry=date(2026, 6, 19))
        assert MultiLegExecutor._build_trading_symbol(leg) == "NIFTY19JUN26F"

    def test_banknifty_symbol(self):
        leg = make_leg(
            symbol="BANKNIFTY", option_type="CE", strike=52000.0, expiry=date(2026, 6, 19)
        )
        assert MultiLegExecutor._build_trading_symbol(leg) == "BANKNIFTY19JUN2652000CE"


# ── TestNetPremiumAcceptable ──────────────────────────────────────────────────

class TestNetPremiumAcceptable:
    def _executor(self, min_pct: float = 0.10) -> MultiLegExecutor:
        return make_executor(min_net_premium_pct=min_pct)

    def _exec_with_received(self, received: float) -> MultiLegExecution:
        return make_execution(legs=[], net_premium_received=received)

    def test_credit_strategy_acceptable(self):
        executor = self._executor(min_pct=0.10)
        ex = self._exec_with_received(950.0)
        strategy = make_strategy(legs=[], net_premium=1000.0)
        # floor = 1000 * 0.90 = 900; 950 >= 900 → True
        assert executor._net_premium_acceptable(ex, strategy) is True

    def test_credit_strategy_not_acceptable(self):
        executor = self._executor(min_pct=0.10)
        ex = self._exec_with_received(800.0)
        strategy = make_strategy(legs=[], net_premium=1000.0)
        # floor = 900; 800 < 900 → False
        assert executor._net_premium_acceptable(ex, strategy) is False

    def test_debit_strategy_acceptable(self):
        executor = self._executor(min_pct=0.10)
        ex = self._exec_with_received(-1050.0)
        strategy = make_strategy(legs=[], net_premium=-1000.0)
        # ceiling = -1000 * 1.10 = -1100; -1050 >= -1100 → True
        assert executor._net_premium_acceptable(ex, strategy) is True

    def test_debit_strategy_not_acceptable(self):
        executor = self._executor(min_pct=0.10)
        ex = self._exec_with_received(-1200.0)
        strategy = make_strategy(legs=[], net_premium=-1000.0)
        # ceiling = -1100; -1200 < -1100 → False
        assert executor._net_premium_acceptable(ex, strategy) is False

    def test_zero_premium_always_acceptable(self):
        executor = self._executor()
        ex = self._exec_with_received(0.0)
        strategy = make_strategy(legs=[], net_premium=0.0)
        assert executor._net_premium_acceptable(ex, strategy) is True


# ── TestFOPositionManager ─────────────────────────────────────────────────────

class TestFOPositionManager:
    def _manager(self) -> FOPositionManager:
        executor = make_executor()
        return FOPositionManager(executor=executor)

    def _filled_execution(self, net_premium: float = 5000.0) -> MultiLegExecution:
        leg = LegExecution(
            leg=make_leg(direction=-1, lots=1, lot_size=65, entry_premium=150.0),
            state="FILLED",
            filled_price=150.0,
        )
        ex = make_execution([leg], state=MultiLegState.FILLED, net_premium_received=net_premium)
        return ex

    def _active_execution(self) -> MultiLegExecution:
        leg = LegExecution(
            leg=make_leg(direction=-1, lots=1, lot_size=65, entry_premium=150.0),
            state="FILLED",
            filled_price=150.0,
        )
        return make_execution([leg], state=MultiLegState.PARTIALLY_FILLED, net_premium_received=500.0)

    def test_add_filled_execution_realises_pnl(self):
        mgr = self._manager()
        ex = self._filled_execution(net_premium=5000.0)
        mgr.add_execution(ex)
        assert mgr.daily_pnl() == pytest.approx(5000.0)

    def test_add_active_execution_tracked(self):
        mgr = self._manager()
        ex = self._active_execution()
        mgr.add_execution(ex)
        active = mgr.get_active_executions()
        assert len(active) == 1
        assert active[0].execution_id == ex.execution_id

    def test_update_execution_to_terminal_removes_from_active(self):
        mgr = self._manager()
        ex = self._active_execution()
        mgr.add_execution(ex)
        assert len(mgr.get_active_executions()) == 1

        # Transition to terminal
        ex.state = MultiLegState.FILLED
        ex.net_premium_received = 9750.0
        mgr.update_execution(ex)

        assert len(mgr.get_active_executions()) == 0

    def test_max_concurrent_lots_tracks_peak(self):
        mgr = self._manager()
        ex1 = self._active_execution()
        mgr.add_execution(ex1)
        # Peak should reflect the 1 filled lot from ex1
        assert mgr.max_concurrent_lots() >= 1
