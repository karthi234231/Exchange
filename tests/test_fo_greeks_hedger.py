"""
tests/test_fo_greeks_hedger.py — Tests for fo/greeks_hedger.py.
"""
from __future__ import annotations

import pytest

from algo_trading.fo.greeks_hedger import DeltaHedger, HedgeRecord
from algo_trading.fo.portfolio_risk import PortfolioGreeks, PortfolioRiskEngine
from algo_trading.fo.pricing import BSMEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine() -> PortfolioRiskEngine:
    return PortfolioRiskEngine(BSMEngine(r=0.065, q=0.015))


@pytest.fixture(scope="module")
def hedger(engine: PortfolioRiskEngine) -> DeltaHedger:
    return DeltaHedger(symbol="NIFTY", risk_engine=engine)


def _greeks(delta: float, S: float = 24_000.0) -> PortfolioGreeks:
    ds = S * 0.01
    return PortfolioGreeks(
        delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0,
        delta_risk_inr=delta * S,
        gamma_pnl_1pct=0.0,
        vega_risk_1pct=0.0,
        underlying_price=S,
    )


# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaHedgerInit:

    def test_nifty_lot_size_from_registry(self, hedger):
        assert hedger.futures_lot_size == 65

    def test_invalid_min_lots_raises(self, engine):
        with pytest.raises(ValueError, match="min_lots"):
            DeltaHedger("NIFTY", engine, min_lots=0)

    def test_max_less_than_min_raises(self, engine):
        with pytest.raises(ValueError, match="max_lots"):
            DeltaHedger("NIFTY", engine, min_lots=5, max_lots=3)


class TestNeedsHedge:

    def test_large_delta_triggers_hedge(self, hedger):
        g = _greeks(delta=-200.0)   # -200 delta → 3+ lots needed
        assert hedger.needs_hedge(g) is True

    def test_small_delta_no_hedge(self, hedger):
        g = _greeks(delta=20.0)     # 20/65 ≈ 0.3 lots → below min_lots=1
        assert hedger.needs_hedge(g) is False

    def test_zero_delta_no_hedge(self, hedger):
        assert hedger.needs_hedge(_greeks(0.0)) is False


class TestComputeHedgeLots:

    def test_short_delta_buys_futures(self, hedger):
        # portfolio_delta = -130 → hedge = +2 lots (buy to offset)
        g = _greeks(delta=-130.0)
        lots = hedger.compute_hedge_lots(g)
        assert lots > 0

    def test_long_delta_sells_futures(self, hedger):
        # portfolio_delta = +130 → hedge = -2 lots (sell to offset)
        g = _greeks(delta=130.0)
        lots = hedger.compute_hedge_lots(g)
        assert lots < 0

    def test_capped_at_max_lots(self, engine):
        h = DeltaHedger("NIFTY", engine, min_lots=1, max_lots=3)
        g = _greeks(delta=-10_000.0)   # would be 153+ lots without cap
        assert h.compute_hedge_lots(g) == 3

    def test_min_lots_suppresses_tiny_hedge(self, engine):
        h = DeltaHedger("NIFTY", engine, min_lots=2)
        g = _greeks(delta=-50.0)   # round(-50/65) = -1 lot → below min_lots=2
        assert h.compute_hedge_lots(g) == 0


class TestRecordHedge:

    def test_records_trade_and_updates_position(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        record = h.record_hedge(g, lots_traded=2, futures_price=24_050.0)
        assert isinstance(record, HedgeRecord)
        assert record.direction == 1
        assert record.lots == 2
        assert record.futures_price == 24_050.0
        assert h.current_hedge_lots == 2

    def test_multiple_hedges_accumulate(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        h.record_hedge(g, lots_traded=2, futures_price=24_000.0)
        h.record_hedge(g, lots_traded=1, futures_price=24_010.0)
        assert h.current_hedge_lots == 3
        assert len(h.hedge_history) == 2

    def test_zero_lots_raises(self, engine):
        h = DeltaHedger("NIFTY", engine)
        with pytest.raises(ValueError, match="non-zero"):
            h.record_hedge(_greeks(0.0), lots_traded=0, futures_price=24_000.0)

    def test_delta_before_after_recorded(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        record = h.record_hedge(g, lots_traded=2, futures_price=24_000.0)
        assert record.portfolio_delta_before == pytest.approx(-130.0)
        # After: -130 + 2×65 = 0
        assert record.portfolio_delta_after == pytest.approx(0.0)


class TestUnwindAndReset:

    def test_unwind_returns_opposite_sign(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        h.record_hedge(g, lots_traded=2, futures_price=24_000.0)
        assert h.unwind_hedge() == -2

    def test_unwind_when_flat_returns_zero(self, engine):
        h = DeltaHedger("NIFTY", engine)
        assert h.unwind_hedge() == 0

    def test_reset_clears_state(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        h.record_hedge(g, lots_traded=2, futures_price=24_000.0)
        h.reset()
        assert h.current_hedge_lots == 0
        assert h.hedge_history == []

    def test_hedge_history_is_defensive_copy(self, engine):
        h = DeltaHedger("NIFTY", engine)
        g = _greeks(delta=-130.0)
        h.record_hedge(g, lots_traded=2, futures_price=24_000.0)
        copy = h.hedge_history
        copy.clear()
        assert len(h.hedge_history) == 1   # original not mutated
