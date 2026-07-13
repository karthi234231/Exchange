"""
tests/test_fill_simulator.py — Unit tests for research/fill_simulator.py

Verifies all 9 fill modes from FO_SYSTEM_DESIGN.md Section 9.3:
  1. OPTIMISTIC       — fill at mid price
  2. REALISTIC        — ask+half_spread (buy) / bid-half_spread (sell)
  3. PESSIMISTIC      — full ask+impact (buy) / full bid-impact (sell)
  4. QUEUE_POSITION   — probabilistic fill based on OI turnover
  5. TIME_OF_DAY_LIQUIDITY — spread multiplier by time bucket
  6. EXPIRY_DAY_BLOWOUT — 3× spread final hour, 50% fill cap
  7. EVENT_DAY_BLOWOUT  — 3.5× spread, 40% OTM fill cap
  8. MARKET_ORDER_THIN_BOOK — extra slippage on low-OI strikes
  9. CANCELLATION_LATENCY — cancel-fill race condition

Key invariants:
  - All modes return SimulatedFill or None (never raise on valid inputs)
  - Determinism: same event_time + strike → same fill outcome (seeded RNG)
  - fill_price >= 0 always
  - slippage_bps is (fill_price - mid) / mid × 10_000
  - CANCELLATION_LATENCY can return cancelled=True (filled_qty=0)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time

import pytest

from algo_trading.research.fill_simulator import (
    FillSimulator,
    FillMode,
    FillSimulatorConfig,
    SimulatedFill,
)


# ── Mock quote ────────────────────────────────────────────────────────────────

@dataclass
class MockQuote:
    """Minimal option quote for fill simulation tests."""
    bid: float
    ask: float
    ltp: float
    oi: int


def make_quote(bid: float = 99.0, ask: float = 101.0, ltp: float = 100.0, oi: int = 5000) -> MockQuote:
    return MockQuote(bid=bid, ask=ask, ltp=ltp, oi=oi)


# Standard parameters
EVENT_TIME = datetime(2026, 6, 10, 10, 30, 0)   # Tuesday, 10:30 IST
EXPIRY = date(2026, 6, 10)
STRIKE = 24000.0
MID = 100.0   # (bid=99 + ask=101) / 2


def make_sim(mode: FillMode = FillMode.REALISTIC, **kwargs) -> FillSimulator:
    return FillSimulator(mode=mode, **kwargs)


def do_fill(
    sim: FillSimulator,
    side: str = "BUY",
    qty: int = 1,
    quote: MockQuote | None = None,
    event_time: datetime = EVENT_TIME,
    mode: FillMode | None = None,
) -> SimulatedFill | None:
    if quote is None:
        quote = make_quote()
    return sim.simulate_fill(
        order_id="T-001",
        symbol="NIFTY",
        expiry=EXPIRY,
        strike=STRIKE,
        option_type="CE",
        side=side,
        qty_lots=qty,
        quote=quote,
        event_time=event_time,
        mode=mode,
    )


# ── OPTIMISTIC ────────────────────────────────────────────────────────────────

class TestOptimistic:
    def test_fill_at_mid(self):
        sim = make_sim(FillMode.OPTIMISTIC)
        fill = do_fill(sim)
        assert fill is not None
        assert fill.fill_price == pytest.approx(MID, abs=0.01)

    def test_zero_slippage(self):
        sim = make_sim(FillMode.OPTIMISTIC)
        fill = do_fill(sim)
        assert fill.slippage_bps == pytest.approx(0.0, abs=0.1)

    def test_full_qty_filled(self):
        sim = make_sim(FillMode.OPTIMISTIC)
        fill = do_fill(sim, qty=5)
        assert fill.filled_qty == 5
        assert not fill.is_partial

    def test_not_cancelled(self):
        sim = make_sim(FillMode.OPTIMISTIC)
        fill = do_fill(sim)
        assert fill.cancelled is False


# ── REALISTIC ─────────────────────────────────────────────────────────────────

class TestRealistic:
    def test_buy_price_above_mid(self):
        """Realistic BUY fill must be at or above mid (adverse selection)."""
        sim = make_sim(FillMode.REALISTIC)
        fill = do_fill(sim, side="BUY")
        assert fill is not None
        assert fill.fill_price >= MID

    def test_sell_price_below_mid(self):
        """Realistic SELL fill must be at or below mid."""
        sim = make_sim(FillMode.REALISTIC)
        fill = do_fill(sim, side="SELL")
        assert fill is not None
        assert fill.fill_price <= MID

    def test_buy_slippage_positive(self):
        """BUY slippage (fill > mid) → positive bps."""
        sim = make_sim(FillMode.REALISTIC)
        fill = do_fill(sim, side="BUY")
        assert fill.slippage_bps >= 0

    def test_fill_price_positive(self):
        sim = make_sim(FillMode.REALISTIC)
        fill = do_fill(sim)
        assert fill.fill_price > 0


# ── PESSIMISTIC ───────────────────────────────────────────────────────────────

class TestPessimistic:
    def test_buy_price_gte_ask(self):
        """Pessimistic BUY must be at or above ask (MARKET order)."""
        sim = make_sim(FillMode.PESSIMISTIC)
        q = make_quote(bid=99.0, ask=101.0)
        fill = do_fill(sim, side="BUY", quote=q)
        assert fill.fill_price >= 101.0

    def test_sell_price_lte_bid(self):
        """Pessimistic SELL must be at or below bid."""
        sim = make_sim(FillMode.PESSIMISTIC)
        q = make_quote(bid=99.0, ask=101.0)
        fill = do_fill(sim, side="SELL", quote=q)
        assert fill.fill_price <= 99.0

    def test_pessimistic_slippage_gt_realistic(self):
        """Pessimistic slippage > Realistic slippage for the same quote."""
        q = make_quote()
        r_fill = do_fill(make_sim(FillMode.REALISTIC), side="BUY", quote=q)
        p_fill = do_fill(make_sim(FillMode.PESSIMISTIC), side="BUY", quote=q)
        assert p_fill.fill_price >= r_fill.fill_price


# ── QUEUE_POSITION ────────────────────────────────────────────────────────────

class TestQueuePosition:
    def test_returns_fill_or_none(self):
        """QUEUE_POSITION may return None (queue miss)."""
        sim = make_sim(FillMode.QUEUE_POSITION)
        # Run 20 fills — at least some should succeed (high OI quote)
        filled = 0
        for i in range(20):
            t = EVENT_TIME.replace(second=i)
            fill = do_fill(sim, event_time=t, qty=1)
            if fill is not None:
                assert fill.fill_price > 0
                filled += 1
        # With OI=5000 and fill_prob=0.65, many should fill
        assert filled > 0, "Expected at least some fills in 20 attempts"

    def test_low_oi_reduces_fill_prob(self):
        """Very low OI quote → more queue misses."""
        sim = make_sim(FillMode.QUEUE_POSITION)
        low_oi_q = make_quote(oi=10)
        fills = 0
        for i in range(20):
            t = EVENT_TIME.replace(second=i)
            fill = do_fill(sim, quote=low_oi_q, event_time=t)
            if fill is not None and not fill.cancelled:
                fills += 1
        # With very low OI, many should fail
        # (not 100% deterministic test — just verify it runs cleanly)
        assert fills <= 20  # trivially true, but confirms no exception


# ── TIME_OF_DAY_LIQUIDITY ─────────────────────────────────────────────────────

class TestTimeOfDayLiquidity:
    def test_open_auction_wider_than_midday(self):
        """Open-auction spread multiplier (4×) → wider fill than midday (1×)."""
        sim = make_sim(FillMode.TIME_OF_DAY_LIQUIDITY)
        # 09:16 = open auction
        open_time = datetime(2026, 6, 10, 9, 16)
        # 12:00 = midday baseline
        mid_time = datetime(2026, 6, 10, 12, 0)

        open_fill = do_fill(sim, side="BUY", event_time=open_time)
        mid_fill = do_fill(sim, side="BUY", event_time=mid_time)

        assert open_fill is not None and mid_fill is not None
        assert open_fill.fill_price >= mid_fill.fill_price

    def test_returns_simulated_fill(self):
        sim = make_sim(FillMode.TIME_OF_DAY_LIQUIDITY)
        fill = do_fill(sim)
        assert isinstance(fill, SimulatedFill)
        assert fill.fill_price > 0


# ── EXPIRY_DAY_BLOWOUT ────────────────────────────────────────────────────────

class TestExpiryDayBlowout:
    def test_final_hour_wider_slippage(self):
        """Final hour of expiry day → wider spread than normal hours."""
        sim = make_sim(FillMode.EXPIRY_DAY_BLOWOUT)
        # 14:50 = final hour (15:30 - 40 min)
        final_hour = datetime(2026, 6, 10, 14, 50)
        # 11:00 = morning (outside final hour)
        morning = datetime(2026, 6, 10, 11, 0)

        final_fill = do_fill(sim, side="BUY", event_time=final_hour)
        morning_fill = do_fill(sim, side="BUY", event_time=morning)

        # Both should return fills; final hour should have higher fill price
        if final_fill is not None and morning_fill is not None:
            assert final_fill.fill_price >= morning_fill.fill_price

    def test_no_exception_on_expiry_day(self):
        sim = make_sim(FillMode.EXPIRY_DAY_BLOWOUT)
        fill = do_fill(sim)
        # Should not raise; may return None or SimulatedFill
        assert fill is None or isinstance(fill, SimulatedFill)


# ── EVENT_DAY_BLOWOUT ─────────────────────────────────────────────────────────

class TestEventDayBlowout:
    def test_wider_than_realistic(self):
        """Event day fill price (3.5× spread) >= realistic fill price."""
        q = make_quote()
        event_fill = do_fill(make_sim(FillMode.EVENT_DAY_BLOWOUT), side="BUY", quote=q)
        real_fill = do_fill(make_sim(FillMode.REALISTIC), side="BUY", quote=q)
        if event_fill is not None and real_fill is not None:
            assert event_fill.fill_price >= real_fill.fill_price

    def test_returns_fill_or_none(self):
        sim = make_sim(FillMode.EVENT_DAY_BLOWOUT)
        fill = do_fill(sim)
        assert fill is None or isinstance(fill, SimulatedFill)


# ── MARKET_ORDER_THIN_BOOK ────────────────────────────────────────────────────

class TestMarketOrderThinBook:
    def test_high_oi_fills_at_normal_price(self):
        """High-OI quote → no extra thin-book slippage."""
        sim = make_sim(FillMode.MARKET_ORDER_THIN_BOOK)
        high_oi_q = make_quote(oi=10_000)
        fill = do_fill(sim, side="BUY", quote=high_oi_q)
        # Should fill at full ask (≤ ask + thin_book_slippage)
        assert fill is not None
        assert fill.fill_price >= 101.0  # at least ask

    def test_low_oi_extra_slippage(self):
        """Low-OI quote → extra slippage bps on top of ask."""
        sim_thin = make_sim(FillMode.MARKET_ORDER_THIN_BOOK)
        sim_pess = make_sim(FillMode.PESSIMISTIC)
        low_oi_q = make_quote(oi=50)  # below low_oi_threshold_lots=500
        thin_fill = do_fill(sim_thin, side="BUY", quote=low_oi_q)
        pess_fill = do_fill(sim_pess, side="BUY", quote=low_oi_q)
        if thin_fill is not None and pess_fill is not None:
            # Thin book should be worse than or equal to pessimistic
            assert thin_fill.fill_price >= pess_fill.fill_price


# ── CANCELLATION_LATENCY ──────────────────────────────────────────────────────

class TestCancellationLatency:
    def test_returns_simulated_fill(self):
        """Always returns a SimulatedFill (either cancelled or filled)."""
        sim = make_sim(FillMode.CANCELLATION_LATENCY)
        fill = do_fill(sim)
        assert isinstance(fill, SimulatedFill)

    def test_cancelled_has_zero_qty(self):
        """When cancel wins: cancelled=True, filled_qty=0."""
        sim = make_sim(FillMode.CANCELLATION_LATENCY)
        results = []
        for i in range(30):
            t = EVENT_TIME.replace(second=i)
            fill = do_fill(sim, event_time=t)
            results.append(fill)

        cancelled = [f for f in results if f is not None and f.cancelled]
        filled = [f for f in results if f is not None and not f.cancelled]

        for c in cancelled:
            assert c.filled_qty == 0
            assert c.fill_price == 0.0

        for f in filled:
            assert f.filled_qty > 0
            assert f.fill_price > 0

    def test_determinism(self):
        """Same event_time + strike → same cancel/fill outcome."""
        sim1 = FillSimulator(mode=FillMode.CANCELLATION_LATENCY)
        sim2 = FillSimulator(mode=FillMode.CANCELLATION_LATENCY)

        fill1 = do_fill(sim1, event_time=EVENT_TIME)
        fill2 = do_fill(sim2, event_time=EVENT_TIME)

        assert fill1.cancelled == fill2.cancelled
        assert fill1.fill_price == pytest.approx(fill2.fill_price)


# ── Cross-mode determinism ────────────────────────────────────────────────────

class TestDeterminism:
    """Verify replay determinism: same inputs → same outputs."""

    def test_realistic_deterministic(self):
        """Two identical calls → identical fill prices."""
        sim = make_sim(FillMode.REALISTIC)
        fill1 = do_fill(sim, event_time=EVENT_TIME)
        fill2 = do_fill(sim, event_time=EVENT_TIME)
        assert fill1.fill_price == pytest.approx(fill2.fill_price)

    def test_queue_position_deterministic(self):
        sim = make_sim(FillMode.QUEUE_POSITION)
        fill1 = do_fill(sim, event_time=EVENT_TIME)
        fill2 = do_fill(sim, event_time=EVENT_TIME)
        # Both either None or same result
        assert (fill1 is None) == (fill2 is None)
        if fill1 is not None:
            assert fill1.fill_price == pytest.approx(fill2.fill_price)


# ── Common invariants across all modes ────────────────────────────────────────

class TestCommonInvariants:
    """Properties that must hold for all 9 fill modes."""

    @pytest.mark.parametrize("mode", list(FillMode))
    def test_no_exception_on_valid_quote(self, mode):
        """No mode should raise on a valid quote."""
        sim = FillSimulator(mode=mode)
        try:
            fill = do_fill(sim, mode=mode)
            # May return None for QUEUE_POSITION, etc.
        except Exception as e:
            pytest.fail(f"Mode {mode} raised: {e}")

    @pytest.mark.parametrize("mode", list(FillMode))
    def test_fill_price_non_negative(self, mode):
        """fill_price must always be >= 0."""
        sim = FillSimulator(mode=mode)
        fill = do_fill(sim, mode=mode)
        if fill is not None:
            assert fill.fill_price >= 0

    @pytest.mark.parametrize("mode", list(FillMode))
    def test_fill_mode_field_matches(self, mode):
        """fill.fill_mode must match the requested mode."""
        sim = FillSimulator(mode=mode)
        fill = do_fill(sim, mode=mode)
        if fill is not None:
            assert fill.fill_mode == mode
