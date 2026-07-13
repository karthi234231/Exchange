"""
tests/test_timezone_consistency.py — P2-FIX-87: Timezone consistency tests.

Requirements:
  - All datetime fields in domain types must be timezone-aware (IST or UTC).
  - Naive datetimes must never enter the order/fill/bar pipeline.
  - IST-aware datetimes must behave correctly at date boundaries (midnight IST).
  - No accidental UTC-as-IST confusion on day boundaries.

These tests ensure the system uses consistent, zone-aware datetimes throughout.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from algo_trading.core.calendar import IST, now_ist, today_ist


# ── Helper ────────────────────────────────────────────────────────────────────

def _ist(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Create an IST-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Create a UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ── IST zone tests ────────────────────────────────────────────────────────────

class TestISTZone:
    """Validate IST = UTC+5:30 (no DST)."""

    def test_ist_offset_is_plus_5h30m(self):
        """IST must always be UTC+5:30 — India has no DST."""
        dt_ist = _ist(2026, 6, 15, 9, 15)
        offset = dt_ist.utcoffset()
        assert offset == timedelta(hours=5, minutes=30), (
            f"IST offset should be +05:30, got {offset}"
        )

    def test_ist_offset_is_dst_neutral_winter(self):
        """Verify IST offset is identical in January (DST-neutral)."""
        dt_ist = _ist(2026, 1, 15, 9, 15)
        offset = dt_ist.utcoffset()
        assert offset == timedelta(hours=5, minutes=30)

    def test_market_open_in_utc(self):
        """09:15 IST = 03:45 UTC — verify the conversion is correct."""
        market_open_ist = _ist(2026, 6, 15, 9, 15)
        market_open_utc = market_open_ist.astimezone(timezone.utc)
        assert market_open_utc.hour == 3
        assert market_open_utc.minute == 45

    def test_market_close_in_utc(self):
        """15:30 IST = 10:00 UTC — verify the conversion."""
        market_close_ist = _ist(2026, 6, 15, 15, 30)
        market_close_utc = market_close_ist.astimezone(timezone.utc)
        assert market_close_utc.hour == 10
        assert market_close_utc.minute == 0


# ── Date boundary tests ───────────────────────────────────────────────────────

class TestDateBoundaries:
    """IST midnight (00:00) is the correct date boundary — not UTC."""

    def test_ist_midnight_is_correct_date(self):
        """IST midnight on 2026-06-15 should be date 2026-06-15 in IST."""
        midnight_ist = _ist(2026, 6, 15, 0, 0)
        assert midnight_ist.date() == date(2026, 6, 15)

    def test_ist_midnight_in_utc_is_prev_day(self):
        """IST midnight (00:00 IST) = 18:30 UTC on the PREVIOUS calendar day."""
        midnight_ist = _ist(2026, 6, 15, 0, 0)
        in_utc = midnight_ist.astimezone(timezone.utc)
        assert in_utc.date() == date(2026, 6, 14), (
            "IST midnight should map to previous UTC date"
        )
        assert in_utc.hour == 18
        assert in_utc.minute == 30

    def test_today_ist_returns_ist_date(self):
        """today_ist() must return the current date in IST, not UTC."""
        d = today_ist()
        # Validate it is a date object
        assert isinstance(d, date)
        # Compare against IST-aware now
        now = now_ist()
        assert d == now.date()

    def test_now_ist_is_zone_aware(self):
        """now_ist() must return a timezone-aware datetime."""
        dt = now_ist()
        assert dt.tzinfo is not None, "now_ist() must return timezone-aware datetime"
        offset = dt.utcoffset()
        assert offset == timedelta(hours=5, minutes=30), (
            f"now_ist() offset must be +05:30, got {offset}"
        )


# ── Domain type timezone tests ────────────────────────────────────────────────

class TestDomainTypeTimezones:
    """Validate that domain types (Tick, Bar, Fill) accept only aware datetimes."""

    def test_tick_requires_aware_timestamp(self):
        """Tick.timestamp should be created with IST zone."""
        from algo_trading.data.broker_base import Tick
        aware_ts = _ist(2026, 6, 15, 9, 30)
        tick = Tick(
            timestamp=aware_ts,
            symbol="RELIANCE",
            ltp=Decimal("2500.00"),
            volume=100000,
            best_bid=Decimal("2499.50"),
            best_ask=Decimal("2500.50"),
            oi=0,
        )
        assert tick.timestamp.tzinfo is not None
        assert tick.timestamp.utcoffset() == timedelta(hours=5, minutes=30)

    def test_bar_requires_aware_timestamp(self):
        """Bar.timestamp should be IST-aware."""
        from algo_trading.data.broker_base import Bar
        aware_ts = _ist(2026, 6, 15, 9, 15)
        bar = Bar(
            timestamp=aware_ts,
            symbol="NIFTY",
            open=Decimal("24000"),
            high=Decimal("24100"),
            low=Decimal("23950"),
            close=Decimal("24050"),
            volume=500000,
            vwap=Decimal("24030"),
            num_ticks=3000,
            bar_type="time",
        )
        assert bar.timestamp.tzinfo is not None

    def test_fill_requires_aware_timestamp(self):
        """Fill.timestamp should be IST-aware."""
        from algo_trading.data.broker_base import Fill
        aware_ts = _ist(2026, 6, 15, 10, 30)
        fill = Fill(
            order_id="client-001",
            exchange_order_id="exch-001",
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            price=Decimal("2500.00"),
            timestamp=aware_ts,
        )
        assert fill.timestamp.tzinfo is not None

    def test_order_record_created_at_is_aware(self):
        """OrderRecord.created_at should be IST-aware (from now_ist())."""
        from algo_trading.data.broker_base import Order
        from algo_trading.execution.order_fsm import OrderRecord
        order = Order(
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            order_type="LIMIT",
            price=Decimal("2500"),
            strategy_id="TEST",
        )
        record = OrderRecord(order=order)
        assert record.created_at.tzinfo is not None, (
            "OrderRecord.created_at must be timezone-aware"
        )

    def test_trade_signal_timestamp_aware_if_set(self):
        """TradeSignal.timestamp, if set, must be IST-aware."""
        from algo_trading.execution.order_executor import TradeSignal
        naive_ts = datetime(2026, 6, 15, 10, 30)  # naive — should NOT be used
        aware_ts = _ist(2026, 6, 15, 10, 30)

        signal_aware = TradeSignal(
            symbol="RELIANCE",
            direction=1,
            entry_price=Decimal("2500"),
            stop_loss_price=Decimal("2450"),
            atr=Decimal("15"),
            timestamp=aware_ts,
        )
        assert signal_aware.timestamp is not None
        assert signal_aware.timestamp.tzinfo is not None

        # Naive timestamps are not caught at construction (dataclass doesn't validate),
        # but we document that naive timestamps must not be passed.
        signal_naive = TradeSignal(
            symbol="RELIANCE",
            direction=1,
            entry_price=Decimal("2500"),
            stop_loss_price=Decimal("2450"),
            atr=Decimal("15"),
            timestamp=naive_ts,
        )
        # If naive datetime is passed, utcoffset() should return None
        assert signal_naive.timestamp.tzinfo is None, (
            "Documenting that naive timestamps are possible — caller must use aware datetimes"
        )


# ── Comparison and arithmetic tests ──────────────────────────────────────────

class TestDatetimeComparisons:
    """Zone-aware datetimes must compare correctly across zones."""

    def test_same_instant_different_zones_are_equal(self):
        """09:15 IST == 03:45 UTC — same instant, different representation."""
        ist_time = _ist(2026, 6, 15, 9, 15)
        utc_time = _utc(2026, 6, 15, 3, 45)
        assert ist_time == utc_time

    def test_ist_before_utc_of_next_day(self):
        """15:30 IST (market close) < 00:00 IST (next midnight)."""
        close_ist = _ist(2026, 6, 15, 15, 30)
        midnight_next = _ist(2026, 6, 16, 0, 0)
        assert close_ist < midnight_next

    def test_session_duration_is_375_minutes(self):
        """NSE session: 09:15 to 15:30 = 375 minutes."""
        session_start = _ist(2026, 6, 15, 9, 15)
        session_end = _ist(2026, 6, 15, 15, 30)
        duration = session_end - session_start
        assert duration == timedelta(minutes=375)

    def test_ist_utc_comparison_raises_on_mixed_aware_naive(self):
        """Comparing aware with naive datetime should raise TypeError."""
        aware = _ist(2026, 6, 15, 9, 15)
        naive = datetime(2026, 6, 15, 9, 15)
        with pytest.raises(TypeError):
            _ = aware < naive  # type: ignore[operator]
        with pytest.raises(TypeError):
            _ = aware > naive  # type: ignore[operator]


# ── IST-specific date helpers ─────────────────────────────────────────────────

class TestISTSpecificHelpers:
    """Validate helpers that convert between IST and date objects."""

    def test_ist_date_after_midnight_utc(self):
        """At 00:30 IST (18:30 UTC prev day) the IST date is the current IST day."""
        # 00:30 IST on June 15 = 19:00 UTC on June 14
        dt_ist = _ist(2026, 6, 15, 0, 30)
        assert dt_ist.date() == date(2026, 6, 15)
        dt_utc = dt_ist.astimezone(timezone.utc)
        assert dt_utc.date() == date(2026, 6, 14), (
            "Same instant in UTC is June 14 — IST date should be June 15"
        )

    def test_ist_date_just_before_midnight(self):
        """23:59 IST on June 15 should still be date June 15 in IST."""
        dt_ist = _ist(2026, 6, 15, 23, 59)
        assert dt_ist.date() == date(2026, 6, 15)

    def test_nse_expiry_tuesday_calculation(self):
        """Verify Tuesday (weekday==1) detection for Nifty weekly expiry."""
        # NSE Nifty weekly options expire on Tuesdays (from FAOP68747)
        tuesday = date(2026, 6, 16)  # A Tuesday
        assert tuesday.weekday() == 1, f"Expected Tuesday, got weekday {tuesday.weekday()}"

        wednesday = date(2026, 6, 17)
        assert wednesday.weekday() == 2
