"""
tests/test_market_calendar.py — NSE market calendar tests.

Validates all calendar and session time functions.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from algo_trading.core.calendar import (
    IST,
    SESSION_END,
    SESSION_START,
    bar_start_time,
    can_enter_new_trade,
    is_market_open,
    is_pre_open,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    seconds_to_force_close,
)


def ist(hour: int, minute: int = 0, second: int = 0, date_override: date | None = None) -> datetime:
    """Helper: create a datetime in IST."""
    d = date_override or date(2026, 6, 8)  # Monday — a valid trading day
    return datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=IST)


class TestIsTradingDay:
    def test_weekday_not_holiday_is_trading(self) -> None:
        # Monday 2026-06-08 — should be trading day
        assert is_trading_day(date(2026, 6, 8))

    def test_saturday_is_not_trading(self) -> None:
        assert not is_trading_day(date(2026, 6, 6))  # Saturday

    def test_sunday_is_not_trading(self) -> None:
        assert not is_trading_day(date(2026, 6, 7))  # Sunday

    def test_republic_day_not_trading(self) -> None:
        assert not is_trading_day(date(2026, 1, 26))

    def test_independence_day_not_trading(self) -> None:
        assert not is_trading_day(date(2026, 8, 15))

    def test_christmas_not_trading(self) -> None:
        assert not is_trading_day(date(2026, 12, 25))


class TestNextPreviousTradingDay:
    def test_next_from_friday(self) -> None:
        friday = date(2026, 6, 5)
        next_day = next_trading_day(friday)
        assert next_day.weekday() not in (5, 6)  # Not weekend
        assert next_day > friday

    def test_next_skips_weekend(self) -> None:
        friday = date(2026, 6, 5)
        next_day = next_trading_day(friday)
        assert next_day == date(2026, 6, 8)  # Monday

    def test_previous_from_monday(self) -> None:
        monday = date(2026, 6, 8)
        prev = previous_trading_day(monday)
        assert prev == date(2026, 6, 5)  # Friday

    def test_previous_skips_holiday(self) -> None:
        # Day after Republic Day (2026-01-27 is a Tuesday — previous should skip Jan 26)
        jan_27 = date(2026, 1, 27)
        prev = previous_trading_day(jan_27)
        assert prev < date(2026, 1, 26)  # Must be before the holiday


class TestIsMarketOpen:
    def test_open_during_session(self) -> None:
        assert is_market_open(ist(10, 30))

    def test_closed_before_open(self) -> None:
        assert not is_market_open(ist(9, 0))

    def test_closed_after_close(self) -> None:
        assert not is_market_open(ist(15, 30))

    def test_open_at_session_start(self) -> None:
        assert is_market_open(ist(9, 15))

    def test_closed_at_session_end(self) -> None:
        # 15:30 is exclusive (not open)
        assert not is_market_open(ist(15, 30))

    def test_closed_on_weekend(self) -> None:
        saturday = datetime(2026, 6, 6, 10, 30, tzinfo=IST)  # Saturday
        assert not is_market_open(saturday)

    def test_closed_on_holiday(self) -> None:
        republic_day = datetime(2026, 1, 26, 11, 0, tzinfo=IST)
        assert not is_market_open(republic_day)


class TestIsPreOpen:
    def test_pre_open_session(self) -> None:
        assert is_pre_open(ist(9, 3))

    def test_not_pre_open_after_9_08(self) -> None:
        assert not is_pre_open(ist(9, 9))

    def test_not_pre_open_before_9_00(self) -> None:
        assert not is_pre_open(ist(8, 59))


class TestCanEnterNewTrade:
    def test_cannot_during_opening_bar(self) -> None:
        """9:15-9:30 is the opening bar — no new entries [SURV-8]."""
        assert not can_enter_new_trade(ist(9, 15))
        assert not can_enter_new_trade(ist(9, 20))
        assert not can_enter_new_trade(ist(9, 29, 59))

    def test_can_trade_after_9_30(self) -> None:
        assert can_enter_new_trade(ist(9, 30))
        assert can_enter_new_trade(ist(12, 0))

    def test_cannot_after_no_new_trades_time(self) -> None:
        """After 15:15 — no new entries [SURV-8]."""
        assert not can_enter_new_trade(ist(15, 15))
        assert not can_enter_new_trade(ist(15, 20))

    def test_cannot_outside_market_hours(self) -> None:
        assert not can_enter_new_trade(ist(8, 0))
        assert not can_enter_new_trade(ist(17, 0))

    def test_cannot_on_weekend(self) -> None:
        saturday = datetime(2026, 6, 6, 11, 0, tzinfo=IST)
        assert not can_enter_new_trade(saturday)


class TestBarStartTime:
    def test_bar_at_session_start(self) -> None:
        """09:17 → bar starts at 09:15."""
        dt = ist(9, 17)
        bar = bar_start_time(dt, 15)
        assert bar.hour == 9 and bar.minute == 15

    def test_bar_mid_session(self) -> None:
        """11:32 → bar starts at 11:30."""
        dt = ist(11, 32)
        bar = bar_start_time(dt, 15)
        assert bar.hour == 11 and bar.minute == 30

    def test_bar_at_slot_boundary(self) -> None:
        """09:30:00 exactly → new bar starts at 09:30."""
        dt = ist(9, 30, 0)
        bar = bar_start_time(dt, 15)
        assert bar.hour == 9 and bar.minute == 30

    def test_bar_just_before_boundary(self) -> None:
        """09:29:59 → still in 09:15 bar."""
        dt = ist(9, 29, 59)
        bar = bar_start_time(dt, 15)
        assert bar.hour == 9 and bar.minute == 15

    def test_bar_seconds_reset_to_zero(self) -> None:
        dt = ist(10, 22, 47)
        bar = bar_start_time(dt, 15)
        assert bar.second == 0


class TestSecondsToForceClose:
    def test_positive_before_force_close(self) -> None:
        dt = ist(11, 0)
        secs = seconds_to_force_close(dt)
        assert secs > 0
        # 15:15 - 11:00 = 4h15m = 15300 seconds
        assert abs(secs - 15300) < 60  # Allow 1 minute tolerance

    def test_zero_at_force_close(self) -> None:
        dt = ist(15, 15)
        secs = seconds_to_force_close(dt)
        assert secs == 0.0

    def test_zero_after_force_close(self) -> None:
        dt = ist(15, 20)
        secs = seconds_to_force_close(dt)
        assert secs == 0.0

    def test_returns_negative_on_non_trading_day(self) -> None:
        saturday = datetime(2026, 6, 6, 11, 0, tzinfo=IST)
        secs = seconds_to_force_close(saturday)
        assert secs == -1.0
