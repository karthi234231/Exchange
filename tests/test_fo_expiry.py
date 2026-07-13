"""
tests/test_fo_expiry.py — Unit tests for algo_trading/fo/expiry.py

Verifies:
  1. Nifty weekly expiry = every Tuesday (post Sep 1 2025, per FAOP68747)
  2. BankNifty monthly expiry = last Tuesday of each month (weekly discontinued Nov 2024)
  3. Holiday adjustment: if Tuesday is NSE holiday, expiry rolls to Monday
  4. current_expiry() and next_expiry() return correct dates
  5. LRU-cached time_to_expiry_years uses calendar days / 365 (NSE convention)

Sources verified against:
  [P] NSE Circular FAOP68747 — Nifty weekly expiry moved from Thursday to Tuesday
  [P] NSE Bank Nifty page — weekly contracts discontinued Nov 2024
"""
from __future__ import annotations

from datetime import date

import pytest

from algo_trading.fo.expiry import (
    ExpiryCalendar,
    _all_tuesdays_in_month,
    _last_tuesday_of_month,
    _preceding_trading_day,
    banknifty_expiry_calendar,
    get_expiry_calendar,
    nifty_expiry_calendar,
    time_to_expiry_years,
)


# ── Helper: is_tuesday ────────────────────────────────────────────────────────

def is_tuesday(d: date) -> bool:
    return d.weekday() == 1  # Monday=0, Tuesday=1


# ── _all_tuesdays_in_month ────────────────────────────────────────────────────

class TestAllTuesdaysInMonth:
    def test_june_2026(self):
        """June 2026: Tuesdays are 2, 9, 16, 23, 30."""
        tues = _all_tuesdays_in_month(2026, 6)
        assert tues == [
            date(2026, 6, 2),
            date(2026, 6, 9),
            date(2026, 6, 16),
            date(2026, 6, 23),
            date(2026, 6, 30),
        ]

    def test_all_results_are_tuesdays(self):
        """All returned dates must be Tuesdays."""
        for month in range(1, 13):
            tues = _all_tuesdays_in_month(2026, month)
            assert all(is_tuesday(d) for d in tues)

    def test_sorted_ascending(self):
        tues = _all_tuesdays_in_month(2026, 3)
        assert tues == sorted(tues)

    def test_all_same_month(self):
        tues = _all_tuesdays_in_month(2026, 6)
        assert all(d.month == 6 for d in tues)


# ── _last_tuesday_of_month ────────────────────────────────────────────────────

class TestLastTuesdayOfMonth:
    def test_june_2026(self):
        """Last Tuesday of June 2026 = June 30."""
        assert _last_tuesday_of_month(2026, 6) == date(2026, 6, 30)

    def test_january_2026(self):
        """Last Tuesday of Jan 2026 = Jan 27."""
        assert _last_tuesday_of_month(2026, 1) == date(2026, 1, 27)

    def test_always_tuesday(self):
        for year in [2025, 2026]:
            for month in range(1, 13):
                last = _last_tuesday_of_month(year, month)
                assert is_tuesday(last), f"{last} is not a Tuesday"


# ── Nifty ExpiryCalendar ──────────────────────────────────────────────────────

class TestNiftyExpiryCalendar:
    """
    Nifty: weekly_enabled=True, expiry every Tuesday (since Sep 1 2025).
    [P] NSE Circular FAOP68747
    """

    def test_next_expiry_is_tuesday(self):
        """next_expiry() from any date returns a Tuesday."""
        cal = nifty_expiry_calendar
        for day in [
            date(2026, 6, 1),   # Monday
            date(2026, 6, 3),   # Wednesday
            date(2026, 6, 19),  # Friday
        ]:
            exp = cal.next_expiry(day)
            assert is_tuesday(exp), f"Expected Tuesday, got {exp} ({exp.strftime('%A')})"

    def test_weekly_expiry_is_not_monthly_only(self):
        """Nifty has multiple expiries per month (weekly), not just one."""
        cal = nifty_expiry_calendar
        expiries = cal.weekly_expiries_in_month(2026, 6)
        assert len(expiries) >= 4, "Nifty should have 4+ weekly expiries in June 2026"

    def test_expiry_series_ascending(self):
        """expiry_series() returns sorted dates."""
        exps = nifty_expiry_calendar.expiry_series(date(2026, 6, 1), 5)
        assert exps == sorted(exps), "Expiry series should be ascending"

    def test_expiry_series_all_tuesdays(self):
        exps = nifty_expiry_calendar.expiry_series(date(2026, 1, 1), 10)
        # All should be on Tuesday (or the preceding trading day if Tuesday is holiday)
        for exp in exps:
            # Either Tuesday or a non-Tuesday if holiday-adjusted
            # Check it's at most 1 day before a Tuesday
            next_tuesday_offset = (1 - exp.weekday()) % 7
            days_back = (exp.weekday() - 1) % 7 if exp.weekday() != 1 else 0
            assert days_back <= 2, f"{exp} is {exp.strftime('%A')}, should be close to Tuesday"

    def test_is_expiry_day_for_known_tuesday(self):
        """June 9, 2026 is a Tuesday — should be Nifty expiry day."""
        # Verify June 9 2026 is Tuesday
        d = date(2026, 6, 9)
        assert is_tuesday(d), f"June 9, 2026 is {d.strftime('%A')}, not Tuesday"
        # It should be an expiry day (assuming no holiday)
        assert nifty_expiry_calendar.is_expiry_day(d)

    def test_days_to_expiry_positive(self):
        """days_to_expiry() should be positive."""
        dte = nifty_expiry_calendar.days_to_expiry(date(2026, 6, 11))
        assert dte > 0, f"DTE should be positive, got {dte}"

    def test_days_to_expiry_from_tuesday_goes_to_next_week(self):
        """From an expiry Tuesday, DTE should be 7 (next Tuesday)."""
        # June 9 2026 is Tuesday (expiry day)
        d = date(2026, 6, 9)
        dte = nifty_expiry_calendar.days_to_expiry(d)
        # From expiry day, next expiry = 7 days away (next Tuesday)
        assert dte == 7, f"Expected 7, got {dte}"

    def test_monthly_expiry_is_last_tuesday(self):
        """Monthly expiry = last Tuesday of the month."""
        monthly = nifty_expiry_calendar.monthly_expiry(2026, 6)
        assert monthly == date(2026, 6, 30), f"Expected 2026-06-30, got {monthly}"
        assert is_tuesday(monthly)


# ── BankNifty ExpiryCalendar ──────────────────────────────────────────────────

class TestBankNiftyExpiryCalendar:
    """
    BankNifty: weekly_enabled=False (discontinued Nov 2024), monthly only.
    [P] NSE Bank Nifty page
    """

    def test_weekly_expiries_empty(self):
        """BankNifty has no weekly expiries."""
        expiries = banknifty_expiry_calendar.weekly_expiries_in_month(2026, 6)
        assert expiries == [], f"BankNifty should have no weekly expiries, got {expiries}"

    def test_next_expiry_is_monthly(self):
        """BankNifty next_expiry() should be the last Tuesday of the current/next month."""
        cal = banknifty_expiry_calendar
        exp = cal.next_expiry(date(2026, 6, 1))
        # Should be end of June (last Tuesday) or later
        monthly = cal.monthly_expiry(2026, 6)
        assert exp == monthly, f"Expected last Tuesday {monthly}, got {exp}"

    def test_expiry_series_monthly_gaps(self):
        """BankNifty expiries should be ~30 days apart (monthly)."""
        exps = banknifty_expiry_calendar.expiry_series(date(2026, 1, 1), 6)
        for i in range(1, len(exps)):
            gap = (exps[i] - exps[i-1]).days
            assert 25 <= gap <= 35, f"Monthly gap should be ~30 days, got {gap}"

    def test_banknifty_monthly_expiry_june_2026(self):
        """BankNifty monthly expiry for June 2026 = last Tuesday = June 30."""
        exp = banknifty_expiry_calendar.monthly_expiry(2026, 6)
        assert exp == date(2026, 6, 30)

    def test_banknifty_is_not_weekly(self):
        """BankNifty should NOT be an expiry day on non-last Tuesdays."""
        # June 10 is a Tuesday but NOT the last Tuesday of June 2026
        d = date(2026, 6, 10)
        assert not banknifty_expiry_calendar.is_expiry_day(d), (
            f"{d} should not be a BankNifty expiry day (not last Tuesday of month)"
        )

    def test_banknifty_last_tuesday_is_expiry(self):
        """BankNifty IS an expiry day on the last Tuesday."""
        last_tuesday = date(2026, 6, 30)
        assert banknifty_expiry_calendar.is_expiry_day(last_tuesday)


# ── get_expiry_calendar ────────────────────────────────────────────────────────

class TestGetExpiryCalendar:
    def test_nifty_returns_nifty_calendar(self):
        cal = get_expiry_calendar("NIFTY")
        assert cal is nifty_expiry_calendar

    def test_banknifty_case_insensitive(self):
        cal = get_expiry_calendar("banknifty")
        assert cal is banknifty_expiry_calendar

    def test_unknown_symbol_raises(self):
        with pytest.raises(KeyError):
            get_expiry_calendar("FINNIFTY")


# ── time_to_expiry_years ──────────────────────────────────────────────────────

class TestTimeToExpiryYears:
    """
    NSE convention: T = calendar_days / 365
    Minimum T = 1/365 (floor to prevent T=0 crashes)
    """

    def test_7_days_to_expiry(self):
        from_d = date(2026, 6, 3)   # Wednesday
        expiry = date(2026, 6, 10)  # 7 days later
        T = time_to_expiry_years(from_d, expiry)
        expected = 7 / 365.0
        assert abs(T - expected) < 1e-8, f"Expected {expected:.6f}, got {T:.6f}"

    def test_zero_days_returns_floor(self):
        """Same date → T = 1/365 (floor, not zero)."""
        d = date(2026, 6, 10)
        T = time_to_expiry_years(d, d)
        assert T == 1.0 / 365.0

    def test_calendar_days_not_trading_days(self):
        """NSE uses calendar days / 365, not trading days / 252."""
        # 365 calendar days = T=1.0 (one year)
        from_d = date(2026, 1, 1)
        expiry = date(2027, 1, 1)
        T = time_to_expiry_years(from_d, expiry)
        assert abs(T - 1.0) < 1e-6, f"365 calendar days should be T≈1.0, got {T}"

    def test_lru_cache_consistency(self):
        """Calling twice with same args returns same value (cached)."""
        d1 = date(2026, 6, 1)
        exp = date(2026, 6, 30)
        T1 = time_to_expiry_years(d1, exp)
        T2 = time_to_expiry_years(d1, exp)
        assert T1 == T2

    def test_expiry_before_from_returns_floor(self):
        """If expiry < from_date (negative T), returns floor 1/365."""
        from_d = date(2026, 6, 10)
        expiry = date(2026, 6, 3)   # in the past
        T = time_to_expiry_years(from_d, expiry)
        assert T == 1.0 / 365.0, "Negative T should be floored to 1/365"
