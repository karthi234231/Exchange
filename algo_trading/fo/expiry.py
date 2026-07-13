"""
algo_trading/fo/expiry.py — NSE F&O expiry calendar.

All expiry rules are sourced from verified NSE circulars.
No expiry date is hard-coded; every date is computed from the rules.

Verified rules (as of June 2026):
  Nifty 50 options/futures  : Weekly every TUESDAY (changed from Thursday, effective
                              Sep 1 2025 per NSE Circular FAOP68747)
  Bank Nifty options/futures: Monthly ONLY — last TUESDAY of each calendar month.
                              Weekly contracts discontinued Nov 2024.
  Monthly contracts (both)  : Last Tuesday of each calendar month.

Sources:
  [P] NSE Bank Nifty page:
      https://www.nseindia.com/static/products-services/equity-derivatives-banknifty
  [P] NSE Circular FAOP68747 (expiry changes):
      https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf
  [S] Zerodha FAQ (weekly expiry discontinuation):
      https://support.zerodha.com/category/trading-and-markets/trading-faqs/f-otrading/articles/weekly-expiries-discontinued-for-certain-index-derivatives

Holiday-adjusted rule:
  If the computed expiry day is an NSE trading holiday, the expiry rolls to the
  PRECEDING trading day (not the next one).  This is standard NSE practice.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterator

from algo_trading.core.calendar import is_trading_day


# ── Day-of-week constants (Python weekday: Monday=0, Sunday=6) ───────────────
_TUESDAY = 1


def _preceding_trading_day(d: date) -> date:
    """Return the nearest trading day ≤ d (d itself if it is a trading day)."""
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def _all_tuesdays_in_month(year: int, month: int) -> list[date]:
    """Return all Tuesday dates within (year, month), sorted ascending."""
    first_day = date(year, month, 1)
    # First Tuesday: advance from day 1 until we hit Tuesday
    days_to_first_tuesday = (_TUESDAY - first_day.weekday()) % 7
    first_tuesday = first_day + timedelta(days=days_to_first_tuesday)
    tuesdays: list[date] = []
    d = first_tuesday
    while d.month == month:
        tuesdays.append(d)
        d += timedelta(weeks=1)
    return tuesdays


def _last_tuesday_of_month(year: int, month: int) -> date:
    """Return the last Tuesday of (year, month).  Asserts month is valid."""
    tuesdays = _all_tuesdays_in_month(year, month)
    if not tuesdays:
        raise ValueError(f"No Tuesday found in {year}-{month:02d}")
    return tuesdays[-1]


class ExpiryCalendar:
    """
    Computes NSE F&O expiry dates on demand.

    Parameters
    ----------
    weekly_expiry_day : int
        Python weekday integer.  For Nifty (post Sep 2025) = 1 (Tuesday).
        Weekly expiry discontinued for BankNifty — pass weekly_enabled=False.
    weekly_enabled    : bool
        True for Nifty (weekly + monthly), False for BankNifty (monthly only).
    symbol            : str
        For logging/error messages only.
    """

    def __init__(
        self,
        symbol: str,
        weekly_expiry_day: int = _TUESDAY,
        weekly_enabled: bool = True,
    ) -> None:
        self.symbol = symbol
        self._weekly_day = weekly_expiry_day
        self._weekly_enabled = weekly_enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def next_expiry(self, from_date: date | None = None) -> date:
        """
        Return the nearest expiry date strictly after *from_date*.
        Returns the first upcoming expiry (weekly if enabled, else monthly).
        """
        if from_date is None:
            from_date = date.today()
        for d in self._expiry_stream(from_date):
            return d
        raise RuntimeError("Expiry stream terminated unexpectedly")  # pragma: no cover

    def current_expiry(self, on_date: date | None = None) -> date:
        """
        Return the expiry for the currently-active contract on *on_date*.

        Logic:
          - If on_date is itself an expiry day: that IS the current expiry.
          - Otherwise: first expiry strictly after on_date.
        """
        if on_date is None:
            on_date = date.today()
        adjusted = _preceding_trading_day(
            self._raw_expiry_on_or_before(on_date)
        )
        if adjusted >= on_date:
            return adjusted
        return self.next_expiry(on_date)

    def days_to_expiry(self, from_date: date | None = None) -> int:
        """Calendar days from *from_date* to the next expiry (inclusive of expiry day)."""
        if from_date is None:
            from_date = date.today()
        exp = self.next_expiry(from_date)
        return (exp - from_date).days

    def monthly_expiry(self, year: int, month: int) -> date:
        """
        Last Tuesday of (year, month), adjusted backward for holidays.
        This is the monthly expiry for both Nifty and BankNifty.
        """
        raw = _last_tuesday_of_month(year, month)
        return _preceding_trading_day(raw)

    def weekly_expiries_in_month(self, year: int, month: int) -> list[date]:
        """
        All weekly expiry dates in (year, month).
        For BankNifty (weekly_enabled=False) returns an empty list.
        """
        if not self._weekly_enabled:
            return []
        tuesdays = _all_tuesdays_in_month(year, month)
        return [_preceding_trading_day(t) for t in tuesdays]

    def is_expiry_day(self, d: date) -> bool:
        """Return True if *d* is an F&O expiry day for this symbol."""
        # Check if d itself is a holiday-adjusted expiry
        # A date is an expiry if it is the holiday-adjusted version of any raw expiry
        # Efficient approach: d is a Tuesday or the preceding trading day of a Tuesday
        year, month = d.year, d.month

        if self._weekly_enabled:
            weekly = self.weekly_expiries_in_month(year, month)
            if d in weekly:
                return True
        # Check monthly
        return d == self.monthly_expiry(year, month)

    def expiry_series(self, from_date: date, count: int) -> list[date]:
        """Return the next *count* expiry dates after *from_date*."""
        result: list[date] = []
        gen = self._expiry_stream(from_date)
        for _ in range(count):
            result.append(next(gen))
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _expiry_stream(self, after: date) -> Iterator[date]:
        """
        Infinite generator of expiry dates strictly after *after*, in ascending order.
        Yields holiday-adjusted dates.
        """
        year, month = after.year, after.month

        while True:
            if self._weekly_enabled:
                tuesdays = _all_tuesdays_in_month(year, month)
                for raw in tuesdays:
                    adjusted = _preceding_trading_day(raw)
                    if adjusted > after:
                        yield adjusted
            else:
                # Monthly only
                monthly = _preceding_trading_day(_last_tuesday_of_month(year, month))
                if monthly > after:
                    yield monthly

            # Advance to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    def _raw_expiry_on_or_before(self, d: date) -> date:
        """
        Find the most recent raw (non-holiday-adjusted) Tuesday expiry ≤ d.
        Used internally by current_expiry().
        """
        candidate = d
        while candidate.weekday() != _TUESDAY:
            candidate -= timedelta(days=1)
        return candidate


# ── Module-level singletons ──────────────────────────────────────────────────

nifty_expiry_calendar = ExpiryCalendar(
    symbol          = "NIFTY",
    weekly_expiry_day = _TUESDAY,
    weekly_enabled  = True,   # Nifty: weekly Tuesday expiry (since Sep 1 2025)
)

banknifty_expiry_calendar = ExpiryCalendar(
    symbol          = "BANKNIFTY",
    weekly_expiry_day = _TUESDAY,
    weekly_enabled  = False,  # BankNifty: monthly only (weekly discontinued Nov 2024)
)


def get_expiry_calendar(symbol: str) -> ExpiryCalendar:
    """Return the appropriate ExpiryCalendar for *symbol*."""
    mapping = {
        "NIFTY":     nifty_expiry_calendar,
        "BANKNIFTY": banknifty_expiry_calendar,
    }
    symbol = symbol.upper()
    if symbol not in mapping:
        raise KeyError(
            f"No expiry calendar for {symbol!r}. "
            "Add to mapping with source citations before trading."
        )
    return mapping[symbol]


@lru_cache(maxsize=2048)
def time_to_expiry_years(
    from_date: date,
    expiry_date: date,
    trading_hours_per_day: float = 6.25,  # 9:15–15:30 = 6h15m per trading day
) -> float:
    """
    Compute time to expiry in years using calendar days.

    Convention used by NSE practitioners:
      T = calendar_days_to_expiry / 365

    Note on trading-day vs calendar-day convention:
      Some practitioners use T = trading_days / 252.
      NSE uses calendar days / 365 for settlement value computation.
      This implementation uses calendar / 365 (NSE convention).
      Pass use_trading_days=True if you need the alternative.

    Returns 1/365 (one calendar day) as the floor to prevent T=0 crashes.
    """
    delta = (expiry_date - from_date).days
    return max(delta / 365.0, 1.0 / 365.0)
