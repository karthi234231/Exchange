"""
algo_trading/market_data/macro_events.py — Indian macro event calendar.

Tracks upcoming high-impact macro events that should suppress or halt
technical trading signals:
  - RBI Monetary Policy Committee (MPC) announcements (~6 per year)
  - CPI inflation data (~monthly)
  - GDP quarterly releases
  - IIP industrial production data
  - US Fed FOMC meetings (affects FII flows)
  - USDINR rate (affects cost of imported goods, FII hedging)
  - Crude oil price (Brent) — India imports ~80% of crude
  - Union Budget (annual, February)
  - NSE F&O expiry dates (every last Thursday of month)

On high-impact event days: the system should reduce position sizes by 50%
and avoid entering new trades 1 hour before/after the announcement.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Sequence
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Impact level ordering for comparison
_IMPACT_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# Default window hours for is_high_impact_window
_DEFAULT_WINDOW_HOURS: float = 1.0

# Rough RBI MPC announcement months (bi-monthly policy; ~Feb, Apr, Jun, Aug, Oct, Dec).
# ASSUMED — verify against official RBI calendar at https://www.rbi.org.in/
_RBI_MPC_MONTHS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)

# Rough RBI MPC announcement day-of-month (third Wednesday of the policy month).
# ASSUMED — actual dates vary; update from RBI website each year.
_RBI_MPC_APPROXIMATE_DAY: int = 8  # first announcement day (placeholder)

# Union Budget is typically on the first working day of February.
_BUDGET_MONTH: int = 2
_BUDGET_APPROXIMATE_DAY: int = 1

# RBI MPC announcement time (IST).
_RBI_MPC_TIME: time = time(10, 0)  # 10:00 IST — ASSUMED

# CPI release time (IST) — typically 5:30 PM on the release day.
_CPI_RELEASE_TIME: time = time(17, 30)  # ASSUMED

# FOMC meeting result time (New York; impact felt in India the following morning).
# We model it as a 09:00 IST impact since that is when Indian market absorbs overnight.
_FOMC_IMPACT_TIME: time = time(9, 0)


class MacroEventType(str, Enum):
    """Categories of high-impact macro events relevant to Indian equity markets.

    Attributes:
        RBI_MPC: RBI Monetary Policy Committee rate decision.
        CPI: Consumer Price Index inflation release.
        GDP: Quarterly Gross Domestic Product release.
        IIP: Index of Industrial Production monthly release.
        FOMC: US Federal Reserve FOMC meeting outcome.
        BUDGET: Indian Union Budget announcement.
        FNO_EXPIRY: NSE Futures & Options monthly expiry (last Thursday).
        OTHER: Any other macro event not covered above.
    """

    RBI_MPC   = "RBI_MPC"
    CPI       = "CPI"
    GDP       = "GDP"
    IIP       = "IIP"
    FOMC      = "FOMC"
    BUDGET    = "BUDGET"
    FNO_EXPIRY = "FNO_EXPIRY"
    OTHER     = "OTHER"


@dataclass
class MacroEvent:
    """A single macro event entry in the calendar.

    Attributes:
        event_type: Category of the event.
        event_date: Calendar date of the event.
        event_time_ist: Scheduled announcement time in IST, or ``None`` if unknown.
        expected_impact: One of ``"high"``, ``"medium"``, or ``"low"``.
        description: Human-readable description of the event.
    """

    event_type: MacroEventType
    event_date: date
    event_time_ist: time | None
    expected_impact: str
    description: str

    def __post_init__(self) -> None:
        """Validate impact level on construction."""
        if self.expected_impact not in _IMPACT_ORDER:
            raise ValueError(
                f"expected_impact must be 'high', 'medium', or 'low'; "
                f"got {self.expected_impact!r}"
            )


class MacroEventCalendar:
    """In-memory calendar of macro events with query helpers.

    Events can be added manually or via :func:`build_default_calendar`.
    The calendar is unsorted internally; queries sort results on the fly.

    Example usage::

        cal = build_default_calendar(2026)
        if cal.is_high_impact_window(datetime.now(IST)):
            position_size_multiplier = 0.5
        days = cal.days_to_next_event(date.today())
    """

    def __init__(self) -> None:
        """Initialise an empty calendar."""
        self._events: list[MacroEvent] = []

    def add_event(self, event: MacroEvent) -> None:
        """Add a macro event to the calendar.

        Args:
            event: The :class:`MacroEvent` to add.
        """
        self._events.append(event)
        log.debug(
            "macro_calendar.event_added",
            event_type=event.event_type.value,
            event_date=str(event.event_date),
            impact=event.expected_impact,
        )

    def get_events_in_window(self, start: date, end: date) -> list[MacroEvent]:
        """Return all events whose date falls within [start, end] inclusive.

        Args:
            start: Window start date.
            end: Window end date.

        Returns:
            Events sorted by date ascending.
        """
        result = [
            e for e in self._events
            if start <= e.event_date <= end
        ]
        result.sort(key=lambda e: (e.event_date, e.event_time_ist or time.min))
        return result

    def is_high_impact_window(
        self,
        dt: datetime,
        window_hours: float = _DEFAULT_WINDOW_HOURS,
    ) -> bool:
        """Return ``True`` if ``dt`` falls within ``window_hours`` of a high-impact event.

        Checks only events with ``expected_impact == "high"``.  If an event
        has no scheduled time, the entire event day is treated as in-window.

        Args:
            dt: Current datetime (timezone-aware recommended; naive treated as IST).
            window_hours: Hours before/after event announcement to flag as in-window.

        Returns:
            ``True`` if the current moment is within the impact window.
        """
        window_delta = timedelta(hours=window_hours)
        check_date = dt.date() if hasattr(dt, "date") else dt

        for event in self._events:
            if event.expected_impact != "high":
                continue
            if event.event_date != check_date:
                # Check cross-midnight boundary
                if (event.event_date - check_date).days not in (0, 1, -1):
                    continue

            if event.event_time_ist is None:
                # No scheduled time — entire event day is in-window
                if event.event_date == check_date:
                    return True
                continue

            # Construct a naive datetime for comparison
            event_naive = datetime.combine(event.event_date, event.event_time_ist)
            dt_naive = dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

            delta = abs((dt_naive - event_naive).total_seconds())
            if delta <= window_delta.total_seconds():
                log.debug(
                    "macro_calendar.in_window",
                    event_type=event.event_type.value,
                    event_date=str(event.event_date),
                    event_time=str(event.event_time_ist),
                    delta_sec=int(delta),
                )
                return True

        return False

    def days_to_next_event(
        self,
        from_date: date,
        min_impact: str = "high",
    ) -> int | None:
        """Return the number of calendar days until the next qualifying event.

        Args:
            from_date: Date from which to measure.
            min_impact: Minimum impact level to consider.  Events with impact
                below this level are ignored.  One of ``"high"``, ``"medium"``,
                ``"low"``.

        Returns:
            Number of days (0 if today is an event day), or ``None`` if no
            future events are in the calendar.
        """
        min_level = _IMPACT_ORDER.get(min_impact, 2)
        upcoming = sorted(
            [
                e for e in self._events
                if e.event_date >= from_date
                and _IMPACT_ORDER.get(e.expected_impact, 0) >= min_level
            ],
            key=lambda e: e.event_date,
        )
        if not upcoming:
            return None
        return (upcoming[0].event_date - from_date).days

    def get_next_fno_expiry(self, from_date: date) -> date:
        """Return the next NSE F&O monthly expiry date on or after ``from_date``.

        NSE F&O expires on the last Thursday of each calendar month.

        Args:
            from_date: Reference date.

        Returns:
            The nearest upcoming last-Thursday expiry date.
        """
        candidate = _last_thursday_of_month(from_date.year, from_date.month)
        if candidate >= from_date:
            return candidate

        # Advance to next month
        if from_date.month == 12:
            return _last_thursday_of_month(from_date.year + 1, 1)
        return _last_thursday_of_month(from_date.year, from_date.month + 1)

    def all_events(self) -> list[MacroEvent]:
        """Return all events sorted by date.

        Returns:
            All registered events, oldest-first.
        """
        return sorted(self._events, key=lambda e: e.event_date)


def build_default_calendar(year: int) -> MacroEventCalendar:
    """Pre-populate a :class:`MacroEventCalendar` with known recurring events.

    Populates:
    - All 12 F&O expiry dates (last Thursday of each month) — computed exactly.
    - Approximate RBI MPC dates (bi-monthly: Feb, Apr, Jun, Aug, Oct, Dec) —
      ASSUMED, update from https://www.rbi.org.in/ before production use.
    - Monthly CPI release (approximate; 12th of following month) — ASSUMED.
    - Union Budget (1 February for non-election years) — ASSUMED.
    - Approximate FOMC meetings (8 per year, odd months approx.) — ASSUMED.
    - Quarterly GDP releases (May, Aug, Nov, Feb for Q4/Q1/Q2/Q3) — ASSUMED.

    Args:
        year: Calendar year for which to build the schedule.

    Returns:
        Populated :class:`MacroEventCalendar`.
    """
    cal = MacroEventCalendar()

    # ── F&O Expiry (computed; exact) ──────────────────────────────────────
    for month in range(1, 13):
        expiry = _last_thursday_of_month(year, month)
        cal.add_event(MacroEvent(
            event_type=MacroEventType.FNO_EXPIRY,
            event_date=expiry,
            event_time_ist=time(15, 30),
            expected_impact="high",
            description=f"NSE F&O Monthly Expiry {calendar.month_abbr[month]} {year}",
        ))

    # ── RBI MPC (ASSUMED — verify from https://www.rbi.org.in/) ──────────
    # Typical schedule: bi-monthly, first announcement ~8th of policy month.
    # Actual dates vary; this is a rough schedule only.
    for month in _RBI_MPC_MONTHS:
        # Use the Wednesday closest to day 8 of that month as approximation
        approx_day = _first_weekday_on_or_after(year, month, 8, calendar.WEDNESDAY)
        cal.add_event(MacroEvent(
            event_type=MacroEventType.RBI_MPC,
            event_date=date(year, month, approx_day),
            event_time_ist=_RBI_MPC_TIME,
            expected_impact="high",
            description=(
                f"RBI MPC Policy Announcement {calendar.month_abbr[month]} {year} "
                f"[ASSUMED — verify from rbi.org.in]"
            ),
        ))

    # ── CPI Release (~monthly, 12th of following month) (ASSUMED) ─────────
    for month in range(1, 13):
        release_month = month + 1 if month < 12 else 1
        release_year  = year if month < 12 else year + 1
        try:
            release_date = date(release_year, release_month, 12)
        except ValueError:
            release_date = date(release_year, release_month, 28)
        cal.add_event(MacroEvent(
            event_type=MacroEventType.CPI,
            event_date=release_date,
            event_time_ist=_CPI_RELEASE_TIME,
            expected_impact="medium",
            description=(
                f"CPI Inflation Data for {calendar.month_abbr[month]} {year} "
                f"[ASSUMED — verify from mospi.gov.in]"
            ),
        ))

    # ── Union Budget (1 February, or nearest working day) (ASSUMED) ───────
    budget_date = date(year, _BUDGET_MONTH, _BUDGET_APPROXIMATE_DAY)
    # Shift to Monday if weekend
    if budget_date.weekday() == 5:  # Saturday
        budget_date += timedelta(days=2)
    elif budget_date.weekday() == 6:  # Sunday
        budget_date += timedelta(days=1)
    cal.add_event(MacroEvent(
        event_type=MacroEventType.BUDGET,
        event_date=budget_date,
        event_time_ist=time(11, 0),
        expected_impact="high",
        description=(
            f"Union Budget {year} [ASSUMED — verify from indiabudget.gov.in]"
        ),
    ))

    # ── FOMC Meetings (~8 per year, roughly every 6-7 weeks) (ASSUMED) ────
    # Approximate dates for typical FOMC schedule:
    # Jan, Mar, May, Jun, Jul, Sep, Nov, Dec — Wednesday of meeting week.
    fomc_approx_months_days = [
        (1, 29), (3, 18), (5, 6), (6, 17),
        (7, 29), (9, 16), (11, 4), (12, 16),
    ]
    for m, d in fomc_approx_months_days:
        try:
            fomc_date = date(year, m, d)
        except ValueError:
            continue
        cal.add_event(MacroEvent(
            event_type=MacroEventType.FOMC,
            event_date=fomc_date,
            event_time_ist=_FOMC_IMPACT_TIME,
            expected_impact="medium",
            description=(
                f"US Fed FOMC Meeting {calendar.month_abbr[m]} {year} "
                f"[ASSUMED — verify from federalreserve.gov]"
            ),
        ))

    # ── GDP Quarterly Release (ASSUMED — MoSPI calendar) ─────────────────
    # Q4 result (Jan-Mar) released ~May; Q1 (Apr-Jun) ~Aug;
    # Q2 (Jul-Sep) ~Nov; Q3 (Oct-Dec) ~Feb of following year.
    gdp_releases = [
        (year,     5,  30, "Q4 (Jan-Mar)"),
        (year,     8,  30, "Q1 (Apr-Jun)"),
        (year,    11,  29, "Q2 (Jul-Sep)"),
        (year + 1, 2,  28, "Q3 (Oct-Dec)"),
    ]
    for y, m, d, label in gdp_releases:
        try:
            gdp_date = date(y, m, d)
        except ValueError:
            gdp_date = date(y, m, 28)
        cal.add_event(MacroEvent(
            event_type=MacroEventType.GDP,
            event_date=gdp_date,
            event_time_ist=time(17, 30),
            expected_impact="medium",
            description=(
                f"GDP {label} {year} [ASSUMED — verify from mospi.gov.in]"
            ),
        ))

    log.info(
        "macro_calendar.built",
        year=year,
        total_events=len(cal.all_events()),
    )
    return cal


# ── Private helpers ───────────────────────────────────────────────────────────

def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month.

    Args:
        year: Calendar year.
        month: Calendar month (1–12).

    Returns:
        Date of the last Thursday in that month.
    """
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # Weekday: Monday=0, ..., Thursday=3, ..., Sunday=6
    offset = (d.weekday() - calendar.THURSDAY) % 7
    return d - timedelta(days=offset)


def _first_weekday_on_or_after(
    year: int, month: int, from_day: int, weekday: int
) -> int:
    """Return the day-of-month of the first occurrence of ``weekday`` on or after ``from_day``.

    Args:
        year: Calendar year.
        month: Calendar month (1–12).
        from_day: Earliest day-of-month to consider.
        weekday: Target weekday (0=Monday, …, 6=Sunday).

    Returns:
        Day-of-month (int) of the first matching weekday.
    """
    last_day = calendar.monthrange(year, month)[1]
    for day in range(from_day, last_day + 1):
        if date(year, month, day).weekday() == weekday:
            return day
    # Fallback: last day of month
    return last_day
