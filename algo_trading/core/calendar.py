"""
market_calendar.py — NSE trading calendar and session utilities.

All timestamps in this module are IST (Asia/Kolkata, UTC+5:30).
Holiday list is loaded from config/nse_holidays.yaml.  If the current year's
holidays are absent, a WARNING is logged at module load time.
Special sessions (Muhurat trading, DR sessions) are loaded from
config/nse_special_sessions.yaml if it exists.
Source: NSE India annual trading holiday list.

NOTE: All entries marked ASSUMED per verified_facts.md.
Recheck before Stage 1 paper trading begins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

IST = ZoneInfo("Asia/Kolkata")

log = structlog.get_logger(__name__)


# ── Custom exceptions ────────────────────────────────────────────────────────

class OutsideTradingHoursError(Exception):
    """Raised when a datetime falls outside the NSE trading session hours."""


class CalendarStalenessError(Exception):
    """Raised when the holiday calendar data is older than the permitted max age."""


class InstrumentMasterStaleError(Exception):
    """Raised when the instrument master file is missing or too old."""


# ── Module-level staleness constant ─────────────────────────────────────────

CALENDAR_MAX_AGE_DAYS: int = 30

# ── Instrument master freshness constant ─────────────────────────────────────

INSTRUMENT_MASTER_MAX_AGE_HOURS: int = 24

# ── Paths ────────────────────────────────────────────────────────────────────

_HOLIDAYS_YAML_PATH = Path(__file__).parent.parent / "config" / "nse_holidays.yaml"
_SPECIAL_SESSIONS_YAML_PATH = (
    Path(__file__).parent.parent / "config" / "nse_special_sessions.yaml"
)
_CALENDAR_STATE_PATH = Path("state") / "calendar_last_updated.json"


# ── NSE session time constants ───────────────────────────────────────────────

SESSION_START = time(9, 15)
SESSION_END = time(15, 30)
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)
NO_NEW_TRADES_AFTER = time(15, 15)   # [SURV-8]
FORCE_CLOSE_TIME = time(15, 15)      # [SURV-8]

# Pre-open segment constants (#64)
FO_PRE_OPEN_START = time(9, 0)         # F&O pre-open start
FO_PRE_OPEN_END = time(9, 15)          # F&O pre-open ends at normal market open
EQUITY_PRE_OPEN_START = time(9, 0)     # Equity pre-open start
EQUITY_PRE_OPEN_END = time(9, 8)       # Equity pre-open ends at 09:08 (call auction)
EQUITY_CALL_AUCTION_END = time(9, 15)  # After 09:08, price discovery until 09:15

# Segment-aware force-close constants (#65)
# Source: NSEarchives circular FAOP68747 for F&O timing changes
# IMPORTANT: F&O 15:40 extension requires official verification before production use
EQUITY_NO_NEW_ENTRY_TIME = time(15, 15)  # No new equity entries after this
EQUITY_FORCE_CLOSE_TIME = time(15, 15)   # Force close all equity intraday by this time
EQUITY_SESSION_END = time(15, 30)        # Exchange closes

FO_NO_NEW_ENTRY_TIME = time(15, 20)      # [ASSUMED] No new F&O entries
FO_FORCE_CLOSE_TIME = time(15, 30)       # [ASSUMED] Force close F&O by this time
# Note: F&O 15:40 extension reported for August 2026. Use ASSUMED until official NSE circular.
FO_SESSION_END = time(15, 30)            # [ASSUMED] F&O session end


# ── Session state enum (#64) ─────────────────────────────────────────────────

class SessionState(str, Enum):
    PRE_OPEN = "PRE_OPEN"
    CALL_AUCTION = "CALL_AUCTION"    # 09:08–09:15 for equity
    NORMAL = "NORMAL"                # Full normal trading
    NO_NEW_ENTRY = "NO_NEW_ENTRY"    # Last 15 minutes before force-close
    FORCE_CLOSE = "FORCE_CLOSE"      # After 15:15 (equity) or 15:30 (F&O)
    CLOSED = "CLOSED"                # Outside all sessions


# ── SpecialSession dataclass (#63) ───────────────────────────────────────────

@dataclass
class SpecialSession:
    date: date
    session_type: str        # "MUHURAT" | "DR_SESSION" | "SHORTENED"
    session_start: time      # overrides SESSION_START for this date
    session_end: time        # overrides SESSION_END for this date
    tradeable: bool = True   # False = market closed (special holiday)
    notes: str = ""


# ── NSE Holiday fallback list ─────────────────────────────────────────────────

_NSE_HOLIDAYS_2026_FALLBACK: list[date] = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
    date(2026, 3, 25),   # Holi
    date(2026, 4, 2),    # Good Friday
    date(2026, 4, 6),    # Gudi Padwa
    date(2026, 4, 10),   # Shri Ram Navami
    date(2026, 4, 14),   # Dr. B.R. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti / Mahatma Gandhi Jayanti
    date(2026, 10, 24),  # Diwali Laxmi Puja (tentative)
    date(2026, 10, 26),  # Diwali Balipratipada (tentative)
    date(2026, 11, 5),   # Guru Nanak Jayanti (tentative)
    date(2026, 12, 25),  # Christmas
]


# ── Holiday loading ──────────────────────────────────────────────────────────

def _load_holidays() -> frozenset[date]:
    """Load NSE holiday dates for the current year from config/nse_holidays.yaml.

    If the YAML file is missing or the current year's holidays are absent,
    logs a WARNING and falls back to the hardcoded 2026 list (if applicable).

    Returns:
        frozenset of holiday dates for the current year.
    """
    current_year = datetime.now(tz=IST).year
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        log.warning(
            "PyYAML not installed; cannot load nse_holidays.yaml. "
            "Install pyyaml or pip install pyyaml. Falling back to hardcoded list."
        )
        if current_year == 2026:
            return frozenset(_NSE_HOLIDAYS_2026_FALLBACK)
        log.warning(
            "NSE holiday list for year %d not found in config/nse_holidays.yaml "
            "— market_is_holiday() will be inaccurate",
            current_year,
        )
        return frozenset()

    if not _HOLIDAYS_YAML_PATH.exists():
        log.warning(
            "config/nse_holidays.yaml not found at %s. Falling back to hardcoded list.",
            _HOLIDAYS_YAML_PATH,
        )
        if current_year == 2026:
            return frozenset(_NSE_HOLIDAYS_2026_FALLBACK)
        log.warning(
            "NSE holiday list for year %d not found in config/nse_holidays.yaml "
            "— market_is_holiday() will be inaccurate",
            current_year,
        )
        return frozenset()

    try:
        with open(_HOLIDAYS_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        log.error("Failed to parse nse_holidays.yaml: %s. Using hardcoded fallback.", exc)
        if current_year == 2026:
            return frozenset(_NSE_HOLIDAYS_2026_FALLBACK)
        return frozenset()

    year_key = str(current_year)
    if year_key not in (data or {}):
        log.warning(
            "NSE holiday list for year %d not found in config/nse_holidays.yaml "
            "— market_is_holiday() will be inaccurate",
            current_year,
        )
        if current_year == 2026:
            return frozenset(_NSE_HOLIDAYS_2026_FALLBACK)
        return frozenset()

    raw_dates = data[year_key]
    holidays: list[date] = []
    for entry in raw_dates:
        if isinstance(entry, date):
            holidays.append(entry)
        elif isinstance(entry, str):
            try:
                holidays.append(date.fromisoformat(entry))
            except ValueError:
                log.warning("Invalid date string in nse_holidays.yaml: %r", entry)
    return frozenset(holidays)


def _load_special_sessions() -> dict[date, SpecialSession]:
    """Load special sessions from config/nse_special_sessions.yaml if it exists.

    Expected YAML structure:
        sessions:
          - date: "2026-10-24"
            session_type: "MUHURAT"
            session_start: "18:15"
            session_end: "19:15"
            tradeable: true
            notes: "Muhurat trading for Diwali"

    Returns:
        dict mapping date -> SpecialSession.
    """
    result: dict[date, SpecialSession] = {}
    if not _SPECIAL_SESSIONS_YAML_PATH.exists():
        return result

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        log.warning(
            "PyYAML not installed; cannot load nse_special_sessions.yaml."
        )
        return result

    try:
        with open(_SPECIAL_SESSIONS_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        log.error("Failed to parse nse_special_sessions.yaml: %s", exc)
        return result

    if not data or "sessions" not in data:
        return result

    for entry in data.get("sessions", []):
        try:
            raw_date = entry["date"]
            if isinstance(raw_date, str):
                ss_date = date.fromisoformat(raw_date)
            elif isinstance(raw_date, date):
                ss_date = raw_date
            else:
                log.warning("Invalid date in nse_special_sessions.yaml: %r", raw_date)
                continue

            raw_start = entry["session_start"]
            raw_end = entry["session_end"]
            ss_start = (
                time.fromisoformat(raw_start) if isinstance(raw_start, str) else raw_start
            )
            ss_end = (
                time.fromisoformat(raw_end) if isinstance(raw_end, str) else raw_end
            )

            ss = SpecialSession(
                date=ss_date,
                session_type=str(entry.get("session_type", "SHORTENED")),
                session_start=ss_start,
                session_end=ss_end,
                tradeable=bool(entry.get("tradeable", True)),
                notes=str(entry.get("notes", "")),
            )
            result[ss_date] = ss
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Skipping invalid special session entry: %s — %r", exc, entry)

    return result


# ── Module-level holiday set and special sessions loaded at import time ───────

_NSE_HOLIDAYS: frozenset[date] = _load_holidays()


# ── MarketCalendar class ──────────────────────────────────────────────────────

class MarketCalendar:
    """NSE market calendar with holiday awareness, special sessions, and staleness checks.

    Parameters
    ----------
    live_mode : bool
        When True, staleness check raises CalendarStalenessError.
        When False (default), staleness check only logs a WARNING.
    max_age_days : int
        Maximum allowed age of calendar data in days (default: CALENDAR_MAX_AGE_DAYS).
    """

    def __init__(
        self,
        live_mode: bool = False,
        max_age_days: int = CALENDAR_MAX_AGE_DAYS,
    ) -> None:
        self._live_mode = live_mode
        self._max_age_days = max_age_days
        self._special_sessions: dict[date, SpecialSession] = _load_special_sessions()
        self.validate(live_mode=live_mode)

    # ── Staleness check (#62) ─────────────────────────────────────────────────

    def check_staleness(self, max_age_days: int = CALENDAR_MAX_AGE_DAYS) -> None:
        """Check whether the holiday calendar data is stale.

        Checks the last-modified time of ``config/nse_holidays.yaml`` first.
        Falls back to checking ``state/calendar_last_updated.json`` if the YAML
        path is unavailable.

        Args:
            max_age_days: Maximum permitted age in days before the calendar is
                considered stale.

        Raises:
            CalendarStalenessError: If the calendar data is older than
                *max_age_days* days.
        """
        age_days: Optional[float] = None
        source_checked: str = ""

        # Prefer checking the holidays YAML directly
        if _HOLIDAYS_YAML_PATH.exists():
            mtime = _HOLIDAYS_YAML_PATH.stat().st_mtime
            age_seconds = datetime.now().timestamp() - mtime
            age_days = age_seconds / 86400
            source_checked = str(_HOLIDAYS_YAML_PATH)
        elif _CALENDAR_STATE_PATH.exists():
            try:
                with open(_CALENDAR_STATE_PATH, encoding="utf-8") as f:
                    state = json.load(f)
                last_updated_str = state.get("last_updated", "")
                last_updated_dt = datetime.fromisoformat(last_updated_str)
                age_days = (datetime.now() - last_updated_dt).total_seconds() / 86400
                source_checked = str(_CALENDAR_STATE_PATH)
            except Exception as exc:
                log.warning(
                    "Could not read calendar state file for staleness check",
                    path=str(_CALENDAR_STATE_PATH),
                    error=str(exc),
                )
        else:
            log.warning(
                "Calendar staleness check: neither holidays YAML nor state file found; "
                "assuming stale",
                holidays_yaml=str(_HOLIDAYS_YAML_PATH),
                state_file=str(_CALENDAR_STATE_PATH),
            )
            age_days = float("inf")

        if age_days is not None and age_days > max_age_days:
            age_int = int(age_days)
            msg = (
                f"Calendar data is {age_int} days old. "
                f"Update config/nse_holidays.yaml (source: {source_checked})"
            )
            raise CalendarStalenessError(msg)

    def validate(self, live_mode: bool = False) -> None:
        """Run calendar validation checks.

        In live mode, raises CalendarStalenessError if the calendar is stale.
        In non-live mode, logs a WARNING instead.

        Args:
            live_mode: If True, raise on staleness. If False, warn only.
        """
        try:
            self.check_staleness(self._max_age_days)
        except CalendarStalenessError as exc:
            if live_mode:
                raise
            log.warning(
                "Calendar staleness warning (non-live mode)",
                detail=str(exc),
            )

    # ── Special session management (#63) ─────────────────────────────────────

    def register_special_session(self, ss: SpecialSession) -> None:
        """Register a SpecialSession override for a specific date.

        Args:
            ss: SpecialSession to register. Overwrites any existing entry for
                that date.
        """
        self._special_sessions[ss.date] = ss
        log.info(
            "Registered special session",
            date=ss.date.isoformat(),
            session_type=ss.session_type,
            session_start=ss.session_start.isoformat(),
            session_end=ss.session_end.isoformat(),
            tradeable=ss.tradeable,
        )

    def get_session_times(self, d: date) -> tuple[time, time]:
        """Return (session_start, session_end) for a given date.

        Uses special session override if a SpecialSession is registered for *d*,
        otherwise returns the standard SESSION_START / SESSION_END.

        Args:
            d: The date to query.

        Returns:
            Tuple of (session_start, session_end) as time objects.
        """
        ss = self._special_sessions.get(d)
        if ss is not None:
            return (ss.session_start, ss.session_end)
        return (SESSION_START, SESSION_END)

    # ── Market open check (special-session aware) (#63) ───────────────────────

    def is_market_open(self, ts: datetime) -> bool:
        """Return True if *ts* falls within the active trading session.

        Checks ``_special_sessions`` first; if a special session exists for the
        date, uses its hours. If the special session has ``tradeable=False``, the
        market is considered closed regardless of time.

        Args:
            ts: Timestamp to check (any timezone; converted to IST internally).

        Returns:
            True if the market is open at *ts*.
        """
        ist_ts = ts.astimezone(IST)
        d = ist_ts.date()
        t = ist_ts.time()

        ss = self._special_sessions.get(d)
        if ss is not None:
            if not ss.tradeable:
                return False
            return ss.session_start <= t < ss.session_end

        if not is_trading_day(d):
            return False
        return SESSION_START <= t < SESSION_END

    # ── Pre-open checks (#64) ─────────────────────────────────────────────────

    def is_pre_open(self, ts: datetime, segment: str = "EQUITY") -> bool:
        """Return True if *ts* is within the pre-open window for *segment*.

        Args:
            ts: Timestamp to check (any timezone; converted to IST internally).
            segment: "EQUITY" or "FO".

        Returns:
            True if the timestamp is in the pre-open window.
        """
        ist_ts = ts.astimezone(IST)
        if not is_trading_day(ist_ts.date()):
            return False
        t = ist_ts.time()
        if segment.upper() == "FO":
            return FO_PRE_OPEN_START <= t < FO_PRE_OPEN_END
        # EQUITY (default)
        return EQUITY_PRE_OPEN_START <= t < EQUITY_PRE_OPEN_END

    def is_tradeable_time(self, ts: datetime, segment: str = "EQUITY") -> bool:
        """Return True if *ts* is a safe tradeable moment for *segment*.

        Returns False during pre-open, outside market hours, or on non-trading
        days. Uses special session overrides when applicable.

        Args:
            ts: Timestamp to check (any timezone; converted to IST internally).
            segment: "EQUITY" or "FO".

        Returns:
            True if new trades are permitted at this timestamp.
        """
        ist_ts = ts.astimezone(IST)
        if self.is_pre_open(ist_ts, segment=segment):
            return False
        if not self.is_market_open(ist_ts):
            return False
        return True

    # ── Session state (#64) ───────────────────────────────────────────────────

    def get_session_state(self, ts: datetime, segment: str = "EQUITY") -> SessionState:
        """Return the SessionState for *ts* and *segment*.

        State machine (EQUITY):
          CLOSED          → outside all sessions or non-trading day
          PRE_OPEN        → 09:00–09:08
          CALL_AUCTION    → 09:08–09:15
          NORMAL          → 09:15–15:15 (EQUITY) / 09:15–15:20 (FO)
          NO_NEW_ENTRY    → 15:15–15:15 (EQUITY) / 15:20–15:30 (FO)
          FORCE_CLOSE     → at/after EQUITY_FORCE_CLOSE_TIME / FO_FORCE_CLOSE_TIME

        For special sessions the market hours are taken from the registered
        SpecialSession; PRE-OPEN / CALL_AUCTION ranges still use fixed constants.

        Args:
            ts: Timestamp to check (any timezone; converted to IST internally).
            segment: "EQUITY" or "FO".

        Returns:
            Appropriate SessionState enum member.
        """
        seg = segment.upper()
        ist_ts = ts.astimezone(IST)
        d = ist_ts.date()
        t = ist_ts.time()

        if not is_trading_day(d):
            # Check if it's a special tradeable session (e.g. Muhurat)
            ss = self._special_sessions.get(d)
            if ss is None or not ss.tradeable:
                return SessionState.CLOSED

        # PRE_OPEN window
        if seg == "FO":
            if FO_PRE_OPEN_START <= t < FO_PRE_OPEN_END:
                return SessionState.PRE_OPEN
        else:
            if EQUITY_PRE_OPEN_START <= t < EQUITY_PRE_OPEN_END:
                return SessionState.PRE_OPEN
            # CALL_AUCTION only for equity
            if EQUITY_PRE_OPEN_END <= t < EQUITY_CALL_AUCTION_END:
                return SessionState.CALL_AUCTION

        # Determine session bounds (special session aware)
        sess_start, sess_end = self.get_session_times(d)

        # Before session start
        if t < sess_start:
            return SessionState.CLOSED

        # FORCE_CLOSE / NO_NEW_ENTRY / NORMAL
        if seg == "FO":
            if t >= FO_FORCE_CLOSE_TIME:
                return SessionState.FORCE_CLOSE
            if t >= FO_NO_NEW_ENTRY_TIME:
                return SessionState.NO_NEW_ENTRY
        else:
            if t >= EQUITY_FORCE_CLOSE_TIME:
                return SessionState.FORCE_CLOSE
            if t >= EQUITY_NO_NEW_ENTRY_TIME:
                return SessionState.NO_NEW_ENTRY

        # After session end (but before force-close — shouldn't normally occur)
        if t >= sess_end:
            return SessionState.CLOSED

        return SessionState.NORMAL

    # ── Segment-aware force-close helpers (#65) ───────────────────────────────

    def get_force_close_time(self, segment: str = "EQUITY") -> time:
        """Return the force-close time for *segment*.

        Args:
            segment: "EQUITY" or "FO".

        Returns:
            time object representing the force-close cut-off.
        """
        if segment.upper() == "FO":
            return FO_FORCE_CLOSE_TIME
        return EQUITY_FORCE_CLOSE_TIME

    def get_no_new_entry_time(self, segment: str = "EQUITY") -> time:
        """Return the no-new-entry cut-off time for *segment*.

        Args:
            segment: "EQUITY" or "FO".

        Returns:
            time object representing the no-new-entry cut-off.
        """
        if segment.upper() == "FO":
            return FO_NO_NEW_ENTRY_TIME
        return EQUITY_NO_NEW_ENTRY_TIME


# ── Module-level singleton ────────────────────────────────────────────────────

_calendar: Optional[MarketCalendar] = None


def get_calendar(live_mode: bool = False) -> MarketCalendar:
    """Return (or create) the module-level MarketCalendar singleton.

    Args:
        live_mode: Passed to MarketCalendar on first construction; ignored on
            subsequent calls (singleton is reused).
    """
    global _calendar
    if _calendar is None:
        _calendar = MarketCalendar(live_mode=live_mode)
    return _calendar


# ── Module-level convenience functions (backward-compatible) ─────────────────

def is_trading_day(d: date | None = None) -> bool:
    """Return True if *d* is an NSE trading day (not weekend, not holiday)."""
    if d is None:
        d = datetime.now(tz=IST).date()
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in _NSE_HOLIDAYS


def next_trading_day(d: date | None = None) -> date:
    """Return the next NSE trading day after *d* (or today's next if d is None)."""
    if d is None:
        d = datetime.now(tz=IST).date()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(d: date | None = None) -> date:
    """Return the most recent NSE trading day before *d*."""
    if d is None:
        d = datetime.now(tz=IST).date()
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def is_market_open(dt: datetime | None = None) -> bool:
    """Return True if *dt* falls within the NSE trading session.

    This module-level function does NOT check special sessions.
    Use ``MarketCalendar.is_market_open()`` for special session awareness.
    """
    if dt is None:
        dt = datetime.now(tz=IST)
    ist_dt = dt.astimezone(IST)
    if not is_trading_day(ist_dt.date()):
        return False
    t = ist_dt.time()
    return SESSION_START <= t < SESSION_END


def is_pre_open(dt: datetime | None = None) -> bool:
    """Return True if *dt* is within the NSE equity pre-open call auction session.

    This module-level function uses equity pre-open constants.
    Use ``MarketCalendar.is_pre_open(segment=...)`` for segment-specific checks.
    """
    if dt is None:
        dt = datetime.now(tz=IST)
    ist_dt = dt.astimezone(IST)
    if not is_trading_day(ist_dt.date()):
        return False
    t = ist_dt.time()
    return EQUITY_PRE_OPEN_START <= t < EQUITY_PRE_OPEN_END


def can_enter_new_trade(dt: datetime | None = None) -> bool:
    """Return True if it is currently safe to enter a new position.

    No new entries after NO_NEW_TRADES_AFTER [SURV-8].
    No entries during the opening 15-min bar (09:15-09:30) [SURV-8].
    """
    if dt is None:
        dt = datetime.now(tz=IST)
    ist_dt = dt.astimezone(IST)
    if not is_market_open(ist_dt):
        return False
    t = ist_dt.time()
    if t < time(9, 30):  # Opening auction artifacts — skip first bar
        return False
    if t >= NO_NEW_TRADES_AFTER:
        return False
    return True


def seconds_to_force_close(dt: datetime | None = None) -> float:
    """Seconds until the forced position-close time (15:15 IST).

    Returns 0 if already past force-close time.
    Returns -1 if not a trading day.
    """
    if dt is None:
        dt = datetime.now(tz=IST)
    ist_dt = dt.astimezone(IST)
    if not is_trading_day(ist_dt.date()):
        return -1.0
    force_close_dt = ist_dt.replace(
        hour=FORCE_CLOSE_TIME.hour, minute=FORCE_CLOSE_TIME.minute, second=0, microsecond=0
    )
    delta = (force_close_dt - ist_dt).total_seconds()
    return max(0.0, delta)


def bar_start_time(ts: datetime, interval_minutes: int = 15) -> datetime:
    """Return the start of the time bar that *ts* falls into.

    Example with 15-min bars:
      09:17 → 09:15, 09:29 → 09:15, 09:31 → 09:30

    Args:
        ts: Input datetime (any timezone; converted to IST internally).
        interval_minutes: Bar size in minutes.

    Returns:
        IST datetime representing the bar start.

    Raises:
        OutsideTradingHoursError: If *ts* falls outside the NSE trading session
            (before SESSION_START or at/after SESSION_END).
    """
    ist_ts = ts.astimezone(IST)
    t = ist_ts.time()

    if t < SESSION_START:
        raise OutsideTradingHoursError(
            f"Timestamp {ist_ts.isoformat()} is before session start {SESSION_START} IST"
        )
    if t >= SESSION_END:
        raise OutsideTradingHoursError(
            f"Timestamp {ist_ts.isoformat()} is at or after session end {SESSION_END} IST"
        )

    minutes_since_session = (
        (ist_ts.hour - SESSION_START.hour) * 60
        + (ist_ts.minute - SESSION_START.minute)
    )
    bar_slot = (minutes_since_session // interval_minutes) * interval_minutes
    bar_minute = SESSION_START.minute + bar_slot
    bar_hour = SESSION_START.hour + bar_minute // 60
    bar_minute = bar_minute % 60
    return ist_ts.replace(hour=bar_hour, minute=bar_minute, second=0, microsecond=0)


def now_ist() -> datetime:
    """Current time in IST as timezone-aware datetime."""
    return datetime.now(tz=IST)


def today_ist() -> date:
    """Current date in IST."""
    return datetime.now(tz=IST).date()


def first_trading_day_of_month(d: date | None = None) -> date:
    """Return the first NSE trading day of the month containing *d*.

    Args:
        d: Any date within the target month. Defaults to today (IST).

    Returns:
        The first trading day of that calendar month.
    """
    if d is None:
        d = today_ist()
    candidate = d.replace(day=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def is_sip_day(d: date | None = None) -> bool:
    """Return True if *d* is the first NSE trading day of the month.

    The first trading day of each month sees 100M+ SIP inflow orders executed
    at market open, predictably pushing equity prices higher in the morning
    session. (Source: ArXiv 2507.04859, 22-year NSE study.)

    Trading rule:
      - Do NOT open new LONG positions on SIP day (buying into inflated prices).
      - Existing long positions: this is an ideal PLANNED EXIT window (good
        liquidity, prices bid up by SIP flow).
      - SHORT entries are neutral — SIP buy pressure fades by mid-session.

    Args:
        d: Date to check. Defaults to today (IST).

    Returns:
        True if *d* is the first trading day of its calendar month.
    """
    if d is None:
        d = today_ist()
    if not is_trading_day(d):
        return False
    return d == first_trading_day_of_month(d)


# ── F&O expiry helpers (#66) ─────────────────────────────────────────────────

def is_expiry_day(d: date, underlying: str = "NIFTY") -> bool:
    """Return True if *d* is an F&O expiry day for *underlying*.

    Delegates to ``algo_trading.fo.expiry.ExpiryCalendar`` when available.
    Falls back to a simple Tuesday check if the expiry module is not importable.

    Args:
        d: Date to check.
        underlying: Underlying symbol, e.g. "NIFTY" or "BANKNIFTY".

    Returns:
        True if *d* is a known expiry day.
    """
    try:
        from algo_trading.fo.expiry import get_expiry_calendar  # type: ignore[import]
        cal = get_expiry_calendar(underlying)
        return cal.is_expiry_day(d)
    except (ImportError, KeyError):
        log.warning(
            "algo_trading.fo.expiry not available; falling back to Tuesday expiry check",
            underlying=underlying,
        )
        # Fallback: Tuesday (weekday == 1) on a trading day
        return d.weekday() == 1 and is_trading_day(d)


def get_expiry_risk_multiplier(d: date, underlying: str = "NIFTY") -> float:
    """Return a position-size multiplier for *d* reflecting expiry-day gamma risk.

    Returns 0.5 on expiry day (half position size due to elevated gamma risk)
    and 1.0 on all other days.

    Args:
        d: Date to evaluate.
        underlying: Underlying symbol.

    Returns:
        0.5 on expiry day, 1.0 otherwise.
    """
    if is_expiry_day(d, underlying=underlying):
        return 0.5
    return 1.0


# ── Instrument master freshness (#67) ────────────────────────────────────────

def check_instrument_master_freshness(
    master_path: Path = Path("state/instrument_master.json"),
) -> bool:
    """Check whether the instrument master file is fresh enough to use.

    Returns True if the file exists and was modified within the last
    ``INSTRUMENT_MASTER_MAX_AGE_HOURS`` hours.  Returns False and logs a
    WARNING if the file is missing or older than the threshold.

    Args:
        master_path: Path to the instrument master JSON file.

    Returns:
        True if fresh, False if stale or missing.
    """
    if not master_path.exists():
        log.warning(
            "Instrument master file not found",
            path=str(master_path),
        )
        return False

    mtime = master_path.stat().st_mtime
    age_seconds = datetime.now().timestamp() - mtime
    age_hours = age_seconds / 3600

    if age_hours > INSTRUMENT_MASTER_MAX_AGE_HOURS:
        log.warning(
            "Instrument master is stale",
            path=str(master_path),
            age_hours=round(age_hours, 1),
            max_age_hours=INSTRUMENT_MASTER_MAX_AGE_HOURS,
        )
        return False

    return True


def validate_fo_prerequisites(underlying: str, expiry: date) -> None:
    """Validate prerequisites before executing F&O strategies.

    Checks instrument master freshness and that the expiry date has not
    already passed.

    Args:
        underlying: Underlying symbol (e.g. "NIFTY").
        expiry: Expiry date for the contract.

    Raises:
        InstrumentMasterStaleError: If the instrument master is missing or stale.
        ValueError: If *expiry* is in the past.
    """
    if not check_instrument_master_freshness():
        raise InstrumentMasterStaleError(
            f"Instrument master is stale or missing. "
            f"Refresh state/instrument_master.json before trading {underlying}."
        )

    today = today_ist()
    if expiry < today:
        raise ValueError(
            f"Expiry {expiry.isoformat()} is in the past (today is {today.isoformat()}). "
            f"Select a valid future expiry for {underlying}."
        )
