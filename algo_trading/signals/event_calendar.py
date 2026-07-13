"""
signals/event_calendar.py — Re-export shim.

The canonical EventCalendarFilter implementation lives in
algo_trading.signals.event_filter. This module is a compatibility shim
that re-exports everything so any code that imports from event_calendar
continues to work.

Prefer importing from algo_trading.signals.event_filter directly.
"""
from algo_trading.signals.event_filter import (  # noqa: F401
    EventCalendarFilter,
    BlackoutEntry,
    is_nse_expiry_day,
    is_weekly_expiry_day,
)

__all__ = [
    "EventCalendarFilter",
    "BlackoutEntry",
    "is_nse_expiry_day",
    "is_weekly_expiry_day",
]
