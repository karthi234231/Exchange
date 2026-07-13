"""
signals/event_filter.py — NSE event-based position size multiplier.

Reduces or eliminates position sizes around scheduled high-volatility events
that make directional signals unreliable:
  - NSE F&O expiry (last Tuesday of every month): 50% size reduction  [P0-FIX-6]
  - NSE weekly options expiry (every Tuesday): 30% size reduction     [P0-FIX-6]
  - High India VIX (> 22): 40% size reduction
  - Very high India VIX (> 30): 80% size reduction
  - Results blackout: symbols with earnings ±1 day: 0% (skip entirely)
  - Budget / RBI policy day: manually set via blackout list

Expiry rules sourced from algo_trading.fo.expiry (NSE Circular FAOP68747):
  - Weekly expiry: every Tuesday (Nifty; changed from Thursday Sep 2025)
  - Monthly expiry: last Tuesday of month (holiday-adjusted)

Design:
  - EventCalendarFilter.get_size_multiplier(symbol, trade_date, india_vix) -> float [0, 1]
  - EventCalendarFilter.add_blackout(symbol, date_str, reason) — for results/corporate events
  - EventCalendarFilter.load_blackout_csv(path) — bulk load from CSV with columns: symbol, date, reason
  - is_nse_expiry_day(date) — True for last Tuesday of month (monthly expiry) [P0-FIX-6]
  - is_weekly_expiry_day(date) — True for any F&O expiry Tuesday            [P0-FIX-6]
  - EventCalendarFilter.get_event_description(date, symbol) -> str — human-readable reason for any size reduction
  - Thread-safe (read-heavy, writes rare)

India VIX fetching is handled externally — this class accepts india_vix as a float parameter.
If india_vix is None: skip VIX-based filtering (safe default).
"""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import structlog

from algo_trading.fo.expiry import nifty_expiry_calendar

log = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# VIX thresholds
_VIX_HIGH_THRESHOLD: float = 22.0
_VIX_VERY_HIGH_THRESHOLD: float = 30.0

# Size multipliers
_MULTIPLIER_MONTHLY_EXPIRY: float = 0.50   # F&O monthly expiry
_MULTIPLIER_WEEKLY_EXPIRY: float = 0.70    # Weekly expiry
_MULTIPLIER_VIX_HIGH: float = 0.60         # VIX > 22
_MULTIPLIER_VIX_VERY_HIGH: float = 0.20    # VIX > 30
_MULTIPLIER_BLACKOUT: float = 0.0          # Results / corporate action blackout

# Blackout covers earnings day ± this many calendar days
_RESULTS_WINDOW_DAYS: int = 1


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlackoutEntry:
    """A single blackout entry for a symbol on a specific date."""
    symbol: str        # "RELIANCE", or "__ALL__" for market-wide blackout
    date: date
    reason: str


# ── Pure calendar helpers (module-level, no class required) ──────────────────

def is_weekly_expiry_day(d: date) -> bool:
    """
    Return True if `d` is an NSE F&O expiry day (Nifty weekly or monthly).

    Delegates to the centralized expiry calendar so rule changes (e.g. the
    Thursday→Tuesday shift per NSE Circular FAOP68747, effective Sep 2025)
    propagate automatically. Holiday-adjusted dates are handled inside the
    calendar — no separate holiday check is needed here. [P0-FIX-6]
    """
    return nifty_expiry_calendar.is_expiry_day(d)


def is_nse_expiry_day(d: date) -> bool:
    """
    Return True if `d` is the NSE monthly F&O expiry (last Tuesday of month,
    holiday-adjusted).

    Delegates to the centralized expiry calendar (NSE Circular FAOP68747). [P0-FIX-6]
    """
    return d == nifty_expiry_calendar.monthly_expiry(d.year, d.month)


# ── Main class ────────────────────────────────────────────────────────────────

class EventCalendarFilter:
    """
    NSE event-based position size multiplier.

    Thread-safe: a RWLock pattern is approximated using a standard threading.Lock.
    Writes (add_blackout, load_blackout_csv) are rare; reads (get_size_multiplier)
    are frequent. A simple lock is sufficient — contention is negligible.

    Usage:
        ecf = EventCalendarFilter()
        ecf.load_blackout_csv("data/results_calendar.csv")
        multiplier = ecf.get_size_multiplier("RELIANCE", date.today(), india_vix=18.5)
        # multiplier ∈ [0.0, 1.0] — multiply computed position size by this
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Key: (symbol_or_ALL, date) → BlackoutEntry
        self._blackouts: dict[tuple[str, date], BlackoutEntry] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_size_multiplier(
        self,
        symbol: str,
        trade_date: date,
        india_vix: Optional[float] = None,
    ) -> float:
        """
        Compute position size multiplier for (symbol, trade_date).

        Returns a float in [0.0, 1.0]. Multiply the nominal position size
        by this value before sending to the broker.

        Rules applied in priority order (most restrictive wins):
          1. Symbol-specific blackout (results ±1 day)          → 0.0
          2. Market-wide blackout (Budget, RBI policy)          → 0.0
          3. Very high India VIX (> 30)                         → 0.2
          4. High India VIX (> 22)                              → 0.6
          5. NSE monthly F&O expiry (last Tuesday)              → 0.5
          6. NSE weekly expiry (any Tuesday)                    → 0.7
          7. No event                                           → 1.0

        Args:
            symbol:     NSE equity symbol (e.g. "RELIANCE").
            trade_date: The date of the prospective trade.
            india_vix:  Current India VIX value (float). Pass None to skip VIX check.

        Returns:
            Position size multiplier ∈ [0.0, 1.0].
        """
        # Rule 1+2: Blackout check (covers results ±1 day and market-wide events)
        if self._is_in_blackout(symbol, trade_date):
            return _MULTIPLIER_BLACKOUT

        # Rule 3+4: VIX-based reduction
        if india_vix is not None:
            if india_vix > _VIX_VERY_HIGH_THRESHOLD:
                return _MULTIPLIER_VIX_VERY_HIGH
            if india_vix > _VIX_HIGH_THRESHOLD:
                return _MULTIPLIER_VIX_HIGH

        # Rule 5: Monthly F&O expiry (last Tuesday) — more severe than weekly
        if is_nse_expiry_day(trade_date):
            return _MULTIPLIER_MONTHLY_EXPIRY

        # Rule 6: Weekly expiry (any Tuesday that is NOT monthly)
        if is_weekly_expiry_day(trade_date):
            return _MULTIPLIER_WEEKLY_EXPIRY

        return 1.0

    def add_blackout(
        self,
        symbol: str,
        date_str: str,
        reason: str = "",
    ) -> None:
        """
        Add a single blackout entry for a symbol on a specific date.

        The blackout covers [date - RESULTS_WINDOW_DAYS, date + RESULTS_WINDOW_DAYS]
        inclusive (i.e., earnings ±1 calendar day).

        Args:
            symbol:   NSE symbol (e.g. "RELIANCE") or "__ALL__" for market-wide.
            date_str: ISO date string "YYYY-MM-DD".
            reason:   Human-readable reason (e.g., "Q3 earnings release").

        Raises:
            ValueError: If date_str is not a valid ISO date.
        """
        centre = date.fromisoformat(date_str)
        entries: list[BlackoutEntry] = []
        for delta in range(-_RESULTS_WINDOW_DAYS, _RESULTS_WINDOW_DAYS + 1):
            d = centre + timedelta(days=delta)
            entries.append(BlackoutEntry(symbol=symbol, date=d, reason=reason))

        with self._lock:
            for entry in entries:
                self._blackouts[(entry.symbol, entry.date)] = entry

        log.info(
            "event_filter.blackout_added",
            symbol=symbol,
            centre_date=str(centre),
            window_days=_RESULTS_WINDOW_DAYS,
            reason=reason,
        )

    def add_blackout_exact(
        self,
        symbol: str,
        date_str: str,
        reason: str = "",
    ) -> None:
        """
        Add a blackout for exactly one date (no ±1 window).
        Useful for market-wide events like Budget day or RBI policy.
        """
        d = date.fromisoformat(date_str)
        entry = BlackoutEntry(symbol=symbol, date=d, reason=reason)
        with self._lock:
            self._blackouts[(symbol, d)] = entry
        log.info(
            "event_filter.blackout_exact_added",
            symbol=symbol,
            date=str(d),
            reason=reason,
        )

    def load_blackout_csv(self, path: str | Path) -> int:
        """
        Bulk-load blackout entries from a CSV file.

        CSV format (with header row):
            symbol,date,reason
            RELIANCE,2026-01-15,Q3 earnings
            __ALL__,2026-02-01,Union Budget

        Lines starting with '#' and blank lines are ignored.
        Each entry is expanded by ±1 day (same as add_blackout).

        Args:
            path: Path to the CSV file.

        Returns:
            Number of centre-date entries successfully loaded.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError:        If a row has malformed data.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Blackout CSV not found: {csv_path}")

        count = 0
        errors: list[str] = []

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for line_num, row in enumerate(reader, start=2):
                # Skip comment lines and blank rows
                symbol = (row.get("symbol") or "").strip()
                date_str = (row.get("date") or "").strip()
                reason = (row.get("reason") or "").strip()

                if not symbol or symbol.startswith("#"):
                    continue
                if not date_str:
                    errors.append(f"Line {line_num}: missing date for symbol {symbol!r}")
                    continue

                try:
                    self.add_blackout(symbol, date_str, reason)
                    count += 1
                except ValueError as exc:
                    errors.append(f"Line {line_num}: {exc}")

        if errors:
            log.warning(
                "event_filter.load_csv_errors",
                path=str(csv_path),
                error_count=len(errors),
                first_error=errors[0],
            )

        log.info(
            "event_filter.blackout_csv_loaded",
            path=str(csv_path),
            entries_loaded=count,
        )
        return count

    def get_event_description(
        self,
        trade_date: date,
        symbol: str = "",
    ) -> str:
        """
        Return a human-readable description of why position sizing is reduced
        (or empty string if no reduction applies).

        Args:
            trade_date: The date to check.
            symbol:     Optional symbol for symbol-specific check.

        Returns:
            Descriptive string, e.g. "NSE monthly F&O expiry" or "" if no event.
        """
        parts: list[str] = []

        # Blackout
        if symbol and self._is_symbol_blackout(symbol, trade_date):
            entry = self._get_blackout_entry(symbol, trade_date)
            parts.append(
                f"Symbol blackout: {entry.reason}" if entry and entry.reason
                else "Symbol blackout (results/corporate event)"
            )
        elif self._is_market_wide_blackout(trade_date):
            entry = self._get_blackout_entry("__ALL__", trade_date)
            parts.append(
                f"Market-wide blackout: {entry.reason}" if entry and entry.reason
                else "Market-wide blackout"
            )

        # Calendar events (independent of blackout)
        if is_nse_expiry_day(trade_date):
            parts.append("NSE monthly F&O expiry (last Tuesday)")  # [P0-FIX-6]
        elif is_weekly_expiry_day(trade_date):
            parts.append("NSE weekly options expiry (Tuesday)")    # [P0-FIX-6]

        return "; ".join(parts) if parts else ""

    def remove_blackout(self, symbol: str, date_str: str) -> None:
        """
        Remove the blackout entry for (symbol, date) if it exists.
        Does NOT remove ±1 window — only the exact date provided.
        """
        d = date.fromisoformat(date_str)
        with self._lock:
            self._blackouts.pop((symbol, d), None)

    def list_blackouts(self, symbol: Optional[str] = None) -> list[BlackoutEntry]:
        """
        Return all blackout entries, optionally filtered by symbol.
        Sorted by date ascending.
        """
        with self._lock:
            entries = list(self._blackouts.values())
        if symbol is not None:
            entries = [e for e in entries if e.symbol == symbol]
        return sorted(entries, key=lambda e: e.date)

    def clear_blackouts(self) -> None:
        """Remove all blackout entries."""
        with self._lock:
            self._blackouts.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_in_blackout(self, symbol: str, d: date) -> bool:
        """True if the date is blacked out for this symbol OR market-wide."""
        with self._lock:
            return (
                (symbol, d) in self._blackouts
                or ("__ALL__", d) in self._blackouts
            )

    def _is_symbol_blackout(self, symbol: str, d: date) -> bool:
        with self._lock:
            return (symbol, d) in self._blackouts

    def _is_market_wide_blackout(self, d: date) -> bool:
        with self._lock:
            return ("__ALL__", d) in self._blackouts

    def _get_blackout_entry(
        self, symbol: str, d: date
    ) -> Optional[BlackoutEntry]:
        with self._lock:
            return self._blackouts.get((symbol, d))
