"""Date-aware lot-size registry for NSE F&O instruments.

NSE revises contract lot sizes periodically via SEBI-mandated reviews.
Backtests must use the lot size that was **in effect on the trade date**,
not the current live lot size.  This module maintains an append-only
history of all lot-size epochs, keyed by symbol and effective-from date.

Source conventions used in ``circular_ref`` fields
---------------------------------------------------
[P]  = Primary source — NSE circular cited directly.
[S]  = Secondary / derived — inferred from multiple NSE circulars or
       widely-published exchange data.

Verified lot-size epochs (as of 2026-06-19)
--------------------------------------------
NIFTY
  75 lots  until 2025-12-31  [P] NSE circular FAOP67267 (and prior circulars)
  65 lots  from  2026-01-01  [P] NSE circular FAOP70616

BANKNIFTY
  35 lots  until 2025-12-31  [P] NSE circular FAOP70616 preamble
  30 lots  from  2026-01-01  [P] NSE circular FAOP70616

To add a future revision: append a new ``LotSizeRecord`` to
``LotSizeHistory._DEFAULT_REGISTRY`` with the appropriate
``effective_from``, a ``None`` ``effective_to``, and the
correct ``circular_ref``.  Update the previous record's
``effective_to`` to the day before the new record's ``effective_from``.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LotSizeRecord:
    """Immutable record describing a single lot-size epoch for one symbol.

    Attributes:
        symbol: Canonical NSE F&O symbol (e.g. ``"NIFTY"``, ``"BANKNIFTY"``).
            Always stored and matched in upper-case.
        lot_size: Number of units per contract during this epoch.
        effective_from: First calendar date (inclusive) on which this
            lot size applies.
        effective_to: Last calendar date (inclusive) on which this lot size
            applies.  ``None`` means the record is still current and has no
            known end date.
        circular_ref: NSE circular reference number (e.g. ``"FAOP70616"``).
            Serves as a human-readable audit trail.  Prefix with ``[P]`` for
            primary (direct) sources and ``[S]`` for secondary / derived.
    """

    symbol: str
    lot_size: int
    effective_from: date
    effective_to: date | None
    circular_ref: str

    def covers(self, on_date: date) -> bool:
        """Return True if *on_date* falls within this record's epoch.

        Args:
            on_date: The calendar date to test.

        Returns:
            ``True`` if ``effective_from <= on_date <= effective_to``
            (or ``effective_to is None`` and ``effective_from <= on_date``).
        """
        if on_date < self.effective_from:
            return False
        if self.effective_to is not None and on_date > self.effective_to:
            return False
        return True


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class LotSizeNotFoundError(KeyError):
    """Raised when no ``LotSizeRecord`` covers the requested symbol/date.

    This typically means the requested date is *before* the earliest known
    record for that symbol.  Add historical records to
    ``LotSizeHistory._DEFAULT_REGISTRY`` to extend coverage.
    """

    def __init__(self, symbol: str, on_date: date) -> None:
        super().__init__(
            f"No lot-size record found for symbol={symbol!r} on date={on_date.isoformat()!r}. "
            "The date may be before the earliest known epoch for this symbol. "
            "Add a LotSizeRecord to LotSizeHistory._DEFAULT_REGISTRY to extend coverage."
        )
        self.symbol = symbol
        self.on_date = on_date


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LotSizeHistory:
    """Registry of NSE F&O lot-size epochs, queryable by symbol and date.

    The class ships with a verified default registry covering NIFTY and
    BANKNIFTY.  Additional symbols or future revisions can be registered
    via :meth:`register` at runtime, or by editing
    ``_DEFAULT_REGISTRY`` in this file.

    Example::

        history = LotSizeHistory()
        assert history.get_lot_size("NIFTY", date(2025, 6, 1)) == 75
        assert history.get_lot_size("NIFTY", date(2026, 1, 1)) == 65
        assert history.get_lot_size("BANKNIFTY", date(2025, 12, 31)) == 35
        assert history.get_lot_size("BANKNIFTY", date(2026, 1, 1)) == 30
    """

    # ------------------------------------------------------------------
    # Verified default registry
    # ------------------------------------------------------------------
    # Each tuple is (symbol, lot_size, effective_from, effective_to, circular_ref).
    # Sources:
    #   FAOP67267  — NSE circular revising NIFTY lot to 75 (and prior revisions)
    #   FAOP70616  — NSE circular effective 2026-01-01: NIFTY→65, BANKNIFTY→30
    #                The preamble of FAOP70616 confirms BANKNIFTY was 35 before.
    # ------------------------------------------------------------------
    _DEFAULT_REGISTRY: ClassVar[list[LotSizeRecord]] = [
        # --- NIFTY 50 -------------------------------------------------------
        # [P] NSE FAOP67267 (and prior circulars): lot size = 75
        #     Effective until the day before FAOP70616 kicks in.
        LotSizeRecord(
            symbol="NIFTY",
            lot_size=75,
            effective_from=date(2010, 1, 1),   # [S] conservative back-fill; 75-lot era
            effective_to=date(2025, 12, 31),
            circular_ref="[P] FAOP67267 and prior NSE circulars",
        ),
        # [P] NSE FAOP70616: NIFTY lot revised to 65 from 2026-01-01
        LotSizeRecord(
            symbol="NIFTY",
            lot_size=65,
            effective_from=date(2026, 1, 1),
            effective_to=None,                  # currently in effect
            circular_ref="[P] NSE FAOP70616",
        ),
        # --- BANK NIFTY -----------------------------------------------------
        # [P] NSE FAOP70616 preamble: BANKNIFTY was 35 before 2026-01-01
        LotSizeRecord(
            symbol="BANKNIFTY",
            lot_size=35,
            effective_from=date(2010, 1, 1),   # [S] conservative back-fill
            effective_to=date(2025, 12, 31),
            circular_ref="[P] NSE FAOP70616 preamble (prior lot = 35)",
        ),
        # [P] NSE FAOP70616: BANKNIFTY lot revised to 30 from 2026-01-01
        LotSizeRecord(
            symbol="BANKNIFTY",
            lot_size=30,
            effective_from=date(2026, 1, 1),
            effective_to=None,                  # currently in effect
            circular_ref="[P] NSE FAOP70616",
        ),
    ]

    def __init__(self) -> None:
        # Build a mutable copy so runtime registrations don't mutate the class var.
        self._records: list[LotSizeRecord] = list(self._DEFAULT_REGISTRY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, record: LotSizeRecord) -> None:
        """Add a new lot-size epoch record to this registry instance.

        Use this to register instruments not in the default registry, or
        to add future revisions at runtime (e.g. loaded from a database).

        Args:
            record: A fully-populated :class:`LotSizeRecord`.

        Raises:
            ValueError: If *record* overlaps with an existing record for
                the same symbol.
        """
        normalised = LotSizeRecord(
            symbol=record.symbol.upper(),
            lot_size=record.lot_size,
            effective_from=record.effective_from,
            effective_to=record.effective_to,
            circular_ref=record.circular_ref,
        )
        self._validate_no_overlap(normalised)
        self._records.append(normalised)
        log.debug(
            "lot_size_record_registered",
            symbol=normalised.symbol,
            lot_size=normalised.lot_size,
            effective_from=normalised.effective_from.isoformat(),
            effective_to=(
                normalised.effective_to.isoformat()
                if normalised.effective_to
                else "current"
            ),
        )

    def get_lot_size(self, symbol: str, on_date: date) -> int:
        """Return the correct lot size for *symbol* on *on_date*.

        Iterates over all registered records for the symbol and returns
        the ``lot_size`` from the record whose epoch covers *on_date*.

        Args:
            symbol: NSE F&O symbol (case-insensitive; e.g. ``"NIFTY"``).
            on_date: The trade or backtest date for which lot size is needed.

        Returns:
            The integer lot size in effect on *on_date*.

        Raises:
            LotSizeNotFoundError: If no registered record covers *on_date*
                for this symbol.  This is **not** a ``ValueError`` — it is a
                subclass of ``KeyError`` so callers can catch it specifically.
        """
        upper_sym = symbol.upper()
        for record in self._records:
            if record.symbol == upper_sym and record.covers(on_date):
                log.debug(
                    "lot_size_resolved",
                    symbol=upper_sym,
                    on_date=on_date.isoformat(),
                    lot_size=record.lot_size,
                    circular_ref=record.circular_ref,
                )
                return record.lot_size

        raise LotSizeNotFoundError(symbol=upper_sym, on_date=on_date)

    def all_records_for(self, symbol: str) -> list[LotSizeRecord]:
        """Return all registered records for *symbol*, sorted by effective_from.

        Args:
            symbol: NSE F&O symbol (case-insensitive).

        Returns:
            List of :class:`LotSizeRecord` objects in chronological order.
            Empty list if symbol has no registered records.
        """
        upper_sym = symbol.upper()
        return sorted(
            (r for r in self._records if r.symbol == upper_sym),
            key=lambda r: r.effective_from,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_no_overlap(self, new_record: LotSizeRecord) -> None:
        """Raise ValueError if *new_record* overlaps an existing record.

        Args:
            new_record: The candidate record to check.

        Raises:
            ValueError: If the candidate epoch overlaps with an existing
                registered record for the same symbol.
        """
        for existing in self._records:
            if existing.symbol != new_record.symbol:
                continue
            # Check if date ranges overlap.
            # Two ranges [a_from, a_to] and [b_from, b_to] overlap iff
            #   a_from <= b_to  AND  b_from <= a_to
            # Use date.max as a proxy for open-ended (None) effective_to.
            a_from = existing.effective_from
            a_to = existing.effective_to or date.max
            b_from = new_record.effective_from
            b_to = new_record.effective_to or date.max

            if a_from <= b_to and b_from <= a_to:
                raise ValueError(
                    f"New LotSizeRecord for {new_record.symbol!r} "
                    f"({b_from} – {new_record.effective_to or 'current'}) "
                    f"overlaps existing record "
                    f"({a_from} – {existing.effective_to or 'current'})."
                )


# ---------------------------------------------------------------------------
# Module-level singleton and convenience function
# ---------------------------------------------------------------------------

#: Module-level default registry instance.  Import and call ``get_lot_size``
#: for the common case; use ``LotSizeHistory()`` when you need a private
#: instance with custom records.
_DEFAULT_HISTORY: LotSizeHistory = LotSizeHistory()


def get_lot_size(symbol: str, on_date: date) -> int:
    """Return the correct lot size for *symbol* on *on_date*.

    This is a thin convenience wrapper around the module-level
    :class:`LotSizeHistory` singleton.  For the vast majority of callers
    this is the only function they need.

    Args:
        symbol: NSE F&O symbol (case-insensitive; e.g. ``"NIFTY"``).
        on_date: The trade or backtest date.

    Returns:
        Integer lot size in effect on *on_date*.

    Raises:
        LotSizeNotFoundError: If no registered record covers the date.

    Examples::

        >>> from datetime import date
        >>> get_lot_size("NIFTY", date(2025, 6, 1))
        75
        >>> get_lot_size("NIFTY", date(2026, 1, 1))
        65
        >>> get_lot_size("BANKNIFTY", date(2025, 12, 31))
        35
        >>> get_lot_size("BANKNIFTY", date(2026, 6, 19))
        30
    """
    return _DEFAULT_HISTORY.get_lot_size(symbol, on_date)
