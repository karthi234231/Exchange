"""
algo_trading/market_data/chain_normalizer.py — Raw chain data normalisation.

Handles naming differences, missing strikes, crossed quotes, and negative IV.
Every correction is logged to an immutable audit trail for research review.

Key rules (design Section 9.2)
--------------------------------
- Reject quotes where bid > ask (crossed/locked market).
- Reject quotes where IV < 0 (excluding the sentinel -1.0 = unavailable).
- Log EVERY correction made — no silent mutation.
- NEVER use corrected data for strategy signals; only for display.
- If > 5% of expected strikes are missing → mark snapshot as ``"UNUSABLE"``
  for backtesting.
- Missing strikes → ``data_quality = "PARTIAL"``; interpolate only for display.

Correction types
----------------
- ``"CROSSED_QUOTE"``     : bid > ask — quote removed from snapshot.
- ``"NEGATIVE_IV"``       : iv < 0 and iv != -1.0 — iv set to -1.0 (unavailable).
- ``"MISSING_STRIKE"``    : expected strike not in snapshot — logged but not added.
- ``"ZERO_VOLUME_FLAG"``  : volume and OI both zero on a non-zero-ltp quote — flagged.

Audit guarantee
---------------
Every ``NormalizationCorrection`` is appended to ``self._corrections`` and is
never removed.  Use ``get_corrections(since=...)`` to retrieve the full or
partial audit log.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from algo_trading.market_data.chain_snapshot import (
    CURRENT_SCHEMA_VERSION,
    ChainSnapshot,
    OptionQuote,
)

log = structlog.get_logger(__name__)

# Fraction of missing strikes above which a snapshot is UNUSABLE.
_UNUSABLE_MISSING_THRESHOLD: float = 0.05   # 5 %


@dataclass
class NormalizationCorrection:
    """Immutable record of a single correction made during normalisation.

    Appended to the audit log; never mutated after creation.

    Attributes:
        timestamp: Wall-clock UTC time when the correction was made.
        correction_type: One of ``"CROSSED_QUOTE"``, ``"NEGATIVE_IV"``,
            ``"MISSING_STRIKE"``, ``"ZERO_VOLUME_FLAG"``.
        original_value: The raw value before correction.
        corrected_value: The value after correction (or ``None`` if the
            quote was removed entirely).
        description: Human-readable explanation including symbol, expiry,
            strike, and option type.
    """

    timestamp: datetime
    correction_type: str
    original_value: Any
    corrected_value: Any
    description: str


class ChainNormalizer:
    """Normalises raw ``ChainSnapshot`` objects before storage.

    The normalizer is a pure function from raw snapshot to normalised snapshot.
    It does NOT persist snapshots — that is the archiver's job.

    Usage::

        normalizer = ChainNormalizer()
        clean_snapshot = normalizer.normalize(raw_snapshot)

        # Retrieve full audit log:
        corrections = normalizer.get_corrections()

        # Retrieve corrections since last hour:
        recent = normalizer.get_corrections(since=datetime.now(utc) - timedelta(hours=1))

    Parameters
    ----------
    strike_generator:
        Optional callable ``(symbol, expiry) -> list[float]`` that returns the
        expected list of strikes from the contract master.  When provided,
        missing-strike detection is enabled.  When ``None``, only present
        quotes are inspected (no missing-strike check).
    log_corrections:
        If ``True`` (default), append each correction to the audit log and emit
        a structlog WARNING.  Set to ``False`` only in unit tests where the
        audit trail is not required.
    """

    def __init__(
        self,
        strike_generator: Callable[[str, Any], list[float]] | None = None,
        log_corrections: bool = True,
    ) -> None:
        self._strike_generator = strike_generator
        self._log_corrections = log_corrections
        self._corrections: list[NormalizationCorrection] = []

        log.info(
            "chain_normalizer.init",
            has_strike_generator=strike_generator is not None,
            log_corrections=log_corrections,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, snapshot: ChainSnapshot) -> ChainSnapshot:
        """Return a normalised copy of *snapshot* with ``data_quality`` set.

        Steps performed in order:
            1. Deep-copy the incoming quotes dict to avoid mutating the caller's
               data.
            2. For each quote: check bid/ask spread and IV sanity.
            3. Remove crossed/locked quotes; set IV to -1.0 for negative IVs.
            4. If a strike_generator is configured: detect missing strikes.
            5. Assess overall completeness and set ``data_quality``.

        NEVER use the corrected/normalised snapshot for strategy signals.
        Corrections fix display issues; they do not recover market truth.

        Args:
            snapshot: Raw ``ChainSnapshot`` from the feed or archiver.

        Returns:
            New ``ChainSnapshot`` with ``data_quality`` set and invalid quotes
            removed or repaired.  The original snapshot is not modified.
        """
        normalised_quotes = copy.copy(snapshot.quotes)  # shallow copy of dict; values immutable-ish

        # Pass 1: spread and IV sanity.
        keys_to_remove: list[tuple] = []
        for key, quote in normalised_quotes.items():
            expiry, strike, opt_type = key

            if not self._check_spread_sanity(quote):
                self._record_correction(
                    correction_type="CROSSED_QUOTE",
                    original_value=(quote.bid, quote.ask),
                    corrected_value=None,
                    description=(
                        f"{snapshot.symbol} {expiry} {strike} {opt_type}: "
                        f"bid={quote.bid} > ask={quote.ask}; quote removed"
                    ),
                )
                keys_to_remove.append(key)
                continue

            if not self._check_iv_sanity(quote):
                # Replace the quote with IV set to -1.0 (unavailable sentinel).
                fixed_quote = _replace_iv(quote, -1.0)
                normalised_quotes[key] = fixed_quote
                self._record_correction(
                    correction_type="NEGATIVE_IV",
                    original_value=quote.iv,
                    corrected_value=-1.0,
                    description=(
                        f"{snapshot.symbol} {expiry} {strike} {opt_type}: "
                        f"iv={quote.iv} is negative (not -1.0 sentinel); reset to -1.0"
                    ),
                )

            # Zero-volume flag: non-zero LTP but zero OI and volume.
            if quote.ltp > 0 and quote.oi == 0 and quote.volume == 0:
                self._record_correction(
                    correction_type="ZERO_VOLUME_FLAG",
                    original_value={"oi": quote.oi, "volume": quote.volume, "ltp": quote.ltp},
                    corrected_value=None,  # flagged only, not changed
                    description=(
                        f"{snapshot.symbol} {expiry} {strike} {opt_type}: "
                        f"ltp={quote.ltp} but oi=0 and volume=0; flagged for review"
                    ),
                )

        for key in keys_to_remove:
            del normalised_quotes[key]

        # Pass 2: missing strikes (if strike_generator is available).
        if self._strike_generator is not None:
            for expiry in snapshot.all_expiries():
                expected_strikes = self._strike_generator(snapshot.symbol, expiry)
                for strike in expected_strikes:
                    if (
                        (expiry, strike, "CE") not in normalised_quotes
                        and (expiry, strike, "PE") not in normalised_quotes
                    ):
                        self._record_correction(
                            correction_type="MISSING_STRIKE",
                            original_value=None,
                            corrected_value=None,
                            description=(
                                f"{snapshot.symbol} {expiry} {strike}: "
                                f"expected strike absent from snapshot; "
                                f"not interpolated (only display interpolation allowed)"
                            ),
                        )

        # Assess completeness and set data_quality.
        data_quality = self._assess_completeness_from_quotes(
            snapshot, normalised_quotes
        )

        normalised = ChainSnapshot(
            symbol=snapshot.symbol,
            snapshot_timestamp=snapshot.snapshot_timestamp,
            event_timestamp=snapshot.event_timestamp,
            underlying_price=snapshot.underlying_price,
            underlying_bid=snapshot.underlying_bid,
            underlying_ask=snapshot.underlying_ask,
            schema_version=CURRENT_SCHEMA_VERSION,
            sequence_number=snapshot.sequence_number,
            data_quality=data_quality,
            quotes=normalised_quotes,
            metadata=snapshot.metadata,
        )

        log.debug(
            "chain_normalizer.normalized",
            symbol=snapshot.symbol,
            original_quote_count=len(snapshot.quotes),
            normalised_quote_count=len(normalised_quotes),
            removed=len(keys_to_remove),
            data_quality=data_quality,
        )
        return normalised

    def get_corrections(
        self,
        since: datetime | None = None,
    ) -> list[NormalizationCorrection]:
        """Return the audit log of all corrections made.

        Args:
            since: If provided, return only corrections at or after this
                timestamp.  Timezone-naive values are assumed UTC.

        Returns:
            List of ``NormalizationCorrection`` objects in insertion order.
        """
        if since is None:
            return list(self._corrections)

        since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        return [
            c
            for c in self._corrections
            if (
                c.timestamp
                if c.timestamp.tzinfo
                else c.timestamp.replace(tzinfo=timezone.utc)
            )
            >= since_utc
        ]

    # ------------------------------------------------------------------
    # Sanity predicates
    # ------------------------------------------------------------------

    def _check_spread_sanity(self, quote: OptionQuote) -> bool:
        """Return ``True`` if bid <= ask or both sides are zero (no depth).

        Args:
            quote: ``OptionQuote`` to inspect.

        Returns:
            ``False`` for a crossed/locked market (bid > ask when both non-zero).
        """
        if quote.bid == 0.0 and quote.ask == 0.0:
            return True  # no depth data — not a crossed quote
        return quote.bid <= quote.ask

    def _check_iv_sanity(self, quote: OptionQuote) -> bool:
        """Return ``True`` if IV is non-negative or the unavailable sentinel (-1.0).

        A value of -1.0 means "not computed" and is explicitly allowed.
        Any other negative IV is a data error.

        Args:
            quote: ``OptionQuote`` to inspect.

        Returns:
            ``False`` when ``iv < 0 and iv != -1.0``.
        """
        if quote.iv == -1.0:
            return True  # sentinel: IV not available
        return quote.iv >= 0.0

    # ------------------------------------------------------------------
    # Completeness assessment
    # ------------------------------------------------------------------

    def _assess_completeness(self, snapshot: ChainSnapshot) -> str:
        """Return ``"COMPLETE"``, ``"PARTIAL"``, or ``"UNUSABLE"`` for *snapshot*.

        Delegates to the internal helper that also accepts a pre-filtered
        quotes dict (used by ``normalize`` so completeness reflects removed quotes).

        Args:
            snapshot: Snapshot to assess.

        Returns:
            Data quality string.
        """
        return self._assess_completeness_from_quotes(snapshot, snapshot.quotes)

    def _assess_completeness_from_quotes(
        self,
        snapshot: ChainSnapshot,
        quotes: dict,
    ) -> str:
        """Assess completeness given a specific quotes dict.

        When no ``strike_generator`` is configured, a snapshot with at
        least one quote is ``"COMPLETE"`` (we have no expectation to compare
        against); an empty snapshot is ``"UNUSABLE"``.

        When a ``strike_generator`` is configured:
            - missing_fraction = missing_strikes / total_expected_strikes
            - UNUSABLE  : missing_fraction > 5 %
            - PARTIAL   : 0 < missing_fraction <= 5 %
            - COMPLETE  : missing_fraction == 0

        Args:
            snapshot: Original snapshot (used for symbol and expiry list).
            quotes: Normalised quotes dict to measure against.

        Returns:
            ``"COMPLETE"``, ``"PARTIAL"``, or ``"UNUSABLE"``.
        """
        if not quotes:
            return "UNUSABLE"

        if self._strike_generator is None:
            return "COMPLETE"

        total_expected = 0
        total_missing = 0

        for expiry in snapshot.all_expiries():
            expected_strikes = self._strike_generator(snapshot.symbol, expiry)
            for strike in expected_strikes:
                total_expected += 1
                if (
                    (expiry, strike, "CE") not in quotes
                    and (expiry, strike, "PE") not in quotes
                ):
                    total_missing += 1

        if total_expected == 0:
            return "COMPLETE"

        missing_fraction = total_missing / total_expected
        if missing_fraction > _UNUSABLE_MISSING_THRESHOLD:
            return "UNUSABLE"
        if missing_fraction > 0:
            return "PARTIAL"
        return "COMPLETE"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_correction(
        self,
        correction_type: str,
        original_value: Any,
        corrected_value: Any,
        description: str,
    ) -> None:
        """Append a correction to the audit log and optionally log it.

        Args:
            correction_type: Category string (e.g. ``"CROSSED_QUOTE"``).
            original_value: Pre-correction value.
            corrected_value: Post-correction value; ``None`` if removed.
            description: Human-readable explanation.
        """
        correction = NormalizationCorrection(
            timestamp=datetime.now(tz=timezone.utc),
            correction_type=correction_type,
            original_value=original_value,
            corrected_value=corrected_value,
            description=description,
        )
        self._corrections.append(correction)

        if self._log_corrections:
            log.warning(
                "chain_normalizer.correction",
                correction_type=correction_type,
                description=description,
                original_value=str(original_value),
                corrected_value=str(corrected_value),
            )


def _replace_iv(quote: OptionQuote, new_iv: float) -> OptionQuote:
    """Return a new ``OptionQuote`` identical to *quote* but with iv=*new_iv*.

    ``OptionQuote`` is a mutable dataclass so we construct a replacement
    manually rather than using ``copy.replace`` (Python 3.13+) to stay
    compatible with Python 3.11+.

    Args:
        quote: Original quote.
        new_iv: Replacement IV value.

    Returns:
        New ``OptionQuote`` with iv set to *new_iv*; all other fields copied.
    """
    return OptionQuote(
        symbol=quote.symbol,
        expiry=quote.expiry,
        strike=quote.strike,
        option_type=quote.option_type,
        bid=quote.bid,
        ask=quote.ask,
        ltp=quote.ltp,
        iv=new_iv,
        oi=quote.oi,
        volume=quote.volume,
        delta=quote.delta,
        gamma=quote.gamma,
        vega=quote.vega,
        theta=quote.theta,
        quote_timestamp=quote.quote_timestamp,
    )
