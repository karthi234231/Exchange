"""
algo_trading/market_data/chain_snapshot.py — Point-in-time options chain snapshot.

Captures ALL strikes, all expiries, bid/ask/IV/OI for a single timestamp.

Replay clock semantics (design Section 9.2)
-------------------------------------------
- ``snapshot_timestamp``  : wall-clock time when the snapshot was taken.
                            Use for live trading decisions.
- ``event_timestamp``     : exchange timestamp on the raw feed.
                            USE THIS for backtesting, not wall clock.
                            Ingestion latency creates false causality if
                            wall-clock time is used during replay.
- ``sequence_number``     : monotonic counter for ordering snapshots that
                            share the same timestamp (e.g. rapid bursts).

All replay must use event time, not ingestion time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

CURRENT_SCHEMA_VERSION: int = 1
"""Bump when the snapshot schema changes in a backward-incompatible way."""


@dataclass
class OptionQuote:
    """Single NSE F&O option contract quote.

    All prices are in INR.  IV is annualised decimal (0.15 = 15 %).
    OI and volume are in lots (not individual contracts).

    Attributes:
        symbol: Underlying symbol (e.g. ``"NIFTY"``).
        expiry: Option expiry date (IST).
        strike: Strike price in INR.
        option_type: ``"CE"`` (call) or ``"PE"`` (put).
        bid: Best bid price.  0.0 if not available.
        ask: Best ask price.  0.0 if not available.
        ltp: Last traded price.
        iv: Implied volatility, annualised decimal.  -1.0 if unavailable.
        oi: Open interest in lots.
        volume: Day volume in lots.
        delta: Delta from pricing engine; ``None`` if not computed.
        gamma: Gamma from pricing engine; ``None`` if not computed.
        vega: Vega from pricing engine; ``None`` if not computed.
        theta: Theta from pricing engine; ``None`` if not computed.
        quote_timestamp: Exchange timestamp on the quote (use for replay).
    """

    symbol: str
    expiry: date
    strike: float
    option_type: str          # "CE" or "PE"
    bid: float
    ask: float
    ltp: float
    iv: float                 # annualised decimal; -1.0 = unavailable
    oi: int
    volume: int
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    quote_timestamp: datetime  # exchange timestamp — use for replay (see module docstring)

    def is_valid(self) -> bool:
        """Return ``True`` if the quote passes basic sanity checks.

        Checks:
            - ``bid >= 0``
            - ``ask >= 0``
            - ``bid <= ask``  (no crossed/locked market)
            - ``iv >= 0``  when IV is available (iv != -1.0)
            - ``ltp >= 0``
            - ``option_type`` is ``"CE"`` or ``"PE"``

        Returns:
            ``False`` on any violation; ``True`` otherwise.
        """
        if self.bid < 0 or self.ask < 0 or self.ltp < 0:
            return False
        if self.bid > self.ask and not (self.bid == 0.0 and self.ask == 0.0):
            return False
        if self.iv != -1.0 and self.iv < 0:
            return False
        if self.option_type not in ("CE", "PE"):
            return False
        return True

    def mid_price(self) -> float:
        """Return the mid-point of the bid/ask spread.

        Falls back to ``ltp`` when bid and ask are both zero (no market depth).

        Returns:
            ``(bid + ask) / 2`` when both sides are non-zero, else ``ltp``.
        """
        if self.bid == 0.0 and self.ask == 0.0:
            return self.ltp
        return (self.bid + self.ask) / 2.0


@dataclass
class ChainSnapshot:
    """Full NSE options chain at a single point in time.

    Keys in ``quotes`` are ``(expiry, strike, option_type)`` tuples so
    individual contracts can be retrieved in O(1).

    Replay clock semantics (design Section 9.2)
    -------------------------------------------
    Use ``event_timestamp`` (not ``snapshot_timestamp``) for all backtesting
    and replay logic.  Ingestion latency introduces false causality when wall-
    clock time is used during historical replay.

    Attributes:
        symbol: Underlying symbol (e.g. ``"NIFTY"``).
        snapshot_timestamp: Wall-clock time when this snapshot was captured.
        event_timestamp: Exchange timestamp; ``None`` if feed does not provide
            it.  Must be used for backtesting instead of ``snapshot_timestamp``.
        underlying_price: Last traded price of the underlying.
        underlying_bid: Best bid for the underlying; ``None`` if unavailable.
        underlying_ask: Best ask for the underlying; ``None`` if unavailable.
        schema_version: Schema version for forward-compatibility checks.
            Bump ``CURRENT_SCHEMA_VERSION`` when fields change.
        sequence_number: Monotonic counter.  Two snapshots for the same symbol
            with the same timestamp are ordered by this field.
        data_quality: ``"COMPLETE"``, ``"PARTIAL"``, or ``"UNUSABLE"``.
            Set by ``ChainNormalizer`` after validation.
        quotes: Map from ``(expiry, strike, option_type)`` to ``OptionQuote``.
        metadata: Free-form dict for archiver metadata, vendor info, etc.
    """

    symbol: str
    snapshot_timestamp: datetime    # wall clock — do NOT use for replay
    event_timestamp: datetime | None  # exchange timestamp — USE for replay
    underlying_price: float
    underlying_bid: float | None
    underlying_ask: float | None
    schema_version: int
    sequence_number: int
    data_quality: str               # "COMPLETE" | "PARTIAL" | "UNUSABLE"
    quotes: dict[tuple[date, float, str], OptionQuote]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_quote(
        self,
        expiry: date,
        strike: float,
        option_type: str,
    ) -> OptionQuote | None:
        """Return the quote for a specific contract, or ``None``.

        Args:
            expiry: Option expiry date.
            strike: Strike price in INR.
            option_type: ``"CE"`` or ``"PE"``.

        Returns:
            ``OptionQuote`` if found, ``None`` otherwise.
        """
        return self.quotes.get((expiry, strike, option_type))

    def completeness_pct(
        self,
        expected_strikes: list[float],
        expiry: date,
    ) -> float:
        """Percentage of expected strikes present for a given expiry.

        Args:
            expected_strikes: Full list of strikes expected from the contract
                master for this expiry.
            expiry: Expiry date to check against.

        Returns:
            Float in [0.0, 100.0].  Returns 100.0 when ``expected_strikes``
            is empty (vacuous completeness).
        """
        if not expected_strikes:
            return 100.0
        present = sum(
            1
            for strike in expected_strikes
            if (expiry, strike, "CE") in self.quotes or (expiry, strike, "PE") in self.quotes
        )
        return (present / len(expected_strikes)) * 100.0

    def strikes_by_expiry(self, expiry: date) -> list[float]:
        """Sorted list of distinct strikes available for a given expiry.

        Args:
            expiry: Expiry date to filter by.

        Returns:
            Sorted list of strike prices.
        """
        return sorted(
            {strike for (exp, strike, _) in self.quotes if exp == expiry}
        )

    def all_expiries(self) -> list[date]:
        """Sorted list of distinct expiry dates in this snapshot.

        Returns:
            List of ``date`` objects in ascending order.
        """
        return sorted({exp for (exp, _, _) in self.quotes})

    def atm_quote(
        self,
        expiry: date,
        option_type: str,
    ) -> OptionQuote | None:
        """Return the quote closest to the current underlying price (ATM).

        When two strikes are equidistant from the underlying price, the
        lower strike is returned.

        Args:
            expiry: Expiry date to search within.
            option_type: ``"CE"`` or ``"PE"``.

        Returns:
            The ATM ``OptionQuote``, or ``None`` if no quotes exist for
            this expiry/option_type combination.
        """
        strikes = [
            strike
            for (exp, strike, opt) in self.quotes
            if exp == expiry and opt == option_type
        ]
        if not strikes:
            return None
        atm_strike = min(strikes, key=lambda s: abs(s - self.underlying_price))
        return self.quotes.get((expiry, atm_strike, option_type))

    def iv_smile(
        self,
        expiry: date,
    ) -> dict[float, tuple[float, float]]:
        """IV smile for a given expiry: strike → (call_iv, put_iv).

        Only strikes with both a CE and PE quote are included.  If a leg
        is missing or its IV is -1.0 (unavailable), ``float('nan')`` is
        used for that side so callers can distinguish from a genuine zero.

        Args:
            expiry: Expiry date to build the smile for.

        Returns:
            Dict mapping strike price to ``(call_iv, put_iv)``.  Both IV
            values are annualised decimals or ``float('nan')``.
        """
        strikes = {
            strike
            for (exp, strike, _) in self.quotes
            if exp == expiry
        }
        smile: dict[float, tuple[float, float]] = {}
        for strike in sorted(strikes):
            ce = self.quotes.get((expiry, strike, "CE"))
            pe = self.quotes.get((expiry, strike, "PE"))
            call_iv = ce.iv if (ce is not None and ce.iv != -1.0) else math.nan
            put_iv = pe.iv if (pe is not None and pe.iv != -1.0) else math.nan
            smile[strike] = (call_iv, put_iv)
        return smile
