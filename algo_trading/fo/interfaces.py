"""Formal abstract base classes for all system interfaces.

This module defines the **interface contracts** that every concrete
implementation must satisfy.  Abstract base classes (ABCs) are used so
that ``isinstance()`` checks, type annotations, and mypy can all
enforce the contract without coupling code to a specific implementation
(Shoonya broker gateway, Parquet snapshot store, etc.).

Design reference
----------------
Section 4 of the NSE F&O System Design Document:

  4.1 SnapshotStore          — write/read/query option-chain snapshots
  4.2 BrokerGateway          — order placement and session management
  4.3 HistoricalDataProvider — historical OHLCV / chain / IV surface data
  4.4 ExecutionVenueAdapter  — order routing and venue status
  4.5 MarginModelProvider    — SPAN + exposure margin estimation

Broker constraint
-----------------
The current implementation target is **Shoonya (Finvasia)** only.
[P] Project constraint documented in project_context.md.

Usage example::

    from algo_trading.fo.interfaces import BrokerGateway

    class ShoonyaGateway(BrokerGateway):
        def place_order(self, order: dict) -> str: ...
        # ... implement all abstract methods ...

    gateway: BrokerGateway = ShoonyaGateway(...)

Backwards compatibility note
-----------------------------
The previous version of this file defined a narrower ``SnapshotStore``
ABC (targeting ``ChainSnapshot`` objects specifically).  That contract
has been replaced by the full Section 4 interface set.  Any code that
imported ``SnapshotStore.write`` / ``read_latest`` / ``count`` must be
updated to the new API (``write_snapshot`` / ``read_snapshot`` / ``list_gaps``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# 4.1  SnapshotStore
# ---------------------------------------------------------------------------


class SnapshotStore(ABC):
    """Interface for a time-series store of NSE option-chain snapshots.

    Each snapshot is a complete point-in-time capture of the option chain
    (all strikes × expiries) for one or more underlying instruments.
    Implementations may use Parquet files, a TimescaleDB table, Redis
    streams, etc.

    Schema versioning is mandatory so that readers can deserialise
    snapshots written by an older or newer writer without silent data
    corruption.
    """

    @abstractmethod
    def write_snapshot(
        self,
        timestamp: datetime,
        chain_data: dict,
        schema_version: int,
    ) -> str:
        """Persist a single option-chain snapshot and return its unique ID.

        Args:
            timestamp: The market timestamp of this snapshot (timezone-aware
                UTC recommended).
            chain_data: Arbitrary dict representing the full chain payload.
                The schema is validated against *schema_version*.
            schema_version: Integer version of the ``chain_data`` schema.
                Callers must increment this whenever the payload structure
                changes in a backwards-incompatible way.

        Returns:
            A stable, opaque snapshot ID (e.g. a UUID string or a
            content-addressed hash) that can be passed to
            :meth:`read_snapshot`.

        Raises:
            ValueError: If *schema_version* is unknown or unsupported.
            StorageError: (implementation-defined) on I/O failure.
        """
        ...

    @abstractmethod
    def read_snapshot(self, timestamp: datetime) -> dict:
        """Retrieve the snapshot closest to *timestamp*.

        Args:
            timestamp: Target market timestamp.  Implementations should
                return the snapshot whose stored timestamp is exactly equal
                to, or immediately precedes, *timestamp*.

        Returns:
            The deserialised chain payload dict, enriched with a
            ``"_snapshot_meta"`` key containing at minimum
            ``{"timestamp": ..., "schema_version": ...}``.

        Raises:
            KeyError: If no snapshot exists at or before *timestamp*.
        """
        ...

    @abstractmethod
    def read_range(
        self,
        start_ts: datetime,
        end_ts: datetime,
    ) -> Iterator[dict]:
        """Yield all snapshots in the half-open interval [start_ts, end_ts).

        Args:
            start_ts: Range start (inclusive).
            end_ts: Range end (exclusive).

        Yields:
            Deserialised chain payload dicts in chronological order,
            each enriched with ``"_snapshot_meta"``.

        Raises:
            ValueError: If *start_ts* >= *end_ts*.
        """
        ...

    @abstractmethod
    def get_schema_version(self, snapshot_id: str) -> int:
        """Return the schema version of the snapshot identified by *snapshot_id*.

        Args:
            snapshot_id: Opaque ID returned by :meth:`write_snapshot`.

        Returns:
            Integer schema version.

        Raises:
            KeyError: If *snapshot_id* does not exist.
        """
        ...

    @abstractmethod
    def list_gaps(
        self,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Identify time ranges within [start_ts, end_ts] that contain no snapshots.

        Useful for data-quality checks and backfill jobs.

        Args:
            start_ts: Range start (inclusive).
            end_ts: Range end (inclusive).

        Returns:
            List of (gap_start, gap_end) tuples, each representing a
            contiguous period with no stored snapshots.  Empty list if
            coverage is complete.

        Raises:
            ValueError: If *start_ts* >= *end_ts*.
        """
        ...


# ---------------------------------------------------------------------------
# 4.2  BrokerGateway
# ---------------------------------------------------------------------------


class BrokerGateway(ABC):
    """Interface for broker API interactions (order placement, session, fills).

    The current target broker is **Shoonya (Finvasia)**.  Any future broker
    integration must implement this interface rather than being wired
    directly into strategy or risk code.

    Session management is included here because Shoonya sessions expire
    daily and must be explicitly refreshed.
    """

    @abstractmethod
    def place_order(self, order: dict) -> str:
        """Submit a new order to the broker and return the broker order ID.

        Args:
            order: Order payload dict.  The required keys and their types
                are defined by the concrete implementation (e.g. Shoonya
                API fields: ``"tradingsymbol"``, ``"exchange"``,
                ``"transactiontype"``, ``"quantity"``, ``"pricetype"``).

        Returns:
            Broker-assigned order ID string (opaque; use with
            :meth:`cancel_order`).

        Raises:
            BrokerError: (implementation-defined) on API rejection or
                connectivity failure.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        """Request cancellation of an open order.

        Args:
            order_id: Broker-assigned order ID returned by
                :meth:`place_order`.

        Returns:
            Broker response dict with at minimum a ``"status"`` key.

        Raises:
            KeyError: If *order_id* is not recognised by the broker.
            BrokerError: (implementation-defined) on API failure.
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return all open positions for the current session.

        Returns:
            List of position dicts.  Each dict contains at minimum:
            ``"tradingsymbol"``, ``"netqty"``, ``"pnl"``.

        Raises:
            BrokerError: (implementation-defined) on API failure.
        """
        ...

    @abstractmethod
    def subscribe_fills(self, callback: Any) -> Any:
        """Register a callback to be invoked on every order fill event.

        The callback signature must be ``callback(fill_event: dict) -> None``.
        Implementations may use a WebSocket subscription, polling loop,
        or broker-specific push mechanism.

        Args:
            callback: Callable invoked with a fill-event dict on each fill.

        Returns:
            An opaque subscription handle that can be used to unsubscribe
            (implementation-defined).

        Raises:
            BrokerError: (implementation-defined) on subscription failure.
        """
        ...

    @abstractmethod
    def get_session_status(self) -> str:
        """Return the current broker session status.

        Returns:
            One of ``"ACTIVE"``, ``"EXPIRED"``, ``"INVALID"``, or
            ``"UNKNOWN"``.  Callers should call :meth:`refresh_session`
            if the status is not ``"ACTIVE"``.
        """
        ...

    @abstractmethod
    def refresh_session(self) -> str:
        """Re-authenticate and obtain a fresh broker session token.

        Returns:
            The new session token string.

        Raises:
            AuthenticationError: (implementation-defined) if credentials
                are invalid or the broker rejects the login.
        """
        ...


# ---------------------------------------------------------------------------
# 4.3  HistoricalDataProvider
# ---------------------------------------------------------------------------


class HistoricalDataProvider(ABC):
    """Interface for sourcing historical market data for backtesting.

    Implementations may read from local Parquet files, a database,
    or a third-party data vendor API.  The interface intentionally
    abstracts away the storage/transport layer so strategy code is
    portable across data sources.
    """

    @abstractmethod
    def get_chain_snapshot(
        self,
        symbol: str,
        expiry: date,
        timestamp: datetime,
    ) -> dict:
        """Return the option chain for *symbol* and *expiry* at *timestamp*.

        Args:
            symbol: Underlying index or stock symbol (e.g. ``"NIFTY"``).
            expiry: Option expiry date.
            timestamp: Point-in-time for which the chain is requested.

        Returns:
            Dict mapping strike prices to option contract data
            (bid/ask, OI, IV, greeks, etc.).

        Raises:
            DataNotAvailableError: (implementation-defined) if no chain
                data exists for the requested timestamp.
        """
        ...

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str,
    ) -> Any:
        """Return OHLCV data for *symbol* between *start_date* and *end_date*.

        Args:
            symbol: Underlying index or stock symbol.
            start_date: First date of the requested range (inclusive).
            end_date: Last date of the requested range (inclusive).
            interval: Candle interval string, e.g. ``"1m"``, ``"5m"``,
                ``"1d"``.

        Returns:
            A time-indexed structure (e.g. ``pandas.DataFrame`` with columns
            ``open``, ``high``, ``low``, ``close``, ``volume``).
            The concrete return type is implementation-defined but must be
            documented by the implementing class.

        Raises:
            DataNotAvailableError: (implementation-defined) if the requested
                range or interval is unavailable.
        """
        ...

    @abstractmethod
    def get_iv_surface(
        self,
        symbol: str,
        expiry: date,
        timestamp: datetime,
    ) -> dict:
        """Return the implied-volatility surface for *symbol* at *timestamp*.

        Args:
            symbol: Underlying index or stock symbol.
            expiry: Option expiry for the surface slice.
            timestamp: Point-in-time for the IV surface.

        Returns:
            Dict mapping strike prices to IV values (and optionally
            Greeks and model parameters).

        Raises:
            DataNotAvailableError: (implementation-defined) if no IV data
                is available.
        """
        ...

    @abstractmethod
    def get_available_dates(self, symbol: str) -> list[date]:
        """Return all calendar dates for which data exists for *symbol*.

        Args:
            symbol: Underlying index or stock symbol.

        Returns:
            Sorted list of ``date`` objects.  Empty list if no data is
            available for this symbol.
        """
        ...


# ---------------------------------------------------------------------------
# 4.4  ExecutionVenueAdapter
# ---------------------------------------------------------------------------


class ExecutionVenueAdapter(ABC):
    """Interface for smart-order routing and venue management.

    In the current architecture the only execution venue is NSE via
    Shoonya.  This interface is defined now so that multi-venue routing
    (e.g. BSE, dark pools) can be added later without touching strategy
    or risk code.
    """

    @abstractmethod
    def route_order(self, order: dict, venue_hint: str = "") -> dict:
        """Route *order* to the optimal (or hinted) execution venue.

        Args:
            order: Order payload dict (same schema as
                :meth:`BrokerGateway.place_order`).
            venue_hint: Optional preferred venue identifier (e.g.
                ``"NSE"``, ``"BSE"``).  Implementations may override
                this hint if the venue is unavailable.

        Returns:
            Routing result dict containing at minimum:
            ``{"order_id": str, "venue": str, "status": str}``.

        Raises:
            RoutingError: (implementation-defined) if no venue can accept
                the order.
        """
        ...

    @abstractmethod
    def get_venue_status(self) -> dict:
        """Return the operational status of all known execution venues.

        Returns:
            Dict mapping venue name to status string, e.g.
            ``{"NSE": "OPEN", "BSE": "CLOSED"}``.
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Return the current status of an order placed through this adapter.

        Args:
            order_id: Order ID returned by :meth:`route_order`.

        Returns:
            Status dict with at minimum:
            ``{"order_id": str, "status": str, "filled_qty": int,
               "remaining_qty": int}``.

        Raises:
            KeyError: If *order_id* is unknown.
        """
        ...

    @abstractmethod
    def get_market_hours(self) -> dict:
        """Return today's market session hours for all known venues.

        Returns:
            Dict mapping venue name to session times, e.g.::

                {
                    "NSE": {
                        "pre_open_start": datetime(...),
                        "market_open": datetime(...),
                        "market_close": datetime(...),
                    }
                }

            All datetimes should be timezone-aware.
        """
        ...


# ---------------------------------------------------------------------------
# 4.5  MarginModelProvider
# ---------------------------------------------------------------------------


class MarginModelProvider(ABC):
    """Interface for margin estimation (SPAN + exposure) for F&O positions.

    NSE uses the **SPAN** (Standard Portfolio Analysis of Risk) methodology
    for initial margin and an additional **exposure margin** on top.
    Implementations may use:

    - A local SPAN parameter file (downloaded from NSE/NSCCL)
    - The broker's margin API (Shoonya has a margin-estimate endpoint)
    - A simplified analytical approximation for backtesting

    All ``positions`` arguments accept the same list-of-dicts schema used
    by :meth:`BrokerGateway.get_positions`.
    """

    @abstractmethod
    def estimate_span_margin(self, positions: list[dict]) -> float:
        """Estimate the SPAN initial margin for *positions*.

        SPAN margin is calculated as the worst-case one-day loss over a
        set of 16 risk scenarios (price up/down × vol up/down combinations).
        [S] NSE/NSCCL SPAN methodology documentation.

        Args:
            positions: List of open position dicts (same schema as
                :meth:`BrokerGateway.get_positions`).

        Returns:
            Estimated SPAN margin in INR.
        """
        ...

    @abstractmethod
    def estimate_exposure_margin(self, positions: list[dict]) -> float:
        """Estimate the exposure (additional) margin for *positions*.

        Exposure margin is charged over and above SPAN margin.  For index
        options it is typically 3 % of notional value.
        [S] NSE/NSCCL margin framework circulars.

        Args:
            positions: List of open position dicts.

        Returns:
            Estimated exposure margin in INR.
        """
        ...

    @abstractmethod
    def estimate_total_margin(self, positions: list[dict]) -> float:
        """Estimate total margin (SPAN + exposure) for *positions*.

        Convenience method that sums :meth:`estimate_span_margin` and
        :meth:`estimate_exposure_margin`.  Implementations may provide
        a more accurate combined calculation if the broker API supports it.

        Args:
            positions: List of open position dicts.

        Returns:
            Total estimated margin in INR.
        """
        ...

    @abstractmethod
    def get_margin_utilization(
        self,
        positions: list[dict],
        available_capital: float,
    ) -> float:
        """Return the fraction of *available_capital* consumed by margin.

        Args:
            positions: List of open position dicts.
            available_capital: Total deployable capital in INR.

        Returns:
            A float in [0.0, inf).  Values > 1.0 indicate a margin breach
            (margin required exceeds available capital).  Callers should
            treat values >= a configurable threshold (e.g. 0.8) as a
            risk-limit trigger.

        Raises:
            ValueError: If *available_capital* is <= 0.
        """
        ...
