"""
algo_trading/execution/tca.py — Transaction Cost Analysis (TCA).

Transaction Cost Analysis decomposes the total cost of execution into
components that pinpoint *where* value is lost between a trade decision and
its final fill.  Three benchmark prices are tracked per trade:

decision_price
    The mid-price (or LTP) at the instant the signal was generated.
    Represents the idealized "zero-latency, zero-cost" execution price.

arrival_price
    The best ask (for BUY) or best bid (for SELL) when the order actually
    reached the exchange.  Captures the market-impact of signal-to-order
    latency — sometimes called "pre-trade drift" or "timing cost".

fill_price
    The actual execution price.  The gap from arrival_price is the
    post-arrival slippage — queue position, market-impact, and spread cost.

Key TCA metrics (all in basis points unless stated otherwise):

implementation_shortfall = fill_price vs decision_price
    Total cost of not executing at the instant the signal fired.

timing_cost = arrival_price vs decision_price
    How much of that shortfall arose *before* the order hit the book
    (signal latency, processing, network).

arrival_shortfall = fill_price vs arrival_price
    How much arose *after* arrival (queue position, adverse selection, spread).

miss_cost (opportunity cost)
    For unfilled orders, the price move against the intended direction
    measured from arrival_price to the next available market price.

A ``best_execution_score`` (0–100) aggregates these metrics into a single
number that can be tracked over time to detect execution quality regression.

Usage example::

    tracker = ImplementationShortfallTracker()
    trade_id = tracker.record_signal("RELIANCE", decision_price=2500.0,
                                     direction="BUY", qty=100, signal_time_ns=t0)
    tracker.record_arrival(trade_id, arrival_price=2500.5, limit_price=2500.25)
    tracker.record_fill(trade_id, fill_price=2500.30, fill_time_ns=t1)
    report = generate_tca_report(tracker, "RELIANCE", period_start, period_end)
    print(format_tca_summary(report))
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BPS conversion factor
_BPS_FACTOR: float = 10_000.0

# best_execution_score thresholds
_SCORE_START: float = 100.0
_ARRIVAL_SHORTFALL_FREE_BPS: float = 3.0          # No penalty below this
_ARRIVAL_SHORTFALL_PENALTY_PER_BPS: float = 1.0   # Points deducted per bps above free tier
_MISS_RATE_FREE_PCT: float = 10.0                  # No penalty below 10% miss rate
_MISS_RATE_PENALTY_PER_PCT: float = 0.5            # Points deducted per % above free tier
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 100.0

# Minimum trades required for p95 calculation to be meaningful
_MIN_TRADES_FOR_P95: int = 5


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

@dataclass
class ArrivalPriceRecord:
    """A single trade's price journey from signal to fill.

    Attributes:
        symbol: NSE trading symbol (e.g. "RELIANCE").
        strategy_id: Strategy identifier that generated the signal.
        signal_time: Wall-clock time when the signal was generated (IST).
        decision_price: Mid/LTP price at signal generation time.
        arrival_price: Best ask (BUY) or best bid (SELL) when order reached
            the exchange.  None if arrival was never recorded.
        limit_price_placed: The LIMIT price actually submitted to the broker.
            None if not yet placed.
        fill_price: Actual execution price.  None if not filled.
        fill_time_ns: Fill timestamp in nanoseconds (from time.time_ns()).
            None if not filled.
        direction: "BUY" or "SELL".
        qty: Number of shares.
        is_filled: True once a fill is recorded; False for cancelled/expired orders.
    """

    symbol: str
    strategy_id: str
    signal_time: datetime
    decision_price: float
    direction: str                        # "BUY" or "SELL"
    qty: int
    arrival_price: Optional[float] = None
    limit_price_placed: Optional[float] = None
    fill_price: Optional[float] = None
    fill_time_ns: Optional[int] = None
    is_filled: bool = False
    miss_reason: Optional[str] = None
    miss_reference_price: Optional[float] = None   # Price observed at miss time


@dataclass(frozen=True)
class TCAReport:
    """Aggregated TCA metrics for a symbol over a time window.

    Attributes:
        symbol: NSE trading symbol.
        period_start: Start of the analysis period.
        period_end: End of the analysis period.
        n_trades: Total filled trades in the period.
        n_misses: Orders that were not filled (cancelled, expired, or skipped).
        avg_decision_shortfall_bps: Mean implementation shortfall in bps
            (fill vs decision price).  Positive = adverse.
        avg_arrival_shortfall_bps: Mean post-arrival slippage in bps
            (fill vs arrival price).  Positive = adverse.
        avg_timing_cost_bps: Mean timing cost in bps (arrival vs decision
            price).  Positive = adverse.
        avg_miss_opportunity_cost_bps: Mean opportunity cost for missed orders,
            measured as the adverse price move from arrival to miss price.
        p95_arrival_shortfall_bps: 95th percentile arrival shortfall in bps.
            Captures tail execution events; requires >= 5 trades.
        total_shortfall_inr: Sum of (fill_price - decision_price) * qty * direction_sign
            across all filled trades, in INR.  Positive = total cost paid.
        best_execution_score: Composite score 0–100.  100 = perfect execution.
            Reduced by arrival shortfall above 3 bps and miss rate above 10%.
    """

    symbol: str
    period_start: datetime
    period_end: datetime
    n_trades: int
    n_misses: int
    avg_decision_shortfall_bps: float
    avg_arrival_shortfall_bps: float
    avg_timing_cost_bps: float
    avg_miss_opportunity_cost_bps: float
    p95_arrival_shortfall_bps: float
    total_shortfall_inr: float
    best_execution_score: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ImplementationShortfallTracker:
    """Track three-price implementation shortfall per trade.

    Each trade passes through up to four lifecycle events:
      1. ``record_signal``  — signal generated, decision_price captured.
      2. ``record_arrival`` — order reached exchange, arrival_price captured.
      3. ``record_fill``    — order filled, fill_price captured.
         OR
      3. ``record_miss``    — order expired/cancelled without fill.

    All completed (filled + missed) records are retained in memory.
    Thread safety: not guaranteed for concurrent writes to the same trade_id.
    Concurrent writes to *different* trade_ids are safe.

    Attributes:
        _records: Maps trade_id → ArrivalPriceRecord.
        _by_symbol: Maps symbol → list of trade_ids for fast symbol queries.
    """

    def __init__(self) -> None:
        self._records: dict[str, ArrivalPriceRecord] = {}
        self._by_symbol: dict[str, list[str]] = defaultdict(list)

    # ---- Lifecycle methods ------------------------------------------------

    def record_signal(
        self,
        symbol: str,
        decision_price: float,
        direction: str,
        qty: int,
        signal_time_ns: int,
        strategy_id: str = "",
    ) -> str:
        """Record a new trade signal and return a unique trade_id.

        Args:
            symbol: NSE trading symbol.
            decision_price: Mid/LTP price at signal generation.
            direction: "BUY" or "SELL".
            qty: Intended order quantity in shares.
            signal_time_ns: Signal wall-clock time in nanoseconds.
            strategy_id: Optional strategy identifier for attribution.

        Returns:
            A unique trade_id string to be passed to subsequent lifecycle calls.
        """
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be 'BUY' or 'SELL', got {direction!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if decision_price <= 0.0:
            raise ValueError(f"decision_price must be positive, got {decision_price}")

        trade_id = str(uuid4())
        signal_dt = datetime.fromtimestamp(signal_time_ns / 1e9)
        record = ArrivalPriceRecord(
            symbol=symbol,
            strategy_id=strategy_id,
            signal_time=signal_dt,
            decision_price=decision_price,
            direction=direction,
            qty=qty,
        )
        self._records[trade_id] = record
        self._by_symbol[symbol].append(trade_id)

        log.debug(
            "tca.signal_recorded",
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            decision_price=f"{decision_price:.4f}",
            qty=qty,
        )
        return trade_id

    def record_arrival(
        self,
        trade_id: str,
        arrival_price: float,
        limit_price: float,
    ) -> None:
        """Record the best market price when the order reached the exchange.

        Args:
            trade_id: Trade identifier returned by ``record_signal``.
            arrival_price: Best ask (BUY) or best bid (SELL) at order arrival.
            limit_price: The LIMIT price actually submitted to the broker.

        Raises:
            KeyError: If trade_id is not found.
            ValueError: If prices are non-positive.
        """
        record = self._get_record(trade_id, "record_arrival")
        if arrival_price <= 0.0:
            raise ValueError(f"arrival_price must be positive, got {arrival_price}")
        if limit_price <= 0.0:
            raise ValueError(f"limit_price must be positive, got {limit_price}")

        record.arrival_price = arrival_price
        record.limit_price_placed = limit_price

        timing_cost_bps = _compute_bps(
            reference=record.decision_price,
            execution=arrival_price,
            direction=record.direction,
        )
        log.debug(
            "tca.arrival_recorded",
            trade_id=trade_id,
            symbol=record.symbol,
            arrival_price=f"{arrival_price:.4f}",
            timing_cost_bps=f"{timing_cost_bps:.2f}",
        )

    def record_fill(
        self,
        trade_id: str,
        fill_price: float,
        fill_time_ns: int,
    ) -> None:
        """Record a successful fill and compute all per-trade TCA metrics.

        Args:
            trade_id: Trade identifier returned by ``record_signal``.
            fill_price: Actual execution price.
            fill_time_ns: Fill timestamp in nanoseconds (from time.time_ns()).

        Raises:
            KeyError: If trade_id is not found.
            ValueError: If fill_price is non-positive.
        """
        record = self._get_record(trade_id, "record_fill")
        if fill_price <= 0.0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")

        record.fill_price = fill_price
        record.fill_time_ns = fill_time_ns
        record.is_filled = True

        # Compute and log per-trade metrics at fill time for immediate visibility
        decision_sf = _compute_bps(record.decision_price, fill_price, record.direction)
        arrival_sf = (
            _compute_bps(record.arrival_price, fill_price, record.direction)
            if record.arrival_price is not None
            else float("nan")
        )
        timing_cost = (
            _compute_bps(record.decision_price, record.arrival_price, record.direction)
            if record.arrival_price is not None
            else float("nan")
        )
        shortfall_inr = (
            (fill_price - record.decision_price)
            * record.qty
            * (1 if record.direction == "BUY" else -1)
        )

        log.info(
            "tca.fill_recorded",
            trade_id=trade_id,
            symbol=record.symbol,
            direction=record.direction,
            decision_price=f"{record.decision_price:.4f}",
            fill_price=f"{fill_price:.4f}",
            decision_shortfall_bps=f"{decision_sf:.2f}",
            arrival_shortfall_bps=f"{arrival_sf:.2f}",
            timing_cost_bps=f"{timing_cost:.2f}",
            shortfall_inr=f"{shortfall_inr:.2f}",
        )

    def record_miss(self, trade_id: str, reason: str) -> None:
        """Record a missed order (not filled — cancelled, expired, or skipped).

        The opportunity cost is computed as the adverse price move from the
        arrival price (or decision price if arrival was not recorded) to the
        ``miss_reference_price``.  The caller should pass a current market
        price at the time of cancellation/expiry so that the drift is captured.

        Args:
            trade_id: Trade identifier returned by ``record_signal``.
            reason: Human-readable reason for the miss (e.g. "stale_timeout",
                "risk_rejected", "spread_too_wide").

        Raises:
            KeyError: If trade_id is not found.
        """
        record = self._get_record(trade_id, "record_miss")
        record.is_filled = False
        record.miss_reason = reason

        # Use current time as miss reference (no price passed — opportunity cost
        # cannot be quantified without a reference price; set to None)
        log.info(
            "tca.miss_recorded",
            trade_id=trade_id,
            symbol=record.symbol,
            direction=record.direction,
            reason=reason,
            arrival_price=record.arrival_price,
        )

    def record_miss_with_price(
        self, trade_id: str, reason: str, reference_price: float
    ) -> None:
        """Record a missed order with a reference market price for opportunity cost.

        Args:
            trade_id: Trade identifier returned by ``record_signal``.
            reason: Human-readable reason for the miss.
            reference_price: Current market price at time of miss (used to
                measure opportunity cost — the adverse move we forfeited).

        Raises:
            KeyError: If trade_id is not found.
            ValueError: If reference_price is non-positive.
        """
        self.record_miss(trade_id, reason)
        record = self._get_record(trade_id, "record_miss_with_price")
        if reference_price <= 0.0:
            raise ValueError(f"reference_price must be positive, got {reference_price}")
        record.miss_reference_price = reference_price

    # ---- Query helpers ---------------------------------------------------

    def get_record(self, trade_id: str) -> Optional[ArrivalPriceRecord]:
        """Return the ArrivalPriceRecord for a given trade_id, or None.

        Args:
            trade_id: Trade identifier returned by ``record_signal``.

        Returns:
            The ArrivalPriceRecord if found, else None.
        """
        return self._records.get(trade_id)

    def get_records_for_symbol(
        self,
        symbol: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[ArrivalPriceRecord]:
        """Return all records for a symbol within a time window.

        Args:
            symbol: NSE trading symbol.
            period_start: Inclusive start of the window.
            period_end: Inclusive end of the window.

        Returns:
            List of ArrivalPriceRecord objects ordered by signal_time.
        """
        ids = self._by_symbol.get(symbol, [])
        records = [
            r for tid in ids
            if (r := self._records.get(tid)) is not None
            and period_start <= r.signal_time <= period_end
        ]
        records.sort(key=lambda r: r.signal_time)
        return records

    # ---- Private helpers --------------------------------------------------

    def _get_record(self, trade_id: str, caller: str) -> ArrivalPriceRecord:
        """Retrieve a record or raise KeyError with a descriptive message.

        Args:
            trade_id: Trade identifier.
            caller: Name of calling method for log context.

        Returns:
            The ArrivalPriceRecord for the given trade_id.

        Raises:
            KeyError: If trade_id is not found.
        """
        record = self._records.get(trade_id)
        if record is None:
            log.warning(
                "tca.unknown_trade_id",
                caller=caller,
                trade_id=trade_id,
            )
            raise KeyError(f"tca: unknown trade_id {trade_id!r} in {caller}()")
        return record


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_tca_report(
    tracker: ImplementationShortfallTracker,
    symbol: str,
    period_start: datetime,
    period_end: datetime,
) -> TCAReport:
    """Generate an aggregated TCA report for a symbol over a time window.

    Collects all ArrivalPriceRecords for the symbol in the window, computes
    per-trade shortfall components, aggregates to means and percentiles, and
    returns a frozen TCAReport dataclass.

    Args:
        tracker: ImplementationShortfallTracker with recorded trades.
        symbol: NSE trading symbol to analyse.
        period_start: Start of analysis window (inclusive).
        period_end: End of analysis window (inclusive).

    Returns:
        TCAReport with all metrics populated.  Returns a zero-valued report
        when no records are found in the window.
    """
    records = tracker.get_records_for_symbol(symbol, period_start, period_end)

    filled = [r for r in records if r.is_filled]
    missed = [r for r in records if not r.is_filled]

    n_trades = len(filled)
    n_misses = len(missed)

    if n_trades == 0 and n_misses == 0:
        log.info(
            "tca.report_empty",
            symbol=symbol,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
        )
        return TCAReport(
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            n_trades=0,
            n_misses=0,
            avg_decision_shortfall_bps=0.0,
            avg_arrival_shortfall_bps=0.0,
            avg_timing_cost_bps=0.0,
            avg_miss_opportunity_cost_bps=0.0,
            p95_arrival_shortfall_bps=0.0,
            total_shortfall_inr=0.0,
            best_execution_score=_SCORE_MAX,
        )

    # ---- Per-trade metrics for filled trades ----------------------------
    decision_shortfalls: list[float] = []
    arrival_shortfalls: list[float] = []
    timing_costs: list[float] = []
    total_shortfall_inr: float = 0.0

    for r in filled:
        assert r.fill_price is not None  # guaranteed by record_fill()
        direction_sign = 1 if r.direction == "BUY" else -1

        decision_sf = _compute_bps(r.decision_price, r.fill_price, r.direction)
        decision_shortfalls.append(decision_sf)

        if r.arrival_price is not None:
            arrival_sf = _compute_bps(r.arrival_price, r.fill_price, r.direction)
            arrival_shortfalls.append(arrival_sf)
            timing_cost = _compute_bps(r.decision_price, r.arrival_price, r.direction)
            timing_costs.append(timing_cost)

        notional_cost = (r.fill_price - r.decision_price) * r.qty * direction_sign
        total_shortfall_inr += notional_cost

    # ---- Opportunity cost for missed trades ----------------------------
    miss_costs: list[float] = []
    for r in missed:
        if r.miss_reference_price is not None:
            ref = r.arrival_price if r.arrival_price is not None else r.decision_price
            miss_cost = _compute_bps(ref, r.miss_reference_price, r.direction)
            miss_costs.append(miss_cost)

    # ---- Aggregation ---------------------------------------------------
    avg_decision_sf = float(np.mean(decision_shortfalls)) if decision_shortfalls else 0.0
    avg_arrival_sf = float(np.mean(arrival_shortfalls)) if arrival_shortfalls else 0.0
    avg_timing_cost = float(np.mean(timing_costs)) if timing_costs else 0.0
    avg_miss_cost = float(np.mean(miss_costs)) if miss_costs else 0.0

    if len(arrival_shortfalls) >= _MIN_TRADES_FOR_P95:
        p95_arrival_sf = float(np.percentile(arrival_shortfalls, 95))
    else:
        p95_arrival_sf = avg_arrival_sf  # fall back to mean when too few observations

    best_execution_score = _compute_best_execution_score(
        avg_arrival_shortfall_bps=avg_arrival_sf,
        n_trades=n_trades,
        n_misses=n_misses,
    )

    report = TCAReport(
        symbol=symbol,
        period_start=period_start,
        period_end=period_end,
        n_trades=n_trades,
        n_misses=n_misses,
        avg_decision_shortfall_bps=avg_decision_sf,
        avg_arrival_shortfall_bps=avg_arrival_sf,
        avg_timing_cost_bps=avg_timing_cost,
        avg_miss_opportunity_cost_bps=avg_miss_cost,
        p95_arrival_shortfall_bps=p95_arrival_sf,
        total_shortfall_inr=total_shortfall_inr,
        best_execution_score=best_execution_score,
    )

    log.info(
        "tca.report_generated",
        symbol=symbol,
        n_trades=n_trades,
        n_misses=n_misses,
        avg_arrival_shortfall_bps=f"{avg_arrival_sf:.2f}",
        best_execution_score=f"{best_execution_score:.1f}",
    )
    return report


def format_tca_summary(report: TCAReport) -> str:
    """Format a TCAReport as a human-readable multi-line text summary.

    Args:
        report: TCAReport returned by ``generate_tca_report``.

    Returns:
        A formatted string suitable for logging, Telegram alerts, or dashboards.
    """
    total = report.n_trades + report.n_misses
    miss_rate_pct = (report.n_misses / total * 100.0) if total > 0 else 0.0

    lines = [
        f"TCA Report — {report.symbol}",
        f"  Period  : {report.period_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{report.period_end.strftime('%Y-%m-%d %H:%M')}",
        f"  Trades  : {report.n_trades} filled, {report.n_misses} missed "
        f"({miss_rate_pct:.1f}% miss rate)",
        f"  Decision shortfall : {report.avg_decision_shortfall_bps:+.2f} bps (avg)",
        f"  Arrival  shortfall : {report.avg_arrival_shortfall_bps:+.2f} bps (avg)",
        f"  Timing cost        : {report.avg_timing_cost_bps:+.2f} bps (avg)",
        f"  P95 arrival slippage: {report.p95_arrival_shortfall_bps:+.2f} bps",
        f"  Opportunity cost   : {report.avg_miss_opportunity_cost_bps:+.2f} bps (missed orders)",
        f"  Total shortfall    : ₹{report.total_shortfall_inr:+,.2f} INR",
        f"  Best-execution score: {report.best_execution_score:.1f} / 100",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_bps(reference: float, execution: float, direction: str) -> float:
    """Compute signed basis-point cost relative to a reference price.

    A positive value always means an adverse move (paid more as a buyer,
    received less as a seller).

    Formula:
        bps = (execution - reference) / reference * 10_000 * direction_sign

    where direction_sign = +1 for BUY, -1 for SELL.

    Args:
        reference: Benchmark price (decision or arrival).
        execution: Realised price (arrival or fill).
        direction: "BUY" or "SELL".

    Returns:
        Signed basis-point cost.  Returns 0.0 if reference is zero or negative.
    """
    if reference <= 0.0:
        return 0.0
    direction_sign = 1.0 if direction == "BUY" else -1.0
    return (execution - reference) / reference * _BPS_FACTOR * direction_sign


def _compute_best_execution_score(
    avg_arrival_shortfall_bps: float,
    n_trades: int,
    n_misses: int,
) -> float:
    """Compute the best-execution composite score (0–100).

    Scoring rules:
      - Start at 100.
      - Subtract 1.0 point per bps of avg arrival shortfall above 3 bps.
      - Subtract 0.5 points per percentage point of miss rate above 10%.
      - Clamp result to [0, 100].

    Args:
        avg_arrival_shortfall_bps: Mean arrival shortfall across filled trades.
        n_trades: Number of filled trades.
        n_misses: Number of missed (unfilled) orders.

    Returns:
        Score in [0.0, 100.0].
    """
    score = _SCORE_START

    # Arrival shortfall penalty
    excess_arrival_bps = max(0.0, avg_arrival_shortfall_bps - _ARRIVAL_SHORTFALL_FREE_BPS)
    score -= excess_arrival_bps * _ARRIVAL_SHORTFALL_PENALTY_PER_BPS

    # Miss-rate penalty
    total = n_trades + n_misses
    if total > 0:
        miss_rate_pct = n_misses / total * 100.0
        excess_miss_pct = max(0.0, miss_rate_pct - _MISS_RATE_FREE_PCT)
        score -= excess_miss_pct * _MISS_RATE_PENALTY_PER_PCT

    return float(max(_SCORE_MIN, min(_SCORE_MAX, score)))
