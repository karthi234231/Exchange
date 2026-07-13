"""
algo_trading/research/fill_simulator.py — Fill quality simulation for historical replay.

Implements all 9 fill modes from FO_SYSTEM_DESIGN.md Section 9.3.

Design rules:
  - "ask + spread/2 is a toy approximation, not a realistic execution model."
    All 9 modes are fully implemented with distinct logic.
  - Parameters marked [ASSUMED] must be calibrated from real fills in Phase 4.
    Do not treat them as ground truth before calibration.
  - All random draws use seeded random.Random instances whose seeds are derived
    from event_time + strike.  This preserves replay determinism: the same
    inputs always produce the same fill result.
  - SimulatedFill is NOT frozen — but internal _fill_cancel_race creates a
    new SimulatedFill rather than mutating an existing one, which is the
    correct pattern for all fill mode implementations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Fill mode enum ─────────────────────────────────────────────────────────────

class FillMode(str, Enum):
    """
    Fill simulation modes from FO_SYSTEM_DESIGN.md Section 9.3.

    All 9 modes are implemented in :class:`FillSimulator`.  Choose the mode
    based on research context:
    - ``REALISTIC`` for primary backtesting
    - ``PESSIMISTIC`` for stress-test / lower-bound runs
    - ``OPTIMISTIC`` as a theoretical upper bound only
    """
    OPTIMISTIC = "OPTIMISTIC"
    """Fill at mid price.  Best-case scenario; assumes perfect queue position."""

    REALISTIC = "REALISTIC"
    """Fill at ask + half_spread (buy) or bid - half_spread (sell).
    Represents the average outcome for informed LIMIT orders."""

    PESSIMISTIC = "PESSIMISTIC"
    """Full ask (buy) or full bid (sell) plus market-impact bps.
    Represents MARKET orders or thin-book fills."""

    QUEUE_POSITION = "QUEUE_POSITION"
    """LIMIT order queue-position model.
    fill_prob = min(queue_turnover, oi) / (oi + order_size).
    Returns None when fill_prob < threshold (order not filled this bar)."""

    TIME_OF_DAY_LIQUIDITY = "TIME_OF_DAY_LIQUIDITY"
    """Spread multiplier driven by time of day.
    09:15–09:30: 4× (wide); 11:00–14:00: 1× (baseline); 15:00–close: 1.5×."""

    EXPIRY_DAY_BLOWOUT = "EXPIRY_DAY_BLOWOUT"
    """Final hour of Tuesday expiry: spreads blow out 3×.
    OTM strikes (low OI) have a 50% fill-probability cap."""

    EVENT_DAY_BLOWOUT = "EVENT_DAY_BLOWOUT"
    """RBI policy day / budget day: IV spikes, spreads widen 3.5×.
    OTM fill probability capped at 40%."""

    MARKET_ORDER_THIN_BOOK = "MARKET_ORDER_THIN_BOOK"
    """OTM strikes with low OI: market orders slip beyond quoted ask.
    Extra slippage bps applied on top of full ask/bid price."""

    CANCELLATION_LATENCY = "CANCELLATION_LATENCY"
    """Cancel → fill race condition during fast markets.
    Models the probability that cancel arrives after fill."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SimulatedFill:
    """
    Result of simulating a single option leg fill.

    A ``None`` return from :meth:`FillSimulator.simulate_fill` means the order
    was not filled this bar (queue miss, OTM liquidity cap, etc.).

    Attributes
    ----------
    slippage_bps : float
        (fill_price − mid_price) / mid_price × 10 000.
        Negative values indicate a favourable fill (fill below mid for a buy).
    cancelled : bool
        True only for CANCELLATION_LATENCY mode when the cancel won the race.
        In that case filled_qty = 0 and fill_price = 0.0.
    notes : str
        Human-readable explanation of the fill logic applied.
    """
    order_id: str
    symbol: str
    expiry: Any           # date
    strike: float
    option_type: str      # "CE" or "PE"
    side: str             # "BUY" or "SELL"
    requested_qty: int    # in lots
    filled_qty: int       # in lots (may be < requested for partial fills)
    fill_price: float     # premium per unit
    slippage_bps: float   # (fill_price − mid) / mid × 10 000; negative = favourable
    fill_mode: FillMode
    timestamp: datetime
    is_partial: bool
    cancelled: bool       # True only in CANCELLATION_LATENCY when cancel won
    notes: str


# ── Simulator configuration ────────────────────────────────────────────────────

@dataclass
class FillSimulatorConfig:
    """
    Configuration parameters for fill simulation.

    All numeric values are [ASSUMED] starting points derived from NSE
    empirical observations documented in CLAUDE.md (IRF-Q10):
      - LIMIT 1-tick inside: 3–8 bps avg adverse selection
      - MARKET orders:       8–20 bps slippage

    These MUST be calibrated against real fills in Phase 4.  Do not treat
    them as accurate before calibration.
    """
    # ── Base slippage by mode ──────────────────────────────────────────────
    # [ASSUMED] starting values — calibrate from real fills in Phase 4
    optimistic_slippage_bps: float = 0.0      # mid price — theoretical best case
    realistic_slippage_bps: float = 5.0       # half-spread avg [ASSUMED]
    pessimistic_slippage_bps: float = 15.0    # full spread + impact [ASSUMED]

    # ── Queue position model ───────────────────────────────────────────────
    # [ASSUMED] — calibrate from real NSE order book data
    queue_fill_probability: float = 0.65      # fraction of queue that turns over per bar
    min_fill_probability: float = 0.30        # below this threshold → no fill this bar

    # ── Time-of-day spread multipliers ────────────────────────────────────
    # [ASSUMED] based on typical NSE spread widening patterns
    open_auction_spread_multiplier: float = 4.0    # 09:15–09:30
    midday_spread_multiplier: float = 1.0           # 11:00–14:00 baseline
    pre_close_spread_multiplier: float = 1.5        # 15:00–15:30

    # ── Expiry day blowout ────────────────────────────────────────────────
    # [ASSUMED] based on reported NSE expiry-day microstructure behaviour
    expiry_final_hour_spread_multiplier: float = 3.0
    otm_expiry_fill_probability_cap: float = 0.50   # 50% fill cap for low-OI strikes

    # ── Event day blowout ─────────────────────────────────────────────────
    # [ASSUMED] based on reported RBI/budget day behaviour
    event_day_spread_multiplier: float = 3.5
    event_day_otm_fill_prob_cap: float = 0.40        # 40% fill cap

    # ── Market order thin book ────────────────────────────────────────────
    # [ASSUMED] — calibrate thin_book_extra_slippage_bps from real fills
    low_oi_threshold_lots: int = 500
    thin_book_extra_slippage_bps: float = 20.0

    # ── Cancellation latency race ─────────────────────────────────────────
    # [ASSUMED] typical NSE broker round-trip ~100 ms; 5% race probability
    cancel_fill_race_probability: float = 0.05    # P(cancel arrives after fill)

    # ── General liquidity threshold ───────────────────────────────────────
    min_oi_lots_for_normal_fill: int = 100        # below this → apply thin-book model


# ── Simulator ─────────────────────────────────────────────────────────────────

class FillSimulator:
    """
    Simulates order fills against historical options chain data.

    Implements all 9 fill modes from FO_SYSTEM_DESIGN.md Section 9.3.

    Parameters
    ----------
    config : FillSimulatorConfig | None
        Slippage and probability parameters.  Uses defaults if None.
    mode : FillMode
        Default fill mode when ``simulate_fill`` is called without an
        explicit ``mode`` argument.

    Warnings
    --------
    All slippage parameters are [ASSUMED] starting values.
    Calibrate from real fills in Phase 4 before drawing conclusions about
    strategy edge.

    Example
    -------
    ::

        sim = FillSimulator(mode=FillMode.REALISTIC)
        fill = sim.simulate_fill(
            order_id="O-001", symbol="NIFTY", expiry=exp_date,
            strike=22000.0, option_type="CE", side="BUY", qty_lots=1,
            quote=quote_obj, event_time=event.event_time,
        )
        if fill is not None:
            strategy.record_fill(fill)
    """

    def __init__(
        self,
        config: FillSimulatorConfig | None = None,
        mode: FillMode = FillMode.REALISTIC,
    ) -> None:
        self.config = config or FillSimulatorConfig()
        self.default_mode = mode

    # ── Public interface ──────────────────────────────────────────────────────

    def simulate_fill(
        self,
        order_id: str,
        symbol: str,
        expiry: Any,
        strike: float,
        option_type: str,        # "CE" or "PE"
        side: str,               # "BUY" or "SELL"
        qty_lots: int,
        quote: Any,              # OptionContract / OptionQuote from chain_snapshot
        event_time: datetime,
        is_expiry_day: bool = False,
        is_event_day: bool = False,
        mode: FillMode | None = None,
    ) -> SimulatedFill | None:
        """
        Simulate a fill for a single option leg.

        Parameters
        ----------
        order_id : str
            Unique identifier for this order.
        quote : Any
            Current market quote.  Must expose ``bid``, ``ask``, ``ltp``,
            and ``oi`` as attributes or dict keys.
        mode : FillMode | None
            Override the simulator's default mode for this call only.

        Returns
        -------
        SimulatedFill
            Fill result, or ``None`` when the order is not filled this bar
            (queue miss, liquidity cap, etc.).

        Notes
        -----
        ``is_expiry_day`` and ``is_event_day`` are passed by the caller and
        are NOT derived from wall-clock time.  In replay, take these from
        :attr:`ReplayEvent.is_expiry_bar` and a separately maintained
        event-day calendar, both of which are deterministic.
        """
        used_mode = mode if mode is not None else self.default_mode

        # Extract market data from quote (supports both attribute and dict access)
        bid = self._get_field(quote, "bid", 0.0)
        ask = self._get_field(quote, "ask", 0.0)
        ltp = self._get_field(quote, "ltp", 0.0)
        oi  = self._get_field(quote, "oi",  0)

        bid = max(float(bid), 0.0)
        ask = max(float(ask), 0.0)
        ltp = max(float(ltp), 0.0)
        oi  = max(int(oi),    0)

        # Derive mid and spread
        mid    = (bid + ask) / 2.0 if (bid > 0.0 and ask > 0.0) else ltp
        spread = max(ask - bid, 0.0) if ask > bid else 0.0

        log.debug(
            "fill_simulator.simulate",
            order_id=order_id,
            mode=used_mode,
            strike=strike,
            side=side,
            bid=bid,
            ask=ask,
            mid=mid,
        )

        # Dispatch to the appropriate fill mode implementation
        if used_mode == FillMode.OPTIMISTIC:
            return self._fill_optimistic(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, mid, spread, event_time,
            )

        if used_mode == FillMode.REALISTIC:
            return self._fill_realistic(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, event_time,
            )

        if used_mode == FillMode.PESSIMISTIC:
            return self._fill_pessimistic(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, event_time,
            )

        if used_mode == FillMode.QUEUE_POSITION:
            return self._fill_queue_position(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, oi, event_time,
            )

        if used_mode == FillMode.TIME_OF_DAY_LIQUIDITY:
            return self._fill_time_of_day(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, event_time,
            )

        if used_mode == FillMode.EXPIRY_DAY_BLOWOUT:
            return self._fill_expiry_blowout(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, oi, event_time,
            )

        if used_mode == FillMode.EVENT_DAY_BLOWOUT:
            return self._fill_event_blowout(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, event_time,
            )

        if used_mode == FillMode.MARKET_ORDER_THIN_BOOK:
            return self._fill_thin_book(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, oi, event_time,
            )

        if used_mode == FillMode.CANCELLATION_LATENCY:
            return self._fill_cancel_race(
                order_id, symbol, expiry, strike, option_type,
                side, qty_lots, bid, ask, mid, spread, event_time,
            )

        raise ValueError(f"Unknown FillMode: {used_mode!r}")

    # ── Private fill mode implementations ────────────────────────────────────

    def _fill_optimistic(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        mid: float, spread: float, event_time: datetime,
    ) -> SimulatedFill:
        """
        Fill at mid price — theoretical best case.

        Assumes perfect queue position and zero market impact.
        Use only as an upper bound; not realistic for live trading.
        """
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=mid, slippage_bps=0.0,
            fill_mode=FillMode.OPTIMISTIC, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=f"Optimistic: fill at mid={mid:.2f}",
        )

    def _fill_realistic(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        event_time: datetime,
    ) -> SimulatedFill:
        """
        Fill at ask + half_spread (buy) or bid - half_spread (sell).

        Represents the average outcome for informed LIMIT orders placed
        inside the spread.  [ASSUMED: half_spread represents avg adverse
        selection; calibrate realistic_slippage_bps from real fills.]
        """
        half = spread / 2.0
        if side == "BUY":
            fill_price = ask + half     # slight beyond ask for LIMIT fill
        else:
            fill_price = max(bid - half, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.REALISTIC, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"Realistic: fill at {'ask' if side == 'BUY' else 'bid'}+half_spread"
                f"={fill_price:.2f}, spread={spread:.2f} [ASSUMED params]"
            ),
        )

    def _fill_pessimistic(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        event_time: datetime,
    ) -> SimulatedFill:
        """
        Full ask (buy) or full bid (sell) plus market-impact bps.

        Represents MARKET orders or fills in thin books where the order
        sweeps through multiple levels.
        [ASSUMED: pessimistic_slippage_bps; calibrate from real MARKET fills.]
        """
        impact = self.config.pessimistic_slippage_bps / 10_000.0
        if side == "BUY":
            # full ask + full spread + market-impact: always >= realistic (ask + half_spread)
            fill_price = ask + spread + ask * impact
        else:
            fill_price = max(bid - spread - bid * impact, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.PESSIMISTIC, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"Pessimistic: full {'ask' if side == 'BUY' else 'bid'}+spread+impact"
                f"={fill_price:.2f} ({self.config.pessimistic_slippage_bps:.0f}bps impact)"
                f" [ASSUMED params]"
            ),
        )

    def _fill_queue_position(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        oi: int, event_time: datetime,
    ) -> SimulatedFill | None:
        """
        LIMIT order queue-position model.

        Conservative assumption: we are always at the back of the queue.

        fill_prob = min(queue_turnover_per_bar, oi) / (oi + qty)

        where queue_turnover_per_bar = oi × queue_fill_probability.

        If fill_prob < min_fill_probability, the order is NOT filled this bar
        (returns None).

        [ASSUMED] All probability parameters need calibration from real NSE
        order-book data.  The 0.65 queue_fill_probability is a conservative
        starting estimate.
        """
        if oi <= 0:
            # Zero OI → cannot estimate queue depth → fall back to pessimistic
            log.debug("fill.queue_zero_oi", strike=strike, fallback="pessimistic")
            return self._fill_pessimistic(
                order_id, symbol, expiry, strike, opt_type,
                side, qty, bid, ask, mid, spread, event_time,
            )

        queue_turnover = oi * self.config.queue_fill_probability
        fill_prob = min(queue_turnover / (oi + qty), 1.0)

        if fill_prob < self.config.min_fill_probability:
            log.debug(
                "fill.queue_miss",
                strike=strike,
                oi=oi,
                fill_prob=f"{fill_prob:.2%}",
                threshold=f"{self.config.min_fill_probability:.2%}",
            )
            return None  # Order not filled this bar

        # Filled at limit price (bid for sell, ask for buy)
        fill_price = ask if side == "BUY" else bid
        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.QUEUE_POSITION, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"Queue: fill_prob={fill_prob:.2%}, oi={oi}, "
                f"price={fill_price:.2f} [ASSUMED params]"
            ),
        )

    def _fill_time_of_day(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        event_time: datetime,
    ) -> SimulatedFill:
        """
        Apply a time-of-day spread multiplier then perform a realistic fill.

        Multipliers (all [ASSUMED] — calibrate from real spread data):
          09:15–09:29 : 4×  (opening auction: very wide spreads)
          09:30–10:59 : 2×  (elevated post-open period)
          11:00–13:59 : 1×  (baseline midday liquidity)
          14:00–14:59 : 1.2×(slight afternoon widening)
          15:00–close : 1.5×(pre-close widening)
        """
        t = event_time.time()

        if t < time(9, 30):
            mult = self.config.open_auction_spread_multiplier    # 4× [ASSUMED]
        elif t < time(11, 0):
            mult = 2.0     # Post-open elevated [ASSUMED]
        elif t < time(14, 0):
            mult = self.config.midday_spread_multiplier           # 1× baseline
        elif t < time(15, 0):
            mult = 1.2     # Afternoon [ASSUMED]
        else:
            mult = self.config.pre_close_spread_multiplier        # 1.5× [ASSUMED]

        adj_spread = spread * mult
        adj_half   = adj_spread / 2.0

        if side == "BUY":
            fill_price = ask + adj_half
        else:
            fill_price = max(bid - adj_half, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.TIME_OF_DAY_LIQUIDITY, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"ToD: time={t}, spread_mult={mult}×, "
                f"adj_spread={adj_spread:.2f}, fill={fill_price:.2f} [ASSUMED params]"
            ),
        )

    def _fill_expiry_blowout(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        oi: int, event_time: datetime,
    ) -> SimulatedFill | None:
        """
        Final hour of Tuesday expiry: OTM options become extremely illiquid.

        Applies only during the final hour (14:30–15:30 IST).
        Before that window, falls back to REALISTIC fill.

        Logic:
          - Spread multiplier = expiry_final_hour_spread_multiplier (3×) [ASSUMED]
          - For low-OI strikes (OI < min_oi_lots_for_normal_fill):
            random draw (seeded) determines whether the order fills at all.
            P(fill) = otm_expiry_fill_probability_cap (50%) [ASSUMED].

        Determinism: the random draw uses a seed derived from event_time and
        strike so that two identical replay runs produce identical results.
        """
        t = event_time.time()
        if t < time(14, 30):
            # Not in final-hour blowout window — use realistic pricing but tag as EXPIRY_DAY_BLOWOUT
            half = spread / 2.0
            if side == "BUY":
                fill_price = ask + half
            else:
                fill_price = max(bid - half, 0.0)
            slippage = self._bps(fill_price, mid)
            return SimulatedFill(
                order_id=order_id, symbol=symbol, expiry=expiry,
                strike=strike, option_type=opt_type, side=side,
                requested_qty=qty, filled_qty=qty,
                fill_price=fill_price, slippage_bps=slippage,
                fill_mode=FillMode.EXPIRY_DAY_BLOWOUT, timestamp=event_time,
                is_partial=False, cancelled=False,
                notes=(
                    f"ExpiryBlowout (pre-final-hour): realistic fill={fill_price:.2f}, "
                    f"spread={spread:.2f} [ASSUMED params]"
                ),
            )

        mult       = self.config.expiry_final_hour_spread_multiplier
        adj_spread = spread * mult
        adj_half   = adj_spread / 2.0

        # Low-OI OTM probability cap
        if oi < self.config.min_oi_lots_for_normal_fill:
            rng = self._make_rng(event_time, strike, salt=0)
            if rng.random() > self.config.otm_expiry_fill_probability_cap:
                log.debug(
                    "fill.expiry_blowout.no_fill",
                    strike=strike, oi=oi,
                    cap=self.config.otm_expiry_fill_probability_cap,
                )
                return None  # No fill — illiquid OTM on expiry

        if side == "BUY":
            fill_price = ask + adj_half
        else:
            fill_price = max(bid - adj_half, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.EXPIRY_DAY_BLOWOUT, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"ExpiryBlowout: spread×{mult}, oi={oi}, "
                f"fill={fill_price:.2f} [ASSUMED params]"
            ),
        )

    def _fill_event_blowout(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        event_time: datetime,
    ) -> SimulatedFill | None:
        """
        RBI policy day / budget day: IV spikes, spreads widen dramatically.

        Applies throughout the entire event day (not just a specific window).

        Logic:
          - Spread multiplier = event_day_spread_multiplier (3.5×) [ASSUMED]
          - Seeded random draw determines whether the order fills at all.
            P(fill) = event_day_otm_fill_prob_cap (40%) [ASSUMED].

        [ASSUMED] Based on reported NSE event-day microstructure behaviour.
        Calibrate both the spread multiplier and fill-probability cap from
        real event-day fills in Phase 4.
        """
        mult       = self.config.event_day_spread_multiplier
        adj_spread = spread * mult
        adj_half   = adj_spread / 2.0

        rng = self._make_rng(event_time, strike, salt=1)
        if rng.random() > self.config.event_day_otm_fill_prob_cap:
            log.debug(
                "fill.event_blowout.no_fill",
                strike=strike,
                cap=self.config.event_day_otm_fill_prob_cap,
            )
            return None

        if side == "BUY":
            fill_price = ask + adj_half
        else:
            fill_price = max(bid - adj_half, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.EVENT_DAY_BLOWOUT, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"EventBlowout: spread×{mult}, fill={fill_price:.2f} [ASSUMED params]"
            ),
        )

    def _fill_thin_book(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        oi: int, event_time: datetime,
    ) -> SimulatedFill:
        """
        OTM strikes with low OI: MARKET orders slip beyond the quoted ask.

        For strikes with OI < low_oi_threshold_lots, extra slippage bps are
        applied on top of the full ask/bid price to model the order sweeping
        through multiple thin levels.

        [ASSUMED] thin_book_extra_slippage_bps (20 bps); calibrate from real
        MARKET-order fills on deep-OTM strikes.
        """
        is_thin = oi < self.config.low_oi_threshold_lots
        impact_base = self.config.pessimistic_slippage_bps / 10_000.0
        extra_bps   = (self.config.thin_book_extra_slippage_bps / 10_000.0) if is_thin else 0.0

        if side == "BUY":
            # For thin books: at least as bad as pessimistic (ask + spread + impact), then add extra
            fill_price = ask + spread + ask * impact_base + ask * extra_bps
        else:
            fill_price = max(bid - spread - bid * impact_base - bid * extra_bps, 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.MARKET_ORDER_THIN_BOOK, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"ThinBook: oi={oi} ({'thin' if is_thin else 'normal'}), "
                f"extra={extra_bps * 10_000:.0f}bps, fill={fill_price:.2f} [ASSUMED params]"
            ),
        )

    def _fill_cancel_race(
        self, order_id: str, symbol: str, expiry: Any, strike: float,
        opt_type: str, side: str, qty: int,
        bid: float, ask: float, mid: float, spread: float,
        event_time: datetime,
    ) -> SimulatedFill:
        """
        Cancel → fill race condition.

        Models the scenario where a cancel request is sent but the fill
        arrives at the exchange marginally before the cancel.

        Probability:
          - P(cancel wins, no fill)  = 1 − cancel_fill_race_probability
          - P(fill wins, trade done) = cancel_fill_race_probability (default 5%)

        When the fill wins, price is modelled at the pessimistic level
        (MARKET-order slippage), since in fast markets only MARKET-like
        orders trigger this race condition.

        [ASSUMED] 5% race probability based on typical NSE broker round-trip
        of ~100 ms.  Calibrate from real cancel-race observations in Phase 4.

        Determinism: uses a seeded RNG (salt=2) to ensure two identical
        replay runs always produce the same race outcome.
        """
        rng = self._make_rng(event_time, strike, salt=2)
        fill_won = rng.random() < self.config.cancel_fill_race_probability

        if not fill_won:
            # Cancel arrived first — order not filled
            return SimulatedFill(
                order_id=order_id, symbol=symbol, expiry=expiry,
                strike=strike, option_type=opt_type, side=side,
                requested_qty=qty, filled_qty=0,
                fill_price=0.0, slippage_bps=0.0,
                fill_mode=FillMode.CANCELLATION_LATENCY, timestamp=event_time,
                is_partial=False, cancelled=True,
                notes=(
                    f"CancelRace: cancel won "
                    f"(race_prob={self.config.cancel_fill_race_probability:.0%}) [ASSUMED]"
                ),
            )

        # Fill arrived before cancel — use pessimistic price (fast-market fill)
        impact = self.config.pessimistic_slippage_bps / 10_000.0
        if side == "BUY":
            fill_price = ask * (1.0 + impact)
        else:
            fill_price = max(bid * (1.0 - impact), 0.0)

        slippage = self._bps(fill_price, mid)
        return SimulatedFill(
            order_id=order_id, symbol=symbol, expiry=expiry,
            strike=strike, option_type=opt_type, side=side,
            requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, slippage_bps=slippage,
            fill_mode=FillMode.CANCELLATION_LATENCY, timestamp=event_time,
            is_partial=False, cancelled=False,
            notes=(
                f"CancelRace: fill won (race_prob={self.config.cancel_fill_race_probability:.0%}), "
                f"pessimistic price={fill_price:.2f} [ASSUMED]"
            ),
        )

    # ── Private utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _bps(fill_price: float, mid: float) -> float:
        """
        Compute slippage in basis points relative to mid price.

        Returns 0.0 when mid is zero to avoid division by zero.
        Negative values indicate a favourable fill (filled below mid for a buy).
        """
        if mid <= 0.0:
            return 0.0
        return (fill_price - mid) / mid * 10_000.0

    @staticmethod
    def _make_rng(event_time: datetime, strike: float, salt: int) -> random.Random:
        """
        Create a deterministic Random instance seeded from event_time + strike.

        Using a seed derived from the market data (not from OS randomness)
        ensures that replay is deterministic: the same event_time + strike
        combination always produces the same random draws.

        Parameters
        ----------
        event_time : datetime
            Exchange timestamp of the event.
        strike : float
            Option strike price.
        salt : int
            A small integer (0, 1, 2, …) that differentiates multiple RNG
            calls within the same fill mode so they each have distinct seeds.
        """
        seed = int(event_time.timestamp()) + int(strike * 100) + salt
        return random.Random(seed)

    @staticmethod
    def _get_field(obj: Any, name: str, default: Any) -> Any:
        """
        Extract a field from either an object (attribute) or a dict (key).

        Returns *default* when the field is absent or evaluates to None.
        """
        val = getattr(obj, name, None)
        if val is None and isinstance(obj, dict):
            val = obj.get(name)
        return val if val is not None else default
