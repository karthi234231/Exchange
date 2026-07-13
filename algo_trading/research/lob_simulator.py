"""
algo_trading/research/lob_simulator.py — Realistic LOB execution simulator.

Extends fill_simulator.py with queue-position modeling, cancel/replace latency,
and broker-disconnect simulation for more realistic backtesting.

Key models:
1. QueuePositionModel: Estimates fill probability based on queue position
2. CancelReplaceLatencyModel: Adds realistic cancel/replace delays
3. BrokerDisconnectSimulator: Simulates intermittent broker connectivity issues

These are used in backtesting to avoid the "always filled at best price"
assumption that inflates backtest profitability.

All probability/latency values marked [ASSUMED] must be calibrated from
real fills and connectivity logs in Phase 4.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# QueuePositionModel
# ---------------------------------------------------------------------------

@dataclass
class QueueFillResult:
    """Result of a queue-position fill estimation.

    Attributes:
        order_id: Unique identifier of the order.
        submitted_qty: Quantity submitted to the exchange.
        filled_qty: Estimated quantity that will be filled.
        avg_fill_price: Estimated average fill price (0.0 if unfilled).
        fill_probability: Estimated probability of fill in the given bar window.
        queue_position: Estimated position in the visible queue (worst-case = depth).
        bars_to_fill: Estimated number of bars until full fill at current turnover.
    """

    order_id: str
    submitted_qty: int
    filled_qty: int
    avg_fill_price: float
    fill_probability: float
    queue_position: int
    bars_to_fill: int


class QueuePositionModel:
    """Estimates limit order fill probability based on visible queue depth.

    Assumptions (NSE liquid stocks, [ASSUMED] — calibrate from L2 data):
      - Queue turnover ≈ 3× visible depth per 15-min bar.
      - Conservative: assume we are at the BACK of the queue (worst case).
      - Adverse selection discount: 70% of theoretical fills are adverse.

    Formula::

        queue_position = visible_depth   (worst case: back of queue)
        turnover_est   = visible_depth × turnover_ratio × n_bars
        fill_prob      = min(turnover_est / (visible_depth + order_qty), 1.0)
                         × adverse_discount

    Args:
        turnover_ratio: Queue turnover multiplier per bar. [ASSUMED 3.0]
        adverse_discount: Fraction of theoretical fills that are non-adverse.
            [ASSUMED 0.7 — 30% of fills experience adverse selection]

    Example::

        model = QueuePositionModel()
        result = model.estimate_fill("O-001", order_qty=500, visible_depth=2000)
        if result.fill_probability >= 0.6:
            # Likely to fill — proceed with limit order
            ...
    """

    def __init__(
        self,
        turnover_ratio: float = 3.0,
        adverse_discount: float = 0.7,
    ) -> None:
        if turnover_ratio <= 0:
            raise ValueError(f"turnover_ratio must be > 0, got {turnover_ratio}")
        if not (0.0 < adverse_discount <= 1.0):
            raise ValueError(f"adverse_discount must be in (0, 1], got {adverse_discount}")
        self._turnover_ratio = turnover_ratio
        self._adverse_discount = adverse_discount

    def estimate_fill(
        self,
        order_id: str,
        order_qty: int,
        visible_depth: int,
        n_bars: int = 1,
    ) -> QueueFillResult:
        """Estimate fill probability and queue metrics for a limit order.

        Args:
            order_id: Unique order identifier (used for logging only).
            order_qty: Order size in shares.
            visible_depth: Shares visible at the limit price in the order book.
            n_bars: Number of bars the order remains live (multiplies turnover).

        Returns:
            QueueFillResult with fill probability, estimated filled qty, and
            queue position.  filled_qty is 0 when fill_probability < 0.3
            [ASSUMED minimum threshold].

        Notes:
            When visible_depth is 0, the order is assumed to be at the top of an
            empty queue — fill_probability is set to 1.0 (immediate fill).
        """
        order_qty = max(1, order_qty)
        n_bars = max(1, n_bars)

        # Edge case: empty queue → no competition, fill immediately
        if visible_depth <= 0:
            log.debug(
                "lob_simulator.queue.empty_depth",
                order_id=order_id,
                note="visible_depth=0, assuming immediate fill",
            )
            return QueueFillResult(
                order_id=order_id,
                submitted_qty=order_qty,
                filled_qty=order_qty,
                avg_fill_price=0.0,   # caller must supply price
                fill_probability=1.0,
                queue_position=0,
                bars_to_fill=1,
            )

        # Worst-case: we are at the back of the visible queue
        queue_position: int = visible_depth

        # Total turnover over n_bars [ASSUMED: 3× depth per bar]
        turnover_est: float = visible_depth * self._turnover_ratio * n_bars

        # Raw fill probability before adverse selection adjustment
        raw_fill_prob: float = min(
            turnover_est / (visible_depth + order_qty), 1.0
        )

        # Apply adverse selection discount
        fill_prob: float = raw_fill_prob * self._adverse_discount
        fill_prob = min(fill_prob, 1.0)

        # Minimum fill threshold [ASSUMED 0.3]
        _MIN_FILL_THRESHOLD = 0.3
        if fill_prob < _MIN_FILL_THRESHOLD:
            filled_qty = 0
        else:
            # Partial fill: proportion of order_qty based on probability
            filled_qty = min(order_qty, int(order_qty * fill_prob))

        # Estimated bars to full fill: how many bar-turnovers to clear our position
        # bars_to_fill = ceil((queue_position + order_qty) / (depth × turnover_ratio))
        denominator = max(visible_depth * self._turnover_ratio, 1.0)
        bars_to_fill: int = math.ceil((queue_position + order_qty) / denominator)

        log.debug(
            "lob_simulator.queue.estimate_fill",
            order_id=order_id,
            order_qty=order_qty,
            visible_depth=visible_depth,
            queue_position=queue_position,
            fill_probability=round(fill_prob, 4),
            filled_qty=filled_qty,
            bars_to_fill=bars_to_fill,
        )
        return QueueFillResult(
            order_id=order_id,
            submitted_qty=order_qty,
            filled_qty=filled_qty,
            avg_fill_price=0.0,   # price is market data, not modelled here
            fill_probability=fill_prob,
            queue_position=queue_position,
            bars_to_fill=bars_to_fill,
        )

    def should_use_market_order(
        self,
        order_qty: int,
        visible_depth: int,
        min_fill_probability: float = 0.6,
    ) -> bool:
        """Return True if limit fill probability is too low — use market instead.

        Uses estimate_fill with n_bars=1 to determine whether a single-bar
        limit fill probability meets the caller's minimum threshold.  If the
        limit order is unlikely to fill, the caller should submit a market
        order to guarantee execution.

        Args:
            order_qty: Order size in shares.
            visible_depth: Shares visible at the limit price.
            min_fill_probability: Minimum acceptable single-bar fill probability.
                [ASSUMED 0.6 default — calibrate from real fill rate data]

        Returns:
            True if the caller should use a market order instead of a limit order.
        """
        result = self.estimate_fill(
            order_id="_check",
            order_qty=order_qty,
            visible_depth=visible_depth,
            n_bars=1,
        )
        use_market = result.fill_probability < min_fill_probability
        log.debug(
            "lob_simulator.queue.should_use_market",
            fill_probability=round(result.fill_probability, 4),
            min_threshold=min_fill_probability,
            use_market=use_market,
        )
        return use_market


# ---------------------------------------------------------------------------
# CancelReplaceLatencyModel
# ---------------------------------------------------------------------------

@dataclass
class LatencyProfile:
    """Realistic NSE order-operation latency profile from a Mumbai VPS.

    All values are in milliseconds [ASSUMED based on retail VPS observations;
    calibrate from real broker API logs in Phase 4].

    Attributes:
        broker_api_ping_ms: Round-trip ping from Mumbai VPS to broker endpoint (~20-30ms).
        zmq_internal_ms: Internal ZeroMQ IPC latency within the trading process.
        feature_compute_ms: Time to recompute features on bar close.
        treelite_inference_ms: ML model inference time (compiled treelite model).
        order_placement_ms: Time from signal to order submission at broker.
        exchange_ack_ms: Exchange acknowledgement round-trip from broker.
        cancel_processing_ms: Time to process a cancel request at the exchange.
        replace_processing_ms: Time to process a replace request at the exchange.
        jitter_factor: Log-normal jitter std dev as fraction of mean. [ASSUMED 0.3]
    """

    broker_api_ping_ms: float = 25.0
    zmq_internal_ms: float = 1.0
    feature_compute_ms: float = 5.0
    treelite_inference_ms: float = 2.0
    order_placement_ms: float = 25.0
    exchange_ack_ms: float = 20.0
    cancel_processing_ms: float = 15.0
    replace_processing_ms: float = 20.0
    jitter_factor: float = 0.3          # [ASSUMED] std dev as fraction of mean


class CancelReplaceLatencyModel:
    """Adds realistic latency delays to cancel/replace operations in simulation.

    Key insight: a cancel+replace sequence on a Mumbai VPS takes approximately
    35-45 ms.  During this window, the market can move 1-2 ticks against the
    order.  This model simulates that price drift so that backtests do not
    assume costless re-quoting.

    Latency is sampled from a log-normal distribution (not a constant) to
    reflect the real-world jitter observed in broker API latencies.

    Args:
        profile: LatencyProfile with per-operation latencies.  Uses defaults if None.
        seed: Random seed for reproducible simulations.

    Example::

        model = CancelReplaceLatencyModel(seed=42)
        cancel_ms = model.sample_cancel_latency_ms()
        replace_ms = model.sample_replace_latency_ms()
        new_price = model.price_drift_during_cancel(
            price=1500.0, direction="BUY", tick_size=0.05
        )
    """

    def __init__(
        self,
        profile: Optional[LatencyProfile] = None,
        seed: int = 42,
    ) -> None:
        self._profile = profile or LatencyProfile()
        self._rng = random.Random(seed)

    def _sample_lognormal(self, mean_ms: float) -> float:
        """Sample a latency from a log-normal distribution.

        Args:
            mean_ms: Target mean in milliseconds.

        Returns:
            Sample in milliseconds drawn from log-normal with mean=mean_ms
            and sigma = jitter_factor * mean_ms.

        Notes:
            Log-normal parameters are derived from the desired mean and std dev
            using the standard moment-matching formulae::

                mu_ln    = ln(mean) - 0.5 * sigma_ln^2
                sigma_ln = sqrt(ln(1 + (std/mean)^2))
        """
        jitter = self._profile.jitter_factor
        if mean_ms <= 0:
            return 0.0
        # moment-match to log-normal
        sigma_ln = math.sqrt(math.log(1.0 + jitter ** 2))
        mu_ln = math.log(mean_ms) - 0.5 * sigma_ln ** 2
        # gauss gives N(0,1); transform to log-normal
        sample = math.exp(mu_ln + sigma_ln * self._rng.gauss(0.0, 1.0))
        return max(0.1, sample)  # floor at 0.1 ms

    def sample_cancel_latency_ms(self) -> float:
        """Sample a cancel-operation latency from log-normal distribution.

        Returns:
            Cancel latency in milliseconds. [ASSUMED mean=15ms with 30% jitter]
        """
        latency = self._sample_lognormal(self._profile.cancel_processing_ms)
        log.debug("lob_simulator.latency.cancel_sample_ms", latency_ms=round(latency, 2))
        return latency

    def sample_replace_latency_ms(self) -> float:
        """Sample a replace-operation latency from log-normal distribution.

        Returns:
            Replace latency in milliseconds. [ASSUMED mean=20ms with 30% jitter]
        """
        latency = self._sample_lognormal(self._profile.replace_processing_ms)
        log.debug("lob_simulator.latency.replace_sample_ms", latency_ms=round(latency, 2))
        return latency

    def price_drift_during_cancel(
        self,
        price: float,
        direction: str,
        tick_size: float,
        volatility_per_ms: float = 0.0001,
    ) -> float:
        """Estimate price drift during the cancel/replace window.

        Models Brownian price drift over the total cancel+replace latency
        window.  For a BUY order the drift is adverse (price moves up),
        for a SELL order it is also adverse (price moves down).

        Formula::

            window_ms     = cancel_ms + replace_ms
            price_std_dev = volatility_per_ms * sqrt(window_ms) * price
            drift         = abs(N(0, price_std_dev))   # always adverse
            new_price     = price ± drift

        Args:
            price: Current mid-price in INR.
            direction: "BUY" or "SELL".
            tick_size: Minimum price increment for the instrument (e.g. 0.05).
            volatility_per_ms: Price std dev per ms as fraction of price.
                [ASSUMED 0.0001 = 0.01% per ms for NSE large-cap]

        Returns:
            New price after adverse drift during cancel/replace window.
            Always moves against the direction (adversely).
        """
        cancel_ms = self.sample_cancel_latency_ms()
        replace_ms = self.sample_replace_latency_ms()
        window_ms = cancel_ms + replace_ms

        # Price std dev over the window (scaled Brownian motion)
        price_std = volatility_per_ms * math.sqrt(max(window_ms, 0.1)) * price

        # Always adverse: use absolute value of normal sample
        drift = abs(self._rng.gauss(0.0, price_std))

        # Round drift to nearest tick
        if tick_size > 0:
            drift = round(drift / tick_size) * tick_size

        if direction.upper() == "BUY":
            # Price rose while we were cancelling — we now pay more
            new_price = price + drift
        else:
            # Price fell while we were cancelling — we now receive less
            new_price = max(0.0, price - drift)

        log.debug(
            "lob_simulator.latency.price_drift",
            price=price,
            direction=direction,
            window_ms=round(window_ms, 2),
            drift=round(drift, 4),
            new_price=round(new_price, 4),
        )
        return new_price


# ---------------------------------------------------------------------------
# BrokerDisconnectSimulator
# ---------------------------------------------------------------------------

@dataclass
class DisconnectEvent:
    """Represents a single broker disconnect episode.

    Attributes:
        start_bar: Bar index when the disconnect begins.
        duration_bars: Number of bars the disconnect lasts.
        reconnect_with_state_mismatch: True when on-reconnect state
            reconciliation is needed (see SURV-6).
    """

    start_bar: int
    duration_bars: int
    reconnect_with_state_mismatch: bool


class BrokerDisconnectSimulator:
    """Simulates intermittent broker connectivity failures.

    Typical NSE retail trading-day disconnect profile [ASSUMED based on
    retail co-location and retail VPS observations; NOT verified against
    Shoonya SLA data]:

      - ~1-2 disconnects per week on average
      - Duration: 1-5 minutes (1-5 bars at 15-min timeframe)
      - Higher probability during: market open (9:15-9:30), high volatility
        periods, and F&O expiry days

    When disconnected:
      - No new orders can be placed (is_connected returns False).
      - Open orders at the exchange may or may not have been received before
        the disconnect — simulate_reconnect_scenario captures this ambiguity.
      - On reconnect: state MUST be reconciled before placing new orders
        (see SURV-6 in the system design).

    Disconnect events are generated lazily: the first call to is_connected()
    for a new bar index triggers event generation for that bar.

    Args:
        disconnect_prob_per_bar: Probability of a NEW disconnect starting per
            bar.  Default 0.002 ≈ 1 disconnect per 500 bars ≈ ~2/week at
            15-min bars.  [ASSUMED]
        min_duration_bars: Minimum disconnect duration in bars. [ASSUMED 1]
        max_duration_bars: Maximum disconnect duration in bars. [ASSUMED 5]
        seed: Random seed for reproducible simulation.

    Example::

        sim = BrokerDisconnectSimulator(seed=42)
        for bar_idx in range(total_bars):
            if not sim.is_connected(bar_idx):
                # skip order submission this bar
                continue
            # ... normal trading logic
        events = sim.get_disconnect_events()
    """

    def __init__(
        self,
        disconnect_prob_per_bar: float = 0.002,
        min_duration_bars: int = 1,
        max_duration_bars: int = 5,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= disconnect_prob_per_bar <= 1.0):
            raise ValueError(
                f"disconnect_prob_per_bar must be in [0, 1], got {disconnect_prob_per_bar}"
            )
        if min_duration_bars < 1:
            raise ValueError("min_duration_bars must be >= 1")
        if max_duration_bars < min_duration_bars:
            raise ValueError("max_duration_bars must be >= min_duration_bars")

        self._prob = disconnect_prob_per_bar
        self._min_dur = min_duration_bars
        self._max_dur = max_duration_bars
        self._rng = random.Random(seed)

        # Mapping from start_bar → DisconnectEvent
        self._events: dict[int, DisconnectEvent] = {}
        # Set of bars that are inside any disconnect window
        self._disconnected_bars: set[int] = set()
        # Highest bar index for which we've already decided connectivity
        self._evaluated_up_to: int = -1

    def _evaluate_bar(self, bar_idx: int) -> None:
        """Evaluate and record connectivity state for a single bar.

        Args:
            bar_idx: Bar index to evaluate.  Must not have been evaluated before.
        """
        # Only start a new disconnect if we're currently connected
        if bar_idx in self._disconnected_bars:
            return  # Already in a disconnect window, don't start another

        if self._rng.random() < self._prob:
            duration = self._rng.randint(self._min_dur, self._max_dur)
            # State mismatch: 40% of reconnects have ambiguous order state [ASSUMED]
            state_mismatch = self._rng.random() < 0.4

            event = DisconnectEvent(
                start_bar=bar_idx,
                duration_bars=duration,
                reconnect_with_state_mismatch=state_mismatch,
            )
            self._events[bar_idx] = event

            # Mark all bars in this window as disconnected
            for offset in range(duration):
                self._disconnected_bars.add(bar_idx + offset)

            log.info(
                "lob_simulator.disconnect.event_generated",
                start_bar=bar_idx,
                duration_bars=duration,
                state_mismatch=state_mismatch,
            )

    def is_connected(self, bar_idx: int) -> bool:
        """Return True if the broker is connected at the given bar.

        Evaluates all bars up to bar_idx that haven't been evaluated yet.
        Safe to call for any bar_idx in any order, but intended to be called
        sequentially (bar 0, 1, 2, ...).

        Args:
            bar_idx: Zero-based bar index in the simulation.

        Returns:
            True if connected (no active disconnect), False otherwise.
        """
        # Evaluate any unevaluated bars up to this one
        for idx in range(self._evaluated_up_to + 1, bar_idx + 1):
            self._evaluate_bar(idx)
        self._evaluated_up_to = max(self._evaluated_up_to, bar_idx)

        connected = bar_idx not in self._disconnected_bars
        if not connected:
            log.debug(
                "lob_simulator.disconnect.bar_disconnected",
                bar_idx=bar_idx,
            )
        return connected

    def get_disconnect_events(self) -> list[DisconnectEvent]:
        """Return all disconnect events generated so far.

        Returns:
            List of DisconnectEvent instances, sorted by start_bar.
        """
        return sorted(self._events.values(), key=lambda e: e.start_bar)

    def simulate_reconnect_scenario(self, disconnect: DisconnectEvent) -> dict:
        """Simulate the outcome of reconnecting after a disconnect.

        On reconnect, the system must determine what happened to open orders
        during the disconnection window.  Three outcomes are possible:

          - "filled": the order was received and executed during the disconnect.
          - "cancelled": NSE IOC/GTD timer expired, order was cancelled by exchange.
          - "unknown": state is ambiguous — requires full reconciliation via
            order book query (SURV-6).

        Args:
            disconnect: The DisconnectEvent to simulate reconnect for.

        Returns:
            Dictionary with keys:
              - "pending_orders_fate" (str): one of "filled", "cancelled", "unknown".
              - "needs_reconciliation" (bool): True if full order-book reconciliation
                is required before placing new orders.
              - "reconnect_delay_bars" (int): extra bars needed before trading resumes.

        Notes:
            All probability values are [ASSUMED] and must be calibrated from
            real Shoonya disconnect/reconnect logs.
        """
        fate_roll = self._rng.random()

        if disconnect.reconnect_with_state_mismatch:
            # Ambiguous state — must reconcile
            pending_fate = "unknown"
            needs_reconciliation = True
        elif fate_roll < 0.45:
            # [ASSUMED] 45% chance order was filled before disconnect
            pending_fate = "filled"
            needs_reconciliation = False
        elif fate_roll < 0.80:
            # [ASSUMED] 35% chance order was cancelled by exchange timer
            pending_fate = "cancelled"
            needs_reconciliation = False
        else:
            # [ASSUMED] 20% chance of unknown state
            pending_fate = "unknown"
            needs_reconciliation = True

        # Extra bars to wait before safe to resume trading [ASSUMED 1 bar minimum]
        reconnect_delay = 1 if not needs_reconciliation else 2

        result = {
            "pending_orders_fate": pending_fate,
            "needs_reconciliation": needs_reconciliation,
            "reconnect_delay_bars": reconnect_delay,
        }
        log.info(
            "lob_simulator.disconnect.reconnect_scenario",
            start_bar=disconnect.start_bar,
            duration_bars=disconnect.duration_bars,
            **result,
        )
        return result
