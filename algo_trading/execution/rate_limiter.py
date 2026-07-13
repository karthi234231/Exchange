"""
execution/rate_limiter.py — Dual-lock order rate limiter. [ARCH-FIX-5]

Two independent locks:
  1. Global token bucket: max 8 orders/second (SEBI limit is 10, 2-order buffer).
  2. Per-order modification counter: max 5 modifications per Order ID.
     Beyond 5 → bad trade; caller must cancel + halt the symbol.

Why this matters (from CLAUDE.md):
  SEBI's surveillance flags static IPs that spam order modifications.
  A logic bug (rejected order retrying, or partial fill looping) can send
  hundreds of modifications/second on the same order ID. This limiter
  hard-stops that scenario before SEBI bans the IP.

Design:
  - TokenBucket: leaky bucket, thread-safe, supports sub-second precision.
  - release_order(): must be called on every FILLED/CANCELLED/REJECTED event.
    Orphaned IDs accumulate counts forever if release is skipped.
  - acquire() is synchronous (blocks if rate limit hit) — caller must handle.
    For the hot execution path, call try_acquire() (non-blocking).
"""
from __future__ import annotations

import collections
import threading
import time
from datetime import time as dtime
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

_MAX_GLOBAL_OPS = 8      # Orders per second (SEBI limit 10, 2-order buffer)
_MAX_MODS_PER_ORDER = 5  # Max modifications per single order ID

# P0-FIX-7: Import authoritative session-end time from market_calendar
# instead of hard-coding 15:30 here. If NSE changes the close time,
# only market_calendar.py needs to be updated.
from algo_trading.core.calendar import SESSION_END as _SESSION_END_TIME  # noqa: E402
SESSION_END: dtime = _SESSION_END_TIME


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Tokens refill at `rate` per second up to `capacity`.
    consume(n): blocks until n tokens are available (up to `timeout_sec`).
    try_consume(n): non-blocking, returns False immediately if not enough tokens.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate        # tokens per second
        self._capacity = capacity
        self._tokens = capacity  # Start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Must be called with self._lock held."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        """
        Non-blocking: consume tokens if available.
        Returns True if consumed, False if not enough tokens.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def consume(self, tokens: float = 1.0, timeout_sec: float = 1.0) -> bool:
        """
        Blocking: consume tokens, waiting up to timeout_sec.
        Returns True if consumed, False if timeout expired.
        """
        deadline = time.monotonic() + timeout_sec
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # Wait a fraction of the refill time to avoid busy-looping
            time.sleep(min(tokens / self._rate * 0.1, 0.01))

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class DualLockRateLimiter:
    """
    Global 8 OPS token bucket + per-order modification counter.

    Usage:
        limiter = DualLockRateLimiter()

        # Before placing or modifying an order:
        allowed, reason = limiter.acquire(order_id="abc123")
        if not allowed:
            # Handle rejection: cancel order, halt symbol, log alert

        # After FILLED / CANCELLED / REJECTED:
        limiter.release_order("abc123")
    """

    # P2-FIX-119: SEBI OTR alert threshold — alert when daily OTR exceeds this value.
    # SEBI/NSE define instrument-specific limits; use a conservative default.
    # Set to 0 to disable OTR alerts.
    OTR_ALERT_THRESHOLD: float = 50.0   # orders per trade — alert when exceeded

    def __init__(
        self,
        max_ops: int = _MAX_GLOBAL_OPS,
        max_mods: int = _MAX_MODS_PER_ORDER,
    ) -> None:
        self._bucket = TokenBucket(rate=float(max_ops), capacity=float(max_ops))
        self._max_mods = max_mods
        self._order_counts: dict[str, int] = {}
        self._order_created_at: dict[str, float] = {}  # order_id → monotonic timestamp
        self._halted_orders: set[str] = set()
        self._lock = threading.Lock()
        # Rolling rejection counter: deque of monotonic timestamps for rejections
        self._rejection_timestamps: collections.deque[float] = collections.deque()
        # P2-FIX-119: OTR tracking — daily order and trade counters
        self._daily_orders: int = 0   # total orders submitted today (any type)
        self._daily_trades: int = 0   # total trades executed today (fills)

    def _cleanup_orphaned_orders(self) -> None:
        """Remove order IDs whose creation time predates the session end wall-clock.

        Must be called with ``self._lock`` held.  An order is considered
        orphaned when the session has ended (current wall-clock time is past
        SESSION_END) and the order was created before that boundary —
        meaning ``release_order`` was never called for it.
        """
        import datetime as _dt
        now = _dt.datetime.now().time()
        if now <= SESSION_END:
            return  # Session still active — keep all orders
        # Session has ended; prune any order created before session end today
        session_end_mono_approx = time.monotonic()  # conservative: now
        stale_ids = [
            oid for oid, ts in self._order_created_at.items()
            if (session_end_mono_approx - ts) > 0  # created before "now"
        ]
        for oid in stale_ids:
            self._order_counts.pop(oid, None)
            self._order_created_at.pop(oid, None)
            self._halted_orders.discard(oid)
        if stale_ids:
            log.info(
                "rate_limiter.orphaned_orders_cleaned",
                count=len(stale_ids),
            )

    def acquire(self, order_id: str, block: bool = False) -> tuple[bool, str]:
        """
        Attempt to consume one global rate-limit token and increment
        the per-order modification counter.

        Returns (True, "") if allowed.
        Returns (False, reason) if blocked.

        block=True: waits up to 1 second for a token (useful for non-latency-sensitive paths).
        block=False (default): non-blocking for hot execution path.
        """
        now_mono = time.monotonic()

        # Check per-order halt first (fast path, no token consumed)
        # Also run TTL cleanup of orphaned orders.
        with self._lock:
            self._cleanup_orphaned_orders()

            if order_id in self._halted_orders:
                self._rejection_timestamps.append(now_mono)
                return False, f"Order {order_id} is halted (exceeded {self._max_mods} modifications)"

            count = self._order_counts.get(order_id, 0)
            if count >= self._max_mods:
                self._halted_orders.add(order_id)
                self._rejection_timestamps.append(now_mono)
                log.error(
                    "rate_limiter.order_halted",
                    order_id=order_id,
                    count=count,
                    max_mods=self._max_mods,
                    action="cancel_order + halt_symbol required",
                )
                return False, (
                    f"Order {order_id} exceeded {self._max_mods} modifications — "
                    f"CANCEL the order and halt the symbol immediately"
                )

        # Global token bucket check
        if block:
            ok = self._bucket.consume(1.0, timeout_sec=1.0)
        else:
            ok = self._bucket.try_consume(1.0)

        if not ok:
            with self._lock:
                self._rejection_timestamps.append(now_mono)
            log.warning(
                "rate_limiter.global_limit_hit",
                available=f"{self._bucket.available:.2f}",
                max_ops=_MAX_GLOBAL_OPS,
            )
            return False, f"Global rate limit: max {_MAX_GLOBAL_OPS} orders/second"

        # Increment modification counter and record creation time for new orders
        with self._lock:
            if order_id not in self._order_created_at:
                self._order_created_at[order_id] = now_mono
            self._order_counts[order_id] = self._order_counts.get(order_id, 0) + 1
            new_count = self._order_counts[order_id]
            # P2-FIX-119: increment daily order counter for OTR tracking
            self._daily_orders += 1

        if new_count >= self._max_mods - 1:
            log.warning(
                "rate_limiter.order_near_limit",
                order_id=order_id,
                count=new_count,
                max_mods=self._max_mods,
            )

        # P2-FIX-119: Alert if daily OTR is approaching threshold
        self._check_otr_threshold()

        return True, ""

    def release_order(self, order_id: str) -> None:
        """
        Free the order slot. MUST be called when order reaches a terminal state
        (FILLED, CANCELLED, REJECTED, HALTED).
        Forgetting to call this causes modification counts to grow indefinitely.
        """
        with self._lock:
            self._order_counts.pop(order_id, None)
            self._order_created_at.pop(order_id, None)
            self._halted_orders.discard(order_id)

    def is_order_halted(self, order_id: str) -> bool:
        with self._lock:
            return order_id in self._halted_orders

    def get_modification_count(self, order_id: str) -> int:
        with self._lock:
            return self._order_counts.get(order_id, 0)

    @property
    def available_tokens(self) -> float:
        return self._bucket.available

    def get_stats(self) -> dict:
        """Return current token bucket and order tracking statistics.

        Returns
        -------
        dict with keys:
            ``current_token_count`` (float): tokens available in the bucket right now.
            ``tracked_orders`` (int): number of order IDs currently being tracked.
            ``orders_rejected_last_minute`` (int): rejections in the last 60 seconds.
        """
        cutoff = time.monotonic() - 60.0
        with self._lock:
            # Purge stale rejection timestamps (older than 1 minute)
            while self._rejection_timestamps and self._rejection_timestamps[0] < cutoff:
                self._rejection_timestamps.popleft()
            rejected_last_minute = len(self._rejection_timestamps)
            tracked = len(self._order_counts)
        return {
            "current_token_count": self._bucket.available,
            "tracked_orders": tracked,
            "orders_rejected_last_minute": rejected_last_minute,
        }

    def record_trade(self) -> None:
        """
        P2-FIX-119: Record a completed trade (fill) for OTR tracking.
        Call this on every fill event from the order executor.
        """
        with self._lock:
            self._daily_trades += 1

    def get_otr(self) -> float:
        """
        P2-FIX-119: Return current order-to-trade ratio for today.
        OTR = total_orders / total_trades (0.0 if no trades yet).
        """
        with self._lock:
            if self._daily_trades == 0:
                return 0.0
            return self._daily_orders / self._daily_trades

    def _check_otr_threshold(self) -> None:
        """Alert if OTR has exceeded the configured warning threshold."""
        if self.OTR_ALERT_THRESHOLD <= 0:
            return
        with self._lock:
            if self._daily_trades == 0:
                return
            otr = self._daily_orders / self._daily_trades
        if otr > self.OTR_ALERT_THRESHOLD:
            log.warning(
                "rate_limiter.otr_threshold_exceeded",
                otr=f"{otr:.1f}",
                threshold=self.OTR_ALERT_THRESHOLD,
                daily_orders=self._daily_orders,
                daily_trades=self._daily_trades,
                note="Review order activity — SEBI OTR monitoring active",
            )

    def reset(self) -> None:
        """Reset all state. Only for testing and end-of-day cleanup."""
        with self._lock:
            self._order_counts.clear()
            self._order_created_at.clear()
            self._halted_orders.clear()
            self._rejection_timestamps.clear()
            self._daily_orders = 0
            self._daily_trades = 0
