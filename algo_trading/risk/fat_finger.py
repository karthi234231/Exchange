"""
risk/fat_finger.py — Fat-finger guard for order placement.

Blocks orders that are outsized relative to:
  1. Absolute value (Rs. cap)
  2. Average Daily Volume percentage (market impact prevention)
  3. Price deviation from Last Traded Price (rogue price guard for LIMIT orders)
  4. Absolute quantity cap per symbol

This is the FIRST check in OrderExecutor.submit_signal() — if it fires,
the order never reaches the rate limiter or broker.

Design:
  - FatFingerConfig: injectable; defaults load from settings.
  - ADVCache: thread-safe LRU with TTL — computes ADV from QuestDB or falls back
    to a configurable safe default when QuestDB is unavailable.
  - FatFingerGuard.check(): returns FatFingerResult (never raises in hot path).
  - All monetary comparisons use Decimal.
  - Logs every rejection at ERROR level with full context.
"""
from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Tuple

import structlog

from algo_trading.config import settings
from algo_trading.data.broker_base import Order

log = structlog.get_logger(__name__)

# ── Types ────────────────────────────────────────────────────────────────────

_ADVEntry = tuple[float, float]  # (adv_shares, cached_at_monotonic)


# ── Enum ─────────────────────────────────────────────────────────────────────

class FatFingerResult(Enum):
    """Result codes returned by FatFingerGuard.check()."""
    OK = "OK"
    REJECT_OVER_VALUE = "REJECT_OVER_VALUE"
    REJECT_OVER_ADV = "REJECT_OVER_ADV"
    REJECT_PRICE_DEVIATION = "REJECT_PRICE_DEVIATION"
    REJECT_QTY_CAP = "REJECT_QTY_CAP"
    # Fix F: returned when an internal guard exception occurs (fail-closed)
    REJECT_GUARD_ERROR = "REJECT_GUARD_ERROR"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FatFingerConfig:
    """
    Configuration for fat-finger checks.
    All monetary values in INR. All defaults are conservative.
    """
    # Maximum order value in INR (default Rs. 2L)
    max_order_value_inr: Decimal = field(
        default_factory=lambda: Decimal("200000")
    )
    # Maximum percentage of symbol's Average Daily Volume (0.02 = 2%)
    max_adv_pct: float = 0.02
    # Maximum deviation from LTP for LIMIT orders in basis points (50bps = 0.5%)
    max_price_deviation_bps: float = 50.0
    # Maximum absolute quantity per single order
    max_quantity: int = 5000
    # ADV cache TTL in seconds (default: 1 hour)
    adv_cache_ttl_sec: float = 3600.0
    # Fallback ADV (shares) used when QuestDB is unavailable. None = skip ADV check.
    adv_fallback: Optional[float] = None


def _default_config() -> FatFingerConfig:
    """Build FatFingerConfig from settings (module defaults)."""
    return FatFingerConfig(
        max_order_value_inr=Decimal("200000"),
        max_adv_pct=0.02,
        max_price_deviation_bps=50.0,
        max_quantity=5000,
        adv_cache_ttl_sec=3600.0,
        adv_fallback=None,
    )


# ── ADV Cache ─────────────────────────────────────────────────────────────────

class ADVCache:
    """
    Thread-safe in-process cache for Average Daily Volume (in shares) per symbol.

    Entries expire after `ttl_sec` seconds (monotonic clock). On expiry the entry
    is removed; callers should repopulate via update() or from QuestDB.

    The cache is intentionally simple (plain dict) — there are at most ~50 symbols
    in an NSE algo portfolio, so LRU eviction overhead is unnecessary.
    """

    def __init__(self, ttl_sec: float = 3600.0) -> None:
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._store: dict[str, _ADVEntry] = {}

    def get(self, symbol: str) -> Optional[float]:
        """
        Return ADV for symbol if cache entry exists and is fresh, else None.
        Thread-safe.
        """
        with self._lock:
            entry = self._store.get(symbol)
            if entry is None:
                return None
            adv_value, cached_at = entry
            if (_time.monotonic() - cached_at) > self._ttl_sec:
                # Expired — evict
                del self._store[symbol]
                return None
            return adv_value

    def update(self, symbol: str, adv_shares: float) -> None:
        """
        Store or refresh ADV for a symbol.
        Thread-safe.
        """
        with self._lock:
            self._store[symbol] = (adv_shares, _time.monotonic())

    def invalidate(self, symbol: str) -> None:
        """Remove a symbol's entry from the cache."""
        with self._lock:
            self._store.pop(symbol, None)

    def clear(self) -> None:
        """Remove all entries (e.g., at market open for forced refresh)."""
        with self._lock:
            self._store.clear()

    def invalidate_stale(self, session_end_time: datetime) -> None:
        """
        Fix G: Remove all cache entries that were computed before *session_end_time*.

        Call this at the start of each new session (e.g., from
        ``RiskManager.reset_for_new_day()``) to ensure ADV values from the
        previous day are not carried forward.

        Args:
            session_end_time: Any cache entry whose wall-clock creation time
                              predates this moment will be evicted.  Typically
                              pass the previous session's 15:30 IST close.
        """
        # Convert the session_end_time to a monotonic reference point.
        # monotonic() has no epoch, so we compute how many seconds ago the
        # session ended and translate that into a monotonic cut-off.
        # Use tz-aware now() if session_end_time is tz-aware, else naive — prevents TypeError
        if session_end_time.tzinfo is not None:
            now_wall = datetime.now(session_end_time.tzinfo)
        else:
            now_wall = datetime.now()
        seconds_since_session_end = (now_wall - session_end_time).total_seconds()
        monotonic_cutoff = _time.monotonic() - seconds_since_session_end

        with self._lock:
            stale_keys = [
                sym for sym, (_, cached_at) in self._store.items()
                if cached_at < monotonic_cutoff
            ]
            for sym in stale_keys:
                del self._store[sym]

        if stale_keys:
            log.info(
                "adv_cache.invalidated_stale",
                count=len(stale_keys),
                symbols=stale_keys,
                session_end_time=session_end_time.isoformat(),
            )

    def size(self) -> int:
        """Return number of cached (possibly stale) entries."""
        with self._lock:
            return len(self._store)


# ── Guard ─────────────────────────────────────────────────────────────────────

class FatFingerGuard:
    """
    Fat-finger pre-order guard.

    Usage:
        guard = FatFingerGuard()  # defaults from settings
        result, reason = guard.check(order, ltp=Decimal("2850.00"))
        if result != FatFingerResult.OK:
            log.error("fat_finger.blocked", reason=reason)
            return None  # don't submit to broker

    The check() method NEVER raises — all exceptions are caught and logged,
    and REJECT_GUARD_ERROR is returned on internal error (fail-closed to prevent
    rogue orders from slipping through a broken guard).
    """

    def __init__(
        self,
        config: Optional[FatFingerConfig] = None,
        questdb_store: Optional[object] = None,
    ) -> None:
        self._config = config if config is not None else _default_config()
        self._questdb_store = questdb_store
        self._adv_cache = ADVCache(ttl_sec=self._config.adv_cache_ttl_sec)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        order: Order,
        ltp: Decimal,
        adv_override: Optional[float] = None,
    ) -> Tuple[FatFingerResult, str]:
        """
        Run all fat-finger checks on the order.

        Args:
            order:        The Order to validate (immutable).
            ltp:          Last Traded Price at time of check (Decimal).
            adv_override: If provided, use this ADV value instead of cache/QuestDB.
                          Useful for backtesting and unit tests.

        Returns:
            (FatFingerResult, human_readable_reason)

        Never raises. On internal error returns (OK, "internal-error").
        """
        try:
            return self._check_impl(order, ltp, adv_override)
        except Exception as exc:
            log.error(
                "fat_finger.internal_error",
                symbol=order.symbol,
                error=str(exc),
                exc_info=True,
            )
            # Fix F: Fail-closed — an internal guard error must never let a
            # potentially rogue order through.
            return FatFingerResult.REJECT_GUARD_ERROR, f"guard_internal_error: {exc}"

    def update_adv(self, symbol: str, adv: float) -> None:
        """
        Update the ADV cache for a symbol.

        Call daily at market open with fresh ADV data from QuestDB or a
        data provider. ADV is shares traded on average per day (30-day rolling).

        Args:
            symbol: NSE symbol (e.g. "RELIANCE").
            adv:    Average daily volume in shares (float).
        """
        if adv <= 0:
            log.warning("fat_finger.adv_update.invalid", symbol=symbol, adv=adv)
            return
        self._adv_cache.update(symbol, adv)
        log.debug("fat_finger.adv_updated", symbol=symbol, adv=f"{adv:,.0f}")

    def invalidate_adv(self, symbol: str) -> None:
        """Force-expire a symbol's cached ADV (e.g., after a stock split)."""
        self._adv_cache.invalidate(symbol)

    def clear_adv_cache(self) -> None:
        """Clear all cached ADV values. Call at market open to force refresh."""
        self._adv_cache.clear()

    def invalidate_stale_adv(self, session_end_time: datetime) -> None:
        """
        Fix G: Delegate to ADVCache.invalidate_stale().

        Call from ``RiskManager.reset_for_new_day()`` passing the previous
        session's end time so stale ADV entries do not carry over into the
        new session.

        Example::
            guard.invalidate_stale_adv(datetime(2026, 6, 21, 15, 30, tzinfo=IST))
        """
        self._adv_cache.invalidate_stale(session_end_time)

    # ── Implementation ────────────────────────────────────────────────────────

    def _check_impl(
        self,
        order: Order,
        ltp: Decimal,
        adv_override: Optional[float],
    ) -> Tuple[FatFingerResult, str]:
        cfg = self._config
        symbol = order.symbol
        qty = order.quantity

        # ── 1. Absolute quantity cap ──────────────────────────────────────────
        if qty > cfg.max_quantity:
            reason = (
                f"[{symbol}] Quantity {qty:,} exceeds cap {cfg.max_quantity:,} — "
                f"reduce order size"
            )
            log.error(
                "fat_finger.reject.qty_cap",
                symbol=symbol,
                quantity=qty,
                cap=cfg.max_quantity,
            )
            return FatFingerResult.REJECT_QTY_CAP, reason

        # Classify order type up front for bypass logic.
        # STOP_LOSS MARKET orders (trigger_price > 0) are safety-critical exits
        # and bypass both ADV and value checks so they can never be blocked.
        # is_exit is True when reducing/closing an existing position.
        is_stop_loss_market = (
            order.order_type == "MARKET"
            and order.trigger_price > Decimal(0)
        )
        # is_exit is set by OrderExecutor for force-close and stop-loss exits.
        is_exit = getattr(order, "is_exit", False)
        _bypass_safety = is_stop_loss_market or is_exit

        # ── 2. ADV percentage check ───────────────────────────────────────────
        # Checked before value so that an ADV violation surfaces first when both
        # constraints would fire (e.g. a large MARKET order with ADV configured).
        # ADV check is SKIPPED when no ADV data is available and no fallback is
        # configured — the guard degrades gracefully rather than fail-closing.
        if not _bypass_safety:
            adv = adv_override if adv_override is not None else self._get_adv(symbol)
            if adv is not None and adv > 0:
                adv_threshold = adv * cfg.max_adv_pct
                if qty > adv_threshold:
                    reason = (
                        f"[{symbol}] Quantity {qty:,} exceeds "
                        f"{cfg.max_adv_pct * 100:.1f}% of ADV "
                        f"({adv:,.0f} shares) — threshold is {adv_threshold:,.0f} shares"
                    )
                    log.error(
                        "fat_finger.reject.over_adv",
                        symbol=symbol,
                        quantity=qty,
                        adv=f"{adv:,.0f}",
                        adv_threshold=f"{adv_threshold:,.0f}",
                        max_adv_pct=cfg.max_adv_pct,
                    )
                    return FatFingerResult.REJECT_OVER_ADV, reason

        # ── 3. Absolute order value cap (INR) ─────────────────────────────────
        # Stop-loss MARKET orders and force-close exits bypass this check —
        # they must always execute to protect open positions.
        if not _bypass_safety:
            # Use limit price for LIMIT orders; use LTP for MARKET orders
            if order.order_type == "LIMIT" and order.price > 0:
                effective_price = order.price
            elif ltp > 0:
                effective_price = ltp
            else:
                effective_price = Decimal(0)

            if effective_price > 0:
                order_value = Decimal(str(qty)) * effective_price
                if order_value > cfg.max_order_value_inr:
                    reason = (
                        f"[{symbol}] Order value ₹{order_value:,.2f} exceeds limit "
                        f"₹{cfg.max_order_value_inr:,.2f} (qty={qty}, px={effective_price})"
                    )
                    log.error(
                        "fat_finger.reject.over_value",
                        symbol=symbol,
                        order_value=str(order_value),
                        limit=str(cfg.max_order_value_inr),
                        qty=qty,
                        price=str(effective_price),
                    )
                    return FatFingerResult.REJECT_OVER_VALUE, reason
        else:
            # Even for bypass orders, still run value check when NOT a stop-loss
            # MARKET (i.e., plain exit orders tagged is_exit only).
            # Stop-loss MARKET orders skip value entirely.
            if not is_stop_loss_market:
                if order.order_type == "LIMIT" and order.price > 0:
                    effective_price = order.price
                elif ltp > 0:
                    effective_price = ltp
                else:
                    effective_price = Decimal(0)

                if effective_price > 0:
                    order_value = Decimal(str(qty)) * effective_price
                    if order_value > cfg.max_order_value_inr:
                        reason = (
                            f"[{symbol}] Order value ₹{order_value:,.2f} exceeds limit "
                            f"₹{cfg.max_order_value_inr:,.2f} (qty={qty}, px={effective_price})"
                        )
                        log.error(
                            "fat_finger.reject.over_value",
                            symbol=symbol,
                            order_value=str(order_value),
                            limit=str(cfg.max_order_value_inr),
                            qty=qty,
                            price=str(effective_price),
                        )
                        return FatFingerResult.REJECT_OVER_VALUE, reason

        # ── 4. LIMIT price deviation from LTP ─────────────────────────────────
        if order.order_type == "LIMIT" and ltp > 0 and order.price > 0:
            deviation_bps = (
                abs(order.price - ltp) / ltp * Decimal("10000")
            )
            threshold = Decimal(str(cfg.max_price_deviation_bps))
            if deviation_bps > threshold:
                reason = (
                    f"[{symbol}] LIMIT price {order.price} deviates "
                    f"{float(deviation_bps):.1f}bps from LTP {ltp} — "
                    f"threshold is {cfg.max_price_deviation_bps:.1f}bps"
                )
                log.error(
                    "fat_finger.reject.price_deviation",
                    symbol=symbol,
                    limit_price=str(order.price),
                    ltp=str(ltp),
                    deviation_bps=f"{float(deviation_bps):.2f}",
                    threshold_bps=cfg.max_price_deviation_bps,
                )
                return FatFingerResult.REJECT_PRICE_DEVIATION, reason

        return FatFingerResult.OK, ""

    def _get_adv(self, symbol: str) -> Optional[float]:
        """
        Retrieve ADV from cache, then optionally from QuestDB.
        Returns None if unavailable — caller skips ADV check gracefully.
        """
        adv = self._adv_cache.get(symbol)
        if adv is not None:
            return adv

        # Attempt live fetch from QuestDB if store is wired up
        if self._questdb_store is not None:
            try:
                fetched = self._fetch_adv_from_questdb(symbol)
                if fetched is not None and fetched > 0:
                    self._adv_cache.update(symbol, fetched)
                    return fetched
            except Exception as exc:
                log.warning(
                    "fat_finger.adv_fetch_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        # Fall back to configured safe default
        if self._config.adv_fallback is not None:
            return self._config.adv_fallback

        return None

    # NSE symbols are all-caps alphanumeric, optionally ending with -EQ or -BE.
    # Reject anything that doesn't match to prevent SQL injection. [P1-FIX-25]
    _SYMBOL_WHITELIST_RE = None  # Lazy-compiled below

    @staticmethod
    def _validate_symbol(symbol: str) -> bool:
        """Return True if symbol matches the NSE allowed pattern."""
        import re
        if FatFingerGuard._SYMBOL_WHITELIST_RE is None:
            FatFingerGuard._SYMBOL_WHITELIST_RE = re.compile(r'^[A-Z0-9&]{1,20}(-[A-Z]{1,4})?$')
        return bool(FatFingerGuard._SYMBOL_WHITELIST_RE.match(symbol))

    def _fetch_adv_from_questdb(self, symbol: str) -> Optional[float]:
        """
        Query QuestDB for 30-day average daily volume for symbol.
        The store must expose a query() method returning a list of dicts.
        P1-FIX-25: symbol is whitelist-validated before interpolation.
        """
        if self._questdb_store is None:
            return None
        # P1-FIX-25: reject non-whitelisted symbols before they reach the query
        if not self._validate_symbol(symbol):
            log.error(
                "fat_finger.adv_fetch.invalid_symbol",
                symbol=symbol,
                note="Symbol failed whitelist check — ADV query blocked",
            )
            return None
        # Attempt a best-effort query; returns None if the table doesn't exist
        try:
            rows = self._questdb_store.query(  # type: ignore[union-attr]
                f"SELECT avg(volume) AS adv FROM daily_bars "
                f"WHERE symbol = '{symbol}' "
                f"AND timestamp > dateadd('d', -30, now())"
            )
            if rows and rows[0].get("adv") is not None:
                return float(rows[0]["adv"])
        except Exception:
            pass
        return None
