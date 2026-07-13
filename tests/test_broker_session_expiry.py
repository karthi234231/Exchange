"""
tests/test_broker_session_expiry.py — Broker session expiry handling tests.

P2-FIX-98: Mock session expiry before place/cancel.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

from algo_trading.data.broker_base import Order, OrderRejectedError, SessionError

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Minimal ShoonyaBroker stub that exercises session-expiry logic without
# requiring the NorenRestApiPy library or a live Shoonya account.
# ---------------------------------------------------------------------------

_SESSION_MAX_AGE_HOURS = 22  # mirrors broker_shoonya.py


class _FakeSession:
    """Thin wrapper around session state for isolation in tests."""

    def __init__(
        self,
        token: Optional[str] = "valid-token-abc",
        login_time: Optional[datetime] = None,
    ) -> None:
        self.token = token
        self.login_time = login_time or datetime.now(tz=IST)


class _TestableBroker:
    """
    Minimal broker that reproduces ShoonyaBroker's session expiry behaviour.

    The real ShoonyaBroker is not instantiable in tests (requires NorenRestApiPy
    and real credentials), so we replicate just the session-management paths.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = "valid-token",
        login_time: Optional[datetime] = None,
    ) -> None:
        self._token = token
        self._login_time: Optional[datetime] = login_time or datetime.now(tz=IST)
        self._lock = threading.RLock()
        self._login_call_count = 0
        self._login_should_fail = False
        self._placed_orders: list[Order] = []
        self._cancelled: list[str] = []

    # ── Session internals ─────────────────────────────────────────────────────

    def _is_session_expired(self) -> bool:
        with self._lock:
            if not self._token or not self._login_time:
                return True
            age = datetime.now(tz=IST) - self._login_time
            return age.total_seconds() / 3600 >= _SESSION_MAX_AGE_HOURS

    def login(self) -> bool:
        self._login_call_count += 1
        if self._login_should_fail:
            raise SessionError("Simulated relogin failure")
        with self._lock:
            self._token = f"refreshed-token-{self._login_call_count}"
            self._login_time = datetime.now(tz=IST)
        return True

    def ensure_session(self) -> None:
        """Mirrors ShoonyaBroker.ensure_session()."""
        if self._is_session_expired():
            self.login()

    # ── Order operations ──────────────────────────────────────────────────────

    def place_order(self, order: Order) -> str:
        self.ensure_session()
        self._placed_orders.append(order)
        return f"EXCH-{len(self._placed_orders):05d}"

    def cancel_order(self, exchange_order_id: str) -> bool:
        self.ensure_session()
        self._cancelled.append(exchange_order_id)
        return True


def _make_order(symbol: str = "RELIANCE-EQ") -> Order:
    return Order(
        symbol=symbol,
        side="BUY",
        quantity=10,
        order_type="LIMIT",
        price=Decimal("2500"),
        strategy_id="SESSION-TEST",
    )


def _expired_session_broker() -> _TestableBroker:
    """Return a broker whose session is older than SESSION_MAX_AGE_HOURS."""
    old_time = datetime.now(tz=IST) - timedelta(hours=_SESSION_MAX_AGE_HOURS + 1)
    return _TestableBroker(token="old-token", login_time=old_time)


def _fresh_session_broker() -> _TestableBroker:
    """Return a broker whose session was just created (not expired)."""
    return _TestableBroker(token="fresh-token", login_time=datetime.now(tz=IST))


# ---------------------------------------------------------------------------
# Test: place_order triggers relogin on expired session
# ---------------------------------------------------------------------------

class TestPlaceOrderSessionExpiry:
    """P2-FIX-98: place_order must call login() if session is expired."""

    def test_place_order_triggers_relogin_on_expired_session(self) -> None:
        broker = _expired_session_broker()

        # Before placing: session is expired
        assert broker._is_session_expired() is True

        order = _make_order()
        exchange_id = broker.place_order(order)

        # login() must have been called once during ensure_session()
        assert broker._login_call_count == 1, (
            f"Expected 1 login call, got {broker._login_call_count}"
        )
        assert exchange_id.startswith("EXCH-")
        assert len(broker._placed_orders) == 1

    def test_place_order_skips_relogin_on_fresh_session(self) -> None:
        broker = _fresh_session_broker()
        order = _make_order()
        broker.place_order(order)

        assert broker._login_call_count == 0, (
            "Fresh session should not trigger relogin before place_order"
        )

    def test_multiple_orders_with_expired_session_relogin_once(self) -> None:
        """Session expires once → relogin once → subsequent orders do not trigger more logins."""
        broker = _expired_session_broker()
        orders = [_make_order() for _ in range(3)]

        for order in orders:
            broker.place_order(order)

        # Relogin happens on first place_order call; subsequent calls find a fresh session
        assert broker._login_call_count == 1, (
            f"Expected exactly 1 relogin, got {broker._login_call_count}"
        )
        assert len(broker._placed_orders) == 3


# ---------------------------------------------------------------------------
# Test: cancel_order triggers relogin on expired session
# ---------------------------------------------------------------------------

class TestCancelOrderSessionExpiry:
    """P2-FIX-98: cancel_order must call login() if session is expired."""

    def test_cancel_order_triggers_relogin_on_expired_session(self) -> None:
        broker = _expired_session_broker()
        assert broker._is_session_expired() is True

        result = broker.cancel_order("EXCH-00001")

        assert broker._login_call_count == 1, (
            f"Expected 1 login call before cancel_order, got {broker._login_call_count}"
        )
        assert result is True
        assert "EXCH-00001" in broker._cancelled

    def test_cancel_order_does_not_relogin_on_fresh_session(self) -> None:
        broker = _fresh_session_broker()
        broker.cancel_order("EXCH-00002")
        assert broker._login_call_count == 0


# ---------------------------------------------------------------------------
# Test: relogin failure raises SessionError
# ---------------------------------------------------------------------------

class TestReloginFailure:
    """P2-FIX-98: If relogin fails, place_order must raise SessionError."""

    def test_relogin_failure_raises_session_error(self) -> None:
        broker = _expired_session_broker()
        broker._login_should_fail = True

        order = _make_order()
        with pytest.raises(SessionError, match="Simulated relogin failure"):
            broker.place_order(order)

    def test_cancel_order_raises_session_error_on_relogin_failure(self) -> None:
        broker = _expired_session_broker()
        broker._login_should_fail = True

        with pytest.raises(SessionError, match="Simulated relogin failure"):
            broker.cancel_order("EXCH-12345")

    def test_relogin_failure_leaves_no_orders_placed(self) -> None:
        broker = _expired_session_broker()
        broker._login_should_fail = True

        try:
            broker.place_order(_make_order())
        except SessionError:
            pass

        # No orders should have been placed
        assert len(broker._placed_orders) == 0


# ---------------------------------------------------------------------------
# Test: session valid → relogin skipped
# ---------------------------------------------------------------------------

class TestSessionValidSkipsRelogin:
    """P2-FIX-98: Fresh session must not trigger relogin."""

    def test_session_valid_skips_relogin(self) -> None:
        broker = _fresh_session_broker()
        initial_token = broker._token

        broker.ensure_session()

        assert broker._login_call_count == 0, (
            "ensure_session() should not call login() when session is valid"
        )
        # Token unchanged
        assert broker._token == initial_token

    def test_is_session_expired_false_for_new_session(self) -> None:
        broker = _fresh_session_broker()
        assert broker._is_session_expired() is False

    def test_is_session_expired_true_when_token_missing(self) -> None:
        broker = _TestableBroker(token=None)
        assert broker._is_session_expired() is True

    def test_is_session_expired_true_when_login_time_missing(self) -> None:
        broker = _TestableBroker(login_time=None)
        # Manually clear login_time after init to bypass constructor default
        broker._login_time = None
        assert broker._is_session_expired() is True


# ---------------------------------------------------------------------------
# Test: heartbeat interval triggers relogin
# ---------------------------------------------------------------------------

class TestHeartbeatSessionRefresh:
    """P2-FIX-98: Heartbeat loop calls login() when session age exceeds threshold."""

    def test_heartbeat_interval_triggers_relogin(self) -> None:
        """
        Advance mock time past SESSION_MAX_AGE_HOURS.
        The heartbeat check (_is_session_expired) must return True,
        and a subsequent ensure_session() call must trigger login().
        """
        # Set up broker with a session that was established 23 hours ago
        old_time = datetime.now(tz=IST) - timedelta(hours=23)
        broker = _TestableBroker(token="old-token", login_time=old_time)

        assert broker._is_session_expired() is True, (
            "Session should be expired after 23 hours (limit is 22 hours)"
        )

        # Simulate what the heartbeat loop does
        if broker._is_session_expired():
            broker.login()

        assert broker._login_call_count == 1, (
            "Heartbeat should have triggered exactly one login()"
        )
        assert not broker._is_session_expired(), (
            "After relogin, session should no longer be expired"
        )

    def test_heartbeat_does_not_relogin_fresh_session(self) -> None:
        """Heartbeat loop on fresh session → no relogin."""
        broker = _fresh_session_broker()

        # Simulate heartbeat check
        if broker._is_session_expired():
            broker.login()

        assert broker._login_call_count == 0

    @pytest.mark.parametrize("age_hours,expected_expired", [
        (0,   False),
        (10,  False),
        (21,  False),   # just under the 22-hour limit
        (22,  True),    # exactly at limit → expired
        (23,  True),
        (36,  True),
    ])
    def test_session_expiry_threshold(
        self, age_hours: int, expected_expired: bool
    ) -> None:
        """Parametrized session age checks around the 22-hour boundary."""
        login_time = datetime.now(tz=IST) - timedelta(hours=age_hours)
        broker = _TestableBroker(token="tok", login_time=login_time)
        assert broker._is_session_expired() == expected_expired, (
            f"Age={age_hours}h: expected expired={expected_expired}, "
            f"got {broker._is_session_expired()}"
        )
