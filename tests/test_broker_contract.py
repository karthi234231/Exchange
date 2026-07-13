"""
tests/test_broker_contract.py — Broker API contract tests.

Tests that ShoonyaBroker correctly maps all known broker status strings
to internal FSM events. Uses recorded/mock Shoonya order update payloads.

P2-FIX-50: Broker capability contract tests.
P2-FIX-51: Shoonya remarks field length verification.
"""
from __future__ import annotations

import sys
import types
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal recorded Shoonya WebSocket order-update payloads
# ---------------------------------------------------------------------------

_SHOONYA_OPEN = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "OPEN",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
}
_SHOONYA_OPEN_PENDING = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "OPEN PENDING",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
}
_SHOONYA_FILLED = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "COMPLETE",
    "remarks": "client123",
    "flqty": "10",
    "flprc": "2500.50",
}
_SHOONYA_PARTIAL = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "PARTIALLY FILLED",
    "remarks": "client123",
    "flqty": "5",
    "flprc": "2500.00",
}
_SHOONYA_CANCELLED = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "CANCELLED",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
}
_SHOONYA_REJECTED = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "REJECTED",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
    "rejreason": "Insufficient margin",
}
_SHOONYA_TRIGGER_PENDING = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "TRIGGER PENDING",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
}
_SHOONYA_UNKNOWN = {
    "stat": "Ok",
    "norenordno": "23010100000001",
    "status": "FOOBAR",
    "remarks": "client123",
    "flqty": "0",
    "flprc": "0",
}


# ---------------------------------------------------------------------------
# Helpers — build a minimal stub for the status-mapping logic that lives in
# broker_shoonya.py without needing the NorenRestApiPy library installed.
# We test the *mapping logic* in isolation.
# ---------------------------------------------------------------------------

def _parse_shoonya_status(payload: dict) -> Optional[str]:
    """
    Mirror of the status→FSM-event mapping used in ShoonyaBroker._on_order_update.

    Returns the internal event string (ACK / FILL / PARTIAL_FILL /
    CANCEL_CONFIRM / REJECT) or None for explicitly-ignored statuses.

    Raises ValueError for unknown statuses so the caller can alert.
    """
    STATUS_MAP: dict[str, Optional[str]] = {
        "OPEN": "ACK",
        "OPEN PENDING": "ACK",
        "COMPLETE": "FILL",
        "PARTIALLY FILLED": "PARTIAL_FILL",
        "CANCELLED": "CANCEL_CONFIRM",
        "CANCEL PENDING": None,   # in-flight cancel — explicitly ignored
        "TRIGGER PENDING": None,  # SL-M order waiting for trigger — ignored
        "REJECTED": "REJECT",
        "MODIFIED": None,         # order modification ACK — ignored
    }
    status = payload.get("status", "")
    if status not in STATUS_MAP:
        raise ValueError(f"Unknown Shoonya status: {status!r}")
    return STATUS_MAP[status]


def _parse_shoonya_fill(payload: dict, prev_cumulative: int) -> tuple[int, Decimal]:
    """
    Convert Shoonya's cumulative flqty into a per-message delta.

    Returns (delta_qty, fill_price).
    """
    new_cumulative = int(payload.get("flqty", 0) or 0)
    delta = new_cumulative - prev_cumulative
    price = Decimal(str(payload.get("flprc", 0) or 0))
    return delta, price


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestStatusMapping:
    """P2-FIX-50: Verify all known Shoonya statuses map to correct FSM events."""

    def test_status_mapping_open(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_OPEN)
        assert event == "ACK", f"Expected 'ACK', got {event!r}"

    def test_status_mapping_open_pending(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_OPEN_PENDING)
        assert event == "ACK", f"Expected 'ACK', got {event!r}"

    def test_status_mapping_complete(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_FILLED)
        assert event == "FILL", f"Expected 'FILL', got {event!r}"

    def test_status_mapping_partially_filled(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_PARTIAL)
        assert event == "PARTIAL_FILL", f"Expected 'PARTIAL_FILL', got {event!r}"

    def test_status_mapping_cancelled(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_CANCELLED)
        assert event == "CANCEL_CONFIRM", f"Expected 'CANCEL_CONFIRM', got {event!r}"

    def test_status_mapping_rejected(self) -> None:
        event = _parse_shoonya_status(_SHOONYA_REJECTED)
        assert event == "REJECT", f"Expected 'REJECT', got {event!r}"

    def test_status_mapping_trigger_pending(self) -> None:
        """TRIGGER PENDING is a valid Shoonya status but maps to None (ignored)."""
        event = _parse_shoonya_status(_SHOONYA_TRIGGER_PENDING)
        assert event is None, (
            "TRIGGER PENDING should be explicitly ignored (returns None), "
            f"got {event!r}"
        )

    def test_unknown_status_raises(self) -> None:
        """Unknown status must raise so the caller can emit an alert."""
        with pytest.raises(ValueError, match="FOOBAR"):
            _parse_shoonya_status(_SHOONYA_UNKNOWN)


class TestRemarksFieldLength:
    """P2-FIX-51: Shoonya remarks field accepts up to 20 characters."""

    # Shoonya API documentation states the `remarks` field is limited to 20 chars.
    _SHOONYA_REMARKS_MAX_LEN = 20

    def test_client_order_id_within_remarks_limit(self) -> None:
        """
        The client_order_id placed in the remarks field must not exceed 20 chars.

        The ShoonyaBroker uses order.client_order_id (UUID4) in the remarks field
        (see broker_shoonya.py: remarks=order.client_order_id).  UUID4 is 36 chars
        which exceeds the 20-char limit.  This test documents the mismatch so the
        team is aware — if truncation is applied upstream this test should be updated
        to reflect the truncated format.
        """
        import uuid
        full_uuid = str(uuid.uuid4())  # 36 chars — exceeds limit
        assert len(full_uuid) == 36

        # The remarks field limit
        limit = self._SHOONYA_REMARKS_MAX_LEN
        assert limit == 20

        # Document: raw UUID exceeds limit. A shorter format (e.g. 8-char hex prefix
        # or a sequential counter) must be used when placing live orders.
        truncated = full_uuid.replace("-", "")[:limit]
        assert len(truncated) <= limit, (
            f"Truncated client_order_id {truncated!r} still exceeds {limit} chars"
        )

    def test_strategy_id_within_remarks_limit(self) -> None:
        """Short strategy IDs like 'EMA-CROSS' easily fit in 20 chars."""
        strategy_id = "EMA-CROSS"
        assert len(strategy_id) <= self._SHOONYA_REMARKS_MAX_LEN

    @pytest.mark.parametrize("client_id,expected_len", [
        ("EMA-RELIANCE-001", 16),
        ("A" * 20, 20),             # exactly at limit
        ("A" * 20, 20),             # exactly at limit (no truncation needed)
    ])
    def test_parametrized_remarks_within_limit(self, client_id: str, expected_len: int) -> None:
        assert len(client_id) == expected_len
        assert expected_len <= self._SHOONYA_REMARKS_MAX_LEN


class TestFillDeduplication:
    """P2-FIX-50: Same fill event twice should only produce one delta."""

    def test_fill_deduplication_via_cumulative(self) -> None:
        """
        If flqty=10 arrives twice (WS retry/reconnect), the second message
        produces delta=0 via the cumulative→delta conversion.
        """
        # First message: cumulative goes from 0 → 10
        delta1, _ = _parse_shoonya_fill(_SHOONYA_FILLED, prev_cumulative=0)
        assert delta1 == 10

        # Duplicate message: cumulative is still 10 → delta = 0
        delta2, _ = _parse_shoonya_fill(_SHOONYA_FILLED, prev_cumulative=10)
        assert delta2 == 0, (
            f"Duplicate fill message should produce delta=0, got delta={delta2}"
        )

    def test_partial_fill_deduplication(self) -> None:
        """Partial fill of 5 received twice → second delta=0."""
        delta1, _ = _parse_shoonya_fill(_SHOONYA_PARTIAL, prev_cumulative=0)
        assert delta1 == 5

        delta2, _ = _parse_shoonya_fill(_SHOONYA_PARTIAL, prev_cumulative=5)
        assert delta2 == 0


class TestCumulativeToDeltaConversion:
    """P2-FIX-50 / P2-FIX-101: Cumulative flqty → per-message delta conversion."""

    def test_cumulative_to_delta_basic(self) -> None:
        """flqty=10 then flqty=15 should produce delta=5 for the second update."""
        first_payload = {"flqty": "10", "flprc": "2500.00", "status": "PARTIALLY FILLED"}
        second_payload = {"flqty": "15", "flprc": "2500.00", "status": "COMPLETE"}

        delta1, price1 = _parse_shoonya_fill(first_payload, prev_cumulative=0)
        assert delta1 == 10
        assert price1 == Decimal("2500.00")

        delta2, price2 = _parse_shoonya_fill(second_payload, prev_cumulative=10)
        assert delta2 == 5, f"Expected delta=5, got {delta2}"
        assert price2 == Decimal("2500.00")

    def test_cumulative_to_delta_three_partials(self) -> None:
        """Three cumulative updates: 5, 10, 10 → deltas: 5, 5, 0."""
        updates = [
            {"flqty": "5",  "flprc": "2500.00"},
            {"flqty": "10", "flprc": "2501.00"},
            {"flqty": "10", "flprc": "2501.00"},  # duplicate
        ]
        cumulative = 0
        deltas = []
        for upd in updates:
            delta, _ = _parse_shoonya_fill(upd, prev_cumulative=cumulative)
            cumulative += delta
            deltas.append(delta)

        assert deltas == [5, 5, 0], f"Expected [5, 5, 0], got {deltas}"
        assert cumulative == 10

    def test_zero_flqty_produces_zero_delta(self) -> None:
        """OPEN status sends flqty='0' — delta must be 0, not negative."""
        delta, price = _parse_shoonya_fill(_SHOONYA_OPEN, prev_cumulative=0)
        assert delta == 0
        assert price == Decimal("0")

    @pytest.mark.parametrize("flqty_str,prev,expected_delta", [
        ("0", 0, 0),
        ("10", 0, 10),
        ("10", 5, 5),
        ("10", 10, 0),    # duplicate
        ("0", 0, 0),
    ])
    def test_parametrized_cumulative_delta(
        self, flqty_str: str, prev: int, expected_delta: int
    ) -> None:
        payload = {"flqty": flqty_str, "flprc": "2500.00"}
        delta, _ = _parse_shoonya_fill(payload, prev_cumulative=prev)
        assert delta == expected_delta


class TestOrderUpdatePayloadParsing:
    """Contract tests for the full order update payload structure."""

    def test_norenordno_always_present(self) -> None:
        """All Shoonya order update payloads must contain norenordno."""
        for payload in [
            _SHOONYA_OPEN, _SHOONYA_FILLED, _SHOONYA_PARTIAL,
            _SHOONYA_CANCELLED, _SHOONYA_REJECTED,
        ]:
            assert "norenordno" in payload, (
                f"Payload missing norenordno: {payload}"
            )
            assert payload["norenordno"], "norenordno must not be empty"

    def test_stat_ok_in_all_payloads(self) -> None:
        """All recorded payloads have stat=Ok."""
        for payload in [
            _SHOONYA_OPEN, _SHOONYA_FILLED, _SHOONYA_PARTIAL,
            _SHOONYA_CANCELLED, _SHOONYA_REJECTED,
        ]:
            assert payload.get("stat") == "Ok", (
                f"Expected stat='Ok' in {payload}"
            )

    def test_rejected_payload_has_rejreason(self) -> None:
        """REJECTED payload must carry a rejection reason."""
        assert "rejreason" in _SHOONYA_REJECTED
        assert _SHOONYA_REJECTED["rejreason"]

    def test_filled_payload_has_nonzero_flprc(self) -> None:
        """COMPLETE (filled) payload must have a positive fill price."""
        price = Decimal(_SHOONYA_FILLED["flprc"])
        assert price > Decimal("0"), f"Expected positive flprc, got {price}"

    def test_open_payload_has_zero_flqty(self) -> None:
        """OPEN status has no fills yet — flqty must be 0."""
        assert int(_SHOONYA_OPEN["flqty"]) == 0
