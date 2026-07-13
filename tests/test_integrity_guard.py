"""
tests/test_integrity_guard.py — Tick validator acceptance tests.

All acceptance criteria from ACCEPTANCE_CRITERIA.md:
  [x] Rejects tick with price move > 3% from previous tick
  [x] Rejects tick with duplicate timestamp
  [x] Rejects tick with zero volume on an active bar (when require_nonzero_volume=True)
  [x] Triggers halt when bad-tick rate exceeds threshold
  [x] Accepts valid tick without error
  [x] Test with synthetic corrupted feed sequences
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from algo_trading.data.broker_base import Tick
from algo_trading.data.integrity_guard import DataQualityMonitor, TickValidator

IST = ZoneInfo("Asia/Kolkata")
BASE_TS = datetime(2026, 6, 6, 10, 0, 0, tzinfo=IST)


def make_tick(
    symbol: str = "RELIANCE-EQ",
    ltp: float = 2850.0,
    volume: int = 100000,
    timestamp: Optional[datetime] = None,
    bid: float = 2849.95,
    ask: float = 2850.05,
) -> Tick:
    return Tick(
        timestamp=timestamp or BASE_TS,
        symbol=symbol,
        ltp=Decimal(str(ltp)),
        volume=volume,
        best_bid=Decimal(str(bid)),
        best_ask=Decimal(str(ask)),
        oi=0,
    )


@pytest.fixture
def validator() -> TickValidator:
    return TickValidator()


@pytest.fixture
def validator_volume() -> TickValidator:
    """Validator with volume check enabled."""
    return TickValidator(require_nonzero_volume=True)


class TestValidTick:
    def test_accepts_first_tick(self, validator: TickValidator) -> None:
        """First tick for a symbol must always be accepted."""
        tick = make_tick()
        result = validator.validate(tick)
        assert result.is_valid, f"First tick should be valid, got: {result.reason}"

    def test_accepts_normal_price_move(self, validator: TickValidator) -> None:
        """Normal 1% move must be accepted."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        t2 = make_tick(ltp=2878.5, timestamp=BASE_TS + timedelta(seconds=15))  # ~1% move

        validator.validate(t1)
        result = validator.validate(t2)
        assert result.is_valid, f"1% move should be valid, got: {result.reason}"

    def test_accepts_exact_boundary(self, validator: TickValidator) -> None:
        """Exactly 3.0% move must be accepted (boundary is EXCLUSIVE at >3%)."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        validator.validate(t1)

        # Exactly 3.0%
        t2 = make_tick(ltp=2850.0 * 1.03, timestamp=BASE_TS + timedelta(seconds=15))
        result = validator.validate(t2)
        assert result.is_valid, f"Exactly 3% move should be accepted, got: {result.reason}"

    def test_tick_count_increments(self, validator: TickValidator) -> None:
        for i in range(5):
            validator.validate(make_tick(timestamp=BASE_TS + timedelta(seconds=i)))
        assert validator.get_tick_count("RELIANCE-EQ") == 5


class TestPriceJump:
    def test_rejects_price_jump_above_3pct(self, validator: TickValidator) -> None:
        """Price jump > 3% from previous must be rejected."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        validator.validate(t1)

        # 3.5% jump
        t2 = make_tick(ltp=2850.0 * 1.035, timestamp=BASE_TS + timedelta(seconds=5))
        result = validator.validate(t2)
        assert not result.is_valid
        assert "jump" in result.reason.lower() or "3%" in result.reason or "%" in result.reason

    def test_rejects_large_drop(self, validator: TickValidator) -> None:
        """Large drop (> 3%) must also be rejected."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        validator.validate(t1)

        t2 = make_tick(ltp=2850.0 * 0.96, timestamp=BASE_TS + timedelta(seconds=5))
        result = validator.validate(t2)
        assert not result.is_valid

    def test_state_not_updated_on_rejection(self, validator: TickValidator) -> None:
        """After rejection, next valid tick uses the ORIGINAL price (not the bad tick)."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        validator.validate(t1)

        # Bad tick — rejected
        bad = make_tick(ltp=5000.0, timestamp=BASE_TS + timedelta(seconds=5))
        validator.validate(bad)

        # Next tick at 2855 (~0.18% from 2850) — should pass
        t3 = make_tick(ltp=2855.0, timestamp=BASE_TS + timedelta(seconds=10))
        result = validator.validate(t3)
        assert result.is_valid, f"Valid tick after rejection should pass: {result.reason}"

    def test_custom_threshold(self) -> None:
        """Custom threshold (e.g. 5%) accepts moves that 3% would reject."""
        v5 = TickValidator(max_price_jump_pct=Decimal("5.0"))
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        v5.validate(t1)

        # 4% move — rejected at 3% threshold, accepted at 5%
        t2 = make_tick(ltp=2850.0 * 1.04, timestamp=BASE_TS + timedelta(seconds=5))
        result = v5.validate(t2)
        assert result.is_valid, "4% move should pass 5% threshold"


class TestDuplicateTimestamp:
    def test_rejects_duplicate_timestamp(self, validator: TickValidator) -> None:
        """Two ticks with the same symbol + timestamp must reject the second."""
        t = make_tick(timestamp=BASE_TS)
        validator.validate(t)

        duplicate = make_tick(ltp=2851.0, timestamp=BASE_TS)  # Same timestamp!
        result = validator.validate(duplicate)
        assert not result.is_valid
        assert "duplicate" in result.reason.lower() or "timestamp" in result.reason.lower()

    def test_accepts_different_timestamps(self, validator: TickValidator) -> None:
        """Different timestamps for same symbol must pass."""
        t1 = make_tick(timestamp=BASE_TS)
        t2 = make_tick(timestamp=BASE_TS + timedelta(seconds=1))
        validator.validate(t1)
        result = validator.validate(t2)
        assert result.is_valid


class TestZeroLTP:
    def test_rejects_zero_ltp(self, validator: TickValidator) -> None:
        tick = make_tick(ltp=0.0)
        result = validator.validate(tick)
        assert not result.is_valid
        assert "ltp" in result.reason.lower() or "zero" in result.reason.lower()

    def test_rejects_negative_ltp(self, validator: TickValidator) -> None:
        tick = make_tick(ltp=-100.0)
        result = validator.validate(tick)
        assert not result.is_valid


class TestVolumeCheck:
    def test_rejects_zero_volume_when_enabled(self, validator_volume: TickValidator) -> None:
        tick = make_tick(volume=0)
        result = validator_volume.validate(tick)
        assert not result.is_valid
        assert "volume" in result.reason.lower()

    def test_accepts_zero_volume_when_disabled(self, validator: TickValidator) -> None:
        tick = make_tick(volume=0)
        result = validator.validate(tick)
        assert result.is_valid  # Volume check disabled by default


class TestOutOfOrder:
    def test_rejects_out_of_order_tick(self, validator: TickValidator) -> None:
        """Tick with timestamp earlier than previous must be rejected."""
        t1 = make_tick(timestamp=BASE_TS + timedelta(seconds=10))
        validator.validate(t1)

        t2 = make_tick(ltp=2851.0, timestamp=BASE_TS)  # Earlier!
        result = validator.validate(t2)
        assert not result.is_valid


class TestMultiSymbol:
    def test_each_symbol_has_independent_state(self, validator: TickValidator) -> None:
        """Price jump threshold applies per symbol."""
        r1 = make_tick(symbol="RELIANCE-EQ", ltp=2850.0, timestamp=BASE_TS)
        t1 = make_tick(symbol="TCS-EQ", ltp=3500.0, timestamp=BASE_TS)

        validator.validate(r1)
        validator.validate(t1)

        # RELIANCE at 2942.5 (3.2% jump) — should fail
        r2 = make_tick(symbol="RELIANCE-EQ", ltp=2942.5, timestamp=BASE_TS + timedelta(seconds=5))
        result_r = validator.validate(r2)

        # TCS at 3535 (~1% move) — should pass
        t2 = make_tick(symbol="TCS-EQ", ltp=3535.0, timestamp=BASE_TS + timedelta(seconds=5))
        result_t = validator.validate(t2)

        assert not result_r.is_valid, "RELIANCE 3.2% jump should fail"
        assert result_t.is_valid, "TCS 1% move should pass"

    def test_reset_symbol(self, validator: TickValidator) -> None:
        """After reset, the next tick is treated as first tick (no price history)."""
        t1 = make_tick(ltp=2850.0, timestamp=BASE_TS)
        validator.validate(t1)

        validator.reset_symbol("RELIANCE-EQ")
        assert validator.get_tick_count("RELIANCE-EQ") == 0

        # Large jump should now pass (no previous price to compare against)
        t2 = make_tick(ltp=5000.0, timestamp=BASE_TS + timedelta(seconds=5))
        result = validator.validate(t2)
        assert result.is_valid, "Post-reset tick should pass (no price history)"


class TestDataQualityMonitor:
    def test_halt_not_triggered_below_threshold(self) -> None:
        halt_cb = MagicMock()
        monitor = DataQualityMonitor(window_size=10, max_bad_rate=0.10, halt_callback=halt_cb)

        # 9 good, 1 bad = 10% — exactly at threshold (not above)
        for _ in range(9):
            monitor.record(True)
        monitor.record(False)

        halt_cb.assert_not_called()
        assert not monitor.is_halted

    def test_halt_triggered_above_threshold(self) -> None:
        halt_cb = MagicMock()
        monitor = DataQualityMonitor(window_size=10, max_bad_rate=0.10, halt_callback=halt_cb)

        # 8 good, 2 bad = 20% > 10% threshold
        for _ in range(8):
            monitor.record(True)
        monitor.record(False)
        monitor.record(False)

        halt_cb.assert_called_once()
        assert monitor.is_halted

    def test_halt_blocks_further_callbacks(self) -> None:
        """After halt, subsequent bad ticks don't re-trigger the callback."""
        halt_cb = MagicMock()
        monitor = DataQualityMonitor(window_size=5, max_bad_rate=0.10, halt_callback=halt_cb)

        for _ in range(5):
            monitor.record(False)  # 100% bad rate — should halt

        assert monitor.is_halted
        initial_count = halt_cb.call_count

        # More bad ticks — callback should NOT fire again
        for _ in range(5):
            monitor.record(False)

        assert halt_cb.call_count == initial_count

    def test_reset_clears_halt(self) -> None:
        halt_cb = MagicMock()
        monitor = DataQualityMonitor(window_size=5, max_bad_rate=0.10, halt_callback=halt_cb)

        for _ in range(5):
            monitor.record(False)
        assert monitor.is_halted

        monitor.reset()
        assert not monitor.is_halted
        assert monitor.current_bad_rate == 0.0

    def test_bad_rate_calculation(self) -> None:
        monitor = DataQualityMonitor(window_size=100, max_bad_rate=0.50)
        for _ in range(25):
            monitor.record(True)
        for _ in range(25):
            monitor.record(False)

        assert abs(monitor.current_bad_rate - 0.50) < 0.01


class TestCorruptedFeedSequence:
    def test_synthetic_corrupted_sequence(self, validator: TickValidator) -> None:
        """
        Simulate a corrupted feed: good ticks, then a burst of bad ticks,
        then good ticks resume. Validate that the system correctly identifies
        the bad ticks and resumes on good ticks.
        """
        results: list[bool] = []
        ts = BASE_TS
        price = Decimal("2850.00")

        # 5 good ticks
        for i in range(5):
            t = make_tick(ltp=float(price), timestamp=ts)
            results.append(validator.validate(t).is_valid)
            ts += timedelta(seconds=15)
            price += Decimal("0.50")

        # 3 bad ticks (price spike)
        for i in range(3):
            bad = make_tick(ltp=5000.0, timestamp=ts)
            results.append(validator.validate(bad).is_valid)
            ts += timedelta(seconds=15)

        # 5 more good ticks (resuming from last valid price ~₹2852.50)
        for i in range(5):
            t = make_tick(ltp=float(price), timestamp=ts)
            results.append(validator.validate(t).is_valid)
            ts += timedelta(seconds=15)
            price += Decimal("0.50")

        assert all(results[:5]), "First 5 ticks should be valid"
        assert not any(results[5:8]), "3 spike ticks should be invalid"
        assert all(results[8:]), "Final 5 ticks should be valid"
