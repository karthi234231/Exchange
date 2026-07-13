"""
tests/test_fat_finger.py — Comprehensive test suite for FatFingerGuard.

Covers all rejection paths, boundary conditions, and edge cases.
Uses realistic NSE price ranges (Rs 100–5000 per share).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from algo_trading.data.broker_base import Order
from algo_trading.risk.fat_finger import (
    ADVCache,
    FatFingerConfig,
    FatFingerGuard,
    FatFingerResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_order(
    symbol: str = "RELIANCE",
    side: str = "BUY",
    quantity: int = 10,
    order_type: str = "LIMIT",
    price: float = 2850.0,
    trigger_price: float = 0.0,
    strategy_id: str = "TEST-STRAT-01",
) -> Order:
    """Build a test Order. Trigger price > 0 marks it as stop-loss MARKET."""
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=Decimal(str(price)) if order_type == "LIMIT" else Decimal("0"),
        strategy_id=strategy_id,
        trigger_price=Decimal(str(trigger_price)),
    )


def _guard(
    max_order_value_inr: float = 200_000,
    max_adv_pct: float = 0.02,
    max_price_deviation_bps: float = 50.0,
    max_quantity: int = 5000,
    adv_fallback: float | None = None,
) -> FatFingerGuard:
    cfg = FatFingerConfig(
        max_order_value_inr=Decimal(str(max_order_value_inr)),
        max_adv_pct=max_adv_pct,
        max_price_deviation_bps=max_price_deviation_bps,
        max_quantity=max_quantity,
        adv_cache_ttl_sec=3600.0,
        adv_fallback=adv_fallback,
    )
    return FatFingerGuard(config=cfg)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNormalOrderOK:
    """Test 1: A normal, well-within-limits order should return OK."""

    def test_normal_buy_limit_order_passes(self):
        guard = _guard()
        # RELIANCE ~2850, 10 shares = Rs 28,500 — well within Rs 2L
        order = _make_order(symbol="RELIANCE", quantity=10, price=2850.0)
        ltp = Decimal("2848.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK
        assert reason == ""

    def test_normal_small_cap_buy_passes(self):
        guard = _guard()
        # Small-cap stock at Rs 120, 100 shares = Rs 12,000
        order = _make_order(symbol="SMALLCAP", quantity=100, price=120.0)
        ltp = Decimal("119.50")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK

    def test_normal_sell_order_passes(self):
        guard = _guard()
        order = _make_order(symbol="TCS", side="SELL", quantity=5, price=3500.0)
        ltp = Decimal("3498.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK


class TestRejectOverValue:
    """Test 2: Order value > Rs 2L should return REJECT_OVER_VALUE."""

    def test_order_over_2_lakh_rejected(self):
        guard = _guard(max_order_value_inr=200_000)
        # 80 shares × Rs 2600 = Rs 208,000 > Rs 2L
        order = _make_order(symbol="HDFCBANK", quantity=80, price=2600.0)
        ltp = Decimal("2598.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_VALUE
        assert "HDFCBANK" in reason
        assert "200000" in reason or "2,00,000" in reason or "200,000" in reason

    def test_order_over_limit_market_order_uses_ltp(self):
        guard = _guard(max_order_value_inr=200_000)
        # MARKET order — no limit price, uses LTP for valuation
        # 80 × 2600 = 208,000 > 2L
        order = _make_order(
            symbol="HDFCBANK", quantity=80, order_type="MARKET", price=0.0
        )
        ltp = Decimal("2600.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_VALUE

    def test_high_priced_stock_rejected(self):
        guard = _guard(max_order_value_inr=200_000)
        # MRF ~Rs 120,000 — even 2 shares = Rs 240,000
        order = _make_order(symbol="MRF", quantity=2, price=120_000.0)
        ltp = Decimal("119_900.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_VALUE


class TestBoundaryAtExactLimit:
    """Test 3: Order value exactly at Rs 2L should be OK (boundary inclusive)."""

    def test_exact_value_at_limit_is_ok(self):
        guard = _guard(max_order_value_inr=200_000)
        # 100 shares × Rs 2000 = exactly Rs 2,00,000
        order = _make_order(symbol="INFY", quantity=100, price=2000.0)
        ltp = Decimal("1999.50")  # LTP within 50bps
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, (
            f"Expected OK at exactly Rs 2L boundary, got {result}: {reason}"
        )

    def test_one_rupee_over_limit_rejected(self):
        guard = _guard(max_order_value_inr=200_000)
        # 100 × 2000.01 = 200,001 > 200,000
        order = _make_order(symbol="INFY", quantity=100, price=2000.01)
        ltp = Decimal("1999.50")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_VALUE


class TestRejectOverADV:
    """Test 4: Quantity > 2% ADV should return REJECT_OVER_ADV."""

    def test_order_over_adv_pct_rejected(self):
        guard = _guard(max_adv_pct=0.02)
        # ADV = 1,000,000 shares → 2% = 20,000 shares
        # Order for 25,000 shares at Rs 100 = Rs 25L (but value cap is 2L,
        # so use a cheaper stock to isolate the ADV check)
        # Use a custom high-value cap to isolate ADV test
        guard2 = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),  # Rs 1 Cr — not the binding constraint
                max_adv_pct=0.02,
                max_price_deviation_bps=500.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard2.update_adv("RELIANCE", 1_000_000)  # 1M shares/day ADV
        # 2% of 1M = 20,000 — order 25,000
        order = _make_order(symbol="RELIANCE", quantity=25_000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard2.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_ADV
        assert "RELIANCE" in reason
        assert "ADV" in reason or "adv" in reason.lower() or "2%" in reason or "2.0%" in reason

    def test_order_just_under_adv_pct_ok(self):
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.02,
                max_price_deviation_bps=500.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard.update_adv("RELIANCE", 1_000_000)
        # 2% of 1M = 20,000 — order exactly 20,000 (boundary OK)
        order = _make_order(symbol="RELIANCE", quantity=20_000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, f"Expected OK at ADV boundary, got {result}: {reason}"


class TestRejectPriceDeviation:
    """Test 5: LIMIT price > 50bps from LTP should return REJECT_PRICE_DEVIATION."""

    def test_price_too_far_above_ltp_rejected(self):
        guard = _guard(max_price_deviation_bps=50.0)
        # LTP = 2848, limit = 2865
        # deviation = (2865 - 2848) / 2848 * 10000 = 59.7 bps > 50 bps
        order = _make_order(symbol="RELIANCE", quantity=5, price=2865.0)
        ltp = Decimal("2848.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_PRICE_DEVIATION
        assert "RELIANCE" in reason

    def test_price_too_far_below_ltp_rejected(self):
        guard = _guard(max_price_deviation_bps=50.0)
        # LTP = 2848, limit = 2830
        # deviation = (2848 - 2830) / 2848 * 10000 = 63.2 bps > 50 bps
        order = _make_order(
            symbol="RELIANCE", side="SELL", quantity=5, price=2830.0
        )
        ltp = Decimal("2848.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_PRICE_DEVIATION


class TestPriceDeviationBoundaryOK:
    """Test 6: LIMIT price at exactly 50bps from LTP should be OK (inclusive boundary)."""

    def test_price_at_exact_50bps_is_ok(self):
        guard = _guard(max_price_deviation_bps=50.0)
        # LTP = 2000.00
        # 50 bps = 2000 * 50 / 10000 = 10.00
        # Limit price = 2010.00 → deviation = 10/2000 * 10000 = 50.0 bps exactly
        order = _make_order(symbol="INFY", quantity=5, price=2010.0)
        ltp = Decimal("2000.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, (
            f"Expected OK at 50bps boundary, got {result}: {reason}"
        )

    def test_price_at_50bps_plus_one_tick_rejected(self):
        guard = _guard(max_price_deviation_bps=50.0)
        # Slightly more than 50bps: 2010.10
        order = _make_order(symbol="INFY", quantity=5, price=2010.10)
        ltp = Decimal("2000.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_PRICE_DEVIATION


class TestRejectQtyCap:
    """Test 7: Quantity > 5000 should return REJECT_QTY_CAP."""

    def test_qty_over_5000_rejected(self):
        guard = _guard(max_quantity=5000)
        order = _make_order(symbol="INFY", quantity=5001, price=100.0)
        ltp = Decimal("99.90")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_QTY_CAP
        assert "INFY" in reason
        assert "5000" in reason or "5,000" in reason

    def test_qty_at_5000_is_ok(self):
        guard = _guard(
            max_quantity=5000,
            max_order_value_inr=10_000_000,  # High cap to not interfere
            max_price_deviation_bps=500.0,
        )
        order = _make_order(symbol="INFY", quantity=5000, price=100.0)
        ltp = Decimal("99.90")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, (
            f"Expected OK at qty cap boundary 5000, got {result}: {reason}"
        )

    def test_qty_over_cap_checked_before_value(self):
        guard = _guard(max_quantity=5000)
        # This would also fail value check, but qty is checked first
        order = _make_order(symbol="TCS", quantity=6000, price=3500.0)
        ltp = Decimal("3498.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_QTY_CAP


class TestStopLossMarketBypassesADV:
    """
    Test 8: MARKET stop-loss order bypasses ADV check.
    Stop-loss orders have trigger_price > 0.
    They are safety-critical and must not be blocked by ADV limits.
    """

    def test_stop_loss_market_bypasses_adv_check(self):
        # Set up guard with very tight ADV — would normally block this qty
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),  # High enough to not block
                max_adv_pct=0.001,  # 0.1% — very tight
                max_price_deviation_bps=5000.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard.update_adv("RELIANCE", 1_000_000)
        # 0.1% of 1M = 1000 — order 5000 (would fail ADV check on regular order)
        # BUT trigger_price > 0 marks it as a stop-loss MARKET order → skip ADV
        order = Order(
            symbol="RELIANCE",
            side="SELL",
            quantity=5000,
            order_type="MARKET",
            price=Decimal("0"),
            strategy_id="TEST-STRAT-01",
            trigger_price=Decimal("2800.00"),  # Stop trigger set → stop-loss MARKET
        )
        ltp = Decimal("2810.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, (
            f"Stop-loss MARKET should bypass ADV check, got {result}: {reason}"
        )

    def test_regular_market_order_still_checked_for_adv(self):
        """A MARKET order WITHOUT trigger_price is NOT exempt from ADV check."""
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.001,
                max_price_deviation_bps=5000.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard.update_adv("RELIANCE", 1_000_000)
        # trigger_price = 0 → regular MARKET order → ADV check applies
        order = Order(
            symbol="RELIANCE",
            side="BUY",
            quantity=5000,
            order_type="MARKET",
            price=Decimal("0"),
            strategy_id="TEST-STRAT-01",
            trigger_price=Decimal("0"),  # No trigger → regular market order
        )
        ltp = Decimal("2810.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_ADV


class TestADVNotAvailable:
    """Test 9: When ADV is not in cache and no fallback, only value and qty checks apply."""

    def test_no_adv_skips_adv_check(self):
        guard = _guard(max_order_value_inr=10_000_000, max_adv_pct=0.001)
        # No ADV set → adv_fallback = None (default) → ADV check skipped
        # This large quantity would fail ADV check if ADV were set
        order = _make_order(symbol="NEWSTOCK", quantity=50_000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard.check(order, ltp)
        # Should NOT fail on ADV (no ADV known) — may fail on value/qty
        assert result != FatFingerResult.REJECT_OVER_ADV

    def test_adv_fallback_used_when_no_cache(self):
        """When adv_fallback is set, it is used as ADV when cache is empty."""
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.02,
                max_price_deviation_bps=5000.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
                adv_fallback=100_000.0,  # 100K shares fallback
            )
        )
        # 2% of 100K = 2000 shares max → order 3000 should fail
        order = _make_order(symbol="UNKNOWN", quantity=3000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_ADV


class TestZeroLTP:
    """Test 10: Zero LTP — price deviation check must be skipped gracefully."""

    def test_zero_ltp_no_price_deviation_check(self):
        guard = _guard(max_price_deviation_bps=50.0)
        order = _make_order(symbol="RELIANCE", quantity=5, price=2850.0)
        ltp = Decimal("0")
        # Should not crash or return REJECT_PRICE_DEVIATION
        result, reason = guard.check(order, ltp)
        assert result != FatFingerResult.REJECT_PRICE_DEVIATION

    def test_zero_ltp_with_large_order_value(self):
        guard = _guard(max_order_value_inr=200_000)
        # LTP=0 → effective price = limit price → value check still runs
        order = _make_order(symbol="RELIANCE", quantity=100, price=2850.0)
        ltp = Decimal("0")
        result, reason = guard.check(order, ltp)
        # 100 × 2850 = 285,000 > 200,000 → should still reject on value
        assert result == FatFingerResult.REJECT_OVER_VALUE


class TestUpdateADV:
    """Test 11: update_adv then check with correct ADV values."""

    def test_update_adv_and_check_pass(self):
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.02,
                max_price_deviation_bps=500.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        # Set ADV to 5 million shares/day (large liquid stock like RELIANCE)
        guard.update_adv("RELIANCE", 5_000_000)
        # 2% of 5M = 100K → order 50K should be fine
        order = _make_order(symbol="RELIANCE", quantity=50_000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK, (
            f"Expected OK after ADV update, got {result}: {reason}"
        )

    def test_update_adv_then_reject_over_adv(self):
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.02,
                max_price_deviation_bps=500.0,
                max_quantity=200_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard.update_adv("ZOMATO", 500_000)  # 500K shares/day
        # 2% of 500K = 10,000 → order 15,000 should fail
        order = _make_order(symbol="ZOMATO", quantity=15_000, price=10.0)
        ltp = Decimal("10.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_ADV

    def test_adv_cache_stores_and_retrieves(self):
        cache = ADVCache(ttl_sec=3600.0)
        cache.update("RELIANCE", 4_000_000.0)
        retrieved = cache.get("RELIANCE")
        assert retrieved == pytest.approx(4_000_000.0)

    def test_adv_cache_miss_returns_none(self):
        cache = ADVCache(ttl_sec=3600.0)
        assert cache.get("UNKNOWN") is None


class TestReasonString:
    """Test 12: Reason string is informative — contains symbol and threshold."""

    def test_value_reject_reason_has_symbol_and_limit(self):
        guard = _guard(max_order_value_inr=200_000)
        order = _make_order(symbol="WIPRO", quantity=100, price=3000.0)
        ltp = Decimal("2998.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_VALUE
        assert "WIPRO" in reason
        # Threshold amount must appear (200000 in some format)
        assert any(s in reason for s in ["200000", "200,000", "2,00,000"])

    def test_qty_cap_reason_has_symbol_and_cap(self):
        guard = _guard(max_quantity=5000)
        order = _make_order(symbol="TECHM", quantity=6000, price=100.0)
        ltp = Decimal("99.80")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_QTY_CAP
        assert "TECHM" in reason
        assert "5000" in reason or "5,000" in reason

    def test_adv_reason_has_symbol_and_adv_info(self):
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.02,
                max_price_deviation_bps=500.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        guard.update_adv("ONGC", 2_000_000)
        # 2% of 2M = 40K → order 50K
        order = _make_order(symbol="ONGC", quantity=50_000, price=5.0)
        ltp = Decimal("5.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_OVER_ADV
        assert "ONGC" in reason

    def test_price_deviation_reason_has_symbol_and_deviation(self):
        guard = _guard(max_price_deviation_bps=50.0)
        order = _make_order(symbol="BAJFINANCE", quantity=2, price=7500.0)
        ltp = Decimal("7460.00")  # Deviation = 40/7460*10000 ≈ 53.6 bps > 50
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_PRICE_DEVIATION
        assert "BAJFINANCE" in reason

    def test_ok_reason_is_empty_string(self):
        guard = _guard()
        order = _make_order(symbol="RELIANCE", quantity=5, price=2850.0)
        ltp = Decimal("2848.00")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.OK
        assert reason == ""


class TestADVCacheExpiry:
    """Test ADV cache TTL expiry."""

    def test_adv_cache_expires_after_ttl(self):
        import time as _t
        cache = ADVCache(ttl_sec=0.05)  # 50ms TTL for fast test
        cache.update("RELIANCE", 1_000_000.0)
        assert cache.get("RELIANCE") == pytest.approx(1_000_000.0)
        _t.sleep(0.1)  # Wait past TTL
        assert cache.get("RELIANCE") is None, "ADV should have expired"

    def test_adv_cache_clear_removes_all(self):
        cache = ADVCache(ttl_sec=3600.0)
        cache.update("RELIANCE", 1_000_000.0)
        cache.update("TCS", 500_000.0)
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0
        assert cache.get("RELIANCE") is None
        assert cache.get("TCS") is None


class TestFatFingerGuardIntegration:
    """Integration tests combining multiple checks."""

    def test_multiple_checks_qty_fails_first(self):
        """Qty cap is checked BEFORE value — verify check order."""
        guard = _guard(max_quantity=5000, max_order_value_inr=200_000)
        # qty=6000 fails qty cap; value = 6000 * 30 = 180K < 200K (would pass value)
        order = _make_order(symbol="NIFTYSTOCK", quantity=6000, price=30.0)
        ltp = Decimal("29.90")
        result, reason = guard.check(order, ltp)
        assert result == FatFingerResult.REJECT_QTY_CAP

    def test_guard_thread_safety_concurrent_updates(self):
        """update_adv and check() can be called from multiple threads."""
        import threading
        guard = FatFingerGuard(
            config=FatFingerConfig(
                max_order_value_inr=Decimal("10000000"),
                max_adv_pct=0.5,  # Very loose ADV to not block
                max_price_deviation_bps=5000.0,
                max_quantity=100_000,
                adv_cache_ttl_sec=3600.0,
            )
        )
        errors: list[Exception] = []

        def updater():
            try:
                for i in range(50):
                    guard.update_adv("RELIANCE", float(1_000_000 + i))
            except Exception as exc:
                errors.append(exc)

        def checker():
            try:
                order = _make_order(symbol="RELIANCE", quantity=10, price=100.0)
                ltp = Decimal("99.80")
                for _ in range(50):
                    guard.check(order, ltp)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=checker),
            threading.Thread(target=updater),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread safety errors: {errors}"
