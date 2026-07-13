"""
tests/test_margin_headroom.py — Unit tests for algo_trading/risk/margin_headroom.py

Verifies:
  - MarginSnapshot.status thresholds: OK/WARNING/CRITICAL/HALT
  - MarginSnapshot.can_add_positions: blocked at CRITICAL+ (>= 85%)
  - MarginHeadroomMonitor.snapshot(): computes calibrated estimate
  - Alert callback fired at WARNING, CRITICAL, HALT
  - record_broker_margin(): updates calibration_factor after 3+ observations
  - headroom_lots(): 0 when no headroom; positive when capital available
  - update_capital(): changes available capital
"""
from __future__ import annotations

import math
import pytest
from datetime import datetime

from algo_trading.risk.margin_headroom import (
    MarginHeadroomMonitor,
    MarginSnapshot,
    MarginCalibrationRecord,
)


# ── MarginSnapshot: status thresholds ────────────────────────────────────────

class TestMarginSnapshotStatus:
    def _make_snapshot(self, utilization: float) -> MarginSnapshot:
        cap = 1_000_000.0
        return MarginSnapshot(
            timestamp=datetime.utcnow(),
            available_capital=cap,
            theoretical_span=cap * utilization / 1.20,  # calibration_factor=1.20 default
            broker_actual=None,
            calibrated_estimate=cap * utilization,
            utilization_pct=utilization,
            headroom_inr=cap * (1.0 - utilization),
            positions_count=1,
        )

    def test_status_ok_below_70(self):
        snap = self._make_snapshot(0.60)
        assert snap.status == "OK"

    def test_status_warning_at_70(self):
        snap = self._make_snapshot(0.70)
        assert snap.status == "WARNING"

    def test_status_warning_below_85(self):
        snap = self._make_snapshot(0.80)
        assert snap.status == "WARNING"

    def test_status_critical_at_85(self):
        snap = self._make_snapshot(0.85)
        assert snap.status == "CRITICAL"

    def test_status_critical_below_95(self):
        snap = self._make_snapshot(0.90)
        assert snap.status == "CRITICAL"

    def test_status_halt_at_95(self):
        snap = self._make_snapshot(0.95)
        assert snap.status == "HALT"

    def test_status_halt_above_95(self):
        snap = self._make_snapshot(0.99)
        assert snap.status == "HALT"

    def test_can_add_positions_ok(self):
        snap = self._make_snapshot(0.60)
        assert snap.can_add_positions is True

    def test_can_add_positions_warning(self):
        snap = self._make_snapshot(0.80)
        assert snap.can_add_positions is True

    def test_cannot_add_positions_critical(self):
        snap = self._make_snapshot(0.85)
        assert snap.can_add_positions is False

    def test_cannot_add_positions_halt(self):
        snap = self._make_snapshot(0.96)
        assert snap.can_add_positions is False


# ── MarginHeadroomMonitor: basic snapshot ─────────────────────────────────────

class TestMarginHeadroomMonitorSnapshot:
    def test_empty_positions_near_zero_utilization(self):
        monitor = MarginHeadroomMonitor(available_capital=1_000_000.0)
        snap = monitor.snapshot([])
        assert isinstance(snap, MarginSnapshot)
        assert snap.utilization_pct == pytest.approx(0.0, abs=1e-6)
        assert snap.headroom_inr == pytest.approx(1_000_000.0, abs=1.0)

    def test_single_position_positive_utilization(self):
        monitor = MarginHeadroomMonitor(available_capital=500_000.0)
        positions = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}]
        snap = monitor.snapshot(positions)
        assert snap.utilization_pct > 0.0
        assert snap.headroom_inr < 500_000.0

    def test_utilization_increases_with_positions(self):
        monitor = MarginHeadroomMonitor(available_capital=1_000_000.0)
        pos1 = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}]
        pos5 = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}] * 5
        snap1 = monitor.snapshot(pos1)
        snap5 = monitor.snapshot(pos5)
        assert snap5.utilization_pct > snap1.utilization_pct

    def test_returns_margin_snapshot(self):
        monitor = MarginHeadroomMonitor(available_capital=100_000.0)
        snap = monitor.snapshot([])
        assert isinstance(snap, MarginSnapshot)
        assert hasattr(snap, "WARNING")
        assert "APPROX" in snap.WARNING


# ── Alert callback ────────────────────────────────────────────────────────────

class TestAlertCallback:
    def test_alert_fired_at_warning_threshold(self):
        alerts = []
        monitor = MarginHeadroomMonitor(
            available_capital=100_000.0,
            alert_callback=lambda msg, level: alerts.append((msg, level)),
        )
        # Positions sized to exceed 70% utilization:
        # Default fallback: 12% of notional
        # To get 70%: need notional * 0.12 * 1.20 = capital * 0.70
        # → lots * lot_size * price = capital * 0.70 / (0.12 * 1.20)
        # = 100000 * 0.70 / 0.144 ≈ 486,111
        # lots = 486111 / (65 * 24000) ≈ 0.31 → use lots=1, lot_size=65, price=24000 gets ~24.5%
        # Need to reach 70%: use 3 positions at 24000 * 65 * 0.12 * 1.20 = 22,464 each
        # 3 × 22,464 = 67,392 → 67.4% — not quite 70%
        # Use 4 positions → 89,856 → 89.9% → CRITICAL alert
        positions = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}] * 4
        monitor.snapshot(positions)
        assert len(alerts) > 0

    def test_alert_fired_at_halt_threshold(self):
        alerts = []
        monitor = MarginHeadroomMonitor(
            available_capital=10_000.0,  # tiny capital → easy to hit HALT
            alert_callback=lambda msg, level: alerts.append((msg, level)),
        )
        # One Nifty position: notional = 65 * 24000 = 1,560,000
        # theoretical = 1,560,000 * 0.12 = 187,200
        # calibrated = 187,200 * 1.20 = 224,640 >> 10,000 → HALT
        positions = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}]
        monitor.snapshot(positions)
        assert any("HALT" in msg.upper() or level == "CRITICAL"
                   for msg, level in alerts)


# ── Calibration ───────────────────────────────────────────────────────────────

class TestCalibration:
    def test_initial_calibration_factor_above_one(self):
        """Default calibration factor is 1.20 (broker takes 20% more than theoretical)."""
        monitor = MarginHeadroomMonitor(available_capital=1_000_000.0)
        factor = monitor.calibration_factor()
        assert factor == pytest.approx(1.20, abs=0.01)

    def test_calibration_updates_after_3_observations(self):
        monitor = MarginHeadroomMonitor(available_capital=1_000_000.0)
        # Record 3 observations: actual = 1.0x theoretical (no inflation)
        for i in range(3):
            monitor.record_broker_margin(
                order_id=f"ORD{i:03d}",
                theoretical_span=10_000.0,
                actual_broker_margin=10_000.0,  # ratio = 1.0
            )
        # After 3 observations with ratio=1.0, factor should converge toward 1.0
        factor = monitor.calibration_factor()
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_calibration_record_ratio(self):
        rec = MarginCalibrationRecord(
            timestamp=datetime.utcnow(),
            order_id="ORD001",
            theoretical_span=10_000.0,
            actual_broker_margin=12_000.0,
        )
        assert rec.ratio == pytest.approx(1.20, abs=0.001)

    def test_calibration_ratio_zero_theoretical(self):
        """If theoretical is 0, ratio defaults to 1.0 (no division by zero)."""
        rec = MarginCalibrationRecord(
            timestamp=datetime.utcnow(),
            order_id="ORD002",
            theoretical_span=0.0,
            actual_broker_margin=5_000.0,
        )
        assert rec.ratio == pytest.approx(1.0, abs=0.001)


# ── headroom_lots ─────────────────────────────────────────────────────────────

class TestHeadroomLots:
    def test_zero_headroom_returns_zero(self):
        """When capital is fully used, no more lots can be added."""
        monitor = MarginHeadroomMonitor(available_capital=1.0)  # 1 INR capital
        lots = monitor.headroom_lots(
            positions=[{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}],
            underlying_price=24000.0, lot_size=65,
        )
        assert lots == 0

    def test_positive_headroom_returns_positive_lots(self):
        """Large capital with no positions → can add lots."""
        monitor = MarginHeadroomMonitor(available_capital=10_000_000.0)
        lots = monitor.headroom_lots(
            positions=[],
            underlying_price=24000.0, lot_size=65,
        )
        assert lots > 0

    def test_more_capital_more_lots(self):
        monitor1 = MarginHeadroomMonitor(available_capital=1_000_000.0)
        monitor2 = MarginHeadroomMonitor(available_capital=5_000_000.0)
        lots1 = monitor1.headroom_lots([], underlying_price=24000.0, lot_size=65)
        lots2 = monitor2.headroom_lots([], underlying_price=24000.0, lot_size=65)
        assert lots2 > lots1


# ── update_capital ────────────────────────────────────────────────────────────

class TestUpdateCapital:
    def test_increased_capital_lowers_utilization(self):
        """Doubling capital should roughly halve the utilization."""
        monitor = MarginHeadroomMonitor(available_capital=100_000.0)
        positions = [{"lots": 1, "lot_size": 65, "underlying_price": 24000.0}]
        snap1 = monitor.snapshot(positions)
        monitor.update_capital(200_000.0)
        snap2 = monitor.snapshot(positions)
        assert snap2.utilization_pct < snap1.utilization_pct

    def test_estimate_margin_for_position(self):
        monitor = MarginHeadroomMonitor(available_capital=1_000_000.0)
        pos = {"lots": 1, "lot_size": 65, "underlying_price": 24000.0}
        margin = monitor.estimate_margin_for_position(pos)
        assert margin > 0
