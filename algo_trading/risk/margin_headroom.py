"""
algo_trading/risk/margin_headroom.py — Real-time margin utilization monitoring.

Monitors current margin used vs available capital and alerts BEFORE a margin
breach occurs. Provides position-level and portfolio-level headroom.

Sources:
  SPAN margin computation: algo_trading/fo/margin.py (SPANMarginEngine)
  Margin estimator caveat (FO_SYSTEM_DESIGN.md Section 9.5):
    "The margin estimator is an estimator, not truth. Live broker margin
     behavior drifts from theoretical SPAN due to broker RMS overlays,
     spread recognition differences, short-option treatment, and intraday
     rule changes."

Required actions (per design doc):
  - Compare modeled vs actual margin in Phase 3 and Phase 4
  - Store per-order pre-trade and post-trade margin snapshots for calibration
  - Maintain calibration layer adjusting theoretical SPAN based on observed behavior

Alerting thresholds:
  > 70% utilization: WARNING
  > 85% utilization: CRITICAL (stop new entries)
  > 95% utilization: HALT (flatten positions)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Alert thresholds (% of available capital used for margin)
_WARN_THRESHOLD     = 0.70   # 70%: warning — reduce new position sizes
_CRITICAL_THRESHOLD = 0.85   # 85%: no new positions
_HALT_THRESHOLD     = 0.95   # 95%: emergency flatten


@dataclass
class MarginSnapshot:
    """Point-in-time margin utilization snapshot."""
    timestamp: datetime
    available_capital: float      # total capital allocated for margin
    theoretical_span: float       # SPAN + exposure margin (from SPANMarginEngine)
    broker_actual: float | None   # actual margin used per broker API (None if not available)
    calibrated_estimate: float    # theoretical × calibration_factor
    utilization_pct: float        # calibrated_estimate / available_capital
    headroom_inr: float           # available_capital - calibrated_estimate
    positions_count: int
    WARNING: str = "[APPROX-NOT-OFFICIAL-SPAN]: theoretical estimate only"

    @property
    def status(self) -> str:
        if self.utilization_pct >= _HALT_THRESHOLD:
            return "HALT"
        if self.utilization_pct >= _CRITICAL_THRESHOLD:
            return "CRITICAL"
        if self.utilization_pct >= _WARN_THRESHOLD:
            return "WARNING"
        return "OK"

    @property
    def can_add_positions(self) -> bool:
        return self.utilization_pct < _CRITICAL_THRESHOLD


@dataclass
class MarginCalibrationRecord:
    """
    Record for calibrating theoretical SPAN against actual broker margin.

    Stored per-order. Used to compute calibration_factor:
      calibration_factor = mean(actual / theoretical) over recent N orders
    """
    timestamp: datetime
    order_id: str
    theoretical_span: float
    actual_broker_margin: float
    ratio: float = field(init=False)  # actual / theoretical

    def __post_init__(self) -> None:
        self.ratio = (
            self.actual_broker_margin / self.theoretical_span
            if self.theoretical_span > 0 else 1.0
        )


class MarginHeadroomMonitor:
    """
    Real-time margin utilization monitor for NSE F&O portfolios.

    Computes utilization = calibrated_margin_estimate / available_capital.
    Fires alert callbacks when thresholds are breached.
    Maintains calibration history for improving margin estimates over time.

    Parameters
    ----------
    available_capital : float
        Capital allocated for margin (not total capital — usually a subset).
    margin_model_provider : MarginModelProvider | None
        Interface to SPAN margin computation (fo.interfaces.MarginModelProvider).
        If None, uses simple 12% notional estimate [ASSUMED].
    alert_callback : callable | None
        Called with (message: str, level: str) on threshold breach.
    calibration_window : int
        Number of recent broker observations used for calibration factor.

    Usage
    -----
    monitor = MarginHeadroomMonitor(available_capital=500_000)

    # Before placing a new trade
    snapshot = monitor.snapshot(current_positions)
    if not snapshot.can_add_positions:
        log.warning("No headroom — skipping trade")
        return

    # Check if enough headroom for a specific new position
    extra_needed = monitor.estimate_margin_for_position(new_position)
    if snapshot.headroom_inr < extra_needed:
        log.warning("Insufficient headroom for new position")
        return
    """

    def __init__(
        self,
        available_capital: float,
        margin_model_provider: Any = None,
        alert_callback: Any = None,
        calibration_window: int = 20,
    ) -> None:
        self._capital = available_capital
        self._margin_provider = margin_model_provider
        self._alert_cb = alert_callback
        self._cal_window = calibration_window
        self._lock = threading.Lock()
        self._calibration_history: list[MarginCalibrationRecord] = []
        self._calibration_factor: float = 1.20  # Default: assume broker takes 20% more [ASSUMED]
        self._last_snapshot: MarginSnapshot | None = None

    def snapshot(self, positions: list[dict]) -> MarginSnapshot:
        """
        Compute current margin utilization snapshot.

        Parameters
        ----------
        positions : list[dict]
            Current open positions in the format accepted by MarginModelProvider.
        """
        with self._lock:
            theoretical = self._compute_theoretical_margin(positions)
            calibrated = theoretical * self._calibration_factor
            utilization = calibrated / self._capital if self._capital > 0 else 0.0
            headroom = self._capital - calibrated

            snap = MarginSnapshot(
                timestamp=datetime.now(timezone.utc),
                available_capital=self._capital,
                theoretical_span=theoretical,
                broker_actual=None,
                calibrated_estimate=calibrated,
                utilization_pct=utilization,
                headroom_inr=headroom,
                positions_count=len(positions),
            )
            self._last_snapshot = snap

            # Fire alerts
            if snap.status == "HALT":
                msg = (f"MARGIN HALT: utilization={utilization:.1%}, "
                       f"headroom=₹{headroom:,.0f}")
                log.critical("margin.halt", utilization=f"{utilization:.1%}")
                self._send_alert(msg, "CRITICAL")
            elif snap.status == "CRITICAL":
                msg = f"MARGIN CRITICAL: utilization={utilization:.1%} — no new positions"
                log.error("margin.critical", utilization=f"{utilization:.1%}")
                self._send_alert(msg, "CRITICAL")
            elif snap.status == "WARNING":
                msg = f"MARGIN WARNING: utilization={utilization:.1%}"
                log.warning("margin.warning", utilization=f"{utilization:.1%}")
                self._send_alert(msg, "WARNING")

            return snap

    def estimate_margin_for_position(self, new_position: dict) -> float:
        """
        Estimate the additional margin required for a hypothetical new position.

        Parameters
        ----------
        new_position : dict
            Position dict with keys: lots, lot_size, underlying_price, etc.

        Returns
        -------
        float
            Estimated additional margin in INR.
        """
        theoretical = self._compute_theoretical_margin([new_position])
        return theoretical * self._calibration_factor

    def update_calibration(
        self, actual_margin: float, theoretical_margin: float
    ) -> None:
        """
        Update calibration factor using EMA after each fill.

        Uses exponential moving average:
            calibration_factor = 0.9 * calibration_factor + 0.1 * (actual / theoretical)

        Parameters
        ----------
        actual_margin : float
            Actual broker-reported margin consumed.
        theoretical_margin : float
            Theoretical SPAN margin estimate used pre-trade.
        """
        if theoretical_margin <= 0:
            log.warning(
                "margin.calibration_skipped",
                reason="theoretical_margin <= 0",
            )
            return
        ratio = actual_margin / theoretical_margin
        with self._lock:
            self._calibration_factor = (
                0.9 * self._calibration_factor + 0.1 * ratio
            )
            log.info(
                "margin.calibration_updated_ema",
                factor=f"{self._calibration_factor:.3f}",
                actual=actual_margin,
                theoretical=theoretical_margin,
                ratio=f"{ratio:.3f}",
            )

    def record_broker_margin(
        self, order_id: str, theoretical_span: float, actual_broker_margin: float
    ) -> None:
        """
        Record a broker margin observation for calibration.

        Call this after each order is filled with the pre-trade theoretical
        margin estimate and the actual broker-reported margin consumed.
        Also updates the calibration factor via EMA.
        """
        with self._lock:
            record = MarginCalibrationRecord(
                timestamp=datetime.now(timezone.utc),
                order_id=order_id,
                theoretical_span=theoretical_span,
                actual_broker_margin=actual_broker_margin,
            )
            self._calibration_history.append(record)

            # Keep only the last N observations
            if len(self._calibration_history) > self._cal_window:
                self._calibration_history = (
                    self._calibration_history[-self._cal_window:]
                )

            # Recompute calibration factor (mean over window for backward compat)
            if len(self._calibration_history) >= 3:
                ratios = [r.ratio for r in self._calibration_history]
                self._calibration_factor = sum(ratios) / len(ratios)
                log.info(
                    "margin.calibration_updated",
                    factor=f"{self._calibration_factor:.3f}",
                    n_observations=len(self._calibration_history),
                )

        # Also apply EMA update so calibration tracks recent fills closely
        self.update_calibration(actual_broker_margin, theoretical_span)

    def calibration_factor(self) -> float:
        """Current calibration factor (actual/theoretical ratio)."""
        with self._lock:
            return self._calibration_factor

    def headroom_lots(
        self,
        positions: list[dict],
        underlying_price: float,
        lot_size: int = 65,
        margin_pct_assumed: float = 0.12,
    ) -> int:
        """
        How many additional lots can be opened with current headroom?

        Uses a simplified per-lot margin estimate.
        [ASSUMED] margin_pct_assumed: check SEBI/NSE margin circular for exact value.

        Parameters
        ----------
        underlying_price : float
            Current Nifty/BankNifty index level.
        lot_size : int
            Lot size for the contract (e.g. 65 for Nifty).
        margin_pct_assumed : float
            Approximate margin as % of notional [ASSUMED].

        Returns
        -------
        int
            Maximum additional lots (conservative estimate).
        """
        snap = self.snapshot(positions)
        if snap.headroom_inr <= 0:
            return 0
        per_lot_notional = lot_size * underlying_price
        per_lot_margin = per_lot_notional * margin_pct_assumed * self._calibration_factor
        return int(snap.headroom_inr / per_lot_margin)

    def update_capital(self, new_capital: float) -> None:
        """Update available capital (e.g. after P&L changes, capital addition)."""
        with self._lock:
            self._capital = new_capital
            log.info("margin.capital_updated", capital=f"₹{new_capital:,.0f}")

    def _compute_theoretical_margin(self, positions: list[dict]) -> float:
        """Compute theoretical SPAN + exposure margin."""
        if self._margin_provider is not None:
            try:
                return self._margin_provider.estimate_total_margin(positions)
            except Exception as e:
                log.warning("margin.provider_error", error=str(e))

        # Fallback: 12% of notional (ASSUMED — verify SEBI circular)
        total = 0.0
        for pos in positions:
            lots = pos.get("lots", 0)
            lot_size = pos.get("lot_size", 1)
            underlying = pos.get("underlying_price", 0.0)
            notional = abs(lots) * lot_size * underlying
            total += notional * 0.12  # [ASSUMED]
        return total

    def _send_alert(self, message: str, level: str = "WARNING") -> None:
        if self._alert_cb:
            try:
                self._alert_cb(message, level)
            except Exception as exc:
                log.error(
                    "margin.alert_callback_failed",
                    error=str(exc),
                    message=message,
                    level=level,
                )
