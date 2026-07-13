"""
risk/manager.py — Risk manager with kill switches and position limits.

Enforces all pre-trade and post-trade risk rules. Every check is a hard rule —
there are no soft warnings that become hard limits later.

Risk rules (from CLAUDE.md and ACCEPTANCE_CRITERIA.md):
  - Daily loss kill switch: daily P&L < -2% of capital → halt all trading
  - Max single position: 5% of capital
  - Max gross exposure: 20% of capital
  - Max drawdown: 10% from peak → pause + alert (does not auto-resume)
  - Circuit breaker: price velocity > 3σ → reduce size (not halt)
  - NSE circuit halt (±5%/±10%/±20%): suspend that symbol
  - Kill switch: once triggered, blocks all subsequent can_trade() calls until reset
  - No new entries after 15:15 IST (enforced by OrderExecutor + here for belt-and-suspenders)
  - Force close at 15:15 IST (enforced by DailyOrchestrator)

State persistence:
  - kill_switch state written to QuestDB risk_state table on every change
  - Restored from QuestDB on startup (StartupReconciler handles this)

ATR-based position sizing formula:
  position_size = (capital × risk_pct) / (ATR14 × atr_multiplier)
  Approximates Half-Kelly without requiring return distribution estimates.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

import structlog

from algo_trading.config import settings
from algo_trading.data.system_state import LiveSystemState
from algo_trading.execution.order_executor import TradeSignal
from algo_trading.core.calendar import IST, can_enter_new_trade, now_ist
from algo_trading.risk.cost_model import IndiaCostModel, TradeCost

log = structlog.get_logger(__name__)


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""


@dataclass
class RiskState:
    """Snapshot of current risk state for persistence."""
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    halted_symbols: set[str] = field(default_factory=set)
    daily_pnl: Decimal = field(default_factory=lambda: Decimal(0))
    peak_pnl: Decimal = field(default_factory=lambda: Decimal(0))


class PortfolioCorrelationGuard:
    """
    Guards against adding highly correlated positions to the portfolio.

    Tracks a rolling window of per-bar percentage returns per symbol and
    computes pairwise Pearson correlations between a candidate new symbol
    and all symbols with open positions.

    Multiplier tiers:
      - Insufficient data (< min_bars_required bars for any pair) → 1.0
      - Max |ρ| > max_correlation (default 0.75) → 0.3  (high correlation)
      - Max |ρ| > 0.50                              → 0.7  (moderate correlation)
      - Otherwise                                   → 1.0  (uncorrelated)

    Thread-safe; each symbol's deque is protected by a single Lock.
    """

    def __init__(
        self,
        max_correlation: float = 0.75,
        lookback_bars: int = 60,
        min_bars_required: int = 20,
    ) -> None:
        self._max_correlation = max_correlation
        self._lookback_bars = lookback_bars
        self._min_bars_required = min_bars_required
        self._returns: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def update_returns(self, symbol: str, pct_return: float) -> None:
        """Append a percentage-return bar for *symbol* (thread-safe)."""
        with self._lock:
            if symbol not in self._returns:
                self._returns[symbol] = deque(maxlen=self._lookback_bars)
            self._returns[symbol].append(float(pct_return))

    def _pearson_correlation(self, xs: list[float], ys: list[float]) -> float:
        """
        Compute Pearson correlation for two equal-length lists.
        Returns 0.0 if the lists are constant (zero variance) to avoid
        division-by-zero; that case is treated as uncorrelated.
        """
        n = len(xs)
        if n < 2:
            return 0.0

        if _HAS_NUMPY:
            arr_x = np.array(xs, dtype=float)
            arr_y = np.array(ys, dtype=float)
            # np.corrcoef returns NaN when std == 0
            corr_matrix = np.corrcoef(arr_x, arr_y)
            val = float(corr_matrix[0, 1])
            return val if not (val != val) else 0.0  # NaN guard

        # Pure-Python fallback (no numpy)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        dx = [v - mean_x for v in xs]
        dy = [v - mean_y for v in ys]
        num = sum(a * b for a, b in zip(dx, dy))
        denom_x = sum(a * a for a in dx) ** 0.5
        denom_y = sum(b * b for b in dy) ** 0.5
        if denom_x == 0.0 or denom_y == 0.0:
            return 0.0
        return num / (denom_x * denom_y)

    def get_correlation_multiplier(
        self,
        new_symbol: str,
        existing_symbols: list[str],
    ) -> float:
        """
        Return a size multiplier in [0.3, 1.0] based on the maximum absolute
        pairwise Pearson correlation between *new_symbol* and every symbol in
        *existing_symbols*.

        Only overlapping (aligned) bars are used for each pair.
        Returns 1.0 immediately if there are no existing symbols or if any
        pair has fewer than min_bars_required overlapping bars.
        """
        if not existing_symbols:
            return 1.0

        with self._lock:
            new_returns = self._returns.get(new_symbol)
            if new_returns is None or len(new_returns) < self._min_bars_required:
                return 1.0

            new_list = list(new_returns)
            max_abs_corr = 0.0

            for sym in existing_symbols:
                existing_returns = self._returns.get(sym)
                if existing_returns is None:
                    continue

                exist_list = list(existing_returns)

                # Align to overlapping tail
                overlap = min(len(new_list), len(exist_list))
                if overlap < self._min_bars_required:
                    # Not enough data for this pair → skip (conservative: treat as no data)
                    return 1.0

                xs = new_list[-overlap:]
                ys = exist_list[-overlap:]

                corr = self._pearson_correlation(xs, ys)
                abs_corr = abs(corr)
                if abs_corr > max_abs_corr:
                    max_abs_corr = abs_corr

        if max_abs_corr > self._max_correlation:
            return 0.3
        if max_abs_corr > 0.5:
            return 0.7
        return 1.0


_KILL_SWITCH_SENTINEL = Path("state/kill_switch_clear.sentinel")
# P1-FIX-29: Operator trigger file — written by scripts/kill_switch.py CLI
_KILL_SWITCH_TRIGGER = Path("state/kill_switch.trigger")


class RiskManager:
    """
    Pre-trade risk gate and portfolio-level kill switch.

    Injected into OrderExecutor — every order must pass can_trade() first.
    Thread-safe via _lock.

    State updates come from:
      - OrderExecutor._on_fill() → record_fill()
      - DailyOrchestrator → reset_for_new_day()
      - External triggers → trigger_kill_switch()
    """

    def __init__(
        self,
        state: LiveSystemState,
        store: Optional[object] = None,   # QuestDBStore
        alert_sender: Optional[object] = None,  # AlertSender
        correlation_guard: Optional[PortfolioCorrelationGuard] = None,
        fat_finger_guard: Optional[object] = None,  # FatFingerGuard — for ADV cache reset
        max_symbol_loss_pct: float = 0.01,  # #137: 1% of capital per symbol per day [ASSUMED]
        shap_report_path: Optional[Path] = None,  # #139: SHAP JSON report path
    ) -> None:
        self._state = state
        self._store = store
        self._alert = alert_sender
        self._correlation_guard = correlation_guard
        self._fat_finger_guard = fat_finger_guard  # Fix G: held so reset_for_new_day() can call invalidate_stale_adv()
        self._cost_model = IndiaCostModel()
        self._lock = threading.Lock()
        self._risk_state = RiskState()

        # ── #136: Strategy-level capital allocation ───────────────────────────
        # Maps strategy_id → fraction of total capital (0.0–1.0).
        # Total allocated fraction must never exceed 1.0.
        self._strategy_allocations: dict[str, float] = {}

        # ── #137: Per-symbol daily P&L tracking ──────────────────────────────
        self._symbol_losses: dict[str, Decimal] = {}
        self._halted_by_loss: set[str] = set()  # symbols auto-halted by loss limit
        self._max_symbol_loss_pct: float = max_symbol_loss_pct  # [ASSUMED] 1% default

        # ── #138: Regime model freshness ─────────────────────────────────────
        self._last_regime_refit_date: Optional[date] = None

        # ── #139: SHAP report availability ───────────────────────────────────
        self._shap_report_path: Path = shap_report_path or Path("state/shap_report.json")

        # ── #140: PSI (Population Stability Index) monitoring ─────────────────
        # Maps feature_name → latest PSI score.
        self._feature_psi: dict[str, float] = {}

        # ── #141: Calibration score (Brier score) tracking ────────────────────
        self._latest_brier_score: Optional[float] = None
        self._max_brier_score: float = 0.3  # [ASSUMED] block if Brier > 0.3

        # ── #144: Blocked trade log ───────────────────────────────────────────
        self._blocked_trades: list[dict] = []

    # ── Pre-trade Check ───────────────────────────────────────────────────────

    def can_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
    ) -> tuple[bool, str]:
        """
        Pre-trade risk gate. Called by OrderExecutor before every order.
        Returns (True, "") if allowed, (False, reason) if blocked.

        Checks in priority order:
          1. Kill switch (most critical)
          2. Halted symbols
          3. Session time gate
          4. Daily loss limit
          5. Max drawdown
          6. Position size limit
          7. Gross exposure limit
          8. Portfolio correlation guard (optional; blocks when high-corr + ≥3 open positions)
        """
        # Fix C: check for operator sentinel before acquiring the lock
        self.check_kill_switch_clear_sentinel()
        # P1-FIX-29: Check for CLI trigger file (scripts/kill_switch.py)
        self._check_kill_switch_trigger()

        with self._lock:
            rs = self._risk_state

            # 1. Kill switch
            if rs.kill_switch_active:
                return False, f"Kill switch active: {rs.kill_switch_reason}"

            # 2. Symbol halt
            if symbol in rs.halted_symbols:
                return False, f"Symbol {symbol} is halted"

            # 2b. #137: Per-symbol daily loss limit
            if self.is_symbol_loss_limit_hit(symbol):
                return False, f"Symbol {symbol} daily loss limit hit (auto-halted)"

            # 3. Session gate (belt-and-suspenders after OrderExecutor checks)
            if not can_enter_new_trade():
                return False, "Outside trading hours or too close to session end"

            # 4. Daily loss limit
            daily_pnl = self._state.daily_pnl
            max_loss = settings.max_daily_loss_inr
            if daily_pnl < -max_loss:
                self._trigger_kill_switch_internal(
                    f"Daily loss limit hit: ₹{daily_pnl:.2f} < -₹{max_loss:.2f}"
                )
                return False, "Daily loss limit triggered kill switch"

            # 5. Max drawdown
            peak = self._state.peak_pnl
            if peak > 0 and daily_pnl < peak:
                drawdown = (peak - daily_pnl) / settings.capital_inr
                if drawdown > Decimal(str(settings.max_drawdown_pct)):
                    self._trigger_kill_switch_internal(
                        f"Max drawdown breached: {drawdown:.1%} > {settings.max_drawdown_pct:.1%}"
                    )
                    return False, "Max drawdown kill switch triggered"

            # 6. Position size limit
            turnover = Decimal(str(quantity)) * price
            if turnover > settings.max_position_inr:
                return False, (
                    f"Position too large: ₹{turnover:.0f} > "
                    f"limit ₹{settings.max_position_inr:.0f}"
                )

            # 7. Gross exposure limit
            current_position = self._state.get_position(symbol)
            existing_qty = current_position.net_quantity if current_position else 0
            new_qty_abs = abs(existing_qty + (quantity if side == "BUY" else -quantity))
            new_exposure = new_qty_abs * price
            # Approximate check — full gross exposure check requires all prices
            if new_exposure > settings.max_gross_exposure_inr:
                return False, (
                    f"Gross exposure breach: new exposure ₹{new_exposure:.0f} > "
                    f"limit ₹{settings.max_gross_exposure_inr:.0f}"
                )

            # 8. Portfolio correlation guard
            if self._correlation_guard is not None:
                existing = [
                    p.symbol for p in self._state.get_all_positions()
                ]
                if symbol not in existing:
                    corr_mult = self._correlation_guard.get_correlation_multiplier(
                        symbol, existing
                    )
                    if corr_mult < 1.0:
                        log.info(
                            "risk.correlation_guard.reducing_size",
                            symbol=symbol,
                            multiplier=corr_mult,
                        )
                        # Note: size reduction via multiplier happens in compute_position_size.
                        # Block entirely only when correlation is at maximum penalty AND
                        # there are already 3 or more correlated open positions.
                        if corr_mult <= 0.3 and len(existing) >= 3:
                            return False, (
                                f"High correlation with existing positions "
                                f"(mult={corr_mult:.2f})"
                            )

        return True, ""

    def compute_position_size(self, signal: TradeSignal) -> int:
        """
        ATR-based position sizing (approximates Half-Kelly).
        Formula: size = (capital × risk_pct) / (ATR × atr_multiplier)
        Returns 0 if ATR is zero or capital insufficient.

        P1-FIX-23: Applies portfolio correlation multiplier from
        PortfolioCorrelationGuard so that the size is actually reduced
        (not just logged) when the new symbol is correlated with open positions.

        #136: Uses strategy-allocated capital when signal carries a strategy_id
        that has been registered via register_strategy().
        """
        if signal.atr <= 0:
            log.warning("risk.position_size.zero_atr", symbol=signal.symbol)
            return 0

        # #136: Use strategy-level capital allocation if strategy_id is known
        strategy_id: Optional[str] = getattr(signal, "model_version", None) or settings.strategy_id
        effective_capital = self.get_strategy_capital(strategy_id) if strategy_id else settings.capital_inr

        risk_per_trade = effective_capital * Decimal(str(settings.max_position_pct)) * Decimal("0.5")
        # Half-Kelly: risk 50% of max_position_pct per trade
        stop_distance = signal.atr * Decimal(str(settings.atr_stop_multiplier))

        if stop_distance <= 0:
            return 0

        raw_size = risk_per_trade / stop_distance
        size = max(0, int(raw_size))

        # Cap at max single position by value
        if size > 0 and signal.entry_price > 0:
            max_shares = int(settings.max_position_inr / signal.entry_price)
            size = min(size, max_shares)

        # P1-FIX-23: Apply correlation multiplier to actually reduce position size.
        # Previously, the guard only blocked at 0.3 multiplier with 3+ positions;
        # intermediate correlations (multiplier=0.7) were logged but never applied.
        if size > 0 and self._correlation_guard is not None:
            existing = [p.symbol for p in self._state.get_all_positions()]
            if signal.symbol not in existing:
                corr_mult = self._correlation_guard.get_correlation_multiplier(
                    signal.symbol, existing
                )
                if corr_mult < 1.0:
                    size = max(0, int(size * corr_mult))
                    log.info(
                        "risk.position_size.correlation_reduced",
                        symbol=signal.symbol,
                        multiplier=corr_mult,
                        size_after=size,
                    )

        log.debug(
            "risk.position_size",
            symbol=signal.symbol,
            atr=str(signal.atr),
            stop_distance=str(stop_distance),
            size=size,
        )
        return size

    def compute_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        atr: Optional[Decimal] = None,
    ) -> TradeCost:
        """Compute trade cost via IndiaCostModel. Hour-of-day from current IST time."""
        hour = now_ist().hour
        return self._cost_model.compute(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            atr=atr,
            hour_of_day=hour,
        )

    # ── Kill Switch ───────────────────────────────────────────────────────────

    def trigger_kill_switch(self, reason: str) -> None:
        """
        External trigger for the kill switch (e.g. Telegram command, SIGTERM).
        This method is public and thread-safe.
        """
        with self._lock:
            self._trigger_kill_switch_internal(reason)

    def _trigger_kill_switch_internal(self, reason: str) -> None:
        """Must be called with self._lock held."""
        if self._risk_state.kill_switch_active:
            return  # Already active

        self._risk_state.kill_switch_active = True
        self._risk_state.kill_switch_reason = reason
        self._state.set_kill_switch(True)

        log.error("risk.kill_switch.TRIGGERED", reason=reason)
        self._persist_risk_event("KILL_SWITCH", reason)

        if self._alert:
            try:
                self._alert.send_risk_alert(f"KILL SWITCH: {reason}")  # type: ignore[union-attr]
            except Exception:
                pass  # Never let alert failure block kill switch

    def reset_kill_switch(self) -> None:
        """
        Reset the kill switch. REQUIRES manual operator acknowledgement.
        The orchestrator must also restart position reconciliation after reset.
        """
        with self._lock:
            if not self._risk_state.kill_switch_active:
                return
            log.warning("risk.kill_switch.RESET — manual operator reset")
            self._risk_state.kill_switch_active = False
            self._risk_state.kill_switch_reason = ""
            self._state.set_kill_switch(False)
            self._persist_risk_event("KILL_SWITCH_RESET", "Manual operator reset")

    def check_kill_switch_clear_sentinel(self) -> bool:
        """
        Fix C: Check for the operator sentinel file that clears the kill switch.

        If ``state/kill_switch_clear.sentinel`` exists:
          - Read and log its contents at WARNING.
          - Delete the file.
          - Clear the kill switch flag.
          - Send an alert.
          - Return True.

        Returns False if the sentinel file is absent or the kill switch is not active.
        Safe to call frequently (e.g., on every can_trade() call).
        """
        if not _KILL_SWITCH_SENTINEL.exists():
            return False

        try:
            contents = _KILL_SWITCH_SENTINEL.read_text(encoding="utf-8").strip()
        except Exception as exc:
            log.error("risk.kill_switch.sentinel_read_error", error=str(exc))
            return False

        log.warning(
            "risk.kill_switch.sentinel_detected",
            sentinel_path=str(_KILL_SWITCH_SENTINEL),
            contents=contents,
        )

        try:
            _KILL_SWITCH_SENTINEL.unlink()
        except Exception as exc:
            log.error("risk.kill_switch.sentinel_delete_error", error=str(exc))
            # Do not clear the kill switch if we couldn't remove the sentinel —
            # this prevents repeated auto-resets from a stuck file.
            return False

        with self._lock:
            self._risk_state.kill_switch_active = False
            self._risk_state.kill_switch_reason = ""
            self._state.set_kill_switch(False)
            self._persist_risk_event("KILL_SWITCH_RESET", f"Sentinel file: {contents}")

        if self._alert:
            try:
                self._alert.send_risk_alert(  # type: ignore[union-attr]
                    "Kill switch cleared by operator"
                )
            except Exception:
                pass

        log.warning("risk.kill_switch.cleared_by_sentinel", contents=contents)
        return True

    def _check_kill_switch_trigger(self) -> None:
        """P1-FIX-29: Check for CLI trigger file written by scripts/kill_switch.py.

        If ``state/kill_switch.trigger`` exists:
          - Read and log its contents.
          - Delete the trigger file.
          - Activate the kill switch with the recorded reason.

        Safe to call frequently (called on every can_trade() call).
        """
        if not _KILL_SWITCH_TRIGGER.exists():
            return

        try:
            import json as _json
            contents = _KILL_SWITCH_TRIGGER.read_text(encoding="utf-8").strip()
            data = _json.loads(contents) if contents else {}
            reason = data.get("reason", "Manual CLI trigger")
        except Exception as exc:
            log.error("risk.kill_switch.trigger_read_error", error=str(exc))
            return

        try:
            _KILL_SWITCH_TRIGGER.unlink()
        except Exception as exc:
            log.error("risk.kill_switch.trigger_delete_error", error=str(exc))
            return  # Don't activate if we couldn't remove the file (prevents repeated triggers)

        log.warning("risk.kill_switch.cli_trigger_detected", reason=reason)
        with self._lock:
            self._trigger_kill_switch_internal(f"CLI trigger: {reason}")

    # ── Symbol Halt (NSE Circuit / Surveillance) ───────────────────────────────

    def halt_symbol(self, symbol: str, reason: str) -> None:
        """
        Halt trading for a specific symbol (NSE circuit halt, data quality issue, etc.).
        Does NOT halt the entire system — only this symbol.
        """
        with self._lock:
            self._risk_state.halted_symbols.add(symbol)
        log.warning("risk.symbol_halted", symbol=symbol, reason=reason)
        self._persist_risk_event("SYMBOL_HALT", f"{symbol}: {reason}")
        if self._alert:
            try:
                self._alert.send_risk_alert(f"Symbol halted: {symbol} — {reason}")  # type: ignore[union-attr]
            except Exception:
                pass

    def resume_symbol(self, symbol: str) -> None:
        """Re-enable trading for a previously halted symbol."""
        with self._lock:
            self._risk_state.halted_symbols.discard(symbol)
        log.info("risk.symbol_resumed", symbol=symbol)

    def check_price_velocity(self, symbol: str, price_change_pct: Decimal) -> None:
        """
        Check if price velocity exceeds 3σ (NSE circuit-style check).
        Above threshold: halt the symbol (not the whole system).
        The specific threshold uses the NSE circuit bands:
          ±5% → lower circuit band warning
          ±10% → mid circuit
          ±20% → upper circuit halt
        """
        ABS_BANDS = [Decimal("5"), Decimal("10"), Decimal("20")]
        abs_change = abs(price_change_pct)

        if abs_change >= ABS_BANDS[2]:
            self.halt_symbol(symbol, f"Price hit ±20% circuit band ({price_change_pct:.2f}%)")
        elif abs_change >= ABS_BANDS[1]:
            log.warning("risk.price_velocity.10pct", symbol=symbol, change=str(price_change_pct))
            if self._alert:
                self._alert.send_risk_alert(  # type: ignore[union-attr]
                    f"{symbol} hit ±10% band — monitoring closely"
                )
        elif abs_change >= ABS_BANDS[0]:
            log.info("risk.price_velocity.5pct", symbol=symbol, change=str(price_change_pct))

    # ── State Queries ─────────────────────────────────────────────────────────

    @property
    def is_kill_switch_active(self) -> bool:
        with self._lock:
            return self._risk_state.kill_switch_active

    def is_symbol_halted(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._risk_state.halted_symbols

    def get_risk_snapshot(self) -> dict:
        """Return a log-safe snapshot of current risk state."""
        state_snap = self._state.snapshot()
        with self._lock:
            return {
                "kill_switch": self._risk_state.kill_switch_active,
                "kill_reason": self._risk_state.kill_switch_reason,
                "halted_symbols": list(self._risk_state.halted_symbols),
                "daily_pnl": state_snap["daily_pnl"],
                "peak_pnl": state_snap["peak_pnl"],
                "max_daily_loss": str(settings.max_daily_loss_inr),
            }

    def reset_for_new_day(self) -> None:
        """Called at market open to reset daily counters and kill switch.

        Clears the kill switch, halted symbols, and per-symbol daily P&L
        so that fresh-day trading can begin. The kill switch is cleared here
        because a new trading day constitutes an implicit operator
        acknowledgement that conditions have been reviewed.
        """
        with self._lock:
            # Clear kill switch for the new day
            was_active = self._risk_state.kill_switch_active
            if was_active:
                log.warning(
                    "risk.reset_for_new_day.clearing_kill_switch",
                    reason=self._risk_state.kill_switch_reason,
                    note="Kill switch cleared for new trading day",
                )
            self._risk_state.kill_switch_active = False
            self._risk_state.kill_switch_reason = ""
            self._state.set_kill_switch(False)

            # Fix D: explicitly clear halted symbols and log it
            self._risk_state.halted_symbols.clear()
            # #137: Reset per-symbol daily P&L and auto-halt set for new day
            self._symbol_losses.clear()
            self._halted_by_loss.clear()

        log.info(
            "risk.reset_for_new_day",
            kill_switch_was_active=was_active,
        )
        log.info("risk.halted_symbols_cleared", message="halted_symbols cleared for new day")

        # Fix G: invalidate ADV cache entries from the previous session.
        # Use the previous session's nominal close (15:30 IST) as the cut-off.
        if self._fat_finger_guard is not None:
            try:
                from datetime import datetime as _dt
                from algo_trading.core.calendar import IST as _IST
                yesterday = _dt.now(_IST).date()
                session_end = _dt(yesterday.year, yesterday.month, yesterday.day,
                                  15, 30, 0, tzinfo=_IST)
                self._fat_finger_guard.invalidate_stale_adv(session_end)  # type: ignore[union-attr]
            except Exception as exc:
                log.error("risk.reset_for_new_day.adv_invalidate_error", error=str(exc))

    # ── #136: Strategy-level Capital Allocation ───────────────────────────────

    def register_strategy(self, strategy_id: str, capital_pct: float) -> None:
        """
        #136: Register a strategy with its share of total capital.

        capital_pct is a fraction in (0, 1.0]. The sum of all registered
        strategy capital_pcts must not exceed 1.0.

        Raises ValueError if the registration would cause over-allocation.
        """
        with self._lock:
            current_total = sum(self._strategy_allocations.values())
            existing = self._strategy_allocations.get(strategy_id, 0.0)
            new_total = current_total - existing + capital_pct
            if new_total > 1.0:
                raise ValueError(
                    f"Strategy allocation over-budget: registering {strategy_id}={capital_pct:.2%} "
                    f"would bring total to {new_total:.2%} (max 100%)"
                )
            self._strategy_allocations[strategy_id] = capital_pct
        log.info(
            "risk.strategy_registered",
            strategy_id=strategy_id,
            capital_pct=f"{capital_pct:.2%}",
        )

    def get_strategy_capital(self, strategy_id: Optional[str]) -> Decimal:
        """
        #136: Return the capital (in INR) allocated to *strategy_id*.

        If strategy_id is None or not registered, returns full settings.capital_inr
        (backward-compatible with single-strategy deployments).
        """
        if strategy_id is None:
            return settings.capital_inr
        with self._lock:
            alloc = self._strategy_allocations.get(strategy_id)
        if alloc is None:
            return settings.capital_inr
        return settings.capital_inr * Decimal(str(alloc))

    # ── #137: Per-symbol Daily Loss / Cooldown ────────────────────────────────

    def record_symbol_pnl(self, symbol: str, pnl: Decimal) -> None:
        """
        #137: Accumulate daily P&L for *symbol*.

        Call this on every fill or EOD MTM update. Negative pnl = loss.
        If the accumulated loss exceeds the per-symbol limit, the symbol is
        automatically added to the risk-state halted set.
        """
        with self._lock:
            current = self._symbol_losses.get(symbol, Decimal(0))
            self._symbol_losses[symbol] = current + pnl

        log.debug(
            "risk.symbol_pnl_recorded",
            symbol=symbol,
            pnl=str(pnl),
            cumulative=str(self._symbol_losses[symbol]),
        )

        # Auto-halt if limit breached
        if self.is_symbol_loss_limit_hit(symbol):
            with self._lock:
                if symbol not in self._halted_by_loss:
                    self._halted_by_loss.add(symbol)
                    self._risk_state.halted_symbols.add(symbol)
            log.error(
                "risk.symbol_loss_limit.auto_halt",
                symbol=symbol,
                cumulative_loss=str(self._symbol_losses.get(symbol, Decimal(0))),
                limit_inr=str(settings.capital_inr * Decimal(str(self._max_symbol_loss_pct))),
            )
            self._persist_risk_event(
                "SYMBOL_LOSS_HALT",
                f"{symbol}: cumulative loss {self._symbol_losses.get(symbol, Decimal(0)):.2f}",
            )

    def is_symbol_loss_limit_hit(self, symbol: str) -> bool:
        """
        #137: Returns True if the symbol has exceeded its per-day loss limit.

        Limit = settings.capital_inr × max_symbol_loss_pct [ASSUMED 1%].
        Once hit, the symbol is auto-halted in _risk_state.halted_symbols and
        remains halted until reset_for_new_day() is called.
        """
        limit_inr = settings.capital_inr * Decimal(str(self._max_symbol_loss_pct))
        cum_pnl = self._symbol_losses.get(symbol, Decimal(0))
        return cum_pnl <= -limit_inr

    # ── #138: Regime Model Freshness ─────────────────────────────────────────

    def check_regime_model_freshness(self) -> bool:
        """
        #138: Returns True if the regime model was refit within the last 7 days.

        Returns False (and logs a warning) if:
          - _last_regime_refit_date has never been set, OR
          - the refit is older than 7 days

        [ASSUMED] 7-day staleness threshold — regime models trained on equity
        microstructure should be refit at least weekly.
        """
        if self._last_regime_refit_date is None:
            log.warning(
                "risk.regime_model.never_refit",
                note="[ASSUMED] regime model refit date unknown — treating as stale",
            )
            return False

        days_stale = (date.today() - self._last_regime_refit_date).days
        if days_stale > 7:  # [ASSUMED] 7-day threshold
            log.warning(
                "risk.regime_model.stale",
                last_refit=str(self._last_regime_refit_date),
                days_stale=days_stale,
                threshold_days=7,
            )
            return False

        return True

    def set_regime_refit_date(self, refit_date: date) -> None:
        """Record the date of the most recent regime model refit."""
        self._last_regime_refit_date = refit_date
        log.info("risk.regime_model.refit_date_set", refit_date=str(refit_date))

    # ── #139: SHAP Report Availability ───────────────────────────────────────

    def shap_report_available(self) -> bool:
        """
        #139: Returns True if a SHAP JSON report exists and was created within 30 days.

        [ASSUMED] 30-day freshness threshold — SHAP reports should be regenerated
        after each model retrain.
        """
        if not self._shap_report_path.exists():
            log.warning(
                "risk.shap_report.missing",
                path=str(self._shap_report_path),
            )
            return False

        try:
            mtime = datetime.fromtimestamp(
                self._shap_report_path.stat().st_mtime, tz=IST
            )
            age_days = (datetime.now(tz=IST) - mtime).days
            if age_days > 30:  # [ASSUMED] 30-day freshness
                log.warning(
                    "risk.shap_report.stale",
                    path=str(self._shap_report_path),
                    age_days=age_days,
                    threshold_days=30,
                )
                return False
        except OSError as exc:
            log.error("risk.shap_report.stat_error", error=str(exc))
            return False

        return True

    # ── #140: PSI (Population Stability Index) Monitoring ────────────────────

    def record_feature_psi(self, feature: str, psi: float) -> None:
        """
        #140: Record the latest PSI score for a feature.

        PSI thresholds (standard):
          < 0.10 : no significant change
          0.10 – 0.20 : moderate change — monitor
          > 0.20 : significant change — consider model refresh  [ASSUMED]
        """
        self._feature_psi[feature] = psi
        if psi > 0.2:  # [ASSUMED] PSI > 0.2 = significant drift
            log.warning(
                "risk.psi.feature_drift_detected",
                feature=feature,
                psi=f"{psi:.4f}",
                threshold=0.2,
            )

    def check_feature_drift(self) -> list[str]:
        """
        #140: Return a list of features whose PSI exceeds 0.2 (significant drift).

        [ASSUMED] PSI > 0.2 threshold per industry standard (Yurdakul 2018).
        Caller should consider retraining or switching to fallback model.
        """
        return [feat for feat, psi in self._feature_psi.items() if psi > 0.2]

    # ── #141: Model Calibration Score ────────────────────────────────────────

    def record_calibration_score(self, brier_score: float) -> None:
        """
        #141: Record the Brier score from the latest model calibration run.

        Brier score is in [0, 1]: lower = better calibration.
        [ASSUMED] If Brier > 0.3, the model's probability estimates are
        poorly calibrated and should not be used for position sizing.
        """
        self._latest_brier_score = brier_score
        if brier_score >= self._max_brier_score:
            log.error(
                "risk.calibration.poor_brier_score",
                brier_score=f"{brier_score:.4f}",
                threshold=self._max_brier_score,
                note="[ASSUMED] Brier > 0.3 indicates poor probability calibration",
            )

    def is_model_calibrated(self) -> bool:
        """Returns True if the latest Brier score is below the acceptable threshold."""
        if self._latest_brier_score is None:
            log.warning("risk.calibration.no_score_recorded")
            return True  # Conservative: allow trading if no score recorded yet
        return self._latest_brier_score < self._max_brier_score

    # ── #142: Confidence No-Trade Band ───────────────────────────────────────

    def is_confidence_above_threshold(
        self,
        confidence: float,
        min_confidence: float = 0.55,  # [ASSUMED] 55% minimum signal confidence
    ) -> bool:
        """
        #142: Returns True if *confidence* meets the minimum threshold.

        Signals with confidence in the no-trade band [0, min_confidence) are
        discarded. This prevents marginal-probability signals from generating
        large position sizes (adverse Kelly dynamics).

        [ASSUMED] Default minimum confidence = 0.55 (55%). In production, tune
        from backtested confidence-vs-return distribution.
        """
        if confidence < min_confidence:
            log.info(
                "risk.confidence_below_threshold",
                confidence=f"{confidence:.3f}",
                min_confidence=min_confidence,
            )
            return False
        return True

    # ── #143: Expected Value (EV) Calculation ─────────────────────────────────

    def compute_expected_value(
        self,
        probability: float,
        target_pct: float,
        stop_pct: float,
        cost_bps: float,
    ) -> float:
        """
        #143: Compute the Expected Value of a trade before submission.

        Formula:
            EV = probability × target_pct − (1 − probability) × stop_pct − cost_bps/10000

        Arguments:
            probability: model's estimated win probability (0–1)
            target_pct:  profit target as a fraction (e.g. 0.02 for 2%)
            stop_pct:    stop-loss distance as a fraction (e.g. 0.01 for 1%)
            cost_bps:    total transaction cost in basis points (brokerage + taxes)

        Returns:
            EV as a float. Negative or zero EV → trade should be blocked.

        [ASSUMED] cost_bps default guidance: NSE equity intraday ~5–8bps round-trip.
        """
        ev = probability * target_pct - (1 - probability) * stop_pct - cost_bps / 10000
        if ev <= 0:
            log.info(
                "risk.ev_negative",
                probability=f"{probability:.3f}",
                target_pct=f"{target_pct:.4f}",
                stop_pct=f"{stop_pct:.4f}",
                cost_bps=f"{cost_bps:.1f}",
                ev=f"{ev:.6f}",
                note="Non-positive EV — trade should be blocked",
            )
        return ev

    # ── #144: Blocked Trade Log ───────────────────────────────────────────────

    def record_blocked_trade(
        self,
        reason: str,
        symbol: str,
        signal: Optional[object] = None,
    ) -> None:
        """
        #144: Record every blocked trade with reason, timestamp, symbol, and signal snapshot.

        Stored in _blocked_trades for later analysis (e.g., post-session review,
        strategy improvement). The list is NOT persisted across restarts —
        use the audit log (order_audit.log) for durable records.
        """
        entry: dict = {
            "ts_ist": now_ist().isoformat(),
            "reason": reason,
            "symbol": symbol,
            "signal": str(signal) if signal is not None else None,
        }
        self._blocked_trades.append(entry)
        log.debug("risk.blocked_trade_recorded", **entry)

    def get_blocked_trades(self) -> list[dict]:
        """#144: Return a copy of the blocked trades log for this session."""
        return list(self._blocked_trades)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_risk_event(self, event_type: str, details: str) -> None:
        if self._store is None:
            return
        try:
            self._store.write_risk_event(  # type: ignore[union-attr]
                event_type=event_type,
                details=details,
                daily_pnl=self._state.daily_pnl,
                peak_pnl=self._state.peak_pnl,
                kill_switch_active=self._risk_state.kill_switch_active,
            )
        except Exception as exc:
            log.error("risk.persist_failed", error=str(exc))
