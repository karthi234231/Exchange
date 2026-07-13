"""
algo_trading/risk/gap_risk_model.py — Overnight gap risk modeling for NSE F&O.

Overnight gaps are a significant risk for options sellers (short-gamma/short-vega
positions). A large gap move on open can:
  1. Breach stop-loss prices before they can be executed (gap through stop)
  2. Cause significant IV expansion (gap + vol spike)
  3. Move short strikes deep ITM before the market even opens

This module models overnight gap risk using historical Nifty gap distribution
and computes worst-case P&L for the current portfolio.

Gap definition: gap% = (open_price - prev_close) / prev_close × 100

Gap statistics for Nifty (historical, approximate — [ASSUMED] calibrate from data):
  Mean daily gap:  ~0.0% (symmetric)
  Std of gaps:     ~0.5% (normal trading days)
  95th percentile: ±1.5%
  99th percentile: ±3.0%
  Extreme (0.1%):  ±5% or more (budget/RBI/geopolitical events)

These are ASSUMED starting values based on general equity market behavior.
MUST be calibrated from actual NSE historical data before use in production.

References:
  - Gap risk is a key consideration for theta strategies; not from a single paper
    but well established in options literature (see Sinclair 2010, "Volatility Trading")
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import structlog

log = structlog.get_logger(__name__)


# ── Assumed Nifty overnight gap distribution parameters ─────────────────────
# [ASSUMED] Must be calibrated from historical NSE data before production use.
# Based on general equity index behavior; Nifty may differ.

_ASSUMED_GAP_MEAN_PCT   = 0.0    # Symmetric around zero [ASSUMED]
_ASSUMED_GAP_STD_PCT    = 0.50   # ~0.5% per day std [ASSUMED]
_ASSUMED_GAP_P95_PCT    = 1.50   # 95th percentile absolute gap [ASSUMED]
_ASSUMED_GAP_P99_PCT    = 3.00   # 99th percentile absolute gap [ASSUMED]
_ASSUMED_GAP_EXTREME_PCT = 5.00  # 99.9th percentile / tail [ASSUMED]


@dataclass(frozen=True)
class GapScenario:
    """Single overnight gap scenario."""
    name: str
    gap_pct: float        # % move from prev_close to open (e.g. -3.0 = -3%)
    iv_multiplier: float  # simultaneous IV change (gaps are usually accompanied by vol spike)
    probability: float    # [ASSUMED] approximate probability of this scenario


# Standard gap scenarios — calibrate probabilities from historical data
GAP_SCENARIOS: list[GapScenario] = [
    GapScenario("FLAT",        0.0,  1.00, 0.40),    # 40% of opens are flat
    GapScenario("UP_0.5PCT",  +0.5,  1.05, 0.15),
    GapScenario("DOWN_0.5PCT", -0.5, 1.10, 0.15),    # down gaps slightly more vol
    GapScenario("UP_1PCT",    +1.0,  1.10, 0.08),
    GapScenario("DOWN_1PCT",   -1.0, 1.20, 0.10),
    GapScenario("UP_2PCT",    +2.0,  1.15, 0.04),
    GapScenario("DOWN_2PCT",   -2.0, 1.40, 0.05),
    GapScenario("UP_3PCT",    +3.0,  1.20, 0.01),
    GapScenario("DOWN_3PCT",   -3.0, 1.60, 0.015),
    GapScenario("DOWN_5PCT",   -5.0, 2.00, 0.002),   # tail: budget/RBI surprise
    GapScenario("UP_5PCT",    +5.0,  1.30, 0.001),
]


@dataclass
class GapPnL:
    """P&L estimate for portfolio under one gap scenario."""
    scenario: GapScenario
    portfolio_pnl: float      # INR
    delta_pnl: float          # from gap move
    vega_pnl: float           # from IV change
    gap_through_stop: bool    # True if gap would breach stop-loss
    notes: str = ""


@dataclass
class GapRiskReport:
    """
    Full overnight gap risk report.

    The expected P&L = Σ(probability × scenario_pnl).
    Weighted by [ASSUMED] probabilities — calibrate before production.
    """
    current_price: float
    prev_close: float
    current_iv: float
    scenarios: list[GapPnL]
    WARNING: str = (
        "[ASSUMED] Gap probabilities are assumed starting values. "
        "Calibrate from historical NSE data before production use."
    )

    def expected_pnl(self) -> float:
        """Probability-weighted expected P&L across all scenarios."""
        total_prob = sum(g.scenario.probability for g in self.scenarios)
        if total_prob == 0:
            return 0.0
        return sum(
            g.portfolio_pnl * g.scenario.probability
            for g in self.scenarios
        ) / total_prob

    def worst_gap_pnl(self) -> float:
        """Worst-case gap P&L (minimum across all scenarios)."""
        if not self.scenarios:
            return 0.0
        return min(g.portfolio_pnl for g in self.scenarios)

    def var_95_pct(self, capital: float) -> float:
        """
        Approximate Value-at-Risk at 95% confidence level.
        Uses scenarios with cumulative probability ≤ 5% tail.
        Returns VaR as % of capital (positive = loss).
        """
        sorted_by_pnl = sorted(self.scenarios, key=lambda x: x.portfolio_pnl)
        cumulative_prob = 0.0
        for s in sorted_by_pnl:
            cumulative_prob += s.scenario.probability
            if cumulative_prob >= 0.05:
                return abs(s.portfolio_pnl) / capital * 100.0
        return 0.0

    def print_summary(self, capital: float = 100_000.0) -> None:
        print(f"\n{'='*55}")
        print(f"Overnight Gap Risk | Nifty prev_close={self.prev_close:.0f}, "
              f"ATM-IV={self.current_iv:.1%}")
        print(f"{'='*55}")
        print(f"{'Scenario':<18} {'Gap%':>6} {'P&L (₹)':>10} {'P&L%':>7} {'Stop?':>6}")
        print(f"{'-'*55}")
        for g in sorted(self.scenarios, key=lambda x: x.portfolio_pnl):
            pct = g.portfolio_pnl / capital * 100 if capital else 0
            stop = "YES" if g.gap_through_stop else "—"
            print(f"{g.scenario.name:<18} {g.scenario.gap_pct:>5.1f}% "
                  f"{g.portfolio_pnl:>10,.0f} {pct:>6.1f}% {stop:>6}")
        print(f"Expected P&L:   ₹{self.expected_pnl():>10,.0f}")
        print(f"Worst-case:     ₹{self.worst_gap_pnl():>10,.0f}")
        print(f"VaR(95%):       {self.var_95_pct(capital):.1f}% of capital")
        print(f"\n⚠ {self.WARNING}")


class GapRiskModel:
    """
    Models overnight gap risk for an NSE F&O portfolio.

    Uses second-order Taylor expansion (same as StressEngine) but
    applies gap-specific IV co-movement.

    Parameters
    ----------
    scenarios : list[GapScenario] | None
        Custom scenarios. Defaults to GAP_SCENARIOS.
    stop_loss_pct : float
        Portfolio stop-loss level as % of capital (e.g. 0.02 = 2%).
        Used to flag gap-through-stop scenarios.
    """

    def __init__(
        self,
        scenarios: list[GapScenario] | None = None,
        stop_loss_pct: float = 0.02,
    ) -> None:
        self.scenarios = scenarios or GAP_SCENARIOS
        self.stop_loss_pct = stop_loss_pct

    def compute(
        self,
        positions: list[dict],
        prev_close: float,
        current_price: float,
        current_atm_iv: float,
        capital: float = 100_000.0,
    ) -> GapRiskReport:
        """
        Compute overnight gap risk for current portfolio.

        Parameters
        ----------
        positions : list[dict]
            Same format as StressEngine.run() positions.
        prev_close : float
            Yesterday's NSE close price.
        current_price : float
            Current price (may differ from prev_close intraday).
        current_atm_iv : float
            Current ATM implied volatility (annualized).
        capital : float
            Total capital for % P&L calculations.
        """
        scenario_pnls: list[GapPnL] = []

        for scenario in self.scenarios:
            gap_price = prev_close * (1.0 + scenario.gap_pct / 100.0)
            delta_s = gap_price - current_price
            delta_sigma = current_atm_iv * (scenario.iv_multiplier - 1.0)

            total_delta = 0.0
            total_vega = 0.0

            for pos in positions:
                lots = pos.get("lots", 0)
                lot_size = pos.get("lot_size", 1)
                direction = pos.get("direction", 1)
                delta = pos.get("delta", 0.0)
                gamma = pos.get("gamma", 0.0)
                vega = pos.get("vega", 0.0)
                units = lots * lot_size * direction

                dp = (delta * units * delta_s
                      + 0.5 * gamma * units * (delta_s ** 2))
                vp = vega * lots * direction * (delta_sigma * 100)
                total_delta += dp
                total_vega += vp

            portfolio_pnl = total_delta + total_vega
            stop_loss_inr = capital * self.stop_loss_pct
            gap_through_stop = portfolio_pnl < -stop_loss_inr

            scenario_pnls.append(GapPnL(
                scenario=scenario,
                portfolio_pnl=portfolio_pnl,
                delta_pnl=total_delta,
                vega_pnl=total_vega,
                gap_through_stop=gap_through_stop,
                notes=f"gap_price={gap_price:.0f}, IV×{scenario.iv_multiplier}",
            ))

        report = GapRiskReport(
            current_price=current_price,
            prev_close=prev_close,
            current_iv=current_atm_iv,
            scenarios=scenario_pnls,
        )

        log.info(
            "gap_risk.computed",
            worst_pnl=f"₹{report.worst_gap_pnl():,.0f}",
            expected_pnl=f"₹{report.expected_pnl():,.0f}",
            var95=f"{report.var_95_pct(capital):.1f}%",
        )
        return report

    @staticmethod
    def calibrate_gap_std_from_history(
        daily_gaps_pct: Sequence[float],
    ) -> dict:
        """
        Compute empirical gap statistics from historical data.

        Parameters
        ----------
        daily_gaps_pct : sequence of float
            Historical overnight gaps as %, e.g. [0.1, -0.5, 1.2, ...]

        Returns
        -------
        dict with keys: mean, std, p95, p99, p999
        """
        if not daily_gaps_pct:
            return {}

        import statistics

        gaps = sorted(daily_gaps_pct)
        n = len(gaps)
        mean = statistics.mean(gaps)
        std = statistics.stdev(gaps) if n > 1 else 0.0
        abs_gaps = sorted(abs(g) for g in gaps)

        def percentile(data: list, pct: float) -> float:
            idx = pct / 100.0 * (len(data) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(data) - 1)
            frac = idx - lo
            return data[lo] * (1 - frac) + data[hi] * frac

        return {
            "mean_pct":    mean,
            "std_pct":     std,
            "p95_abs_pct": percentile(abs_gaps, 95),
            "p99_abs_pct": percentile(abs_gaps, 99),
            "p999_abs_pct": percentile(abs_gaps, 99.9),
            "n_observations": n,
            "WARNING": "[EMPIRICAL] calibrated from actual NSE data",
        }
