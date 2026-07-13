"""
algo_trading/risk/stress_engine.py — Portfolio scenario stress testing.

Applies instantaneous shocks to underlying price and implied volatility,
then recomputes portfolio P&L using the BSM Greeks.

Scenarios (from FO_SYSTEM_DESIGN.md Section 9.5):
  Nifty: ±3%, ±5%, ±10%
  VIX:   ×2, ×3 (multiplier on current IV surface)
  Extreme: Nifty ±15%, VIX ×4 (tail scenario)

P&L approximation uses second-order Taylor expansion:
  ΔP ≈ Δ·ΔS + ½·Γ·(ΔS)² + Vega·Δσ + Θ·Δt

where:
  Δ    = portfolio delta (₹/₹ underlying)
  Γ    = portfolio gamma (₹²/₹ underlying²)
  Vega = portfolio vega (₹/vol point)
  Δt   = 0 (instantaneous — no theta decay in stress scenario)

This is an APPROXIMATION. For large moves the Taylor expansion breaks down.
Use as a risk indicator, not as a precise P&L forecast.

Normalized risk units (FO_SYSTEM_DESIGN.md Section 9.5):
  delta_risk_inr  = portfolio_delta × underlying_price  (₹ P&L per 1% move)
  gamma_pnl_1pct  = 0.5 × portfolio_gamma × (underlying × 0.01)²
  vega_inr        = portfolio_vega × 1.0  (₹ per 1 vol point = 0.01 IV change)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StressScenario:
    """
    A single stress scenario specification.

    Attributes
    ----------
    name : str
        Human-readable name for reporting.
    underlying_shock_pct : float
        Instantaneous % move in underlying (e.g. -0.05 = -5%).
    vol_multiplier : float
        Multiplier on current implied volatility (e.g. 2.0 = double IV).
    description : str
        One-line description for audit reports.
    """
    name: str
    underlying_shock_pct: float   # e.g. -0.05 = -5%
    vol_multiplier: float         # e.g. 2.0 = IV × 2
    description: str


# ── Standard NSE F&O Stress Scenarios (FO_SYSTEM_DESIGN.md Section 9.5) ─────

SCENARIOS: list[StressScenario] = [
    # Nifty directional shocks
    StressScenario("UP_3PCT",    +0.03, 1.0,  "Nifty +3%, IV unchanged"),
    StressScenario("DOWN_3PCT",  -0.03, 1.0,  "Nifty -3%, IV unchanged"),
    StressScenario("UP_5PCT",    +0.05, 1.0,  "Nifty +5%, IV unchanged"),
    StressScenario("DOWN_5PCT",  -0.05, 1.0,  "Nifty -5%, IV unchanged"),
    StressScenario("UP_10PCT",   +0.10, 1.0,  "Nifty +10%, IV unchanged"),
    StressScenario("DOWN_10PCT", -0.10, 1.0,  "Nifty -10%, IV unchanged"),

    # VIX expansion only (no directional move — overnight vol spike)
    StressScenario("VIX_2X",  0.0, 2.0, "IV × 2, Nifty unchanged"),
    StressScenario("VIX_3X",  0.0, 3.0, "IV × 3, Nifty unchanged"),

    # Combined adverse scenarios
    StressScenario("DOWN_5PCT_VIX_2X", -0.05, 2.0, "Nifty -5%, IV × 2 (high-VIX crash)"),
    StressScenario("DOWN_10PCT_VIX_3X", -0.10, 3.0, "Nifty -10%, IV × 3 (tail event)"),
    StressScenario("UP_5PCT_VIX_2X",   +0.05, 2.0, "Nifty +5%, IV × 2 (upside gap with uncertainty)"),

    # Tail scenarios (extreme)
    StressScenario("DOWN_15PCT_VIX_4X", -0.15, 4.0, "Nifty -15%, IV × 4 (circuit breaker event)"),
    StressScenario("UP_15PCT",          +0.15, 1.0,  "Nifty +15% (extreme short squeeze)"),
]


@dataclass
class ScenarioPnL:
    """P&L estimate for one position under one stress scenario."""
    position_id: str
    scenario: StressScenario
    delta_pnl: float       # Δ·ΔS contribution
    gamma_pnl: float       # ½·Γ·(ΔS)² contribution
    vega_pnl: float        # Vega·Δσ contribution
    total_pnl: float       # sum of above
    notes: str = ""


@dataclass
class StressResult:
    """
    Full stress test result for the portfolio under one scenario.

    Attributes
    ----------
    scenario : StressScenario
        The scenario that was applied.
    portfolio_pnl : float
        Total estimated portfolio P&L in INR.
    delta_contribution : float
        Delta P&L component in INR.
    gamma_contribution : float
        Gamma P&L component in INR.
    vega_contribution : float
        Vega P&L component in INR.
    position_pnls : list[ScenarioPnL]
        Per-position breakdown.
    vix_trigger_breached : bool
        True if scenario triggers a VIX risk rule (Section 9.5):
          VIX 22–25: reduce new position size
          VIX 25–30: no new positions
          VIX > 30 : emergency flatten all short-vega positions
    WARNING : str
        Always set — this is an APPROXIMATION, not official risk measure.
    """
    scenario: StressScenario
    portfolio_pnl: float
    delta_contribution: float
    gamma_contribution: float
    vega_contribution: float
    position_pnls: list[ScenarioPnL] = field(default_factory=list)
    vix_trigger_breached: bool = False
    WARNING: str = "APPROX-TAYLOR-EXPANSION: large moves not modeled accurately"

    def pnl_as_pct_of_capital(self, capital: float) -> float:
        """Portfolio P&L as % of capital."""
        if capital == 0:
            return 0.0
        return self.portfolio_pnl / capital * 100.0


@dataclass
class StressReport:
    """Aggregated stress report across all scenarios."""
    underlying_price: float
    current_iv: float            # current ATM IV (annualized)
    scenarios: list[StressResult]

    def worst_case(self) -> StressResult | None:
        """Return the scenario with the worst (most negative) P&L."""
        if not self.scenarios:
            return None
        return min(self.scenarios, key=lambda s: s.portfolio_pnl)

    def worst_pnl_inr(self) -> float:
        worst = self.worst_case()
        return worst.portfolio_pnl if worst else 0.0

    def print_summary(self, capital: float = 100_000.0) -> None:
        print(f"\n{'='*65}")
        print(f"Stress Test Report | Nifty={self.underlying_price:.0f}, ATM-IV={self.current_iv:.1%}")
        print(f"{'='*65}")
        print(f"{'Scenario':<28} {'P&L (₹)':>10} {'P&L %':>8} {'Delta':>10} {'Gamma':>10} {'Vega':>10}")
        print(f"{'-'*65}")
        for r in sorted(self.scenarios, key=lambda x: x.portfolio_pnl):
            pct = r.pnl_as_pct_of_capital(capital)
            print(f"{r.scenario.name:<28} {r.portfolio_pnl:>10,.0f} {pct:>7.1f}%"
                  f" {r.delta_contribution:>10,.0f} {r.gamma_contribution:>10,.0f}"
                  f" {r.vega_contribution:>10,.0f}")
        print(f"{'='*65}")
        worst = self.worst_case()
        if worst:
            print(f"Worst case: {worst.scenario.name} → ₹{worst.portfolio_pnl:,.0f}"
                  f" ({worst.pnl_as_pct_of_capital(capital):.1f}% of capital)")
        print(f"⚠ {self.scenarios[0].WARNING if self.scenarios else 'No scenarios'}")


class StressEngine:
    """
    Scenario stress testing engine for NSE F&O portfolios.

    Computes P&L under instantaneous shocks using second-order Taylor
    expansion of option price w.r.t. underlying and volatility.

    This is an APPROXIMATION — not an official stress test or regulatory
    capital computation.

    Parameters
    ----------
    scenarios : list[StressScenario] | None
        Scenarios to run. Defaults to the standard NSE F&O scenario set.
    """

    def __init__(self, scenarios: list[StressScenario] | None = None) -> None:
        self.scenarios = scenarios or SCENARIOS

    def run(
        self,
        positions: list[dict],
        underlying_price: float,
        current_atm_iv: float,
    ) -> StressReport:
        """
        Run all scenarios against current portfolio.

        Parameters
        ----------
        positions : list[dict]
            Each position must have keys:
            {
              "position_id": str,
              "lots": int,
              "lot_size": int,
              "direction": int,     # +1 long, -1 short
              "delta": float,       # per unit
              "gamma": float,       # per unit
              "vega": float,        # per unit (₹ per 1% vol change)
            }
        underlying_price : float
            Current Nifty / BankNifty index level.
        current_atm_iv : float
            Current ATM implied volatility (annualized, e.g. 0.15 = 15%).

        Returns
        -------
        StressReport
        """
        results: list[StressResult] = []

        for scenario in self.scenarios:
            delta_s = underlying_price * scenario.underlying_shock_pct
            delta_sigma = current_atm_iv * (scenario.vol_multiplier - 1.0)

            total_delta_pnl = 0.0
            total_gamma_pnl = 0.0
            total_vega_pnl = 0.0
            position_pnls: list[ScenarioPnL] = []

            for pos in positions:
                lots = pos.get("lots", 0)
                lot_size = pos.get("lot_size", 1)
                direction = pos.get("direction", 1)
                delta = pos.get("delta", 0.0)    # per unit
                gamma = pos.get("gamma", 0.0)    # per unit
                vega = pos.get("vega", 0.0)      # ₹ per 1% vol change (per lot)
                units = lots * lot_size * direction

                dp = delta * units * delta_s
                gp = 0.5 * gamma * units * (delta_s ** 2)
                vp = vega * lots * direction * (delta_sigma * 100)  # vega × Δ(vol%)

                total_delta_pnl += dp
                total_gamma_pnl += gp
                total_vega_pnl += vp

                position_pnls.append(ScenarioPnL(
                    position_id=pos.get("position_id", "?"),
                    scenario=scenario,
                    delta_pnl=dp,
                    gamma_pnl=gp,
                    vega_pnl=vp,
                    total_pnl=dp + gp + vp,
                ))

            portfolio_pnl = total_delta_pnl + total_gamma_pnl + total_vega_pnl

            # VIX trigger check (FO_SYSTEM_DESIGN.md Section 9.5)
            implied_vix = current_atm_iv * 100 * scenario.vol_multiplier
            vix_breach = implied_vix > 22

            results.append(StressResult(
                scenario=scenario,
                portfolio_pnl=portfolio_pnl,
                delta_contribution=total_delta_pnl,
                gamma_contribution=total_gamma_pnl,
                vega_contribution=total_vega_pnl,
                position_pnls=position_pnls,
                vix_trigger_breached=vix_breach,
            ))

            log.debug(
                "stress.scenario",
                scenario=scenario.name,
                pnl=f"₹{portfolio_pnl:,.0f}",
                delta_s=f"{delta_s:.0f}",
                delta_sigma=f"{delta_sigma:.3f}",
            )

        return StressReport(
            underlying_price=underlying_price,
            current_iv=current_atm_iv,
            scenarios=results,
        )

    def vix_action_required(
        self, current_vix: float
    ) -> tuple[str, str]:
        """
        Return (action, description) based on current VIX level.

        Rules from FO_SYSTEM_DESIGN.md Section 9.5:
          VIX 22–25: reduce new position size by 50%
          VIX 25–30: no new positions, manage existing to expiry or stop
          VIX > 30  : emergency flatten all short-vega positions

        Parameters
        ----------
        current_vix : float
            India VIX value (e.g. 14.5, not 0.145).

        Returns
        -------
        (action_code, description)
        """
        if current_vix > 30.0:
            return ("EMERGENCY_FLATTEN", f"India VIX={current_vix:.1f} > 30 — flatten all short-vega")
        elif current_vix > 25.0:
            return ("NO_NEW_POSITIONS", f"India VIX={current_vix:.1f} > 25 — no new entries")
        elif current_vix > 22.0:
            return ("REDUCE_SIZE_50PCT", f"India VIX={current_vix:.1f} > 22 — halve new position sizes")
        else:
            return ("NORMAL", f"India VIX={current_vix:.1f} — normal regime")
