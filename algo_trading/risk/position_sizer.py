"""
position_sizer.py — Position sizing for NSE intraday strategies.

Three sizing methods, all enforcing the same hard caps:

  1. ATR-based (primary, matches CLAUDE.md formula):
       position_size = (capital × risk_pct) / (ATR14 × atr_multiplier)
     risk_pct defaults to 1% of capital; atr_multiplier defaults to 1.5.

  2. Half-Kelly:
       f* = (p × b − q) / b   where p=win_rate, b=avg_win/avg_loss, q=1−p
       half_kelly_size = f* × 0.5 × capital / price
     Kelly fraction is clipped to [0, 1] before halving to prevent
     negative sizing on edge cases (b < 1 or p < 0.5).

  3. Target-volatility:
       target_vol_position = (capital × target_annual_vol) / (realized_vol × price × sqrt(252))
     realized_vol is the daily standard deviation of returns (dimensionless).
     target_annual_vol defaults to 15% (CLAUDE.md spec).

Hard caps (enforced by ALL methods via _apply_hard_caps()):
  - Max single position value: max_position_pct (default 5%) of capital.
    Rationale: if a stock locks in a down circuit you hold that position
    to EOD with no exit — the 5% cap limits worst-case overnight gap risk.
  - No fractional shares (floor division).
  - Minimum 1 share.

All monetary arithmetic uses Decimal. float is used only for intermediate
volatility / Kelly ratio calculations where IEEE-754 precision is acceptable
and Decimal would add unnecessary overhead with no correctness benefit.

Config consumed:
  settings.capital_inr            — total capital
  settings.max_position_pct       — 5% single-position cap
  settings.atr_stop_multiplier    — 1.5 (atr_multiplier for ATR sizing)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

import structlog

from algo_trading.config import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — all sourced from config or method-level parameters.
# These module-level names exist ONLY as defaults; every public method accepts
# the relevant config-driven values via __init__ parameters.
# ---------------------------------------------------------------------------
_SQRT_252: float = math.sqrt(252)
_D_ZERO:   Decimal = Decimal("0")
_D_ONE:    Decimal = Decimal("1")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SizingResult:
    """Output from any PositionSizer method.

    Attributes:
        shares: Final share count after all caps (≥ 1 if a valid size exists,
                0 if the sizing formula produced zero).
        position_value: shares × price as Decimal (INR).
        risk_per_trade: Capital at risk = stop_distance × shares.
                        ATR method: (ATR14 × atr_multiplier) × shares.
                        Kelly / target-vol: position_value (full position is
                        at risk since stop distance is unknown in those modes).
        size_method: One of "atr", "kelly", "target_vol".
        capped: True if any hard cap was applied.
        cap_reason: "circuit_5pct" | "max_position" | "" (empty if uncapped).
    """
    shares:         int
    position_value: Decimal
    risk_per_trade: Decimal
    size_method:    str
    capped:         bool
    cap_reason:     str


# ---------------------------------------------------------------------------
# Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """Position sizer with three methods and hard caps.

    Instantiate once per strategy run (or once per day after reset).
    All methods are stateless after construction — safe for concurrent calls
    from different threads (reads only, no mutation after __init__).

    Args:
        capital: Total available capital in INR.
        max_position_pct: Maximum fraction of capital in a single position.
            Default 0.05 (5%). Used as circuit-breaker cap.
        atr_risk_pct: Fraction of capital risked per ATR-based trade.
            Default 0.01 (1%). Maps to risk_pct in the CLAUDE.md formula.
        atr_multiplier: Multiplier applied to ATR14 for the stop distance.
            Default 1.5. Sourced from settings.atr_stop_multiplier.
        target_annual_vol: Annualised target volatility for target-vol sizing.
            Default 0.15 (15%).
    """

    def __init__(
        self,
        capital: Decimal,
        max_position_pct: float = 0.05,
        atr_risk_pct: float = 0.01,
        atr_multiplier: float = 1.5,
        target_annual_vol: float = 0.15,
    ) -> None:
        if capital <= _D_ZERO:
            raise ValueError(f"capital must be positive, got {capital}")
        if not (0.0 < max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0, 1], got {max_position_pct}")
        if not (0.0 < atr_risk_pct <= 1.0):
            raise ValueError(f"atr_risk_pct must be in (0, 1], got {atr_risk_pct}")
        if atr_multiplier <= 0.0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")
        if not (0.0 < target_annual_vol <= 2.0):
            raise ValueError(f"target_annual_vol must be in (0, 2], got {target_annual_vol}")

        self._capital:          Decimal = capital
        self._max_pos_pct:      Decimal = Decimal(str(max_position_pct))
        self._atr_risk_pct:     Decimal = Decimal(str(atr_risk_pct))
        self._atr_multiplier:   Decimal = Decimal(str(atr_multiplier))
        self._target_annual_vol: float  = target_annual_vol

        # Pre-compute the max position value cap (used by every method)
        self._max_position_value: Decimal = (
            self._capital * self._max_pos_pct
        ).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        log.debug(
            "position_sizer.init",
            capital=str(capital),
            max_position_pct=max_position_pct,
            atr_risk_pct=atr_risk_pct,
            atr_multiplier=atr_multiplier,
            target_annual_vol=target_annual_vol,
            max_position_value=str(self._max_position_value),
        )

    # -----------------------------------------------------------------------
    # Public sizing methods
    # -----------------------------------------------------------------------

    def size_atr(self, price: Decimal, atr: Decimal) -> SizingResult:
        """ATR-based position sizing (primary method).

        Formula (from CLAUDE.md):
            position_size = (capital × risk_pct) / (ATR14 × atr_multiplier)

        The result is the number of shares such that a 1×ATR adverse move
        costs exactly risk_pct of capital.

        Args:
            price: Current market price of the instrument (INR).
            atr: 14-period ATR in price units (INR).

        Returns:
            SizingResult with size_method="atr".
        """
        if price <= _D_ZERO:
            log.warning("position_sizer.atr.zero_price", price=str(price))
            return self._zero_result("atr", price)
        if atr <= _D_ZERO:
            log.warning("position_sizer.atr.zero_atr", atr=str(atr))
            return self._zero_result("atr", price)

        risk_amount:   Decimal = self._capital * self._atr_risk_pct
        stop_distance: Decimal = atr * self._atr_multiplier

        raw_shares: int = int(risk_amount / stop_distance)  # floor division

        shares, capped, cap_reason = self._apply_hard_caps(raw_shares, price)

        # Compute risk_per_trade using final (capped) share count
        final_risk: Decimal = stop_distance * Decimal(str(shares))
        position_value: Decimal = price * Decimal(str(shares))

        log.debug(
            "position_sizer.atr",
            capital=str(self._capital),
            price=str(price),
            atr=str(atr),
            stop_distance=str(stop_distance),
            raw_shares=raw_shares,
            final_shares=shares,
            capped=capped,
            cap_reason=cap_reason,
        )

        return SizingResult(
            shares=shares,
            position_value=position_value,
            risk_per_trade=final_risk,
            size_method="atr",
            capped=capped,
            cap_reason=cap_reason,
        )

    def size_kelly(
        self,
        price: Decimal,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> SizingResult:
        """Half-Kelly position sizing.

        Formula:
            f* = (p × b − q) / b   where p=win_rate, q=1−p, b=avg_win/avg_loss
            half_kelly_size = f* × 0.5 × capital / price

        The Kelly fraction is clamped to [0, 1] before halving. A negative
        Kelly (edge < 0) yields 0 shares (no trade).

        Args:
            price: Current market price (INR).
            win_rate: Historical win rate in [0, 1] (e.g. 0.55 for 55%).
            avg_win_loss_ratio: Average win / average loss ratio (e.g. 1.5
                means wins are 1.5× the average loss). Must be > 0.

        Returns:
            SizingResult with size_method="kelly".
        """
        if price <= _D_ZERO:
            log.warning("position_sizer.kelly.zero_price", price=str(price))
            return self._zero_result("kelly", price)
        if not (0.0 <= win_rate <= 1.0):
            raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
        if avg_win_loss_ratio <= 0.0:
            raise ValueError(
                f"avg_win_loss_ratio must be positive, got {avg_win_loss_ratio}"
            )

        p: float = win_rate
        q: float = 1.0 - p
        b: float = avg_win_loss_ratio

        # Kelly fraction
        kelly_f: float = (p * b - q) / b

        if kelly_f <= 0.0:
            log.info(
                "position_sizer.kelly.negative_edge",
                win_rate=win_rate,
                avg_win_loss_ratio=avg_win_loss_ratio,
                kelly_f=kelly_f,
            )
            return self._zero_result("kelly", price)

        # Clamp to [0, 1] (full Kelly), then halve
        half_kelly: float = min(kelly_f, 1.0) * 0.5

        capital_f: float  = float(self._capital)
        price_f:   float  = float(price)
        raw_shares: int   = int(half_kelly * capital_f / price_f)

        shares, capped, cap_reason = self._apply_hard_caps(raw_shares, price)
        position_value: Decimal = price * Decimal(str(shares))

        # risk_per_trade for Kelly = position_value (no explicit stop defined)
        risk_per_trade: Decimal = position_value

        log.debug(
            "position_sizer.kelly",
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
            kelly_f=round(kelly_f, 6),
            half_kelly=round(half_kelly, 6),
            raw_shares=raw_shares,
            final_shares=shares,
            capped=capped,
        )

        return SizingResult(
            shares=shares,
            position_value=position_value,
            risk_per_trade=risk_per_trade,
            size_method="kelly",
            capped=capped,
            cap_reason=cap_reason,
        )

    def size_target_vol(
        self,
        price: Decimal,
        realized_vol_daily: float,
    ) -> SizingResult:
        """Target-volatility position sizing.

        Formula:
            target_vol_position = (capital × target_annual_vol) /
                                  (realized_vol × price × sqrt(252))

        Annualises the daily realised volatility by multiplying by sqrt(252).

        Args:
            price: Current market price (INR).
            realized_vol_daily: Daily realised volatility of the instrument
                (standard deviation of daily log-returns, dimensionless).
                E.g. 0.015 for 1.5% daily vol.

        Returns:
            SizingResult with size_method="target_vol".
        """
        if price <= _D_ZERO:
            log.warning("position_sizer.target_vol.zero_price", price=str(price))
            return self._zero_result("target_vol", price)
        if realized_vol_daily <= 0.0:
            log.warning(
                "position_sizer.target_vol.zero_vol",
                realized_vol_daily=realized_vol_daily,
            )
            return self._zero_result("target_vol", price)

        capital_f:      float = float(self._capital)
        price_f:        float = float(price)
        annual_vol_f:   float = realized_vol_daily * _SQRT_252

        if annual_vol_f <= 0.0:
            return self._zero_result("target_vol", price)

        raw_float: float = (
            capital_f * self._target_annual_vol
        ) / (annual_vol_f * price_f)

        raw_shares: int = int(raw_float)  # floor

        shares, capped, cap_reason = self._apply_hard_caps(raw_shares, price)
        position_value: Decimal = price * Decimal(str(shares))
        risk_per_trade: Decimal = position_value

        log.debug(
            "position_sizer.target_vol",
            realized_vol_daily=realized_vol_daily,
            annual_vol=round(annual_vol_f, 6),
            target_annual_vol=self._target_annual_vol,
            raw_shares=raw_shares,
            final_shares=shares,
            capped=capped,
        )

        return SizingResult(
            shares=shares,
            position_value=position_value,
            risk_per_trade=risk_per_trade,
            size_method="target_vol",
            capped=capped,
            cap_reason=cap_reason,
        )

    def adjust_for_circuit_risk(
        self,
        shares: int,
        price: Decimal,
    ) -> tuple[int, bool]:
        """Enforce the 5% circuit-breaker cap on an already-sized position.

        Call this when you already have a share count from an external source
        (e.g., a signal router that sizes independently) and want to enforce
        the cap without going through a full sizing method.

        Args:
            shares: Proposed share count.
            price: Current market price (INR).

        Returns:
            Tuple of (adjusted_shares, was_capped). was_capped is True if the
            returned share count is less than the input.
        """
        adjusted, capped, _ = self._apply_hard_caps(shares, price)
        return adjusted, capped

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_hard_caps(
        self,
        raw_shares: int,
        price: Decimal,
    ) -> tuple[int, bool, str]:
        """Apply all hard caps and return (final_shares, capped, cap_reason).

        Caps applied in order:
          1. Circuit cap: position_value ≤ max_position_pct of capital.
          2. Minimum: 1 share (a zero raw_shares remains 0 — caller decides).

        Args:
            raw_shares: Uncapped share count from a sizing formula.
            price: Instrument price used to compute position value.

        Returns:
            Tuple (final_shares, capped, cap_reason).
        """
        if raw_shares <= 0:
            return 0, False, ""

        position_value: Decimal = price * Decimal(str(raw_shares))
        capped:    bool = False
        cap_reason: str = ""

        if position_value > self._max_position_value:
            # Floor-divide to find max shares within the cap
            if price <= _D_ZERO:
                raw_shares = 0
            else:
                raw_shares = int(
                    self._max_position_value / price
                )  # floor via int()
            capped     = True
            cap_reason = "circuit_5pct"
            log.debug(
                "position_sizer.cap_applied",
                cap=cap_reason,
                original_value=str(position_value),
                max_value=str(self._max_position_value),
                capped_shares=raw_shares,
            )

        # Ensure minimum 1 share (only when raw sizing produced ≥ 1 before capping)
        if capped and raw_shares < 1:
            # Cap drove shares to zero — respect the cap (no trade)
            return 0, True, cap_reason

        final_shares: int = max(raw_shares, 1) if raw_shares > 0 else 0
        return final_shares, capped, cap_reason

    def _zero_result(self, method: str, price: Decimal) -> SizingResult:
        """Return a zero-size SizingResult (used on invalid inputs).

        Args:
            method: The sizing method name ("atr", "kelly", "target_vol").
            price: Price passed to the sizing method (unused in result).

        Returns:
            SizingResult with shares=0 and all Decimal fields at zero.
        """
        return SizingResult(
            shares=0,
            position_value=_D_ZERO,
            risk_per_trade=_D_ZERO,
            size_method=method,
            capped=False,
            cap_reason="",
        )

    # -----------------------------------------------------------------------
    # Convenience: build from config singleton
    # -----------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "PositionSizer":
        """Construct a PositionSizer using the global config singleton.

        All parameters are sourced from ``algo_trading.config.settings``.

        Returns:
            A fully configured PositionSizer instance.
        """
        return cls(
            capital=settings.capital_inr,
            max_position_pct=settings.max_position_pct,
            atr_risk_pct=0.01,                       # 1% per CLAUDE.md spec
            atr_multiplier=settings.atr_stop_multiplier,
            target_annual_vol=0.15,                  # 15% annualised target
        )
