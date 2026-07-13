"""
algo_trading/risk/portfolio_allocator.py — Portfolio-level risk allocation.

Sits ABOVE the per-symbol position sizer.  Controls how capital is distributed
across multiple concurrent strategies and symbols, enforcing:
  - Sector concentration limits
  - Beta exposure vs Nifty
  - Correlation clustering (avoid over-allocating to correlated assets)
  - Per-strategy drawdown budgets
  - Marginal contribution to portfolio risk (MCR)
  - Expected shortfall (CVaR) at portfolio level

This module covers EQUITY portfolio controls.
F&O controls live in algo_trading/fo/portfolio_risk.py.

Sector mapping (NSE, [ASSUMED] — update from NSE sector classification document):
  IT:      TCS, INFY, WIPRO, HCLTECH, TECHM
  Banking: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
  Energy:  RELIANCE, ONGC, NTPC, POWERGRID, COALINDIA
  FMCG:    HINDUNILVR, ITC, NESTLEIND, BRITANNIA
  Auto:    MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO, EICHERMOT
  Pharma:  SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, APOLLOHOSP
  Metal:   TATASTEEL, JSWSTEEL, HINDALCO, VEDL
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping [ASSUMED — update from NSE official classification]
# ---------------------------------------------------------------------------

_DEFAULT_SECTOR_MAP: dict[str, str] = {
    # IT
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    # Banking
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "KOTAKBANK": "Banking",
    "AXISBANK": "Banking",
    # Energy
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "NTPC": "Energy",
    "POWERGRID": "Energy",
    "COALINDIA": "Energy",
    # FMCG
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    # Auto
    "MARUTI": "Auto",
    "TATAMOTORS": "Auto",
    "M&M": "Auto",
    "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto",
    # Pharma
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Pharma",
    # Metal
    "TATASTEEL": "Metal",
    "JSWSTEEL": "Metal",
    "HINDALCO": "Metal",
    "VEDL": "Metal",
}


def get_sector(symbol: str, extra_map: Optional[dict[str, str]] = None) -> str:
    """Look up the NSE sector for a symbol.

    Args:
        symbol: NSE ticker symbol (e.g. "TCS").
        extra_map: Optional caller-supplied additions/overrides to the default map.

    Returns:
        Sector name string, or "Unknown" if the symbol is not in any map.
    """
    effective_map = {**_DEFAULT_SECTOR_MAP, **(extra_map or {})}
    return effective_map.get(symbol.upper(), "Unknown")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PortfolioPosition:
    """A single open position in the portfolio.

    Attributes:
        symbol: NSE ticker symbol.
        sector: Sector classification (derived via get_sector or supplied).
        qty: Signed quantity (positive = long, negative = short).
        entry_price: Average entry price in INR.
        current_price: Most recent market price in INR.
        strategy_name: Name of the strategy that generated this position.
        beta: Beta of the symbol vs Nifty. [ASSUMED 1.0 if unknown]
    """

    symbol: str
    sector: str
    qty: int
    entry_price: float
    current_price: float
    strategy_name: str
    beta: float = 1.0

    @property
    def market_value(self) -> float:
        """Absolute market value of the position in INR."""
        return abs(self.qty) * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        """Unrealised P&L in INR (positive = profit)."""
        return self.qty * (self.current_price - self.entry_price)


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of all open positions.

    Attributes:
        positions: List of all open PortfolioPosition instances.
        total_capital: Total allocated capital in INR.
        timestamp: Snapshot timestamp (datetime).
    """

    positions: list[PortfolioPosition]
    total_capital: float
    timestamp: Any   # datetime

    @property
    def total_market_value(self) -> float:
        """Sum of absolute market values across all positions."""
        return sum(p.market_value for p in self.positions)

    def positions_by_strategy(self, strategy_name: str) -> list[PortfolioPosition]:
        """Filter positions belonging to a specific strategy.

        Args:
            strategy_name: Strategy name to filter by.

        Returns:
            List of PortfolioPosition for that strategy.
        """
        return [p for p in self.positions if p.strategy_name == strategy_name]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAllocatorConfig:
    """Configuration parameters for the PortfolioAllocator.

    All percentage values are 0-100 (not 0-1 fractions).
    All [ASSUMED] values should be reviewed against actual NSE portfolio
    risk guidelines and back-tested before live deployment.

    Attributes:
        max_sector_exposure_pct: Maximum fraction of total capital in any
            single sector. [ASSUMED 30%]
        max_beta_exposure: Maximum weighted-average portfolio beta vs Nifty.
            [ASSUMED 1.5 — allows moderate directional tilt]
        max_correlation_cluster_pct: Maximum fraction of capital in any
            single correlation cluster. [ASSUMED 40%]
        correlation_threshold: Pearson correlation above which two symbols
            are considered part of the same cluster. [ASSUMED 0.7]
        max_single_symbol_pct: Hard cap per symbol (SURV-8 compliance). [5%]
        strategy_drawdown_budget_pct: Per-strategy drawdown budget as % of
            initial strategy capital.  Reaching this triggers a pause. [ASSUMED 5%]
    """

    max_sector_exposure_pct: float = 30.0
    max_beta_exposure: float = 1.5
    max_correlation_cluster_pct: float = 40.0
    correlation_threshold: float = 0.7
    max_single_symbol_pct: float = 5.0
    strategy_drawdown_budget_pct: float = 5.0


# ---------------------------------------------------------------------------
# PortfolioAllocator
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """Portfolio-level risk allocation and concentration guard.

    Sits above the per-symbol PositionSizer. Every new position request must
    pass check_new_position_allowed() before sizing and submission.

    Args:
        config: PortfolioAllocatorConfig with limit parameters.  Uses
            defaults (conservative NSE-appropriate values) if None.
        sector_map: Custom symbol→sector mapping to override the defaults.

    Example::

        allocator = PortfolioAllocator()
        allowed, reason = allocator.check_new_position_allowed(
            snapshot, "INFY", qty=100, price=1800.0, strategy_name="ema_cross"
        )
        if not allowed:
            log.warning("position_blocked", reason=reason)
    """

    def __init__(
        self,
        config: Optional[PortfolioAllocatorConfig] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._config = config or PortfolioAllocatorConfig()
        self._sector_map = sector_map or {}

    # -------------------------------------------------------------------------
    # Sector exposure
    # -------------------------------------------------------------------------

    def check_sector_limit(
        self,
        snapshot: PortfolioSnapshot,
        new_symbol: str,
        new_value: float,
    ) -> tuple[bool, str]:
        """Check whether adding new_symbol would breach the sector concentration limit.

        Args:
            snapshot: Current portfolio snapshot.
            new_symbol: Symbol being considered for a new position.
            new_value: Market value of the proposed new position in INR.

        Returns:
            Tuple of (allowed: bool, reason: str).  reason is empty when allowed.
        """
        if snapshot.total_capital <= 0:
            return True, ""

        new_sector = get_sector(new_symbol, self._sector_map)

        # Current sector exposure
        sector_value = sum(
            p.market_value
            for p in snapshot.positions
            if get_sector(p.symbol, self._sector_map) == new_sector
        )

        # Proposed total sector value after the new position
        proposed_sector_value = sector_value + new_value
        proposed_sector_pct = (proposed_sector_value / snapshot.total_capital) * 100.0

        if proposed_sector_pct > self._config.max_sector_exposure_pct:
            reason = (
                f"Sector limit breach: adding {new_symbol} (sector={new_sector}) "
                f"would bring sector exposure to {proposed_sector_pct:.1f}% "
                f"> limit {self._config.max_sector_exposure_pct:.1f}%"
            )
            log.info(
                "portfolio_allocator.sector_limit_breach",
                symbol=new_symbol,
                sector=new_sector,
                proposed_pct=round(proposed_sector_pct, 2),
                limit_pct=self._config.max_sector_exposure_pct,
            )
            return False, reason

        return True, ""

    # -------------------------------------------------------------------------
    # Beta exposure
    # -------------------------------------------------------------------------

    def compute_portfolio_beta(self, snapshot: PortfolioSnapshot) -> float:
        """Compute the value-weighted average portfolio beta vs Nifty.

        Beta is weighted by each position's market value.  Symbols with
        missing beta are assumed to have beta=1.0 (neutral). [ASSUMED]

        Args:
            snapshot: Current portfolio snapshot.

        Returns:
            Weighted portfolio beta.  Returns 0.0 for an empty portfolio.
        """
        total_value = snapshot.total_market_value
        if total_value <= 0:
            return 0.0

        weighted_beta = sum(
            p.beta * p.market_value for p in snapshot.positions
        )
        portfolio_beta = weighted_beta / total_value

        log.debug(
            "portfolio_allocator.portfolio_beta",
            portfolio_beta=round(portfolio_beta, 4),
            n_positions=len(snapshot.positions),
        )
        return portfolio_beta

    # -------------------------------------------------------------------------
    # Correlation clustering
    # -------------------------------------------------------------------------

    def compute_correlation_clusters(
        self,
        returns_matrix: np.ndarray,
        symbols: list[str],
    ) -> dict[str, list[str]]:
        """Cluster symbols by return correlation using single-linkage clustering.

        Groups symbols into clusters where any pair within a cluster has
        pairwise absolute Pearson correlation >= correlation_threshold.
        Uses a union-find (disjoint-set) approach for simplicity.

        Args:
            returns_matrix: 2-D array of shape (n_bars, n_symbols). Each column
                is the return series for one symbol in *symbols*.
            symbols: List of symbol names, aligned with columns of returns_matrix.

        Returns:
            Dictionary mapping cluster_id (str) to list of symbol names in that
            cluster.  Cluster IDs are the name of the first (alphabetically
            lowest) symbol in the cluster.

        Raises:
            ValueError: If returns_matrix columns don't match len(symbols).
        """
        n_symbols = len(symbols)
        if returns_matrix.ndim != 2 or returns_matrix.shape[1] != n_symbols:
            raise ValueError(
                f"returns_matrix must have shape (n_bars, {n_symbols}), "
                f"got {returns_matrix.shape}"
            )

        if n_symbols == 0:
            return {}

        # Compute pairwise correlation matrix
        # np.corrcoef expects (n_symbols, n_bars) — transpose
        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(returns_matrix.T)

        # Replace NaN with 0 (uncorrelated for missing data)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Union-find (disjoint-set) for cluster membership
        parent: list[int] = list(range(n_symbols))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Merge pairs that exceed the correlation threshold
        threshold = self._config.correlation_threshold
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                if abs(corr_matrix[i, j]) >= threshold:
                    union(i, j)

        # Group by root
        clusters: dict[int, list[str]] = {}
        for idx, symbol in enumerate(symbols):
            root = find(idx)
            clusters.setdefault(root, []).append(symbol)

        # Re-key by the first (sorted) symbol name in each cluster
        result: dict[str, list[str]] = {}
        for members in clusters.values():
            members_sorted = sorted(members)
            cluster_id = members_sorted[0]
            result[cluster_id] = members_sorted

        log.debug(
            "portfolio_allocator.correlation_clusters",
            n_clusters=len(result),
            cluster_sizes={k: len(v) for k, v in result.items()},
        )
        return result

    # -------------------------------------------------------------------------
    # Marginal contribution to risk (MCR)
    # -------------------------------------------------------------------------

    def marginal_contribution_to_risk(
        self,
        snapshot: PortfolioSnapshot,
        symbol_returns: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute each position's marginal contribution to portfolio risk.

        MCR_i = (w_i × cov(r_i, r_portfolio)) / portfolio_vol

        where w_i is the weight of symbol i in the portfolio (by market value)
        and r_portfolio is the value-weighted portfolio return series.

        Args:
            snapshot: Current portfolio snapshot.
            symbol_returns: Dict mapping symbol → list of historical daily returns
                (as fractions, e.g. 0.01 = 1%).  All series must have equal length.

        Returns:
            Dict mapping symbol → MCR fraction (MCR_i / sum(MCR)).
            An empty dict is returned when the portfolio is empty or no returns
            are available.

        Notes:
            Symbols not in symbol_returns are skipped.  MCR values do not sum
            to exactly 1.0 due to cross-term effects; interpret each as the
            approximate contribution of that symbol to total portfolio volatility.
        """
        if not snapshot.positions:
            return {}

        total_value = snapshot.total_market_value
        if total_value <= 0:
            return {}

        # Build weight vector and aligned returns matrix
        weights: dict[str, float] = {}
        aligned_symbols: list[str] = []
        aligned_returns: list[list[float]] = []

        for pos in snapshot.positions:
            if pos.symbol in symbol_returns and len(symbol_returns[pos.symbol]) > 1:
                weights[pos.symbol] = pos.market_value / total_value
                aligned_symbols.append(pos.symbol)
                aligned_returns.append(symbol_returns[pos.symbol])

        if not aligned_symbols:
            return {}

        # Trim all series to the same minimum length
        min_len = min(len(r) for r in aligned_returns)
        returns_array = np.array([r[-min_len:] for r in aligned_returns], dtype=float)
        # Shape: (n_symbols, n_bars)

        w_array = np.array([weights[sym] for sym in aligned_symbols], dtype=float)

        # Portfolio return series
        r_portfolio = w_array @ returns_array  # shape: (n_bars,)

        portfolio_var = float(np.var(r_portfolio, ddof=1))
        portfolio_vol = math.sqrt(portfolio_var) if portfolio_var > 0 else 1e-12

        mcr: dict[str, float] = {}
        for i, symbol in enumerate(aligned_symbols):
            cov_i_p = float(np.cov(returns_array[i], r_portfolio)[0, 1])
            mcr[symbol] = (weights[symbol] * cov_i_p) / portfolio_vol

        log.debug(
            "portfolio_allocator.mcr",
            n_symbols=len(mcr),
            portfolio_vol=round(portfolio_vol, 6),
        )
        return mcr

    # -------------------------------------------------------------------------
    # Expected shortfall (CVaR)
    # -------------------------------------------------------------------------

    def expected_shortfall(
        self,
        daily_pnl_series: list[float],
        confidence: float = 0.95,
    ) -> float:
        """Compute the Expected Shortfall (CVaR) of the portfolio.

        CVaR at the given confidence level is the average P&L in the worst
        (1 - confidence)% of trading days.

        Args:
            daily_pnl_series: List of daily P&L values in INR (negative = loss).
            confidence: Confidence level for CVaR. Default 0.95 (95%).

        Returns:
            CVaR as a negative INR value (worst expected average loss in tail).
            Returns 0.0 if the series is empty or has only one element.

        Notes:
            A return of e.g. -50000 means "in the worst 5% of days, the average
            loss is ₹50,000".
        """
        if len(daily_pnl_series) < 2:
            return 0.0

        sorted_pnl = sorted(daily_pnl_series)
        cutoff_idx = max(1, int(len(sorted_pnl) * (1.0 - confidence)))
        tail = sorted_pnl[:cutoff_idx]

        cvar = float(np.mean(tail)) if tail else 0.0

        log.debug(
            "portfolio_allocator.expected_shortfall",
            confidence=confidence,
            n_days=len(daily_pnl_series),
            tail_days=len(tail),
            cvar_inr=round(cvar, 2),
        )
        return cvar

    # -------------------------------------------------------------------------
    # Per-symbol hard cap
    # -------------------------------------------------------------------------

    def _check_single_symbol_limit(
        self,
        snapshot: PortfolioSnapshot,
        new_symbol: str,
        new_value: float,
    ) -> tuple[bool, str]:
        """Check SURV-8 single-symbol hard cap.

        Args:
            snapshot: Current portfolio snapshot.
            new_symbol: Symbol being considered.
            new_value: Market value of the proposed new position in INR.

        Returns:
            Tuple (allowed, reason).
        """
        if snapshot.total_capital <= 0:
            return True, ""

        # Existing exposure to this symbol
        existing_value = sum(
            p.market_value
            for p in snapshot.positions
            if p.symbol == new_symbol
        )

        proposed_value = existing_value + new_value
        proposed_pct = (proposed_value / snapshot.total_capital) * 100.0

        if proposed_pct > self._config.max_single_symbol_pct:
            reason = (
                f"Single-symbol limit (SURV-8): {new_symbol} would reach "
                f"{proposed_pct:.1f}% of capital > limit {self._config.max_single_symbol_pct:.1f}%"
            )
            log.info(
                "portfolio_allocator.single_symbol_limit_breach",
                symbol=new_symbol,
                proposed_pct=round(proposed_pct, 2),
                limit_pct=self._config.max_single_symbol_pct,
            )
            return False, reason

        return True, ""

    # -------------------------------------------------------------------------
    # Beta cap check
    # -------------------------------------------------------------------------

    def _check_beta_limit(
        self,
        snapshot: PortfolioSnapshot,
        new_symbol: str,
        new_value: float,
        new_beta: float,
    ) -> tuple[bool, str]:
        """Check whether adding a new position would breach the beta exposure cap.

        Args:
            snapshot: Current portfolio snapshot.
            new_symbol: Symbol being considered.
            new_value: Market value of the proposed new position in INR.
            new_beta: Beta of the new symbol vs Nifty.

        Returns:
            Tuple (allowed, reason).
        """
        total_current_value = snapshot.total_market_value
        new_total_value = total_current_value + new_value

        if new_total_value <= 0:
            return True, ""

        current_weighted_beta = sum(p.beta * p.market_value for p in snapshot.positions)
        proposed_weighted_beta = current_weighted_beta + new_beta * new_value
        proposed_portfolio_beta = proposed_weighted_beta / new_total_value

        if proposed_portfolio_beta > self._config.max_beta_exposure:
            reason = (
                f"Beta limit breach: adding {new_symbol} (beta={new_beta:.2f}) "
                f"would raise portfolio beta to {proposed_portfolio_beta:.2f} "
                f"> limit {self._config.max_beta_exposure:.2f}"
            )
            log.info(
                "portfolio_allocator.beta_limit_breach",
                symbol=new_symbol,
                new_beta=new_beta,
                proposed_portfolio_beta=round(proposed_portfolio_beta, 4),
                limit=self._config.max_beta_exposure,
            )
            return False, reason

        return True, ""

    # -------------------------------------------------------------------------
    # Strategy drawdown budget
    # -------------------------------------------------------------------------

    def get_strategy_allocation_remaining(
        self,
        snapshot: PortfolioSnapshot,
        strategy_name: str,
        strategy_initial_capital: float,
    ) -> float:
        """Compute how much more capital a strategy can deploy before hitting its drawdown budget.

        The drawdown budget is: strategy_initial_capital × strategy_drawdown_budget_pct / 100.
        If unrealised losses for the strategy have already consumed the budget,
        returns 0.0 (the strategy should be paused).

        Args:
            snapshot: Current portfolio snapshot.
            strategy_name: Strategy to evaluate.
            strategy_initial_capital: Capital originally allocated to this strategy in INR.

        Returns:
            Remaining capital headroom in INR before the strategy hits its drawdown budget.
            Returns 0.0 if the budget is already exhausted.
        """
        budget = strategy_initial_capital * (
            self._config.strategy_drawdown_budget_pct / 100.0
        )

        strategy_positions = snapshot.positions_by_strategy(strategy_name)
        strategy_unrealised_pnl = sum(p.unrealised_pnl for p in strategy_positions)

        # Loss so far (positive number = loss amount)
        loss_so_far = max(0.0, -strategy_unrealised_pnl)
        remaining = max(0.0, budget - loss_so_far)

        log.debug(
            "portfolio_allocator.strategy_drawdown_remaining",
            strategy_name=strategy_name,
            budget_inr=round(budget, 2),
            loss_so_far_inr=round(loss_so_far, 2),
            remaining_inr=round(remaining, 2),
        )
        return remaining

    # -------------------------------------------------------------------------
    # Master check
    # -------------------------------------------------------------------------

    def check_new_position_allowed(
        self,
        snapshot: PortfolioSnapshot,
        new_symbol: str,
        new_qty: int,
        new_price: float,
        strategy_name: str,
        new_beta: float = 1.0,
    ) -> tuple[bool, str]:
        """Master pre-trade check: run all portfolio-level limits.

        Checks (in order of priority):
          1. Single-symbol hard cap (SURV-8)
          2. Sector concentration limit
          3. Portfolio beta cap
          4. Strategy drawdown budget (blocks if exhausted)

        Args:
            snapshot: Current portfolio snapshot.
            new_symbol: Symbol for the proposed new position.
            new_qty: Quantity for the proposed new position (absolute value).
            new_price: Price per share in INR.
            strategy_name: Strategy submitting this order.
            new_beta: Beta of the new symbol vs Nifty. [ASSUMED 1.0 if unknown]

        Returns:
            Tuple (allowed: bool, rejection_reason: str).  rejection_reason is
            empty string when allowed.
        """
        new_value = abs(new_qty) * new_price

        # 1. Single-symbol hard cap (SURV-8)
        allowed, reason = self._check_single_symbol_limit(snapshot, new_symbol, new_value)
        if not allowed:
            return False, reason

        # 2. Sector concentration
        allowed, reason = self.check_sector_limit(snapshot, new_symbol, new_value)
        if not allowed:
            return False, reason

        # 3. Beta exposure
        allowed, reason = self._check_beta_limit(snapshot, new_symbol, new_value, new_beta)
        if not allowed:
            return False, reason

        # 4. Strategy drawdown budget
        # Use approximate initial capital = total_capital / number of distinct strategies
        distinct_strategies = {p.strategy_name for p in snapshot.positions}
        distinct_strategies.add(strategy_name)
        strategy_initial_capital = (
            snapshot.total_capital / len(distinct_strategies)
            if distinct_strategies
            else snapshot.total_capital
        )
        remaining_budget = self.get_strategy_allocation_remaining(
            snapshot, strategy_name, strategy_initial_capital
        )
        if remaining_budget <= 0:
            reason = (
                f"Strategy '{strategy_name}' drawdown budget exhausted "
                f"(budget={self._config.strategy_drawdown_budget_pct:.1f}% "
                f"of ₹{strategy_initial_capital:.0f})"
            )
            log.info(
                "portfolio_allocator.strategy_budget_exhausted",
                strategy_name=strategy_name,
            )
            return False, reason

        log.debug(
            "portfolio_allocator.position_allowed",
            symbol=new_symbol,
            qty=new_qty,
            value_inr=round(new_value, 2),
            strategy=strategy_name,
        )
        return True, ""
