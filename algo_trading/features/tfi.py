"""Trade Flow Imbalance (TFI) — multi-level order book imbalance feature.

Stage 3+ only: requires L2 order book depth data (bid/ask at multiple levels).
TFI computed from snapped 1-3 sec Shoonya feed produces unreliable signals.
Only valid when DATA_FEED='truedata' and using sub-minute bars.

Based on Cont et al. (2014) "Price impact of order book events":
  OFI_k(t) = ΔB_k(t) - ΔA_k(t)
  TFI(t)   = Σ_k w_k·OFI_k(t) / (Σ_k (|ΔB_k|+|ΔA_k|) + ε)

Weights w_k = 1/k (geometric decay: level 1 is most important).
Output range: [-1, +1] — positive = buying pressure, negative = selling pressure.
"""
from __future__ import annotations

STAGE = "3+"

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderBookSnapshot:
    """A single L2 order book snapshot.

    Attributes:
        timestamp_ns: UTC timestamp in nanoseconds.
        symbol: Instrument identifier (e.g. "RELIANCE").
        bids: Top-5 bid levels as [(price, qty)], sorted best (highest) first.
        asks: Top-5 ask levels as [(price, qty)], sorted best (lowest) first.
    """
    timestamp_ns: int
    symbol: str
    bids: list[tuple[float, int]]
    asks: list[tuple[float, int]]


@dataclass
class OFILevel:
    """Order Flow Imbalance contribution for a single price level.

    Attributes:
        level: 1-indexed depth level (1 = best bid/ask).
        bid_delta: Change in bid quantity at this level (positive = increase).
        ask_delta: Change in ask quantity at this level (positive = increase).
        ofi: bid_delta - ask_delta.
    """
    level: int
    bid_delta: int
    ask_delta: int
    ofi: int


@dataclass
class TFIResult:
    """Output of a single TFI computation step.

    Attributes:
        timestamp_ns: UTC timestamp (nanoseconds) from the current snapshot.
        tfi: Normalised imbalance in [-1, +1].
        raw_ofi: Weighted but un-normalised OFI sum (before dividing by total_flow).
        total_flow: Sum of absolute bid and ask deltas across all levels (+ ε=1).
        n_levels_used: Count of levels where at least one delta is non-zero.
        buy_pressure: Fraction of levels with strictly positive OFI.
        signal: Human-readable label ("strong_buy", "buy", "neutral", "sell",
            "strong_sell") based on tfi thresholds ±0.1 and ±0.3.
    """
    timestamp_ns: int
    tfi: float
    raw_ofi: float
    total_flow: int
    n_levels_used: int
    buy_pressure: float
    signal: str


# ---------------------------------------------------------------------------
# Signal classification helper
# ---------------------------------------------------------------------------

def _classify_signal(tfi: float) -> str:
    """Map a TFI value to a categorical signal label.

    Thresholds:
        > +0.3  → "strong_buy"
        > +0.1  → "buy"
        < -0.3  → "strong_sell"
        < -0.1  → "sell"
        else    → "neutral"

    Args:
        tfi: Normalised TFI value in [-1, +1].

    Returns:
        One of "strong_buy", "buy", "neutral", "sell", "strong_sell".
    """
    if tfi > 0.3:
        return "strong_buy"
    if tfi > 0.1:
        return "buy"
    if tfi < -0.3:
        return "strong_sell"
    if tfi < -0.1:
        return "sell"
    return "neutral"


# ---------------------------------------------------------------------------
# TFICalculator
# ---------------------------------------------------------------------------

class TFICalculator:
    """Compute Trade Flow Imbalance from consecutive L2 order book snapshots.

    Each call to :meth:`update` (or :meth:`compute_tfi`) ingests one snapshot.
    The first call has no previous snapshot to diff against, so it returns None
    and stores the snapshot for the next call.

    Args:
        n_levels: How many order book depth levels to use (1–5, default 5).
        decay_weights: If True, weight level k with 1/k (level 1 is most
            important). If False, all levels share equal weight 1/n_levels.
    """

    def __init__(self, n_levels: int = 5, decay_weights: bool = True) -> None:
        if not (1 <= n_levels <= 5):
            raise ValueError(f"n_levels must be between 1 and 5; got {n_levels}")
        self._n_levels = n_levels
        self._decay_weights = decay_weights
        self._prev_snapshot: Optional[OrderBookSnapshot] = None

        # Pre-compute weight vector
        if decay_weights:
            # w_k = 1/k, k in 1..n_levels
            self._weights = np.array([1.0 / k for k in range(1, n_levels + 1)], dtype="float64")
        else:
            self._weights = np.full(n_levels, 1.0 / n_levels, dtype="float64")

    # ------------------------------------------------------------------
    def compute_ofi(
        self,
        prev: OrderBookSnapshot,
        curr: OrderBookSnapshot,
    ) -> list[OFILevel]:
        """Compute per-level Order Flow Imbalance between two consecutive snapshots.

        For each level k (1-indexed), quantities are compared at the same
        rank position.  If the price at a given rank has changed between
        snapshots, the delta is set to 0 (price-level shift is not attributed
        to a single side).  Only levels present in both snapshots are used.

        Args:
            prev: The earlier order book snapshot.
            curr: The more recent order book snapshot.

        Returns:
            List of :class:`OFILevel` objects, one per depth level.
        """
        levels: list[OFILevel] = []
        for k in range(1, self._n_levels + 1):
            idx = k - 1  # 0-based index into bids/asks lists

            # --- Bid side ---
            if idx < len(prev.bids) and idx < len(curr.bids):
                prev_bid_price, prev_bid_qty = prev.bids[idx]
                curr_bid_price, curr_bid_qty = curr.bids[idx]
                if prev_bid_price == curr_bid_price:
                    bid_delta = curr_bid_qty - prev_bid_qty
                else:
                    bid_delta = 0
            else:
                bid_delta = 0

            # --- Ask side ---
            if idx < len(prev.asks) and idx < len(curr.asks):
                prev_ask_price, prev_ask_qty = prev.asks[idx]
                curr_ask_price, curr_ask_qty = curr.asks[idx]
                if prev_ask_price == curr_ask_price:
                    ask_delta = curr_ask_qty - prev_ask_qty
                else:
                    ask_delta = 0
            else:
                ask_delta = 0

            levels.append(OFILevel(
                level=k,
                bid_delta=bid_delta,
                ask_delta=ask_delta,
                ofi=bid_delta - ask_delta,
            ))

        return levels

    # ------------------------------------------------------------------
    def compute_tfi(self, curr_snapshot: OrderBookSnapshot) -> Optional[TFIResult]:
        """Ingest a snapshot and return the TFI result, or None on the first call.

        On the first call there is no previous snapshot to diff against.
        The snapshot is stored and None is returned.  Subsequent calls
        compute OFI across all levels, weight and normalise, and return a
        fully populated :class:`TFIResult`.

        Args:
            curr_snapshot: The latest L2 order book snapshot.

        Returns:
            :class:`TFIResult` or None (if this is the first snapshot).
        """
        if self._prev_snapshot is None:
            self._prev_snapshot = curr_snapshot
            logger.debug("tfi.first_snapshot_stored", symbol=curr_snapshot.symbol)
            return None

        ofi_levels = self.compute_ofi(self._prev_snapshot, curr_snapshot)

        # Weighted raw OFI
        raw_ofi = 0.0
        total_abs_delta = 0
        n_levels_used = 0
        n_positive_ofi = 0

        for lvl, weight in zip(ofi_levels, self._weights):
            raw_ofi += weight * lvl.ofi
            abs_delta = abs(lvl.bid_delta) + abs(lvl.ask_delta)
            total_abs_delta += abs_delta
            if abs_delta > 0:
                n_levels_used += 1
            if lvl.ofi > 0:
                n_positive_ofi += 1

        # ε = 1 to avoid division-by-zero
        total_flow = total_abs_delta + 1

        # Normalised TFI, clipped to [-1, +1]
        tfi = float(np.clip(raw_ofi / total_flow, -1.0, 1.0))

        # Buy pressure: fraction of levels with positive OFI (out of n_levels)
        buy_pressure = n_positive_ofi / self._n_levels if self._n_levels > 0 else 0.0

        result = TFIResult(
            timestamp_ns=curr_snapshot.timestamp_ns,
            tfi=tfi,
            raw_ofi=raw_ofi,
            total_flow=total_flow,
            n_levels_used=n_levels_used,
            buy_pressure=buy_pressure,
            signal=_classify_signal(tfi),
        )

        self._prev_snapshot = curr_snapshot

        logger.debug(
            "tfi.computed",
            symbol=curr_snapshot.symbol,
            tfi=round(tfi, 4),
            signal=result.signal,
            n_levels_used=n_levels_used,
        )

        return result

    def update(self, snapshot: OrderBookSnapshot) -> Optional[TFIResult]:
        """Alias for :meth:`compute_tfi` — preferred name for streaming usage.

        Args:
            snapshot: The latest L2 order book snapshot.

        Returns:
            :class:`TFIResult` or None if this is the first snapshot.
        """
        return self.compute_tfi(snapshot)


# ---------------------------------------------------------------------------
# TFIRollingWindow
# ---------------------------------------------------------------------------

class TFIRollingWindow:
    """Maintain a rolling window of TFI results and compute aggregate statistics.

    Tracks recent TFI values and provides z-score normalisation and trend
    detection via linear regression slope (numpy polyfit degree 1).

    Args:
        window_size: Number of TFI results to retain (default 20).
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._tfi_values: deque[float] = deque(maxlen=window_size)

    def add(self, result: TFIResult) -> dict:
        """Add a TFI result to the rolling window and return aggregate stats.

        Statistics returned:
            - ``tfi_mean``: Mean TFI over the window.
            - ``tfi_std``: Standard deviation of TFI over the window.
            - ``tfi_zscore``: (current_tfi - mean) / (std + 1e-8).
            - ``tfi_trend``: "increasing", "decreasing", or "flat" based on
              linear regression slope sign over the window (|slope| < 1e-8
              is classified as "flat").
            - ``current_signal``: The signal label from the most recent result.

        Args:
            result: A :class:`TFIResult` from :class:`TFICalculator`.

        Returns:
            Dict with keys: tfi_mean, tfi_std, tfi_zscore, tfi_trend,
            current_signal.
        """
        self._tfi_values.append(result.tfi)
        arr = np.array(self._tfi_values, dtype="float64")

        tfi_mean = float(np.mean(arr))
        tfi_std = float(np.std(arr, ddof=0))
        tfi_zscore = (result.tfi - tfi_mean) / (tfi_std + 1e-8)

        # Trend: linear regression slope over window indices
        if len(arr) >= 2:
            xs = np.arange(len(arr), dtype="float64")
            slope = float(np.polyfit(xs, arr, deg=1)[0])
            if slope > 1e-8:
                trend = "increasing"
            elif slope < -1e-8:
                trend = "decreasing"
            else:
                trend = "flat"
        else:
            trend = "flat"

        return {
            "tfi_mean": tfi_mean,
            "tfi_std": tfi_std,
            "tfi_zscore": float(tfi_zscore),
            "tfi_trend": trend,
            "current_signal": result.signal,
        }
