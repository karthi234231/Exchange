"""Tests for algo_trading/features/tfi.py — Trade Flow Imbalance calculator.

All tests use synthetic order book snapshots. No file I/O, no external data.
"""
from __future__ import annotations

import numpy as np
import pytest

from algo_trading.features.tfi import (
    OFILevel,
    OrderBookSnapshot,
    TFICalculator,
    TFIResult,
    TFIRollingWindow,
)

# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

_DEFAULT_BIDS: list[tuple[float, int]] = [
    (2800.0, 1000),
    (2799.5, 800),
    (2799.0, 600),
    (2798.5, 400),
    (2798.0, 200),
]
_DEFAULT_ASKS: list[tuple[float, int]] = [
    (2800.5, 900),
    (2801.0, 700),
    (2801.5, 500),
    (2802.0, 300),
    (2802.5, 100),
]


def make_snapshot(
    timestamp_ns: int,
    bids: list[tuple[float, int]] | None = None,
    asks: list[tuple[float, int]] | None = None,
    symbol: str = "RELIANCE",
) -> OrderBookSnapshot:
    """Create a synthetic OrderBookSnapshot.

    Args:
        timestamp_ns: UTC timestamp in nanoseconds.
        bids: List of (price, qty) bid levels. Defaults to _DEFAULT_BIDS.
        asks: List of (price, qty) ask levels. Defaults to _DEFAULT_ASKS.
        symbol: Instrument identifier.

    Returns:
        OrderBookSnapshot with the provided or default book state.
    """
    return OrderBookSnapshot(
        timestamp_ns=timestamp_ns,
        symbol=symbol,
        bids=bids if bids is not None else list(_DEFAULT_BIDS),
        asks=asks if asks is not None else list(_DEFAULT_ASKS),
    )


# ---------------------------------------------------------------------------
# Test 1: first snapshot returns None (no previous state to diff against)
# ---------------------------------------------------------------------------

def test_first_snapshot_returns_none() -> None:
    calc = TFICalculator()
    snap = make_snapshot(timestamp_ns=1_000_000_000)
    result = calc.compute_tfi(snap)
    assert result is None, "First snapshot must return None (no previous to diff)"


# ---------------------------------------------------------------------------
# Test 2: second snapshot returns a TFIResult
# ---------------------------------------------------------------------------

def test_second_snapshot_returns_result() -> None:
    calc = TFICalculator()
    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    snap2 = make_snapshot(timestamp_ns=2_000_000_000)
    calc.compute_tfi(snap1)
    result = calc.compute_tfi(snap2)
    assert isinstance(result, TFIResult), "Second snapshot must return a TFIResult"


# ---------------------------------------------------------------------------
# Test 3: TFI is always in [-1, +1]
# ---------------------------------------------------------------------------

def test_tfi_range() -> None:
    """Normalised TFI must lie in [-1, +1] for arbitrary book changes."""
    rng = np.random.default_rng(0)
    calc = TFICalculator()

    # Seed with first snapshot
    calc.compute_tfi(make_snapshot(timestamp_ns=0))

    for i in range(1, 50):
        bids = [(2800.0 - k * 0.5, int(rng.integers(0, 5000))) for k in range(5)]
        asks = [(2800.5 + k * 0.5, int(rng.integers(0, 5000))) for k in range(5)]
        result = calc.compute_tfi(make_snapshot(timestamp_ns=i * 1_000_000_000, bids=bids, asks=asks))
        assert result is not None
        assert -1.0 <= result.tfi <= 1.0, f"TFI out of range at step {i}: {result.tfi}"


# ---------------------------------------------------------------------------
# Test 4: buying pressure → positive TFI
# ---------------------------------------------------------------------------

def test_buy_pressure_positive_tfi() -> None:
    """When all bid quantities increase and all ask quantities decrease, TFI > 0."""
    calc = TFICalculator()

    # Initial snapshot
    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    calc.compute_tfi(snap1)

    # Large bid increase, ask decrease — pure buy pressure
    new_bids = [(p, qty + 500) for p, qty in _DEFAULT_BIDS]
    new_asks = [(p, max(1, qty - 400)) for p, qty in _DEFAULT_ASKS]
    snap2 = make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids, asks=new_asks)
    result = calc.compute_tfi(snap2)

    assert result is not None
    assert result.tfi > 0, f"Expected positive TFI for buy pressure, got {result.tfi}"


# ---------------------------------------------------------------------------
# Test 5: selling pressure → negative TFI
# ---------------------------------------------------------------------------

def test_sell_pressure_negative_tfi() -> None:
    """When all bid quantities decrease and all ask quantities increase, TFI < 0."""
    calc = TFICalculator()

    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    calc.compute_tfi(snap1)

    # Bid decrease, large ask increase — pure sell pressure
    new_bids = [(p, max(1, qty - 400)) for p, qty in _DEFAULT_BIDS]
    new_asks = [(p, qty + 500) for p, qty in _DEFAULT_ASKS]
    snap2 = make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids, asks=new_asks)
    result = calc.compute_tfi(snap2)

    assert result is not None
    assert result.tfi < 0, f"Expected negative TFI for sell pressure, got {result.tfi}"


# ---------------------------------------------------------------------------
# Test 6: balanced flow → TFI near zero
# ---------------------------------------------------------------------------

def test_balanced_flow_near_zero() -> None:
    """When bid and ask deltas are equal in magnitude, TFI should be near zero."""
    calc = TFICalculator()

    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    calc.compute_tfi(snap1)

    # Identical delta on both sides: bid +100, ask +100
    new_bids = [(p, qty + 100) for p, qty in _DEFAULT_BIDS]
    new_asks = [(p, qty + 100) for p, qty in _DEFAULT_ASKS]
    snap2 = make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids, asks=new_asks)
    result = calc.compute_tfi(snap2)

    assert result is not None
    # bid_delta = 100, ask_delta = 100, ofi = 0 at every level
    assert abs(result.tfi) < 1e-6, f"Expected TFI ≈ 0 for balanced flow, got {result.tfi}"


# ---------------------------------------------------------------------------
# Test 7: signal classification
# ---------------------------------------------------------------------------

def test_signal_classification() -> None:
    """Signal labels are assigned based on TFI threshold boundaries."""
    calc = TFICalculator()

    def _get_tfi_with_target(target_tfi: float) -> TFIResult:
        """Helper: manufacture a scenario that forces TFI near target_tfi."""
        # We drive TFI by creating large one-sided flow at level 1 only.
        # Use a fresh calculator each time to avoid state contamination.
        c = TFICalculator(n_levels=1, decay_weights=False)
        c.compute_tfi(make_snapshot(1))  # seed

        if target_tfi > 0:
            # Buy: bid increases by X, ask decreases by 0
            bid_qty_delta = int(abs(target_tfi) * 10_000)
            new_bids = [(2800.0, 1000 + bid_qty_delta)] + list(_DEFAULT_BIDS[1:])
            new_asks = list(_DEFAULT_ASKS)
        else:
            # Sell: ask increases by X, bid decreases by 0
            ask_qty_delta = int(abs(target_tfi) * 10_000)
            new_bids = list(_DEFAULT_BIDS)
            new_asks = [(2800.5, 900 + ask_qty_delta)] + list(_DEFAULT_ASKS[1:])

        snap = make_snapshot(2, bids=new_bids, asks=new_asks)
        r = c.compute_tfi(snap)
        assert r is not None
        return r

    # tfi=0.5 → strong_buy
    r_strong_buy = _get_tfi_with_target(0.5)
    assert r_strong_buy.signal == "strong_buy", (
        f"Expected strong_buy for tfi={r_strong_buy.tfi:.3f}, got {r_strong_buy.signal}"
    )

    # Near-zero tfi → neutral; manufacture a snapshot with tiny identical deltas
    c_neutral = TFICalculator()
    c_neutral.compute_tfi(make_snapshot(1))
    # Same quantities — no change at all
    r_neutral = c_neutral.compute_tfi(make_snapshot(2))
    assert r_neutral is not None
    assert r_neutral.signal == "neutral", (
        f"Expected neutral for tfi={r_neutral.tfi:.3f}, got {r_neutral.signal}"
    )

    # tfi=-0.4 → strong_sell
    r_strong_sell = _get_tfi_with_target(-0.4)
    assert r_strong_sell.signal == "strong_sell", (
        f"Expected strong_sell for tfi={r_strong_sell.tfi:.3f}, got {r_strong_sell.signal}"
    )


# ---------------------------------------------------------------------------
# Test 8: n_levels_used matches levels with non-zero deltas
# ---------------------------------------------------------------------------

def test_n_levels_used() -> None:
    """n_levels_used must equal the count of levels where |bid_delta|+|ask_delta| > 0."""
    calc = TFICalculator(n_levels=5)

    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    calc.compute_tfi(snap1)

    # Change only levels 1 and 3 (bid side); leave 2, 4, 5 unchanged
    new_bids = [
        (2800.0, 1000 + 200),   # level 1 — changed
        (2799.5, 800),           # level 2 — unchanged
        (2799.0, 600 + 100),    # level 3 — changed
        (2798.5, 400),           # level 4 — unchanged
        (2798.0, 200),           # level 5 — unchanged
    ]
    snap2 = make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids)
    result = calc.compute_tfi(snap2)

    assert result is not None
    assert result.n_levels_used == 2, (
        f"Expected 2 levels used, got {result.n_levels_used}"
    )


# ---------------------------------------------------------------------------
# Test 9: rolling window z-score is finite after 20+ updates
# ---------------------------------------------------------------------------

def test_rolling_window_zscore() -> None:
    """After 20+ TFI updates, z-score must be a finite float."""
    calc = TFICalculator()
    window = TFIRollingWindow(window_size=20)

    rng = np.random.default_rng(7)
    calc.compute_tfi(make_snapshot(0))  # seed

    last_stats: dict | None = None
    for i in range(1, 30):
        bids = [(2800.0 - k * 0.5, int(rng.integers(100, 2000))) for k in range(5)]
        asks = [(2800.5 + k * 0.5, int(rng.integers(100, 2000))) for k in range(5)]
        result = calc.compute_tfi(make_snapshot(i * 1_000_000_000, bids=bids, asks=asks))
        if result is not None:
            last_stats = window.add(result)

    assert last_stats is not None, "No TFI results were generated"
    assert np.isfinite(last_stats["tfi_zscore"]), (
        f"Z-score is not finite: {last_stats['tfi_zscore']}"
    )


# ---------------------------------------------------------------------------
# Test 10: monotonically increasing TFI series gives trend="increasing"
# ---------------------------------------------------------------------------

def test_rolling_window_trend() -> None:
    """A monotonically increasing TFI series must yield tfi_trend='increasing'."""
    window = TFIRollingWindow(window_size=10)

    # Synthesise TFIResult objects with strictly increasing TFI values
    last_stats: dict | None = None
    for i in range(10):
        tfi_val = -0.5 + i * 0.1  # -0.5, -0.4, ..., 0.4
        dummy = TFIResult(
            timestamp_ns=i * 1_000_000_000,
            tfi=tfi_val,
            raw_ofi=float(tfi_val),
            total_flow=1000,
            n_levels_used=5,
            buy_pressure=max(0.0, tfi_val),
            signal="neutral",
        )
        last_stats = window.add(dummy)

    assert last_stats is not None
    assert last_stats["tfi_trend"] == "increasing", (
        f"Expected 'increasing', got '{last_stats['tfi_trend']}'"
    )


# ---------------------------------------------------------------------------
# Test 11: OFILevel.ofi == bid_delta - ask_delta
# ---------------------------------------------------------------------------

def test_ofi_level_computation() -> None:
    """OFILevel.ofi must equal bid_delta - ask_delta for every level."""
    calc = TFICalculator(n_levels=5)

    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    snap2 = make_snapshot(
        timestamp_ns=2_000_000_000,
        bids=[(p, qty + 50 * (i + 1)) for i, (p, qty) in enumerate(_DEFAULT_BIDS)],
        asks=[(p, qty - 30 * (i + 1)) for i, (p, qty) in enumerate(_DEFAULT_ASKS)],
    )

    ofi_levels = calc.compute_ofi(snap1, snap2)

    for lvl in ofi_levels:
        assert lvl.ofi == lvl.bid_delta - lvl.ask_delta, (
            f"Level {lvl.level}: ofi={lvl.ofi} != bid_delta-ask_delta="
            f"{lvl.bid_delta}-{lvl.ask_delta}={lvl.bid_delta - lvl.ask_delta}"
        )


# ---------------------------------------------------------------------------
# Test 12: decay_weights=True gives different result than decay_weights=False
# ---------------------------------------------------------------------------

def test_decay_weights_vs_equal_weights() -> None:
    """Decay weights and equal weights produce different TFI for asymmetric OFI."""
    # Asymmetric book: large OFI at level 1, near-zero OFI at deeper levels
    # Decay weights amplify level 1 vs equal weights → different TFI values.
    new_bids = [
        (2800.0, 1000 + 2000),  # level 1: very large positive delta
        (2799.5, 800 - 10),     # level 2: tiny negative delta
        (2799.0, 600 - 10),     # level 3: tiny negative delta
        (2798.5, 400 - 10),     # level 4: tiny negative delta
        (2798.0, 200 - 10),     # level 5: tiny negative delta
    ]
    new_asks = list(_DEFAULT_ASKS)  # no change on ask side

    calc_decay = TFICalculator(n_levels=5, decay_weights=True)
    calc_equal = TFICalculator(n_levels=5, decay_weights=False)

    snap1 = make_snapshot(timestamp_ns=1_000_000_000)
    calc_decay.compute_tfi(snap1)
    calc_equal.compute_tfi(make_snapshot(timestamp_ns=1_000_000_000))

    snap2 = make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids, asks=new_asks)
    result_decay = calc_decay.compute_tfi(snap2)
    result_equal = calc_equal.compute_tfi(
        make_snapshot(timestamp_ns=2_000_000_000, bids=new_bids, asks=new_asks)
    )

    assert result_decay is not None
    assert result_equal is not None
    assert result_decay.tfi != result_equal.tfi, (
        f"Expected different TFI values for decay vs equal weights, "
        f"got both={result_decay.tfi:.6f}"
    )
