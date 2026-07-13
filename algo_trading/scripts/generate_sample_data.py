"""Generate synthetic NSE-like 15-min OHLCV Parquet files for offline testing.

NO network calls, NO downloads.  All data is produced from Geometric Brownian
Motion (GBM) and parametric noise so the full pipeline (feature engine,
labeling, backtest) can be exercised without real market data.

Run as a module::

    python -m algo_trading.scripts.generate_sample_data

Output files written to ``data/historical/`` relative to the project root.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Seed for full reproducibility across runs.
RNG_SEED: Final[int] = 42

# 252 trading days/year × 3 years
TRADING_DAYS: Final[int] = 252 * 3          # 756 days

# NSE intraday 15-min bar schedule: 9:15 … 15:00 → 25 bars per day.
BAR_MINUTES: Final[int] = 15
BARS_PER_DAY: Final[int] = 25              # 9:15, 9:30, …, 15:00 (inclusive)

TOTAL_BARS: Final[int] = TRADING_DAYS * BARS_PER_DAY   # ~19,656 per symbol

# GBM annual drift/vol parameters per symbol.
# Format: (start_price, annual_mu, annual_sigma)
SYMBOL_PARAMS: Final[dict[str, tuple[float, float, float]]] = {
    "RELIANCE":  (2800.0, 0.12, 0.25),
    "TCS":       (3700.0, 0.10, 0.22),
    "HDFCBANK":  (1650.0, 0.11, 0.24),
    "INFY":      (1400.0, 0.09, 0.26),
    "ICICIBANK": (1250.0, 0.13, 0.28),
    "NIFTY50":   (19500.0, 0.10, 0.18),   # Index: lower vol
}

# Mean daily volume (shares); scaled to per-bar with intraday profile.
DAILY_VOL_MEAN: Final[int] = 1_000_000

# Black-Scholes risk-free rate for F&O Greeks.
RISK_FREE_RATE: Final[float] = 0.065   # 6.5 % RBI repo rate proxy

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _historical_dir() -> Path:
    out = _project_root() / "data" / "historical"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def _build_timestamps(n_days: int, bars_per_day: int) -> list[datetime]:
    """Build a flat list of naive datetime objects for NSE 15-min bars.

    Starts from 2021-01-04 (Monday).  Skips weekends only (no holiday calendar
    for synthetic data).

    Returns:
        List of ``n_days * bars_per_day`` datetime objects in IST (naive).
    """
    # Anchor start on a Monday so weekend skips work cleanly.
    start_date = date(2021, 1, 4)
    timestamps: list[datetime] = []

    # First bar of the day is 09:15.
    bar_start_hour, bar_start_minute = 9, 15

    day_count = 0
    current = start_date
    while day_count < n_days:
        # Skip Saturday (5) and Sunday (6).
        if current.weekday() < 5:
            for bar_idx in range(bars_per_day):
                total_minutes = bar_start_hour * 60 + bar_start_minute + bar_idx * BAR_MINUTES
                h, m = divmod(total_minutes, 60)
                timestamps.append(
                    datetime(current.year, current.month, current.day, h, m, 0)
                )
            day_count += 1
        current += timedelta(days=1)

    return timestamps


# ---------------------------------------------------------------------------
# GBM price path
# ---------------------------------------------------------------------------

def _gbm_closes(
    rng: np.random.Generator,
    start_price: float,
    annual_mu: float,
    annual_sigma: float,
    n_bars: int,
    dt: float,
) -> np.ndarray:
    """Generate GBM close prices.

    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z~N(0,1)

    Args:
        rng: Seeded NumPy random generator.
        start_price: Initial price.
        annual_mu: Annual drift.
        annual_sigma: Annual volatility.
        n_bars: Number of bars to generate.
        dt: Time step in years (BAR_MINUTES / (252 * 390) roughly).

    Returns:
        Array of shape (n_bars,) of close prices.
    """
    drift = (annual_mu - 0.5 * annual_sigma ** 2) * dt
    diffusion = annual_sigma * math.sqrt(dt)

    z = rng.standard_normal(n_bars)
    log_returns = drift + diffusion * z
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    return np.exp(log_prices)


# ---------------------------------------------------------------------------
# ATR-14 computation
# ---------------------------------------------------------------------------

def _compute_atr14(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Compute 14-bar ATR (simple rolling mean of True Range).

    TrueRange[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)

    Returns:
        Array of shape (n,).  First 13 values are forward-filled from the first
        valid ATR so that no NaN values appear in the output.
    """
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # 14-bar simple rolling mean; pad initial NaN window.
    atr = np.empty(n)
    atr[:] = np.nan
    for i in range(13, n):
        atr[i] = tr[i - 13: i + 1].mean()

    # Forward-fill the first 13 rows with the first valid ATR value.
    first_valid = atr[13] if n > 13 else (highs[0] - lows[0])
    atr[:13] = first_valid
    return atr


# ---------------------------------------------------------------------------
# Intraday volume profile
# ---------------------------------------------------------------------------

def _intraday_volume_profile(bars_per_day: int) -> np.ndarray:
    """Return normalised intraday volume weights (sum to 1).

    Higher weight at open (bar 0) and close (last bar), lower in midday.
    Approximated by a U-shaped cosine curve.
    """
    x = np.linspace(0, np.pi, bars_per_day)
    # cos(x) goes 1 → -1 → we want U-shape, so negate and shift.
    profile = 1.1 - np.cos(x)      # U-shaped, range [0.1, 2.1]
    return profile / profile.sum()


# ---------------------------------------------------------------------------
# OHLCV bar builder
# ---------------------------------------------------------------------------

def _build_ohlcv(
    rng: np.random.Generator,
    symbol: str,
    start_price: float,
    annual_mu: float,
    annual_sigma: float,
    timestamps: list[datetime],
) -> pl.DataFrame:
    """Build a complete OHLCV DataFrame for one symbol.

    Returns a Polars DataFrame with columns:
        timestamp, open, high, low, close, volume, atr14, symbol
    """
    n = len(timestamps)
    # 15-min bar as fraction of a trading year (252 days × 390 minutes/day).
    dt = BAR_MINUTES / (252 * 390)

    # ---- Close prices via GBM ----
    closes = _gbm_closes(rng, start_price, annual_mu, annual_sigma, n, dt)

    # ---- Open: previous close + tiny gap (~0.01-0.05% of price) ----
    gap_pct = rng.uniform(0.0001, 0.0005, size=n)
    gap_sign = rng.choice([-1.0, 1.0], size=n)
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1] * (1.0 + gap_sign[1:] * gap_pct[1:])

    # ---- Provisional ATR from |close - prev_close| to seed OHLC ----
    # We use a rough ATR proxy to set realistic high/low expansions.
    rough_atr = np.abs(np.diff(closes, prepend=closes[0])) * 2.0 + closes * 0.002

    # ---- High / Low ----
    upper_expansion = rng.uniform(0.0, 0.5, size=n) * rough_atr
    lower_expansion = rng.uniform(0.0, 0.5, size=n) * rough_atr
    highs = np.maximum(opens, closes) + upper_expansion
    lows  = np.minimum(opens, closes) - lower_expansion

    # Ensure OHLC coherence (numerical safety).
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows,  np.minimum(opens, closes))

    # ---- Proper ATR-14 from final OHLC ----
    atr14 = _compute_atr14(highs, lows, closes)

    # ---- Volume: log-normal with intraday profile ----
    vol_profile = _intraday_volume_profile(BARS_PER_DAY)
    # Tile the profile across all days.
    n_days = n // BARS_PER_DAY
    leftover = n % BARS_PER_DAY
    tiled_profile = np.tile(vol_profile, n_days)
    if leftover:
        tiled_profile = np.concatenate([tiled_profile, vol_profile[:leftover]])

    # Log-normal noise (sigma=0.5 gives realistic day-to-day spread).
    base_volume = rng.lognormal(mean=0.0, sigma=0.5, size=n)
    volumes = (base_volume * tiled_profile * DAILY_VOL_MEAN).astype(np.int64)
    volumes = np.clip(volumes, 100, None)   # floor at 100 shares per bar

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open":      opens.astype(np.float64),
            "high":      highs.astype(np.float64),
            "low":       lows.astype(np.float64),
            "close":     closes.astype(np.float64),
            "volume":    volumes,
            "atr14":     atr14.astype(np.float64),
            "symbol":    [symbol] * n,
        },
        schema={
            "timestamp": pl.Datetime,
            "open":      pl.Float64,
            "high":      pl.Float64,
            "low":       pl.Float64,
            "close":     pl.Float64,
            "volume":    pl.Int64,
            "atr14":     pl.Float64,
            "symbol":    pl.Utf8,
        },
    )


# ---------------------------------------------------------------------------
# Tick data (Hawkes-like) for RELIANCE
# ---------------------------------------------------------------------------

def _build_tick_data(rng: np.random.Generator) -> pl.DataFrame:
    """Generate 100,000 synthetic ticks for RELIANCE over one trading day.

    Inter-arrival times follow a Poisson process (lambda=5 ticks/sec) with
    Hawkes-like clustering: every 50th tick triggers a burst of 5-10 ticks
    within 1 second.

    Returns a DataFrame with columns:
        timestamp_ns, price, quantity, side, bid_qty, ask_qty
    """
    TARGET_TICKS = 100_000
    DAY_START_NS: int = int(
        datetime(2021, 1, 4, 9, 15, 0).replace(tzinfo=timezone.utc).timestamp() * 1e9
    )
    DAY_END_NS: int = int(
        datetime(2021, 1, 4, 15, 30, 0).replace(tzinfo=timezone.utc).timestamp() * 1e9
    )
    TOTAL_SECONDS = (DAY_END_NS - DAY_START_NS) / 1e9   # 22500 s

    LAMBDA = 5.0   # baseline ticks per second

    # ---- Build arrival times ----
    timestamps_ns: list[int] = []
    current_ns = DAY_START_NS
    tick_count = 0

    while tick_count < TARGET_TICKS and current_ns < DAY_END_NS:
        # Standard Poisson inter-arrival.
        ia_s = rng.exponential(1.0 / LAMBDA)
        current_ns += int(ia_s * 1e9)

        if current_ns >= DAY_END_NS:
            break

        timestamps_ns.append(current_ns)
        tick_count += 1

        # Hawkes-like burst every 50th tick.
        if tick_count % 50 == 0:
            burst_size = int(rng.integers(5, 11))
            for _ in range(burst_size):
                if tick_count >= TARGET_TICKS:
                    break
                # Burst ticks within 1 second of the trigger tick.
                offset_ns = int(rng.uniform(0, 1e9))
                burst_ts = timestamps_ns[-burst_size] + offset_ns
                if burst_ts < DAY_END_NS:
                    timestamps_ns.append(burst_ts)
                    tick_count += 1

    # Truncate / pad to exactly TARGET_TICKS.
    n = min(len(timestamps_ns), TARGET_TICKS)
    timestamps_ns = timestamps_ns[:n]
    timestamps_ns.sort()

    ts_arr = np.array(timestamps_ns, dtype=np.int64)

    # ---- Price random walk around 2800 ----
    price_steps = rng.choice([-0.05, 0.0, 0.05], size=n, p=[0.35, 0.30, 0.35])
    prices = 2800.0 + np.cumsum(price_steps)
    prices = np.clip(prices, 2500.0, 3100.0)

    # ---- Quantity: log-normal, typical NSE lot sizes ----
    quantities = np.maximum(1, rng.lognormal(mean=5.5, sigma=1.2, size=n)).astype(np.int64)

    # ---- Side ----
    side_idx = rng.integers(0, 2, size=n)
    sides = np.where(side_idx == 0, "BUY", "SELL")

    # ---- Bid/Ask qty (correlated with side) ----
    bid_qty = np.maximum(1, rng.lognormal(mean=6.0, sigma=1.0, size=n)).astype(np.int64)
    ask_qty = np.maximum(1, rng.lognormal(mean=6.0, sigma=1.0, size=n)).astype(np.int64)

    return pl.DataFrame(
        {
            "timestamp_ns": ts_arr,
            "price":        prices,
            "quantity":     quantities,
            "side":         sides.tolist(),
            "bid_qty":      bid_qty,
            "ask_qty":      ask_qty,
        },
        schema={
            "timestamp_ns": pl.Int64,
            "price":        pl.Float64,
            "quantity":     pl.Int64,
            "side":         pl.Utf8,
            "bid_qty":      pl.Int64,
            "ask_qty":      pl.Int64,
        },
    )


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (inline)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> tuple[float, float, float, float, float, float]:
    """Return (ltp, delta, gamma, theta, vega, iv=sigma) for a European option.

    Uses Black-Scholes closed-form.  ``T`` is time to expiry in years.

    Returns:
        (ltp, delta, gamma, theta, vega, iv)
    """
    if T <= 1e-9 or sigma <= 1e-9:
        # At or past expiry.
        intrinsic = max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
        return (intrinsic, float(intrinsic > 0), 0.0, 0.0, 0.0, sigma)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "CE":
        ltp   = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        ltp   = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100.0   # per 1 % vol move
    # Theta: per calendar day.
    theta = (
        -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
        - r * K * math.exp(-r * T) * (_norm_cdf(d2) if option_type == "CE" else _norm_cdf(-d2))
    ) / 365.0

    ltp = max(ltp, 0.0)
    return (ltp, delta, gamma, theta, vega, sigma)


# ---------------------------------------------------------------------------
# F&O chain
# ---------------------------------------------------------------------------

def _build_fo_chain(rng: np.random.Generator) -> pl.DataFrame:
    """Generate a synthetic F&O options chain for RELIANCE.

    10 strikes ±ATM at 50-point intervals, 2 expiries (current + next month).
    """
    SPOT = 2800.0
    ATM_STRIKE = round(SPOT / 50) * 50   # nearest 50-point strike
    STRIKES = [ATM_STRIKE + i * 50 for i in range(-5, 6)]   # 11 strikes (include ATM)
    OPTION_TYPES = ["CE", "PE"]

    today = date(2021, 1, 4)
    # Current-month expiry: last Thursday of January 2021 → 2021-01-28.
    # Next-month expiry: last Thursday of February 2021 → 2021-02-25.
    expiries = [date(2021, 1, 28), date(2021, 2, 25)]

    rows: list[dict] = []
    for expiry in expiries:
        T = max((expiry - today).days / 365.0, 1 / 365.0)
        for strike in STRIKES:
            for opt_type in OPTION_TYPES:
                # Synthetic implied vol: higher for OTM options (vol smile).
                moneyness = abs(SPOT - strike) / SPOT
                iv = 0.25 + 0.10 * moneyness + rng.uniform(-0.03, 0.03)
                iv = float(np.clip(iv, 0.15, 0.50))

                ltp, delta, gamma, theta, vega, _ = _bs_greeks(
                    SPOT, float(strike), T, RISK_FREE_RATE, iv, opt_type
                )

                # Synthetic OI and volume.
                oi_base = int(rng.lognormal(mean=11.0, sigma=1.0))
                vol_base = int(rng.lognormal(mean=9.5, sigma=1.0))

                rows.append(
                    {
                        "expiry":      expiry,
                        "strike":      float(strike),
                        "option_type": opt_type,
                        "oi":          oi_base,
                        "volume":      vol_base,
                        "iv":          iv,
                        "ltp":         round(ltp, 2),
                        "delta":       round(delta, 4),
                        "gamma":       round(gamma, 6),
                        "theta":       round(theta, 4),
                        "vega":        round(vega, 4),
                    }
                )

    return pl.DataFrame(
        rows,
        schema={
            "expiry":      pl.Date,
            "strike":      pl.Float64,
            "option_type": pl.Utf8,
            "oi":          pl.Int64,
            "volume":      pl.Int64,
            "iv":          pl.Float64,
            "ltp":         pl.Float64,
            "delta":       pl.Float64,
            "gamma":       pl.Float64,
            "theta":       pl.Float64,
            "vega":        pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all synthetic data files."""
    rng = np.random.default_rng(RNG_SEED)

    out_dir = _historical_dir()
    timestamps = _build_timestamps(TRADING_DAYS, BARS_PER_DAY)

    print(f"Generating synthetic data → {out_dir}")
    print(f"  Bars per symbol : {len(timestamps):,}")

    # ---- OHLCV for each equity symbol + NIFTY50 ----
    for symbol, (start_price, mu, sigma) in SYMBOL_PARAMS.items():
        print(f"  [{symbol}] generating {len(timestamps):,} bars …", end=" ", flush=True)
        df = _build_ohlcv(rng, symbol, start_price, mu, sigma, timestamps)
        path = out_dir / f"{symbol}_15min_sample.parquet"
        df.write_parquet(path)
        print(f"saved → {path.name}")

    # ---- Tick data ----
    print(f"  [RELIANCE ticks] generating 100,000 ticks …", end=" ", flush=True)
    ticks_df = _build_tick_data(rng)
    tick_path = out_dir / "RELIANCE_ticks_sample.parquet"
    ticks_df.write_parquet(tick_path)
    print(f"saved → {tick_path.name}  (rows={len(ticks_df):,})")

    # ---- F&O chain ----
    print(f"  [RELIANCE F&O chain] generating options chain …", end=" ", flush=True)
    fo_df = _build_fo_chain(rng)
    fo_path = out_dir / "RELIANCE_fo_chain_sample.parquet"
    fo_df.write_parquet(fo_path)
    print(f"saved → {fo_path.name}  (rows={len(fo_df):,})")

    print("\nDone.  All synthetic files written successfully.")


if __name__ == "__main__":
    main()
