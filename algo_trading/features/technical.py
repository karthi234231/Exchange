"""Technical indicator features (features 1-12, 18-19).

Computes RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, OBV, Supertrend,
EMA cross, Ichimoku, VWAP deviation, Heikin-Ashi direction, and Pivot distance.

All outputs are float32. NaN when lookback period is not met.
No lookahead bias — all features are computable from data available at bar close.

NOTE: pandas_ta is NOT available for Python 3.13. All indicators are implemented
natively using pandas/numpy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import structlog

from algo_trading.core.calendar import SESSION_START as _SESSION_START  # [P0-FIX-7]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# P0-FIX-7: Use market_calendar authoritative constants instead of local copies.
_SESSION_START_HOUR: int = _SESSION_START.hour
_SESSION_START_MINUTE: int = _SESSION_START.minute
_RSI_PERIOD: int = 14
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9
_BB_PERIOD: int = 20
_BB_STD: float = 2.0
_ATR_PERIOD: int = 14
_ADX_PERIOD: int = 14
_STOCH_K: int = 14
_STOCH_D: int = 3
_STOCH_SMOOTH_K: int = 3
_SUPERTREND_PERIOD: int = 10
_SUPERTREND_MULT: float = 3.0
_EMA_FAST: int = 9
_EMA_SLOW: int = 21
_ICHIMOKU_TENKAN: int = 9
_ICHIMOKU_KIJUN: int = 26
_ICHIMOKU_SENKOU_B: int = 52


class TechnicalFeatures:
    """Computes technical indicator features 1-12, 18-19.

    Uses native pandas/numpy for indicator computation. Polars DataFrames are
    converted to pandas at the boundary, indicators computed, and results
    immediately converted back to Polars as float32.

    All NaN values are preserved — never filled with zeros — when lookback
    periods are not met.
    """

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add technical indicator feature columns to input DataFrame.

        Args:
            df: Polars DataFrame with columns: open, high, low, close, volume.
                Optional column ``timestamp`` (datetime) is used for VWAP
                session resets (9:15 AM IST). If absent, VWAP is computed
                cumulatively without session resets.

        Returns:
            The input DataFrame with the following float32 columns appended:
            feat_rsi, feat_macd_hist, feat_bb_pct_b, feat_bb_width,
            feat_atr, feat_adx, feat_stoch_k, feat_stoch_d, feat_obv,
            feat_supertrend_dir, feat_ema_cross, feat_ichimoku_pos,
            feat_vwap_dev, feat_ha_dir, feat_pivot_dist.
        """
        _validate_ohlcv(df)
        log = logger.bind(rows=len(df))
        log.debug("computing technical features")

        # Convert once to pandas for all indicator work
        pdf = df.to_pandas()

        pdf = _add_rsi(pdf)
        pdf = _add_macd_hist(pdf)
        pdf = _add_bollinger(pdf)
        pdf = _add_atr(pdf)
        pdf = _add_adx(pdf)
        pdf = _add_stochastic(pdf)
        pdf = _add_obv(pdf)
        pdf = _add_supertrend(pdf)
        pdf = _add_ema_cross(pdf)
        pdf = _add_ichimoku(pdf)
        pdf = _add_vwap_deviation(pdf)
        pdf = _add_heikin_ashi_dir(pdf)
        pdf = _add_pivot_dist(pdf)

        # Convert feature columns back to Polars and cast to float32
        feature_cols = [
            "feat_rsi", "feat_macd_hist", "feat_bb_pct_b", "feat_bb_width",
            "feat_atr", "feat_adx", "feat_stoch_k", "feat_stoch_d",
            "feat_obv", "feat_supertrend_dir", "feat_ema_cross",
            "feat_ichimoku_pos", "feat_vwap_dev", "feat_ha_dir",
            "feat_pivot_dist",
        ]
        feat_df = pl.from_pandas(pdf[feature_cols]).cast(pl.Float32)
        result = pl.concat([df, feat_df], how="horizontal")

        log.debug("technical features computed", n_features=len(feature_cols))
        return result

    def compute_single_bar(self, recent_bars: pl.DataFrame) -> dict[str, float]:
        """Compute features for the most recent (last) bar only.

        Designed for low-latency live inference. Computes features over the
        full window, then returns only the final row as a dict.

        Args:
            recent_bars: At least 60 bars (Ichimoku requires 52 bars of
                lookback; 60 gives a small buffer). Must have OHLCV columns.

        Returns:
            Dict mapping feature_name to float value. NaN represented as
            ``float('nan')``.

        Raises:
            ValueError: If ``recent_bars`` has fewer than 60 rows.
        """
        if len(recent_bars) < 60:
            raise ValueError(
                f"compute_single_bar requires >= 60 bars; got {len(recent_bars)}"
            )
        enriched = self.compute(recent_bars)
        last = enriched.row(-1, named=True)
        # Map internal column names to canonical FEATURE_NAMES from engine.py
        return {
            "rsi_14":            float(last.get("feat_rsi",            float("nan"))),
            "macd_hist":         float(last.get("feat_macd_hist",      float("nan"))),
            "bb_pct_b":          float(last.get("feat_bb_pct_b",       float("nan"))),
            "bb_width":          float(last.get("feat_bb_width",       float("nan"))),
            "atr_14":            float(last.get("feat_atr",            float("nan"))),
            "adx_14":            float(last.get("feat_adx",            float("nan"))),
            "stoch_k":           float(last.get("feat_stoch_k",        float("nan"))),
            "stoch_d":           float(last.get("feat_stoch_d",        float("nan"))),
            "obv_normalized":    float(last.get("feat_obv",            float("nan"))),
            "supertrend_dir":    float(last.get("feat_supertrend_dir", float("nan"))),
            "ema_cross_ratio":   float(last.get("feat_ema_cross",      float("nan"))),
            "ichimoku_position": float(last.get("feat_ichimoku_pos",   float("nan"))),
            "vwap_dev_bps":      float(last.get("feat_vwap_dev",       float("nan"))),
            "ha_direction":      float(last.get("feat_ha_dir",         float("nan"))),
            "pivot_dist_bps":    float(last.get("feat_pivot_dist",     float("nan"))),
        }


# ---------------------------------------------------------------------------
# Internal helpers — each takes a pandas DataFrame and returns it augmented
# ---------------------------------------------------------------------------

def _validate_ohlcv(df: pl.DataFrame) -> None:
    """Raise ValueError if required OHLCV columns are missing or values are invalid.

    Checks performed (on every row):
      - high >= low
      - close in [low, high]
      - open > 0
      - volume >= 0
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Row-level OHLCV integrity checks
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    volume = df["volume"].to_numpy()

    bad_hl = np.where(high < low)[0]
    if len(bad_hl) > 0:
        i = int(bad_hl[0])
        raise ValueError(
            f"OHLCV integrity: high={high[i]} < low={low[i]} at row {i}"
        )

    bad_close = np.where((close < low) | (close > high))[0]
    if len(bad_close) > 0:
        i = int(bad_close[0])
        raise ValueError(
            f"OHLCV integrity: close={close[i]} not in [low={low[i]}, high={high[i]}] at row {i}"
        )

    bad_open = np.where(open_ <= 0)[0]
    if len(bad_open) > 0:
        i = int(bad_open[0])
        raise ValueError(
            f"OHLCV integrity: open={open_[i]} must be > 0 at row {i}"
        )

    bad_vol = np.where(volume < 0)[0]
    if len(bad_vol) > 0:
        i = int(bad_vol[0])
        raise ValueError(
            f"OHLCV integrity: volume={volume[i]} must be >= 0 at row {i}"
        )


def _to_float32_series(series: pd.Series) -> pd.Series:
    """Cast a pandas Series to float32, preserving NaN."""
    return series.astype("float32")


# ---------------------------------------------------------------------------
# Feature 1 — RSI(14)
# ---------------------------------------------------------------------------

def _add_rsi(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI(14) and add as feat_rsi (float32).

    Uses Wilder's smoothing (equivalent to EWM with alpha=1/period).
    """
    close = pdf["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing: EWM with alpha = 1/period, adjust=False
    alpha = 1.0 / _RSI_PERIOD
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # First (_RSI_PERIOD - 1) values are NaN due to insufficient lookback
    rsi.iloc[:_RSI_PERIOD] = float("nan")

    pdf["feat_rsi"] = _to_float32_series(rsi)
    return pdf


# ---------------------------------------------------------------------------
# Feature 2 — MACD histogram (12, 26, 9)
# ---------------------------------------------------------------------------

def _add_macd_hist(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD histogram and add as feat_macd_hist.

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    close = pdf["close"]
    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line

    # Insufficient lookback: set NaN for first (slow - 1) bars
    hist.iloc[:_MACD_SLOW - 1] = float("nan")

    pdf["feat_macd_hist"] = _to_float32_series(hist)
    return pdf


# ---------------------------------------------------------------------------
# Features 3 & 4 — Bollinger %B and Width
# ---------------------------------------------------------------------------

def _add_bollinger(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Bollinger %B and Band Width.

    %B = (close - lower) / (upper - lower)
    Width = (upper - lower) / middle
    """
    close = pdf["close"]
    mid = close.rolling(_BB_PERIOD).mean()
    std = close.rolling(_BB_PERIOD).std(ddof=1)
    upper = mid + _BB_STD * std
    lower = mid - _BB_STD * std

    band_range = upper - lower
    pct_b = (close - lower) / band_range.replace(0.0, float("nan"))
    width = band_range / mid.replace(0.0, float("nan"))

    pdf["feat_bb_pct_b"] = _to_float32_series(pct_b)
    pdf["feat_bb_width"] = _to_float32_series(width)
    return pdf


# ---------------------------------------------------------------------------
# Feature 5 — ATR(14)
# ---------------------------------------------------------------------------

def _add_atr(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute ATR(14) using Wilder's smoothing."""
    high = pdf["high"]
    low = pdf["low"]
    close = pdf["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing
    alpha = 1.0 / _ATR_PERIOD
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    atr.iloc[:_ATR_PERIOD] = float("nan")

    pdf["feat_atr"] = _to_float32_series(atr)
    return pdf


# ---------------------------------------------------------------------------
# Feature 6 — ADX(14)
# ---------------------------------------------------------------------------

def _add_adx(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute ADX(14) using Wilder's smoothing."""
    high = pdf["high"]
    low = pdf["low"]
    close = pdf["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    move_up = high - prev_high
    move_down = prev_low - low

    plus_dm = move_up.where((move_up > move_down) & (move_up > 0), 0.0)
    minus_dm = move_down.where((move_down > move_up) & (move_down > 0), 0.0)

    alpha = 1.0 / _ADX_PERIOD

    atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di_raw = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_di_raw = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    atr_safe = atr_wilder.replace(0.0, float("nan"))
    plus_di = 100.0 * plus_di_raw / atr_safe
    minus_di = 100.0 * minus_di_raw / atr_safe

    dx_denom = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom

    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx.iloc[:2 * _ADX_PERIOD] = float("nan")

    pdf["feat_adx"] = _to_float32_series(adx)
    return pdf


# ---------------------------------------------------------------------------
# Feature 7 — Stochastic %K and %D
# ---------------------------------------------------------------------------

def _add_stochastic(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Stochastic %K (smoothed) and %D (3-period SMA of %K)."""
    high = pdf["high"]
    low = pdf["low"]
    close = pdf["close"]

    highest_high = high.rolling(_STOCH_K).max()
    lowest_low = low.rolling(_STOCH_K).min()

    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, float("nan"))
    # Smooth %K
    smooth_k = raw_k.rolling(_STOCH_SMOOTH_K).mean()
    # %D is SMA of smoothed %K
    stoch_d = smooth_k.rolling(_STOCH_D).mean()

    pdf["feat_stoch_k"] = _to_float32_series(smooth_k)
    pdf["feat_stoch_d"] = _to_float32_series(stoch_d)
    return pdf


# ---------------------------------------------------------------------------
# Feature 8 — OBV (On-Balance Volume)
# ---------------------------------------------------------------------------

def _add_obv(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute OBV normalized to millions and add as feat_obv.

    Raw OBV values can reach billions for large-cap NSE stocks, causing
    float32 overflow (~3.4e38 max, but precision loss starts much earlier).
    Dividing by 1,000,000 scales the values to a float32-safe range while
    preserving the sign and relative magnitude needed for ML features.

    The canonical feature name exposed via compute_single_bar() is
    ``obv_normalized``, consistent with this scaling.
    """
    close = pdf["close"]
    volume = pdf["volume"]
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    pdf["feat_obv"] = _to_float32_series(obv / 1_000_000)
    return pdf


# ---------------------------------------------------------------------------
# Feature 9 — Supertrend direction (10, 3)
# ---------------------------------------------------------------------------

def _add_supertrend(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Supertrend(10, 3) direction (+1 or -1) and add as feat_supertrend_dir."""
    high = pdf["high"].to_numpy(dtype=float)
    low = pdf["low"].to_numpy(dtype=float)
    close = pdf["close"].to_numpy(dtype=float)
    n = len(close)

    # ATR using Wilder's smoothing
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder EWM for ATR
    alpha = 1.0 / _SUPERTREND_PERIOD
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    hl2 = (high + low) / 2.0
    upper_band = hl2 + _SUPERTREND_MULT * atr
    lower_band = hl2 - _SUPERTREND_MULT * atr

    # Final bands and direction
    final_upper = np.empty(n)
    final_lower = np.empty(n)
    direction = np.empty(n)

    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    direction[0] = 1.0  # default bullish

    for i in range(1, n):
        # Upper band: tighten only (never widen)
        if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band: tighten only (never widen)
        if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: 1 = bullish (price > lower band), -1 = bearish (price < upper band)
        if direction[i - 1] == -1.0:
            if close[i] > final_upper[i]:
                direction[i] = 1.0
            else:
                direction[i] = -1.0
        else:
            if close[i] < final_lower[i]:
                direction[i] = -1.0
            else:
                direction[i] = 1.0

    # NaN for warmup period
    dir_series = pd.Series(direction.astype("float32"), index=pdf.index)
    dir_series.iloc[:_SUPERTREND_PERIOD] = float("nan")

    pdf["feat_supertrend_dir"] = dir_series
    return pdf


# ---------------------------------------------------------------------------
# Feature 10 — EMA cross (9/21 ratio - 1)
# ---------------------------------------------------------------------------

def _add_ema_cross(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA9/EMA21 ratio minus 1 (centred at 0 when EMAs are equal)."""
    close = pdf["close"]
    ema_fast = close.ewm(span=_EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_EMA_SLOW, adjust=False).mean()
    cross = (ema_fast / ema_slow.where(ema_slow != 0)) - 1.0
    pdf["feat_ema_cross"] = _to_float32_series(cross)
    return pdf


# ---------------------------------------------------------------------------
# Feature 11 — Ichimoku cloud position
# ---------------------------------------------------------------------------

def _add_ichimoku(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Ichimoku cloud position: -1 (below), 0 (in), +1 (above)."""
    high = pdf["high"]
    low = pdf["low"]
    close = pdf["close"]

    tenkan = (high.rolling(_ICHIMOKU_TENKAN).max() + low.rolling(_ICHIMOKU_TENKAN).min()) / 2
    kijun = (high.rolling(_ICHIMOKU_KIJUN).max() + low.rolling(_ICHIMOKU_KIJUN).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (
        high.rolling(_ICHIMOKU_SENKOU_B).max() + low.rolling(_ICHIMOKU_SENKOU_B).min()
    ) / 2

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # +1 above cloud, -1 below cloud, 0 inside cloud
    pos = pd.Series(np.where(close > cloud_top, 1.0,
                             np.where(close < cloud_bot, -1.0, 0.0)),
                    index=pdf.index, dtype="float32")
    # NaN where cloud values are NaN (lookback not met)
    nan_mask = cloud_top.isna() | cloud_bot.isna()
    pos[nan_mask] = np.nan

    pdf["feat_ichimoku_pos"] = pos
    return pdf


# ---------------------------------------------------------------------------
# Feature 12 — VWAP deviation (in basis points)
# ---------------------------------------------------------------------------

def _add_vwap_deviation(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute VWAP deviation = (close - VWAP) / VWAP * 10000 in basis points.

    VWAP resets at each trading session start (9:15 AM IST). If the
    ``timestamp`` column is absent, VWAP is computed cumulatively without
    session resets (suitable for single-session DataFrames).
    """
    typical_price = (pdf["high"] + pdf["low"] + pdf["close"]) / 3.0
    pv = typical_price * pdf["volume"]

    if "timestamp" in pdf.columns:
        ts = pd.to_datetime(pdf["timestamp"])
        # Mark session start rows (9:15 AM) as group boundaries
        session_id = (
            (ts.dt.hour == _SESSION_START_HOUR) & (ts.dt.minute == _SESSION_START_MINUTE)
        ).cumsum()
        cum_pv = pv.groupby(session_id).cumsum()
        cum_vol = pdf["volume"].groupby(session_id).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = pdf["volume"].cumsum()

    vwap = cum_pv / cum_vol
    dev = (pdf["close"] - vwap) / vwap * 10_000.0
    pdf["feat_vwap_dev"] = _to_float32_series(dev)
    return pdf


# ---------------------------------------------------------------------------
# Feature 18 — Heikin-Ashi direction
# ---------------------------------------------------------------------------

def _add_heikin_ashi_dir(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi direction: +1 (HA close > HA open), -1 otherwise.

    HA Close  = (open + high + low + close) / 4
    HA Open   = (prev_HA_open + prev_HA_close) / 2  (initialised at row 0 open)
    """
    n = len(pdf)
    ha_open = np.empty(n, dtype="float64")
    ha_close = (pdf["open"].values + pdf["high"].values +
                pdf["low"].values + pdf["close"].values) / 4.0

    ha_open[0] = pdf["open"].iloc[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    direction = np.where(ha_close > ha_open, 1.0, -1.0).astype("float32")
    pdf["feat_ha_dir"] = direction
    return pdf


# ---------------------------------------------------------------------------
# Feature 19 — Pivot distance (in basis points)
# ---------------------------------------------------------------------------

def _add_pivot_dist(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute distance from pivot point in basis points.

    Pivot = (prev_high + prev_low + prev_close) / 3
    Distance = (close - pivot) / pivot * 10000

    The first bar has NaN because there is no previous bar.
    """
    prev_high = pdf["high"].shift(1)
    prev_low = pdf["low"].shift(1)
    prev_close = pdf["close"].shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    dist = (pdf["close"] - pivot) / pivot.where(pivot != 0) * 10_000.0
    pdf["feat_pivot_dist"] = _to_float32_series(dist)
    return pdf
