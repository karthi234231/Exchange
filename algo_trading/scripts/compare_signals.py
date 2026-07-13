"""
scripts/compare_signals.py — Compare signal variants on historical data.

Loads Parquet files from data/historical/, runs walk-forward backtest on
each signal variant, and prints a comparison table.

Usage:
    python -m algo_trading.scripts.compare_signals
    python -m algo_trading.scripts.compare_signals --symbols RELIANCE INFY
    python -m algo_trading.scripts.compare_signals --data-dir data/historical
    python -m algo_trading.scripts.compare_signals --symbols RELIANCE --output data/results.json

Signal variants compared:
  1. EMA cross (raw)
  2. EMA + ADX filter
  3. EMA + ADX + Volume filter
  4. EMA + ADX + Volume + RS filter
  5. Opening Range Breakout

Backtest rules (same as backtest/runner.py):
  - 5 walk-forward folds, purge=51 bars minimum
  - 1-bar fill delay
  - Full India cost model (IndiaCostModel.compute_round_trip)
  - 5bps adverse selection on LIMIT fills
  - Force close at 15:15 IST
  - ATR-based stop (1.5× ATR)

Output table:
  Signal Variant | Sharpe | PF | WR | MaxDD | Trades/Day | Avg Cost bps
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Path configuration ────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent.parent.parent  # project root
_DATA_DIR_DEFAULT = _BASE_DIR / "data" / "historical"
_OUTPUT_DIR = _BASE_DIR / "data"

# ── Backtest constants (mirrors backtest/runner.py) ───────────────────────────
_BARS_PER_DAY: int = 25          # ~25 × 15-min bars per NSE session
_INITIAL_CAPITAL = Decimal("500000")
_MAX_POSITION_PCT: float = 0.05
_ATR_STOP_MULT: float = 1.5
_MAX_HOLDING_BARS: int = 20
_ADVERSE_SEL_BPS: float = 5.0
_MAX_PRICE_CHASE_BPS: float = 30.0
_NO_TRADE_BEFORE_HM = (9, 30)   # (hour, minute)
_SQUARE_OFF_HM = (15, 15)       # (hour, minute)
_TRADING_DAYS_PER_YEAR: int = 250
_N_FOLDS: int = 5


# ── Signal variant definitions ────────────────────────────────────────────────

@dataclass
class VariantConfig:
    name: str
    use_adx: bool = False
    use_volume: bool = False
    use_rs: bool = False
    use_orb: bool = False


VARIANTS: list[VariantConfig] = [
    VariantConfig("EMA cross (raw)",               use_adx=False, use_volume=False, use_rs=False, use_orb=False),
    VariantConfig("EMA + ADX filter",              use_adx=True,  use_volume=False, use_rs=False, use_orb=False),
    VariantConfig("EMA + ADX + Volume",            use_adx=True,  use_volume=True,  use_rs=False, use_orb=False),
    VariantConfig("EMA + ADX + Volume + RS",       use_adx=True,  use_volume=True,  use_rs=True,  use_orb=False),
    VariantConfig("Opening Range Breakout",        use_adx=False, use_volume=False, use_rs=False, use_orb=True),
]


# ── Bar dataclass (local to this module — avoids ZMQ/broker deps) ─────────────

@dataclass
class _SimpleBar:
    """Lightweight bar for backtesting — mirrors broker_base.Bar fields used here."""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal
    num_ticks: int = 0
    bar_type: str = "time"
    interval_minutes: int = 15
    is_first_bar_of_day: bool = False


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class VariantResult:
    name: str
    sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    trading_days: int = 0
    trades_per_day: float = 0.0
    avg_cost_bps: float = 0.0
    total_net_pnl: float = 0.0
    passes_gate: bool = False
    error: Optional[str] = None


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_parquet_bars(parquet_path: Path, symbol: str) -> list[_SimpleBar]:
    """Load a Parquet file and convert to list of _SimpleBar objects.

    Expected columns: timestamp, open, high, low, close, volume
    Optional: vwap, num_ticks

    Args:
        parquet_path: Path to .parquet file.
        symbol: NSE symbol name.

    Returns:
        List of _SimpleBar sorted by timestamp ascending.
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required: pip install polars")

    df = pl.read_parquet(parquet_path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet {parquet_path.name} missing columns: {missing}")

    # Ensure timestamp is datetime
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    # Sort chronologically
    df = df.sort("timestamp")

    bars: list[_SimpleBar] = []
    prev_date: Optional[date] = None

    rows = df.to_dicts()
    for row in rows:
        ts = row["timestamp"]
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            else:
                ts = ts.astimezone(IST)
        else:
            # polars may return date — skip
            continue

        bar_date = ts.date()
        is_first = prev_date is None or bar_date != prev_date
        prev_date = bar_date

        vwap = row.get("vwap", row["close"])
        if vwap is None:
            vwap = row["close"]

        bars.append(_SimpleBar(
            timestamp=ts,
            symbol=symbol,
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
            vwap=Decimal(str(vwap)),
            num_ticks=int(row.get("num_ticks", 0) or 0),
            bar_type="time",
            interval_minutes=15,
            is_first_bar_of_day=is_first,
        ))

    log.info(
        "compare_signals.data_loaded",
        symbol=symbol,
        bars=len(bars),
        path=str(parquet_path),
    )
    return bars


def _find_parquet_files(data_dir: Path, symbols: list[str]) -> dict[str, Path]:
    """Find parquet files for the given symbols in data_dir.

    Matches files with pattern: {SYMBOL}*.parquet (case-insensitive).

    Returns:
        Dict mapping symbol → parquet path (first match per symbol).
    """
    found: dict[str, Path] = {}
    for symbol in symbols:
        sym_upper = symbol.upper()
        matches = sorted(data_dir.glob(f"{sym_upper}*.parquet"))
        if not matches:
            # Also try lowercase
            matches = sorted(data_dir.glob(f"{symbol.lower()}*.parquet"))
        if matches:
            found[symbol] = matches[0]

    return found


# ── Signal generators ─────────────────────────────────────────────────────────

def _generate_ema_signals(bars: list[_SimpleBar], fast: int = 9, slow: int = 21) -> list[int]:
    """Generate EMA crossover signals from bar list. Returns list of {-1,0,+1}."""
    n = len(bars)
    signals = [0] * n

    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    ema_f = float(bars[0].close) if bars else 0.0
    ema_s = ema_f
    prev_f = ema_f
    prev_s = ema_s
    position = 0  # track current position to avoid duplicate crosses

    for i, bar in enumerate(bars):
        bar_time = bar.timestamp.time()

        prev_f = ema_f
        prev_s = ema_s

        close = float(bar.close)
        ema_f = alpha_fast * close + (1 - alpha_fast) * ema_f
        ema_s = alpha_slow * close + (1 - alpha_slow) * ema_s

        if i < slow:
            continue

        # Skip first bar of day [SURV-8]
        if bar.is_first_bar_of_day:
            continue

        bullish = prev_f < prev_s and ema_f > ema_s
        bearish = prev_f > prev_s and ema_f < ema_s

        if bullish and position <= 0:
            signals[i] = 1
            position = 1
        elif bearish and position >= 0:
            signals[i] = -1
            position = -1

    return signals


def _generate_filtered_ema_signals(
    bars: list[_SimpleBar],
    use_adx: bool,
    use_volume: bool,
    use_rs: bool,
) -> list[int]:
    """Generate ADX/Volume/RS filtered EMA signals.

    This uses the actual FilteredEMACrossStrategy from the signals module.
    Falls back to raw signals if import fails.
    """
    try:
        from algo_trading.signals.ema_cross_filtered import FilteredEMACrossStrategy
        from algo_trading.data.broker_base import Bar as BrokerBar
    except ImportError as e:
        log.warning("compare_signals.import_failed", error=str(e), variant="filtered_ema")
        return _generate_ema_signals(bars)

    strategy = FilteredEMACrossStrategy(
        use_adx=use_adx,
        use_volume=use_volume,
        use_rs=use_rs,
    )

    signals = [0] * len(bars)
    for i, sbar in enumerate(bars):
        # Convert _SimpleBar to BrokerBar
        broker_bar = BrokerBar(
            timestamp=sbar.timestamp,
            symbol=sbar.symbol,
            open=sbar.open,
            high=sbar.high,
            low=sbar.low,
            close=sbar.close,
            volume=sbar.volume,
            vwap=sbar.vwap,
            num_ticks=sbar.num_ticks,
            bar_type=sbar.bar_type,
            interval_minutes=sbar.interval_minutes,
            is_first_bar_of_day=sbar.is_first_bar_of_day,
        )
        sig = strategy.feed_bar(broker_bar)
        if sig is not None:
            signals[i] = sig.direction

    return signals


def _generate_orb_signals(bars: list[_SimpleBar]) -> list[int]:
    """Generate Opening Range Breakout signals."""
    try:
        from algo_trading.signals.opening_range_breakout import OpeningRangeBreakoutStrategy
        from algo_trading.data.broker_base import Bar as BrokerBar
    except ImportError as e:
        log.warning("compare_signals.import_failed", error=str(e), variant="orb")
        return [0] * len(bars)

    strategy = OpeningRangeBreakoutStrategy()
    signals = [0] * len(bars)

    for i, sbar in enumerate(bars):
        broker_bar = BrokerBar(
            timestamp=sbar.timestamp,
            symbol=sbar.symbol,
            open=sbar.open,
            high=sbar.high,
            low=sbar.low,
            close=sbar.close,
            volume=sbar.volume,
            vwap=sbar.vwap,
            num_ticks=sbar.num_ticks,
            bar_type=sbar.bar_type,
            interval_minutes=sbar.interval_minutes,
            is_first_bar_of_day=sbar.is_first_bar_of_day,
        )
        sig = strategy.feed_bar(broker_bar)
        if sig is not None:
            signals[i] = sig.direction

    return signals


# ── Walk-forward backtest (self-contained, no ZMQ deps) ───────────────────────

@dataclass
class _TradeResult:
    direction: int
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    cost_inr: float
    cost_bps: float
    exit_reason: str


def _simulate_bars(
    bars: list[_SimpleBar],
    signals: list[int],
    symbol: str,
) -> list[_TradeResult]:
    """
    Simulate trades on a bar list with the standard backtest rules.

    Rules:
      - 1-bar fill delay
      - 5bps adverse selection
      - ATR stop at 1.5×
      - Force close at 15:15 IST
      - Max holding = 20 bars
      - 30bps stale signal abort
      - Full India cost model
    """
    try:
        from algo_trading.risk.cost_model import IndiaCostModel
        cost_model = IndiaCostModel()
    except ImportError:
        cost_model = None

    n = len(bars)
    trades: list[_TradeResult] = []

    # Pre-compute ATR(14) for each bar
    atrs = _compute_atr_series(bars)

    i = 0
    while i < n - 1:
        sig = signals[i]
        if sig == 0:
            i += 1
            continue

        # Time gate [SURV-8]
        bar_time = bars[i].timestamp.time()
        if (bar_time.hour, bar_time.minute) < _NO_TRADE_BEFORE_HM:
            i += 1
            continue

        signal_close = float(bars[i].close)
        atr_i = atrs[i] if atrs[i] > 0 else signal_close * 0.005
        direction = int(sig)

        # 1-bar fill delay
        fill_bar = i + 1
        fill_open = float(bars[fill_bar].open)

        # Stale signal abort
        if signal_close > 0:
            drift_bps = abs(fill_open - signal_close) / signal_close * 10_000
            if drift_bps > _MAX_PRICE_CHASE_BPS:
                i += 1
                continue

        # Adverse selection on entry
        adv_mult = 1.0 + direction * (_ADVERSE_SEL_BPS / 10_000)
        entry_f = fill_open * adv_mult
        entry_price_dec = Decimal(str(round(entry_f, 2)))

        # Position size
        max_invest = _INITIAL_CAPITAL * Decimal(str(_MAX_POSITION_PCT))
        qty = max(1, int(max_invest / entry_price_dec))

        # Stop loss
        atr_dec = Decimal(str(round(atr_i, 4)))
        stop_mult = Decimal(str(_ATR_STOP_MULT))
        if direction > 0:
            stop_price = entry_price_dec - atr_dec * stop_mult
        else:
            stop_price = entry_price_dec + atr_dec * stop_mult

        # Exit search
        exit_bar = fill_bar + 1
        exit_reason = "force_close"
        exit_price_dec = Decimal(str(float(bars[min(fill_bar, n - 1)].close)))

        while exit_bar < n:
            ts_exit = bars[exit_bar].timestamp.time()
            exit_hm = (ts_exit.hour, ts_exit.minute)

            if exit_hm >= _SQUARE_OFF_HM:
                exit_price_dec = Decimal(str(float(bars[exit_bar].open)))
                exit_reason = "time_exit"
                break

            cur_high = float(bars[exit_bar].high)
            cur_low = float(bars[exit_bar].low)
            cur_open = float(bars[exit_bar].open)

            if direction > 0 and cur_low <= float(stop_price):
                exit_price_dec = Decimal(str(cur_open))
                exit_reason = "stop"
                break
            if direction < 0 and cur_high >= float(stop_price):
                exit_price_dec = Decimal(str(cur_open))
                exit_reason = "stop"
                break

            bars_held = exit_bar - fill_bar
            if bars_held >= _MAX_HOLDING_BARS:
                exit_price_dec = Decimal(str(float(bars[exit_bar].open)))
                exit_reason = "force_close"
                break

            exit_bar += 1
        else:
            exit_price_dec = Decimal(str(float(bars[n - 1].close)))
            exit_reason = "force_close"
            exit_bar = n - 1

        # Cost computation
        qty_dec = Decimal(str(qty))
        gross_pnl = direction * (exit_price_dec - entry_price_dec) * qty_dec

        if cost_model is not None:
            fill_hour = bars[fill_bar].timestamp.hour
            try:
                entry_cost = cost_model.compute(
                    symbol=symbol,
                    side="BUY" if direction > 0 else "SELL",
                    quantity=qty,
                    price=entry_price_dec,
                    atr=atr_dec,
                    hour_of_day=fill_hour,
                )
                exit_cost = cost_model.compute(
                    symbol=symbol,
                    side="SELL" if direction > 0 else "BUY",
                    quantity=qty,
                    price=exit_price_dec,
                    atr=atr_dec,
                    hour_of_day=fill_hour,
                )
                total_cost_inr = float(entry_cost.total_inr + exit_cost.total_inr)
                turnover = float(entry_price_dec * qty_dec)
                cost_bps = (total_cost_inr / turnover * 10_000) if turnover > 0 else 0.0
            except Exception:
                total_cost_inr = float(entry_price_dec * qty_dec) * 0.003  # 30bps fallback
                turnover = float(entry_price_dec * qty_dec)
                cost_bps = 30.0
        else:
            total_cost_inr = float(entry_price_dec * qty_dec) * 0.003
            turnover = float(entry_price_dec * qty_dec)
            cost_bps = 30.0

        net_pnl = float(gross_pnl) - total_cost_inr

        trades.append(_TradeResult(
            direction=direction,
            entry_price=float(entry_price_dec),
            exit_price=float(exit_price_dec),
            quantity=qty,
            gross_pnl=float(gross_pnl),
            net_pnl=net_pnl,
            cost_inr=total_cost_inr,
            cost_bps=cost_bps,
            exit_reason=exit_reason,
        ))

        i = exit_bar + 1

    return trades


def _compute_atr_series(bars: list[_SimpleBar]) -> list[float]:
    """Compute ATR(14) for each bar using Wilder's smoothing."""
    n = len(bars)
    atrs = [0.0] * n
    atr = None

    for i in range(n):
        high = float(bars[i].high)
        low = float(bars[i].low)
        close = float(bars[i].close)

        if i == 0:
            tr = high - low
        else:
            prev_c = float(bars[i - 1].close)
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))

        if atr is None:
            atr = tr
        else:
            atr = (atr * 13 + tr) / 14

        atrs[i] = atr

    return atrs


# ── Walk-forward split (simple fixed-window) ──────────────────────────────────

def _walk_forward_splits(n: int, n_folds: int = _N_FOLDS, purge_bars: int = 51) -> list[tuple[int, int, int, int]]:
    """
    Generate walk-forward fold indices (train_start, train_end, test_start, test_end).

    Uses anchored expanding window (walk-forward, not sliding window).
    All indices are half-open: [start, end).

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.
    """
    if n < n_folds * 2 + purge_bars:
        # Not enough bars for folds — return a single split using 60/40
        split = int(n * 0.6)
        gap_end = min(split + purge_bars, n)
        return [(0, split, gap_end, n)] if gap_end < n else []

    fold_size = n // (n_folds + 1)
    splits = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end + purge_bars
        test_end = test_start + fold_size

        if test_start >= n or test_end > n:
            break

        splits.append((0, train_end, test_start, test_end))

    return splits


# ── Metric computation ────────────────────────────────────────────────────────

def _compute_sharpe(returns: list[float]) -> float:
    """Annualised Sharpe ratio from per-trade returns."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    # Annualise: assume each trade lasts ~1 bar on average; bars/year = 25 * 250
    annual_factor = math.sqrt(_BARS_PER_DAY * _TRADING_DAYS_PER_YEAR)
    return round(mean_r / std * annual_factor, 3)


def _compute_max_drawdown(returns: list[float]) -> float:
    """Maximum peak-to-trough drawdown from list of per-trade returns."""
    if not returns:
        return 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cum += r
        if cum > peak:
            peak = cum
        dd = (peak - cum) / (peak + 1e-10)
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def _compute_metrics(
    trades: list[_TradeResult],
    bars: list[_SimpleBar],
) -> dict:
    """Compute full metrics from a list of trade results."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades_per_day": 0.0,
            "avg_cost_bps": 0.0,
            "total_net_pnl": 0.0,
        }

    capital = float(_INITIAL_CAPITAL)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = wins / n

    gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_loss = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    returns = [t.net_pnl / capital for t in trades]
    sharpe = _compute_sharpe(returns)
    max_dd = _compute_max_drawdown(returns)

    total_net_pnl = sum(t.net_pnl for t in trades)

    # Trading days in bar series
    trading_days = len({b.timestamp.date() for b in bars})
    trades_per_day = round(n / trading_days, 2) if trading_days > 0 else 0.0

    avg_cost_bps = sum(t.cost_bps for t in trades) / n if n > 0 else 0.0

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "sharpe": sharpe,
        "max_drawdown": round(max_dd, 4),
        "trades_per_day": trades_per_day,
        "avg_cost_bps": round(avg_cost_bps, 2),
        "total_net_pnl": round(total_net_pnl, 2),
    }


# ── Main comparison logic ─────────────────────────────────────────────────────

def _run_variant(
    variant: VariantConfig,
    bars: list[_SimpleBar],
    symbol: str,
) -> VariantResult:
    """Run one signal variant on a bar series and return metrics."""
    result = VariantResult(name=variant.name)

    try:
        # Generate signals
        if variant.use_orb:
            signals = _generate_orb_signals(bars)
        elif variant.use_adx or variant.use_volume or variant.use_rs:
            signals = _generate_filtered_ema_signals(
                bars,
                use_adx=variant.use_adx,
                use_volume=variant.use_volume,
                use_rs=variant.use_rs,
            )
        else:
            signals = _generate_ema_signals(bars)

        # Walk-forward splits — test fold only
        n = len(bars)
        splits = _walk_forward_splits(n, n_folds=_N_FOLDS, purge_bars=51)

        if not splits:
            result.error = f"Not enough bars ({n}) for {_N_FOLDS} walk-forward folds"
            return result

        all_trades: list[_TradeResult] = []
        for train_start, train_end, test_start, test_end in splits:
            test_bars = bars[test_start:test_end]
            test_signals = signals[test_start:test_end]

            fold_trades = _simulate_bars(test_bars, test_signals, symbol)
            all_trades.extend(fold_trades)

        metrics = _compute_metrics(all_trades, bars)

        result.sharpe = metrics["sharpe"]
        result.profit_factor = metrics["profit_factor"]
        result.win_rate = metrics["win_rate"]
        result.max_drawdown = metrics["max_drawdown"]
        result.total_trades = metrics["total_trades"]
        result.trading_days = len({b.timestamp.date() for b in bars})
        result.trades_per_day = metrics["trades_per_day"]
        result.avg_cost_bps = metrics["avg_cost_bps"]
        result.total_net_pnl = metrics["total_net_pnl"]
        result.passes_gate = (
            metrics["sharpe"] >= 1.0
            and metrics["profit_factor"] >= 1.2
            and metrics["win_rate"] >= 0.40
        )

    except Exception as exc:
        result.error = str(exc)
        log.exception("compare_signals.variant_failed", variant=variant.name, error=str(exc))

    return result


def _print_comparison_table(results: list[tuple[str, list[VariantResult]]]) -> None:
    """Print formatted comparison table to stdout using only stdlib."""
    # Column widths
    col_variant = 36
    col_num = 10

    header = (
        f"{'Signal Variant':<{col_variant}}"
        f"{'Sharpe':>{col_num}}"
        f"{'PF':>{col_num}}"
        f"{'WR%':>{col_num}}"
        f"{'MaxDD%':>{col_num}}"
        f"{'Tr/Day':>{col_num}}"
        f"{'CostBPS':>{col_num}}"
        f"{'Net PnL':>{col_num}}"
        f"  Gate"
    )
    sep = "-" * len(header)

    for symbol, variant_results in results:
        print(f"\n{'='*len(header)}")
        print(f"  Symbol: {symbol}")
        print(sep)
        print(header)
        print(sep)

        for vr in variant_results:
            if vr.error:
                print(f"  {vr.name:<{col_variant-2}}  ERROR: {vr.error}")
                continue

            gate_mark = "PASS" if vr.passes_gate else "FAIL"
            print(
                f"{vr.name:<{col_variant}}"
                f"{vr.sharpe:>{col_num}.3f}"
                f"{vr.profit_factor:>{col_num}.3f}"
                f"{vr.win_rate * 100:>{col_num}.1f}"
                f"{vr.max_drawdown * 100:>{col_num}.1f}"
                f"{vr.trades_per_day:>{col_num}.2f}"
                f"{vr.avg_cost_bps:>{col_num}.1f}"
                f"{vr.total_net_pnl:>{col_num},.0f}"
                f"  {gate_mark}"
            )

        print(sep)
        print()


def _save_results(
    results: list[tuple[str, list[VariantResult]]],
    output_path: Path,
) -> None:
    """Save results to JSON file."""
    output = []
    for symbol, variant_results in results:
        for vr in variant_results:
            output.append({
                "symbol": symbol,
                "variant": vr.name,
                "sharpe": vr.sharpe,
                "profit_factor": vr.profit_factor,
                "win_rate": vr.win_rate,
                "max_drawdown": vr.max_drawdown,
                "total_trades": vr.total_trades,
                "trades_per_day": vr.trades_per_day,
                "avg_cost_bps": vr.avg_cost_bps,
                "total_net_pnl": vr.total_net_pnl,
                "passes_gate": vr.passes_gate,
                "error": vr.error,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare signal variants on historical NSE data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        help="NSE symbols to compare (default: RELIANCE TCS INFY HDFCBANK ICICIBANK)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_DATA_DIR_DEFAULT),
        help=f"Directory containing historical Parquet files (default: {_DATA_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/signal_comparison_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[v.name for v in VARIANTS] + ["all"],
        default=["all"],
        help="Signal variants to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    # ── Check data directory ──────────────────────────────────────────────────
    if not data_dir.exists() or not any(data_dir.glob("*.parquet")):
        print(f"\nERROR: No Parquet files found in {data_dir}")
        print("\nTo download historical data, run:")
        print("  python -m algo_trading.scripts.download_history --symbols RELIANCE TCS INFY --source yfinance --years 1")
        print("  python -m algo_trading.scripts.download_history --source shoonya --days 400")
        print(f"\nThen re-run: python -m algo_trading.scripts.compare_signals --data-dir {data_dir}")
        sys.exit(1)

    symbols: list[str] = args.symbols
    parquet_map = _find_parquet_files(data_dir, symbols)

    if not parquet_map:
        print(f"\nERROR: No matching Parquet files found for symbols: {symbols}")
        print(f"Files in {data_dir}:")
        for p in sorted(data_dir.glob("*.parquet"))[:20]:
            print(f"  {p.name}")
        sys.exit(1)

    # Filter variants
    if "all" in args.variants:
        active_variants = VARIANTS
    else:
        active_variants = [v for v in VARIANTS if v.name in args.variants]

    print(f"\nNSE Signal Comparison — {date.today().isoformat()}")
    print(f"Symbols: {list(parquet_map.keys())}")
    print(f"Variants: {len(active_variants)}")
    print(f"Data directory: {data_dir}")
    print(f"Walk-forward folds: {_N_FOLDS}\n")

    all_results: list[tuple[str, list[VariantResult]]] = []

    for symbol, parquet_path in sorted(parquet_map.items()):
        print(f"Loading {symbol} from {parquet_path.name} ...")
        try:
            bars = _load_parquet_bars(parquet_path, symbol)
        except Exception as e:
            print(f"  ERROR loading {symbol}: {e}")
            continue

        if len(bars) < 100:
            print(f"  SKIP: {symbol} has only {len(bars)} bars (need >= 100)")
            continue

        print(f"  {len(bars)} bars loaded. Running {len(active_variants)} variants ...")
        symbol_results: list[VariantResult] = []

        for variant in active_variants:
            print(f"    Running: {variant.name} ...", end="", flush=True)
            vr = _run_variant(variant, bars, symbol)
            symbol_results.append(vr)
            if vr.error:
                print(f" ERROR: {vr.error}")
            else:
                print(f" Sharpe={vr.sharpe:.3f}  PF={vr.profit_factor:.3f}  WR={vr.win_rate:.1%}  Trades={vr.total_trades}")

        all_results.append((symbol, symbol_results))

    if not all_results:
        print("\nNo results to display.")
        sys.exit(1)

    # ── Print table ───────────────────────────────────────────────────────────
    _print_comparison_table(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        today = date.today().isoformat()
        output_path = _OUTPUT_DIR / f"signal_comparison_{today}.json"

    _save_results(all_results, output_path)


if __name__ == "__main__":
    main()
