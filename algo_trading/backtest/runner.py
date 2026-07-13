"""
backtest/runner.py — Walk-forward backtesting harness with full India cost model.

Primary tool for validating strategy edge before any live trading.
All simulated trades go through IndiaCostModel.  Strict walk-forward splits
only — no random shuffle, no in-sample evaluation.

Design decisions:
  - 1-bar fill delay: signal fires at bar close → fill at NEXT bar open.
  - Stop-loss always uses MARKET order semantics (next-bar open, full slippage).
  - Stale signal abort: price moved >30 bps from signal bar close → skip trade.
  - Adverse selection penalty: 5 bps per side applied to LIMIT fills.
  - ATR-based dynamic slippage via IndiaCostModel (no static slippage).
  - Regime classification: bull/bear/sideways based on 50-bar EMA slope.
  - Edge gate enforced on OOS (walk-forward test folds) only, never in-sample.

Constraints: [R-11] No static 0.05% slippage. [SURV-8] No trade 9:15–9:30 or after 15:15.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import polars as pl
import structlog

from algo_trading.labeling.walk_forward import WalkForwardConfig, WalkForwardSplitter
from algo_trading.risk.cost_model import IndiaCostModel, TradeCost

log = structlog.get_logger(__name__)

# ── Edge gate thresholds ──────────────────────────────────────────────────────
_GATE_MIN_SHARPE: float = 1.0
_GATE_MIN_PROFIT_FACTOR: float = 1.2
_GATE_MIN_WIN_RATE: float = 0.40
_GATE_MIN_POSITIVE_REGIMES: int = 2       # of 3: bull/bear/sideways
_GATE_MIN_TRADES_PER_FOLD: int = 200

# ── Regime classification ─────────────────────────────────────────────────────
_REGIME_EMA_PERIOD: int = 50              # bars for trend regime EMA
_REGIME_FLAT_THRESHOLD: float = 0.001    # <0.1% slope → sideways

# ── Annualisation ─────────────────────────────────────────────────────────────
_BARS_PER_DAY: int = 25                   # ~25 × 15-min bars per NSE session
_TRADING_DAYS_PER_YEAR: int = 250


@dataclass
class BacktestConfig:
    """Configuration for the walk-forward backtester.

    Attributes:
        initial_capital: Starting capital in INR.  Defaults to Rs. 5 lakh.
        max_position_pct: Maximum fraction of capital per position (0.05 = 5%).
        stop_loss_atr_mult: Stop placed at entry ± multiplier × ATR.
        max_holding_bars: Force-close after this many bars.
        no_trade_before: Skip any signal whose bar timestamp is before this
            intraday time (HH:MM, IST). Prevents trading the opening auction
            bar [SURV-8].
        square_off_time: Force-close all open positions by this time (HH:MM,
            IST) [SURV-8].
        adverse_selection_bps: Extra cost in basis points per side applied
            to LIMIT order fills to model adverse selection.
        max_price_chase_bps: Abort trade if the fill-bar open has moved more
            than this many bps away from the signal-bar close.
    """

    initial_capital: Decimal = field(default_factory=lambda: Decimal("500000"))
    max_position_pct: float = 0.05
    stop_loss_atr_mult: float = 1.5
    max_holding_bars: int = 20
    no_trade_before: str = "09:30"
    square_off_time: str = "15:15"
    adverse_selection_bps: float = 5.0
    max_price_chase_bps: float = 50.0


@dataclass
class TradeResult:
    """Complete record for one simulated round-trip trade.

    Attributes:
        symbol: NSE symbol.
        entry_bar: DataFrame row index of the fill bar (signal bar + 1).
        exit_bar: DataFrame row index of the exit fill bar.
        entry_timestamp: Timestamp of the entry fill bar (for stress window filtering).
        direction: +1 long, -1 short.
        entry_price: Actual fill price at entry (next-bar open ± slippage).
        exit_price: Actual fill price at exit.
        quantity: Number of shares traded.
        gross_pnl: P&L before any costs (Decimal, INR).
        net_pnl: P&L after all costs from IndiaCostModel (Decimal, INR).
        cost_breakdown: Full cost breakdown from IndiaCostModel.
        exit_reason: One of "target", "stop", "time_exit", "force_close",
            "stale_abort".
        signal_confidence: Model confidence at signal time [0, 1].
    """

    symbol: str
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    gross_pnl: Decimal
    net_pnl: Decimal
    cost_breakdown: TradeCost
    exit_reason: str
    signal_confidence: float
    entry_timestamp: Optional[object] = None  # datetime or None


@dataclass
class BacktestReport:
    """Aggregated results from a complete walk-forward backtest run.

    Attributes:
        total_trades: Total simulated trades across all OOS folds.
        win_rate: Fraction of trades with net_pnl > 0.
        profit_factor: Net profit / net loss (absolute) — costs included. [P0-FIX-9]
        sharpe_ratio: Annualised Sharpe ratio over walk-forward OOS returns.
        max_drawdown: Maximum drawdown as a fraction of peak capital.
        total_net_pnl: Sum of all net P&L across all trades (INR).
        per_fold_metrics: List of per-fold metric dicts.
        regime_breakdown: Nested dict of metrics per regime.
        cost_analysis: Aggregate cost statistics.
        per_symbol_metrics: Per-symbol breakdown of Sharpe, profit factor,
            and win rate.
        passes_edge_gate: True if all edge gate criteria are satisfied.
        gate_failures: List of human-readable gate failure descriptions.
    """

    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_net_pnl: Decimal
    per_fold_metrics: list[dict]
    regime_breakdown: dict[str, dict]
    cost_analysis: dict
    passes_edge_gate: bool
    gate_failures: list[str]
    per_symbol_metrics: dict[str, dict] = field(default_factory=dict)


class BacktestRunner:
    """Walk-forward backtesting harness with full India cost model.

    Every simulated trade applies:
      - IndiaCostModel (STT, exchange charge, SEBI fee, GST, stamp, slippage).
      - ATR-based dynamic slippage with time-of-day multiplier.
      - 5 bps adverse selection penalty per side for LIMIT fills.
      - 1-bar fill delay (signal at bar N close → fill at bar N+1 open).
      - Stale signal abort (price moved > max_price_chase_bps → no trade).
      - Stop-loss as MARKET at next bar open with full slippage.
      - Force-close after max_holding_bars or at square_off_time.

    Usage:
        runner = BacktestRunner(BacktestConfig())
        report = runner.run(ohlcv_df, signals, feature_df)
        if report.passes_edge_gate:
            ...  # proceed to ML
        else:
            print(report.gate_failures)

    Args:
        config: Backtest configuration.  Defaults to BacktestConfig().
    """

    def __init__(self, config: BacktestConfig = BacktestConfig()) -> None:
        self._cfg = config
        self._cost_model = IndiaCostModel()
        self._log = log.bind(component="BacktestRunner")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        df: pl.DataFrame,
        signals: pl.Series,
        feature_df: pl.DataFrame,
    ) -> BacktestReport:
        """Run a full walk-forward backtest.

        Args:
            df: Chronologically ordered OHLCV Polars DataFrame.  Required
                columns: timestamp, open, high, low, close, volume.
            signals: Integer Series of {-1, 0, +1}, one per bar, aligned to df.
                Generated by an ML model or indicator on a per-bar basis.
            feature_df: Feature DataFrame aligned to df.  Must contain
                feat_atr (ATR-14) used for stop-loss sizing and slippage.

        Returns:
            BacktestReport with full metrics and edge gate result.

        Raises:
            ValueError: If df, signals, or feature_df have mismatched lengths,
                or if required columns are absent.
        """
        self._validate_inputs(df, signals, feature_df)

        splitter = WalkForwardSplitter(
            WalkForwardConfig(
                n_folds=5,
                max_holding_bars=self._cfg.max_holding_bars,
                min_test_trades=_GATE_MIN_TRADES_PER_FOLD,
            )
        )
        splits = splitter.split(df)

        all_trades: list[TradeResult] = []
        per_fold_metrics: list[dict] = []

        for train_df, test_df, meta in splits:
            fold_signals = signals.slice(meta.test_start, meta.test_size)
            fold_features = feature_df.slice(meta.test_start, meta.test_size)

            fold_trades = self._simulate_fold(
                ohlcv=test_df,
                signals=fold_signals,
                features=fold_features,
                symbol=str(df["symbol"][0]) if "symbol" in df.columns else "UNKNOWN",
                bar_offset=meta.test_start,
            )
            all_trades.extend(fold_trades)

            fold_metrics = self._compute_fold_metrics(fold_trades, meta.fold_index)
            per_fold_metrics.append(fold_metrics)
            self._log.info(
                "backtest.fold_complete",
                fold=meta.fold_index,
                trades=len(fold_trades),
                net_pnl=str(fold_metrics.get("net_pnl", 0)),
                sharpe=fold_metrics.get("sharpe", 0.0),
            )

        # Classify regimes over the full df, then bucket trades by regime
        regimes = self._classify_regimes(df)
        regime_breakdown = self._compute_regime_breakdown(all_trades, regimes)

        report = self._build_report(all_trades, per_fold_metrics, regime_breakdown)
        self._log.info(
            "backtest.complete",
            total_trades=report.total_trades,
            sharpe=report.sharpe_ratio,
            passes_gate=report.passes_edge_gate,
            gate_failures=report.gate_failures,
        )
        return report

    def run_ema_baseline(
        self,
        df: pl.DataFrame,
        fast: int = 9,
        slow: int = 21,
    ) -> BacktestReport:
        """Run EMA cross baseline backtest — used by validate_edge.py.

        Computes EMA(fast)/EMA(slow) crossover signals on the full DataFrame,
        then runs the walk-forward backtest.  This is the profitability gate
        before adding any ML complexity.

        Args:
            df: OHLCV Polars DataFrame with columns open, high, low, close,
                volume, timestamp.
            fast: Fast EMA period (default 9).
            slow: Slow EMA period (default 21).

        Returns:
            BacktestReport from the EMA baseline walk-forward run.

        Raises:
            ValueError: If df is missing required columns.
        """
        self._validate_ohlcv(df)
        signals = self._compute_ema_signals(df, fast, slow)

        # Build minimal feature_df with ATR for cost model
        feature_df = self._compute_atr_features(df)

        self._log.info(
            "backtest.ema_baseline.start",
            rows=len(df),
            fast=fast,
            slow=slow,
            nonzero_signals=(signals != 0).sum(),
        )
        return self.run(df, signals, feature_df)

    # ── Edge gate ─────────────────────────────────────────────────────────────

    def _check_edge_gate(self, report: BacktestReport) -> tuple[bool, list[str]]:
        """Evaluate all edge gate criteria against the BacktestReport.

        Args:
            report: Completed backtest report.

        Returns:
            Tuple of (passed: bool, failed_gates: list[str]).
        """
        failures: list[str] = []

        if report.sharpe_ratio < _GATE_MIN_SHARPE:
            failures.append(
                f"Sharpe {report.sharpe_ratio:.3f} < {_GATE_MIN_SHARPE} (walk-forward OOS)"
            )
        if report.profit_factor < _GATE_MIN_PROFIT_FACTOR:
            failures.append(
                f"Profit factor {report.profit_factor:.3f} < {_GATE_MIN_PROFIT_FACTOR}"
            )
        if report.win_rate < _GATE_MIN_WIN_RATE:
            failures.append(
                f"Win rate {report.win_rate:.1%} < {_GATE_MIN_WIN_RATE:.0%}"
            )

        # Regime gate: positive Sharpe in at least 2 of 3 regimes
        positive_regimes = sum(
            1
            for regime, metrics in report.regime_breakdown.items()
            if metrics.get("sharpe", -999) > 0
        )
        if positive_regimes < _GATE_MIN_POSITIVE_REGIMES:
            failures.append(
                f"Positive Sharpe in only {positive_regimes}/3 regimes "
                f"(need {_GATE_MIN_POSITIVE_REGIMES})"
            )

        # Per-fold minimum trades gate
        for fold_m in report.per_fold_metrics:
            fold_idx = fold_m.get("fold_index", "?")
            n = fold_m.get("n_trades", 0)
            if n < _GATE_MIN_TRADES_PER_FOLD:
                failures.append(
                    f"Fold {fold_idx}: only {n} trades (need {_GATE_MIN_TRADES_PER_FOLD})"
                )

        passed = len(failures) == 0
        return passed, failures

    # ── Simulation core ───────────────────────────────────────────────────────

    def _simulate_fold(
        self,
        ohlcv: pl.DataFrame,
        signals: pl.Series,
        features: pl.DataFrame,
        symbol: str,
        bar_offset: int,
    ) -> list[TradeResult]:
        """Simulate all trades for one walk-forward test fold.

        Implements 1-bar fill delay: signal at bar i → fill attempt at bar i+1.
        Stop-loss is market order.  Limit fills carry adverse selection penalty.
        Stale signals (price moved > max_price_chase_bps) are aborted.

        Args:
            ohlcv: OHLCV DataFrame for the test fold.
            signals: Signal Series aligned to the fold.
            features: Feature DataFrame aligned to the fold.
            symbol: Trading symbol string for cost model logging.
            bar_offset: Absolute row offset of this fold within the full df
                (used only for TradeResult.entry_bar / exit_bar).

        Returns:
            List of TradeResult for all completed trades in this fold.
        """
        n = len(ohlcv)
        trades: list[TradeResult] = []

        closes = ohlcv["close"].cast(pl.Float64).to_list()
        opens = ohlcv["open"].cast(pl.Float64).to_list()
        highs = ohlcv["high"].cast(pl.Float64).to_list()
        lows = ohlcv["low"].cast(pl.Float64).to_list()
        timestamps = ohlcv["timestamp"].to_list()

        # ATR column — fall back to close * 0.005 if absent
        if "feat_atr" in features.columns:
            atrs = features["feat_atr"].cast(pl.Float64).to_list()
        else:
            atrs = [c * 0.005 for c in closes]

        sigs = signals.cast(pl.Int8).to_list()

        # Session time limits (HH:MM parsed to (h, m) tuples)
        no_trade_hm = self._parse_hhmm(self._cfg.no_trade_before)
        sqoff_hm = self._parse_hhmm(self._cfg.square_off_time)

        cfg = self._cfg

        i = 0
        while i < n - 1:
            sig = sigs[i]
            if sig == 0:
                i += 1
                continue

            # [SURV-8] Skip signal before no_trade_before time
            ts_i = timestamps[i]
            if ts_i is not None and hasattr(ts_i, "hour"):
                bar_hm = (ts_i.hour, ts_i.minute)
                if bar_hm < no_trade_hm:
                    i += 1
                    continue

            signal_close = closes[i]
            atr_i = atrs[i] if atrs[i] is not None and not math.isnan(float(atrs[i])) else signal_close * 0.005
            direction = int(sig)

            # 1-bar fill delay: fill at bar i+1 open
            fill_bar = i + 1
            fill_open = opens[fill_bar]

            # Stale signal abort: price moved > max_price_chase_bps from signal close
            if signal_close > 0:
                price_move_bps = abs(fill_open - signal_close) / signal_close * 10_000
                if price_move_bps > cfg.max_price_chase_bps:
                    self._log.debug(
                        "backtest.stale_signal_aborted",
                        bar=i,
                        move_bps=price_move_bps,
                    )
                    i += 1
                    continue

            # Apply adverse selection penalty to entry fill (LIMIT order assumption)
            adv_sel_mult = 1.0 + direction * (cfg.adverse_selection_bps / 10_000)
            entry_price = Decimal(str(fill_open)) * Decimal(str(adv_sel_mult))
            entry_price = entry_price.quantize(Decimal("0.01"), ROUND_HALF_UP)

            # Position size: max_position_pct of initial capital
            capital = cfg.initial_capital
            max_invest = capital * Decimal(str(cfg.max_position_pct))
            qty = max(1, int(max_invest / entry_price))

            # Stop-loss level
            atr_dec = Decimal(str(round(atr_i, 4)))
            stop_mult = Decimal(str(cfg.stop_loss_atr_mult))
            if direction > 0:
                stop_price = entry_price - atr_dec * stop_mult
            else:
                stop_price = entry_price + atr_dec * stop_mult

            # Entry cost (BUY for long, SELL for short)
            entry_side = "BUY" if direction > 0 else "SELL"
            fill_hour = timestamps[fill_bar].hour if hasattr(timestamps[fill_bar], "hour") else None
            entry_cost = self._cost_model.compute(
                symbol=symbol,
                side=entry_side,
                quantity=qty,
                price=entry_price,
                atr=atr_dec,
                hour_of_day=fill_hour,
            )

            # Seek exit bar
            exit_bar = fill_bar + 1
            exit_reason = "force_close"
            exit_price = Decimal(str(closes[min(fill_bar, n - 1)]))

            while exit_bar < n:
                ts_exit = timestamps[exit_bar]
                exit_hm = (ts_exit.hour, ts_exit.minute) if hasattr(ts_exit, "hour") else (0, 0)

                # [SURV-8] Square-off deadline
                if exit_hm >= sqoff_hm:
                    exit_price = Decimal(str(opens[exit_bar]))
                    exit_reason = "time_exit"
                    break

                cur_high = highs[exit_bar]
                cur_low = lows[exit_bar]
                cur_open = opens[exit_bar]

                # Stop-loss hit check (MARKET order at next bar open)
                if direction > 0 and cur_low <= float(stop_price):
                    exit_price = Decimal(str(cur_open))  # market fill next open
                    exit_reason = "stop"
                    break
                if direction < 0 and cur_high >= float(stop_price):
                    exit_price = Decimal(str(cur_open))
                    exit_reason = "stop"
                    break

                # Force-close after max_holding_bars
                bars_held = exit_bar - fill_bar
                if bars_held >= cfg.max_holding_bars:
                    exit_price = Decimal(str(opens[exit_bar]))
                    exit_reason = "force_close"
                    break

                exit_bar += 1
            else:
                # Reached end of fold
                exit_price = Decimal(str(closes[n - 1]))
                exit_reason = "force_close"
                exit_bar = n - 1

            # Exit cost — stop is MARKET (full slippage), otherwise LIMIT (adv sel)
            exit_side = "SELL" if direction > 0 else "BUY"
            exit_atr = atr_dec  # use entry ATR as proxy
            if exit_reason == "stop":
                # MARKET order: no adverse selection, but full ATR slippage
                exit_cost = self._cost_model.compute(
                    symbol=symbol,
                    side=exit_side,
                    quantity=qty,
                    price=exit_price,
                    atr=exit_atr,
                    hour_of_day=fill_hour,
                )
            else:
                # LIMIT order: apply adverse selection penalty
                adv_exit_mult = 1.0 - direction * (cfg.adverse_selection_bps / 10_000)
                exit_price_adj = exit_price * Decimal(str(adv_exit_mult))
                exit_price_adj = exit_price_adj.quantize(Decimal("0.01"), ROUND_HALF_UP)
                exit_cost = self._cost_model.compute(
                    symbol=symbol,
                    side=exit_side,
                    quantity=qty,
                    price=exit_price_adj,
                    atr=exit_atr,
                    hour_of_day=fill_hour,
                )
                exit_price = exit_price_adj

            # P&L
            qty_dec = Decimal(str(qty))
            gross_pnl = direction * (exit_price - entry_price) * qty_dec
            total_cost = entry_cost.total_inr + exit_cost.total_inr
            net_pnl = gross_pnl - total_cost

            # Combined cost breakdown for TradeResult
            combined_cost = self._cost_model.compute_round_trip(
                symbol=symbol,
                quantity=qty,
                entry_price=entry_price,
                exit_price=exit_price,
                atr=atr_dec,
                direction=direction,
            )

            trade = TradeResult(
                symbol=symbol,
                entry_bar=bar_offset + fill_bar,
                exit_bar=bar_offset + exit_bar,
                entry_timestamp=timestamps[fill_bar],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                gross_pnl=gross_pnl.quantize(Decimal("0.01"), ROUND_HALF_UP),
                net_pnl=net_pnl.quantize(Decimal("0.01"), ROUND_HALF_UP),
                cost_breakdown=combined_cost,
                exit_reason=exit_reason,
                signal_confidence=1.0,  # placeholder; override in ML path
            )
            trades.append(trade)

            # Advance past the exit bar to avoid overlapping trades
            i = exit_bar + 1

        return trades

    # ── Report builders ───────────────────────────────────────────────────────

    def _build_report(
        self,
        all_trades: list[TradeResult],
        per_fold_metrics: list[dict],
        regime_breakdown: dict[str, dict],
    ) -> BacktestReport:
        """Assemble the final BacktestReport from all collected data.

        Args:
            all_trades: All TradeResult objects across all folds.
            per_fold_metrics: List of per-fold metric dicts.
            regime_breakdown: Per-regime metric dicts.

        Returns:
            Fully populated BacktestReport.
        """
        n = len(all_trades)
        if n == 0:
            report = BacktestReport(
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_net_pnl=Decimal("0"),
                per_fold_metrics=per_fold_metrics,
                regime_breakdown=regime_breakdown,
                cost_analysis={"total_cost": Decimal("0"), "avg_cost_bps": 0.0, "slippage_pct": 0.0},
                per_symbol_metrics={},
                passes_edge_gate=False,
                gate_failures=["No trades executed"],
            )
            return report

        wins = sum(1 for t in all_trades if t.net_pnl > 0)
        win_rate = wins / n

        # P0-FIX-9: profit_factor must use net P&L so edge gate rejects
        # strategies whose gross edge is eaten by costs.
        net_profit = sum(float(t.net_pnl) for t in all_trades if t.net_pnl > 0)
        net_loss = abs(sum(float(t.net_pnl) for t in all_trades if t.net_pnl < 0))
        profit_factor = net_profit / net_loss if net_loss > 0 else float("inf")

        returns = [float(t.net_pnl) / float(self._cfg.initial_capital) for t in all_trades]
        sharpe = self._compute_sharpe(returns)

        max_dd = self._compute_max_drawdown(returns)

        total_net_pnl = sum((t.net_pnl for t in all_trades), Decimal("0"))

        total_cost = sum((t.cost_breakdown.total_inr for t in all_trades), Decimal("0"))
        total_turnover = sum((t.cost_breakdown.turnover for t in all_trades), Decimal("0"))
        avg_cost_bps = float(
            (total_cost / total_turnover * 10_000) if total_turnover > 0 else Decimal("0")
        )
        total_slip = sum(float(t.cost_breakdown.slippage_inr) for t in all_trades)
        slippage_pct = float(total_slip) / float(total_cost) if float(total_cost) > 0 else 0.0

        cost_analysis = {
            "total_cost": total_cost,
            "avg_cost_bps": avg_cost_bps,
            "slippage_pct": slippage_pct,
        }

        # Per-symbol PnL attribution: Sharpe, profit factor, win rate
        per_symbol_metrics: dict[str, dict] = {}
        symbols_seen: set[str] = {t.symbol for t in all_trades}
        for sym in symbols_seen:
            sym_trades = [t for t in all_trades if t.symbol == sym]
            sym_wins = sum(1 for t in sym_trades if t.net_pnl > 0)
            sym_n = len(sym_trades)
            sym_returns = [float(t.net_pnl) / float(self._cfg.initial_capital) for t in sym_trades]
            # P0-FIX-9: use net P&L for profit_factor
            sym_np = sum(float(t.net_pnl) for t in sym_trades if t.net_pnl > 0)
            sym_nl = abs(sum(float(t.net_pnl) for t in sym_trades if t.net_pnl < 0))
            per_symbol_metrics[sym] = {
                "n_trades": sym_n,
                "win_rate": sym_wins / sym_n if sym_n > 0 else 0.0,
                "sharpe": self._compute_sharpe(sym_returns),
                "profit_factor": sym_np / sym_nl if sym_nl > 0 else float("inf"),
                "net_pnl": sum((t.net_pnl for t in sym_trades), Decimal("0")),
            }

        report = BacktestReport(
            total_trades=n,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_net_pnl=total_net_pnl,
            per_fold_metrics=per_fold_metrics,
            regime_breakdown=regime_breakdown,
            cost_analysis=cost_analysis,
            per_symbol_metrics=per_symbol_metrics,
            passes_edge_gate=False,  # set below
            gate_failures=[],
        )

        passed, failures = self._check_edge_gate(report)
        report.passes_edge_gate = passed
        report.gate_failures = failures
        return report

    def _compute_fold_metrics(
        self, trades: list[TradeResult], fold_index: int
    ) -> dict:
        """Compute summary metrics for a single fold.

        Args:
            trades: All TradeResult objects for the fold.
            fold_index: Zero-based fold number.

        Returns:
            Dict with fold_index, n_trades, net_pnl, win_rate, sharpe, profit_factor.
        """
        n = len(trades)
        if n == 0:
            return {
                "fold_index": fold_index,
                "n_trades": 0,
                "net_pnl": Decimal("0"),
                "win_rate": 0.0,
                "sharpe": 0.0,
                "profit_factor": 0.0,
            }

        wins = sum(1 for t in trades if t.net_pnl > 0)
        returns = [float(t.net_pnl) / float(self._cfg.initial_capital) for t in trades]
        # P0-FIX-9: net P&L for profit_factor (consistent with Sharpe/win_rate)
        np_ = sum(float(t.net_pnl) for t in trades if t.net_pnl > 0)
        nl_ = abs(sum(float(t.net_pnl) for t in trades if t.net_pnl < 0))

        return {
            "fold_index": fold_index,
            "n_trades": n,
            "net_pnl": sum((t.net_pnl for t in trades), Decimal("0")),
            "win_rate": wins / n,
            "sharpe": self._compute_sharpe(returns),
            "profit_factor": np_ / nl_ if nl_ > 0 else float("inf"),
        }

    # ── Regime utilities ──────────────────────────────────────────────────────

    def _classify_regimes(self, df: pl.DataFrame) -> list[str]:
        """Classify each bar into bull/bear/sideways using 50-bar EMA slope.

        Args:
            df: Full OHLCV DataFrame.

        Returns:
            List of regime strings (length == len(df)).
        """
        closes = df["close"].cast(pl.Float64).to_list()
        n = len(closes)
        regimes: list[str] = ["sideways"] * n

        alpha = 2.0 / (_REGIME_EMA_PERIOD + 1)
        ema = closes[0] if closes else 0.0
        prev_ema = ema

        for i in range(n):
            prev_ema = ema
            ema = alpha * closes[i] + (1 - alpha) * ema
            if i < _REGIME_EMA_PERIOD:
                regimes[i] = "sideways"
                continue
            slope = (ema - prev_ema) / prev_ema if prev_ema != 0 else 0.0
            if slope > _REGIME_FLAT_THRESHOLD:
                regimes[i] = "bull"
            elif slope < -_REGIME_FLAT_THRESHOLD:
                regimes[i] = "bear"
            else:
                regimes[i] = "sideways"

        return regimes

    def _compute_regime_breakdown(
        self,
        trades: list[TradeResult],
        regimes: list[str],
    ) -> dict[str, dict]:
        """Bucket trades by entry-bar regime and compute per-regime metrics.

        Args:
            trades: All TradeResult objects.
            regimes: Full list of regime labels (one per bar in the full df).

        Returns:
            Dict mapping regime name → metrics dict.
        """
        bucketed: dict[str, list[TradeResult]] = {"bull": [], "bear": [], "sideways": []}

        for trade in trades:
            entry_idx = trade.entry_bar
            if 0 <= entry_idx < len(regimes):
                regime = regimes[entry_idx]
                bucketed[regime].append(trade)

        breakdown: dict[str, dict] = {}
        for regime, bucket_trades in bucketed.items():
            breakdown[regime] = self._compute_fold_metrics(bucket_trades, fold_index=-1)
            breakdown[regime].pop("fold_index", None)

        return breakdown

    # ── Signal generators (EMA baseline) ─────────────────────────────────────

    def _compute_ema_signals(
        self, df: pl.DataFrame, fast: int, slow: int
    ) -> pl.Series:
        """Compute EMA(fast)/EMA(slow) crossover signals for the baseline.

        Signals: +1 on bullish cross, -1 on bearish cross, 0 otherwise.
        No signal is generated before bar ``slow`` (warmup period).

        Args:
            df: OHLCV DataFrame with a close column.
            fast: Fast EMA period.
            slow: Slow EMA period.

        Returns:
            Int8 Polars Series of {-1, 0, +1}.
        """
        closes = df["close"].cast(pl.Float64).to_list()
        n = len(closes)
        sigs: list[int] = [0] * n

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)

        ema_f = closes[0]
        ema_s = closes[0]
        prev_f = ema_f
        prev_s = ema_s

        for i in range(n):
            prev_f = ema_f
            prev_s = ema_s
            ema_f = alpha_fast * closes[i] + (1 - alpha_fast) * ema_f
            ema_s = alpha_slow * closes[i] + (1 - alpha_slow) * ema_s

            if i < slow:
                continue

            bullish = prev_f <= prev_s and ema_f > ema_s
            bearish = prev_f >= prev_s and ema_f < ema_s
            if bullish:
                sigs[i] = 1
            elif bearish:
                sigs[i] = -1

        return pl.Series("signal", sigs, dtype=pl.Int8)

    def _compute_atr_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute a minimal feature DataFrame with ATR(14) for the cost model.

        Falls back to close * 0.5% if there is insufficient data for ATR.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Polars DataFrame with a single feat_atr column.
        """
        closes = df["close"].cast(pl.Float64).to_list()
        highs = df["high"].cast(pl.Float64).to_list()
        lows = df["low"].cast(pl.Float64).to_list()
        n = len(closes)
        atr_period = 14
        atrs: list[float] = [closes[0] * 0.005] * n

        if n < 2:
            return pl.DataFrame({"feat_atr": atrs})

        true_ranges: list[float] = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)
            if len(true_ranges) < atr_period:
                atrs[i] = sum(true_ranges) / len(true_ranges)
            else:
                atrs[i] = (atrs[i - 1] * (atr_period - 1) + tr) / atr_period

        return pl.DataFrame({"feat_atr": atrs})

    # ── Statistics helpers ────────────────────────────────────────────────────

    def _compute_sharpe(self, returns: list[float]) -> float:
        """Compute annualised Sharpe ratio from a list of per-trade returns.

        Uses risk-free rate of zero (appropriate for short-horizon intraday).
        Annualises by the square-root of trades-per-year proxy.

        Args:
            returns: List of fractional returns per trade.

        Returns:
            Annualised Sharpe ratio, or 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        mu = statistics.mean(returns)
        sigma = statistics.stdev(returns)
        if sigma == 0.0:
            return 0.0
        # Approximate annualisation: trades/year ≈ bars_per_day × trading_days / avg_hold
        trades_per_year = _BARS_PER_DAY * _TRADING_DAYS_PER_YEAR
        annual_factor = math.sqrt(trades_per_year)
        return (mu / sigma) * annual_factor

    def _compute_max_drawdown(self, returns: list[float]) -> float:
        """Compute maximum drawdown as a fraction of peak equity.

        Args:
            returns: List of fractional per-trade returns.

        Returns:
            Maximum drawdown (positive fraction, e.g. 0.15 for 15% drawdown).
        """
        if not returns:
            return 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= (1.0 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate_inputs(
        self,
        df: pl.DataFrame,
        signals: pl.Series,
        feature_df: pl.DataFrame,
    ) -> None:
        """Validate that inputs are aligned and have required columns.

        Args:
            df: OHLCV DataFrame.
            signals: Signal Series.
            feature_df: Feature DataFrame.

        Raises:
            ValueError: On any mismatch or missing column.
        """
        self._validate_ohlcv(df)
        if len(df) != len(signals):
            raise ValueError(
                f"df ({len(df)} rows) and signals ({len(signals)}) must have equal length."
            )
        if len(df) != len(feature_df):
            raise ValueError(
                f"df ({len(df)} rows) and feature_df ({len(feature_df)}) must have equal length."
            )

    @staticmethod
    def _validate_ohlcv(df: pl.DataFrame) -> None:
        """Raise ValueError if required OHLCV columns are absent or data is invalid.

        Checks for: missing columns, NaN values, inf values, high < low,
        negative prices, and zero volume on price-bearing rows.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If any required column is missing or data is invalid.
        """
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            series = df[col].cast(pl.Float64)
            if series.is_nan().any():
                raise ValueError(f"OHLCV column '{col}' contains NaN values")
            if series.is_infinite().any():
                raise ValueError(f"OHLCV column '{col}' contains inf values")
            if (series < 0).any():
                raise ValueError(f"OHLCV column '{col}' contains negative values")

        vol_series = df["volume"].cast(pl.Float64)
        if vol_series.is_nan().any():
            raise ValueError("OHLCV column 'volume' contains NaN values")
        if vol_series.is_infinite().any():
            raise ValueError("OHLCV column 'volume' contains inf values")

        high_series = df["high"].cast(pl.Float64)
        low_series = df["low"].cast(pl.Float64)
        if (high_series < low_series).any():
            raise ValueError("OHLCV data contains bars where high < low")

        if (vol_series == 0).any():
            n_zero = int((vol_series == 0).sum())
            raise ValueError(
                f"OHLCV data contains {n_zero} bar(s) with zero volume — "
                "filter out illiquid bars before backtesting"
            )

    @staticmethod
    def _parse_hhmm(hhmm: str) -> tuple[int, int]:
        """Parse a HH:MM string into a (hour, minute) tuple.

        Args:
            hhmm: Time string in HH:MM format (e.g. "09:30").

        Returns:
            Tuple of (hour, minute) as integers.

        Raises:
            ValueError: If the format is invalid.
        """
        parts = hhmm.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected HH:MM, got {hhmm!r}")
        return int(parts[0]), int(parts[1])
