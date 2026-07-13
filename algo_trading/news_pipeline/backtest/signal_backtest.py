"""
SignalBacktester: evaluates how well FusedSignals predicted actual price direction.

For each signal:
  - BUY/STRONG_BUY: predicted price would go UP
  - SELL/STRONG_SELL: predicted price would go DOWN
  - NEUTRAL: skip (no directional bet)

Evaluation:
  - entry_price = close price on signal date
  - exit_price  = close price on signal date + 1 trading day
  - hit = True if actual direction matches predicted direction
  - return_pct = (exit - entry) / entry * 100

  If no price data available for a signal, skip and log.

Aggregate metrics:
  - total_signals: int
  - neutral_skipped: int
  - evaluated: int
  - wins: int
  - losses: int
  - win_rate: float  (wins / evaluated)
  - avg_return_pct: float
  - avg_win_pct: float
  - avg_loss_pct: float
  - sharpe_ratio: float  (annualized: mean / std * sqrt(252))
  - max_drawdown: float
  - by_signal_type: dict[str, dict]  — metrics by STRONG_BUY / BUY / SELL / STRONG_SELL
  - by_symbol: dict[str, dict]       — metrics by symbol
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import structlog

from algo_trading.news_pipeline.models import SignalPerformance, make_id
from algo_trading.news_pipeline.config import cfg

if TYPE_CHECKING:
    import pandas as pd

log = structlog.get_logger(__name__)

# Signal types that imply a directional bet
_BULL_SIGNALS = frozenset({"STRONG_BUY", "BUY"})
_BEAR_SIGNALS = frozenset({"STRONG_SELL", "SELL"})
_DIRECTIONAL   = _BULL_SIGNALS | _BEAR_SIGNALS

# NSE market hours in India are 9:15–15:30, i.e. 6.25 h per session
_NSE_HOURS_PER_DAY: float = 6.25

# ─────────────────────────────────────────────────────────────────────────────
# Helper: empty aggregate skeleton
# ─────────────────────────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return {
        "total_signals": 0,
        "neutral_skipped": 0,
        "evaluated": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_return_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "by_signal_type": {},
        "by_symbol": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SignalBacktester:
    """Evaluate FusedSignal accuracy against next-day NSE close prices.

    Parameters
    ----------
    pg_store:
        :class:`~algo_trading.news_pipeline.storage.pg_store.PGStore` instance
        (connected) used to read signals and write performance records.
    parquet_store:
        :class:`~algo_trading.news_pipeline.storage.parquet_store.ParquetStore`
        instance used to read OHLCV parquet files.
    """

    def __init__(self, pg_store, parquet_store) -> None:
        self._pg = pg_store
        self._parquet = parquet_store

    # ─────────────────────────────────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────────────────────────────────

    async def evaluate_yesterday(self) -> dict:
        """Evaluate signals generated yesterday against today's price moves.

        Shorthand wrapper around :meth:`evaluate_date`.
        """
        yesterday = date.today() - timedelta(days=1)
        return await self.evaluate_date(yesterday)

    async def evaluate_date(self, signal_date: date) -> dict:
        """Evaluate all directional signals generated on *signal_date*.

        Steps
        -----
        1. Load all signals from PG via ``get_latest_signals()`` and filter
           those whose ``generated_at`` date matches *signal_date*.
        2. For each unique symbol load OHLCV for *signal_date* **and**
           *signal_date* + 1 calendar day from Parquet.
        3. Compute entry/exit prices and hit/return for each signal.
        4. Persist :class:`SignalPerformance` records to PG.
        5. Return aggregate metrics dict.

        Returns a safe empty-metrics dict on any unhandled error.
        """
        try:
            # ── 1. Pull signals and filter to the requested date ──────────
            all_signals: list[dict] = await self._pg.get_latest_signals(limit=500)
            signals_on_date = [
                s for s in all_signals
                if self._signal_date(s) == signal_date
            ]

            if not signals_on_date:
                log.info(
                    "backtest.evaluate_date.no_signals",
                    signal_date=signal_date.isoformat(),
                )
                return _empty_metrics()

            # ── 2. Pre-load OHLCV for all unique symbols ──────────────────
            exit_date = signal_date + timedelta(days=1)
            symbols = list({s["symbol"] for s in signals_on_date})
            ohlcv_cache: dict[str, "pd.DataFrame"] = {}
            for sym in symbols:
                try:
                    df = self._parquet.read_ohlcv(sym, signal_date, exit_date)
                    if not df.empty:
                        ohlcv_cache[sym] = df
                except Exception as exc:
                    log.warning(
                        "backtest.evaluate_date.ohlcv_load_failed",
                        symbol=sym,
                        signal_date=signal_date.isoformat(),
                        error=str(exc),
                    )

            # ── 3. Compute per-signal performance ─────────────────────────
            performances: list[SignalPerformance] = []
            total_signals = len(signals_on_date)
            neutral_skipped = 0

            for signal in signals_on_date:
                signal_type = signal.get("signal_type", "NEUTRAL")

                # Skip non-directional signals
                if signal_type not in _DIRECTIONAL:
                    neutral_skipped += 1
                    continue

                sym = signal["symbol"]
                entry_price = self._get_ohlcv_close(sym, signal_date, ohlcv_cache)
                exit_price  = self._get_ohlcv_close(sym, exit_date,   ohlcv_cache)

                if entry_price is None or exit_price is None:
                    log.warning(
                        "backtest.evaluate_date.missing_price",
                        symbol=sym,
                        signal_date=signal_date.isoformat(),
                        exit_date=exit_date.isoformat(),
                        entry_price_ok=(entry_price is not None),
                        exit_price_ok=(exit_price is not None),
                    )
                    neutral_skipped += 1   # count as skipped, not a failed bet
                    continue

                perf = self._build_performance_record(
                    signal, entry_price, exit_price, signal_date
                )
                performances.append(perf)

            # ── 4. Persist to PG ──────────────────────────────────────────
            for perf in performances:
                try:
                    await self._pg.insert_signal_performance(perf)
                except Exception as exc:
                    log.error(
                        "backtest.evaluate_date.insert_failed",
                        perf_id=perf.id,
                        error=str(exc),
                    )

            # ── 5. Aggregate and return ───────────────────────────────────
            metrics = self._aggregate_metrics(performances)
            metrics["total_signals"]   = total_signals
            metrics["neutral_skipped"] = neutral_skipped

            log.info(
                "backtest.evaluate_date.done",
                signal_date=signal_date.isoformat(),
                evaluated=metrics["evaluated"],
                win_rate=round(metrics["win_rate"], 4),
            )
            return metrics

        except Exception as exc:
            log.error(
                "backtest.evaluate_date.unhandled_error",
                signal_date=str(signal_date),
                error=str(exc),
            )
            return _empty_metrics()

    async def evaluate_range(self, start: date, end: date) -> dict:
        """Evaluate signals over a date range [*start*, *end*] inclusive.

        Calls :meth:`evaluate_date` for each calendar date and aggregates the
        daily metrics into a single combined result.

        Returns a safe empty-metrics dict on unhandled error.
        """
        try:
            if start > end:
                log.warning(
                    "backtest.evaluate_range.invalid_range",
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
                return _empty_metrics()

            combined_performances: list[SignalPerformance] = []
            total_signals_all   = 0
            neutral_skipped_all = 0

            current = start
            while current <= end:
                daily = await self.evaluate_date(current)
                total_signals_all   += daily.get("total_signals", 0)
                neutral_skipped_all += daily.get("neutral_skipped", 0)
                # We can't reconstruct raw SignalPerformance objects from the
                # returned dict, so pull them from PG for the aggregation.
                current += timedelta(days=1)

            # Re-pull everything from PG for accurate aggregate stats
            # (avoids holding potentially large lists in memory per day)
            all_perfs = await self._fetch_performances_range(start, end)

            metrics = self._aggregate_metrics(all_perfs)
            metrics["total_signals"]   = total_signals_all
            metrics["neutral_skipped"] = neutral_skipped_all

            log.info(
                "backtest.evaluate_range.done",
                start=start.isoformat(),
                end=end.isoformat(),
                evaluated=metrics["evaluated"],
                win_rate=round(metrics["win_rate"], 4),
            )
            return metrics

        except Exception as exc:
            log.error(
                "backtest.evaluate_range.unhandled_error",
                start=str(start),
                end=str(end),
                error=str(exc),
            )
            return _empty_metrics()

    async def summary_report(self, days: int = 30) -> dict:
        """Pull signal_performance records from PG for the last *days* days
        and compute a full aggregate report.

        Returns
        -------
        dict with keys:
            period_days, total_evaluated, win_rate, avg_return_pct,
            sharpe_ratio, max_drawdown,
            by_signal_type  (STRONG_BUY / BUY / SELL / STRONG_SELL),
            by_symbol,
            best_signal_type, best_symbol,
            recent_10
        """
        _empty_report: dict = {
            "period_days": days,
            "total_evaluated": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "by_signal_type": {},
            "by_symbol": {},
            "best_signal_type": "",
            "best_symbol": "",
            "recent_10": [],
        }

        try:
            end   = date.today()
            start = end - timedelta(days=days)
            perfs = await self._fetch_performances_range(start, end)

            if not perfs:
                log.info("backtest.summary_report.no_data", days=days)
                return _empty_report

            metrics = self._aggregate_metrics(perfs)

            # Best signal type by win_rate (ties broken by count)
            by_type = metrics.get("by_signal_type", {})
            best_signal_type = ""
            if by_type:
                best_signal_type = max(
                    by_type,
                    key=lambda k: (by_type[k]["win_rate"], by_type[k]["count"]),
                )

            # Best symbol by win_rate (ties broken by count)
            by_sym = metrics.get("by_symbol", {})
            best_symbol = ""
            if by_sym:
                best_symbol = max(
                    by_sym,
                    key=lambda k: (by_sym[k]["win_rate"], by_sym[k]["count"]),
                )

            # Most recent 10 records sorted by evaluation_date desc
            sorted_perfs = sorted(
                perfs,
                key=lambda p: p.evaluation_date
                if isinstance(p.evaluation_date, datetime)
                else datetime.combine(
                    p.evaluation_date
                    if isinstance(p.evaluation_date, date)
                    else date.today(),
                    datetime.min.time(),
                ),
                reverse=True,
            )
            recent_10 = [p.to_dict() for p in sorted_perfs[:10]]

            report = {
                "period_days":      days,
                "total_evaluated":  metrics["evaluated"],
                "win_rate":         metrics["win_rate"],
                "avg_return_pct":   metrics["avg_return_pct"],
                "sharpe_ratio":     metrics["sharpe_ratio"],
                "max_drawdown":     metrics["max_drawdown"],
                "by_signal_type":   {
                    st: {
                        "count":      v["count"],
                        "win_rate":   v["win_rate"],
                        "avg_return": v["avg_return_pct"],
                    }
                    for st, v in by_type.items()
                },
                "by_symbol": {
                    sym: {
                        "count":      v["count"],
                        "win_rate":   v["win_rate"],
                        "avg_return": v["avg_return_pct"],
                    }
                    for sym, v in by_sym.items()
                },
                "best_signal_type": best_signal_type,
                "best_symbol":      best_symbol,
                "recent_10":        recent_10,
            }

            log.info(
                "backtest.summary_report.done",
                days=days,
                evaluated=metrics["evaluated"],
                win_rate=round(metrics["win_rate"], 4),
            )
            return report

        except Exception as exc:
            log.error(
                "backtest.summary_report.unhandled_error",
                days=days,
                error=str(exc),
            )
            return _empty_report

    # ─────────────────────────────────────────────────────────────────────────
    # Statistical helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sharpe(self, returns: list[float]) -> float:
        """Annualized Sharpe ratio from a list of per-trade return percentages.

        ``sharpe = mean(returns) / stdev(returns) * sqrt(252)``

        Returns 0.0 when fewer than 2 data points or standard deviation is 0.
        """
        if len(returns) < 2:
            return 0.0
        try:
            mean_r = statistics.mean(returns)
            std_r  = statistics.stdev(returns)
            if std_r == 0.0:
                return 0.0
            return round(mean_r / std_r * math.sqrt(252), 6)
        except Exception:
            return 0.0

    def _compute_max_drawdown(self, returns: list[float]) -> float:
        """Maximum drawdown from a cumulative return (equity) series.

        Equity curve: ``equity[i] = product(1 + r/100  for r in returns[:i+1])``
        Drawdown at i: ``(peak_so_far - equity[i]) / peak_so_far``

        Returns 0.0 when fewer than 2 data points.
        """
        if len(returns) < 2:
            return 0.0
        try:
            equity: list[float] = []
            running = 1.0
            for r in returns:
                running *= 1.0 + r / 100.0
                equity.append(running)

            max_dd   = 0.0
            peak     = equity[0]
            for val in equity:
                if val > peak:
                    peak = val
                if peak > 0.0:
                    dd = (peak - val) / peak
                    if dd > max_dd:
                        max_dd = dd

            return round(max_dd, 6)
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Price lookup
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ohlcv_close(
        self,
        symbol: str,
        trade_date: date,
        ohlcv_cache: "dict[str, pd.DataFrame]",
    ) -> Optional[float]:
        """Return closing price for *symbol* on *trade_date* from the cache.

        The cache stores DataFrames keyed by symbol.  Each DataFrame may span
        multiple dates.  We look for a row whose ``date`` column (stored as a
        string ``YYYY-MM-DD``) matches *trade_date*.

        Returns ``None`` if the symbol or date is absent.
        """
        df = ohlcv_cache.get(symbol)
        if df is None or df.empty:
            return None

        date_str = trade_date.isoformat()   # "YYYY-MM-DD"

        # The "date" column may be stored as strings, dates, or datetimes
        if "date" not in df.columns:
            return None

        # Normalise to string comparison — fast and avoids dtype issues
        try:
            mask = df["date"].astype(str).str[:10] == date_str
            matched = df.loc[mask, "close"]
            if matched.empty:
                return None
            return float(matched.iloc[0])
        except Exception as exc:
            log.warning(
                "backtest.get_ohlcv_close.failed",
                symbol=symbol,
                trade_date=date_str,
                error=str(exc),
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Record construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_performance_record(
        self,
        signal: dict,
        entry_price: float,
        exit_price: float,
        eval_date: date,
    ) -> SignalPerformance:
        """Build a :class:`SignalPerformance` from a signal dict and prices.

        Hit logic
        ---------
        * STRONG_BUY / BUY  → hit = (exit_price > entry_price)
        * STRONG_SELL / SELL → hit = (exit_price < entry_price)

        Return pct
        ----------
        * BUY direction  : ``(exit - entry) / entry * 100``
        * SELL direction : ``(entry - exit) / entry * 100``  (profit when short)
        """
        signal_type = signal.get("signal_type", "BUY")
        is_bull     = signal_type in _BULL_SIGNALS

        if is_bull:
            hit        = exit_price > entry_price
            return_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            # SELL / STRONG_SELL — positive when price falls (short profit)
            hit        = exit_price < entry_price
            return_pct = (entry_price - exit_price) / entry_price * 100.0

        # Normalise generated_at to a datetime
        gen_at_raw = signal.get("generated_at")
        if isinstance(gen_at_raw, str):
            signal_generated_at = datetime.fromisoformat(gen_at_raw)
        elif isinstance(gen_at_raw, datetime):
            signal_generated_at = gen_at_raw
        else:
            # Fallback: use eval_date at market open
            signal_generated_at = datetime.combine(eval_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            )

        # Holding time = 1 NSE trading session
        holding_hours = _NSE_HOURS_PER_DAY

        perf_id = make_id(
            f"{signal.get('symbol', '')}|{signal_generated_at.isoformat()}|{eval_date.isoformat()}"
        )

        eval_dt = datetime.combine(eval_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )

        return SignalPerformance(
            id=perf_id,
            evaluation_date=eval_dt,
            symbol=signal.get("symbol", "UNKNOWN"),
            signal_type=signal_type,
            signal_generated_at=signal_generated_at,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            return_pct=round(return_pct, 6),
            hit=hit,
            holding_hours=holding_hours,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _aggregate_metrics(self, performances: list[SignalPerformance]) -> dict:
        """Compute all aggregate metrics from a list of :class:`SignalPerformance`.

        Handles empty lists gracefully by returning an all-zeros dict.
        """
        if not performances:
            return _empty_metrics()

        returns_all: list[float] = [p.return_pct for p in performances]
        wins   = [p for p in performances if p.hit]
        losses = [p for p in performances if not p.hit]

        win_returns  = [p.return_pct for p in wins]
        loss_returns = [p.return_pct for p in losses]

        evaluated    = len(performances)
        win_count    = len(wins)
        loss_count   = len(losses)
        win_rate     = win_count / evaluated if evaluated > 0 else 0.0
        avg_return   = statistics.mean(returns_all) if returns_all else 0.0
        avg_win      = statistics.mean(win_returns)  if win_returns  else 0.0
        avg_loss     = statistics.mean(loss_returns) if loss_returns else 0.0
        sharpe       = self._compute_sharpe(returns_all)
        max_dd       = self._compute_max_drawdown(returns_all)

        # ── By signal type ────────────────────────────────────────────────
        by_signal_type: dict[str, dict] = {}
        for stype in ("STRONG_BUY", "BUY", "SELL", "STRONG_SELL"):
            group = [p for p in performances if p.signal_type == stype]
            if not group:
                continue
            g_returns = [p.return_pct for p in group]
            g_wins    = sum(1 for p in group if p.hit)
            by_signal_type[stype] = {
                "count":         len(group),
                "wins":          g_wins,
                "losses":        len(group) - g_wins,
                "win_rate":      round(g_wins / len(group), 6),
                "avg_return_pct": round(statistics.mean(g_returns), 6),
                "sharpe_ratio":  self._compute_sharpe(g_returns),
                "max_drawdown":  self._compute_max_drawdown(g_returns),
            }

        # ── By symbol ─────────────────────────────────────────────────────
        by_symbol: dict[str, dict] = {}
        all_symbols = {p.symbol for p in performances}
        for sym in sorted(all_symbols):
            group = [p for p in performances if p.symbol == sym]
            g_returns = [p.return_pct for p in group]
            g_wins    = sum(1 for p in group if p.hit)
            by_symbol[sym] = {
                "count":         len(group),
                "wins":          g_wins,
                "losses":        len(group) - g_wins,
                "win_rate":      round(g_wins / len(group), 6),
                "avg_return_pct": round(statistics.mean(g_returns), 6),
                "sharpe_ratio":  self._compute_sharpe(g_returns),
                "max_drawdown":  self._compute_max_drawdown(g_returns),
            }

        return {
            # These two are overwritten by the caller with the full counts
            # (including neutral/skipped) so we set them to 0 here.
            "total_signals":   0,
            "neutral_skipped": 0,
            # Evaluated subset only
            "evaluated":       evaluated,
            "wins":            win_count,
            "losses":          loss_count,
            "win_rate":        round(win_rate, 6),
            "avg_return_pct":  round(avg_return, 6),
            "avg_win_pct":     round(avg_win, 6),
            "avg_loss_pct":    round(avg_loss, 6),
            "sharpe_ratio":    sharpe,
            "max_drawdown":    max_dd,
            "by_signal_type":  by_signal_type,
            "by_symbol":       by_symbol,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _signal_date(signal: dict) -> date:
        """Extract the calendar date from a signal's ``generated_at`` field."""
        raw = signal.get("generated_at")
        if isinstance(raw, datetime):
            return raw.date()
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw).date()
            except ValueError:
                pass
        return date.today()

    async def _fetch_performances_range(
        self, start: date, end: date
    ) -> list[SignalPerformance]:
        """Fetch signal_performance rows from PG between *start* and *end*.

        Uses PGStore's ``get_signal_summary`` data as the source when direct
        row access is not available, but falls back to the summary dict and
        reconstructs lightweight objects for aggregation.

        Note: PGStore does not expose a raw row-level list for
        signal_performance.  We use the existing ``get_signal_summary`` to
        obtain raw aggregate data and additionally query for individual rows
        via a direct SQL call if the pool is available; otherwise we return
        an empty list and rely on the data already written to PG during the
        evaluate_date() calls.
        """
        performances: list[SignalPerformance] = []
        try:
            pool = self._pg._pool_or_raise()
            import datetime as _dt

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        id, evaluation_date, symbol, signal_type,
                        signal_generated_at, entry_price, exit_price,
                        return_pct, hit, holding_hours
                    FROM signal_performance
                    WHERE evaluation_date >= $1
                      AND evaluation_date <= $2
                    ORDER BY evaluation_date DESC
                    """,
                    start,
                    end,
                )
            for row in rows:
                eval_date_val = row["evaluation_date"]
                if isinstance(eval_date_val, _dt.date) and not isinstance(eval_date_val, _dt.datetime):
                    eval_dt = _dt.datetime.combine(
                        eval_date_val, _dt.time.min, tzinfo=timezone.utc
                    )
                elif isinstance(eval_date_val, _dt.datetime):
                    eval_dt = eval_date_val
                else:
                    eval_dt = _dt.datetime.fromisoformat(str(eval_date_val))

                gen_at_val = row["signal_generated_at"]
                if isinstance(gen_at_val, _dt.datetime):
                    gen_at = gen_at_val
                else:
                    gen_at = _dt.datetime.fromisoformat(str(gen_at_val))

                performances.append(
                    SignalPerformance(
                        id=row["id"],
                        evaluation_date=eval_dt,
                        symbol=row["symbol"],
                        signal_type=row["signal_type"],
                        signal_generated_at=gen_at,
                        entry_price=float(row["entry_price"]),
                        exit_price=float(row["exit_price"]),
                        return_pct=float(row["return_pct"]),
                        hit=bool(row["hit"]),
                        holding_hours=float(row["holding_hours"]),
                    )
                )
        except Exception as exc:
            log.warning(
                "backtest.fetch_performances_range.failed",
                start=str(start),
                end=str(end),
                error=str(exc),
            )
        return performances
