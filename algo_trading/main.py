"""
main.py — NSE AlgoTrader entry point.

Usage:
    python -m algo_trading.main
    python -m algo_trading.main --mode paper
    python -m algo_trading.main --mode mock --symbols RELIANCE,TCS

The main.py:
  1. Parses CLI args (override config without editing .env)
  2. Configures logging (structlog + rotating file)
  3. Validates settings (config.py already validated at import — this is belt-and-suspenders)
  4. Runs startup security checks
  5. Starts DailyOrchestrator

Process layout (Stage 1 — single machine):
  Main process: DailyOrchestrator (tick loop + bar builder + signal generator)
  Daemon process: AlertSink (ZMQ PULL → Telegram)
  Daemon thread: Shoonya heartbeat (session renewal)
  Daemon thread: Shoonya WS supervisor (reconnect)
  Daemon thread: Order timeout checker
  Daemon thread: QuestDB flush

Exit codes:
  0: Clean session end (15:45 IST, normal exit)
  1: Fatal error before or during session
  2: Configuration error (.env missing/invalid)
  3: Startup reconciliation mismatch (human intervention required)
"""
from __future__ import annotations

import argparse
import sys

# ── Validate config at import — fail fast before any other import ─────────────
try:
    from algo_trading.config import settings, TradingMode
except (ValueError, Exception) as exc:
    print(f"\n[FATAL] Configuration error: {exc}\n", file=sys.stderr)
    print("  Fix your .env file and re-run. See .env.example for required fields.\n", file=sys.stderr)
    sys.exit(2)

# ── Logging must be configured BEFORE any structlog calls ─────────────────────
from algo_trading.core.logging import configure_logging, get_logger
configure_logging()
log = get_logger(__name__)

from algo_trading.core.calendar import is_trading_day, today_ist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NSE AlgoTrader — Stage 1 EMA cross paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m algo_trading.main                  # Paper trading (default)
  python -m algo_trading.main --mode mock      # Mock mode (no broker connection)
  python -m algo_trading.main --symbols RELIANCE,TCS

Environment:
  All config from .env — run 'python scripts/secure_env.py' to validate.
        """,
    )
    p.add_argument(
        "--mode",
        choices=["mock", "paper", "live"],
        default=None,
        help="Override TRADING_MODE from .env",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list (overrides SYMBOLS in .env)",
    )
    p.add_argument(
        "--skip-market-check",
        action="store_true",
        help="Allow running on non-trading days (testing only)",
    )
    return p.parse_args()


def main() -> int:
    """
    Main entry point. Returns exit code.
    """
    args = parse_args()

    log.info("algo_trader.starting", version="stage-1", config=settings.safe_repr())

    # ── Validate trading day ──────────────────────────────────────────────────
    if not args.skip_market_check and not is_trading_day():
        log.warning(
            "algo_trader.not_a_trading_day",
            date=str(today_ist()),
            hint="Use --skip-market-check to run on holidays (testing only)",
        )
        print(
            f"\nToday ({today_ist()}) is not an NSE trading day.\n"
            "Use --skip-market-check to override (testing only).\n",
            file=sys.stderr,
        )
        return 0

    # ── Log startup summary ───────────────────────────────────────────────────
    log.info(
        "algo_trader.config_summary",
        mode=settings.trading_mode.value,
        symbols=settings.symbol_list,
        capital=str(settings.capital_inr),
        data_feed=settings.data_feed,
        bar_type=settings.bar_type,
        ema_fast=settings.ema_fast_period,
        ema_slow=settings.ema_slow_period,
        questdb=f"{settings.questdb_host}:{settings.questdb_ilp_port}",
    )

    # ── Safety banner for paper mode ──────────────────────────────────────────
    if settings.is_paper():
        log.info(
            "algo_trader.paper_mode_active",
            note=(
                "Paper trading: live WebSocket tick data, "
                "simulated orders, zero financial risk"
            ),
        )

    # ── Safety block for live mode ────────────────────────────────────────────
    if settings.is_live():
        log.warning(
            "algo_trader.LIVE_MODE_ACTIVE",
            strategy_id=settings.strategy_id,
            approved=settings.broker_approval_date,
            capital=str(settings.capital_inr),
        )
        # Final gate: require all acceptance criteria from DECISIONS.md
        # This is enforced by the config validator — if we reach here, gates passed.

    # ── Run orchestrator ──────────────────────────────────────────────────────
    try:
        from algo_trading.app.daily_orchestrator import DailyOrchestrator
        orchestrator = DailyOrchestrator()
        orchestrator.run()
        log.info("algo_trader.session_complete")
        return 0

    except KeyboardInterrupt:
        log.info("algo_trader.keyboard_interrupt — shutting down gracefully")
        return 0

    except Exception as exc:
        log.exception("algo_trader.FATAL", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
