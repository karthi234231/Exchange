"""
scripts/run_fo.py — Entry point for the F&O session orchestrator.

Launches FOOrchestrator for a single NSE underlying (NIFTY or BANKNIFTY).
Blocks until session end (15:45 IST) or SIGTERM/SIGINT.

Usage
-----
    # Paper mode (default):
    python -m algo_trading.scripts.run_fo
    python -m algo_trading.scripts.run_fo --symbol BANKNIFTY --dte 14

    # Live mode (only after all 13 gates pass):
    TRADING_MODE=live python -m algo_trading.scripts.run_fo --symbol NIFTY --dte 21

Environment variables (from .env / shell):
    TRADING_MODE        paper | live   (default: paper)
    SHOONYA_USER_ID     required for broker login
    SHOONYA_PASSWORD    required for broker login
    SHOONYA_TOTP_SECRET required for broker login
    TELEGRAM_BOT_TOKEN  optional — Telegram alerts
    TELEGRAM_CHAT_ID    optional — Telegram alerts

Notes
-----
- This script must be run during NSE market hours (09:00–15:45 IST).
  Outside market hours it exits immediately with a warning.
- One process per underlying. Run two processes for NIFTY + BANKNIFTY.
- QuestDB must be running: docker run -p 9000:9000 questdb/questdb
"""
from __future__ import annotations

import argparse
import sys

import structlog

from algo_trading.core.logging import configure_logging
from algo_trading.core.calendar import is_trading_day, today_ist

log = structlog.get_logger(__name__)

# Valid NSE F&O underlyings supported by this orchestrator
_VALID_SYMBOLS = {"NIFTY", "BANKNIFTY"}

# DTE bounds — same as FOSignalGenerator defaults
_MIN_DTE = 7
_MAX_DTE = 45


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE F&O session orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m algo_trading.scripts.run_fo
  python -m algo_trading.scripts.run_fo --symbol BANKNIFTY --dte 14
  python -m algo_trading.scripts.run_fo --symbol NIFTY --dte 21 --dry-run
        """,
    )
    parser.add_argument(
        "--symbol",
        default="NIFTY",
        choices=list(_VALID_SYMBOLS),
        help="NSE underlying to trade (default: NIFTY)",
    )
    parser.add_argument(
        "--dte",
        type=int,
        default=21,
        metavar="DAYS",
        help=f"Calendar days to target expiry [{_MIN_DTE}–{_MAX_DTE}] (default: 21)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without connecting to broker",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Skip the trading-day check (useful for testing on weekends)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Raise SystemExit with a clear message if args are invalid."""
    if args.dte < _MIN_DTE or args.dte > _MAX_DTE:
        print(
            f"ERROR: --dte {args.dte} is outside allowed range [{_MIN_DTE}, {_MAX_DTE}].\n"
            f"  DTE < {_MIN_DTE}: gamma risk too high.\n"
            f"  DTE > {_MAX_DTE}: theta decay too slow to realize within the session.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    configure_logging()
    args = _parse_args()
    _validate_args(args)

    log.info(
        "run_fo.starting",
        symbol=args.symbol,
        dte=args.dte,
        dry_run=args.dry_run,
        force_run=args.force_run,
    )

    # ── Trading day guard ─────────────────────────────────────────────────────
    if not args.force_run and not is_trading_day():
        log.warning(
            "run_fo.not_a_trading_day",
            date=str(today_ist()),
            hint="Use --force-run to override (testing only)",
        )
        print(
            f"Today ({today_ist()}) is not an NSE trading day. Exiting.\n"
            "Use --force-run to override for testing.",
            file=sys.stderr,
        )
        sys.exit(0)

    # ── Dry run: validate imports and config only ─────────────────────────────
    if args.dry_run:
        print("Dry run — validating imports...")
        try:
            from algo_trading.app.fo_orchestrator import FOOrchestrator
            from algo_trading.config import settings
            print(f"  FOOrchestrator: OK")
            print(f"  Config trading_mode: {settings.trading_mode.value}")
            print(f"  Symbol: {args.symbol}")
            print(f"  DTE: {args.dte}")
            print("Dry run complete. No connections made.")
        except Exception as exc:
            print(f"ERROR during dry run: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Live / paper run ──────────────────────────────────────────────────────
    from algo_trading.app.fo_orchestrator import FOOrchestrator
    from algo_trading.config import settings

    log.info(
        "run_fo.config",
        trading_mode=settings.trading_mode.value,
        symbol=args.symbol,
        dte=args.dte,
        bar_interval_minutes=settings.bar_interval_minutes,
    )

    orchestrator = FOOrchestrator(
        symbol=args.symbol,
        days_to_expiry=args.dte,
    )

    log.info("run_fo.handing_off_to_orchestrator")
    orchestrator.run()
    log.info("run_fo.session_complete")


if __name__ == "__main__":
    main()
