"""
F&O historical data downloader for Stage 4 features.

Downloads option chain snapshots from NSE India (when available)
or generates synthetic F&O chain data for development/testing.

For development (no real data):
    python -m algo_trading.scripts.download_fo_history --synthetic

For production (requires NseIndiaApi):
    pip install nse
    python -m algo_trading.scripts.download_fo_history --symbols RELIANCE --days 30

NseIndiaApi (BennyThadikaran/NseIndiaApi) uses the option-chain-v3 endpoint and
handles the mandatory two-step NSE cookie warm-up automatically. nsepython is
deprecated and breaks against the current NSE website.

Saves parquet files to: data/fo_chain/{symbol}/{date}_chain.parquet
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── NSEPython: deferred import (production only) ───────────────────────────────
# Imported lazily inside _run_live_mode() with a clear error message.

# ── Default configuration ──────────────────────────────────────────────────────
_DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BAJFINANCE", "KOTAKBANK", "LT",
]
_DEFAULT_DAYS = 30
_DEFAULT_OUTPUT_DIR = Path("data/fo_chain")
_STALENESS_WARN_DAYS = 7       # Skip+warn if existing file is < 7 days old
_SEED_BASE = 20260618           # Reproducible seed base for synthetic noise
_N_STRIKES_SYNTHETIC = 10       # Strikes per chain in synthetic mode
_RISK_FREE_RATE = 0.065         # 6.5% annual risk-free rate


# ══════════════════════════════════════════════════════════════════════════════
# Parquet helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chain_to_dataframe(chain):
    """Convert an OptionChain object to a pandas DataFrame row-per-contract."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. pip install pandas") from exc

    rows = []
    for contract in chain.contracts:
        rows.append({
            "symbol": contract.symbol,
            "expiry": contract.expiry.isoformat(),
            "strike": contract.strike,
            "option_type": contract.option_type,
            "oi": contract.oi,
            "change_in_oi": contract.change_in_oi,
            "volume": contract.volume,
            "iv": contract.iv,
            "ltp": contract.ltp,
            "delta": contract.delta,
            "gamma": contract.gamma,
            "theta": contract.theta,
            "vega": contract.vega,
            "bid": contract.bid,
            "ask": contract.ask,
            "underlying_price": chain.underlying_price,
            "timestamp": chain.timestamp.isoformat(),
            "chain_date": chain.timestamp.date().isoformat(),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _save_chain_parquet(df, output_path: Path) -> None:
    """Save a DataFrame to parquet, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _validate_output_file(path: Path) -> bool:
    """
    Validate that the saved parquet file has at least 1 row.

    Returns:
        True if row count > 0, False otherwise.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if len(df) == 0:
            log.warning("fo_downloader.empty_file", path=str(path))
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("fo_downloader.validate_error", path=str(path), error=str(exc))
        return False


def _is_stale(path: Path) -> bool:
    """
    Return True if the file at path is newer than _STALENESS_WARN_DAYS.

    'Stale' in this context means 'recently created' — we skip files that
    already exist and are fresh (< 7 days old) to avoid redundant downloads.
    """
    if not path.exists():
        return False  # Does not exist → not stale → proceed with download
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    age_days = (date.today() - mtime).days
    return age_days < _STALENESS_WARN_DAYS


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic mode
# ══════════════════════════════════════════════════════════════════════════════

# Representative spot prices for synthetic chains (INR)
_SYNTHETIC_SPOT_PRICES: dict[str, float] = {
    "RELIANCE": 2950.0,
    "TCS": 3800.0,
    "HDFCBANK": 1620.0,
    "INFY": 1480.0,
    "ICICIBANK": 1150.0,
    "HINDUNILVR": 2540.0,
    "SBIN": 820.0,
    "BAJFINANCE": 7200.0,
    "KOTAKBANK": 1780.0,
    "LT": 3400.0,
}
_FALLBACK_SPOT = 1000.0


def _run_synthetic_mode(
    symbols: list[str],
    days: int,
    output_dir: Path,
) -> tuple[int, int, list[date]]:
    """
    Generate synthetic F&O chain snapshots for development/testing.

    Creates days × len(symbols) parquet files using FOChainLoader.load_synthetic().
    Uses seeded random for reproducibility.

    Args:
        symbols:    List of NSE equity symbols.
        days:       Number of calendar days to generate.
        output_dir: Root output directory (files go under output_dir/{symbol}/).

    Returns:
        (files_saved, total_contracts, date_range) tuple for summary.
    """
    from algo_trading.data.fo_chain import FOChainLoader

    loader = FOChainLoader()
    today = date.today()
    date_range = [today - timedelta(days=d) for d in range(days - 1, -1, -1)]

    files_saved = 0
    total_contracts = 0
    saved_dates: list[date] = []

    total_tasks = len(symbols) * len(date_range)
    completed = 0

    print(f"\nGenerating synthetic F&O chain data:")
    print(f"  Symbols: {len(symbols)}  |  Days: {days}  |  Total files: {total_tasks}")
    print()

    for sym in symbols:
        spot = _SYNTHETIC_SPOT_PRICES.get(sym, _FALLBACK_SPOT)
        sym_dir = output_dir / sym
        sym_dir.mkdir(parents=True, exist_ok=True)

        for chain_date in date_range:
            output_path = sym_dir / f"{chain_date.isoformat()}_chain.parquet"

            # Staleness check: skip if file exists and is < 7 days old
            if _is_stale(output_path):
                log.info(
                    "fo_downloader.skip_fresh_file",
                    path=str(output_path),
                    reason="file < 7 days old",
                )
                completed += 1
                _print_progress(completed, total_tasks, sym, chain_date, skipped=True)
                continue

            # Reproducible seed: deterministic per (symbol, date)
            seed = _SEED_BASE + hash(f"{sym}{chain_date.isoformat()}") % (2 ** 31)

            # Add small day-specific price drift for realism
            rng = random.Random(seed)
            drift = rng.gauss(0, spot * 0.01)  # ±1% daily noise
            synthetic_spot = max(spot * 0.5, spot + drift)

            # Adjust expiry_days so it's always a positive future expiry
            expiry_days = max(1, (date.today() - chain_date).days + 7)

            try:
                chain = loader.load_synthetic(
                    symbol=sym,
                    underlying_price=synthetic_spot,
                    n_strikes=_N_STRIKES_SYNTHETIC,
                    expiry_days=expiry_days,
                )
                # Override timestamp to reflect chain_date for historical accuracy
                from algo_trading.data.fo_chain import OptionChain
                from dataclasses import replace as _replace
                chain = _replace(
                    chain,
                    timestamp=datetime.combine(chain_date, datetime.min.time()),
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "fo_downloader.synthetic_gen_error",
                    symbol=sym,
                    date=str(chain_date),
                    error=str(exc),
                )
                completed += 1
                _print_progress(completed, total_tasks, sym, chain_date, skipped=True)
                continue

            df = _chain_to_dataframe(chain)
            _save_chain_parquet(df, output_path)

            # Validate
            if not _validate_output_file(output_path):
                log.warning("fo_downloader.validation_failed", path=str(output_path))
                completed += 1
                _print_progress(completed, total_tasks, sym, chain_date, skipped=True)
                continue

            files_saved += 1
            total_contracts += len(df)
            saved_dates.append(chain_date)

            completed += 1
            _print_progress(completed, total_tasks, sym, chain_date)

    return files_saved, total_contracts, sorted(set(saved_dates))


# ══════════════════════════════════════════════════════════════════════════════
# Live / production mode (requires NSEPython)
# ══════════════════════════════════════════════════════════════════════════════

def _run_live_mode(
    symbols: list[str],
    days: int,
    output_dir: Path,
) -> tuple[int, int, list[date]]:
    """
    Download live NSE option chain data using NseIndiaApi (BennyThadikaran/NseIndiaApi).

    Requires: pip install nse

    NseIndiaApi uses the option-chain-v3 endpoint and handles the mandatory
    two-step NSE cookie warm-up automatically (GET nseindia.com, then
    GET nseindia.com/market-data/live-equity-market before any API calls).

    For each symbol downloads today's option chain, parses it into
    OptionContract objects, computes BS Greeks via BlackScholesGreeks,
    and saves as parquet.

    Note: Public NSE API provides current-session data only — historical daily
    snapshots are not available. For historical data use --synthetic or a
    paid data vendor (TrueData, etc.).

    Args:
        symbols:    List of NSE equity symbols (or "NIFTY"/"BANKNIFTY" for indices).
        days:       Retained for API compatibility (only today's data available).
        output_dir: Root output directory.

    Returns:
        (files_saved, total_contracts, date_range) tuple for summary.
    """
    try:
        from nse import NSE  # type: ignore[import]  # pip install nse
    except ImportError:
        raise SystemExit(
            "NseIndiaApi is required for --live mode.\n"
            "Install with: pip install nse\n"
            "Source: https://github.com/BennyThadikaran/NseIndiaApi\n"
            "Or use --synthetic for development without network access.\n"
            "Note: nsepython is deprecated — do NOT use it."
        )

    from algo_trading.data.fo_chain import (
        BlackScholesGreeks,
        OptionChain,
        OptionContract,
    )

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. pip install pandas") from exc

    today = date.today()
    files_saved = 0
    total_contracts = 0
    saved_dates: list[date] = []

    # NSE() constructor performs the two-step cookie warm-up automatically:
    #   Step 1: GET https://www.nseindia.com  (loads session cookies)
    #   Step 2: GET https://www.nseindia.com/market-data/live-equity-market
    # This is required before any NSE archive/API requests.
    try:
        nse = NSE()
    except Exception as exc:
        raise SystemExit(
            f"Failed to initialize NSE session (cookie warm-up failed): {exc}\n"
            "Check network connectivity. Corporate proxies may block NSE."
        ) from exc

    total_tasks = len(symbols)
    completed = 0

    print(f"\nDownloading live NSE F&O chain data (via NseIndiaApi):")
    print(f"  Symbols: {len(symbols)}  |  Date: {today}")
    print()

    for sym in symbols:
        output_path = output_dir / sym / f"{today.isoformat()}_chain.parquet"

        # Staleness check
        if _is_stale(output_path):
            log.info(
                "fo_downloader.skip_fresh_file",
                symbol=sym,
                path=str(output_path),
                reason=f"file already downloaded within {_STALENESS_WARN_DAYS} days",
            )
            print(
                f"  [{completed + 1}/{total_tasks}] {sym:>12}  {today}  "
                f"SKIPPED (fresh file exists)"
            )
            completed += 1
            continue

        try:
            # NseIndiaApi: optionChain() uses the option-chain-v3 endpoint.
            # Returns same structure as legacy nsepython for compatibility.
            chain_json = nse.optionChain(sym)

            if not chain_json or "records" not in chain_json:
                log.warning("fo_downloader.empty_response", symbol=sym)
                completed += 1
                _print_progress(completed, total_tasks, sym, today, skipped=True)
                continue

            records = chain_json["records"]
            underlying_price = float(
                records.get(
                    "underlyingValue",
                    records.get("data", [{}])[0].get("PE", {}).get("underlyingValue", 0),
                )
            )
            expiry_dates_raw: list[str] = records.get("expiryDates", [])

            if not expiry_dates_raw:
                log.warning("fo_downloader.no_expiries", symbol=sym)
                completed += 1
                _print_progress(completed, total_tasks, sym, today, skipped=True)
                continue

            # Use nearest expiry
            nearest_expiry_str = expiry_dates_raw[0]
            try:
                nearest_expiry = datetime.strptime(nearest_expiry_str, "%d-%b-%Y").date()
            except ValueError:
                nearest_expiry = today + timedelta(days=7)

            T = max((nearest_expiry - today).days, 1) / 365.0

            contracts: list[OptionContract] = []
            for row in records.get("data", []):
                for opt_type, key in [("CE", "CE"), ("PE", "PE")]:
                    entry = row.get(key)
                    if not entry:
                        continue
                    try:
                        contract = OptionContract(
                            symbol=sym,
                            expiry=nearest_expiry,
                            strike=float(entry.get("strikePrice", 0)),
                            option_type=opt_type,
                            oi=int(entry.get("openInterest", 0)),
                            change_in_oi=int(entry.get("changeinOpenInterest", 0)),
                            volume=int(entry.get("totalTradedVolume", 0)),
                            iv=float(entry.get("impliedVolatility", 0.0)) / 100.0,
                            ltp=float(entry.get("lastPrice", 0.0)),
                            bid=float(entry.get("bidprice", 0.0)),
                            ask=float(entry.get("askPrice", 0.0)),
                        )
                        # Compute Greeks via Black-Scholes
                        if contract.ltp > 0 and underlying_price > 0 and contract.strike > 0:
                            contract = BlackScholesGreeks.compute_all_greeks(
                                contract, underlying_price, T, _RISK_FREE_RATE
                            )
                        contracts.append(contract)
                    except (KeyError, ValueError, TypeError) as exc:
                        log.debug("fo_downloader.contract_parse_skip", error=str(exc))
                        continue

            if not contracts:
                log.warning("fo_downloader.no_contracts_parsed", symbol=sym)
                completed += 1
                _print_progress(completed, total_tasks, sym, today, skipped=True)
                continue

            chain = OptionChain(
                symbol=sym,
                underlying_price=underlying_price,
                timestamp=datetime.now(),
                expiry=nearest_expiry,
                contracts=contracts,
            )

            df = _chain_to_dataframe(chain)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _save_chain_parquet(df, output_path)

            if not _validate_output_file(output_path):
                log.warning("fo_downloader.validation_failed", path=str(output_path))
                completed += 1
                _print_progress(completed, total_tasks, sym, today, skipped=True)
                continue

            files_saved += 1
            total_contracts += len(df)
            saved_dates.append(today)

            completed += 1
            _print_progress(completed, total_tasks, sym, today)

        except Exception as exc:  # noqa: BLE001
            log.error(
                "fo_downloader.live_fetch_error",
                symbol=sym,
                error=str(exc),
            )
            completed += 1
            _print_progress(completed, total_tasks, sym, today, skipped=True)

    return files_saved, total_contracts, sorted(set(saved_dates))


# ══════════════════════════════════════════════════════════════════════════════
# Progress display
# ══════════════════════════════════════════════════════════════════════════════

def _print_progress(
    completed: int,
    total: int,
    symbol: str,
    chain_date: date,
    skipped: bool = False,
) -> None:
    """Print a simple progress line (no tqdm dependency)."""
    pct = int(completed / total * 100) if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * completed / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    status = "SKIPPED" if skipped else "OK     "
    print(
        f"  [{bar}] {pct:3d}%  {completed:>4}/{total}  "
        f"{symbol:>12}  {chain_date}  {status}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(
    mode: str,
    files_saved: int,
    total_contracts: int,
    date_range: list[date],
    output_dir: Path,
) -> None:
    """Print final summary to stdout."""
    print("\n" + "=" * 60)
    print("  F&O CHAIN DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Mode:              {mode}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Files saved:       {files_saved:,}")
    print(f"  Total contracts:   {total_contracts:,}")
    if date_range:
        print(f"  Date range:        {date_range[0]}  →  {date_range[-1]}")
        print(f"  Days covered:      {len(date_range)}")
    else:
        print("  Date range:        (no files saved)")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSE F&O Historical Chain Downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        metavar="SYMBOL",
        help="NSE equity symbols to download F&O chains for",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=_DEFAULT_DAYS,
        help="Number of calendar days of data to generate/download",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help=(
            "Generate synthetic F&O chain data (default for development; "
            "no network required)"
        ),
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Download live NSE option chain data (requires: pip install nse)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Root output directory for parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    output_dir = Path(args.output_dir)

    # Default mode: synthetic unless --live is explicitly requested
    use_live = args.live
    use_synthetic = args.synthetic or not use_live

    mode_label = "live (NseIndiaApi)" if use_live else "synthetic (seeded random)"

    log.info(
        "fo_downloader.start",
        mode=mode_label,
        symbols=args.symbols,
        days=args.days,
        output_dir=str(output_dir),
    )

    if use_live:
        files_saved, total_contracts, date_range = _run_live_mode(
            symbols=args.symbols,
            days=args.days,
            output_dir=output_dir,
        )
    else:
        files_saved, total_contracts, date_range = _run_synthetic_mode(
            symbols=args.symbols,
            days=args.days,
            output_dir=output_dir,
        )

    _print_summary(
        mode=mode_label,
        files_saved=files_saved,
        total_contracts=total_contracts,
        date_range=date_range,
        output_dir=output_dir,
    )

    log.info(
        "fo_downloader.complete",
        files_saved=files_saved,
        total_contracts=total_contracts,
        mode=mode_label,
    )


if __name__ == "__main__":
    main()
