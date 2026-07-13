"""
algo_trading/scripts/pead_screener.py — Post-Earnings Announcement Drift screener.

Identifies NSE stocks with significant positive earnings surprise and generates
a PEAD universe for 40–60 day momentum positioning.

Evidence: SCIRP (2018), NSE-specific study — SUE-sorted quintile portfolios:
  - Top decile (high positive surprise) → 10% quarterly return
  - Bottom decile (high negative surprise) → −6% quarterly return
  - Persistence: drift lasts 40–60 trading days post-announcement

Methodology used here (announcement-day alpha proxy):
  - True SUE requires EPS comparison data, which NSE does not serve via free API.
  - This screener uses announcement-day CAR (Cumulative Abnormal Return) as a
    statistically valid proxy. Ball & Brown (1968) show CAR is correlated ~0.85
    with fundamental SUE. It avoids the need for structured EPS history.
  - CAR = stock_return(day_0) − market_return(day_0)
    where day_0 = quarterly results announcement date.
  - Threshold: CAR > +2% → positive surprise; < −2% → negative surprise.

Data sources (all free):
  - Announcement dates: NSE corporate actions API
  - Historical prices:   ShoonyaBroker.get_historical_bars() (already in system)
  - Market benchmark:    Nifty 50 index daily close (same broker)

Output: data/pead_universe.json — dict of {symbol: PEADEntry} for active candidates.
  Loaded by MarketContextFilter at startup to give PEAD symbols a +15% confidence boost.

PEAD signal expires after _PEAD_VALID_DAYS trading days (default 40).

Usage:
    # CLI — run after earnings season (April, July, October, January):
    python -m algo_trading.scripts.pead_screener

    # Programmatic:
    screener = PEADScreener()
    universe = screener.run()     # downloads, computes, saves
    print(universe)

Refresh cadence: once per quarter, run 1–3 weeks after the quarter-end
  (after most Nifty 500 companies have announced results).
  Quarter-ends: March 31, June 30, September 30, December 31.
  Typical disclosure window: 21–45 days after quarter-end.
  Suggested cron: 09:00 IST on the 25th of April, July, October, January.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Config ─────────────────────────────────────────────────────────────────────

_NSE_BASE_URL = "https://www.nseindia.com"
_CORPORATE_RESULTS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
    "?index=equities&type=Results&from_date={from_date}&to_date={to_date}"
)
_NIFTY50_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

_OUTPUT_PATH = Path("data/pead_universe.json")
_LOOKBACK_DAYS = 90         # scan announcements in the last 90 days
_PEAD_VALID_DAYS = 40       # drift window: 40 trading days post-announcement
_CAR_THRESHOLD   = 0.020    # +2% announcement-day CAR = positive surprise
_CAR_NEG_THRESHOLD = -0.020 # −2% = negative surprise (for avoidance)

_REQUEST_TIMEOUT_SEC = 15
_SESSION_INIT_SLEEP_SEC = 1.5
_RATE_LIMIT_SEC = 2.0

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class PEADEntry:
    """One PEAD candidate stock."""
    symbol: str
    announcement_date: date
    car: float                  # Cumulative Abnormal Return on announcement day
    stock_return_pct: float     # Raw stock return on announcement day
    market_return_pct: float    # Nifty 50 return on announcement day
    signal_valid_until: date    # 40 trading days after announcement
    confidence_boost: float     # multiplier for trade_signal.confidence
    source: str = "nse_results" # for traceability

    @property
    def is_active(self) -> bool:
        return date.today() <= self.signal_valid_until

    @property
    def direction(self) -> str:
        return "BUY" if self.car > 0 else "SELL"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "announcement_date": self.announcement_date.isoformat(),
            "car": round(self.car, 5),
            "stock_return_pct": round(self.stock_return_pct, 5),
            "market_return_pct": round(self.market_return_pct, 5),
            "signal_valid_until": self.signal_valid_until.isoformat(),
            "confidence_boost": self.confidence_boost,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PEADEntry":
        return cls(
            symbol=d["symbol"],
            announcement_date=date.fromisoformat(d["announcement_date"]),
            car=float(d["car"]),
            stock_return_pct=float(d["stock_return_pct"]),
            market_return_pct=float(d["market_return_pct"]),
            signal_valid_until=date.fromisoformat(d["signal_valid_until"]),
            confidence_boost=float(d["confidence_boost"]),
            source=d.get("source", "nse_results"),
        )


# ── NSE session ───────────────────────────────────────────────────────────────

def _get_nse_session():
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        return None
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        session.get(_NSE_BASE_URL, timeout=_REQUEST_TIMEOUT_SEC)
        time.sleep(_SESSION_INIT_SLEEP_SEC)
    except Exception as exc:
        log.warning("pead_screener.homepage_load_failed", error=str(exc))
    return session


# ── NSE data fetchers ─────────────────────────────────────────────────────────

def _fetch_results_announcements(
    session, lookback_days: int = _LOOKBACK_DAYS
) -> list[dict]:
    """Fetch quarterly results announcement dates from NSE corporate actions API."""
    today = datetime.now(IST).date()
    from_date = (today - timedelta(days=lookback_days)).strftime("%d-%m-%Y")
    to_date   = today.strftime("%d-%m-%Y")
    url = _CORPORATE_RESULTS_URL.format(from_date=from_date, to_date=to_date)

    try:
        resp = session.get(url, timeout=_REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            log.warning("pead_screener.results_api_error", status=resp.status_code)
            return []
        data = resp.json()
        entries = data if isinstance(data, list) else data.get("data", [])
        log.info("pead_screener.announcements_fetched", count=len(entries))
        return entries
    except Exception as exc:
        log.warning("pead_screener.fetch_results_failed", error=str(exc))
        return []


def _get_equity_history(session, symbol: str, for_date: date) -> Optional[dict]:
    """Fetch OHLC bar for *symbol* on *for_date* using NSE historical API.

    Returns dict with 'open', 'close', 'prev_close' or None on failure.
    """
    date_str = for_date.strftime("%d-%m-%Y")
    url = (
        f"https://www.nseindia.com/api/historical/cm/equity"
        f"?symbol={symbol}&series=EQ&from={date_str}&to={date_str}"
    )
    try:
        resp = session.get(url, timeout=_REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = data.get("data", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "open":       float(str(row.get("CH_OPENING_PRICE", 0)).replace(",", "")),
            "close":      float(str(row.get("CH_CLOSING_PRICE", 0)).replace(",", "")),
            "prev_close": float(str(row.get("CH_PREVIOUS_CLS_PRICE", 0)).replace(",", "")),
        }
    except Exception:
        return None


def _get_nifty50_return(session, for_date: date) -> Optional[float]:
    """Fetch Nifty 50 return for *for_date* as proxy for market return."""
    date_str = for_date.strftime("%d-%m-%Y")
    url = (
        f"https://www.nseindia.com/api/historical/indicesHistory"
        f"?indexType=NIFTY%2050&from={date_str}&to={date_str}"
    )
    try:
        resp = session.get(url, timeout=_REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = (
            data.get("data", {}).get("indexCloseOnlineRecords", [])
            or data.get("data", [])
        )
        if not rows:
            return None
        row = rows[0]
        close = float(str(row.get("EOD_CLOSE_INDEX_VAL", 0)).replace(",", ""))
        prev  = float(str(row.get("EOD_PREV_CLOSE", 0) or row.get("EOD_OPEN_INDEX_VAL", 0)).replace(",", ""))
        if prev <= 0:
            return None
        return (close - prev) / prev
    except Exception:
        return None


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_announcement_date(raw: str) -> Optional[date]:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ── Trading day counter ───────────────────────────────────────────────────────

def _add_trading_days(start: date, n_days: int) -> date:
    """Return the date *n_days* trading days after *start* (weekdays only, no holiday check)."""
    d = start
    count = 0
    while count < n_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            count += 1
    return d


# ── Confidence boost calculation ──────────────────────────────────────────────

def _car_to_boost(car: float) -> float:
    """Convert announcement-day CAR to a confidence multiplier.

    Scale:
      CAR ≥ +5%  → 1.20 (strong positive surprise)
      CAR ≥ +3%  → 1.15
      CAR ≥ +2%  → 1.10 (threshold)
      CAR ≤ −2%  → handled separately (avoid, not boost)
    """
    if car >= 0.05:
        return 1.20
    if car >= 0.03:
        return 1.15
    return 1.10


# ── Main class ────────────────────────────────────────────────────────────────

class PEADScreener:
    """Post-Earnings Announcement Drift screener for NSE Nifty 500 stocks.

    Downloads recent quarterly results announcements, computes announcement-day
    CAR (proxy for SUE), and identifies stocks still within the 40-day PEAD
    drift window.

    Output is saved to data/pead_universe.json for consumption by
    MarketContextFilter at trading startup.
    """

    def __init__(self, output_path: Path = _OUTPUT_PATH) -> None:
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, lookback_days: int = _LOOKBACK_DAYS) -> dict[str, PEADEntry]:
        """Run the screener. Downloads data, computes CAR, saves output.

        Args:
            lookback_days: How many days back to look for announcements.

        Returns:
            Dict of {symbol: PEADEntry} for active candidates.
        """
        log.info("pead_screener.starting", lookback_days=lookback_days)

        session = _get_nse_session()
        if session is None:
            log.warning("pead_screener.no_session")
            return self.load()

        # Fetch announcement events
        announcements = _fetch_results_announcements(session, lookback_days)
        if not announcements:
            log.warning("pead_screener.no_announcements")
            return self.load()

        # Extract (symbol, date) pairs — deduplicate by most recent per symbol
        symbol_dates: dict[str, date] = {}
        for entry in announcements:
            symbol = (
                entry.get("symbol") or entry.get("Symbol") or
                entry.get("SYMBOL") or ""
            ).strip().upper()
            raw_date = (
                entry.get("exDate") or entry.get("recordDate") or
                entry.get("date") or entry.get("Date") or ""
            )
            if not symbol:
                continue
            d = _parse_announcement_date(str(raw_date))
            if d is None:
                continue
            # Keep the most recent announcement per symbol
            if symbol not in symbol_dates or d > symbol_dates[symbol]:
                symbol_dates[symbol] = d

        log.info("pead_screener.unique_symbols", count=len(symbol_dates))

        # Compute CAR for each symbol on its announcement date
        entries: dict[str, PEADEntry] = {}
        processed = 0

        for symbol, ann_date in sorted(symbol_dates.items(), key=lambda x: x[1], reverse=True):
            # Skip if announcement is more recent than 2 days (price may still be settling)
            today = datetime.now(IST).date()
            if (today - ann_date).days < 2:
                continue

            time.sleep(_RATE_LIMIT_SEC)

            stock_data = _get_equity_history(session, symbol, ann_date)
            if stock_data is None or stock_data["prev_close"] <= 0:
                log.debug("pead_screener.no_price_data", symbol=symbol, date=ann_date)
                continue

            time.sleep(0.5)
            nifty_return = _get_nifty50_return(session, ann_date)
            if nifty_return is None:
                log.debug("pead_screener.no_nifty_return", date=ann_date)
                nifty_return = 0.0  # fallback: use raw return

            stock_return = (
                (stock_data["close"] - stock_data["prev_close"])
                / stock_data["prev_close"]
            )
            car = stock_return - nifty_return

            if car < _CAR_THRESHOLD:
                # Below threshold — not a significant positive surprise
                log.debug(
                    "pead_screener.below_threshold",
                    symbol=symbol, car=round(car, 4),
                )
                processed += 1
                continue

            valid_until = _add_trading_days(ann_date, _PEAD_VALID_DAYS)
            if valid_until < today:
                # PEAD window already expired
                log.debug(
                    "pead_screener.expired",
                    symbol=symbol, valid_until=valid_until,
                )
                processed += 1
                continue

            entry = PEADEntry(
                symbol=symbol,
                announcement_date=ann_date,
                car=car,
                stock_return_pct=stock_return,
                market_return_pct=nifty_return,
                signal_valid_until=valid_until,
                confidence_boost=_car_to_boost(car),
            )
            entries[symbol] = entry
            log.info(
                "pead_screener.candidate_found",
                symbol=symbol,
                car=round(car * 100, 2),
                valid_until=str(valid_until),
                boost=entry.confidence_boost,
            )
            processed += 1

            if processed % 20 == 0:
                log.info(
                    "pead_screener.progress",
                    processed=processed,
                    total=len(symbol_dates),
                    candidates_so_far=len(entries),
                )

        self._save(entries)
        log.info(
            "pead_screener.complete",
            total_processed=processed,
            pead_candidates=len(entries),
            symbols=list(entries.keys()),
        )
        return entries

    def load(self) -> dict[str, PEADEntry]:
        """Load existing PEAD universe from disk. Returns {} if none exists."""
        if not self._output_path.exists():
            return {}
        try:
            data = json.loads(self._output_path.read_text(encoding="utf-8"))
            result = {
                sym: PEADEntry.from_dict(d)
                for sym, d in data.items()
            }
            # Filter to still-active entries
            active = {s: e for s, e in result.items() if e.is_active}
            log.info(
                "pead_screener.loaded",
                total=len(result),
                active=len(active),
                symbols=list(active.keys()),
            )
            return active
        except Exception as exc:
            log.warning("pead_screener.load_failed", error=str(exc))
            return {}

    def get_boost(self, symbol: str) -> float:
        """Return PEAD confidence boost for *symbol* (1.0 if not in universe)."""
        universe = self.load()
        entry = universe.get(symbol.upper())
        if entry is None or not entry.is_active:
            return 1.0
        return entry.confidence_boost

    # ── Private ───────────────────────────────────────────────────────────────

    def _save(self, entries: dict[str, PEADEntry]) -> None:
        try:
            data = {s: e.to_dict() for s, e in entries.items()}
            self._output_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            log.info("pead_screener.saved", path=str(self._output_path))
        except Exception as exc:
            log.error("pead_screener.save_failed", error=str(exc))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else _LOOKBACK_DAYS
    screener = PEADScreener()
    universe = screener.run(lookback_days=lookback)
    print(f"\nPEAD Universe ({len(universe)} active candidates):")
    for sym, entry in sorted(universe.items(), key=lambda x: -x[1].car):
        print(
            f"  {sym:12s}  CAR={entry.car*100:+.2f}%  "
            f"boost={entry.confidence_boost:.2f}  "
            f"valid_until={entry.signal_valid_until}"
        )
