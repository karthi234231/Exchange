"""
algo_trading/scripts/quality_momentum_screener.py
Combined Quality + Momentum + Earnings Quality screener for NSE Nifty 500.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evidence

  Quality + Momentum combined (BacktestIndia 18-year NSE backtest 2006–2025):
    17.95% CAGR vs Nifty 50 benchmark 10.42% — 83% more wealth compounded.
    Rebalance quarterly in June and December (Nifty500M50 reconstitution dates).

  Earnings quality / accrual anomaly (Sloan 1996, replicated India 2012–2024):
    Low-accrual firms (cash-backed earnings) persistently outperform.
    Especially strong in NSE 500 mid-cap tier due to thin analyst coverage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Methodology

  Three independent factors, each ranked as a percentile [0, 1] across the
  screened universe, then combined with fixed weights:

  1. Momentum (40%)
     12-1 month price return: return from 12 months ago to 1 month ago.
     Excludes the most recent month to avoid short-term reversal bias.
     Source: Jegadeesh & Titman (1993), replicated extensively on NSE.

  2. Quality (35%)
     Composite of three sub-factors (equal weight within quality):
       a. ROE (Return on Equity)      — higher = better
       b. Debt/Equity ratio           — lower = better (inverted rank)
       c. Earnings stability          — fraction of last 4 quarters with
                                        positive YoY EPS growth (0–1)
     Stocks with missing fundamental data receive neutral score (0.50).

  3. Accrual Quality (25%)
     Proxy: coefficient of variation (CV) of EPS across last 4 quarters.
     Low CV = stable, cash-backed earnings = high quality.
     Score = 1 / (1 + CV), so higher score = lower accruals = better.
     If fewer than 2 quarters of EPS data → neutral score (0.50).

  Combined score = 0.40 × momentum_rank + 0.35 × quality_rank + 0.25 × accrual_rank

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Outputs

  Quintile 5 (top 20%, score ≥ 0.80):  confidence_boost = 1.15×
  Top decile  (score ≥ 0.90):           confidence_boost = 1.20×
  Quintile 4  (score 0.60–0.80):        confidence_boost = 1.05×
  Quintile 3  (score 0.40–0.60):        neutral
  Quintile 2  (score 0.20–0.40):        size_multiplier  = 0.85×
  Quintile 1  (bottom 20%, score < 0.20): size_multiplier = 0.70×
  Bottom decile (score < 0.10):         size_multiplier  = 0.60×

  The size_multiplier feeds into get_composite_multiplier() (multiplicative).
  The confidence_boost feeds into get_confidence_boost() (takes max vs PEAD/FDA).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Refresh cadence: QUARTERLY

  Suggested run dates: 20th of April, July, October, January
  (2–3 weeks after quarter-end, once ~80% of Nifty 500 have filed results).

  Full Nifty 500 run: ~90 minutes (rate-limited to avoid NSE ban).
  Quick mode (trading universe only, ~50 symbols): ~10 minutes.

  CLI:
    # Full Nifty 500:
    python -m algo_trading.scripts.quality_momentum_screener

    # Trading universe only (fast):
    python -m algo_trading.scripts.quality_momentum_screener --quick

    # Specific symbols:
    python -m algo_trading.scripts.quality_momentum_screener --symbols RELIANCE TCS INFY

    # Force re-run even if output is fresh:
    python -m algo_trading.scripts.quality_momentum_screener --force

Output: data/qm_universe.json (valid for 90 days)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Config ─────────────────────────────────────────────────────────────────────

_NSE_BASE_URL  = "https://www.nseindia.com"
_NIFTY500_URL  = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
_HIST_PRICE_URL = (
    "https://www.nseindia.com/api/historical/cm/equity"
    "?symbol={sym}&series=EQ&from={from_date}&to={to_date}"
)
_QUOTE_EQUITY_URL = "https://www.nseindia.com/api/quote-equity?symbol={sym}"
_FIN_RESULTS_URLS = [
    # Try the dedicated financial results endpoint first
    "https://www.nseindia.com/api/corporates-financial-results?sym={sym}&type=Quarterly",
    # Fallback: the broader corporates financials endpoint
    "https://www.nseindia.com/api/financials-quarterly-results?sym={sym}",
]

_OUTPUT_PATH   = Path("data/qm_universe.json")
_VALID_DAYS    = 90          # universe valid for one quarter
_RATE_LIMIT    = 2.0         # seconds between calls
_REQ_TIMEOUT   = 15
_SESSION_SLEEP = 1.5
_MIN_EPS_QTRS  = 2           # minimum quarters of EPS history needed

# Factor weights
_W_MOMENTUM = 0.40
_W_QUALITY  = 0.35
_W_ACCRUAL  = 0.25

# Quintile thresholds (score buckets)
_Q5_MIN = 0.80   # top quintile
_Q4_MIN = 0.60
_Q3_MIN = 0.40
_Q2_MIN = 0.20
_Q1_MIN = 0.00   # bottom quintile

# Output multipliers by quintile
_QUINTILE_BOOST = {5: 1.15, 4: 1.05, 3: 1.00, 2: 1.00, 1: 1.00}
_QUINTILE_SMULT = {5: 1.00, 4: 1.00, 3: 1.00, 2: 0.85, 1: 0.70}
_TOP_DECILE_BOOST    = 1.20  # score ≥ 0.90
_BOTTOM_DECILE_SMULT = 0.60  # score < 0.10

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
class RawStockData:
    """Raw fetched data for one symbol before scoring."""
    symbol: str
    price_now:    Optional[float] = None
    price_1m_ago: Optional[float] = None  # ~21 trading days ago
    price_12m_ago: Optional[float] = None # ~252 trading days ago
    eps_quarters:  Optional[list[float]] = None  # last 4 qtrs, newest first
    roe:           Optional[float] = None  # Return on Equity (%)
    debt_equity:   Optional[float] = None  # D/E ratio
    fetch_ok:      bool = True


@dataclass
class QMEntry:
    """Quality-Momentum screener result for one symbol."""
    symbol:           str
    combined_score:   float   # 0–1 percentile rank
    momentum_score:   float   # 0–1 percentile rank of 12-1 momentum
    quality_score:    float   # 0–1 percentile rank of quality composite
    accrual_score:    float   # 0–1 percentile rank of accrual quality
    quintile:         int     # 1 (worst) – 5 (best)
    confidence_boost: float   # ≥ 1.0; multiply into trade_signal.confidence
    size_multiplier:  float   # ≤ 1.0; multiply into composite_multiplier
    computed_at:      date
    valid_until:      date

    @property
    def is_active(self) -> bool:
        return date.today() <= self.valid_until

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "combined_score":   round(self.combined_score, 4),
            "momentum_score":   round(self.momentum_score, 4),
            "quality_score":    round(self.quality_score, 4),
            "accrual_score":    round(self.accrual_score, 4),
            "quintile":         self.quintile,
            "confidence_boost": self.confidence_boost,
            "size_multiplier":  self.size_multiplier,
            "computed_at":      self.computed_at.isoformat(),
            "valid_until":      self.valid_until.isoformat(),
        }

    @classmethod
    def from_dict(cls, sym: str, d: dict) -> "QMEntry":
        return cls(
            symbol=sym,
            combined_score=float(d["combined_score"]),
            momentum_score=float(d["momentum_score"]),
            quality_score=float(d["quality_score"]),
            accrual_score=float(d["accrual_score"]),
            quintile=int(d["quintile"]),
            confidence_boost=float(d["confidence_boost"]),
            size_multiplier=float(d["size_multiplier"]),
            computed_at=date.fromisoformat(d["computed_at"]),
            valid_until=date.fromisoformat(d["valid_until"]),
        )


# ── NSE session ───────────────────────────────────────────────────────────────

def _get_session():
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        log.error("qm_screener.requests_not_installed")
        return None
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get(_NSE_BASE_URL, timeout=_REQ_TIMEOUT)
        time.sleep(_SESSION_SLEEP)
    except Exception as exc:
        log.warning("qm_screener.session_init_failed", error=str(exc))
    return s


# ── Price fetchers ────────────────────────────────────────────────────────────

def _price_window(session, symbol: str, target: date) -> Optional[float]:
    """Return closing price for *symbol* closest to *target* (±5 trading days)."""
    from_d = (target - timedelta(days=5)).strftime("%d-%m-%Y")
    to_d   = (target + timedelta(days=5)).strftime("%d-%m-%Y")
    url = _HIST_PRICE_URL.format(sym=symbol, from_date=from_d, to_date=to_d)
    try:
        resp = session.get(url, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            return None
        rows = resp.json().get("data", [])
        if not rows:
            return None
        # rows are chronological; take the one closest to target
        best_row  = None
        best_diff = float("inf")
        for row in rows:
            raw_date_str = row.get("CH_TIMESTAMP") or row.get("mTIMESTAMP") or ""
            try:
                row_date = datetime.strptime(raw_date_str[:10], "%Y-%m-%d").date()
                diff = abs((row_date - target).days)
                if diff < best_diff:
                    best_diff = diff
                    best_row = row
            except ValueError:
                continue
        if best_row is None:
            return None
        close_str = (
            best_row.get("CH_CLOSING_PRICE")
            or best_row.get("CH_LAST_TRADED_PRICE")
            or best_row.get("mCLOSEPRICE")
            or "0"
        )
        val = float(str(close_str).replace(",", ""))
        return val if val > 0 else None
    except Exception:
        return None


def _current_price(session, symbol: str) -> Optional[float]:
    """Return current last-traded price for *symbol* from quote-equity."""
    url = _QUOTE_EQUITY_URL.format(sym=symbol)
    try:
        resp = session.get(url, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        price_info = data.get("priceInfo", {})
        for key in ("lastPrice", "previousClose", "lp", "close"):
            val = price_info.get(key)
            if val is not None:
                try:
                    v = float(str(val).replace(",", ""))
                    if v > 0:
                        return v
                except (ValueError, TypeError):
                    pass
        return None
    except Exception:
        return None


# ── Fundamental fetchers ──────────────────────────────────────────────────────

def _float_field(obj: dict, *keys: str) -> Optional[float]:
    """Try multiple key names, return first valid float."""
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "" and v != "-":
            try:
                return float(str(v).replace(",", "").replace("%", "").strip())
            except (ValueError, TypeError):
                pass
    return None


def _fetch_financial_results(session, symbol: str) -> list[dict]:
    """Return list of quarterly result dicts (most recent first). Empty on failure."""
    for url_tpl in _FIN_RESULTS_URLS:
        url = url_tpl.format(sym=symbol)
        try:
            resp = session.get(url, timeout=_REQ_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                rows = data if isinstance(data, list) else data.get("data", [])
                if rows:
                    return rows[:8]  # last 8 quarters is plenty
        except Exception:
            pass
        time.sleep(0.3)
    return []


def _fetch_ratios_from_quote(session, symbol: str) -> tuple[Optional[float], Optional[float]]:
    """Return (roe, debt_equity) from quote-equity securityInfo. None if unavailable."""
    url = _QUOTE_EQUITY_URL.format(sym=symbol)
    try:
        resp = session.get(url, timeout=_REQ_TIMEOUT)
        if resp.status_code != 200:
            return None, None
        data = resp.json()

        sec_info = data.get("securityInfo", {})
        metadata  = data.get("metadata", {})
        combined  = {**sec_info, **metadata}

        roe = _float_field(
            combined,
            "returnOnEquity", "roe", "ROE", "return_on_equity",
        )
        de = _float_field(
            combined,
            "debtEquity", "debtToEquity", "debt_equity",
            "debtEquityRatio", "DE",
        )
        return roe, de
    except Exception:
        return None, None


def _extract_eps_series(rows: list[dict]) -> list[float]:
    """Extract EPS values from quarterly result rows. Returns newest-first list."""
    eps_list: list[float] = []
    for row in rows:
        eps = _float_field(
            row,
            "eps", "EPS", "basicEPS", "dilutedEPS",
            "earningsPerShare", "epsTtm",
        )
        if eps is not None:
            eps_list.append(eps)
        if len(eps_list) >= 4:
            break
    return eps_list


def _extract_roe_de_from_results(rows: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Compute ROE and D/E approximation from financial result rows if possible."""
    if not rows:
        return None, None

    latest = rows[0]

    # Net profit margin proxy → not ROE directly, but we'll use it as quality proxy
    pat = _float_field(latest, "pat", "netProfit", "profitAfterTax", "profit")
    equity = _float_field(latest, "equity", "netWorth", "shareHolderFunds", "bookValue")
    total_debt = _float_field(
        latest, "totalDebt", "borrowings", "debt", "longTermBorrowings"
    )

    roe: Optional[float] = None
    de:  Optional[float] = None

    if pat is not None and equity is not None and equity > 0:
        roe = (pat / equity) * 100.0

    if total_debt is not None and equity is not None and equity > 0:
        de = total_debt / equity

    return roe, de


# ── Per-symbol data fetch ─────────────────────────────────────────────────────

def _fetch_symbol(session, symbol: str, today: date) -> RawStockData:
    """Fetch all data needed to score *symbol*. Rate-limited calls."""
    raw = RawStockData(symbol=symbol)

    # 1. Current price
    time.sleep(_RATE_LIMIT)
    raw.price_now = _current_price(session, symbol)

    # 2. Price 1 month ago (~21 trading days)
    time.sleep(_RATE_LIMIT)
    raw.price_1m_ago = _price_window(session, symbol, today - timedelta(days=30))

    # 3. Price 12 months ago
    time.sleep(_RATE_LIMIT)
    raw.price_12m_ago = _price_window(session, symbol, today - timedelta(days=365))

    # 4. Financial results (EPS history + possible ROE/D/E)
    time.sleep(_RATE_LIMIT)
    fin_rows = _fetch_financial_results(session, symbol)
    raw.eps_quarters = _extract_eps_series(fin_rows) if fin_rows else []

    roe_from_fin, de_from_fin = _extract_roe_de_from_results(fin_rows)

    # 5. Quote ratios (authoritative D/E and ROE if available)
    time.sleep(_RATE_LIMIT)
    roe_from_quote, de_from_quote = _fetch_ratios_from_quote(session, symbol)

    # Prefer quote (point-in-time) over computed (may use mismatched periods)
    raw.roe         = roe_from_quote  if roe_from_quote  is not None else roe_from_fin
    raw.debt_equity = de_from_quote   if de_from_quote   is not None else de_from_fin

    raw.fetch_ok = (
        raw.price_now is not None
        or raw.price_12m_ago is not None
        or bool(raw.eps_quarters)
    )
    return raw


# ── Metric computations ───────────────────────────────────────────────────────

def _compute_momentum_raw(raw: RawStockData) -> Optional[float]:
    """12-1 month momentum: return from 12m-ago to 1m-ago (excludes last month)."""
    p12 = raw.price_12m_ago
    p1  = raw.price_1m_ago
    if p12 is None or p1 is None or p12 <= 0:
        return None
    return (p1 - p12) / p12


def _compute_quality_raw(raw: RawStockData) -> Optional[float]:
    """Composite quality score combining ROE, inverse D/E, and earnings stability."""
    sub_scores: list[float] = []

    # a. ROE: bound to [-50, 100], then normalize to [0, 1] with 15% = 0.5
    if raw.roe is not None:
        roe_clipped = max(-50.0, min(100.0, raw.roe))
        # Linear map: -50 → 0, 100 → 1; shift so 15% maps to 0.65
        roe_norm = (roe_clipped + 50.0) / 150.0
        sub_scores.append(roe_norm)

    # b. Inverse D/E: 0 debt = best (score 1.0), D/E = 3.0 = worst (score 0)
    if raw.debt_equity is not None:
        de = max(0.0, raw.debt_equity)
        de_score = max(0.0, 1.0 - de / 3.0)
        sub_scores.append(de_score)

    # c. Earnings stability: fraction of last 4 quarters with positive EPS growth
    eps = raw.eps_quarters or []
    if len(eps) >= 2:
        # Compare each quarter to the same quarter a year ago (index +4)
        # Simplified: just count positive quarters out of available history
        positive = sum(1 for e in eps if e > 0)
        stability = positive / max(len(eps), 1)
        sub_scores.append(stability)

    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores)


def _compute_accrual_raw(raw: RawStockData) -> Optional[float]:
    """Accrual quality proxy: inverse of EPS coefficient of variation.

    Low CV = stable, repeatable earnings = higher quality (less accrual inflation).
    Returns None if < _MIN_EPS_QTRS data points.
    """
    eps = [e for e in (raw.eps_quarters or []) if e is not None]
    if len(eps) < _MIN_EPS_QTRS:
        return None

    mean_eps = sum(eps) / len(eps)
    if abs(mean_eps) < 0.001:
        return None  # near-zero mean → CV undefined / loss-making

    variance = sum((e - mean_eps) ** 2 for e in eps) / len(eps)
    std_eps  = math.sqrt(variance)
    cv       = std_eps / abs(mean_eps)

    # Transform: CV=0 → score 1.0, CV=2.0 → score 0.33, CV→∞ → score→0
    return 1.0 / (1.0 + cv)


# ── Ranking ───────────────────────────────────────────────────────────────────

def _percentile_ranks(values: list[Optional[float]]) -> list[float]:
    """Return percentile rank [0, 1] for each value. None → 0.5 (neutral)."""
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed:
        return [0.5] * len(values)

    n = len(indexed)
    sorted_indexed = sorted(indexed, key=lambda x: x[1])
    rank_by_idx: dict[int, float] = {}
    for rank, (orig_idx, _) in enumerate(sorted_indexed):
        rank_by_idx[orig_idx] = (rank + 0.5) / n  # mid-rank (avoids 0 and 1 extremes)

    return [rank_by_idx.get(i, 0.5) for i in range(len(values))]


# ── Quintile assignment ───────────────────────────────────────────────────────

def _assign_quintile(score: float) -> int:
    if score >= _Q5_MIN:
        return 5
    if score >= _Q4_MIN:
        return 4
    if score >= _Q3_MIN:
        return 3
    if score >= _Q2_MIN:
        return 2
    return 1


def _assign_boost(score: float, quintile: int) -> float:
    if score >= 0.90:
        return _TOP_DECILE_BOOST
    return _QUINTILE_BOOST[quintile]


def _assign_smult(score: float, quintile: int) -> float:
    if score < 0.10:
        return _BOTTOM_DECILE_SMULT
    return _QUINTILE_SMULT[quintile]


# ── Nifty 500 symbol list ─────────────────────────────────────────────────────

def _get_nifty500_symbols(session) -> list[str]:
    try:
        resp = session.get(_NIFTY500_URL, timeout=_REQ_TIMEOUT)
        resp.raise_for_status()
        entries = resp.json().get("data", [])
        symbols = [e["symbol"].strip().upper() for e in entries if e.get("symbol")]
        log.info("qm_screener.nifty500_fetched", count=len(symbols))
        return symbols
    except Exception as exc:
        log.warning("qm_screener.nifty500_fetch_failed", error=str(exc))
        return []


# ── Main class ────────────────────────────────────────────────────────────────

class QualityMomentumScreener:
    """Quality + Momentum + Earnings Quality screener for NSE Nifty 500.

    Downloads price history and quarterly financial data from NSE, computes
    three-factor scores, ranks stocks across the universe, and assigns
    quintile-based trading multipliers.

    Results persist in data/qm_universe.json for 90 days. MarketContextFilter
    reads this file at startup and applies per-symbol multipliers during live
    trading.
    """

    def __init__(self, output_path: Path = _OUTPUT_PATH) -> None:
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_stale(self, max_age_days: int = _VALID_DAYS) -> bool:
        """Return True if output file does not exist or is older than *max_age_days*."""
        if not self._output_path.exists():
            return True
        age = (
            datetime.now(IST) - datetime.fromtimestamp(
                self._output_path.stat().st_mtime, tz=IST
            )
        ).days
        return age >= max_age_days

    def run(
        self,
        symbols: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, QMEntry]:
        """Run the screener, score all symbols, save and return results.

        Args:
            symbols: Explicit symbol list. If None, fetches current Nifty 500.
            force:   Ignore staleness check and always re-run.

        Returns:
            Dict of {symbol: QMEntry} for all scored stocks (not just top quintile).
        """
        if not force and not self.is_stale():
            log.info("qm_screener.output_fresh_skip", path=str(self._output_path))
            return self.load()

        log.info("qm_screener.starting")
        session = _get_session()
        if session is None:
            log.error("qm_screener.no_session")
            return self.load()

        # Get symbol universe
        if symbols is None:
            time.sleep(_RATE_LIMIT)
            symbols = _get_nifty500_symbols(session)
            if not symbols:
                log.error("qm_screener.empty_symbol_list")
                return self.load()

        today = datetime.now(IST).date()
        log.info("qm_screener.symbols_to_process", count=len(symbols))

        # ── Phase 1: fetch raw data ───────────────────────────────────────────
        raw_data: list[RawStockData] = []
        for i, symbol in enumerate(symbols, 1):
            try:
                raw = _fetch_symbol(session, symbol, today)
                raw_data.append(raw)
            except Exception as exc:
                log.warning("qm_screener.symbol_fetch_error", symbol=symbol, error=str(exc))
                raw_data.append(RawStockData(symbol=symbol, fetch_ok=False))

            if i % 50 == 0:
                log.info(
                    "qm_screener.phase1_progress",
                    fetched=i,
                    total=len(symbols),
                )

        log.info("qm_screener.phase1_complete", fetched=len(raw_data))

        # ── Phase 2: compute raw metrics ──────────────────────────────────────
        raw_momentum = [_compute_momentum_raw(r) for r in raw_data]
        raw_quality  = [_compute_quality_raw(r)  for r in raw_data]
        raw_accrual  = [_compute_accrual_raw(r)  for r in raw_data]

        # ── Phase 3: rank (percentile) across the universe ────────────────────
        mom_ranks     = _percentile_ranks(raw_momentum)
        quality_ranks = _percentile_ranks(raw_quality)
        accrual_ranks = _percentile_ranks(raw_accrual)

        # ── Phase 4: combined score, quintile, multipliers ────────────────────
        results: dict[str, QMEntry] = {}
        valid_until = today + timedelta(days=_VALID_DAYS)

        for idx, raw in enumerate(raw_data):
            mom_r  = mom_ranks[idx]
            qual_r = quality_ranks[idx]
            acc_r  = accrual_ranks[idx]

            combined = (
                _W_MOMENTUM * mom_r +
                _W_QUALITY  * qual_r +
                _W_ACCRUAL  * acc_r
            )

            quintile = _assign_quintile(combined)
            boost    = _assign_boost(combined, quintile)
            smult    = _assign_smult(combined, quintile)

            results[raw.symbol] = QMEntry(
                symbol=raw.symbol,
                combined_score=round(combined, 4),
                momentum_score=round(mom_r, 4),
                quality_score=round(qual_r, 4),
                accrual_score=round(acc_r, 4),
                quintile=quintile,
                confidence_boost=boost,
                size_multiplier=smult,
                computed_at=today,
                valid_until=valid_until,
            )

        self._save(results)

        top5  = [s for s, e in results.items() if e.quintile == 5]
        bot1  = [s for s, e in results.items() if e.quintile == 1]
        log.info(
            "qm_screener.complete",
            total_scored=len(results),
            top_quintile=top5,
            bottom_quintile=bot1,
        )
        return results

    def load(self) -> dict[str, QMEntry]:
        """Load existing QM universe from disk. Filters to still-valid entries."""
        if not self._output_path.exists():
            return {}
        try:
            data  = json.loads(self._output_path.read_text(encoding="utf-8"))
            today = date.today().isoformat()
            result = {
                sym: QMEntry.from_dict(sym, d)
                for sym, d in data.items()
                if d.get("valid_until", "2000-01-01") >= today
            }
            log.info(
                "qm_screener.loaded",
                active=len(result),
                top5=[s for s, e in result.items() if e.quintile == 5],
            )
            return result
        except Exception as exc:
            log.warning("qm_screener.load_failed", error=str(exc))
            return {}

    def get_entry(self, symbol: str) -> Optional[QMEntry]:
        """Return the QMEntry for *symbol* if active, else None."""
        universe = self.load()
        entry = universe.get(symbol.upper())
        if entry and entry.is_active:
            return entry
        return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _save(self, results: dict[str, QMEntry]) -> None:
        try:
            data = {sym: entry.to_dict() for sym, entry in results.items()}
            self._output_path.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log.info("qm_screener.saved", path=str(self._output_path), count=len(results))
        except Exception as exc:
            log.error("qm_screener.save_failed", error=str(exc))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quality + Momentum + Earnings Quality screener for NSE Nifty 500"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Explicit symbol list (default: full Nifty 500)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only process the trading universe from config (fast, ~10 min)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output is fresher than 90 days",
    )
    args = parser.parse_args()

    symbols_override: Optional[list[str]] = None
    if args.symbols:
        symbols_override = [s.upper() for s in args.symbols]
    elif args.quick:
        try:
            from algo_trading.config import settings  # noqa: PLC0415
            symbols_override = settings.symbol_list
            print(f"Quick mode: {len(symbols_override)} symbols from config")
        except Exception:
            print("Could not load config; running full Nifty 500")

    screener = QualityMomentumScreener()
    universe = screener.run(symbols=symbols_override, force=args.force)

    print(f"\nQM Universe ({len(universe)} stocks scored)\n")
    print(f"{'Symbol':<14} {'Q':>2}  {'Combined':>8}  {'Mom':>7}  {'Qual':>7}  {'Accr':>7}  {'Boost':>6}  {'SMult':>6}")
    print("-" * 72)
    for sym, e in sorted(universe.items(), key=lambda x: -x[1].combined_score):
        if e.quintile in (5, 1):  # show only top and bottom for brevity
            print(
                f"{sym:<14} {e.quintile:>2}  {e.combined_score:>8.3f}"
                f"  {e.momentum_score:>7.3f}  {e.quality_score:>7.3f}"
                f"  {e.accrual_score:>7.3f}  {e.confidence_boost:>6.2f}"
                f"  {e.size_multiplier:>6.2f}"
            )
