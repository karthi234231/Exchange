"""
algo_trading/market_data/india_vix.py — India VIX regime classifier.

India VIX (NSE Volatility Index) measures expected 30-day volatility based on
Nifty 50 options prices. It is analogous to the CBOE VIX.

Regime classification:
  VIX < 12   → "low_vol"    — tight ranges, mean-reversion strategies perform
  12 ≤ VIX < 18 → "normal"  — standard regime, trend strategies work
  18 ≤ VIX < 25 → "elevated" — reduce position sizes 25%
  25 ≤ VIX < 35 → "high_vol" — reduce position sizes 50%, avoid new longs
  VIX ≥ 35   → "extreme"   — halt new entries, only close existing positions

VIX spike detection: VIX > 20-day moving average by 2 standard deviations
  → "vix_spike" — immediate position size reduction to 25%

Data source: NSE India historical VIX data (CSV download from NSE website)
ASSUMED URL — verify before use.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
import structlog

log = structlog.get_logger(__name__)

# ── URL constants (ASSUMED — verify before production use) ────────────────────

# ASSUMED: NSE India VIX historical data download endpoint.
# Check https://www.nseindia.com/market-data/india-vix for the current URL.
INDIA_VIX_URL_BASE = (
    "https://www.nseindia.com/api/historical/vixhistory"
)

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/market-data/india-vix",
}

_HTTP_TIMEOUT: int = 10  # seconds

# ── Regime boundary constants ─────────────────────────────────────────────────

_VIX_LOW_VOL_MAX: float      = 12.0
_VIX_NORMAL_MAX: float       = 18.0
_VIX_ELEVATED_MAX: float     = 25.0
_VIX_HIGH_VOL_MAX: float     = 35.0

# Position size multipliers per regime
_SIZE_MULTIPLIER: dict[str, float] = {
    "low_vol":  1.0,
    "normal":   1.0,
    "elevated": 0.75,
    "high_vol": 0.50,
    "extreme":  0.0,
}

# VIX spike: multiplier drops to 25% regardless of base regime
_SPIKE_SIZE_MULTIPLIER: float = 0.25

# NSE VIX CSV date format
_NSE_VIX_DATE_FORMAT: str = "%d-%b-%Y"

# Column names in NSE VIX CSV (based on observed NSE downloads)
_NSE_VIX_DATE_COL:  str = "Date"
_NSE_VIX_CLOSE_COL: str = "Close"


class VIXRegime(str, Enum):
    """India VIX volatility regime.

    Attributes:
        LOW_VOL: VIX < 12 — tight ranges, mean-reversion works.
        NORMAL: 12 ≤ VIX < 18 — standard trending conditions.
        ELEVATED: 18 ≤ VIX < 25 — reduce size by 25%.
        HIGH_VOL: 25 ≤ VIX < 35 — reduce size by 50%, avoid new longs.
        EXTREME: VIX ≥ 35 — halt new entries, exit only.
    """

    LOW_VOL  = "low_vol"
    NORMAL   = "normal"
    ELEVATED = "elevated"
    HIGH_VOL = "high_vol"
    EXTREME  = "extreme"


@dataclass
class VIXSnapshot:
    """Point-in-time VIX reading with derived regime and size guidance.

    Attributes:
        date: Trading date of this snapshot.
        vix_value: Raw VIX index value.
        regime: Classified :class:`VIXRegime`.
        size_multiplier: Recommended position-size multiplier (0.0–1.0).
        is_spike: ``True`` if VIX is more than 2 std dev above its 20-day MA.
    """

    date: date
    vix_value: float
    regime: VIXRegime
    size_multiplier: float
    is_spike: bool


class IndiaVIXClassifier:
    """Classifies the India VIX regime and maintains a rolling history.

    The classifier maintains a rolling deque of the last ``lookback_days``
    VIX values internally.  Call :meth:`update` as each new daily VIX
    value arrives; query :meth:`current_regime` and
    :meth:`position_size_multiplier` at any time.

    Example usage::

        clf = IndiaVIXClassifier(lookback_days=20)
        clf.load_from_csv(Path("data/raw/vix_history.csv"))
        snap = clf.update(date.today(), 22.5)
        mult = clf.position_size_multiplier()
    """

    def __init__(self, lookback_days: int = 20) -> None:
        """Initialise the classifier with an empty history buffer.

        Args:
            lookback_days: Number of recent VIX values to use when
                computing the rolling mean and standard deviation for
                spike detection.
        """
        self._lookback_days = lookback_days
        self._history: deque[tuple[date, float]] = deque(maxlen=lookback_days)
        self._latest_snapshot: VIXSnapshot | None = None

        log.debug("vix.classifier_init", lookback_days=lookback_days)

    def update(self, trade_date: date, vix_value: float) -> VIXSnapshot:
        """Record a new VIX value and return the resulting snapshot.

        Args:
            trade_date: Date of the VIX reading.
            vix_value: Raw India VIX value.

        Returns:
            :class:`VIXSnapshot` for this date with regime and size guidance.
        """
        regime = _classify_regime(vix_value)
        is_spike = self._detect_spike(vix_value)

        if is_spike:
            size_mult = _SPIKE_SIZE_MULTIPLIER
        else:
            size_mult = _SIZE_MULTIPLIER.get(regime.value, 1.0)

        snapshot = VIXSnapshot(
            date=trade_date,
            vix_value=vix_value,
            regime=regime,
            size_multiplier=size_mult,
            is_spike=is_spike,
        )

        self._history.append((trade_date, vix_value))
        self._latest_snapshot = snapshot

        log.debug(
            "vix.update",
            date=str(trade_date),
            vix=vix_value,
            regime=regime.value,
            size_mult=size_mult,
            is_spike=is_spike,
        )
        return snapshot

    def current_regime(self) -> VIXRegime:
        """Return the regime from the most recent :meth:`update` call.

        Returns:
            :class:`VIXRegime` for the latest snapshot.

        Raises:
            RuntimeError: If no :meth:`update` has been called yet.
        """
        if self._latest_snapshot is None:
            raise RuntimeError(
                "IndiaVIXClassifier has no data. Call update() or load_from_csv() first."
            )
        return self._latest_snapshot.regime

    def position_size_multiplier(self) -> float:
        """Return the recommended position-size multiplier for the current regime.

        Returns:
            Float in the range [0.0, 1.0].  A value of 0.0 means no new
            entries should be opened.

        Raises:
            RuntimeError: If no :meth:`update` has been called yet.
        """
        if self._latest_snapshot is None:
            raise RuntimeError(
                "IndiaVIXClassifier has no data. Call update() or load_from_csv() first."
            )
        return self._latest_snapshot.size_multiplier

    def load_from_csv(self, path: Path) -> None:
        """Load historical VIX data from an NSE-format CSV file.

        Expected CSV columns (from NSE VIX download):
        ``Date, Open, High, Low, Close, Previous Close, Change, % Change``

        Only ``Date`` and ``Close`` columns are used.  All rows are ingested
        oldest-first; the rolling history will reflect the most recent
        ``lookback_days`` entries after loading.

        Args:
            path: Path to the CSV file.
        """
        path = Path(path)
        if not path.exists():
            log.error("vix.csv_not_found", path=str(path))
            return

        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().splitlines()
        if not lines:
            log.warning("vix.csv_empty", path=str(path))
            return

        header = [h.strip() for h in lines[0].split(",")]

        date_idx  = _col_index(header, _NSE_VIX_DATE_COL)
        close_idx = _col_index(header, _NSE_VIX_CLOSE_COL)

        if date_idx < 0 or close_idx < 0:
            log.error(
                "vix.csv_missing_columns",
                path=str(path),
                header=header,
                required=[_NSE_VIX_DATE_COL, _NSE_VIX_CLOSE_COL],
            )
            return

        loaded = 0
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                date_str = parts[date_idx]
                close_str = parts[close_idx]
                trade_date = datetime.strptime(date_str, _NSE_VIX_DATE_FORMAT).date()
                vix_val = float(close_str.replace(",", ""))
                if math.isnan(vix_val):
                    continue
                self.update(trade_date, vix_val)
                loaded += 1
            except (ValueError, IndexError) as exc:
                log.debug("vix.csv_row_skip", line=line[:80], error=str(exc))
                continue

        log.info("vix.csv_loaded", path=str(path), rows_loaded=loaded)

    def fetch_vix(self, n_days: int = 30) -> list[VIXSnapshot]:
        """Download recent VIX data from NSE and update internal state.

        Attempts to download from the NSE API.  Returns an empty list on any
        failure — does NOT raise.

        Args:
            n_days: Number of recent trading days to request.

        Returns:
            List of :class:`VIXSnapshot` sorted oldest-first.
            Returns an empty list on any fetch failure.
        """
        url = f"{INDIA_VIX_URL_BASE}?endDate=&startDate=&numberOfDays={n_days}"
        log.info("vix.fetch_start", url=url, n_days=n_days)

        try:
            response = requests.get(
                url,
                headers=_BROWSER_HEADERS,
                timeout=_HTTP_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as exc:
            log.warning("vix.connection_error", url=url, error=str(exc))
            return []
        except requests.exceptions.Timeout:
            log.warning("vix.timeout", url=url, timeout_sec=_HTTP_TIMEOUT)
            return []
        except requests.exceptions.RequestException as exc:
            log.warning("vix.request_error", url=url, error=str(exc))
            return []

        if response.status_code == 403:
            log.warning(
                "vix.http_403",
                url=url,
                note="NSE blocking scraper — browser session may be required",
            )
            return []

        if response.status_code != 200:
            log.warning("vix.http_error", url=url, status_code=response.status_code)
            return []

        try:
            payload: Any = response.json()
        except ValueError as exc:
            log.warning("vix.json_parse_error", url=url, error=str(exc))
            return []

        snapshots = self._parse_vix_payload(payload)
        log.info("vix.fetch_complete", snapshots=len(snapshots))
        return snapshots

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_spike(self, vix_value: float) -> bool:
        """Return ``True`` if ``vix_value`` is a statistical spike vs recent history.

        A spike is defined as VIX more than 2 standard deviations above the
        rolling mean of the last ``lookback_days`` values (excluding the
        current value being tested).

        Args:
            vix_value: The new VIX value to evaluate.

        Returns:
            ``True`` if the value is a spike; ``False`` if insufficient
            history or within normal range.
        """
        if len(self._history) < 3:
            return False

        historical = [v for _, v in self._history]

        try:
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical)
        except statistics.StatisticsError:
            return False

        if stdev == 0.0:
            return False

        z_score = (vix_value - mean) / stdev
        return z_score > 2.0

    def _parse_vix_payload(self, payload: Any) -> list[VIXSnapshot]:
        """Parse the NSE VIX JSON response into snapshots.

        Args:
            payload: Parsed JSON from NSE API.

        Returns:
            List of :class:`VIXSnapshot` sorted oldest-first.
        """
        if isinstance(payload, dict):
            items = (
                payload.get("data")
                or payload.get("grapthData")
                or payload.get("vixData")
                or []
            )
        elif isinstance(payload, list):
            items = payload
        else:
            log.warning("vix.unexpected_payload_type", type=type(payload).__name__)
            return []

        snapshots: list[VIXSnapshot] = []
        for item in items:
            try:
                snap = self._parse_vix_item(item)
                if snap is not None:
                    snapshots.append(snap)
            except Exception as exc:
                log.debug("vix.item_parse_skip", item=str(item)[:120], error=str(exc))
                continue

        snapshots.sort(key=lambda s: s.date)
        return snapshots

    def _parse_vix_item(self, item: Any) -> VIXSnapshot | None:
        """Parse a single VIX data item from NSE.

        Args:
            item: A dict or list entry from the payload.

        Returns:
            :class:`VIXSnapshot` or ``None`` if unparseable.
        """
        if isinstance(item, dict):
            date_str = (
                item.get("EOD_TIMESTAMP")
                or item.get("date")
                or item.get("Date")
                or ""
            )
            vix_raw = (
                item.get("EOD_VIX_CLOSE")
                or item.get("close")
                or item.get("Close")
                or item.get("vixClose")
                or None
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Some NSE endpoints return [timestamp_ms, vix_value]
            date_str = str(item[0])
            vix_raw = item[1]
        else:
            return None

        if not date_str or vix_raw is None:
            return None

        # Parse date — might be UNIX ms timestamp or string
        try:
            if str(date_str).isdigit():
                trade_date = datetime.utcfromtimestamp(int(date_str) / 1000).date()
            else:
                trade_date = _parse_date(str(date_str))
            if trade_date is None:
                return None
        except (ValueError, OSError):
            return None

        try:
            vix_val = float(str(vix_raw).replace(",", "").strip())
            if math.isnan(vix_val):
                return None
        except (ValueError, TypeError):
            return None

        return self.update(trade_date, vix_val)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _classify_regime(vix_value: float) -> VIXRegime:
    """Map a VIX value to a :class:`VIXRegime`.

    Args:
        vix_value: Raw India VIX index value.

    Returns:
        Corresponding :class:`VIXRegime`.
    """
    if vix_value < _VIX_LOW_VOL_MAX:
        return VIXRegime.LOW_VOL
    if vix_value < _VIX_NORMAL_MAX:
        return VIXRegime.NORMAL
    if vix_value < _VIX_ELEVATED_MAX:
        return VIXRegime.ELEVATED
    if vix_value < _VIX_HIGH_VOL_MAX:
        return VIXRegime.HIGH_VOL
    return VIXRegime.EXTREME


def _parse_date(s: str) -> date | None:
    """Parse common NSE date string formats.

    Args:
        s: Raw date string.

    Returns:
        Parsed :class:`datetime.date` or ``None``.
    """
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _col_index(header: list[str], name: str) -> int:
    """Return the column index for ``name`` in ``header`` (case-insensitive).

    Args:
        header: List of column names from CSV header.
        name: Target column name.

    Returns:
        Zero-based index or -1 if not found.
    """
    name_lower = name.lower()
    for i, h in enumerate(header):
        if h.lower() == name_lower:
            return i
    return -1
