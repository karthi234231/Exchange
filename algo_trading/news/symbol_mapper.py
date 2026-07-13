"""
news/symbol_mapper.py — Maps company names in news to NSE symbols.

Uses exact match first, then alias lookup, then fuzzy match.
Covers NIFTY 50 + NIFTY Next 50 (100 stocks covers 85% of NSE news flow).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias dictionary — maps lowercase company names / tickers → NSE symbol
# ---------------------------------------------------------------------------

_NIFTY100_ALIASES: dict[str, str] = {
    # ── Direct NSE symbols (map to themselves) ────────────────────────────
    "RELIANCE":   "RELIANCE",
    "TCS":        "TCS",
    "HDFCBANK":   "HDFCBANK",
    "INFY":       "INFY",
    "ICICIBANK":  "ICICIBANK",
    "SBIN":       "SBIN",
    "BHARTIARTL": "BHARTIARTL",
    "WIPRO":      "WIPRO",
    "AXISBANK":   "AXISBANK",
    "KOTAKBANK":  "KOTAKBANK",
    "MARUTI":     "MARUTI",
    "SUNPHARMA":  "SUNPHARMA",
    "BAJFINANCE": "BAJFINANCE",
    "LT":         "LT",
    "HINDUNILVR": "HINDUNILVR",
    "ITC":        "ITC",
    "ASIANPAINT": "ASIANPAINT",
    "ULTRACEMCO": "ULTRACEMCO",
    "TITAN":      "TITAN",
    "NESTLEIND":  "NESTLEIND",
    "TATAMOTORS": "TATAMOTORS",
    "TATASTEEL":  "TATASTEEL",
    "POWERGRID":  "POWERGRID",
    "NTPC":       "NTPC",
    "ONGC":       "ONGC",
    "DRREDDY":    "DRREDDY",
    "CIPLA":      "CIPLA",
    "DIVISLAB":   "DIVISLAB",
    "APOLLOHOSP": "APOLLOHOSP",
    "ADANIENT":   "ADANIENT",
    "ADANIPORTS": "ADANIPORTS",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "EICHERMOT":  "EICHERMOT",
    "M&M":        "M&M",
    "HEROMOTOCO": "HEROMOTOCO",
    "INDUSINDBK": "INDUSINDBK",
    "TECHM":      "TECHM",
    "HCLTECH":    "HCLTECH",
    "SHREECEM":   "SHREECEM",
    "GRASIM":     "GRASIM",
    "JSWSTEEL":   "JSWSTEEL",
    "DMART":      "DMART",
    "TATACONSUM": "TATACONSUM",
    "BRITANNIA":  "BRITANNIA",
    "HDFCLIFE":   "HDFCLIFE",
    "SBILIFE":    "SBILIFE",
    "BAJAJFINSV": "BAJAJFINSV",
    "BPCL":       "BPCL",
    "COALINDIA":  "COALINDIA",
    "UPL":        "UPL",
    "VEDL":       "VEDL",

    # ── Full company names (lowercase) ────────────────────────────────────
    "reliance industries":         "RELIANCE",
    "reliance":                    "RELIANCE",
    "tata consultancy services":   "TCS",
    "tata consultancy":            "TCS",
    "hdfc bank":                   "HDFCBANK",
    "infosys":                     "INFY",
    "infosys limited":             "INFY",
    "icici bank":                  "ICICIBANK",
    "state bank of india":         "SBIN",
    "sbi":                         "SBIN",
    "bharti airtel":               "BHARTIARTL",
    "airtel":                      "BHARTIARTL",
    "wipro":                       "WIPRO",
    "axis bank":                   "AXISBANK",
    "kotak mahindra bank":         "KOTAKBANK",
    "kotak bank":                  "KOTAKBANK",
    "maruti suzuki":               "MARUTI",
    "maruti":                      "MARUTI",
    "sun pharma":                  "SUNPHARMA",
    "sun pharmaceutical":          "SUNPHARMA",
    "sun pharmaceuticals":         "SUNPHARMA",
    "bajaj finance":               "BAJFINANCE",
    "larsen & toubro":             "LT",
    "larsen and toubro":           "LT",
    "l&t":                         "LT",
    "hindustan unilever":          "HINDUNILVR",
    "hul":                         "HINDUNILVR",
    "itc":                         "ITC",
    "itc limited":                 "ITC",
    "asian paints":                "ASIANPAINT",
    "ultratech cement":            "ULTRACEMCO",
    "titan":                       "TITAN",
    "titan company":               "TITAN",
    "nestle india":                "NESTLEIND",
    "nestle":                      "NESTLEIND",
    "tata motors":                 "TATAMOTORS",
    "tata steel":                  "TATASTEEL",
    "power grid":                  "POWERGRID",
    "power grid corporation":      "POWERGRID",
    "ntpc":                        "NTPC",
    "ongc":                        "ONGC",
    "oil and natural gas":         "ONGC",
    "oil and natural gas corporation": "ONGC",
    "dr reddy":                    "DRREDDY",
    "dr reddys":                   "DRREDDY",
    "dr. reddy's":                 "DRREDDY",
    "cipla":                       "CIPLA",
    "divis laboratories":          "DIVISLAB",
    "divi's laboratories":         "DIVISLAB",
    "apollo hospitals":            "APOLLOHOSP",
    "adani enterprises":           "ADANIENT",
    "adani ports":                 "ADANIPORTS",
    "adani ports and sez":         "ADANIPORTS",
    "bajaj auto":                  "BAJAJ-AUTO",
    "eicher motors":               "EICHERMOT",
    "mahindra & mahindra":         "M&M",
    "mahindra and mahindra":       "M&M",
    "m&m":                         "M&M",
    "hero motocorp":               "HEROMOTOCO",
    "hero moto":                   "HEROMOTOCO",
    "indusind bank":               "INDUSINDBK",
    "tech mahindra":               "TECHM",
    "hcl technologies":            "HCLTECH",
    "hcl tech":                    "HCLTECH",
    "shree cement":                "SHREECEM",
    "grasim industries":           "GRASIM",
    "jsw steel":                   "JSWSTEEL",
    "dmart":                       "DMART",
    "avenue supermarts":           "DMART",
    "tata consumer":               "TATACONSUM",
    "tata consumer products":      "TATACONSUM",
    "britannia":                   "BRITANNIA",
    "britannia industries":        "BRITANNIA",
    "hdfc life":                   "HDFCLIFE",
    "hdfc life insurance":         "HDFCLIFE",
    "sbi life":                    "SBILIFE",
    "sbi life insurance":          "SBILIFE",
    "bajaj finserv":               "BAJAJFINSV",
    "bpcl":                        "BPCL",
    "bharat petroleum":            "BPCL",
    "coal india":                  "COALINDIA",
    "vedanta":                     "VEDL",
    # ── Index names (not tradeable stocks but referenced in news) ─────────
    "nifty":                       "NIFTY50",
    "nifty 50":                    "NIFTY50",
    "nifty50":                     "NIFTY50",
    "sensex":                      "SENSEX",
    "banknifty":                   "BANKNIFTY",
    "bank nifty":                  "BANKNIFTY",
}

# Lower-cased lookup table used at runtime (keys are already lowercase above,
# but we build it explicitly so callers can add entries without worrying about case)
_ALIAS_LOWER: dict[str, str] = {k.lower(): v for k, v in _NIFTY100_ALIASES.items()}

# ---------------------------------------------------------------------------
# NIFTY 50 constituents (as of 2026)
# ---------------------------------------------------------------------------

_NIFTY50_SYMBOLS: frozenset[str] = frozenset({
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "AXISBANK",
    "KOTAKBANK", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "WIPRO", "TECHM", "HCLTECH",
    "NTPC", "POWERGRID", "ULTRACEMCO", "NESTLEIND", "DIVISLAB",
    "ONGC", "BAJAJ-AUTO", "EICHERMOT", "TATAMOTORS", "TATASTEEL",
    "HEROMOTOCO", "CIPLA", "DRREDDY", "APOLLOHOSP", "ADANIPORTS",
    "ADANIENT", "JSWSTEEL", "GRASIM", "M&M", "INDUSINDBK",
    "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "BPCL", "COALINDIA",
    "BRITANNIA", "TATACONSUM", "DMART", "SHREECEM", "UPL",
})

# Keywords indicating macro / market-wide news
_MACRO_TERMS: frozenset[str] = frozenset({
    "rbi", "repo rate", "reverse repo", "monetary policy committee",
    "monetary policy", "mpc", "inflation", "cpi", "wpi",
    "budget", "union budget", "gst", "goods and services tax",
    "gdp", "fiscal deficit", "current account deficit",
    "forex", "rupee", "sebi", "capital markets regulator",
    "fii", "fpi", "foreign portfolio", "rate cut", "rate hike",
    "interest rate", "liquidity",
})


# ---------------------------------------------------------------------------
# SymbolMapper
# ---------------------------------------------------------------------------

class SymbolMapper:
    """
    Resolves news text + symbol hints to canonical NSE ticker symbols.

    Resolution order:
      1. Exact uppercase match of each hint against _ALIAS_LOWER keys.
      2. Alias lookup on lowercase hint and on words extracted from text.
      3. Fuzzy match via rapidfuzz (score_cutoff=80), graceful fallback if not installed.
    """

    def __init__(self) -> None:
        self._fuzzy_available = self._check_rapidfuzz()

    @staticmethod
    def _check_rapidfuzz() -> bool:
        try:
            import rapidfuzz  # noqa: F401
            return True
        except ImportError:
            log.debug("rapidfuzz_not_installed", detail="fuzzy matching disabled")
            return False

    def resolve(self, text: str, hints: list[str]) -> list[str]:
        """
        Return a deduplicated list of matched NSE symbols (max 5).

        Parameters
        ----------
        text:
            Full headline + body text of the article.
        hints:
            Symbol hints extracted by the scraper (may be raw NSE symbols or
            any all-caps tokens from the article).
        """
        matched: list[str] = []
        seen: set[str] = set()

        def _add(sym: str) -> None:
            if sym not in seen:
                seen.add(sym)
                matched.append(sym)

        # ── Pass 1: exact direct-symbol lookup ────────────────────────────────
        for hint in hints:
            upper = hint.strip().upper()
            found = _ALIAS_LOWER.get(upper.lower())
            if found:
                _add(found)

        # ── Pass 2: alias lookup on hints + on words/phrases from text ────────
        text_lower = text.lower()

        # Check multi-word aliases against the full text (longest first to avoid
        # matching a shorter alias before a longer one)
        for alias_key in sorted(_ALIAS_LOWER, key=len, reverse=True):
            if alias_key in text_lower:
                _add(_ALIAS_LOWER[alias_key])
                if len(matched) >= 5:
                    return matched

        # ── Pass 3: fuzzy match (rapidfuzz) ──────────────────────────────────
        if self._fuzzy_available and len(matched) < 5:
            try:
                from rapidfuzz import process as rp  # type: ignore[import]
                alias_keys = list(_ALIAS_LOWER.keys())
                result = rp.extractOne(
                    text_lower,
                    alias_keys,
                    score_cutoff=80,
                )
                if result is not None:
                    best_key = result[0]
                    _add(_ALIAS_LOWER[best_key])
            except Exception as exc:  # noqa: BLE001
                log.debug("fuzzy_match_failed", error=str(exc))

        return matched[:5]

    def is_macro_news(self, text: str) -> bool:
        """
        Return True if the text contains macro/market-wide keywords
        (RBI, budget, GST, inflation, repo rate, etc.).
        """
        text_lower = text.lower()
        return any(term in text_lower for term in _MACRO_TERMS)

    def is_nifty50(self, symbol: str) -> bool:
        """Return True if the symbol is a NIFTY 50 constituent."""
        return symbol.upper() in _NIFTY50_SYMBOLS
