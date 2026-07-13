"""
NER pipeline for mapping news text -> NSE ticker symbols.

Strategy (4 passes, in priority order):
1. EntityRuler: custom patterns for 100+ Indian company names (faster, exact)
2. spaCy ORG entity detection: matches ORG entities -> fuzzy alias lookup
3. Uppercase token scan: fallback regex for all-caps 2-10 char tokens
4. Alias dict scan: direct substring match against known company names

Returns up to MAX_SYMBOLS_PER_ARTICLE (from config) unique NSE symbols.

Gracefully degrades to alias-only matching if spaCy is not installed.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

import structlog

from algo_trading.news_pipeline.config import cfg

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# spaCy availability flag
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy  # type: ignore[import]  # noqa: F401
    _SPACY_AVAILABLE = True
except ImportError:
    logger.warning("ner_pipeline_spacy_unavailable", fallback="alias-only matching")


# ---------------------------------------------------------------------------
# Alias dictionary (mirrors symbol_mapper._ALIAS_LOWER + extended coverage)
# Used for both spaCy ORG fallback and pure-alias mode
# ---------------------------------------------------------------------------

_ALIAS_LOWER: dict[str, str] = {
    # ── Direct NSE symbols ────────────────────────────────────────────────
    "reliance":             "RELIANCE",
    "tcs":                  "TCS",
    "hdfcbank":             "HDFCBANK",
    "infy":                 "INFY",
    "icicibank":            "ICICIBANK",
    "sbin":                 "SBIN",
    "bhartiartl":           "BHARTIARTL",
    "wipro":                "WIPRO",
    "axisbank":             "AXISBANK",
    "kotakbank":            "KOTAKBANK",
    "maruti":               "MARUTI",
    "sunpharma":            "SUNPHARMA",
    "bajfinance":           "BAJFINANCE",
    "lt":                   "LT",
    "hindunilvr":           "HINDUNILVR",
    "itc":                  "ITC",
    "asianpaint":           "ASIANPAINT",
    "ultracemco":           "ULTRACEMCO",
    "titan":                "TITAN",
    "nestleind":            "NESTLEIND",
    "tatamotors":           "TATAMOTORS",
    "tatasteel":            "TATASTEEL",
    "powergrid":            "POWERGRID",
    "ntpc":                 "NTPC",
    "ongc":                 "ONGC",
    "drreddy":              "DRREDDY",
    "cipla":                "CIPLA",
    "divislab":             "DIVISLAB",
    "apollohosp":           "APOLLOHOSP",
    "adanient":             "ADANIENT",
    "adaniports":           "ADANIPORTS",
    "bajaj-auto":           "BAJAJ-AUTO",
    "eichermot":            "EICHERMOT",
    "m&m":                  "M&M",
    "heromotoco":           "HEROMOTOCO",
    "indusindbk":           "INDUSINDBK",
    "techm":                "TECHM",
    "hcltech":              "HCLTECH",
    "shreecem":             "SHREECEM",
    "grasim":               "GRASIM",
    "jswsteel":             "JSWSTEEL",
    "dmart":                "DMART",
    "tataconsum":           "TATACONSUM",
    "britannia":            "BRITANNIA",
    "hdfclife":             "HDFCLIFE",
    "sbilife":              "SBILIFE",
    "bajajfinsv":           "BAJAJFINSV",
    "bpcl":                 "BPCL",
    "coalindia":            "COALINDIA",
    "upl":                  "UPL",
    "vedl":                 "VEDL",
    # ── NIFTY Next 50 / sector heavyweights ──────────────────────────────
    "zomato":               "ZOMATO",
    "paytm":                "PAYTM",
    "nykaa":                "NYKAA",
    "delhivery":            "DELHIVERY",
    "policybzr":            "POLICYBZR",
    "irctc":                "IRCTC",
    "abb":                  "ABB",
    "siemens":              "SIEMENS",
    "havells":              "HAVELLS",
    "piind":                "PIIND",
    "srf":                  "SRF",
    "astral":               "ASTRAL",
    "pageind":              "PAGEIND",
    "muthootfin":           "MUTHOOTFIN",
    "cholafin":             "CHOLAFIN",
    "aubank":               "AUBANK",
    "sbicard":              "SBICARD",
    "bandhanbnk":           "BANDHANBNK",
    "federalbnk":           "FEDERALBNK",
    "idfcfirstb":           "IDFCFIRSTB",
    "rblbank":              "RBLBANK",
    "ioc":                  "IOC",
    "gail":                 "GAIL",
    "bhel":                 "BHEL",
    "hpcl":                 "HPCL",
    "hindalco":             "HINDALCO",
    "tatapower":            "TATAPOWER",
    "jswenergy":            "JSWENERGY",
    "torntpharm":           "TORNTPHARM",
    "alkem":                "ALKEM",
    "lauruslabs":           "LAURUSLABS",
    # ── Index names ───────────────────────────────────────────────────────
    "nifty50":              "NIFTY50",
    "banknifty":            "BANKNIFTY",
    "sensex":               "SENSEX",

    # ── Full company names (lowercase) ────────────────────────────────────
    "reliance industries":              "RELIANCE",
    "tata consultancy services":        "TCS",
    "tata consultancy":                 "TCS",
    "hdfc bank":                        "HDFCBANK",
    "infosys":                          "INFY",
    "infosys limited":                  "INFY",
    "icici bank":                       "ICICIBANK",
    "state bank of india":              "SBIN",
    "sbi":                              "SBIN",
    "bharti airtel":                    "BHARTIARTL",
    "airtel":                           "BHARTIARTL",
    "axis bank":                        "AXISBANK",
    "kotak mahindra bank":              "KOTAKBANK",
    "kotak bank":                       "KOTAKBANK",
    "maruti suzuki":                    "MARUTI",
    "sun pharma":                       "SUNPHARMA",
    "sun pharmaceutical":               "SUNPHARMA",
    "sun pharmaceuticals":              "SUNPHARMA",
    "bajaj finance":                    "BAJFINANCE",
    "larsen & toubro":                  "LT",
    "larsen and toubro":                "LT",
    "l&t":                              "LT",
    "hindustan unilever":               "HINDUNILVR",
    "hul":                              "HINDUNILVR",
    "itc limited":                      "ITC",
    "asian paints":                     "ASIANPAINT",
    "ultratech cement":                 "ULTRACEMCO",
    "titan company":                    "TITAN",
    "nestle india":                     "NESTLEIND",
    "nestle":                           "NESTLEIND",
    "tata motors":                      "TATAMOTORS",
    "tata steel":                       "TATASTEEL",
    "power grid":                       "POWERGRID",
    "power grid corporation":           "POWERGRID",
    "oil and natural gas":              "ONGC",
    "oil and natural gas corporation":  "ONGC",
    "dr reddy":                         "DRREDDY",
    "dr reddys":                        "DRREDDY",
    "dr. reddy's":                      "DRREDDY",
    "divis laboratories":               "DIVISLAB",
    "divi's laboratories":              "DIVISLAB",
    "apollo hospitals":                 "APOLLOHOSP",
    "adani enterprises":                "ADANIENT",
    "adani ports":                      "ADANIPORTS",
    "adani ports and sez":              "ADANIPORTS",
    "bajaj auto":                       "BAJAJ-AUTO",
    "eicher motors":                    "EICHERMOT",
    "mahindra & mahindra":              "M&M",
    "mahindra and mahindra":            "M&M",
    "hero motocorp":                    "HEROMOTOCO",
    "hero moto":                        "HEROMOTOCO",
    "indusind bank":                    "INDUSINDBK",
    "tech mahindra":                    "TECHM",
    "hcl technologies":                 "HCLTECH",
    "hcl tech":                         "HCLTECH",
    "shree cement":                     "SHREECEM",
    "grasim industries":                "GRASIM",
    "jsw steel":                        "JSWSTEEL",
    "avenue supermarts":                "DMART",
    "tata consumer":                    "TATACONSUM",
    "tata consumer products":           "TATACONSUM",
    "britannia industries":             "BRITANNIA",
    "hdfc life":                        "HDFCLIFE",
    "hdfc life insurance":              "HDFCLIFE",
    "sbi life":                         "SBILIFE",
    "sbi life insurance":               "SBILIFE",
    "bajaj finserv":                    "BAJAJFINSV",
    "bharat petroleum":                 "BPCL",
    "coal india":                       "COALINDIA",
    "vedanta":                          "VEDL",
    "nifty":                            "NIFTY50",
    "nifty 50":                         "NIFTY50",
    "bank nifty":                       "BANKNIFTY",
    # ── Extended aliases for NIFTY Next 50 ───────────────────────────────
    "abb india":                        "ABB",
    "pi industries":                    "PIIND",
    "page industries":                  "PAGEIND",
    "muthoot finance":                  "MUTHOOTFIN",
    "cholamandalam":                    "CHOLAFIN",
    "cholamandalam investment":         "CHOLAFIN",
    "au small finance bank":            "AUBANK",
    "au bank":                          "AUBANK",
    "sbi cards":                        "SBICARD",
    "bandhan bank":                     "BANDHANBNK",
    "federal bank":                     "FEDERALBNK",
    "federalbank":                      "FEDERALBNK",
    "idfc first bank":                  "IDFCFIRSTB",
    "idfc first":                       "IDFCFIRSTB",
    "rbl bank":                         "RBLBANK",
    "indian oil":                       "IOC",
    "indian oil corporation":           "IOC",
    "tata power":                       "TATAPOWER",
    "jsw energy":                       "JSWENERGY",
    "torrent pharma":                   "TORNTPHARM",
    "torrent pharmaceuticals":          "TORNTPHARM",
    "laurus labs":                      "LAURUSLABS",
    "policybazaar":                     "POLICYBZR",
    "policy bazaar":                    "POLICYBZR",
}

# Pre-sort by length descending so longer multi-word aliases match before shorter ones
_ALIAS_SORTED: list[tuple[str, str]] = sorted(
    _ALIAS_LOWER.items(), key=lambda x: len(x[0]), reverse=True
)

# ---------------------------------------------------------------------------
# EntityRuler patterns for spaCy
# Each tuple: (list_of_tokens_lowercase, nse_symbol)
# ---------------------------------------------------------------------------

_COMPANY_PATTERNS: list[tuple[list[str], str]] = [
    (["reliance", "industries"], "RELIANCE"),
    (["reliance"], "RELIANCE"),
    (["tata", "consultancy", "services"], "TCS"),
    (["tcs"], "TCS"),
    (["hdfc", "bank"], "HDFCBANK"),
    (["infosys"], "INFY"),
    (["icici", "bank"], "ICICIBANK"),
    (["state", "bank", "of", "india"], "SBIN"),
    (["sbi"], "SBIN"),
    (["bharti", "airtel"], "BHARTIARTL"),
    (["airtel"], "BHARTIARTL"),
    (["wipro"], "WIPRO"),
    (["axis", "bank"], "AXISBANK"),
    (["kotak", "mahindra", "bank"], "KOTAKBANK"),
    (["kotak", "bank"], "KOTAKBANK"),
    (["maruti", "suzuki"], "MARUTI"),
    (["maruti"], "MARUTI"),
    (["sun", "pharma"], "SUNPHARMA"),
    (["sun", "pharmaceutical"], "SUNPHARMA"),
    (["bajaj", "finance"], "BAJFINANCE"),
    (["larsen", "toubro"], "LT"),
    (["l&t"], "LT"),
    (["hindustan", "unilever"], "HINDUNILVR"),
    (["hul"], "HINDUNILVR"),
    (["itc"], "ITC"),
    (["asian", "paints"], "ASIANPAINT"),
    (["ultratech", "cement"], "ULTRACEMCO"),
    (["titan"], "TITAN"),
    (["nestle", "india"], "NESTLEIND"),
    (["tata", "motors"], "TATAMOTORS"),
    (["tata", "steel"], "TATASTEEL"),
    (["power", "grid"], "POWERGRID"),
    (["ntpc"], "NTPC"),
    (["ongc"], "ONGC"),
    (["dr", "reddy"], "DRREDDY"),
    (["cipla"], "CIPLA"),
    (["divis", "laboratories"], "DIVISLAB"),
    (["apollo", "hospitals"], "APOLLOHOSP"),
    (["adani", "enterprises"], "ADANIENT"),
    (["adani", "ports"], "ADANIPORTS"),
    (["bajaj", "auto"], "BAJAJ-AUTO"),
    (["eicher", "motors"], "EICHERMOT"),
    (["mahindra", "mahindra"], "M&M"),
    (["hero", "motocorp"], "HEROMOTOCO"),
    (["indusind", "bank"], "INDUSINDBK"),
    (["tech", "mahindra"], "TECHM"),
    (["hcl", "technologies"], "HCLTECH"),
    (["hcl", "tech"], "HCLTECH"),
    (["shree", "cement"], "SHREECEM"),
    (["grasim", "industries"], "GRASIM"),
    (["jsw", "steel"], "JSWSTEEL"),
    (["avenue", "supermarts"], "DMART"),
    (["dmart"], "DMART"),
    (["tata", "consumer"], "TATACONSUM"),
    (["britannia", "industries"], "BRITANNIA"),
    (["britannia"], "BRITANNIA"),
    (["hdfc", "life"], "HDFCLIFE"),
    (["sbi", "life"], "SBILIFE"),
    (["bajaj", "finserv"], "BAJAJFINSV"),
    (["bharat", "petroleum"], "BPCL"),
    (["bpcl"], "BPCL"),
    (["coal", "india"], "COALINDIA"),
    (["vedanta"], "VEDL"),
    (["upl"], "UPL"),
    (["nifty", "50"], "NIFTY50"),
    (["banknifty"], "BANKNIFTY"),
    (["bank", "nifty"], "BANKNIFTY"),
    # NIFTY Next 50 and sector heavyweights
    (["zomato"], "ZOMATO"),
    (["paytm"], "PAYTM"),
    (["nykaa"], "NYKAA"),
    (["delhivery"], "DELHIVERY"),
    (["policybazaar"], "POLICYBZR"),
    (["policy", "bazaar"], "POLICYBZR"),
    (["irctc"], "IRCTC"),
    (["abb", "india"], "ABB"),
    (["siemens"], "SIEMENS"),
    (["havells"], "HAVELLS"),
    (["pi", "industries"], "PIIND"),
    (["srf"], "SRF"),
    (["astral"], "ASTRAL"),
    (["page", "industries"], "PAGEIND"),
    (["muthoot", "finance"], "MUTHOOTFIN"),
    (["cholamandalam"], "CHOLAFIN"),
    (["au", "small", "finance", "bank"], "AUBANK"),
    (["au", "bank"], "AUBANK"),
    (["sbi", "cards"], "SBICARD"),
    (["bandhan", "bank"], "BANDHANBNK"),
    (["federal", "bank"], "FEDERALBNK"),
    (["federalbank"], "FEDERALBNK"),
    (["idfc", "first", "bank"], "IDFCFIRSTB"),
    (["idfc", "first"], "IDFCFIRSTB"),
    (["rbl", "bank"], "RBLBANK"),
    (["indian", "oil"], "IOC"),
    (["ioc"], "IOC"),
    (["gail"], "GAIL"),
    (["bhel"], "BHEL"),
    (["hpcl"], "HPCL"),
    (["hindalco"], "HINDALCO"),
    (["tata", "power"], "TATAPOWER"),
    (["jsw", "energy"], "JSWENERGY"),
    (["torrent", "pharma"], "TORNTPHARM"),
    (["torrent", "pharmaceuticals"], "TORNTPHARM"),
    (["alkem"], "ALKEM"),
    (["laurus", "labs"], "LAURUSLABS"),
]

# ---------------------------------------------------------------------------
# Macro keywords for is_macro_news
# ---------------------------------------------------------------------------

_MACRO_KEYWORDS: frozenset[str] = frozenset({
    "rbi", "repo rate", "reverse repo", "monetary policy",
    "mpc", "inflation", "cpi", "wpi", "budget", "union budget",
    "gst", "gdp", "fiscal deficit", "current account deficit",
    "forex reserves", "rupee", "rate cut", "rate hike", "interest rate",
    "liquidity", "open market operations", "standing deposit facility",
})

# Regex for bare uppercase ticker tokens (2-10 uppercase letters, optionally
# followed by - and more uppercase letters, e.g. BAJAJ-AUTO)
_TICKER_REGEX = re.compile(r"\b([A-Z][A-Z0-9\-]{1,9})\b")

# All known canonical NSE symbols for regex pass validation
_ALL_SYMBOLS: frozenset[str] = frozenset(_ALIAS_LOWER.values())


class NERPipeline:
    """
    Extracts NSE ticker symbols from news text using spaCy + custom patterns.

    Lazy-loads spaCy model on first use (avoids startup delay).
    Thread-safe (spaCy nlp objects are thread-safe for inference).
    Instance-level lock guards lazy initialisation only.
    """

    _nlp: Optional[object] = None  # class-level shared nlp object
    _nlp_lock = threading.Lock()
    _nlp_load_attempted = False

    # -------------------------------------------------------------------------
    # Lazy model loader
    # -------------------------------------------------------------------------

    @classmethod
    def preload(cls) -> None:
        """
        Eagerly load the spaCy model on the calling thread.

        Call this at application startup (e.g. DailyOrchestrator._startup or
        Prefect flow startup) so that the first live call to extract_symbols()
        does not block for 10-30 seconds downloading/loading the model.
        """
        cls._get_nlp()

    @classmethod
    def _get_nlp(cls) -> Optional[object]:
        """
        Lazy-load spaCy model + EntityRuler. Returns None if spaCy unavailable.

        Thread-safety note: ``_nlp_load_attempted`` is only set to True AFTER
        the load completes (success or failure), so concurrent callers that see
        the flag as False will block on the lock rather than racing through with
        a None result while loading is still in progress.
        """
        # Fast path: flag True only after load is fully complete.
        if cls._nlp_load_attempted:
            return cls._nlp

        with cls._nlp_lock:
            # Double-checked locking — re-check inside the lock.
            if cls._nlp_load_attempted:
                return cls._nlp

            # NOTE: _nlp_load_attempted is set at the END of this block (after
            # the try/except), so other threads don't escape via the fast-path
            # check above while the load is still in progress.

            if not _SPACY_AVAILABLE:
                logger.warning("ner_pipeline_spacy_not_installed", detail="pure alias mode")
                cls._nlp_load_attempted = True
                return None

            try:
                import spacy  # type: ignore[import]
                try:
                    nlp = spacy.load(cfg.SPACY_MODEL, disable=["parser", "lemmatizer"])
                except OSError:
                    # Model not downloaded — use blank English pipeline
                    logger.warning(
                        "ner_pipeline_model_not_found",
                        model=cfg.SPACY_MODEL,
                        fallback="blank en pipeline",
                    )
                    nlp = spacy.blank("en")
                    # Add sentencizer so we can still run entity ruler
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")

                # Add EntityRuler before ner (if ner exists)
                ruler_config = {"overwrite_ents": True}
                if "ner" in nlp.pipe_names:
                    ruler = nlp.add_pipe(
                        "entity_ruler", config=ruler_config, before="ner"
                    )
                else:
                    ruler = nlp.add_pipe("entity_ruler", config=ruler_config)

                # Build patterns from _COMPANY_PATTERNS
                patterns = []
                for token_list, symbol in _COMPANY_PATTERNS:
                    pattern = [{"LOWER": tok} for tok in token_list]
                    patterns.append(
                        {"label": "NSE_TICKER", "pattern": pattern, "id": symbol}
                    )
                ruler.add_patterns(patterns)  # type: ignore[attr-defined]

                cls._nlp = nlp
                logger.info(
                    "ner_pipeline_spacy_loaded",
                    model=cfg.SPACY_MODEL,
                    patterns=len(patterns),
                )
            except Exception as exc:
                logger.error(
                    "ner_pipeline_spacy_load_failed",
                    error=str(exc),
                    fallback="alias-only mode",
                )
                cls._nlp = None

            # Mark load as complete only after _nlp is fully set.
            cls._nlp_load_attempted = True

        return cls._nlp

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def extract_symbols(
        self,
        text: str,
        hints: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Returns up to cfg.MAX_SYMBOLS_PER_ARTICLE unique NSE symbols.

        Pass 1 (spaCy EntityRuler): custom NSE_TICKER entities.
        Pass 2 (spaCy ORG entities): standard ORG -> alias lookup.
        Pass 3 (hints): provided uppercase tokens -> alias lookup.
        Pass 4 (regex): bare uppercase 2-10 char tokens validated against known symbols.

        Falls back gracefully to passes 3+4 if spaCy is unavailable.
        """
        max_sym = cfg.MAX_SYMBOLS_PER_ARTICLE
        seen: set[str] = set()
        results: list[str] = []

        def _add(sym: str) -> bool:
            if sym not in seen and len(results) < max_sym:
                seen.add(sym)
                results.append(sym)
                return True
            return False

        nlp = self._get_nlp()

        if nlp is not None:
            try:
                doc = nlp(text[:5000])  # cap text length for performance

                # Pass 1: EntityRuler NSE_TICKER entities
                for ent in doc.ents:
                    if ent.label_ == "NSE_TICKER" and ent.ent_id_:
                        _add(ent.ent_id_)
                    if len(results) >= max_sym:
                        return results

                # Pass 2: Generic ORG entities -> alias lookup
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        sym = _ALIAS_LOWER.get(ent.text.lower().strip())
                        if sym:
                            _add(sym)
                    if len(results) >= max_sym:
                        return results

            except Exception as exc:
                logger.warning("ner_pipeline_spacy_inference_error", error=str(exc))

        # Pass 3: hints (uppercase tokens from scraper)
        if hints:
            for hint in hints:
                sym = _ALIAS_LOWER.get(hint.strip().lower())
                if sym:
                    _add(sym)
                if len(results) >= max_sym:
                    return results

        # Pass 4: alias substring scan on lowercased text (longest alias first).
        # Short aliases (e.g. "lt" → LT, "itc" → ITC) ARE included here because
        # the alias dict is curated — false positives are handled by word-boundary
        # checking below.  A minimum of 2 chars prevents single-letter matches.
        text_lower = text.lower()
        for alias_key, symbol in _ALIAS_SORTED:
            if len(alias_key) < 2:
                continue
            # For very short aliases (≤3 chars) require word boundary to prevent
            # false positives (e.g. "lt" inside "culture" or "lt." in titles).
            if len(alias_key) <= 3:
                import re as _re
                if not _re.search(r"\b" + _re.escape(alias_key) + r"\b", text_lower):
                    continue
            elif alias_key not in text_lower:
                continue
            _add(symbol)
            if len(results) >= max_sym:
                return results

        # Pass 5: regex scan for bare uppercase tickers (2-10 chars) in original text
        for match in _TICKER_REGEX.finditer(text):
            token = match.group(1)
            if token in _ALL_SYMBOLS:
                _add(token)
            if len(results) >= max_sym:
                return results

        return results

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Returns dict of entity type -> list of entity texts found.
        Example: {"ORG": ["Reliance Industries", "SEBI"], "MONEY": ["Rs 500 crore"]}

        Falls back to an empty dict if spaCy is unavailable.
        """
        entities: dict[str, list[str]] = {}
        nlp = self._get_nlp()
        if nlp is None:
            return entities

        try:
            doc = nlp(text[:5000])
            for ent in doc.ents:
                label = ent.label_
                if label not in entities:
                    entities[label] = []
                if ent.text not in entities[label]:
                    entities[label].append(ent.text)
        except Exception as exc:
            logger.warning("ner_pipeline_extract_entities_error", error=str(exc))

        return entities

    def is_macro_news(self, text: str) -> bool:
        """Return True if text contains RBI/repo-rate/budget/inflation/GDP keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in _MACRO_KEYWORDS)
