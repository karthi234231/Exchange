"""
Event tagger: classifies news into EventType categories and extracts fine-grained tags.

Primary classification: keyword rules (fast, deterministic) reusing the exact keyword
sets and priority logic from algo_trading.news.event_classifier.EventClassifier.

Secondary signal: FinBERT label helps break ties within GENERIC events — if FinBERT
says "bearish" for an otherwise GENERIC event we escalate to check REGULATORY/LEGAL
keyword sets again before falling back.

EventTypes mirror algo_trading.news.event_classifier.EventType:
    EARNINGS, DIVIDEND, BUYBACK, M_AND_A, MANAGEMENT, REGULATORY,
    LEGAL, MACRO, IPO_LISTING, SECTOR, GENERIC

Impact levels mirror algo_trading.news.event_classifier.ImpactLevel:
    CRITICAL, HIGH, MEDIUM, LOW
"""
from __future__ import annotations

import re
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets (verbatim from algo_trading.news.event_classifier)
# Kept local so this module has zero runtime import dependency on that module.
# ---------------------------------------------------------------------------

_EARNINGS_KWORDS: frozenset[str] = frozenset([
    "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
    "net profit", "pat", "eps", "revenue", "ebitda", "net loss", "annual results",
    "q1fy", "q2fy", "q3fy", "q4fy", "profit after tax", "earnings per share",
    "operating profit", "gross profit", "turnover",
])

_DIVIDEND_KWORDS: frozenset[str] = frozenset([
    "dividend", "interim dividend", "final dividend", "dividend declared",
    "dividend announcement", "special dividend",
])

_BUYBACK_KWORDS: frozenset[str] = frozenset([
    "buyback", "buy back", "share repurchase", "share buyback",
    "repurchases shares",
])

_M_AND_A_KWORDS: frozenset[str] = frozenset([
    "acquisition", "merger", "acquires", "stake purchase", "takeover",
    "strategic investment", "joint venture", "stake acquisition",
    "buys stake", "acquires stake", "merges with", "amalgamation",
    "demerger", "hive off", "spin off", "stake sale",
])

_MANAGEMENT_KWORDS: frozenset[str] = frozenset([
    "ceo", "md", "managing director", "board appoints", "resignation",
    "steps down", "new cfo", "director", "chairman", "executive director",
    "coo", "board meeting", "management change", "appoints new",
    "elevates", "promoted to", "takes charge",
])

_REGULATORY_KWORDS: frozenset[str] = frozenset([
    "sebi", "show cause notice", "sebi order", "ban", "suspended from trading",
    "nse action", "bse action", "penalty imposed", "investigation",
    "sebi probe", "regulatory action", "adjudication order",
    "debarred", "impounded", "exchange action",
])

_LEGAL_KWORDS: frozenset[str] = frozenset([
    "fraud", "fir", "court order", "arrested", "cheating", "embezzlement",
    "money laundering", "enforcement directorate", "ed raid",
    "income tax raid", "cbi", "chargesheet", "contempt of court",
    "insolvency", "nclt", "default", "wilful defaulter",
])

_MACRO_KWORDS: frozenset[str] = frozenset([
    "rbi", "repo rate", "reverse repo", "monetary policy", "inflation",
    "cpi", "wpi", "budget", "gst", "gdp", "fiscal deficit",
    "current account", "mpc", "rate cut", "rate hike",
    "forex reserves", "rupee", "interest rate",
])

_IPO_KWORDS: frozenset[str] = frozenset([
    "ipo", "initial public offering", "listing day", "ipo allotment",
    "share allotment", "grey market premium", "gmp",
    "listing gains", "oversubscribed", "nfo",
])

_BULLISH_KWORDS: frozenset[str] = frozenset([
    "beats", "beat estimates", "record profit", "strong growth", "upgrades",
    "buyback", "dividend", "acquisition target", "bullish", "positive outlook",
    "profit surges", "revenue jumps", "raises guidance", "upgrade",
    "buy rating", "outperform", "strong buy", "multibagger",
])

_BEARISH_KWORDS: frozenset[str] = frozenset([
    "misses", "miss estimates", "net loss", "sebi ban", "fraud", "fir",
    "show cause", "downgrade", "negative outlook", "probe", "penalty",
    "loss widens", "revenue falls", "guidance cut", "sell rating",
    "underperform", "default", "bankruptcy", "insolvency",
])

_RBI_RATE_TERMS: frozenset[str] = frozenset([
    "repo rate", "reverse repo", "rate cut", "rate hike",
    "rate change", "monetary policy", "mpc decision",
])

_CEO_RESIGN_TERMS: frozenset[str] = frozenset([
    "ceo resigns", "md resigns", "ceo steps down", "md steps down",
    "chief executive resigns", "managing director resigns",
    "resignation of ceo", "resignation of md",
])

# ---------------------------------------------------------------------------
# Sector keywords for tag extraction
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, list[str]] = {
    "banking":      ["bank", "npa", "credit", "deposit", "loan", "nbfc", "casa"],
    "it":           ["software", "it services", "technology", "digital", "cloud", "saas"],
    "pharma":       ["pharma", "drug", "fda", "medicine", "therapeutic", "api", "bulk drug"],
    "auto":         ["automobile", "vehicle", "ev", "electric vehicle", "two-wheeler", "passenger vehicle"],
    "energy":       ["oil", "gas", "petroleum", "refinery", "ongc", "gail", "power"],
    "fmcg":         ["fmcg", "consumer goods", "packaged foods", "beverages", "personal care"],
    "metals":       ["steel", "aluminium", "zinc", "copper", "iron ore", "metal"],
    "realty":       ["real estate", "realty", "housing", "property", "residential"],
    "telecom":      ["telecom", "spectrum", "5g", "broadband", "tariff"],
    "insurance":    ["insurance", "premium", "claim", "life insurance", "general insurance"],
    "cement":       ["cement", "clinker", "capacity expansion"],
    "chemicals":    ["chemical", "agrochemical", "specialty chemical", "fertilizer"],
    "infra":        ["infrastructure", "road", "highway", "bridge", "nhai", "construction"],
}

# Quarter reference patterns — e.g. Q1FY27, Q2FY26
_QUARTER_REGEX = re.compile(r"\b(q[1-4]fy\d{2,4})\b", re.IGNORECASE)
_QUARTER_BARE_REGEX = re.compile(r"\b(q[1-4])\s*(fy\s*\d{2,4})?\b", re.IGNORECASE)
_FY_REGEX = re.compile(r"\b(fy\s*\d{2,4})\b", re.IGNORECASE)

# Beat/miss/inline patterns
_BEAT_REGEX = re.compile(r"\b(beat|beats|exceeded|exceeded estimates|above estimates|topped estimates)\b", re.IGNORECASE)
_MISS_REGEX = re.compile(r"\b(miss|misses|missed|below estimates|fell short|missed estimates)\b", re.IGNORECASE)
_INLINE_REGEX = re.compile(r"\b(in line|inline|as expected|met estimates|in-line)\b", re.IGNORECASE)


def _contains_any(text: str, kwords: frozenset[str]) -> bool:
    """Return True if any keyword appears in text."""
    return any(kw in text for kw in kwords)


def _detect_event_type(text: str) -> str:
    """
    Return the most specific EventType string matching the text.
    Priority order mirrors EventClassifier._detect_event_type():
    LEGAL > REGULATORY > MACRO > IPO_LISTING > EARNINGS > DIVIDEND > BUYBACK >
    M_AND_A > MANAGEMENT > GENERIC
    """
    if _contains_any(text, _LEGAL_KWORDS):
        return "LEGAL"
    if _contains_any(text, _REGULATORY_KWORDS):
        return "REGULATORY"
    if _contains_any(text, _MACRO_KWORDS):
        return "MACRO"
    if _contains_any(text, _IPO_KWORDS):
        return "IPO_LISTING"
    if _contains_any(text, _EARNINGS_KWORDS):
        return "EARNINGS"
    if _contains_any(text, _DIVIDEND_KWORDS):
        return "DIVIDEND"
    if _contains_any(text, _BUYBACK_KWORDS):
        return "BUYBACK"
    if _contains_any(text, _M_AND_A_KWORDS):
        return "M_AND_A"
    if _contains_any(text, _MANAGEMENT_KWORDS):
        return "MANAGEMENT"
    return "GENERIC"


def _compute_impact(event_type: str, sentiment_label: str, text: str) -> str:
    """
    Derive impact level from event_type + sentiment + contextual signals.
    Mirrors EventClassifier._compute_impact() logic exactly.

    Priority (first match wins):
        REGULATORY + bearish     -> CRITICAL
        LEGAL                    -> CRITICAL
        EARNINGS + "fraud"       -> CRITICAL
        EARNINGS                 -> HIGH
        M_AND_A                  -> HIGH
        MACRO + rate action      -> HIGH
        MANAGEMENT + resign      -> HIGH
        DIVIDEND / BUYBACK       -> MEDIUM
        IPO_LISTING              -> MEDIUM
        MACRO (generic)          -> MEDIUM
        REGULATORY (non-bearish) -> HIGH
        everything else          -> LOW
    """
    if event_type == "REGULATORY" and sentiment_label == "bearish":
        return "CRITICAL"
    if event_type == "LEGAL":
        return "CRITICAL"
    if event_type == "EARNINGS":
        if "fraud" in text:
            return "CRITICAL"
        return "HIGH"
    if event_type == "M_AND_A":
        return "HIGH"
    if event_type == "MACRO":
        if _contains_any(text, _RBI_RATE_TERMS):
            return "HIGH"
        return "MEDIUM"
    if event_type == "MANAGEMENT":
        if _contains_any(text, _CEO_RESIGN_TERMS):
            return "HIGH"
        return "MEDIUM"
    if event_type in ("DIVIDEND", "BUYBACK"):
        return "MEDIUM"
    if event_type == "IPO_LISTING":
        return "MEDIUM"
    if event_type == "REGULATORY":
        return "HIGH"
    return "LOW"


class EventTagger:
    """
    Tags a news item with its event type and impact level, and extracts fine-grained tags.

    Returns: (event_type: str, impact: str, tags: list[str])
      event_type: one of EARNINGS|DIVIDEND|BUYBACK|M_AND_A|MANAGEMENT|REGULATORY|
                  LEGAL|MACRO|IPO_LISTING|SECTOR|GENERIC
      impact:     CRITICAL|HIGH|MEDIUM|LOW
      tags:       free-form list, e.g. ["earnings", "beat", "q1fy27", "it-sector"]
    """

    def tag(
        self,
        headline: str,
        body: str,
        sentiment_label: str,
    ) -> tuple[str, str, list[str]]:
        """
        Classify a news item.

        Args:
            headline:        Article headline.
            body:            Article body text (or partial body).
            sentiment_label: "bullish" | "bearish" | "neutral" (from FinBERTAdapter).

        Returns:
            (event_type, impact, tags) tuple.
        """
        if not headline and not body:
            logger.warning("event_tagger_empty_input")
            return "GENERIC", "LOW", []

        combined = (headline + " " + body).lower()

        event_type = _detect_event_type(combined)

        # Secondary FinBERT signal: GENERIC + bearish -> re-check legal/regulatory
        if event_type == "GENERIC" and sentiment_label == "bearish":
            # Escalate check with looser matching for REGULATORY/LEGAL
            # (some regulatory headlines use indirect language)
            if any(kw in combined for kw in ["notice", "action", "probe", "inquiry", "suspended"]):
                event_type = "REGULATORY"
            elif any(kw in combined for kw in ["allegation", "complaint", "case filed", "summon"]):
                event_type = "LEGAL"

        impact = _compute_impact(event_type, sentiment_label, combined)
        tags = self._extract_tags(combined, event_type)

        logger.debug(
            "event_tagger_classified",
            event_type=event_type,
            impact=impact,
            tags=tags,
            sentiment=sentiment_label,
        )

        return event_type, impact, tags

    def _extract_tags(self, text: str, event_type: str) -> list[str]:
        """
        Extract fine-grained tags from lowercased combined text.

        Tags included:
        - event_type slug (always)
        - Quarterly refs: q1fy27, q2fy26, etc.
        - FY refs: fy27, fy26
        - Result quality: "beat", "miss", "inline"
        - Sector: "banking", "it", "pharma", etc.
        - Action hints: "buyback", "dividend", "merger", "demerger", "ipo"
        """
        tags: list[str] = []
        seen: set[str] = set()

        def _add(tag: str) -> None:
            tag = tag.strip().lower().replace(" ", "-")
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)

        # Always include event type slug
        _add(event_type.lower().replace("_", "-"))

        # Quarter references
        for m in _QUARTER_REGEX.finditer(text):
            _add(m.group(1).lower().replace(" ", ""))

        # Bare quarter references (Q1, Q2, etc.) with optional FY
        for m in _QUARTER_BARE_REGEX.finditer(text):
            quarter_part = m.group(1).lower()
            fy_part = (m.group(2) or "").lower().replace(" ", "")
            if fy_part:
                _add(quarter_part + fy_part)
            else:
                _add(quarter_part)

        # FY references
        for m in _FY_REGEX.finditer(text):
            _add(m.group(1).lower().replace(" ", ""))

        # Beat / miss / inline
        if _BEAT_REGEX.search(text):
            _add("beat")
        if _MISS_REGEX.search(text):
            _add("miss")
        if _INLINE_REGEX.search(text):
            _add("inline")

        # Sector hints
        for sector, keywords in _SECTOR_MAP.items():
            if any(kw in text for kw in keywords):
                _add(sector + "-sector")

        # Action hints
        if "buyback" in text or "buy back" in text:
            _add("buyback")
        if "dividend" in text:
            _add("dividend")
        if "merger" in text:
            _add("merger")
        if "demerger" in text or "hive off" in text or "spin off" in text:
            _add("demerger")
        if "ipo" in text:
            _add("ipo")
        if "acquisition" in text or "acquires" in text:
            _add("acquisition")
        if "stake sale" in text or "stake purchase" in text:
            _add("stake-deal")

        # Result quality for earnings
        if event_type == "EARNINGS":
            if _contains_any(text, {"profit", "net profit", "pat"}):
                _add("profit")
            if _contains_any(text, {"net loss", "loss"}):
                _add("loss")
            if _contains_any(text, {"revenue", "turnover", "topline"}):
                _add("revenue")
            if _contains_any(text, {"ebitda", "operating profit"}):
                _add("ebitda")

        return tags
