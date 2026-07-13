"""
news/event_classifier.py — Classify news events by type and market impact.

Rule-based classification (no ML). Fast, deterministic, no model loading.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algo_trading.news.scraper import RawArticle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    EARNINGS    = "EARNINGS"     # quarterly results, PAT, EPS
    DIVIDEND    = "DIVIDEND"     # dividend declared
    BUYBACK     = "BUYBACK"      # share buyback
    M_AND_A     = "M_AND_A"      # merger, acquisition, stake
    MANAGEMENT  = "MANAGEMENT"   # CEO change, board change
    REGULATORY  = "REGULATORY"   # SEBI order, show cause, NSE/BSE action
    LEGAL       = "LEGAL"        # court order, FIR, fraud allegation
    MACRO       = "MACRO"        # RBI, budget, GST, inflation, repo rate
    IPO_LISTING = "IPO_LISTING"  # IPO allotment, listing day
    SECTOR      = "SECTOR"       # sectoral news (no specific stock)
    GENERIC     = "GENERIC"      # unclassified


class ImpactLevel(str, Enum):
    CRITICAL = "CRITICAL"  # halt trading for symbol immediately
    HIGH     = "HIGH"      # reduce position size, heightened alert
    MEDIUM   = "MEDIUM"    # flag for operator, sentiment only
    LOW      = "LOW"       # sentiment feature only


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsEvent:
    """A classified, symbol-tagged news event ready for downstream consumption."""
    article_id:   str
    symbol:       str           # NSE symbol; empty string for macro/sector news
    headline:     str
    url:          str
    published_at: datetime
    event_type:   EventType
    impact:       ImpactLevel
    sentiment_hint: str         # "bullish" | "bearish" | "neutral"
    source:       str
    raw_body:     str


# ---------------------------------------------------------------------------
# Keyword rule sets
# ---------------------------------------------------------------------------

_EARNINGS_KWORDS = frozenset([
    "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
    "net profit", "pat", "eps", "revenue", "ebitda", "net loss", "annual results",
    "q1fy", "q2fy", "q3fy", "q4fy", "profit after tax", "earnings per share",
    "operating profit", "gross profit", "turnover",
])

_DIVIDEND_KWORDS = frozenset([
    "dividend", "interim dividend", "final dividend", "dividend declared",
    "dividend announcement", "special dividend",
])

_BUYBACK_KWORDS = frozenset([
    "buyback", "buy back", "share repurchase", "share buyback",
    "repurchases shares",
])

_M_AND_A_KWORDS = frozenset([
    "acquisition", "merger", "acquires", "stake purchase", "takeover",
    "strategic investment", "joint venture", "stake acquisition",
    "buys stake", "acquires stake", "merges with", "amalgamation",
    "demerger", "hive off", "spin off", "stake sale",
])

_MANAGEMENT_KWORDS = frozenset([
    "ceo", "md", "managing director", "board appoints", "resignation",
    "steps down", "new cfo", "director", "chairman", "executive director",
    "coo", "board meeting", "management change", "appoints new",
    "elevates", "promoted to", "takes charge",
])

_REGULATORY_KWORDS = frozenset([
    "sebi", "show cause notice", "sebi order", "ban", "suspended from trading",
    "nse action", "bse action", "penalty imposed", "investigation",
    "sebi probe", "regulatory action", "adjudication order",
    "debarred", "impounded", "exchange action",
])

_LEGAL_KWORDS = frozenset([
    "fraud", "fir", "court order", "arrested", "cheating", "embezzlement",
    "money laundering", "enforcement directorate", "ed raid",
    "income tax raid", "cbi", "chargesheet", "contempt of court",
    "insolvency", "nclt", "default", "wilful defaulter",
])

_MACRO_KWORDS = frozenset([
    "rbi", "repo rate", "reverse repo", "monetary policy", "inflation",
    "cpi", "wpi", "budget", "gst", "gdp", "fiscal deficit",
    "current account", "mpc", "rate cut", "rate hike",
    "forex reserves", "rupee", "interest rate",
])

_IPO_KWORDS = frozenset([
    "ipo", "initial public offering", "listing day", "ipo allotment",
    "share allotment", "grey market premium", "gmp",
    "listing gains", "oversubscribed", "nfo",
])

_BULLISH_KWORDS = frozenset([
    "beats", "beat estimates", "record profit", "strong growth", "upgrades",
    "buyback", "dividend", "acquisition target", "bullish", "positive outlook",
    "profit surges", "revenue jumps", "raises guidance", "upgrade",
    "buy rating", "outperform", "strong buy", "multibagger",
])

_BEARISH_KWORDS = frozenset([
    "misses", "miss estimates", "net loss", "sebi ban", "fraud", "fir",
    "show cause", "downgrade", "negative outlook", "probe", "penalty",
    "loss widens", "revenue falls", "guidance cut", "sell rating",
    "underperform", "default", "bankruptcy", "insolvency",
])

# Terms that indicate RBI rate action specifically (HIGH impact within MACRO)
_RBI_RATE_TERMS = frozenset([
    "repo rate", "reverse repo", "rate cut", "rate hike",
    "rate change", "monetary policy", "mpc decision",
])

# CEO-resignation signals within MANAGEMENT (HIGH impact)
_CEO_RESIGN_TERMS = frozenset([
    "ceo resigns", "md resigns", "ceo steps down", "md steps down",
    "chief executive resigns", "managing director resigns",
    "resignation of ceo", "resignation of md",
])


# ---------------------------------------------------------------------------
# EventClassifier
# ---------------------------------------------------------------------------

class EventClassifier:
    """
    Classifies a RawArticle into one or more NewsEvent objects.

    Each matched NSE symbol produces a separate NewsEvent (up to 3 per article).
    Macro news (no specific symbol) produces one event with symbol="MARKET".
    """

    def classify(
        self,
        article: "RawArticle",
        symbols: list[str],
    ) -> list[NewsEvent]:
        """
        Classify a RawArticle into NewsEvent objects.

        Parameters
        ----------
        article:
            The raw article from the scraper.
        symbols:
            Resolved NSE symbols from SymbolMapper (may be empty for generic news).

        Returns
        -------
        list[NewsEvent]
            Empty list if no symbols AND not macro news.
        """
        combined_text = (article.headline + " " + article.body).lower()

        event_type = self._detect_event_type(combined_text)
        sentiment  = self._detect_sentiment(combined_text)
        is_macro   = event_type == EventType.MACRO

        # Nothing to tag and not macro → skip
        if not symbols and not is_macro:
            return []

        events: list[NewsEvent] = []

        if is_macro and not symbols:
            # Macro/market-wide news with no specific stock
            impact = self._compute_impact(event_type, sentiment, combined_text)
            events.append(
                NewsEvent(
                    article_id   = article.id,
                    symbol       = "MARKET",
                    headline     = article.headline,
                    url          = article.url,
                    published_at = article.published_at,
                    event_type   = event_type,
                    impact       = impact,
                    sentiment_hint = sentiment,
                    source       = article.source,
                    raw_body     = article.body,
                )
            )
        else:
            # Stock-specific events (up to 3 symbols)
            for symbol in symbols[:3]:
                impact = self._compute_impact(event_type, sentiment, combined_text)
                events.append(
                    NewsEvent(
                        article_id   = article.id,
                        symbol       = symbol,
                        headline     = article.headline,
                        url          = article.url,
                        published_at = article.published_at,
                        event_type   = event_type,
                        impact       = impact,
                        sentiment_hint = sentiment,
                        source       = article.source,
                        raw_body     = article.body,
                    )
                )

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contains_any(text: str, kwords: frozenset[str]) -> bool:
        """Return True if any keyword from kwords appears in text."""
        return any(kw in text for kw in kwords)

    def _detect_event_type(self, text: str) -> EventType:
        """
        Return the most specific EventType that matches the text.
        Checks in priority order so that more specific types win.
        """
        if self._contains_any(text, _LEGAL_KWORDS):
            return EventType.LEGAL
        if self._contains_any(text, _REGULATORY_KWORDS):
            return EventType.REGULATORY
        if self._contains_any(text, _MACRO_KWORDS):
            return EventType.MACRO
        if self._contains_any(text, _IPO_KWORDS):
            return EventType.IPO_LISTING
        if self._contains_any(text, _EARNINGS_KWORDS):
            return EventType.EARNINGS
        if self._contains_any(text, _DIVIDEND_KWORDS):
            return EventType.DIVIDEND
        if self._contains_any(text, _BUYBACK_KWORDS):
            return EventType.BUYBACK
        if self._contains_any(text, _M_AND_A_KWORDS):
            return EventType.M_AND_A
        if self._contains_any(text, _MANAGEMENT_KWORDS):
            return EventType.MANAGEMENT
        return EventType.GENERIC

    def _detect_sentiment(self, text: str) -> str:
        """
        Return "bullish", "bearish", or "neutral" based on keyword presence.
        Bearish wins if both sets match (e.g. 'net loss beats estimates' → bearish).
        """
        has_bullish = self._contains_any(text, _BULLISH_KWORDS)
        has_bearish = self._contains_any(text, _BEARISH_KWORDS)

        if has_bearish:
            return "bearish"
        if has_bullish:
            return "bullish"
        return "neutral"

    def _compute_impact(
        self,
        event_type: EventType,
        sentiment: str,
        text: str,
    ) -> ImpactLevel:
        """
        Derive ImpactLevel from event_type + sentiment + contextual signals.

        Priority rules (first match wins):
          REGULATORY + bearish → CRITICAL
          LEGAL                → CRITICAL
          EARNINGS + fraud     → CRITICAL
          EARNINGS             → HIGH
          M_AND_A              → HIGH
          MACRO + rate action  → HIGH
          MANAGEMENT + resign  → HIGH
          DIVIDEND / BUYBACK   → MEDIUM
          IPO_LISTING          → MEDIUM
          MACRO (generic)      → MEDIUM
          everything else      → LOW
        """
        if event_type == EventType.REGULATORY and sentiment == "bearish":
            return ImpactLevel.CRITICAL

        if event_type == EventType.LEGAL:
            return ImpactLevel.CRITICAL

        if event_type == EventType.EARNINGS:
            if "fraud" in text:
                return ImpactLevel.CRITICAL
            return ImpactLevel.HIGH

        if event_type == EventType.M_AND_A:
            return ImpactLevel.HIGH

        if event_type == EventType.MACRO:
            if self._contains_any(text, _RBI_RATE_TERMS):
                return ImpactLevel.HIGH
            return ImpactLevel.MEDIUM

        if event_type == EventType.MANAGEMENT:
            if self._contains_any(text, _CEO_RESIGN_TERMS):
                return ImpactLevel.HIGH
            return ImpactLevel.MEDIUM

        if event_type in (EventType.DIVIDEND, EventType.BUYBACK):
            return ImpactLevel.MEDIUM

        if event_type == EventType.IPO_LISTING:
            return ImpactLevel.MEDIUM

        if event_type == EventType.REGULATORY:
            # Regulatory that isn't bearish — still elevated
            return ImpactLevel.HIGH

        return ImpactLevel.LOW
