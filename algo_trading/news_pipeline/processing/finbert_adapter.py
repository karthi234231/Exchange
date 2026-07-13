"""
FinBERT sentiment adapter for the news pipeline.

Two modes (auto-detected at first call):
1. Direct mode (preferred): loads FinBERT via transformers.pipeline() directly.
   Lazy-loaded on first call; cached in class variable for the process lifetime.
   CPU-only (project constraint: no GPU).

2. Keyword fallback: used when transformers is not installed or model fails to load.
   Mirrors the keyword sets used in algo_trading.news.event_classifier and
   algo_trading.models.sentiment.SentimentWorker.

Score semantics:
    score = positive_prob - negative_prob   (range -1.0 to +1.0)
    label = "bullish" | "bearish" | "neutral"
    confidence = max(positive_prob, negative_prob, neutral_prob)

The FinBERT model from HuggingFace uses label names "positive", "negative", "neutral".
We map these to "bullish", "bearish", "neutral" respectively.

Input text is truncated to 512 characters of clean text before tokenisation to stay
within FinBERT's 512-token limit. The transformers pipeline also handles truncation
internally (truncation=True), so this is a belt-and-suspenders guard.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional

import structlog

from algo_trading.news_pipeline.config import cfg

# ---------------------------------------------------------------------------
# Inference result cache
# ---------------------------------------------------------------------------
# Key: SHA-256 of the truncated text (first 1800 chars).
# Value: (score, label, confidence) tuple.
# Bounded at _CACHE_MAXSIZE entries (LRU eviction via OrderedDict).
# Thread-safe via _cache_lock.
_CACHE_MAXSIZE  = 2_000
_result_cache: OrderedDict[str, tuple[float, str, float]] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _cache_get(key: str) -> Optional[tuple[float, str, float]]:
    with _cache_lock:
        if key in _result_cache:
            _result_cache.move_to_end(key)
            return _result_cache[key]
    return None


def _cache_put(key: str, value: tuple[float, str, float]) -> None:
    with _cache_lock:
        if key in _result_cache:
            _result_cache.move_to_end(key)
        else:
            if len(_result_cache) >= _CACHE_MAXSIZE:
                _result_cache.popitem(last=False)  # evict oldest
        _result_cache[key] = value

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets (fallback when FinBERT is unavailable)
# Drawn from event_classifier._BULLISH_KWORDS / _BEARISH_KWORDS +
# SentimentWorker._BULLISH_WORDS / _BEARISH_WORDS.
# ---------------------------------------------------------------------------

_BULLISH_KWORDS: frozenset[str] = frozenset({
    "beats", "beat estimates", "record profit", "strong growth", "upgrades",
    "buyback", "dividend", "acquisition target", "bullish", "positive outlook",
    "profit surges", "revenue jumps", "raises guidance", "upgrade",
    "buy rating", "outperform", "strong buy", "multibagger",
    "surge", "rally", "gain", "profit", "beat", "record", "growth",
    "up", "high", "strong", "jumps", "soars", "rises", "turnaround",
    "expansion", "green", "upbeat", "optimistic", "overweight",
})

_BEARISH_KWORDS: frozenset[str] = frozenset({
    "misses", "miss estimates", "net loss", "sebi ban", "fraud", "fir",
    "show cause", "downgrade", "negative outlook", "probe", "penalty",
    "loss widens", "revenue falls", "guidance cut", "sell rating",
    "underperform", "default", "bankruptcy", "insolvency",
    "fall", "drop", "loss", "miss", "decline", "weak", "down",
    "low", "crash", "sell", "plunges", "slumps", "tanks", "warning",
    "concern", "risk", "uncertainty", "headwind", "charges",
})


def _keyword_score(text: str) -> tuple[float, str, float]:
    """Keyword-based sentiment scoring. Returns (score, label, confidence=0.5)."""
    text_lower = text.lower()
    words = text_lower.split()

    # Check multi-word phrases first
    bullish_hits = sum(1 for kw in _BULLISH_KWORDS if kw in text_lower)
    bearish_hits = sum(1 for kw in _BEARISH_KWORDS if kw in text_lower)
    total = bullish_hits + bearish_hits

    # Also check individual words for the shorter keyword sets
    bull_words = sum(1 for w in words if w in _BULLISH_KWORDS)
    bear_words = sum(1 for w in words if w in _BEARISH_KWORDS)
    bullish_hits += bull_words
    bearish_hits += bear_words
    total = bullish_hits + bearish_hits

    if total == 0:
        return 0.0, "neutral", 0.5

    # Bearish wins ties (mirrors EventClassifier._detect_sentiment logic)
    if bearish_hits >= bullish_hits:
        raw_score = -(bearish_hits / (total + 1))
        score = max(-0.9, raw_score)
        return score, "bearish", 0.5

    raw_score = bullish_hits / (total + 1)
    score = min(0.9, raw_score)
    return score, "bullish", 0.5


def _truncate_text(text: str, max_chars: int = 1800) -> str:
    """Truncate text to approximately 512 tokens (1800 chars ~= 450 words ~= 512 tokens)."""
    if len(text) <= max_chars:
        return text
    # Truncate at a word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space]
    return truncated


class FinBERTAdapter:
    """
    Computes sentiment score for financial text using FinBERT.

    Returns: (score: float, label: str, confidence: float)
      score:      -1.0 (very bearish) to +1.0 (very bullish)
      label:      "bullish" | "bearish" | "neutral"
      confidence: 0.0 to 1.0

    FinBERT model: cfg.FINBERT_MODEL ("ProsusAI/finbert")
    Label mapping: "positive" -> "bullish", "negative" -> "bearish", "neutral" -> "neutral"
    Score: positive_prob - negative_prob (ranges -1 to 1)

    Thread-safe: uses a class-level lock for lazy model loading.
    """

    _pipeline: Optional[object] = None
    _model_loaded: bool = False
    _load_error: Optional[str] = None
    _lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    @classmethod
    def _load_model(cls) -> bool:
        """
        Load transformers pipeline. Returns True on success.
        Sets _load_error and returns False on failure.
        Thread-safe via class-level lock.
        """
        if cls._model_loaded:
            return cls._pipeline is not None

        with cls._lock:
            # Double-checked locking
            if cls._model_loaded:
                return cls._pipeline is not None

            logger.info(
                "finbert_adapter_loading_model",
                model=cfg.FINBERT_MODEL,
                device="cpu",
            )
            t0 = time.monotonic()
            try:
                from transformers import pipeline  # type: ignore[import]

                cls._pipeline = pipeline(
                    "sentiment-analysis",
                    model=cfg.FINBERT_MODEL,
                    device=-1,          # CPU only — project constraint
                    max_length=512,
                    truncation=True,
                    top_k=None,         # return all labels with their probabilities
                )
                elapsed = time.monotonic() - t0
                logger.info(
                    "finbert_adapter_model_loaded",
                    model=cfg.FINBERT_MODEL,
                    load_time_s=round(elapsed, 2),
                )
                cls._model_loaded = True
                return True

            except ImportError:
                cls._load_error = "transformers not installed"
                cls._model_loaded = True
                logger.warning(
                    "finbert_adapter_transformers_unavailable",
                    fallback="keyword-based sentiment",
                )
                return False

            except Exception as exc:
                cls._load_error = str(exc)
                cls._model_loaded = True
                logger.error(
                    "finbert_adapter_model_load_failed",
                    error=str(exc),
                    fallback="keyword-based sentiment",
                )
                return False

    # -------------------------------------------------------------------------
    # Eager pre-load (call at startup)
    # -------------------------------------------------------------------------

    @classmethod
    def preload(cls) -> bool:
        """
        Eagerly load the FinBERT model on the calling thread.

        Call this at application startup (e.g. DailyOrchestrator._startup or
        Prefect flow startup) so that the first live inference call does not
        block for 20-60 seconds downloading/loading the model.

        Returns True if the model loaded successfully, False if it fell back
        to keyword scoring.
        """
        return cls._load_model()

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def score(self, text: str) -> tuple[float, str, float]:
        """
        Analyse text sentiment. Returns (score, label, confidence).

        Results are cached by text content (SHA-256 key, LRU eviction at
        2,000 entries) so repeated headlines are free after the first call.

        Never raises — always returns a result. Falls back to keyword
        heuristic when FinBERT is unavailable.

        Args:
            text: Headline or body text (arbitrarily long; will be truncated).

        Returns:
            (score, label, confidence) where:
              score      in [-1.0, 1.0]
              label      "bullish" | "bearish" | "neutral"
              confidence in [0.0, 1.0]
        """
        if not text or not text.strip():
            return 0.0, "neutral", 0.0

        truncated = _truncate_text(text)
        cache_key = _cache_key(truncated)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        model_ok = self._load_model()

        if model_ok and self._pipeline is not None:
            result = self._score_finbert(truncated)
        else:
            result = _keyword_score(text)

        _cache_put(cache_key, result)
        return result

    def score_batch(self, texts: list[str]) -> list[tuple[float, str, float]]:
        """
        Batch sentiment scoring. More efficient than calling score() in a loop
        when FinBERT is available (single forward pass through the model).

        Falls back to keyword scoring per-item when FinBERT is unavailable.

        Args:
            texts: List of text strings to score.

        Returns:
            List of (score, label, confidence) tuples, same order as input.
        """
        if not texts:
            return []

        model_ok = self._load_model()

        if model_ok and self._pipeline is not None:
            return self._score_finbert_batch(texts)

        return [_keyword_score(t) for t in texts]

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _score_finbert(self, text: str) -> tuple[float, str, float]:
        """Run FinBERT pipeline on a single (already-truncated) text. Never raises."""
        try:
            result = self._pipeline(text)  # type: ignore[call-arg]

            # top_k=None returns a list of dicts per input; for single input
            # it is [[{"label": ..., "score": ...}, ...]]
            return _parse_finbert_result(result)

        except Exception as exc:
            logger.warning(
                "finbert_adapter_inference_error",
                error=str(exc),
                fallback="keyword",
            )
            return _keyword_score(text)

    def _score_finbert_batch(self, texts: list[str]) -> list[tuple[float, str, float]]:
        """Run FinBERT pipeline on a batch. Never raises."""
        try:
            truncated_texts = [_truncate_text(t) for t in texts]
            results = self._pipeline(truncated_texts)  # type: ignore[call-arg]

            # results is a list with one entry per text
            out: list[tuple[float, str, float]] = []
            for i, result in enumerate(results):
                try:
                    out.append(_parse_finbert_result(result))
                except Exception as exc:
                    logger.warning(
                        "finbert_adapter_batch_item_error",
                        index=i,
                        error=str(exc),
                    )
                    out.append(_keyword_score(texts[i]))
            return out

        except Exception as exc:
            logger.warning(
                "finbert_adapter_batch_inference_error",
                error=str(exc),
                fallback="keyword-per-item",
            )
            return [_keyword_score(t) for t in texts]


def _parse_finbert_result(result: object) -> tuple[float, str, float]:
    """
    Parse the output from a transformers pipeline call with top_k=None.

    Expected structure (top_k=None, single input):
        [[{"label": "positive", "score": 0.85}, {"label": "negative", "score": 0.05}, ...]]
    OR (top_k=None, within batch):
        [{"label": "positive", "score": 0.85}, ...]

    Returns (score, label, confidence) with:
        score = positive_prob - negative_prob
        label = "bullish" | "bearish" | "neutral"
        confidence = max probability across all labels
    """
    # Normalise: unwrap outer list if present
    if isinstance(result, list) and result and isinstance(result[0], list):
        items = result[0]
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        items = result
    else:
        # Unexpected format — neutral fallback
        return 0.0, "neutral", 0.5

    probs: dict[str, float] = {}
    for item in items:
        label_raw = str(item.get("label", "neutral")).lower()
        prob = float(item.get("score", 0.0))
        probs[label_raw] = prob

    positive_prob = probs.get("positive", 0.0)
    negative_prob = probs.get("negative", 0.0)
    neutral_prob = probs.get("neutral", 0.0)

    score = positive_prob - negative_prob  # range [-1, 1]
    confidence = max(positive_prob, negative_prob, neutral_prob)

    # Map to user-facing label
    if positive_prob >= negative_prob and positive_prob >= neutral_prob:
        label = "bullish"
    elif negative_prob > positive_prob and negative_prob >= neutral_prob:
        label = "bearish"
    else:
        label = "neutral"

    return round(score, 6), label, round(confidence, 6)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

finbert = FinBERTAdapter()
