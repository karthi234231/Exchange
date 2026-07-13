from __future__ import annotations

# Stage 4: Sentiment feature (#24 in extended feature set)
# Uses FinBERT for financial sentiment (HuggingFace transformers)
# ZMQ PUSH/PULL pattern — same as alert_sink.py — execution never blocks on sentiment
# Deferred import: transformers loaded only in worker process, not in main process
STAGE = "4"
SENTIMENT_ZMQ_PORT = 5565  # PUSH:5565 (main) → PULL:5565 (worker) → PUSH:5566 (results back)

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SentimentScore:
    """Sentiment result for a single news headline.

    Attributes:
        timestamp: UTC datetime when the headline was scored.
        symbol: NSE symbol the headline relates to.
        headline: Original headline text.
        score: Sentiment score in [-1.0, +1.0].
            Negative values are bearish; positive values are bullish.
        label: One of "positive", "negative", or "neutral".
        confidence: Model confidence in [0, 1].
        source: Backend used — "finbert" or "fallback_keyword".
    """

    timestamp: datetime
    symbol: str
    headline: str
    score: float
    label: str
    confidence: float
    source: str


# ---------------------------------------------------------------------------
# Publisher (main process — fire-and-forget)
# ---------------------------------------------------------------------------


class SentimentPublisher:
    """Push news headlines to the sentiment worker via ZMQ PUSH.

    Non-blocking. Returns in < 1 µs. Never raises.
    If zmq is not installed the publisher silently becomes a no-op.

    Args:
        port: ZMQ port to connect to (default: SENTIMENT_ZMQ_PORT = 5565).
    """

    def __init__(self, port: int = SENTIMENT_ZMQ_PORT) -> None:
        self._port = port
        self._socket: Optional[object] = None
        try:
            import zmq  # type: ignore[import]
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 50)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://127.0.0.1:{port}")
            self._socket = sock
            logger.debug("sentiment_publisher_connected", port=port)
        except ImportError:
            logger.warning("sentiment_publisher_zmq_unavailable")
            self._socket = None

    def publish(
        self,
        symbol: str,
        headline: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Fire-and-forget push of a headline to the sentiment worker.

        If the worker queue is full the headline is silently dropped.
        This method never raises an exception — it is safe to call from
        the execution hot-path.

        Args:
            symbol: NSE symbol the headline relates to.
            headline: Raw headline text.
            timestamp: UTC datetime of the headline (defaults to now).
        """
        if self._socket is None:
            return
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)
        payload = json.dumps(
            {
                "symbol": symbol,
                "headline": headline,
                "timestamp_iso": timestamp.isoformat(),
            }
        )
        try:
            import zmq  # type: ignore[import]
            self._socket.send_string(payload, flags=zmq.NOBLOCK)
        except Exception as exc:  # zmq.Again or anything else
            try:
                import zmq  # type: ignore[import]
                if isinstance(exc, zmq.Again):
                    logger.warning(
                        "sentiment_publisher_queue_full",
                        symbol=symbol,
                        headline=headline[:60],
                    )
            except ImportError:
                pass


# ---------------------------------------------------------------------------
# Worker (separate process — NOT a thread)
# ---------------------------------------------------------------------------


class SentimentWorker:
    """FinBERT sentiment worker.

    Designed to run as a standalone OS process::

        python -m algo_trading.models.sentiment --worker

    The worker binds to ``pull_port`` (5565) to receive headlines from the
    main process and binds to ``push_port`` (5566) to push results back to
    :class:`SentimentAggregator`.

    FinBERT (``ProsusAI/finbert``) is loaded lazily on first headline to
    avoid slowing down the main process import.  If ``transformers`` is not
    installed a keyword-based fallback is used instead.

    Args:
        pull_port: ZMQ port to PULL headlines from (default 5565).
        push_port: ZMQ port to PUSH results to (default 5566).
    """

    # Keyword sets for the fallback scorer
    _BULLISH_WORDS: frozenset[str] = frozenset(
        {"surge", "rally", "gain", "profit", "beat", "record", "growth", "up", "high", "strong"}
    )
    _BEARISH_WORDS: frozenset[str] = frozenset(
        {"fall", "drop", "loss", "miss", "decline", "weak", "down", "low", "crash", "sell"}
    )

    def __init__(
        self,
        pull_port: int = SENTIMENT_ZMQ_PORT,
        push_port: int = 5566,
    ) -> None:
        self._pull_port = pull_port
        self._push_port = push_port
        self._pipeline: Optional[object] = None
        self._backend: str = "fallback_keyword"
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load FinBERT on first use.

        Falls back to keyword-based scoring if ``transformers`` is not
        installed.  Model loading is deferred so it never happens at import
        time and never blocks the main process.
        """
        try:
            from transformers import pipeline  # type: ignore[import]

            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,       # CPU only
                max_length=512,
                truncation=True,
            )
            self._backend = "finbert"
            logger.info("sentiment_worker_finbert_loaded")
        except ImportError:
            self._pipeline = None
            self._backend = "fallback_keyword"
            logger.warning("sentiment_worker_transformers_unavailable_using_keyword_fallback")
        self._model_loaded = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_headline(self, headline: str) -> tuple[float, str, float]:
        """Score a headline and return (score, label, confidence).

        FinBERT label mapping::

            "positive" → score = +confidence
            "negative" → score = -confidence
            "neutral"  → score = 0.0

        Keyword fallback::

            score = (bullish_count - bearish_count) / (total_count + 1)
            confidence = 0.5  (keyword fallback is low-confidence)

        Args:
            headline: Raw headline text.

        Returns:
            Tuple of (score, label, confidence) where score ∈ [-1.0, +1.0].
        """
        if not self._model_loaded:
            self._load_model()

        if self._backend == "finbert" and self._pipeline is not None:
            return self._score_finbert(headline)
        return self._score_keyword(headline)

    def _score_finbert(self, headline: str) -> tuple[float, str, float]:
        """Run FinBERT pipeline on a single headline."""
        try:
            result = self._pipeline(headline)  # type: ignore[call-arg, index]
            if isinstance(result, list) and result:
                item = result[0]
            else:
                item = result
            label: str = str(item.get("label", "neutral")).lower()
            confidence: float = float(item.get("score", 0.5))
            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                label = "neutral"
                score = 0.0
            return score, label, confidence
        except Exception as exc:
            logger.warning("sentiment_finbert_error", error=str(exc))
            return self._score_keyword(headline)

    def _score_keyword(self, headline: str) -> tuple[float, str, float]:
        """Keyword-based fallback scorer."""
        words = headline.lower().split()
        bullish_count = sum(1 for w in words if w in self._BULLISH_WORDS)
        bearish_count = sum(1 for w in words if w in self._BEARISH_WORDS)
        total_count = bullish_count + bearish_count
        score = (bullish_count - bearish_count) / (total_count + 1)
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        confidence = 0.5
        return score, label, confidence

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main worker loop. Blocking — runs until SIGTERM/SIGINT.

        Binds a PULL socket on ``pull_port`` to receive headlines from the
        main process and a PUSH socket on ``push_port`` to forward results to
        :class:`SentimentAggregator`.  Exceptions within the loop are logged
        and skipped — the loop never exits on its own.
        """
        import zmq  # type: ignore[import]

        ctx = zmq.Context()

        pull_sock = ctx.socket(zmq.PULL)
        pull_sock.setsockopt(zmq.LINGER, 0)
        pull_sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s poll so we can catch signals
        pull_sock.bind(f"tcp://127.0.0.1:{self._pull_port}")

        push_sock = ctx.socket(zmq.PUSH)
        push_sock.setsockopt(zmq.LINGER, 0)
        push_sock.setsockopt(zmq.SNDHWM, 100)
        push_sock.bind(f"tcp://127.0.0.1:{self._push_port}")

        logger.info(
            "sentiment_worker_started",
            pull_port=self._pull_port,
            push_port=self._push_port,
        )

        # Load model once before the loop
        self._load_model()

        while True:
            try:
                raw = pull_sock.recv_string()
            except zmq.Again:
                continue  # timeout — re-enter loop to check for signals
            except zmq.ZMQError as exc:
                logger.error("sentiment_worker_recv_error", error=str(exc))
                break

            try:
                data = json.loads(raw)
                symbol: str = data.get("symbol", "")
                headline: str = data.get("headline", "")
                timestamp_iso: str = data.get("timestamp_iso", "")

                try:
                    ts = datetime.fromisoformat(timestamp_iso)
                except (ValueError, TypeError):
                    ts = datetime.now(tz=timezone.utc)

                score, label, confidence = self.score_headline(headline)

                result = {
                    "timestamp_iso": ts.isoformat(),
                    "symbol": symbol,
                    "headline": headline,
                    "score": score,
                    "label": label,
                    "confidence": confidence,
                    "source": self._backend,
                }
                push_sock.send_string(json.dumps(result), flags=zmq.NOBLOCK)

                logger.debug(
                    "sentiment_scored",
                    symbol=symbol,
                    label=label,
                    score=round(score, 3),
                    source=self._backend,
                )
            except Exception as exc:
                logger.error("sentiment_worker_loop_error", error=str(exc))
                # Never exit — keep processing next message

        pull_sock.close()
        push_sock.close()
        ctx.term()
        logger.info("sentiment_worker_stopped")


# ---------------------------------------------------------------------------
# Aggregator (main process — rolling sentiment window)
# ---------------------------------------------------------------------------


class SentimentAggregator:
    """Maintains a rolling sentiment window and provides feature #24.

    Subscribes to results pushed by :class:`SentimentWorker` (port 5566).
    Results are accumulated in a time-bounded deque (default: 4-hour window).
    All polling is non-blocking — this class is safe to call from the
    execution thread.

    Args:
        pull_port: ZMQ port to PULL results from (default 5566).
        window_minutes: Rolling window length in minutes (default 240 = 4 h).
    """

    def __init__(
        self,
        pull_port: int = 5566,
        window_minutes: int = 240,
    ) -> None:
        self._window_minutes = window_minutes
        self._scores: deque[SentimentScore] = deque()
        self._socket: Optional[object] = None

        try:
            import zmq  # type: ignore[import]
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PULL)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking
            sock.connect(f"tcp://127.0.0.1:{pull_port}")
            self._socket = sock
            logger.debug("sentiment_aggregator_connected", pull_port=pull_port)
        except ImportError:
            logger.warning("sentiment_aggregator_zmq_unavailable")
            self._socket = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self, timeout_ms: int = 0) -> list[SentimentScore]:
        """Drain all available results from the worker (non-blocking).

        Args:
            timeout_ms: Not used — always non-blocking.  Kept for API
                consistency with the alert pattern.

        Returns:
            List of :class:`SentimentScore` objects received since the last
            call.  May be empty.
        """
        if self._socket is None:
            return []

        received: list[SentimentScore] = []
        try:
            import zmq  # type: ignore[import]
            while True:
                try:
                    raw = self._socket.recv_string(flags=zmq.NOBLOCK)
                    data = json.loads(raw)
                    ts_str: str = data.get("timestamp_iso", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except (ValueError, TypeError):
                        ts = datetime.now(tz=timezone.utc)

                    score = SentimentScore(
                        timestamp=ts,
                        symbol=data.get("symbol", ""),
                        headline=data.get("headline", ""),
                        score=float(data.get("score", 0.0)),
                        label=str(data.get("label", "neutral")),
                        confidence=float(data.get("confidence", 0.5)),
                        source=str(data.get("source", "unknown")),
                    )
                    self._scores.append(score)
                    received.append(score)
                except zmq.Again:
                    break  # no more messages
        except Exception as exc:
            logger.warning("sentiment_aggregator_poll_error", error=str(exc))

        self._evict_old()
        return received

    def get_current_score(self, symbol: str) -> float:
        """Compute the rolling exponentially-weighted sentiment score for a symbol.

        Weights are computed as::

            weight = exp(-age_minutes / 60)

        where ``age_minutes`` is how many minutes ago the score was produced.
        Returns 0.0 if no scores exist for the symbol in the current window.

        Args:
            symbol: NSE symbol to query.

        Returns:
            Weighted average score in [-1.0, +1.0], or 0.0 if no data.
        """
        self._evict_old()
        now = datetime.now(tz=timezone.utc)

        weighted_sum = 0.0
        weight_total = 0.0

        for entry in self._scores:
            if entry.symbol != symbol:
                continue
            try:
                ts = entry.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_seconds = (now - ts).total_seconds()
                age_minutes = max(0.0, age_seconds / 60.0)
            except Exception:
                age_minutes = 0.0

            decay = math.exp(-age_minutes / 60.0)
            weighted_sum += entry.score * decay
            weight_total += decay

        if weight_total == 0.0:
            return 0.0
        return weighted_sum / weight_total

    def get_feature_value(self, symbol: str) -> float:
        """Return feature #24 — rolling weighted sentiment score.

        This is the value injected into the live feature vector.

        Args:
            symbol: NSE symbol to query.

        Returns:
            Sentiment score in [-1.0, +1.0], or 0.0 if no recent data.
        """
        return self.get_current_score(symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_old(self) -> None:
        """Remove scores older than ``window_minutes`` from the deque."""
        now = datetime.now(tz=timezone.utc)
        cutoff_seconds = self._window_minutes * 60.0
        while self._scores:
            oldest = self._scores[0]
            try:
                ts = oldest.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_seconds = (now - ts).total_seconds()
            except Exception:
                age_seconds = 0.0
            if age_seconds > cutoff_seconds:
                self._scores.popleft()
            else:
                break


# ---------------------------------------------------------------------------
# Entry point — run as worker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NSE AlgoTrader sentiment worker (Stage 4)")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as ZMQ PULL/PUSH sentiment worker process.",
    )
    args = parser.parse_args()

    if args.worker:
        worker = SentimentWorker()
        worker.run()
