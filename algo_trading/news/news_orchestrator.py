"""
news/news_orchestrator.py — Main entry point for the news intelligence layer.

Launches and wires all news components:
  1. NewsScraper (async, polls RSS + NSE announcements)
  2. EventClassifier + SymbolMapper (processes raw articles)
  3. SentimentPublisher → SentimentWorker (FinBERT scoring)
  4. NewsSignalBridge (routes events to orchestrator)

Runs as a separate OS process launched by daily_orchestrator at startup:
  python -m algo_trading.news.news_orchestrator

ZMQ ports used:
  5570: Scraper PUSH → Processor PULL (raw articles)
  5565: Processor PUSH → SentimentWorker PULL (headlines for FinBERT)
  5566: SentimentWorker PUSH → SentimentAggregator PULL (scores back)
  5571: Processor PUSH → NewsSignalBridge PULL (classified events)
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Optional

import structlog

from algo_trading.models.sentiment import SentimentPublisher

# These are built by the parallel agent.  They will be present by the time
# this module is imported in a fully-assembled deployment.
from algo_trading.news.event_classifier import EventClassifier, NewsEvent, SymbolMapper
from algo_trading.news.news_scraper import NewsScraper  # noqa: F401 — import triggers registration

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ZMQ port constants
# ---------------------------------------------------------------------------

_SCRAPER_PUSH_PORT: int = 5570      # Scraper PUSH → Processor PULL (raw articles)
_SENTIMENT_PUSH_PORT: int = 5565    # Processor → SentimentWorker (headlines)
_EVENT_PUSH_PORT: int = 5571        # Processor → NewsSignalBridge (classified events)


class NewsOrchestrator:
    """Wires all news-intelligence components into a single runnable object.

    Designed to run as an independent OS process::

        python -m algo_trading.news.news_orchestrator --symbols RELIANCE,TCS,...

    Internal ZMQ topology::

        NewsScraper ──PUSH:5570──► _process_articles_loop
                                          │
                     ┌────────────────────┤
                     │                    │
              SentimentPublisher   PUSH:5571 → NewsSignalBridge
              (PUSH:5565)

    Args:
        symbols: List of NSE symbols to monitor.  Used by
            :class:`~algo_trading.news.event_classifier.SymbolMapper` to
            filter articles that mention at least one watched symbol.
    """

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

        # Components — instantiate here; wired together in start()
        self._scraper = NewsScraper(push_port=_SCRAPER_PUSH_PORT)
        self._classifier = EventClassifier()
        self._symbol_mapper = SymbolMapper(symbols=symbols)
        self._sentiment_pub = SentimentPublisher(port=_SENTIMENT_PUSH_PORT)

        # ZMQ sockets created in start() / _process_articles_loop()
        self._event_push_socket: Optional[object] = None
        self._article_pull_socket: Optional[object] = None

        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        log.info(
            "news_orchestrator.init",
            symbols=symbols,
            scraper_pull_port=_SCRAPER_PUSH_PORT,
            event_push_port=_EVENT_PUSH_PORT,
        )

    # ------------------------------------------------------------------
    # Article processing loop — background thread
    # ------------------------------------------------------------------

    def _process_articles_loop(self) -> None:
        """Background thread: PULL raw articles from scraper, classify, publish.

        ZMQ PULL binds on port 5570 to receive raw article dicts from
        :class:`NewsScraper`.  For each article:

        1. :meth:`SymbolMapper.resolve` maps the article text to NSE symbols.
        2. :meth:`EventClassifier.classify` converts the article to a
           :class:`NewsEvent`.
        3. The event headline is pushed to :class:`SentimentPublisher` (→
           FinBERT worker on port 5565).
        4. The serialised event JSON is pushed to the ``NewsSignalBridge`` on
           port 5571.
        """
        try:
            import zmq  # type: ignore[import]
        except ImportError:
            log.error("news_orchestrator.zmq_unavailable — article processor not started")
            return

        ctx = zmq.Context.instance()

        # PULL raw articles from NewsScraper
        pull_sock = ctx.socket(zmq.PULL)
        pull_sock.setsockopt(zmq.LINGER, 0)
        pull_sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s timeout so we can check _stop_event
        pull_sock.setsockopt(zmq.RCVHWM, 1000)
        pull_sock.bind(f"tcp://127.0.0.1:{_SCRAPER_PUSH_PORT}")
        self._article_pull_socket = pull_sock

        # PUSH classified events to NewsSignalBridge
        push_sock = ctx.socket(zmq.PUSH)
        push_sock.setsockopt(zmq.LINGER, 0)
        push_sock.setsockopt(zmq.SNDHWM, 500)
        push_sock.connect(f"tcp://127.0.0.1:{_EVENT_PUSH_PORT}")
        self._event_push_socket = push_sock

        log.info(
            "news_orchestrator.processor_started",
            pull_port=_SCRAPER_PUSH_PORT,
            push_port=_EVENT_PUSH_PORT,
        )

        while not self._stop_event.is_set():
            # Receive raw article dict from scraper
            try:
                raw = pull_sock.recv_string()
            except zmq.Again:
                continue  # poll timeout — loop back
            except zmq.ZMQError as exc:
                log.error("news_orchestrator.pull_recv_error", error=str(exc))
                break

            try:
                article_dict = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("news_orchestrator.article_json_error", error=str(exc))
                continue

            # Map article → NSE symbols
            try:
                symbols = self._symbol_mapper.resolve(article_dict)
            except Exception as exc:
                log.warning("news_orchestrator.symbol_map_error", error=str(exc))
                symbols = []

            if not symbols:
                log.debug(
                    "news_orchestrator.article_no_symbols",
                    headline=str(article_dict.get("headline", ""))[:60],
                )
                continue

            # Classify article into a NewsEvent for each matched symbol
            for symbol in symbols:
                try:
                    event: NewsEvent = self._classifier.classify(article_dict, symbol=symbol)
                except Exception as exc:
                    log.warning(
                        "news_orchestrator.classify_error",
                        symbol=symbol,
                        error=str(exc),
                    )
                    continue

                # Push headline to FinBERT sentiment worker
                try:
                    headline: str = event.headline or str(article_dict.get("headline", ""))
                    ts: datetime = event.timestamp if hasattr(event, "timestamp") else datetime.now(
                        timezone.utc
                    )
                    self._sentiment_pub.publish(
                        symbol=symbol,
                        headline=headline,
                        timestamp=ts,
                    )
                except Exception as exc:
                    log.warning(
                        "news_orchestrator.sentiment_publish_error",
                        symbol=symbol,
                        error=str(exc),
                    )

                # Push classified event JSON to NewsSignalBridge
                try:
                    event_json = json.dumps(event.to_dict())
                    push_sock.send_string(event_json, flags=zmq.NOBLOCK)
                    log.debug(
                        "news_orchestrator.event_published",
                        symbol=symbol,
                        impact=event.impact.value,
                        event_type=event.event_type.value,
                    )
                except zmq.Again:
                    log.warning(
                        "news_orchestrator.event_push_queue_full",
                        symbol=symbol,
                        headline=headline[:60],
                    )
                except Exception as exc:
                    log.warning(
                        "news_orchestrator.event_push_error",
                        symbol=symbol,
                        error=str(exc),
                    )

        pull_sock.close()
        push_sock.close()
        log.info("news_orchestrator.processor_stopped")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the article processor thread.

        The :class:`NewsScraper` async loop is started separately via
        :meth:`run_async`.  This method only starts the synchronous ZMQ
        processing thread so that :meth:`run` can then hand control to the
        async scraper loop.

        Safe to call once per process lifetime.
        """
        if self._processor_thread is not None and self._processor_thread.is_alive():
            log.debug("news_orchestrator.processor_already_running")
            return

        self._processor_thread = threading.Thread(
            target=self._process_articles_loop,
            name="news-article-processor",
            daemon=True,
        )
        self._processor_thread.start()
        log.info("news_orchestrator.processor_thread_started")

    async def run_async(self) -> None:
        """Async entry-point: runs the scraper's async polling loop.

        Starts the article processor thread first, then hands control to the
        scraper's ``run_scraper()`` coroutine which polls RSS feeds and the
        NSE announcements endpoint on an asyncio event loop.

        This coroutine runs until the process is killed (SIGTERM/SIGINT handled
        by the OS) or until the scraper raises a fatal exception.
        """
        log.info("news_orchestrator.async_starting", symbols=self._symbols)
        self.start()  # Ensure processor thread is running before the scraper pushes articles

        try:
            await self._scraper.run_scraper()
        except asyncio.CancelledError:
            log.info("news_orchestrator.scraper_cancelled")
        except Exception as exc:
            log.error("news_orchestrator.scraper_fatal_error", error=str(exc))
            raise
        finally:
            self._stop_event.set()
            log.info("news_orchestrator.async_stopped")

    def run(self) -> None:
        """Synchronous entry-point: block until the scraper exits.

        Wraps :meth:`run_async` in ``asyncio.run()``.  This is the method
        invoked by :func:`main` when the module is run as a script::

            python -m algo_trading.news.news_orchestrator
        """
        asyncio.run(self.run_async())


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for running the news orchestrator as a standalone process.

    Usage::

        python -m algo_trading.news.news_orchestrator
        python -m algo_trading.news.news_orchestrator --symbols RELIANCE,TCS,HDFCBANK

    The orchestrator will keep running until it receives a SIGTERM or SIGINT.
    Launched automatically by ``DailyOrchestrator._startup()`` at 09:00 IST.
    """
    import argparse
    import signal as _signal
    import sys

    from algo_trading.core.logging import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(
        description="NSE AlgoTrader — news intelligence orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "ZMQ ports: 5570 (scraper→processor), 5565 (processor→FinBERT),\n"
            "           5566 (FinBERT→aggregator), 5571 (processor→signal bridge)"
        ),
    )
    parser.add_argument(
        "--symbols",
        default="NIFTY50,BANKNIFTY,RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK",
        help="Comma-separated list of NSE symbols to monitor (default: top 7)",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    log.info("news_orchestrator.main", symbols=symbols)

    orchestrator = NewsOrchestrator(symbols)

    # Honour SIGTERM for clean daemon shutdown
    def _handle_term(signum: int, frame: object) -> None:  # noqa: ARG001
        log.warning("news_orchestrator.sigterm_received")
        orchestrator._stop_event.set()
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _handle_term)

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        log.info("news_orchestrator.keyboard_interrupt")
        orchestrator._stop_event.set()


if __name__ == "__main__":
    main()
