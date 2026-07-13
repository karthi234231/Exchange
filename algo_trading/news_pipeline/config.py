"""algo_trading.news_pipeline.config — Env-var driven configuration for the news pipeline.

All values have sensible defaults so the system runs without any environment variables set.
Override any value by exporting the corresponding uppercase env var before process start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    POSTGRES_DSN: str = "postgresql://trader:trader@localhost:5432/trading_news"

    # ------------------------------------------------------------------
    # Parquet storage (relative to project root)
    # ------------------------------------------------------------------
    PARQUET_DIR: str = "data/news_pipeline"

    # ------------------------------------------------------------------
    # FinBERT / NLP
    # ------------------------------------------------------------------
    FINBERT_MODEL: str = "ProsusAI/finbert"
    SPACY_MODEL: str = "en_core_web_sm"
    NER_CONFIDENCE_THRESHOLD: float = 0.7

    # ------------------------------------------------------------------
    # Pulse scraper
    # ------------------------------------------------------------------
    PULSE_URL: str = "https://pulse.zerodha.com/"
    PULSE_POLL_INTERVAL_SEC: int = 300
    PULSE_TIMEOUT_SEC: int = 15

    # ------------------------------------------------------------------
    # nselib / jugaad
    # ------------------------------------------------------------------
    NSELIB_POLL_INTERVAL_SEC: int = 600
    JUGAAD_POLL_INTERVAL_SEC: int = 3600

    # ------------------------------------------------------------------
    # RSS fallback (used when Pulse fails)
    # ------------------------------------------------------------------
    RSS_POLL_INTERVAL_SEC: int = 300
    RSS_FEEDS: list[str] = field(default_factory=lambda: [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://www.business-standard.com/rss/markets-106.rss",
        "https://www.livemint.com/rss/markets",
    ])

    # ------------------------------------------------------------------
    # Signal fusion weights (must sum to 1.0)
    # ------------------------------------------------------------------
    WEIGHT_NEWS: float = 0.40
    WEIGHT_BULK_DEAL: float = 0.30
    WEIGHT_DELIVERY: float = 0.20
    WEIGHT_VIX: float = 0.10

    # ------------------------------------------------------------------
    # Signal thresholds
    # ------------------------------------------------------------------
    STRONG_BUY_THRESHOLD: float = 0.60
    BUY_THRESHOLD: float = 0.30
    SELL_THRESHOLD: float = -0.30
    STRONG_SELL_THRESHOLD: float = -0.60

    # ------------------------------------------------------------------
    # VIX dampening
    # ------------------------------------------------------------------
    VIX_HIGH_THRESHOLD: float = 20.0
    VIX_DAMPEN_FACTOR: float = 0.7

    # ------------------------------------------------------------------
    # Bulk deal thresholds (crore INR)
    # ------------------------------------------------------------------
    BULK_DEAL_HIGH_CR: float = 5.0
    BLOCK_DEAL_HIGH_CR: float = 50.0

    # ------------------------------------------------------------------
    # Delivery pct z-score window
    # ------------------------------------------------------------------
    DELIVERY_ZSCORE_WINDOW_DAYS: int = 30

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    NEWS_TTL_HOURS: int = 4
    SIGNAL_TTL_MINUTES: int = 60
    MAX_SYMBOLS_PER_ARTICLE: int = 3
    DEDUP_TTL_HOURS: int = 48

    # ------------------------------------------------------------------
    # FastAPI
    # ------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8765

    # ------------------------------------------------------------------
    # Streamlit
    # ------------------------------------------------------------------
    DASHBOARD_PORT: int = 8501

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    MAX_CONSECUTIVE_FAILURES: int = 3
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ------------------------------------------------------------------
    # Watched symbols (default: NIFTY 50 top 10 + indices)
    # ------------------------------------------------------------------
    WATCHED_SYMBOLS: list[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "WIPRO", "AXISBANK", "KOTAKBANK",
        "NIFTY50", "BANKNIFTY",
    ])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Config:
        """Construct a Config by reading environment variables.

        Each field maps to an uppercase env var of the same name.
        Unset variables fall back to the dataclass defaults shown above.
        """
        def _str(name: str, default: str) -> str:
            return os.getenv(name, default)

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        def _strlist(name: str, default: list[str]) -> list[str]:
            raw = os.getenv(name)
            if raw is None:
                return default
            # Env var format: comma-separated values, e.g. "URL1,URL2,URL3"
            return [item.strip() for item in raw.split(",") if item.strip()]

        _default_rss: list[str] = [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "https://www.business-standard.com/rss/markets-106.rss",
            "https://www.livemint.com/rss/markets",
        ]
        _default_symbols: list[str] = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "WIPRO", "AXISBANK", "KOTAKBANK",
            "NIFTY50", "BANKNIFTY",
        ]

        return cls(
            # PostgreSQL
            POSTGRES_DSN=_str("POSTGRES_DSN", "postgresql://trader:trader@localhost:5432/trading_news"),
            # Parquet
            PARQUET_DIR=_str("PARQUET_DIR", "data/news_pipeline"),
            # NLP
            FINBERT_MODEL=_str("FINBERT_MODEL", "ProsusAI/finbert"),
            SPACY_MODEL=_str("SPACY_MODEL", "en_core_web_sm"),
            NER_CONFIDENCE_THRESHOLD=_float("NER_CONFIDENCE_THRESHOLD", 0.7),
            # Pulse
            PULSE_URL=_str("PULSE_URL", "https://pulse.zerodha.com/"),
            PULSE_POLL_INTERVAL_SEC=_int("PULSE_POLL_INTERVAL_SEC", 300),
            PULSE_TIMEOUT_SEC=_int("PULSE_TIMEOUT_SEC", 15),
            # nselib / jugaad
            NSELIB_POLL_INTERVAL_SEC=_int("NSELIB_POLL_INTERVAL_SEC", 600),
            JUGAAD_POLL_INTERVAL_SEC=_int("JUGAAD_POLL_INTERVAL_SEC", 3600),
            # RSS
            RSS_POLL_INTERVAL_SEC=_int("RSS_POLL_INTERVAL_SEC", 300),
            RSS_FEEDS=_strlist("RSS_FEEDS", _default_rss),
            # Fusion weights
            WEIGHT_NEWS=_float("WEIGHT_NEWS", 0.40),
            WEIGHT_BULK_DEAL=_float("WEIGHT_BULK_DEAL", 0.30),
            WEIGHT_DELIVERY=_float("WEIGHT_DELIVERY", 0.20),
            WEIGHT_VIX=_float("WEIGHT_VIX", 0.10),
            # Signal thresholds
            STRONG_BUY_THRESHOLD=_float("STRONG_BUY_THRESHOLD", 0.60),
            BUY_THRESHOLD=_float("BUY_THRESHOLD", 0.30),
            SELL_THRESHOLD=_float("SELL_THRESHOLD", -0.30),
            STRONG_SELL_THRESHOLD=_float("STRONG_SELL_THRESHOLD", -0.60),
            # VIX dampening
            VIX_HIGH_THRESHOLD=_float("VIX_HIGH_THRESHOLD", 20.0),
            VIX_DAMPEN_FACTOR=_float("VIX_DAMPEN_FACTOR", 0.7),
            # Bulk/block deal thresholds
            BULK_DEAL_HIGH_CR=_float("BULK_DEAL_HIGH_CR", 5.0),
            BLOCK_DEAL_HIGH_CR=_float("BLOCK_DEAL_HIGH_CR", 50.0),
            # Delivery
            DELIVERY_ZSCORE_WINDOW_DAYS=_int("DELIVERY_ZSCORE_WINDOW_DAYS", 30),
            # Pipeline lifecycle
            NEWS_TTL_HOURS=_int("NEWS_TTL_HOURS", 4),
            SIGNAL_TTL_MINUTES=_int("SIGNAL_TTL_MINUTES", 60),
            MAX_SYMBOLS_PER_ARTICLE=_int("MAX_SYMBOLS_PER_ARTICLE", 3),
            DEDUP_TTL_HOURS=_int("DEDUP_TTL_HOURS", 48),
            # FastAPI
            API_HOST=_str("API_HOST", "0.0.0.0"),
            API_PORT=_int("API_PORT", 8765),
            # Streamlit
            DASHBOARD_PORT=_int("DASHBOARD_PORT", 8501),
            # Monitoring
            MAX_CONSECUTIVE_FAILURES=_int("MAX_CONSECUTIVE_FAILURES", 3),
            TELEGRAM_BOT_TOKEN=_str("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=_str("TELEGRAM_CHAT_ID", ""),
            # Symbols
            WATCHED_SYMBOLS=_strlist("WATCHED_SYMBOLS", _default_symbols),
        )


# Module-level singleton — import and use directly:
#   from algo_trading.news_pipeline.config import cfg
cfg: Config = Config.from_env()
