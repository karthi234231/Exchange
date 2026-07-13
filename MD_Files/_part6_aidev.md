## 15. SCRIPTS

### 15.1 algo_trading/scripts/download_history.py

**Constants:**
- `NSE_SYMBOL_MAP: dict[str,str]` — NSE → Yahoo Finance symbol mapping
- `_SHOONYA_VALID_INTERVALS`, `_YFINANCE_VALID_INTERVALS`
- `_SHOONYA_MAX_HISTORY_DAYS: int = 400`, `_YFINANCE_15MIN_MAX_DAYS: int = 60`

**Class HistoricalDataDownloader:**
- `__init__(output_dir=Path("data/historical")) -> None`
- `download_yfinance(symbol, years=3, interval="15m") -> pl.DataFrame`
- `download_shoonya(symbol, days=400, interval="15") -> pl.DataFrame` — paginated API calls
- `save_parquet(df, symbol, timeframe) -> Path`
- `load_parquet(symbol, timeframe) -> pl.DataFrame`
- `_pandas_to_polars_ohlcv(raw_df, symbol) -> pl.DataFrame` — IST timezone

**Function:** `main() -> None` — CLI with argparse

---

### 15.2 algo_trading/scripts/download_fo_history.py

**Constants:**
- `_DEFAULT_SYMBOLS`, `_DEFAULT_DAYS: int = 30`, `_STALENESS_WARN_DAYS: int = 7`
- `_SEED_BASE: int = 20260618`, `_N_STRIKES_SYNTHETIC: int = 10`, `_RISK_FREE_RATE: float = 0.065`
- `_SYNTHETIC_SPOT_PRICES: dict`

**Functions:**
- `_chain_to_dataframe(chain) -> pd.DataFrame`
- `_save_chain_parquet(df, output_path) -> None`
- `_validate_output_file(path) -> bool`, `_is_stale(path) -> bool`
- `_run_synthetic_mode(symbols, days, output_dir) -> tuple[int,int,list[date]]`
- `_run_live_mode(symbols, days, output_dir) -> tuple[int,int,list[date]]` — NseIndiaApi
- `_print_progress(...)`, `_print_summary(...)`, `_parse_args() -> Namespace`, `main() -> None`

---

### 15.3 algo_trading/scripts/generate_sample_data.py

**Constants:**
- `RNG_SEED: int = 42`, `TRADING_DAYS: int = 756`, `BAR_MINUTES: int = 15`
- `BARS_PER_DAY: int = 25`, `RISK_FREE_RATE: float = 0.065`
- `SYMBOL_PARAMS: dict` — per-symbol start_price, annual_mu, annual_sigma
- `DAILY_VOL_MEAN: int = 1_000_000`

**Functions:**
- `_project_root() -> Path`, `_historical_dir() -> Path`
- `_build_timestamps(n_days, bars_per_day) -> list[datetime]` — NSE schedule from 2021-01-04
- `_gbm_closes(rng, start_price, annual_mu, annual_sigma, n_bars, dt) -> np.ndarray` — GBM price path
- `_compute_atr14(highs, lows, closes) -> np.ndarray` — forward fill
- `_intraday_volume_profile(bars_per_day) -> np.ndarray` — U-shaped weights
- `_build_ohlcv(rng, symbol, start_price, annual_mu, annual_sigma, timestamps) -> pl.DataFrame`
- `_build_tick_data(rng) -> pl.DataFrame` — 100K ticks with Hawkes clustering
- `_norm_cdf(x) -> float`, `_norm_pdf(x) -> float`
- `_bs_greeks(S, K, T, r, sigma, option_type) -> tuple[float,float,float,float,float,float]`
- `_build_fo_chain(rng) -> pl.DataFrame` — 11 strikes × 2 expiries × 2 types
- `main() -> None` — generates 5 symbols × 19,656 bars + ticks + F&O chain Parquet files

---

### 15.4 algo_trading/scripts/paper_trade.py

**Constants:**
- `_DEFAULT_SYMBOLS`, `_SESSION_OPEN="09:15"`, `_SESSION_CLOSE="15:30"`, `_FORCE_FLAT="15:15"`, `_DAILY_SUMMARY_TIME="15:30"`
- `_PUB_PORT_DATA=5555`, `_PUB_PORT_FEATURES=5556`, `_PULL_PORT_SIGNALS=5557`

**Dataclasses:**
- `PaperFill`: symbol, side, price, quantity, timestamp, signal_confidence
- `_OpenPosition`: symbol, side, entry_price, quantity, entry_time
- `_ClosedTrade`: symbol, side, entry_price, exit_price, quantity, pnl_gross, pnl_net, entry_time, exit_time

**Class PaperTradeTracker:**
- `__init__(capital_inr) -> None`
- `open_position(fill) -> None`, `close_position(symbol, exit_price, timestamp) -> Optional[dict]`
- `daily_summary() -> dict` — pnl, win_rate, sharpe, max_dd, equity, open_positions
- `_compute_sharpe(pnl_series) -> float`, `_compute_max_drawdown() -> float`

**Class _PaperTradeLogger:**
- `__init__(log_dir, session_date) -> None`
- `log_fill(fill, event="open") -> None`, `log_close(symbol, exit_price, pnl_dict) -> None`
- `write_summary(summary) -> None`, `close(self) -> None`

**Functions:**
- `_simulate_fill(symbol, signal, capital, max_position_pct=0.05) -> PaperFill`
- `_decode_signal(raw: bytes) -> dict`
- `_launch_zmq_processes(symbols) -> list[subprocess.Popen]` — DataIngestor, FeatureComputer, MLInference
- `run_paper_session(symbols, capital, duration_days, log_dir) -> None` — main loop
- `main() -> None`

---

### 15.5 algo_trading/scripts/train_models.py

**Constants:**
- `_RETRAIN_DAY_OF_WEEK: int = 6`, `_RETRAIN_HOUR_IST: int = 20`
- `_MODEL_OUTPUT_DIR: str = "models/trained"`, `_OPTUNA_N_TRIALS: int = 50`
- `_SESSION_LOCK_FILE: str = "state/session_active.lock"`, `_MIN_SYMBOLS_FOR_TRAINING: int = 3`

**Functions:**
- `is_safe_to_retrain() -> bool` — Sunday >= 20:00 IST only
- `run_training_pipeline(symbols, data_dir="data/historical", output_dir=..., optuna_trials=50, validate_only=False) -> bool` — 10-step pipeline
- `_load_symbol_data(symbol, data_dir) -> pl.DataFrame | None`
- `_fit_and_save_statistical_models(featured_datasets, output_dir) -> None` — HMM+GARCH
- `_compute_features(df, symbol) -> pl.DataFrame | None`
- `_apply_triple_barrier(df, symbol) -> pl.DataFrame | None`
- `_run_optuna_tuning(all_splits, n_trials) -> dict`
- `_quick_fold_score(train_df, test_df, params) -> float | None`
- `_default_hyperparams() -> dict`
- `_train_ensemble(all_splits, best_params, output_dir) -> object | None`
- `_compile_treelite(model, output_dir) -> None`
- `_generate_shap_report(model, all_splits, output_dir) -> None`
- `_track_experiment(model, best_params, edge_report, output_dir) -> None`
- `_register_frozen_spec(model, symbols, output_dir) -> None` — writes FrozenModelSpec JSON
- `main() -> None` — CLI with --force gate bypass

---

### 15.6 algo_trading/scripts/train_rl.py

**Constants:**
- `_DEFAULT_TIMESTEPS: int = 500_000`, `_VALIDATION_SPLIT_PCT: float = 0.20`
- `_MIN_SHARPE_WARN: float = 0.5`

**Functions:**
- `_step(n, msg) -> None`, `_check_stage_gate() -> None` — verifies Stage 2A spec
- `_load_historical_data(symbols) -> dict`
- `_compute_features(data) -> dict` — Technical+Statistical+Custom features
- `_load_ensemble_signals(data) -> dict`
- `_df_to_env_arrays(df) -> tuple` — (features, closes, atrs, timestamps)
- `_build_train_envs(data, val_split) -> tuple[list, object]` — rolling 1-year windows
- `_is_safe_to_retrain() -> bool`
- `_log_to_mlflow(metrics, config_params, model_path) -> None`
- `_run_validate_only(model_path) -> None`
- `run_training_pipeline(symbols, timesteps, force, model_path) -> None` — 10-step RL pipeline
- `main() -> None`

---

### 15.7 algo_trading/scripts/validate_edge.py

**Constants:**
- `_OPEN_HOUR_START: int = 9`, `_OPEN_HOUR_END: int = 10`
- `_HIGH_ATR_PERCENTILE: float = 0.90`, `_STRESS_MAX_DRAWDOWN: float = 0.20`
- `_MIN_FOLD_TRADES: int = 100`

**Dataclass EdgeReport:**
- `symbols, n_folds, total_trades, sharpe, profit_factor, win_rate, regime_sharpe: dict[str,float], stress_window_results: dict[str,float], gate_passed, gate_failures: list[str]`

**Functions:**
- `validate_ema_edge(symbols, years=3, data_dir="data/historical") -> EdgeReport`
- `_bootstrap_sharpe_ci(returns, n_boot=1000, ci=0.95) -> tuple[float,float]`
- `_aggregate_reports(reports) -> BacktestReport`
- `_compute_stress_windows(reports, all_trades_by_report=None) -> dict[str,float]`
- `_check_stress_gates(stress_results) -> list[str]`
- `_print_report(report) -> None`
- `main() -> None`

---

### 15.8 algo_trading/scripts/compare_signals.py

**Dataclasses:**
- `VariantConfig`: name, use_adx, use_volume, use_rs, ema_fast=9, ema_slow=21, use_orb=False
- `_SimpleBar`: symbol, timestamp, open, high, low, close, volume, atr
- `VariantResult`: variant_name, symbol, sharpe, max_drawdown, win_rate, trades, profit_factor, filter_stats
- `_TradeResult`: entry_bar, exit_bar, entry_price, exit_price, pnl_gross, pnl_net, direction

**Constants:**
- `VARIANTS: list[VariantConfig]` — 5 variants: baseline EMA, +ADX, +Volume, +RS, ORB
- `_INITIAL_CAPITAL: Decimal = 500_000`
- `_N_FOLDS: int = 5`

**Functions:**
- `_load_parquet_bars(parquet_path, symbol) -> list[_SimpleBar]`
- `_generate_ema_signals(bars, fast=9, slow=21) -> list[int]`
- `_generate_filtered_ema_signals(bars, use_adx, use_volume, use_rs) -> list[int]`
- `_generate_orb_signals(bars) -> list[int]`
- `_simulate_bars(bars, signals, symbol) -> list[_TradeResult]` — full backtest with IndiaCostModel
- `_walk_forward_splits(n, n_folds=5, purge_bars=51) -> list[tuple]`
- `_compute_sharpe(returns) -> float`, `_compute_max_drawdown(returns) -> float`
- `_compute_metrics(trades, bars) -> dict`
- `_run_variant(variant, bars, symbol) -> VariantResult`
- `_print_comparison_table(results) -> None`, `_save_results(results, output_path) -> None`
- `main() -> None`

---

### 15.9 algo_trading/scripts/model_stability_report.py

**Constants:**
- `_SHARPE_RED_FLAG: float = -0.5`, `_RANK_CORR_RED_FLAG: float = 0.5`
- `_PROB_NEAR_HALF_THRESHOLD: float = 0.05`, `_REGIME_MIN_POSITIVE_FRACTION: float = 2/3`

**Functions:**
- `_compute_fold_sharpe(pred_probas, y_true, class_to_signal=None) -> float`
- `_spearman_rank_corr_matrix(importance_arrays) -> np.ndarray`
- `_compute_calibration_metrics(pred_probas, y_true, n_bins=10) -> dict` — reliability diagram
- `_compute_confusion_matrix(y_true, y_pred, n_classes=3) -> list[list[int]]`
- `_regime_stratified_sharpe(pred_probas, y_true, regime_labels, regime_names=None) -> dict[str,float]`
- `run_stability_report(model_version=None, oof_file=None, registry_dir=..., output_dir=...) -> dict`
- `_try_load_regime_labels(oof_path, n_samples) -> np.ndarray | None`
- `_print_stability_summary(report) -> None`, `_save_plots(report, out_path, ver_tag) -> None`
- `main() -> None`

---

### 15.10 algo_trading/scripts/run_fo.py

**Constants:** `_VALID_SYMBOLS: set = {"NIFTY","BANKNIFTY"}`, `_MIN_DTE: int = 7`, `_MAX_DTE: int = 45`

**Functions:**
- `_parse_args() -> Namespace` — symbol, dte, dry-run, force-run
- `_validate_args(args) -> None`, `main() -> None`

---

### 15.11 algo_trading/scripts/select_features.py

**Constants:**
- `_SHAP_LOW_THRESHOLD: float = 0.001`, `_CORR_HIGH_THRESHOLD: float = 0.90`
- `_INSTABILITY_RATIO: float = 0.5`, `_N_FOLDS_STABILITY: int = 5`

**Functions:**
- `_compute_shap_values_for_model(model, X, model_name) -> np.ndarray | None`
- `_fallback_importance(model, model_name) -> np.ndarray | None`
- `_compute_rank_stability(fold_importance_arrays) -> np.ndarray`
- `_recompute_fold_importances(model, model_name, X, y, n_folds) -> list[np.ndarray]`
- `_find_correlated_pairs(X, feature_names, mean_shap, threshold=0.90) -> list[dict]`
- `run_feature_selection(model_version, data_dir=..., registry_dir=..., output_dir=..., quick=False, n_shap_samples=2000) -> dict`
- `_load_feature_matrix(data_dir, feature_names) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]`
- `_print_selection_table(feature_names, mean_shap, rank_stds, corr_pairs, keep) -> None`
- `main() -> None`

---

## 16. NEWS MODULE

### 16.1 algo_trading/news/event_classifier.py

**Enum EventType(str, Enum):**
- `EARNINGS, DIVIDEND, BUYBACK, M_AND_A, MANAGEMENT, REGULATORY, LEGAL, MACRO, IPO_LISTING, SECTOR, GENERIC`

**Enum ImpactLevel(str, Enum):** CRITICAL, HIGH, MEDIUM, LOW

**Keyword sets (frozensets):**
- `_EARNINGS_KWORDS`, `_DIVIDEND_KWORDS`, `_BUYBACK_KWORDS`, `_M_AND_A_KWORDS`
- `_MANAGEMENT_KWORDS`, `_REGULATORY_KWORDS`, `_LEGAL_KWORDS`, `_MACRO_KWORDS`, `_IPO_KWORDS`
- `_BULLISH_KWORDS`, `_BEARISH_KWORDS`, `_RBI_RATE_TERMS`, `_CEO_RESIGN_TERMS`

**Dataclass NewsEvent:**
- `article_id, symbol, headline, url, published_at, event_type, impact, sentiment_hint, source, raw_body`

**Class EventClassifier:**
- `classify(article, symbols) -> list[NewsEvent]` — rule-based (no ML)
- `_contains_any(text, kwords) -> bool`
- `_detect_event_type(text) -> EventType` — priority-ordered detection
- `_detect_sentiment(text) -> str` — "bullish"|"bearish"|"neutral"
- `_compute_impact(event_type, sentiment, text) -> ImpactLevel`

---

### 16.2 algo_trading/news/news_orchestrator.py

**Constants:**
- `_SCRAPER_PUSH_PORT: int = 5570`, `_SENTIMENT_PUSH_PORT: int = 5565`, `_EVENT_PUSH_PORT: int = 5571`

**Class NewsOrchestrator:**
- `__init__(symbols) -> None` — scraper, classifier, symbol_mapper, SentimentPublisher
- `_process_articles_loop(self) -> None` — background thread: ZMQ PULL → classify → PUSH
- `start(self) -> None`, `run_async(self) -> None`, `run(self) -> None`

**Function:** `main() -> None` — CLI entry point

---

### 16.3 algo_trading/news/scraper.py

**Constants:**
- `_RSS_FEEDS: list` — 4 RSS sources
- `_NSE_ANNOUNCEMENT_URL: str`, `_SEEN_IDS_FILE: Path`
- `_RSS_INTERVAL_SEC: int = 300`, `_NSE_INTERVAL_SEC: int = 120`, `_HTTP_TIMEOUT_SEC: int = 10`

**Dataclass RawArticle:**
- `id, source, headline, url, published_at, body, symbol_hints: list[str]`

**Functions:**
- `_make_id(url) -> str` — SHA-256 of URL
- `_extract_symbol_hints(text) -> list[str]` — regex uppercase 2-15 char words
- `_feedparser_to_datetime(parsed_time) -> datetime`
- `fetch_rss(session, source, url) -> list[RawArticle]` (async)
- `fetch_nse_announcements(session) -> list[RawArticle]` (async)
- `_parse_nse_datetime(an_dt) -> datetime`
- `_article_to_payload(article) -> bytes`
- `run_scraper(zmq_push_port=5570) -> None` — async main loop

**Class NewsDeduplicator:**
- `__init__(state_file=_SEEN_IDS_FILE) -> None`
- `_load() -> None`, `_save() -> None`
- `is_new(article_id) -> bool`, `mark_seen(article_id) -> None`
- `cleanup_old(older_than_hours=48) -> None`

---

### 16.4 algo_trading/news/signal_bridge.py

**Constants:**
- `_NEWS_STATE_TTL_HOURS: int = 4`, `_MACRO_PAUSE_MINUTES: int = 30`, `_EVENT_PUSH_PORT: int = 5571`

**Dataclass SymbolNewsState:**
- `symbol, last_updated, impact, sentiment_hint, event_type, headline`
- `halted=False, confidence_multiplier=1.0, block_long=False, block_short=False, earnings_pending=False`

**Class NewsSignalBridge:**
- `__init__(zmq_pull_port=5571, alert_sender=None, halt_callback=None) -> None`
- `start_listener(self) -> None` — background ZMQ SUB thread
- `get_symbol_state(symbol) -> Optional[SymbolNewsState]` — TTL-expired returns None
- `should_allow_trade(symbol, direction) -> tuple[bool,str]`
- `get_confidence_multiplier(symbol) -> float`
- `clear_symbol(symbol) -> None`, `cleanup_expired_states() -> None`
- `_listener_loop(self) -> None`, `_process_event(event_dict) -> None`, `_send_alert(msg, level="CRITICAL") -> None`

**Function:** `_impact_rank(impact: ImpactLevel) -> int`

---

### 16.5 algo_trading/news/symbol_mapper.py

**Constants:**
- `_NIFTY100_ALIASES: dict[str,str]` — 100+ company name → ticker mappings
- `_NIFTY50_SYMBOLS: frozenset[str]`, `_MACRO_TERMS: frozenset[str]`

**Class SymbolMapper:**
- `__init__() -> None`
- `_check_rapidfuzz() -> bool` (static)
- `resolve(text, hints) -> list[str]` — 3-pass: exact, alias, fuzzy (rapidfuzz); max 5 results
- `is_macro_news(text) -> bool`
- `is_nifty50(symbol) -> bool`

---

## 17. NEWS PIPELINE

### 17.1 news_pipeline/config.py

**Dataclass Config (frozen=True):**
- DB: `POSTGRES_DSN`, `PARQUET_DIR`
- NLP: `FINBERT_MODEL`, `SPACY_MODEL`, `NER_CONFIDENCE_THRESHOLD`
- Signal fusion weights: `WEIGHT_NEWS=0.40, WEIGHT_BULK_DEAL=0.30, WEIGHT_DELIVERY=0.20, WEIGHT_VIX=0.10`
- Signal thresholds: `STRONG_BUY=0.60, BUY=0.30, SELL=-0.30, STRONG_SELL=-0.60`
- VIX: `VIX_HIGH_THRESHOLD=20.0, VIX_DAMPEN_FACTOR=0.7`
- Deals: `BULK_DEAL_HIGH_CR=5.0, BLOCK_DEAL_HIGH_CR=50.0`
- TTL: `NEWS_TTL_HOURS=4, SIGNAL_TTL_MINUTES=60, DEDUP_TTL_HOURS=48`
- API: `API_HOST="0.0.0.0", API_PORT=8765`, Dashboard: `DASHBOARD_PORT=8501`
- Poll intervals: `PULSE_POLL_INTERVAL_SEC`, `NSELIB_POLL_INTERVAL_SEC`, `JUGAAD_POLL_INTERVAL_SEC`, `RSS_POLL_INTERVAL_SEC`
- `@classmethod from_env() -> Config`

**Module singleton:** `cfg: Config`

---

### 17.2 news_pipeline/models.py

**Functions:**
- `make_id(data: str) -> str` — SHA-256 hex, `_dt_to_iso(dt) -> str | None`

**Dataclasses:**
- `RawNewsItem`: id, source, headline, url, published_at, fetched_at, body, symbol_hints; `to_dict()`
- `ProcessedNewsEvent`: id, source, headline, url, published_at, fetched_at, symbols, event_type, sentiment, sentiment_score, sentiment_confidence, raw_body, entities, tags; `to_dict()`
- `BulkDealRecord`: id, deal_date, symbol, client_name, deal_type, direction, quantity, price, deal_value_cr, fetched_at; `to_dict()`
- `DeliveryRecord`: trade_date, symbol, delivery_qty, traded_qty, delivery_pct, fetched_at; `to_dict()`
- `FusedSignal`: id, generated_at, symbol, signal_type, alpha_score, news_score, bulk_deal_score, delivery_score, vix_score, contributing_events, ttl_minutes; `to_dict()`, `is_expired()`
- `PipelineRun`: run_id, flow_name, started_at, completed_at, status, records_processed, errors, latency_ms; `to_dict()`
- `SignalPerformance`: id, evaluation_date, symbol, signal_type, signal_generated_at, entry_price, exit_price, return_pct, hit, holding_hours; `to_dict()`

**Function:** `new_signal_id() -> str` — UUID4

---

### 17.3 news_pipeline/client.py

**Constants:** `_SIGNAL_PUB_PORT: int = 5572`

**Class NewsPipelineClient:**
- `_BUY: int = 1`, `_SELL: int = -1`
- `__init__(port=5572) -> None`
- `start(self) -> None` — daemon subscriber thread
- `stop(self) -> None`
- `_subscriber_loop(self) -> None` — ZMQ SUB, caches FusedSignal per symbol
- `_parse_signal(data) -> dict | None` (static)
- `_get_live_signal(symbol) -> dict | None` — non-expired with lazy cleanup
- `evaluate_trade(symbol, direction) -> tuple[bool,float,str]` — STRONG_BUY→1.2×, BUY→1.0×, NEUTRAL→1.0×, SELL→block, STRONG_SELL→block

---

### 17.4 news_pipeline/orchestration/flows.py

**Constants:** `_SIGNAL_PUB_PORT: int = 5572`

**Prefect Tasks:**
- `fetch_pulse_task() -> list[dict]`, `fetch_nselib_task() -> list[dict]`, `fetch_rss_task() -> list[dict]`
- `store_raw_items_task(pulse_items, nselib_items, rss_items) -> int`
- `fetch_unprocessed_task(limit=100) -> list[dict]`
- `run_nlp_task(raw_items) -> list[dict]` — NER + FinBERT + tagging
- `store_processed_task(events) -> int`
- `fuse_signals_task(symbols) -> list[dict]`
- `download_bhavcopy_task(trade_date) -> dict`
- `download_ohlcv_task(symbols, trade_date) -> dict`
- `download_delivery_task(trade_date) -> int`
- `evaluate_signals_task(eval_date) -> dict`

**Prefect Flows:**
- `ingest_flow() -> dict` — every 5min: Pulse + nselib + RSS → raw to PG
- `process_flow() -> dict` — every 5min+2s: NLP → processed events to PG
- `fusion_flow(symbols=None) -> dict` — every 5min+4s: fusion → signals to PG + ZMQ PUB
- `history_flow(trade_date=None) -> dict` — daily 16:30 IST: OHLCV + bhavcopy → Parquet
- `backtest_eval_flow(eval_date=None) -> dict` — daily 20:00 IST: signal evaluation

**Functions:**
- `run_pipeline_once(symbols=None) -> None` — one-shot
- `serve_all_flows() -> None` — entry point for Prefect serve mode

---

### 17.5 news_pipeline/ingestion/ (8 files)

**base.py:**
- `FetchResult`: source, success, records, error, latency_ms
- `BaseFetcher(ABC)`: `max_retries=3`, `@abstractmethod source_name`, `@abstractmethod async _fetch_internal()`, `async fetch() -> tuple[list, FetchResult]` — retry + backoff, `is_market_hours() -> bool`

**http_session.py:**
- `CircuitBreaker`: source, failure_threshold=5, recovery_timeout=60.0; `allow()`, `record_success()`, `record_failure()`, `is_open`, `state`
- `NSESession`: `connect()`, `close()`, `get(url)` — rate-limited with auto cookie refresh

**jugaad_fetcher.py:**
- `JugaadFetcher(BaseFetcher)`: source_name="jugaad"; `fetch_ohlcv()`, `fetch_bhavcopy()`, `fetch_corporate_actions()`

**nse_direct_client.py:**
- `NSEDirectClient`: `get_bulk_deals(trade_date)`, `get_block_deals(trade_date)`, `get_delivery_data(trade_date)`, `get_india_vix()`, `get_corporate_announcements(days=3)`

**nselib_fetcher.py:**
- `NSElibFetcher(BaseFetcher)`: source_name="nselib"; bulk deals, block deals, delivery, India VIX

**pulse_scraper.py:**
- `PulseScraper(BaseFetcher)`: source_name="pulse"; parses __NEXT_DATA__ + inline JSON + anchor links

**rss_fallback.py:**
- `RSSFallback(BaseFetcher)`: source_name="rss_fallback"; 8 RSS feeds + NSE announcements (concurrent)

---

### 17.6 news_pipeline/processing/ (4 files)

**deduplication.py:**
- `Deduplicator`: TTLCache + PG two-layer dedup; `filter_new(items)`, `is_new(item)`, `mark_seen(id)`

**event_tagger.py:**
- Keyword sets for 9 event types + sector map (13 sectors)
- `EventTagger`: `tag(headline, body, sentiment_label) -> tuple[str,str,list[str]]`

**finbert_adapter.py:**
- LRU cache (2000 entries), keyword fallback
- `FinBERTAdapter`: `score(text) -> tuple[float,str,float]`, `score_batch(texts) -> list[tuple]`
- Module singleton: `finbert: FinBERTAdapter`

**ner_pipeline.py:**
- 100+ company aliases, spaCy EntityRuler, regex fallback
- `NERPipeline`: `extract_symbols(text, hints=None) -> list[str]` (4-pass), `extract_entities(text) -> dict`, `is_macro_news(text) -> bool`

---

### 17.7 news_pipeline/fusion/ (2 files)

**alpha_scorer.py:**
- `ComponentScores`: news_score, bulk_deal_score, delivery_score, vix_score, alpha_score, signal_type
- `AlphaScorer`: `compute_news_score()`, `compute_bulk_deal_score()`, `compute_delivery_score()`, `compute_vix_adjustment()`, `combine() -> ComponentScores`
- Weights: news=0.40, bulk_deal=0.30, delivery=0.20, vix=0.10

**signal_fuser.py:**
- `SignalFuser`: `fuse_all(symbols) -> list[FusedSignal]`, `fuse_one(symbol) -> FusedSignal | None`, `get_cached_signal(symbol) -> FusedSignal | None`

---

### 17.8 news_pipeline/storage/ (3 files)

**cache.py:**
- `TTLCache`: `set(key, value, ttl_seconds=None)`, `get(key)`, `exists(key)`, `delete(key)`, `cleanup() -> int`, `size()`
- Singletons: `news_id_cache` (48h TTL), `signal_cache` (60min TTL)

**parquet_store.py:**
- `ParquetStore`: `write_news_events()`, `write_bulk_deals()`, `write_signals()`, `write_ohlcv()`, `read_signals()`, `read_ohlcv()`, `read_news_events()`

**pg_store.py:**
- 6 PostgreSQL tables + indexes
- `PGStore`: news events CRUD, bulk deals CRUD, delivery CRUD, signals CRUD, pipeline runs CRUD, signal performance CRUD; uses asyncpg connection pool

---

### 17.9 news_pipeline/api/ (6 files)

**main.py:**
- FastAPI app with lifespan (connect PG + Parquet, create backtester), CORS (allow all origins)
- Routers: signals, news, health, bulk_deals, backtest
- `GET /` → redirect to /docs

**schemas.py (Pydantic v2):**
- `SignalResponse`: symbol, signal_type, alpha_score, all component scores, generated_at, ttl_minutes, contributing_events, expired
- `NewsEventResponse`: id, source, headline, url, published_at, symbols, event_type, sentiment, sentiment_score, tags
- `BulkDealResponse`: all deal fields
- `HealthResponse`: status, flows, overall_healthy, checked_at
- `BacktestSummaryResponse`: period_days, total_evaluated, win_rate, avg_return_pct, sharpe_ratio, max_drawdown, by_signal_type, by_symbol, best_signal_type, best_symbol, recent_10

**routes/signals.py:** `GET /signals/` (cache→PG fallback), `GET /signals/{symbol}`
**routes/news.py:** `GET /news/` (filters: symbol, hours, sentiment), `GET /news/{symbol}`
**routes/bulk_deals.py:** `GET /bulk-deals/`, `GET /bulk-deals/{symbol}/history`
**routes/health.py:** `GET /health/`, `GET /health/flows/{flow_name}`
**routes/backtest.py:** `GET /backtest/summary`

---

### 17.10 news_pipeline/monitoring/tracker.py

**Class LatencyProfiler:** context manager measuring wall-clock duration

**Class FailureCounter:** thread-safe consecutive failure counter
- `@classmethod increment(flow_name) -> int`, `reset(flow_name)`, `get(flow_name) -> int`

**Class PipelineTracker:**
- `start() -> PipelineTracker`, `add_records(count)`, `add_error(msg)`
- `async finish(success=True) -> PipelineRun`
- `@classmethod @asynccontextmanager async run(flow_name, pg_store)` — async context manager

**Class PipelineHealthSummary:**
- `@staticmethod async compute(pg_store) -> dict` — {flows: {...}, overall_healthy: bool}

---

### 17.11 news_pipeline/dashboard/app.py (Streamlit)

**Functions:**
- `@st.cache_data fetch_signals(symbol, limit) -> list[dict]`
- `@st.cache_data fetch_news(symbol, hours, limit) -> list[dict]`
- `@st.cache_data fetch_bulk_deals(symbol) -> list[dict]`
- `@st.cache_data fetch_health() -> dict`
- `@st.cache_data fetch_backtest_summary(days) -> dict`
- `render_signal_badge(signal_type) -> str`
- `render_top_signals_panel(signals) -> None`
- `render_all_signals_panel(signals) -> None`
- `render_news_feed_panel(news_items) -> None`
- `render_bulk_deals_panel(deals) -> None`
- `render_pipeline_health_panel(health) -> None`
- `render_backtest_panel(summary) -> None`
- `main() -> None`

---

### 17.12 news_pipeline/backtest/signal_backtest.py

**Class SignalBacktester:**
- `__init__(pg, parquet) -> None`
- `evaluate_yesterday(self) -> dict`
- `evaluate_date(signal_date) -> dict` — signals vs next-day price move
- `evaluate_range(start, end) -> dict`
- `summary_report(days=30) -> dict` — full backtest: hit_rate, avg_return, sharpe, max_drawdown, by_signal_type, by_symbol
- `_compute_sharpe(returns) -> float`, `_compute_max_drawdown(returns) -> float`
- `_aggregate_metrics(performances) -> dict`

---

END OF PART 6.
