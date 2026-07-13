# NSE AlgoTrader — Complete Codebase Reference
*Auto-generated 2026-06-22. Covers 164 Python files across 16 modules.*

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Infrastructure](#2-infrastructure)
3. [Data Layer](#3-data-layer)
4. [Features](#4-features)
5. [Signals](#5-signals)
6. [Labeling](#6-labeling)
7. [Models](#7-models)
8. [Execution](#8-execution)
9. [Risk](#9-risk)
10. [Alerts](#10-alerts)
11. [Backtest](#11-backtest)
12. [F&O Module](#12-fo-module)
13. [News & Sentiment Pipeline](#13-news--sentiment-pipeline)
14. [Orchestrator](#14-orchestrator)
15. [Scripts](#15-scripts)
16. [Market Data Utilities](#16-market-data-utilities)
17. [Miscellaneous / Top-Level](#17-miscellaneous--top-level)

---

## 1. PROJECT OVERVIEW & ARCHITECTURE

### 1.1 System Description
NSE AlgoTrader is a stage-gated algorithmic trading system for Indian equity and F&O markets. Primary broker: Shoonya (Finvasia) — zero brokerage. Storage: QuestDB (ILP port 9000, PG wire 8812) for all time-series data. ML: XGBoost + LightGBM + CatBoost ensemble → LogisticRegression meta-learner, compiled to Treelite .so. Stage-gated build: Stage 1 (EMA cross, 15-min time bars) → Stage 2A (ML ensemble) → Stage 2B (TrueData+microstructure) → Stage 3 (Hawkes/TFI, sub-minute) → Stage 4 (F&O+RL+sentiment). Hardware: Intel i5-1345U, 16GB RAM, CPU-only. SEBI-regulated — all orders tagged with strategy_id for compliance.

### 1.2 Inter-Process Communication (ZMQ)
- Port 5555: Tick PUB (broker_shoonya/truedata_feed → feature engine)
- Port 5558: Bar PUB (candle_store → downstream consumers)
- Port 5556: Feature PUB (feature engine → inference engine)
- Port 5557: Signal PUSH (inference engine → order executor)
- Port 5560: Alert PUSH (anywhere → alert_sink)
- Port 5565: Sentiment PUSH (SentimentPublisher → SentimentWorker)
- Port 5566: Sentiment PUSH (SentimentWorker → SentimentAggregator)
- Port 5570: News raw PUSH (scraper → news orchestrator)
- Port 5571: News event PUSH (classifier → signal bridge)
- Port 5572: News signal PUB (news pipeline → DailyOrchestrator)

### 1.3 Feature Vector by Stage
- Stage 1–2A: 25 features (FEATURE_NAMES in features/engine.py)
- Stage 2B: 27 features (+vpin_signal_enc, reservation_price_offset)
- Stage 3: 31 features (+hawkes_ratio, hawkes_signal_enc, tfi, tfi_zscore)
- Stage 4: 38 features (+pcr_oi, max_pain_dist_pct, iv_rank, iv_skew, atm_iv, sentiment_score, oi_buildup_enc)

### 1.4 Key Constraints
- Primary broker: Shoonya only. AngelOne = dev reference, forbidden in production
- Storage: QuestDB only for time-series (no SQLite)
- Live trading: hard-disabled until 13 gates pass (3 months paper, Sharpe ≥1.0, etc.)
- Bar type: 15-min time bars Stage 1, volume bars Stage 2A+ (TRAIN_BAR_TYPE == LIVE_BAR_TYPE always)
- ML: No TabPFN, RL, deep models in production until deterministic baseline is profitable
- Microstructure (VPIN/Avellaneda-Stoikov): Stage 2B+ only, requires TrueData feed
- Hawkes/TFI: Stage 3+ only (sub-minute data required — raises NotImplementedError at 15-min)
- RL sizer: Stage 4 only

---

## 2. INFRASTRUCTURE

### 2.1 algo_trading/config.py

**Module-level:**
- `log: logging.Logger` — module logger
- `_ENV_FILE: Path` — path to .env file

**Enum TradingMode(str, Enum):**
- `MOCK = "mock"`
- `PAPER = "paper"`
- `LIVE = "live"`

**Class _SecretRedactFilter(logging.Filter):**
- `__init__(self, secrets: list[str]) -> None` — stores secrets sorted longest-first
- `filter(self, record: logging.LogRecord) -> bool` — replaces secrets in log messages with "[REDACTED]"

**Class Settings(BaseSettings):**
Config: `SettingsConfigDict(env_file, case_sensitive=False, extra="ignore")`

Fields — Shoonya credentials:
- `shoonya_user: str` (required)
- `shoonya_password: SecretStr` (required)
- `shoonya_totp_secret: SecretStr` (required, base32-validated)
- `shoonya_vendor_code: str` (required)
- `shoonya_api_secret: SecretStr` (required)
- `shoonya_imei: str` (required)

Fields — TrueData (optional):
- `truedata_user: Optional[str] = None`
- `truedata_password: Optional[SecretStr] = None`

Fields — Trading mode:
- `trading_mode: TradingMode = TradingMode.PAPER`
- `enable_live_trading: bool = False`

Fields — Data feed:
- `data_feed: Literal["shoonya","truedata"] = "shoonya"`
- `use_microstructure: bool = False`

Fields — Bar configuration:
- `bar_type: Literal["time","volume"] = "time"`
- `bar_interval_minutes: int = 15` (ge=1, le=60)

Fields — NSE session times (IST):
- `session_start_hhmm: str = "09:15"`
- `session_end_hhmm: str = "15:30"`
- `no_new_trades_after_hhmm: str = "15:15"`
- `force_close_hhmm: str = "15:15"`
- `skip_first_bar: bool = True`

Fields — Strategy/compliance:
- `strategy_id: Optional[str] = None`
- `broker_approval_date: Optional[str] = None`

Fields — Capital & risk:
- `capital_inr: Decimal = Decimal("100000")`
- `max_daily_loss_pct: float = 0.02` (ge=0.005, le=0.05)
- `max_position_pct: float = 0.05` (ge=0.01, le=0.20)
- `max_gross_exposure_pct: float = 0.20` (ge=0.05, le=1.0)
- `max_drawdown_pct: float = 0.10` (ge=0.02, le=0.30)

Fields — Order execution:
- `max_orders_per_second: int = 8` (ge=1, le=10)
- `stale_order_timeout_sec: int = 20` (ge=5)
- `max_price_drift_bps: int = 30` (ge=5)
- `max_order_modifications: int = 5` (ge=1)
- `product_type: Literal["MIS","CNC"] = "MIS"`
- `exchange_segment: Literal["NSE","BSE"] = "NSE"`

Fields — EMA Cross:
- `ema_fast_period: int = 9` (ge=2, le=50)
- `ema_slow_period: int = 21` (ge=5, le=200)
- `atr_period: int = 14` (ge=5, le=50)
- `atr_stop_multiplier: float = 1.5` (ge=0.5, le=5.0)
- `min_bars_before_signal: int = 21`

Fields — ZMQ ports:
- `zmq_tick_pub_port: int = 5555`
- `zmq_bar_pub_port: int = 5558`
- `zmq_feature_pub_port: int = 5556`
- `zmq_ml_signal_push_port: int = 5557`
- `zmq_alert_push_port: int = 5560`

Fields — QuestDB:
- `questdb_host: str = "localhost"`
- `questdb_ilp_port: int = 9000`
- `questdb_pg_port: int = 8812`
- `questdb_pg_user: str = "admin"`
- `questdb_pg_password: str = "quest"`
- `questdb_pg_database: str = "qdb"`

Fields — Alerts:
- `telegram_bot_token: Optional[SecretStr] = None`
- `telegram_chat_id: Optional[str] = None`

Fields — System:
- `log_level: str = "INFO"`
- `log_levels: dict[str,str]` (default: {"apscheduler":"WARNING","zmq":"WARNING"})
- `symbols: str = "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK"`
- `log_dir: str = "logs"`

Validators:
- `@field_validator("shoonya_totp_secret")` → `_validate_totp` — validates base32 encoding
- `@field_validator("trading_mode")` → `_validate_mode` — validates mode ∈ {mock,paper,live}
- `@field_validator("log_level")` → `_validate_log_level` — validates level ∈ {DEBUG,INFO,WARNING,ERROR,CRITICAL}
- `@field_validator("ema_slow_period")` → `_validate_ema_periods` — enforces fast < slow
- `@model_validator(mode="after")` → `_live_gate` — BLOCKS live trading unless 13 gates satisfied; checks enable_live_trading, broker_approval_date, strategy_id
- `@model_validator(mode="after")` → `_microstructure_gate` — enforces ARCH-FIX-1: use_microstructure requires truedata feed

Properties:
- `symbol_list -> list[str]` — splits comma-separated symbols
- `max_daily_loss_inr -> Decimal` — capital_inr × max_daily_loss_pct
- `max_position_inr -> Decimal` — capital_inr × max_position_pct
- `max_gross_exposure_inr -> Decimal` — capital_inr × max_gross_exposure_pct

Methods:
- `is_live(self) -> bool` — returns True if LIVE mode + enable_live_trading
- `is_paper(self) -> bool` — returns True if PAPER mode
- `safe_repr(self) -> str` — log-safe summary without secrets
- `install_log_secret_filter(self) -> None` — attaches _SecretRedactFilter to root logger

**Module singleton:** `settings: Settings` — instantiated at import time

---

### 2.2 algo_trading/main.py

**Functions:**
- `parse_args() -> argparse.Namespace` — parses CLI: --mode {mock,paper,live}, --symbols, --skip-market-check
- `main() -> int` — entry point; validates trading day, instantiates DailyOrchestrator, calls .run(); exit codes: 0=clean, 1=exception, 2=config error, 3=reconciliation mismatch

**Module guard:** `if __name__ == "__main__": sys.exit(main())`

---

### 2.3 algo_trading/logging_setup.py

**Constants:**
- `_DISK_SPACE_MIN_BYTES: int = 2 * 1024 * 1024 * 1024` — 2 GB minimum disk threshold

**Functions:**
- `configure_logging() -> None` — idempotent (uses _done flag); creates log dir, 50MB rotating file handler (10 backups), stdout handler (human format), file handler (JSON), structlog processors (contextvars, logger name, level, timestamp, exceptions), suppresses noisy third-party loggers, applies per-module level overrides, checks disk space and sends Telegram alert if < 2GB, installs secret filter
- `get_logger(name: str) -> structlog.stdlib.BoundLogger` — returns structlog logger

---

### 2.4 algo_trading/market_calendar.py

**Constants:**
- `IST: ZoneInfo = ZoneInfo("Asia/Kolkata")`
- `_NSE_HOLIDAYS_2026_FALLBACK: list[date]` — 14 hardcoded 2026 holiday dates
- `_HOLIDAYS_YAML_PATH: Path` — config/nse_holidays.yaml
- `SESSION_START: time = time(9, 15)`
- `SESSION_END: time = time(15, 30)`
- `PRE_OPEN_START: time = time(9, 0)`
- `PRE_OPEN_END: time = time(9, 8)`
- `NO_NEW_TRADES_AFTER: time = time(15, 15)` — [SURV-8]
- `FORCE_CLOSE_TIME: time = time(15, 15)` — [SURV-8]
- `_NSE_HOLIDAYS: frozenset[date]` — loaded at import time from YAML or fallback

**Exceptions:**
- `OutsideTradingHoursError(Exception)` — raised when timestamp outside trading session

**Functions:**
- `_load_holidays() -> frozenset[date]` — loads holidays from YAML or falls back to hardcoded list
- `is_trading_day(d: date | None = None) -> bool` — not weekend AND not in holiday set
- `next_trading_day(d: date | None = None) -> date` — next calendar day that is a trading day
- `previous_trading_day(d: date | None = None) -> date` — previous calendar day that is a trading day
- `is_market_open(dt: datetime | None = None) -> bool` — SESSION_START <= t < SESSION_END
- `is_pre_open(dt: datetime | None = None) -> bool` — 09:00–09:08 window
- `can_enter_new_trade(dt: datetime | None = None) -> bool` — after 09:30, before 15:15, market open [SURV-8]
- `seconds_to_force_close(dt: datetime | None = None) -> float` — seconds until 15:15; 0 if past; -1 if not trading day
- `bar_start_time(ts: datetime, interval_minutes: int = 15) -> datetime` — floor timestamp to bar boundary; raises OutsideTradingHoursError if outside SESSION_START–SESSION_END
- `now_ist() -> datetime` — current time in IST
- `today_ist() -> date` — current date in IST

---

## 3. DATA LAYER

### 3.1 algo_trading/data/broker_base.py

**Dataclass Tick (frozen=True, slots=True):**
- `timestamp: datetime` — timezone-aware (IST)
- `symbol: str`
- `ltp: Decimal` — last traded price
- `volume: int` — cumulative day volume
- `best_bid: Decimal`
- `best_ask: Decimal`
- `oi: int` — open interest (0 for equities)
- `exchange: str = "NSE"`
- `token: str = ""`
- Property `spread -> Decimal` — ask - bid (0 if unavailable)
- Property `mid_price -> Decimal` — (bid+ask)/2 or ltp

**Dataclass Bar (frozen=False, slots=True):**
- `timestamp: datetime` — bar open time (IST)
- `symbol: str`
- `open: Decimal`, `high: Decimal`, `low: Decimal`, `close: Decimal`
- `volume: int`, `vwap: Decimal`, `num_ticks: int`
- `bar_type: str` — "time" or "volume"
- `interval_minutes: int = 15`
- `volume_threshold: int = 0`
- `is_first_bar_of_day: bool = False`
- Property `typical_price -> Decimal` — (high+low+close)/3
- Property `bar_range -> Decimal` — high - low

**Dataclass Order (frozen=True):**
- `symbol: str`, `side: str` ("BUY"|"SELL"), `quantity: int`
- `order_type: str` ("LIMIT"|"MARKET"), `price: Decimal` (0 for MARKET)
- `strategy_id: str` — SEBI compliance field
- `product_type: str = "MIS"`, `exchange: str = "NSE"`
- `tradingsymbol: str = ""`, `trigger_price: Decimal = Decimal(0)`
- `client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))`
- `validity: str = "DAY"`, `disclose_qty: int = 0`
- `__post_init__` validates: side ∈ {BUY,SELL}, order_type ∈ {LIMIT,MARKET}, quantity > 0, price >= 0, strategy_id non-empty

**Dataclass OrderUpdate (frozen=False):**
- `client_order_id: str`, `exchange_order_id: str`
- `event: str` ("ACK"|"REJECT"|"FILL"|"PARTIAL_FILL"|"CANCEL_CONFIRM")
- `filled_quantity: int = 0`, `fill_price: Decimal = Decimal(0)`
- `reject_reason: str = ""`, `timestamp: Optional[datetime] = None`

**Dataclass Fill (frozen=True):**
- `order_id: str`, `exchange_order_id: str`, `symbol: str`
- `side: str`, `quantity: int`, `price: Decimal`
- `timestamp: datetime`, `exchange: str = "NSE"`

**Dataclass Position (frozen=False):**
- `symbol: str`, `net_quantity: int`, `average_price: Decimal`, `exchange: str = "NSE"`
- Property `is_long -> bool`, `is_short -> bool`, `is_flat -> bool`
- `unrealized_pnl(current_price: Decimal) -> Decimal` — (current-avg)×qty

**Type aliases:**
- `TickCallback = Callable[[Tick], None]`
- `OrderUpdateCallback = Callable[[OrderUpdate], None]`

**Abstract class BrokerBase(ABC):**
All methods are abstract:
- `login(self) -> bool` — authenticate, raise SessionError on failure
- `logout(self) -> None` — invalidate session
- `ensure_session(self) -> None` — check health, re-login if needed
- `place_order(self, order: Order) -> str` — returns exchange_order_id
- `cancel_order(self, exchange_order_id: str) -> bool` — idempotent
- `get_positions(self) -> list[Position]`
- `get_order_book(self) -> list[dict]`
- `subscribe_ws(self, symbols: list[str], on_tick: TickCallback, on_order_update: OrderUpdateCallback) -> None`
- `unsubscribe_ws(self, symbols: list[str]) -> None`
- `get_historical_bars(self, symbol: str, interval_minutes: int, start: datetime, end: datetime) -> list[Bar]`
- `get_ltp(self, symbol: str) -> Decimal`

**Exceptions:**
- `BrokerError(Exception)` — base
- `SessionError(BrokerError)` — session invalid/expired
- `OrderRejectedError(BrokerError)` — stores `reason` attribute
- `RateLimitError(BrokerError)` — rate limit exceeded

---

### 3.2 algo_trading/data/broker_shoonya.py

**Constants:**
- `_HEARTBEAT_INTERVAL_SEC = 60`
- `_RECONNECT_BASE_DELAY_SEC = 2`, `_RECONNECT_MAX_DELAY_SEC = 60`
- `_SESSION_MAX_AGE_HOURS = 22`

**Exceptions:**
- `SymbolNotFoundError(Exception)` — symbol cannot be resolved to token
- `BrokerFetchError(Exception)` — API call failed

**Class ShoonyaBroker(BrokerBase):**
`__init__`:
- `_api: ShoonyaApiPy`, `_lock: threading.RLock`
- `_session_token: Optional[str]`, `_session_login_time: Optional[datetime]`
- `_symbol_token_map: dict[str,str]`
- `_on_tick_cb: Optional[TickCallback]`, `_on_order_update_cb: Optional[OrderUpdateCallback]`
- `_ws_connected: threading.Event`, `_ws_subscribed_symbols: list[str]`
- `_heartbeat_thread: Optional[threading.Thread]`
- `_ws_supervisor_thread: Optional[threading.Thread]`
- `_stop_event: threading.Event`
- `_paper_orders: dict[str,dict]` — exchange_order_id → order dict

Authentication:
- `login(self) -> bool` — TOTP auth via pyotp, stores token, starts heartbeat [SURV-10]
- `logout(self) -> None` — stops threads, calls api.logout(), clears token
- `ensure_session(self) -> None` — checks 22h TTL, re-logins if expired
- `_is_session_expired(self) -> bool` — age > _SESSION_MAX_AGE_HOURS
- `_start_heartbeat_thread(self) -> None` — spawns daemon thread
- `_heartbeat_loop(self) -> None` — checks session every 60s, proactive re-login [SURV-10]

Symbol resolution:
- `_resolve_token(self, symbol: str) -> str` — returns numeric token, cached after first lookup
- `resolve_all_symbols(self) -> None` — pre-resolves all configured symbols at startup

Order methods:
- `place_order(self, order: Order) -> str` — PAPER: _simulate_paper_order(); LIVE: api.place_order()
- `_simulate_paper_order(self, order: Order) -> str` — fake order, sends ACK callback
- `simulate_paper_fill(self, exchange_order_id: str, fill_price: Decimal) -> Optional[Fill]` — sends FILL callback
- `cancel_order(self, exchange_order_id: str) -> bool` — idempotent
- `_cancel_paper_order(self, exchange_order_id: str) -> bool`

Position methods:
- `get_positions(self) -> list[Position]`
- `_get_paper_positions(self) -> list[Position]` — derived from filled paper orders
- `get_order_book(self) -> list[dict]`
- `get_ltp(self, symbol: str) -> Decimal` — api.get_quotes()

Historical:
- `get_historical_bars(self, symbol: str, interval_minutes: int, start: datetime, end: datetime) -> list[Bar]` — api.get_time_price_series()

WebSocket:
- `subscribe_ws(...)` — stores callbacks, starts WS + supervisor
- `_start_ws(self, symbols: list[str]) -> None` — binds API WS, waits 10s, subscribes
- `_subscribe_symbols(self, symbols: list[str]) -> None` — exchange|token list
- `unsubscribe_ws(self, symbols: list[str]) -> None`
- `_start_ws_supervisor(self, symbols: list[str]) -> None` — spawns reconnect daemon
- `_ws_reconnect_loop(self, symbols: list[str]) -> None` — exponential backoff reconnection

WS Callbacks:
- `_on_ws_open(self) -> None` — sets _ws_connected
- `_on_ws_close(self) -> None` — clears _ws_connected
- `_on_ws_error(self, error: object) -> None` — logs, clears _ws_connected
- `_on_ws_tick(self, tick_data: dict) -> None` — parses → Tick, calls on_tick_cb
- `_on_ws_order_update(self, update_data: dict) -> None` — parses → OrderUpdate, maps status→event

---

### 3.3 algo_trading/data/candle_store.py

**Constants:**
- `_DEAD_LETTER_PATH: Path` — dead-letter JSONL for failed ILP writes
- `_HAS_QDB_ILP: bool`, `_HAS_PSYCOPG2: bool`
- DDL strings for 4 QuestDB tables: time_bars, orders, fills, risk_state

**Dataclass _CurrentBar:**
- `symbol: str`, `bar_start: datetime`
- `open: Decimal`, `high: Decimal`, `low: Decimal`, `close: Decimal`
- `volume: int` — cumulative day volume at latest tick
- `bar_start_volume: int` — cumulative at bar start
- `vwap_num: Decimal` — price×volume numerator
- `num_ticks: int`, `is_first: bool`

**Class TimeBarBuilder:**
- `__init__(self, interval_minutes: int = 15) -> None`
- `feed_tick(self, tick: Tick) -> Optional[Bar]` — accumulates tick, returns completed Bar when slot advances
- `_emit_bar(self, current: _CurrentBar) -> Bar` — converts _CurrentBar to Bar, computes VWAP
- `flush_all(self) -> list[Bar]` — emits all open bars at session end
- `reset(self) -> None`

**Class QuestDBStore:**
- Class attribute `_FLUSH_INTERVAL_ROWS: int = 50`
- `__init__(self) -> None` — ILP sender + psycopg2 state
- `connect(self) -> None` — retries with exponential backoff [1,2,4,8,16,32]
- `close(self) -> None` — flushes, closes connections
- `create_tables(self) -> None` — executes DDL for all 4 tables
- `write_bar(self, bar: Bar) -> None` — buffers bar, auto-flushes
- `write_order_event(self, ts, client_order_id, exchange_order_id, symbol, side, quantity, price, order_type, state, strategy_id, reject_reason="") -> None`
- `write_fill(self, fill: Fill, cost_inr: Decimal, slippage_bps: Decimal) -> None`
- `write_risk_event(self, event_type: str, details: str, daily_pnl: Decimal, peak_pnl: Decimal, kill_switch_active: bool) -> None`
- `flush(self) -> None` — sends buffered rows via ILP, replays dead-letter on failure
- `_auto_flush(self) -> None` — flushes if buffer >= threshold
- `get_bars(self, symbol: str, start: datetime, end: datetime, bar_type: str = "time") -> pl.DataFrame`
- `get_last_order_states(self, date_str: str) -> pl.DataFrame` — LATEST ON ts PARTITION BY client_order_id
- `get_daily_pnl(self, date_str: str) -> Decimal` — SUM(-cost_inr) from fills

---

### 3.4 algo_trading/data/ring_buffer.py

**Constants:**
- `TICK_DTYPE: np.dtype` — structured array: ltp(float32), volume(int32), bid(float32), ask(float32), timestamp(int64)

**Class RingBuffer:**
- `__init__(self, capacity: int = 10_000) -> None` — validates capacity >= 1; initializes numpy structured array, head, count, lock
- `push(self, ltp: float, volume: int, bid: float, ask: float, timestamp_ns: int) -> None` — lock only for (_head, _count) update; non-blocking write
- Property `count -> int`, `capacity -> int`
- `last(self, k: int) -> np.ndarray` — last k ticks (copy, handles wrap-around)
- `last_ltp_array(self, k: int) -> np.ndarray` — last k prices as float32
- `last_as_polars(self, k: int) -> pl.DataFrame` — last k ticks as Polars DataFrame
- `is_ready(self, min_count: int) -> bool` — count >= min_count (warm-up guard)
- `reset(self) -> None`

---

### 3.5 algo_trading/data/system_state.py

**Dataclass ReconcileResult:**
- `is_clean: bool`, `broker_positions: list[Position]`, `db_positions: list[Position]`
- `mismatches: list[str]`, `stale_positions: list[str]`, `extra_broker_positions: list[str]`

**Dataclass LiveSystemState:**
- `_lock: threading.Lock`, `_state_file: Path`
- `positions: dict[str,Position]`, `active_orders: dict[str,str]`
- `daily_pnl: Decimal`, `peak_pnl: Decimal`
- `session_date: Optional[date]`, `kill_switch_active: bool`, `startup_reconcile_clean: bool`
- `update_position_from_fill(self, symbol, side, quantity, fill_price, trade_cost) -> None` — updates position, computes P&L on close, saves to disk
- `get_position(self, symbol: str) -> Optional[Position]`
- `get_all_positions(self) -> list[Position]` — excludes flat positions
- `gross_exposure(self, current_prices: dict[str,Decimal]) -> Decimal` — Σ|qty|×price
- `record_order(self, client_order_id: str, state: str) -> None`
- `remove_order(self, client_order_id: str) -> None`
- `set_kill_switch(self, active: bool) -> None`
- `reset_for_new_day(self, session_date: date) -> None`
- `snapshot(self) -> dict` — thread-safe snapshot for logging
- `save(self) -> None` — atomic JSON persist (temp file + rename)
- `@classmethod load_from_disk(cls, path: Path) -> Optional[LiveSystemState]`

**Class StartupReconciler:**
- `__init__(self, broker: BrokerBase, state: LiveSystemState) -> None`
- `reconcile(self) -> ReconcileResult` — compares broker vs DB positions; writes sentinel file on mismatch to block trading

---

### 3.6 algo_trading/data/integrity_guard.py

**Constants:**
- `_DEFAULT_MAX_PRICE_JUMP_PCT: Decimal = Decimal("3.0")`
- `_DEFAULT_BAD_TICK_WINDOW: int = 100`
- `_DEFAULT_MAX_BAD_TICK_RATE: float = 0.10`

**Dataclass ValidationResult (frozen=True):**
- `is_valid: bool`, `reason: str = ""`
- `@classmethod ok(cls) -> ValidationResult`
- `@classmethod reject(cls, reason: str) -> ValidationResult`

**Dataclass _SymbolState:**
- `last_ltp: Optional[Decimal]`, `last_timestamp: Optional[datetime]`, `tick_count: int = 0`

**Class TickValidator:**
- `__init__(self, max_price_jump_pct: Decimal = 3.0, require_nonzero_volume: bool = False) -> None`
- `validate(self, tick: Tick) -> ValidationResult` — rejects: LTP ≤ 0, zero volume (if required), duplicate timestamp (rolling 20-tick deque [Fix F]), price jump > 3%, out-of-order timestamp
- `reset_symbol(self, symbol: str) -> None`
- `reset_all(self) -> None`
- `get_tick_count(self, symbol: str) -> int`

**Class DataQualityMonitor:**
- `__init__(self, window_size: int = 100, max_bad_rate: float = 0.10, halt_callback: Optional[callable] = None) -> None`
- `record(self, is_valid: bool, symbol: str = "GLOBAL") -> None` — per-symbol rolling window [Fix E]
- `should_halt_trading(self, symbol: str = "GLOBAL") -> bool`
- Property `is_halted -> bool` — any symbol halted
- `current_bad_rate(self, symbol: str = "GLOBAL") -> float`
- `reset(self, symbol: Optional[str] = None) -> None`

---

### 3.7 algo_trading/data/websocket_feed.py

**Functions:**
- `_encode(obj: dict) -> bytes` — msgpack or JSON
- `_decode(data: bytes) -> dict`
- `_tick_to_dict(tick: Tick) -> dict` — type="tick", ts_ns, symbol, ltp, volume, bid, ask, oi, exchange, token
- `_bar_to_dict(bar: Bar) -> dict` — type="bar", ts_ns, symbol, ohlcv, vwap, num_ticks, bar_type, interval_min, is_first_bar
- `_dict_to_tick(d: dict) -> Tick`
- `_dict_to_bar(d: dict) -> Bar`

**Class TickPublisher:**
- `__init__(self, port: Optional[int] = None) -> None`
- `start(self) -> None` — PUB socket, LINGER=0, SNDHWM=10000; retries alternate ports (base+1,+2,+3) [Fix C]
- `publish_tick(self, tick: Tick) -> None` — NOBLOCK; tracks/logs drops per-symbol [Fix D]
- `stop(self) -> None`

**Class BarPublisher:** (analogous to TickPublisher)
- `start()`, `publish_bar(bar: Bar) -> None`, `stop()`

**Class ZMQTickSubscriber:**
- `start(self) -> None` — SUB socket, LINGER=0, RCVHWM=10000, RCVTIMEO, subscribes all
- `recv_tick(self) -> Optional[Tick]` — None on timeout
- `stop(self) -> None`

**Class ZMQBarSubscriber:** (analogous)
- `start()`, `recv_bar() -> Optional[Bar]`, `stop()`

---

### 3.8 algo_trading/data/fo_chain.py

**Constants:** `STAGE: str = "4"`

**Dataclass OptionContract:**
- `symbol: str`, `expiry: date`, `strike: float`, `option_type: str` ("CE"|"PE")
- `oi: int`, `change_in_oi: int`, `volume: int`, `iv: float`, `ltp: float`
- `delta: float = 0.0`, `gamma: float = 0.0`, `theta: float = 0.0`, `vega: float = 0.0`
- `bid: float = 0.0`, `ask: float = 0.0`

**Dataclass OptionChain:**
- `symbol: str`, `underlying_price: float`, `timestamp: datetime`, `expiry: date`, `contracts: list[OptionContract]`
- Property `calls -> list[OptionContract]`, `puts -> list[OptionContract]`
- Property `atm_strike -> float` — closest strike to underlying_price

**Class BlackScholesGreeks (all static methods):**
- `_d1d2(S, K, T, r, sigma) -> tuple[float,float]`
- `call_price(S, K, T, r, sigma) -> float`
- `put_price(S, K, T, r, sigma) -> float`
- `delta(S, K, T, r, sigma, option_type) -> float` — N(d1) for call, N(d1)-1 for put
- `gamma(S, K, T, r, sigma) -> float` — N'(d1)/(S×σ×√T)
- `theta(S, K, T, r, sigma, option_type) -> float` — per day (annual/365)
- `vega(S, K, T, r, sigma) -> float` — per 1% IV move (annual/100)
- `implied_volatility(market_price, S, K, T, r, option_type, max_iter=100, tol=1e-6) -> Optional[float]` — Newton-Raphson solver; returns None if intrinsic or vega too flat
- `compute_all_greeks(contract, S, T, r=0.065) -> OptionContract` — computes IV then Greeks; falls back to IV=0.30

**Class FOChainLoader:**
- `load_from_parquet(self, path, symbol, expiry=None) -> OptionChain`
- `load_synthetic(self, symbol, underlying_price, n_strikes=10, expiry_days=7) -> OptionChain` — GBM synthetic; 50-point intervals; BS prices + Greeks

---

### 3.9 algo_trading/data/tick_aggregator.py

**Constants:**
- `VALID_INTERVALS: tuple[int] = (1, 5, 10, 30, 60)`
- `NS_PER_SECOND: int = 1_000_000_000`

**Dataclass TickBar:**
- `timestamp_ns: int`, `open: float`, `high: float`, `low: float`, `close: float`
- `volume: int`, `n_trades: int`, `buy_volume: int`, `sell_volume: int`, `vwap: float`
- `to_dict() -> dict`

**Dataclass _BarAccumulator:**
- `bar_start_ns, first_price, high, low, last_price, volume, n_trades, buy_volume, sell_volume, _pq_sum, _qty_sum`
- `@classmethod from_first_tick(bar_start_ns, price, quantity, side) -> _BarAccumulator`
- `_absorb(price, quantity, side) -> None`, `add(price, quantity, side) -> None`
- Property `vwap -> float`
- `to_bar() -> TickBar`

**Class TickAggregator:**
- `__init__(interval_seconds: int = 5) -> None`
- `add_tick(timestamp_ns, price, quantity, side) -> Optional[TickBar]` — returns completed bar or None
- `get_current_incomplete_bar() -> Optional[TickBar]`
- `reset() -> None`
- `_floor_to_interval(timestamp_ns: int) -> int`

**Class TickBarStream:**
- `__init__(interval_seconds: int = 5, symbol: str = "") -> None`
- `@staticmethod from_zmq_message(msg: bytes) -> dict`
- `process_message(msg: bytes) -> Optional[TickBar]`
- `publish_bar(bar: TickBar, zmq_pub_socket) -> None`
- `get_current_incomplete_bar() -> Optional[TickBar]`
- `reset() -> None`

---

### 3.10 algo_trading/data/truedata_feed.py

**Constants:**
- `_RECONNECT_BASE_SEC: float = 2.0`, `_RECONNECT_MAX_SEC: float = 300.0`, `_RECONNECT_MULTIPLIER: float = 2.0`
- `_ZMQ_SNDHWM: int = 50_000`, `_TRUEDATA_PORT: int = 8082`, `_BACKFILL_GAP_THRESHOLD_SEC: int = 60`

**Dataclass TrueDataTick:**
- `symbol, ltp, bid, ask, bid_qty, ask_qty, volume, timestamp_ns, seq`

**Functions:**
- `_tick_to_zmq_dict(tick: TrueDataTick) -> dict`
- `_encode(obj: dict) -> bytes`

**Class TrueDataFeed:**
- `__init__(user, password, symbols, zmq_pub_port=5555, option_chain_callback=None)`
- `start() -> None` — connects TrueData, subscribes, publishes via ZMQ
- `stop() -> None`
- `reconnect() -> None` — exponential backoff
- `start_option_chain(symbol, expiry=None, *, greek=True) -> None`
- `stop_option_chain(symbol) -> None`
- Private: `_bind_zmq()`, `_connect_truedata()`, `_on_tick(tick_data)`, `_on_option_chain(chain_data)`, `_publish(tick)`, `_backfill_ticks_if_stale()` — REST gap backfill on >60s stale

**Class TrueDataHistorical:**
- `get_bars(symbol, from_date, to_date, bar_size=15) -> pl.DataFrame`
- `get_ticks(symbol, from_date, to_date) -> pl.DataFrame`

---

*END OF PART 1*
# NSE AlgoTrader — Codebase Reference (Part 2)

---

## 4. FEATURES

### 4.1 algo_trading/features/engine.py

**Constants:**
- `FEATURE_NAMES: list[str]` (25 features): rsi_14, macd_hist, bb_pct_b, bb_width, atr_14, adx_14, stoch_k, stoch_d, obv_normalized, supertrend_dir, ema_cross_ratio, ichimoku_position, vwap_dev_bps, kalman_price, hmm_regime, garch_vol, obi, price_vol_divergence, ha_direction, pivot_dist_bps, autocorr_lag1, relative_strength, tod_sin, tod_cos, spread_percentile
- `N_FEATURES: int = 25`
- `_ZMQ_RCVHWM: int`, `_ZMQ_RCVTIMEO_MS: int = 5_000`, `_ZMQ_SNDHWM: int`, `_ZMQ_LINGER: int = 0`
- `_MAX_ALLOWED_GAP: int = 3`, `_MIN_TICKS_FOR_FEATURES: int`, `_RING_BUFFER_CAPACITY: int`

**Functions:**
- `compute_feature_schema_hash(feature_names: list[str]) -> str` — SHA-256 of feature names list
- `FEATURE_SCHEMA_HASH: str` — pre-computed hash at import time

**Dataclass _VolumeBar:**
- `symbol, open_, high, low, close, volume, turnover, tick_count, start_ts, end_ts, bar_type, vwap`

**Class VolumeBarAggregator:**
- `__init__(symbol, volume_threshold=50_000, tick_threshold=500, dollar_threshold=2_00_00_000.0, on_bar=None)`
- `push_tick(ltp, volume, timestamp_ns) -> Optional[_VolumeBar]` — feeds tick, returns bar when threshold crossed
- `calibrate_from_history(daily_volumes: list[int], target_bars: int = 200) -> None`
- Property `bar_seq -> int`

**Class SequenceGuard (SURV-7):**
- `__init__()` — initializes tracking state
- `check(symbol: str, seq: int) -> bool` — validates sequence, returns True if should process
- `is_halted(symbol: str) -> bool`
- `reset_symbol(symbol: str) -> None`

**Class FeatureComputer:**
- `__init__(symbols, sub_port=5555, pub_port=5556, min_bars_for_features=60)`
- `start() -> None` — main event loop, blocks until shutdown
- `shutdown() -> None`
- Private: `_setup_sockets()`, `_setup_bar_aggregators()`, `_handle_raw_message(parts)`, `_process_tick(symbol, tick_data)`, `_on_bar_complete(bar)`, `_compute_features(symbol, bar) -> Optional[np.ndarray]`, `_publish_features(symbol, features, timestamp_ns, bar_seq)`, `_teardown()`

**Function:**
- `run_feature_computer(symbols, sub_port=5555, pub_port=5556) -> None` — entry for multiprocessing

---

### 4.2 algo_trading/features/technical.py

**Constants:**
- `_SESSION_START_HOUR: int = 9`, `_SESSION_START_MINUTE: int = 15`
- `_RSI_PERIOD: int = 14`, `_MACD_FAST: int = 12`, `_MACD_SLOW: int = 26`, `_MACD_SIGNAL: int = 9`
- `_BB_PERIOD: int = 20`, `_BB_STD: float = 2.0`
- `_ATR_PERIOD: int = 14`, `_ADX_PERIOD: int = 14`
- `_STOCH_K: int = 14`, `_STOCH_D: int = 3`, `_STOCH_SMOOTH_K: int = 3`
- `_SUPERTREND_PERIOD: int = 10`, `_SUPERTREND_MULT: float = 3.0`
- `_EMA_FAST: int = 9`, `_EMA_SLOW: int = 21`
- `_ICHIMOKU_TENKAN: int = 9`, `_ICHIMOKU_KIJUN: int = 26`, `_ICHIMOKU_SENKOU_B: int = 52`

**Class TechnicalFeatures:**
- `compute(df: pl.DataFrame) -> pl.DataFrame` — adds all 15 technical feature columns to DataFrame
- `compute_single_bar(recent_bars: pl.DataFrame) -> dict[str,float]` — last bar only

**Helper functions (private):**
- `_validate_ohlcv(df) -> None`, `_to_float32_series(series) -> pd.Series`
- `_add_rsi(pdf) -> pd.DataFrame` — RSI(14)
- `_add_macd_hist(pdf) -> pd.DataFrame` — MACD histogram
- `_add_bollinger(pdf) -> pd.DataFrame` — BB %B and width
- `_add_atr(pdf) -> pd.DataFrame` — ATR(14)
- `_add_adx(pdf) -> pd.DataFrame` — ADX(14)
- `_add_stochastic(pdf) -> pd.DataFrame` — Stoch %K and %D
- `_add_obv(pdf) -> pd.DataFrame` — OBV normalized
- `_add_supertrend(pdf) -> pd.DataFrame` — Supertrend direction
- `_add_ema_cross(pdf) -> pd.DataFrame` — EMA9/EMA21 ratio
- `_add_ichimoku(pdf) -> pd.DataFrame` — Ichimoku cloud position
- `_add_vwap_deviation(pdf) -> pd.DataFrame` — VWAP deviation in basis points
- `_add_heikin_ashi_dir(pdf) -> pd.DataFrame` — HA direction
- `_add_pivot_dist(pdf) -> pd.DataFrame` — pivot distance in basis points

---

### 4.3 algo_trading/features/statistical.py

**Exceptions:**
- `HMMFitError` — raised when HMM cannot be fitted

**Constants:**
- `_AUTOCORR_LAGS: int = 20`, `_HMM_N_STATES: int = 3`, `_HMM_MIN_DURATION: int = 5`, `_HMM_N_ITER: int = 200`
- `_GARCH_ANNUALISE_FACTOR: float = 252.0 * 25.0`

**Class OFIKalmanFilter:**
- `__init__(process_noise=0.01, obs_noise=1.0, ofi_beta=0.05) -> None`
- `update(price: float, ofi: float) -> float` — 2-state constant-velocity Kalman update; returns smoothed price
- Property `state -> tuple[float,float]`

**Class HMMRegimeDetector:**
- `__init__(n_states=3, min_duration_bars=5) -> None`
- `fit(returns: np.ndarray) -> None` — Gaussian HMM on log returns
- `predict_regime(recent_returns: np.ndarray) -> tuple[int, np.ndarray]` — with min-duration smoothing
- `regime_to_label(regime_idx: int, means: np.ndarray) -> str` — "bull"|"bear"|"sideways"
- Property `state_means -> Optional[np.ndarray]`

**Class GARCHVolForecaster:**
- `__init__() -> None`
- `fit(log_returns: np.ndarray) -> None` — fits GARCH(1,1)
- `forecast_one_step() -> float` — annualized 1-step-ahead vol
- `refit_at_open(gap_adjusted_returns: np.ndarray) -> None`

**Class StatisticalFeatures:**
- `__init__()` — instantiates Kalman, HMM, GARCH
- `compute(df: pl.DataFrame) -> pl.DataFrame` — adds statistical columns
- `compute_single_bar(recent_bars, current_ofi) -> dict[str,float]`

**Private functions:**
- `_lag1_autocorr(log_returns, window=20) -> np.ndarray` — rolling lag-1 autocorrelation
- `_validate_close(df) -> None`

---

### 4.4 algo_trading/features/microstructure.py (Stage 2B+)

**Constants:**
- `_VPIN_WINDOW: int = 50`, `_VPIN_DEFAULT_BUCKETS: int = 50`
- `_VPIN_ELEVATED_THRESHOLD: float = 0.35`, `_VPIN_TOXIC_THRESHOLD: float = 0.70`
- `_KYLE_LAMBDA_WINDOW: int = 20`, `_KYLE_MIN_BARS: int = 5`
- `_SESSION_TOTAL_MINUTES: int = 375`, `_GAMMA_DEFAULT: float = 0.1`

**Class MicrostructureFeatures:**
- `__init__(adv: float, stage: str = "2A") -> None`
- `compute_vpin(volume_bars: pl.DataFrame, bucket_size=None) -> float` — VPIN using BVC (Bulk Volume Classification)
- `get_vpin_signal(vpin: float) -> tuple[float,str]` — (size_multiplier, label)
- `compute_reservation_price(mid, inventory, sigma_daily, minutes_remaining, gamma=0.1) -> float` — Avellaneda-Stoikov
- `compute_kyle_lambda(bars: pl.DataFrame) -> float`
- `compute_all(bars, inventory, minutes_remaining) -> dict[str,float]`
- `_require_stage_2b(feature: str)` — raises if stage < 2B

**Stage 3+ stubs (raise NotImplementedError):**
- `compute_hawkes_branching_ratio(*args, **kwargs) -> float`
- `compute_trade_flow_imbalance(*args, **kwargs) -> float`

---

### 4.5 algo_trading/features/custom.py

**Constants:**
- `_OBI_ALPHA: float = 0.01`, `_DIVERGENCE_WINDOW: int = 20`
- `_RS_WINDOW: int = 20`, `_SPREAD_WINDOW: int = 20`
- `_NSE_MINUTES_PER_DAY: int = 375`, `_VOL_PROFILE_PATH: str`

**Numba kernel:**
- `_compute_obi_numba(bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price, alpha) -> float` — exponentially-weighted Order Book Imbalance

**Functions:**
- `_load_volume_profile(path) -> Optional[np.ndarray]`
- `_validate_ohlcv(df) -> None`

**Class CustomFeatures:**
- `__init__(bar_type: str = "volume") -> None`
- `compute_obi(bid_prices, bid_volumes, ask_prices, ask_volumes, mid) -> float`
- `compute_price_vol_divergence(closes, volumes) -> float` — correlation of |price_change| with volume
- `compute_relative_strength(symbol_closes, nifty_closes) -> float` — z-score of excess return
- `encode_time_of_day(hour, minute, cumulative_volume=None, total_day_volume=None) -> tuple[float,float]` — (sin, cos) encoding
- `compute_single_bar(recent_bars: pl.DataFrame) -> dict[str,float]`
- `compute_spread_percentile(current_spread, recent_spreads) -> float` — rolling percentile rank

**Function:** `warmup_numba() -> None` — pre-compiles numba JIT kernels

---

### 4.6 algo_trading/features/hawkes.py (Stage 3+)

**Constants:** `STAGE: str = "3+"`, `MIN_TIMEFRAME_SECONDS: int = 60`

**Dataclass HawkesParams:**
- `mu: float`, `alpha: float`, `beta: float`
- Property `branching_ratio -> float` — alpha/beta
- Property `is_stationary -> bool` — branching_ratio < 1.0

**Enum HawkesSignal:** NORMAL, ELEVATED, BURST

**Class HawkesProcess:**
- `__init__(params: Optional[HawkesParams] = None) -> None`
- `fit(timestamps: np.ndarray) -> HawkesParams` — MLE via L-BFGS-B with O(n) log-likelihood
- `log_likelihood(timestamps, params) -> float`
- `current_intensity(timestamps, eval_time) -> float` — lambda(eval_time)
- `intensity_ratio(timestamps, eval_time) -> float` — lambda(eval_time)/mu

**Class HawkesFeatureEngine:**
- `__init__(window_seconds=300, refit_every_n_events=500) -> None`
- `update(timestamp_ns, price, qty) -> dict` — ingests tick, returns feature dict with hawkes_ratio and hawkes_signal_enc
- `_refit() -> None`, `_default_features() -> dict`

**Function:** `get_hawkes_signal(intensity_ratio: float) -> HawkesSignal`

---

### 4.7 algo_trading/features/tfi.py (Stage 3+)

**Constants:** `STAGE: str = "3+"`

**Dataclasses:**
- `OrderBookSnapshot`: timestamp_ns, symbol, bids: list[tuple[float,int]], asks: list[tuple[float,int]]
- `OFILevel`: level, bid_delta, ask_delta, ofi
- `TFIResult`: timestamp_ns, tfi, raw_ofi, total_flow, n_levels_used, buy_pressure, signal (str)

**Function:** `_classify_signal(tfi: float) -> str` — strong_buy/buy/neutral/sell/strong_sell

**Class TFICalculator:**
- `__init__(n_levels=5, decay_weights=True) -> None`
- `compute_ofi(prev: OrderBookSnapshot, curr: OrderBookSnapshot) -> list[OFILevel]`
- `compute_tfi(curr_snapshot) -> Optional[TFIResult]` — 1/k decay weights on OFI
- `update(snapshot) -> Optional[TFIResult]` — alias for compute_tfi

**Class TFIRollingWindow:**
- `__init__(window_size=20) -> None`
- `add(result: TFIResult) -> dict` — returns {tfi_mean, tfi_std, tfi_zscore, tfi_trend, current_signal}

---

### 4.8 algo_trading/features/fo_features.py (Stage 4)

**Constants:** `STAGE: str = "4"`, `_VALID_OI_BUILDUP: frozenset`, `_VALID_PCR_SIGNAL: frozenset`

**Dataclass FOFeaturesResult:**
- `timestamp, symbol, pcr_oi, pcr_volume, max_pain, iv_rank, iv_percentile, iv_skew, atm_iv`
- `oi_buildup_signal, call_wall, put_wall, price_vs_max_pain_pct, pcr_signal`

**Class FOFeatureEngine:**
- `__init__(iv_history_window=252) -> None`
- `compute(chain: OptionChain, price_change_pct=0.0, oi_change_pct=0.0) -> FOFeaturesResult`
- Private: `_compute_atm_iv()`, `_compute_pcr()`, `_compute_max_pain()`, `_compute_iv_rank()`, `_compute_iv_skew()`, `_compute_oi_buildup()`, `_compute_walls()`

---

### 4.9 algo_trading/features/stage_router.py

**Constants:**
- `_HAWKES_SIGNAL_ENC: dict[str,float]` — {normal:0.0, elevated:0.5, burst:1.0}
- `_OI_BUILDUP_ENC: dict[str,float]` — encoding for OI buildup signals
- `_BASE_NAMES: list[str]` — 25 base feature names
- `_STAGE_2B_EXTRA, _STAGE_3_EXTRA, _STAGE_4_EXTRA` — additional features per stage
- `STAGE_FEATURE_NAMES: dict[str,list[str]]` — features by stage
- `_VALID_STAGES: frozenset[str]`

**Dataclass StageConfig:**
- `stage: str`, `data_feed: str`
- `enable_hawkes: bool = False`, `enable_tfi: bool = False`
- `enable_fo_features: bool = False`, `enable_sentiment: bool = False`, `enable_microstructure: bool = False`
- `@classmethod from_env() -> StageConfig` — reads ALGO_STAGE env var

**Class FeatureRouter:**
- `__init__(config: StageConfig) -> None`
- `compute(bar_df, tick_timestamps_ns, ob_snapshot, fo_chain, symbol, inventory, minutes_remaining, price_change_pct, oi_change_pct) -> dict[str,float]`
- `get_feature_names() -> list[str]`
- `get_feature_vector(feature_dict: dict[str,float]) -> np.ndarray` — float32 ndarray

**Helpers:** `_is_finite(value) -> bool`, `_safe_float(value) -> float`

---

## 5. SIGNALS

### 5.1 algo_trading/signals/ema_cross.py

**Dataclass EMACrossState:**
- `ema_fast: Optional[float]`, `ema_slow: Optional[float]`
- `prev_ema_fast: Optional[float]`, `prev_ema_slow: Optional[float]`
- `bar_count: int`, `closes: deque`, `highs: deque`, `lows: deque`
- `atr: Optional[float]`, `position: int`

**Class EMACrossStrategy:**
- `__init__() -> None` — reads settings for EMA periods
- `feed_bar(bar: Bar) -> Optional[TradeSignal]` — EMA9/EMA21 cross + ATR stops
- `reset_symbol(symbol: str)`, `reset_all()`
- `get_state(symbol: str) -> Optional[EMACrossState]`
- `is_warmed_up(symbol: str) -> bool`
- `_compute_atr(state) -> float`, `_compute_confidence(ema_fast, ema_slow, atr) -> float`

---

### 5.2 algo_trading/signals/ema_cross_filtered.py

**Constants:**
- `_ADX_PERIOD: int = 14`, `_VOLUME_LOOKBACK: int = 20`, `_RS_LOOKBACK: int = 5`

**Dataclasses:**
- `ADXState`: bar_count, prev_high, prev_low, prev_close, smooth_atr, smooth_plus_dm, smooth_minus_dm, adx, plus_di, minus_di, seed arrays, seeded, adx_seeded
- `FilterStats`: total_signals, adx_blocked, volume_blocked, rs_blocked, passed_all

**Class ADXComputer:**
- `__init__(period=14) -> None`
- `update(bar: Bar, state: ADXState) -> None` — Wilder's smoothing
- `_compute_di_dx_adx(state) -> None`

**Class VolumeFilter:**
- `__init__(lookback=20, multiplier=1.5) -> None`
- `update_and_check(bar: Bar) -> bool` — True if volume > 1.5x 20-bar average
- `reset_symbol(symbol) -> None`

**Class RelativeStrengthFilter:**
- `__init__(lookback=5, nifty_symbol="NIFTY50-INDEX") -> None`
- `update(bar: Bar) -> None`
- `check(symbol: str, direction: int) -> bool`
- `reset_symbol(symbol) -> None`

**Class FilteredEMACrossStrategy:**
- `__init__(use_adx=True, use_volume=True, use_rs=True, adx_threshold=25.0, volume_multiplier=1.5, nifty_symbol="NIFTY50-INDEX")`
- `feed_bar(bar: Bar, nifty_bar: Optional[Bar] = None) -> Optional[TradeSignal]`
- `get_filter_stats() -> dict[str,dict]` — per-symbol filter hit rates
- `reset_symbol(symbol)`, `reset_all()`
- `get_adx(symbol) -> Optional[float]`
- `is_warmed_up(symbol) -> bool`

---

### 5.3 algo_trading/signals/event_filter.py

**Constants:**
- `_THURSDAY: int = 3`
- `_VIX_HIGH_THRESHOLD: float = 22.0`, `_VIX_VERY_HIGH_THRESHOLD: float = 30.0`
- `_MULTIPLIER_MONTHLY_EXPIRY: float = 0.50`, `_MULTIPLIER_WEEKLY_EXPIRY: float = 0.70`
- `_BLACKOUT_MULTIPLIER: float = 0.0`, `_RESULTS_WINDOW_DAYS: int = 1`

**Dataclass BlackoutEntry:** `symbol: str`, `date: date`, `reason: str`

**Functions:**
- `is_weekly_expiry_day(d: date) -> bool` — True if Thursday
- `is_nse_expiry_day(d: date) -> bool` — True if last Thursday of month

**Class EventCalendarFilter:**
- `__init__() -> None`
- `get_size_multiplier(symbol: str, trade_date: date, india_vix: Optional[float] = None) -> float` — returns [0.0, 1.0]
- `add_blackout(symbol, date_str, reason="")`, `add_blackout_exact(symbol, date_str, reason="")`
- `load_blackout_csv(path: str | Path) -> int`
- `get_event_description(trade_date: date, symbol: str = "") -> str`
- `remove_blackout(symbol, date_str)`, `list_blackouts(symbol=None) -> list[BlackoutEntry]`, `clear_blackouts()`

---

### 5.4 algo_trading/signals/event_calendar.py

Re-export shim for `algo_trading.signals.event_filter`:
- Re-exports: `EventCalendarFilter`, `BlackoutEntry`, `is_nse_expiry_day`, `is_weekly_expiry_day`

---

### 5.5 algo_trading/signals/mean_reversion.py

**Constants:**
- `_RSI_PERIOD: int = 14`, `_ADX_PERIOD: int = 14`
- `_RSI_OVERBOUGHT: float = 72.0`, `_RSI_OVERSOLD: float = 28.0`
- `_ADX_MAX: float = 20.0`, `_ATR_MULTIPLIER: float = 1.0`, `_WARMUP_BARS: int = 28`

**Dataclasses:**
- `RSIState`: bar_count, prev_close, seed_gains, seed_losses, seeded, avg_gain, avg_loss, rsi
- `MRSignalStats`: total_checks, adx_blocked, rsi_no_extreme, signals_emitted

**Class RSIComputer:**
- `__init__(period=14) -> None`
- `update(bar, state: RSIState) -> None` — Wilder's smoothing

**Class MeanReversionStrategy:**
- `__init__(rsi_overbought=72.0, rsi_oversold=28.0, adx_max=20.0, atr_multiplier=1.0) -> None`
- `feed_bar(bar: Bar) -> Optional[TradeSignal]` — RSI extreme in ADX<20 sideways regime
- `is_warmed_up(symbol) -> bool`
- `get_rsi(symbol) -> Optional[float]`, `get_adx(symbol) -> Optional[float]`
- `get_stats() -> dict[str,dict]`
- `reset_symbol(symbol)`, `reset_all()`

---

### 5.6 algo_trading/signals/multi_timeframe.py

**Constants:**
- `_BARS_PER_HOUR: int = 4`, `_EMA_FAST_PERIOD: int = 9`, `_EMA_SLOW_PERIOD: int = 21`
- `_SIDEWAYS_BAND_BPS: float = 5.0`
- `_COUNTER_TREND_MULTIPLIER: float = 0.0`, `_SIDEWAYS_MULTIPLIER: float = 0.5`

**Enum HTFTrend:** BULL, BEAR, SIDEWAYS

**Dataclasses:**
- `HourlyBarAccumulator`: open, high, low, close, volume, count
- `HTFState`: accumulator, ema_fast, ema_slow, hourly_bar_count, trend

**Class MultiTimeframeConfirmation:**
- `__init__(bars_per_hour=4, ema_fast_period=9, ema_slow_period=21, sideways_band_bps=5.0, counter_trend_multiplier=0.0, sideways_multiplier=0.5)`
- `update(bar: Bar) -> Optional[HTFTrend]` — 4x15min->hourly resampling, EMA9/21 trend filter
- `get_size_multiplier(symbol, signal_direction) -> float` — 0.0 (counter-trend), 0.5 (sideways), 1.0 (aligned)
- `get_trend(symbol) -> Optional[HTFTrend]`
- `reset_symbol(symbol)`, `reset_all()`
- `is_warmed_up(symbol) -> bool`

---

### 5.7 algo_trading/signals/opening_range_breakout.py

**Constants:**
- `_ORB_BARS: int = 2`, `_ORB_BUILD_COMPLETE_TIME: time = time(9, 44)`
- `_ORB_CUTOFF_TIME: time = time(12, 0)`, `_NO_NEW_TRADES_TIME: time = time(15, 15)`
- `_VOLUME_MULTIPLIER: float = 1.5`, `_MIN_RANGE_FRACTION: float = 0.002`

**Dataclass ORBState:**
- `opening_high, opening_low, bars_in_opening, range_ready, signal_fired`
- `volume_history: deque`, `closes: deque`, `highs: deque`, `lows: deque`
- `atr: Optional[float]`, `total_bars: int`

**Class OpeningRangeBreakoutStrategy:**
- `__init__(orb_bars=2, cutoff_time=time(12,0), atr_period=14, atr_stop_multiplier=1.5, volume_multiplier=1.5, min_range_fraction=0.002)`
- `feed_bar(bar: Bar) -> Optional[TradeSignal]` — first 2 bars define range; breakout with volume confirm; 12:00 cutoff
- `reset_symbol(symbol)`, `reset_all()`
- `get_state(symbol) -> Optional[ORBState]`
- `is_range_ready(symbol) -> bool`
- `_compute_atr(state) -> float`

---

## 6. LABELING

### 6.1 algo_trading/labeling/triple_barrier.py

**Constants:**
- `REQUIRED_INPUT_COLUMNS: Final[list[str]] = ["timestamp","open","high","low","close","volume","atr14"]`
- `IMBALANCE_WARN_LOW: Final[float] = 0.15`
- `IMBALANCE_WARN_HIGH: Final[float] = 0.55`

**Dataclass BarrierSpec:**
- `upper_atr_mult: float = 1.5`, `lower_atr_mult: float = 1.5`
- `max_holding_bars: int = 20`, `min_atr_floor: float = 0.0005`
- `__post_init__` — validates symmetric barriers and parameter ranges

**Class TripleBarrierLabeler:**
- `__init__(spec: BarrierSpec = BarrierSpec()) -> None`
- `label(df: pl.DataFrame) -> pl.DataFrame` — returns df with: label(-1/0/+1), upper_barrier, lower_barrier, time_barrier, sample_weight, barrier_hit columns
- `compute_sample_weights(labels: pl.Series) -> pl.Series` — inverse-frequency weights normalized to mean=1.0
- `compute_uniqueness_weights(events_df: pl.DataFrame) -> pl.Series` — AFML Ch.4 uniqueness-weighted sample weights via label span concurrency
- `compute_label_distribution(labels: pl.Series) -> dict[int,float]` — per-class proportions; warns if outside [15%,55%]
- `validate_no_lookahead(df: pl.DataFrame) -> bool` — verifies barriers don't use future data
- `_validate_input(df: pl.DataFrame) -> pl.DataFrame` — raises ValueError on missing cols or empty; drops null-ATR rows

---

### 6.2 algo_trading/labeling/walk_forward.py

**Dataclasses:**
- `SplitMetadata`: fold_index, train_start, train_end, test_start, test_end, purge_bars, embargo_bars, bar_type, train_size, test_size
- `WalkForwardConfig`: n_folds=5, max_holding_bars=20, max_feature_lookback_bars=52 (Ichimoku), embargo_bars=5, min_test_trades=200, bar_type="time_15min"; `__post_init__` validates n_folds>=5, embargo_bars>=1

**Class WalkForwardSplitter:**
- `__init__(config: WalkForwardConfig = WalkForwardConfig()) -> None`
- Property `purge_bars -> int` — max(computed, 51); computed = max_holding + max_lookback + embargo
- `split(df: pl.DataFrame) -> list[tuple[pl.DataFrame, pl.DataFrame, SplitMetadata]]` — n_folds walk-forward with fixed window
- `validate_no_future_leakage(splits) -> bool` — max(train.timestamp) < min(test.timestamp)
- `get_expanding_window_splits(df) -> list[...]` — train_start always 0, tagged "_expanding"
- `_validate_dataset_size(n, purge) -> None` — raises if insufficient data
- `_check_min_test_trades(test_df, fold_index) -> None`

---

END OF PART 2
## 7. EXECUTION

### 7.1 algo_trading/execution/order_executor.py

**Dataclass TradeSignal (frozen=False, slots=True):**
- `symbol: str`, `direction: int` (+1=BUY, -1=SELL)
- `entry_price: Decimal`, `stop_loss_price: Decimal`, `atr: Decimal`
- `confidence: float = 1.0`, `timestamp: Optional[datetime] = None`, `trace_id: str = ""`

**Exception:** `RiskCheckFailed(Exception)` — stores `reason` attribute

**Class OrderExecutor:**
- `__init__(broker, fsm, rate_limiter, risk_manager, state, store=None, alert_sender=None, fat_finger=None) -> None`
- `start(self) -> None` — starts stale-order timeout checker thread
- `stop(self) -> None`
- `submit_signal(self, signal: TradeSignal) -> Optional[str]` — [SURV-8] session guard, 30bps drift check, limit price calc, ATR-based sizing, risk pre-trade check, fat-finger guard; returns client_order_id or None
- `place_stop_loss(self, symbol, side, quantity, stop_price) -> Optional[str]` — MARKET order
- `close_position(self, symbol, position_qty) -> Optional[str]` — MARKET order
- `_place_order(self, order, trace_id="") -> Optional[str]` — FSM record, rate limiter, broker submit
- `cancel_order(self, client_order_id) -> bool` — FSM + broker cancel, releases rate limiter
- `on_broker_update(self, update: OrderUpdate) -> None` — delegates to FSM
- `_on_fill(self, record, fill) -> None` — logs fill, computes cost, updates position, persists, releases rate limiter, sends alert
- `_timeout_checker_loop(self) -> None` — background thread, checks every 5s
- `_check_stale_orders(self) -> None` — cancels ACKED/PARTIAL LIMIT orders older than stale_order_timeout_sec [SURV-11: LIMIT never converted to MARKET]
- `_get_limit_price(self, signal, current_ltp) -> Decimal` — passive offset based on confidence

---

### 7.2 algo_trading/execution/order_fsm.py

**Constants:**
- `_WAL_PATH: Path = Path("state/orders_wal.jsonl")`
- `_PERSIST_FAILURE_THRESHOLD: int = 5`

**Exception:** `PersistenceHaltedError(Exception)` — trading halted after 5 persistence failures
**Exception:** `TransitionError(Exception)` — illegal state transition

**Enum OrderState(str, Enum):**
- `PENDING, SENT, ACKED, PARTIAL, FILLED` (terminal)
- `CANCEL_PENDING, CANCELLED` (terminal), `REJECTED` (terminal), `HALTED` (terminal)
- Property `is_terminal -> bool`, `is_active -> bool`

**Enum OrderEvent(str, Enum):**
- `SEND, ACK, REJECT, FILL, PARTIAL_FILL, CANCEL_REQUEST, CANCEL_CONFIRM, TIMEOUT`

**Dataclass OrderRecord:**
- `order: Order`, `state: OrderState = PENDING`
- `exchange_order_id: str = ""`, `filled_quantity: int = 0`
- `fills: list[Fill] = []`, `reject_reason: str = ""`
- `created_at: datetime`, `updated_at: datetime`, `sent_at: Optional[datetime]`
- `fill_price_avg: Decimal = Decimal(0)` — VWAP of all fills
- Property `remaining_quantity -> int`
- `add_fill(self, fill: Fill) -> None` — appends fill, updates VWAP

**Abstract class FillCallback:**
- `on_fill(self, record: OrderRecord, fill: Fill) -> None`

**Class OrderFSM:**
- `__init__(store=None) -> None`
- `register_fill_callback(cb: FillCallback) -> None`
- `replay_wal(self) -> int` — reads WAL, replays to QuestDB, clears on success
- `create(self, order: Order) -> OrderRecord` — idempotent; raises ValueError on collision
- `transition(self, client_order_id, event, exchange_order_id="", filled_quantity=0, fill_price=Decimal(0), reject_reason="") -> Optional[OrderRecord]`
- `on_broker_update(self, update: OrderUpdate) -> Optional[OrderRecord]`
- `get_record(self, client_order_id) -> Optional[OrderRecord]`
- `get_active_orders(self) -> list[OrderRecord]`
- `get_all_records(self) -> list[OrderRecord]`
- `_persist(self, record: OrderRecord) -> None` — QuestDB write; WAL fallback; halts after 5 failures

---

### 7.3 algo_trading/execution/rate_limiter.py

**Constants:**
- `_MAX_GLOBAL_OPS: int = 8` — SEBI limit 10 with 2-order buffer
- `_MAX_MODS_PER_ORDER: int = 5`
- `SESSION_END: dtime = dtime(15, 30)`

**Class TokenBucket:**
- `__init__(rate: float, capacity: float) -> None`
- `_refill(self) -> None` — must be called with lock held
- `try_consume(self, tokens=1.0) -> bool` — non-blocking
- `consume(self, tokens=1.0, timeout_sec=1.0) -> bool` — blocking
- Property `available -> float`

**Class DualLockRateLimiter:**
- `__init__(max_ops=8, max_mods=5) -> None` — TokenBucket + per-order counters + halted set + rejection deque
- `_cleanup_orphaned_orders(self) -> None` — must be called with lock held; removes pre-session orders
- `acquire(self, order_id, block=False) -> tuple[bool,str]` — checks halt, increments mod counter, acquires token
- `release_order(self, order_id) -> None` — removes tracking on terminal state
- `is_order_halted(self, order_id) -> bool`
- `get_modification_count(self, order_id) -> int`
- Property `available_tokens -> float`
- `get_stats(self) -> dict` — current_token_count, tracked_orders, orders_rejected_last_minute
- `reset(self) -> None` — end-of-day cleanup

---

### 7.4 algo_trading/execution/signal_hierarchy.py

**Constants:**
- `_VETO_AUDIT_PATH: Path = Path("state/veto_audit.jsonl")`
- `_DRIFT_MULTIPLIER: dict[DriftSeverity,float]` = {NONE:1.0, MODERATE:0.5, SEVERE:0.25}

**Enum SignalPriority(IntEnum):**
- `ADWIN_DRIFT = 1`, `ML_ENSEMBLE = 2`, `HMM_REGIME = 3`, `RISK_MANAGER = 4`

**Enum HMMRegime(str, Enum):** BULL, BEAR, SIDEWAYS, UNKNOWN

**Enum DriftSeverity(str, Enum):** NONE, MODERATE, SEVERE

**Enum RiskStatus(str, Enum):** OK, WARNING, HALT

**Dataclass TradingDecision:**
- `action: Literal["BUY","SELL","HOLD","HALT"]`
- `size_multiplier: float = 1.0`, `veto_reason: str = ""`, `signal_strength: float = 0.0`, `resolved_by: str = ""`

**Functions:**
- `log_veto(timestamp, symbol, signal_direction, veto_reason, veto_source, signal_confidence) -> None` — JSONL append to veto_audit
- `resolve_signal(ml_signal, ml_confidence, regime, drift_flag, risk_status, symbol="") -> TradingDecision` — 4-level priority chain: RISK_MANAGER(4) > HMM_REGIME(3) > ML_ENSEMBLE(2) > ADWIN_DRIFT(1)

**Class SignalConflictLogger:**
- `__init__(self)` — _counters per SignalPriority, _total = 0
- `record(self, decision: TradingDecision) -> None`
- `get_counts(self) -> dict[str,int]`
- `get_rates(self) -> dict[str,float]`
- `log_summary(self) -> None`
- `reset(self) -> None`

---

### 7.5 algo_trading/execution/trade_logger.py

**Constants:**
- `BATCH_SIZE: int = 50`, `FLUSH_INTERVAL_SEC: float = 5.0`
- `_FALLBACK_LOG_FILE: Path = Path(settings.log_dir) / "trade_log_fallback.jsonl"`
- `_STOP_SENTINEL: object`

**Dataclass TradeRecord:**
- `symbol, direction, quantity, entry_price, exit_price=0, entry_time_ns=0, exit_time_ns=0`
- `gross_pnl=0, net_pnl=0, slippage_bps=0, spread_cost_bps=0, latency_ms=0.0`
- `partial_fill_ratio=1.0, abort_reason="", label=0, model_version="", regime_at_entry="", signal_confidence=0.0`
- `trade_id: str = field(default_factory=uuid4)` 
- `to_dict() -> dict`, `to_ilp_columns() -> dict`

**Class TradeLogger:**
- `__init__(questdb_host, questdb_ilp_port) -> None` — creates queue, starts worker thread
- `log_entry(record: TradeRecord) -> None`
- `log_exit(entry_id, exit_price, exit_time_ns, gross_pnl, net_pnl, slippage_bps, partial_fill_ratio=1.0, model_version="") -> None`
- `log_abort(symbol, reason, signal_price, model_version="", regime_at_entry="", signal_confidence=0.0) -> None`
- `flush(self) -> None`
- `close(self) -> None` — blocks up to 30s; safe to call multiple times
- `_run(self) -> None` — batches BATCH_SIZE rows or FLUSH_INTERVAL_SEC
- `_flush_batch(self, batch) -> None` — QuestDB ILP; fallback JSONL
- `_write_to_questdb(self, batch) -> None`
- `_write_to_jsonl(self, batch) -> None`

---

### 7.6 algo_trading/execution/latency_profiler.py

**Constants:**
- `_COMPONENTS: tuple = ("tick_to_feature","feature_to_signal","signal_to_order","order_to_ack","total")`
- `_NS_TO_MS: float = 1e-6`
- `_STATS_PATH: Path = Path("state/latency_stats.json")`
- `_PERSIST_INTERVAL_SEC: float = 900` (15 min)

**Dataclass LatencySnapshot:**
- `symbol, bar_seq, tick_received_ns, feature_computed_ns, signal_generated_ns, order_placed_ns, exchange_ack_ns`
- Properties: `tick_to_feature_ms, feature_to_signal_ms, signal_to_order_ms, order_to_ack_ms, total_ms`
- `component_values(self) -> dict[str,float]`

**Class LatencyProfiler:**
- `__init__(window=100, alert_threshold_multiplier=2.0) -> None`
- `_load_persisted_stats(self) -> None`
- `_persist_stats(self) -> None` — writes P50/P95/P99 to disk
- `_schedule_persist(self) -> None` — reschedules every 15min
- `is_rate_reduced(self) -> bool` — True when 5 consecutive spike bars
- `record(self, snap: LatencySnapshot) -> None` — appends to rolling buffers, checks spikes
- `get_stats(self) -> dict[str,dict[str,float]]` — P50/P95/P99 per component
- `check_spike(self, snap) -> list[str]` — components exceeding 2×P95; tracks consecutive spikes
- `log_to_questdb(self, snap, sender) -> None`

---

## 8. RISK MANAGEMENT

### 8.1 algo_trading/risk/manager.py

**Constants:**
- `_KILL_SWITCH_SENTINEL: Path = Path("state/kill_switch_clear.sentinel")`

**Dataclasses:**
- `RiskCheckResult`: allowed: bool, reason: str = ""
- `RiskState`: kill_switch_active, kill_switch_reason, halted_symbols: set[str], daily_pnl: Decimal, peak_pnl: Decimal

**Class PortfolioCorrelationGuard:**
- `__init__(max_correlation=0.75, lookback_bars=60, min_bars_required=20) -> None`
- `update_returns(symbol, pct_return) -> None` — thread-safe
- `_pearson_correlation(xs, ys) -> float`
- `get_correlation_multiplier(new_symbol, existing_symbols) -> float` — [0.3, 1.0] based on max |ρ|

**Class RiskManager:**
- `__init__(state, store=None, alert_sender=None, correlation_guard=None, fat_finger_guard=None) -> None`
- `can_trade(symbol, side, quantity, price) -> tuple[bool,str]` — 8-check chain: kill switch, halted symbols, session time, daily loss limit, max drawdown, position size, gross exposure, portfolio correlation
- `compute_position_size(signal: TradeSignal) -> int` — ATR-based: size = (capital×risk_pct)/(ATR×multiplier)
- `compute_trade_cost(symbol, side, quantity, price, atr=None) -> TradeCost`
- `trigger_kill_switch(reason: str) -> None` — external trigger (thread-safe)
- `_trigger_kill_switch_internal(reason: str) -> None` — must be called with lock
- `reset_kill_switch(self) -> None` — requires operator acknowledgement
- `check_kill_switch_clear_sentinel(self) -> bool` — checks sentinel file, returns True if cleared
- `halt_symbol(symbol, reason) -> None`
- `resume_symbol(symbol) -> None`
- `check_price_velocity(symbol, price_change_pct) -> None` — halts if ±20% circuit band
- Property `is_kill_switch_active -> bool`
- `is_symbol_halted(symbol) -> bool`
- `get_risk_snapshot(self) -> dict`
- `reset_for_new_day(self) -> None` — clears daily counters, halted symbols, stale ADV cache
- `_persist_risk_event(self, event_type, details) -> None`

---

### 8.2 algo_trading/risk/cost_model.py

**Constants:**
- `_STT_INTRADAY_SELL_PCT: Decimal = 0.00025` (0.025%)
- `_STT_DELIVERY_BOTH_PCT: Decimal = 0.001` (0.1%)
- `_EXCHANGE_CHARGE_PCT: Decimal = 0.0000345` (0.00345%)
- `_SEBI_FEE_PER_CRORE: Decimal = 10` (₹10/Cr)
- `_GST_PCT: Decimal = 0.18` (18%)
- `_STAMP_DUTY_PCT: Decimal = 0.00003` (0.003% buy-side only)
- `_BROKERAGE_PCT: Decimal = 0` (Shoonya zero-brokerage)
- `_ATR_SLIPPAGE_BASE_PCT: Decimal = 0.30`
- `_OPEN_CLOSE_MULTIPLIER: Decimal = 1.5`

**Dataclass TradeCost (frozen=True):**
- `brokerage, stt, exchange_charge, sebi_fee, gst, stamp_duty`
- `slippage_inr, slippage_bps, total_inr, turnover`
- `as_bps_of_turnover(self) -> Decimal`

**Class IndiaCostModel:**
- `compute(symbol, side, quantity, price, is_delivery=False, atr=None, hour_of_day=None, spread=None) -> TradeCost`
- `compute_round_trip(symbol, quantity, entry_price, exit_price=None, atr=None, direction=1) -> TradeCost`
- `_compute_stt(turnover, side, is_delivery) -> Decimal` (static)
- `_compute_stamp_duty(turnover, side) -> Decimal` (static — buy only)
- `_compute_slippage(price, quantity, atr, hour_of_day, spread) -> tuple[Decimal,Decimal]` (static)

---

### 8.3 algo_trading/risk/fat_finger.py

**Enum FatFingerResult(Enum):**
- `OK`, `REJECT_OVER_VALUE`, `REJECT_OVER_ADV`, `REJECT_PRICE_DEVIATION`, `REJECT_QTY_CAP`, `REJECT_GUARD_ERROR`

**Dataclass FatFingerConfig:**
- `max_order_value_inr: Decimal = 200000`, `max_adv_pct: float = 0.02`
- `max_price_deviation_bps: float = 50.0`, `max_quantity: int = 5000`
- `adv_cache_ttl_sec: float = 3600.0`, `adv_fallback: Optional[float] = None`

**Class ADVCache:**
- `__init__(ttl_sec=3600.0) -> None`
- `get(symbol) -> Optional[float]` — evicts expired
- `update(symbol, adv_shares) -> None`
- `invalidate(symbol) -> None`, `clear() -> None`
- `invalidate_stale(session_end_time) -> None`
- `size() -> int`

**Class FatFingerGuard:**
- `__init__(config=None, questdb_store=None) -> None`
- `check(order, ltp, adv_override=None) -> Tuple[FatFingerResult,str]` — fail-closed: never raises; checks qty cap, value cap, price deviation (50bps), ADV% (2%)
- `update_adv(symbol, adv) -> None`, `invalidate_adv(symbol) -> None`
- `clear_adv_cache() -> None`, `invalidate_stale_adv(session_end_time) -> None`
- `_check_impl(order, ltp, adv_override) -> Tuple[FatFingerResult,str]`
- `_get_adv(symbol) -> Optional[float]` — cache → QuestDB → fallback
- `_fetch_adv_from_questdb(symbol) -> Optional[float]` — 30-day average volume query

---

### 8.4 algo_trading/risk/kill_switch_manager.py

**Enum KillLevel(str, Enum):** STRATEGY, PORTFOLIO, HUMAN

**Enum KillReason(str, Enum):**
- `DAILY_LOSS_LIMIT, MAX_DRAWDOWN, DATA_QUALITY, BROKER_ERROR`
- `STUCK_LEG, PARTIAL_ABORT_FAILED, VIX_EMERGENCY`
- `MANUAL, STARTUP_RECONCILE, SESSION_END, UNKNOWN`

**Dataclass KillEvent:**
- `event_id, timestamp, event_type ("TRIGGERED"|"RESET"), level, reason, strategy_id, triggered_by, details, schema_version=1`
- `to_json_line() -> str`, `@classmethod from_json_line(cls, line) -> KillEvent`

**Class KillSwitchManager:**
- `__init__(audit_log_path="audit/kill_switch.jsonl", alert_callback=None) -> None` — loads state from audit log
- `is_trading_allowed(strategy_id=None) -> tuple[bool,str]` — checks human override, portfolio, strategy kills
- `trigger(level, reason, strategy_id=None, triggered_by="SYSTEM", details=None) -> None` — appends audit event
- `reset(level, strategy_id=None, reset_by="SYSTEM", confirm_human=False) -> bool` — requires confirm_human for HUMAN level
- `status(self) -> dict`
- `load_audit_log(self) -> list[KillEvent]`
- `_load_state(self) -> None` — reconstructs by replaying audit log
- `_append_audit(self, event: KillEvent) -> None` — append-only JSONL (SEBI compliance)
- `_send_alert(self, message, level="CRITICAL") -> None`

---

### 8.5 algo_trading/risk/stress_engine.py

**Constants:**
- `SCENARIOS: list[StressScenario]` — 13 NSE F&O stress scenarios

**Dataclass StressScenario (frozen=True):**
- `name: str`, `underlying_shock_pct: float`, `vol_multiplier: float`, `description: str`

**Dataclass ScenarioPnL:**
- `position_id, scenario, delta_pnl, gamma_pnl, vega_pnl, total_pnl, notes=""`

**Dataclass StressResult:**
- `scenario, portfolio_pnl, delta_contribution, gamma_contribution, vega_contribution, position_pnls`
- `vix_trigger_breached: bool = False`
- `WARNING: str = "APPROX-TAYLOR-EXPANSION: large moves not modeled accurately"`
- `pnl_as_pct_of_capital(self, capital) -> float`

**Dataclass StressReport:**
- `underlying_price, current_iv, scenarios: list[StressResult]`
- `worst_case(self) -> StressResult | None`
- `worst_pnl_inr(self) -> float`
- `print_summary(self, capital=100_000.0) -> None`

**Class StressEngine:**
- `__init__(scenarios=None) -> None`
- `run(positions, underlying_price, current_atm_iv) -> StressReport` — second-order Taylor expansion: Δ·ΔS + ½·Γ·ΔS² + Vega·Δσ
- `vix_action_required(current_vix) -> tuple[str,str]` — VIX thresholds: 22→reduce, 25→hedge, 30→halt

---

### 8.6 algo_trading/risk/gap_risk_model.py

**Constants:**
- `GAP_SCENARIOS: list[GapScenario]` — 11 predefined gap scenarios with probabilities
- Assumed statistics: `_ASSUMED_GAP_MEAN_PCT=0.0`, `_ASSUMED_GAP_STD_PCT=0.50`, P95=1.50%, P99=3.00%, extreme=5.00%

**Dataclass GapScenario (frozen=True):**
- `name, gap_pct, iv_multiplier, probability`

**Dataclass GapPnL:**
- `scenario, portfolio_pnl, delta_pnl, vega_pnl, gap_through_stop: bool, notes=""`

**Dataclass GapRiskReport:**
- `current_price, prev_close, current_iv, scenarios: list[GapPnL]`
- `WARNING: str` — disclaimer on assumed probabilities
- `expected_pnl(self) -> float` — probability-weighted
- `worst_gap_pnl(self) -> float`
- `var_95_pct(self, capital) -> float`
- `print_summary(self, capital=100_000.0) -> None`

**Class GapRiskModel:**
- `__init__(scenarios=None, stop_loss_pct=0.02) -> None`
- `compute(positions, prev_close, current_price, current_atm_iv, capital=100_000.0) -> GapRiskReport`
- `@staticmethod calibrate_gap_std_from_history(daily_gaps_pct) -> dict`

---

### 8.7 algo_trading/risk/margin_headroom.py

**Constants:**
- `_WARN_THRESHOLD: float = 0.70`, `_CRITICAL_THRESHOLD: float = 0.85`, `_HALT_THRESHOLD: float = 0.95`

**Dataclass MarginSnapshot:**
- `timestamp, available_capital, theoretical_span, broker_actual, calibrated_estimate, utilization_pct, headroom_inr, positions_count`
- `WARNING: str = "[APPROX-NOT-OFFICIAL-SPAN]: theoretical estimate only"`
- Property `status -> str` — "HALT"|"CRITICAL"|"WARNING"|"OK"
- Property `can_add_positions -> bool` — utilization < 0.85

**Dataclass MarginCalibrationRecord:**
- `timestamp, order_id, theoretical_span, actual_broker_margin, ratio` (computed field = actual/theoretical)

**Class MarginHeadroomMonitor:**
- `__init__(available_capital, margin_model_provider=None, alert_callback=None, calibration_window=20) -> None`
- `snapshot(positions) -> MarginSnapshot` — fires alerts on threshold breach
- `estimate_margin_for_position(new_position) -> float`
- `update_calibration(actual_margin, theoretical_margin) -> None` — EMA: 0.9×old + 0.1×ratio
- `record_broker_margin(order_id, theoretical_span, actual_broker_margin) -> None`
- Property `calibration_factor -> float`
- `headroom_lots(positions, underlying_price, lot_size=65, margin_pct_assumed=0.12) -> int`
- `update_capital(new_capital) -> None`
- `_compute_theoretical_margin(positions) -> float`
- `_send_alert(message, level="WARNING") -> None`

---

### 8.8 algo_trading/risk/position_sizer.py

**Constants:**
- `_SQRT_252: float = math.sqrt(252)`

**Dataclass SizingResult:**
- `shares: int`, `position_value: Decimal`, `risk_per_trade: Decimal`
- `size_method: str` ("atr"|"kelly"|"target_vol"), `capped: bool`, `cap_reason: str`

**Class PositionSizer:**
- `__init__(capital, max_position_pct=0.05, atr_risk_pct=0.01, atr_multiplier=1.5, target_annual_vol=0.15) -> None`
- `size_atr(price, atr) -> SizingResult` — (capital×risk_pct)/(ATR14×atr_multiplier)
- `size_kelly(price, win_rate, avg_win_loss_ratio) -> SizingResult` — Half-Kelly; clamps fraction to [0,1]
- `size_target_vol(price, realized_vol_daily) -> SizingResult` — annualises by √252
- `adjust_for_circuit_risk(shares, price) -> tuple[int,bool]` — 5% circuit-breaker cap
- `_apply_hard_caps(raw_shares, price) -> tuple[int,bool,str]`
- `_zero_result(method, price) -> SizingResult`
- `@classmethod from_settings(cls) -> PositionSizer` — from global config

---

END OF PART 3.
# NSE AlgoTrader Codebase Reference — Part 4: ML Models & Research

---

## 9. ML MODELS

### 9.1 algo_trading/models/ensemble.py

**Constants:**
- `XGB_PARAMS: dict` — {tree_method:"hist", device:"cpu", n_estimators:300, max_depth:8, learning_rate:0.05, subsample:0.8, colsample_bytree:0.8, min_child_weight:5, gamma:0.1, reg_alpha:0.1, reg_lambda:1.0, objective:"multi:softprob", num_class:3, eval_metric:"mlogloss", use_label_encoder:False}
- `LGB_PARAMS: dict` — {device:"cpu", num_leaves:127, n_estimators:300, learning_rate:0.05, subsample:0.8, colsample_bytree:0.8, min_child_samples:20, reg_alpha:0.1, reg_lambda:1.0, objective:"multiclass", num_class:3, verbose:-1}
- `CAT_PARAMS: dict` — {task_type:"CPU", iterations:300, depth:5, learning_rate:0.05, l2_leaf_reg:3.0, border_count:64, verbose:False}
- `_CLASS_TO_SIGNAL: dict[int,int]` — {0:-1, 1:0, 2:1}
- `_SIGNAL_TO_CLASS: dict[int,int]` — {-1:0, 0:1, 1:2}
- `_N_CLASSES: int = 3`

**Dataclass TrainingConfig:**
- `n_folds: int = 5`, `early_stopping_rounds: int = 50`, `eval_split_pct: float = 0.1`

**Class RegimeConditionedMeta:**
- `_MIN_SAMPLES_PER_REGIME: int = 200`
- `__init__(self) -> None`
- `fit(meta_features, y_mapped, regime_labels) -> None` — per-regime LogisticRegression models
- `predict_proba(meta_features, regime_label) -> np.ndarray` — returns class probabilities for regime

**Class MLEnsemble:**
- `MIN_CALIBRATION_SAMPLES: int = 10`
- `__init__(config: TrainingConfig = TrainingConfig()) -> None`
- `train(X, y, sample_weights, feature_names, output_dir, bar_type="time_15min", bar_spec=None, symbol_universe=(), version="v1.0.0", regime_labels=None) -> FrozenModelSpec` — full walk-forward pipeline: OOF predictions, isotonic calibration per model, RegimeConditionedMeta meta-learner, feature importance stability, Treelite compilation; returns FrozenModelSpec
- `predict_proba(X: np.ndarray) -> np.ndarray` — stacked ensemble with meta-learner
- `compute_feature_importance(feature_names) -> dict[str,float]`
- `_walk_forward_oof(X, y, sample_weights, model_type, holdout_fold_reserve=True) -> np.ndarray` — returns (n_samples,3) OOF probability matrix; last fold reserved as holdout
- `_extract_feature_importance(model, model_type) -> np.ndarray | None`
- `_compute_feature_importance_stability(feature_names) -> tuple[int, dict[str,float]]` — Spearman rank correlation across folds
- `_build_and_fit(model_type, X_tr, y_tr, sw_tr, X_ev, y_ev, fold_idx) -> Any`
- `_predict_proba_raw(model, model_type, X) -> np.ndarray` (static)
- `_build_xgb() -> xgb.XGBClassifier` (static), `_build_lgb() -> lgb.LGBMClassifier` (static), `_build_cat() -> CatBoostClassifier` (static)
- `_assert_trained() -> None`

**Function:**
- `compile_for_production(xgb_model, lgb_model, cat_model, output_dir) -> tuple[str,str,str]` — Treelite .so compilation; returns (xgb_so, lgb_so, cat_so)

---

### 9.2 algo_trading/models/inference_engine.py

**Constants:**
- `SIGNAL_SELL: int = -1`, `SIGNAL_HOLD: int = 0`, `SIGNAL_BUY: int = 1`
- `_CLASS_TO_SIGNAL: dict[int,int]` — {0:-1, 1:0, 2:1}
- `CONFIDENCE_THRESHOLD: float = 0.55`

**Dataclass LatencySnapshot:**
- `symbol, t_recv_ns, t_infer_ns, t_push_ns, signal, confidence, published`
- Property `inference_ms -> float`, `total_ms -> float`

**Functions:**
- `_deserialise_feature_vector(data: bytes) -> tuple[str, np.ndarray]` — ZMQ msg → (symbol, feature_vector)
- `_serialise_signal(symbol, signal, confidence) -> bytes`

**Class EnsembleInferenceEngine:**
- `CONFIDENCE_THRESHOLD: float = 0.55`
- `__init__(spec: FrozenModelSpec, zmq_sub_port=5556, zmq_push_port=5557) -> None`
- `load_models(self) -> None` — Treelite .so + meta_lr.pkl; heartbeat monitor (30s silence alert)
- `predict(X: np.ndarray) -> tuple[int,float]` — hot-path inference; returns (signal, confidence)
- `run(symbols: list[str]) -> None` — ZMQ SUB→PUSH main loop
- `latency_stats(self) -> dict[str,float]` — {mean_ms, p50_ms, p95_ms, p99_ms, max_ms, n_samples}
- `_assert_loaded(self) -> None`

---

### 9.3 algo_trading/models/model_registry.py

**Exceptions:**
- `ModelMismatchError(Exception)` — feature schema drift
- `RegistryError(Exception)` — registry-level failures
- `InsufficientDataError(Exception)` — insufficient training data

**Dataclass FrozenModelSpec (frozen=True):**
- `version, feature_names, feature_schema_hash, bar_type, bar_spec, trained_on, symbol_universe`
- `validation_metrics, treelite_xgb_path, treelite_lgb_path, treelite_cat_path, meta_lr_path`
- `package_versions, approval_status, holdout_sharpe, holdout_accuracy, unstable_features_count, min_train_bars_required, min_trades_per_fold_required`
- `validate_input(X_columns) -> None` — raises ModelMismatchError on schema drift
- `validate_bar_type(live_bar_type, live_bar_spec) -> None`
- `@classmethod compute_feature_hash(feature_names) -> str` — 64-char SHA-256
- `save(path) -> None`, `@classmethod load(path) -> FrozenModelSpec`

**Class ModelRegistry:**
- `RETRAIN_DAY: int = 6` (Sunday), `RETRAIN_HOUR: int = 20` (8 PM IST)
- `DEFAULT_MIN_BARS: int = 5000`, `DEFAULT_MIN_TRADES: int = 200`
- `__init__(registry_dir=Path("models/registry")) -> None`
- `register(spec: FrozenModelSpec) -> None` — persists; demotes previous approved
- `get_production_model(self) -> FrozenModelSpec` — raises RegistryError if zero/multiple
- `is_safe_to_retrain(self) -> bool` — Sunday >= 20:00 IST, not market hours
- `list_versions(self) -> list[FrozenModelSpec]` — newest-first
- `validate_training_data(n_bars, n_trades, min_bars=5000, min_trades=200) -> None`
- `_spec_path(version) -> Path` (static), `_demote_current_approved() -> None`
- `_is_market_open(now_ist) -> bool` (static) — 09:15–15:30 weekday

---

### 9.4 algo_trading/models/online_model.py

**Enum AdaptationResult(str, Enum):**
- `NO_CHANGE, ADAPTED, BLOCKED_INSUFFICIENT_DATA, BLOCKED_COOLDOWN, BLOCKED_HIGH_VOL, BLOCKED_EDGE_VALIDATION`

**Enum _DriftSeverity(str, Enum):** NONE, MODERATE, SEVERE

**Dataclass ADWINGuardrails:**
- `MIN_BARS_BEFORE_ADAPT: int = 200`, `MIN_DAYS_BETWEEN_ADAPT: int = 5`
- `FREEZE_WHEN_ATR_RATIO: float = 2.0`, `REQUIRE_EDGE_REVALIDATION: bool = True`

**Class SafeADWINAdapter:**
- `_SIZE_MULT: dict` — {NONE:1.0, MODERATE:0.5, SEVERE:0.25}
- `__init__(guardrails=ADWINGuardrails()) -> None`
- `update(prediction_error, current_atr, baseline_atr_20d) -> AdaptationResult` — ADWIN with 4 guardrails
- `get_size_multiplier(self) -> float` — {0.25, 0.5, 1.0}
- `reset(self) -> None` — clears after Sunday retrain
- Property `total_adaptations -> int`, `total_blocked -> int`, `severity -> str`
- `_rolling_accuracy(self) -> float`
- `_passes_edge_validation(self) -> bool` — accuracy > 50%
- `_compute_severity(self) -> _DriftSeverity`

---

### 9.5 algo_trading/models/tuner.py

**Dataclass TunerConfig:**
- `n_trials=50, timeout_seconds=3600, n_jobs=-1, metric="sharpe", use_fold_minus_2=True, random_seed=42, verbosity=0`

**Functions:**
- `_get_is_safe_to_retrain() -> bool`
- `_complexity_penalty(params) -> float`

**Class HyperparamTuner:**
- `__init__(config=TunerConfig()) -> None`
- `_check_safety(self) -> None` — raises RuntimeError if not Sunday
- `_make_study(study_name) -> optuna.Study` — TPESampler + MedianPruner
- `_compute_sharpe(y_true, y_pred_proba) -> float` (static)
- `tune_xgboost(X_train, y_train, X_val, y_val, sample_weights=None) -> dict` — search: max_depth[3,8], lr[0.01,0.3], n_est[100,500], subsample[0.6,1.0], colsample[0.6,1.0], min_child_weight[1,20]; raises if best_sharpe < 0.3
- `tune_lightgbm(X_train, y_train, X_val, y_val, sample_weights=None) -> dict` — search: num_leaves[15,127], lr[0.01,0.3], n_est[100,500], min_child_samples[5,50], feature_fraction[0.6,1.0]
- `_tune_catboost(X_train, y_train, X_val, y_val, sample_weights=None) -> dict` — search: depth[3,8], lr[0.01,0.3], iterations[100,500], l2_leaf_reg[1.0,10.0]
- `tune_ensemble(X, y, folds, sample_weights=None) -> dict` — tunes all 3 on fold[-2]; returns {"xgb":params, "lgb":params, "cat":params}

---

### 9.6 algo_trading/models/rl_env.py (Stage 4)

**Constants:**
- `_STAGE: str = "4"`, `_N_FEATURES: int = 25`, `_N_STATE_EXTRAS: int = 4`
- `STATE_DIM: int = 29`, `_N_BARS_PER_DAY: int = 26`, `_FORCE_CLOSE_BAR: int = 25`
- `_DIRECTION_CHANGE_PENALTY: float = -0.01`, `_TERMINAL_SHARPE_SCALE: float = 0.1`
- `_ROLLING_WINDOW: int = 20`

**Function:** `_require_stage_4() -> None` — raises NotImplementedError if ALGO_STAGE < 4

**Class NSETradingEnv(gymnasium.Env):**
- `metadata = {"render_modes": ["human"]}`
- `__init__(features, closes, atrs, timestamps, initial_capital=500_000.0, max_position_fraction=0.05, cost_model=None, symbol="RELIANCE", render_mode=None) -> None`
- `reset(seed=None, options=None) -> tuple[np.ndarray, dict]` — resets to bar 0
- `step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]`
- `_get_observation(bar_override=None) -> np.ndarray` — 29-dim float32 state vector
- `_compute_reward(prev_pos, curr_pos, close, prev_close, atr, cost_inr) -> float` — Sharpe-style reward
- `render() -> None`

**Class MultiStockTradingEnv(gymnasium.Env):**
- `__init__(envs: list[NSETradingEnv]) -> None`
- `reset(seed=None, options=None) -> tuple[np.ndarray, dict]` — resets all envs
- `step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]` — concatenated obs
- `render() -> None`

---

### 9.7 algo_trading/models/rl_sizer.py (Stage 4)

**Constants:**
- `_STAGE: str = "4"`, `_WEEKDAYS: frozenset = frozenset(range(5))`

**Dataclass RLSizerConfig:**
- `n_timesteps=500_000, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, device="cpu", policy="MlpPolicy", model_save_path, tensorboard_log`
- `__post_init__` — validates device == "cpu"

**Dataclass RLSizerGuardrails:**
- `min_training_episodes=100, min_sharpe=0.5, max_position_override=0.05, cooldown_days=7, freeze_in_high_vol=True`

**Class RLPositionSizer:**
- `__init__(config, guardrails) -> None`
- `train(train_envs, val_env) -> dict` — PPO training; returns {status, sharpe, episodes}
- `load(path=None) -> None` — loads pre-trained PPO zip
- `get_position_fraction(observation, current_atr, baseline_atr) -> float` — returns [-max_override, +max_override]
- `is_safe_to_retrain(self) -> bool` — cooldown passed, not market hours
- `_require_sb3(operation) -> None`, `_make_env_factory(env)` (static), `_run_validation_episode(model, val_env) -> float`

---

### 9.8 algo_trading/models/sentiment.py (Stage 4)

**Constants:**
- `STAGE: str = "4"`, `SENTIMENT_ZMQ_PORT: int = 5565`

**Dataclass SentimentScore:**
- `timestamp, symbol, headline, score, label, confidence, source`

**Class SentimentPublisher:**
- `__init__(port=5565) -> None`
- `publish(symbol, headline, timestamp=None) -> None` — ZMQ PUSH, fire-and-forget, never raises

**Class SentimentWorker:**
- `_BULLISH_WORDS: frozenset[str]`, `_BEARISH_WORDS: frozenset[str]`
- `__init__(pull_port=5565, push_port=5566) -> None`
- `_load_model(self) -> None` — FinBERT via transformers; fallback to keyword-based
- `score_headline(headline) -> tuple[float,str,float]` — (score, label, confidence)
- `_score_finbert(headline) -> tuple[float,str,float]`
- `_score_keyword(headline) -> tuple[float,str,float]`
- `run(self) -> None` — ZMQ PULL→PUSH main loop; blocks until SIGTERM

**Class SentimentAggregator:**
- `__init__(pull_port=5566, window_minutes=240) -> None`
- `poll(timeout_ms=0) -> list[SentimentScore]` — drains all available (non-blocking)
- `get_current_score(symbol) -> float` — 4h EWM window in [-1.0, +1.0]
- `get_feature_value(symbol) -> float` — feature #24
- `_evict_old(self) -> None`

---

### 9.9 algo_trading/models/tracker.py

**Function:**
- `_mlflow_with_retry(operation, fallback_payload=None, operation_name="mlflow_write") -> T | None` — 3 retries with exponential backoff; JSON fallback on permanent failure

**Class ExperimentTracker:**
- `__init__(tracking_uri="./mlruns", experiment_name="nse_algotrader") -> None`
- `@contextmanager start_run(run_name, tags=None) -> Generator`
- `log_params(params) -> None`, `log_metrics(metrics, step=None) -> None`
- `log_model(model, model_name, feature_names) -> None` — logs with feature schema + package versions
- `log_feature_importance(importance_dict) -> None`
- `log_walk_forward_results(fold_metrics) -> None`
- `get_best_run(metric="sharpe") -> dict`
- `register_production_model(run_id, model_name) -> None` — promotes to "Production" stage

**Class DVCManager:**
- `__init__(repo_root=Path(".")) -> None`
- `_run(cmd, check=True) -> subprocess.CompletedProcess`
- `add_data(data_path) -> None`, `push_model(model_path) -> None`, `pull_model(version_tag) -> None`

---

### 9.10 algo_trading/models/explainer.py

**Class EnsembleExplainer:**
- `_DEFAULT_CLASS_LABELS: dict` — {0:"bear", 1:"bull", 2:"sideways"}
- `__init__(xgb_model, lgb_model, cat_model, feature_names, class_labels=None) -> None`
- `compute_shap_values(X: np.ndarray) -> dict[str, np.ndarray]` — {"xgb","lgb","cat"} → (n_samples, n_features, n_classes)
- `compute_average_importance(X) -> dict[str,float]` — sorted descending
- `generate_sebi_report(X, output_path, model_version, symbol) -> Path` — SEBI-compliant JSON with top features, model version, timestamp
- `plot_summary(X, output_path) -> None` — SHAP summary bar plot PNG
- `explain_single_prediction(x: np.ndarray) -> dict[str,float]` — sorted by absolute SHAP value

---

## 10. BACKTESTING & RESEARCH

### 10.1 algo_trading/backtest/runner.py

**Constants:**
- `_GATE_MIN_SHARPE: float = 1.0`, `_GATE_MIN_PROFIT_FACTOR: float = 1.2`
- `_GATE_MIN_WIN_RATE: float = 0.40`, `_GATE_MIN_POSITIVE_REGIMES: int = 2`
- `_GATE_MIN_TRADES_PER_FOLD: int = 200`
- `_REGIME_EMA_PERIOD: int = 50`, `_REGIME_FLAT_THRESHOLD: float = 0.001`
- `_BARS_PER_DAY: int = 25`, `_TRADING_DAYS_PER_YEAR: int = 250`

**Dataclasses:**
- `BacktestConfig`: initial_capital, max_position_pct=0.05, stop_loss_atr_mult=1.5, max_holding_bars=20, no_trade_before="09:30", square_off_time="15:15", adverse_selection_bps=5.0, max_price_chase_bps=30.0
- `TradeResult`: symbol, entry_bar, exit_bar, entry_timestamp, direction, entry_price, exit_price, quantity, gross_pnl, net_pnl, cost_breakdown, exit_reason, signal_confidence
- `BacktestReport`: total_trades, win_rate, profit_factor, sharpe_ratio, max_drawdown, total_net_pnl, per_fold_metrics, regime_breakdown, cost_analysis, per_symbol_metrics, passes_edge_gate, gate_failures

**Class BacktestRunner:**
- `__init__(config=BacktestConfig()) -> None`
- `run(df, signals, feature_df) -> BacktestReport` — full walk-forward backtest
- `run_ema_baseline(df, fast=9, slow=21) -> BacktestReport`
- `_check_edge_gate(report) -> tuple[bool, list[str]]` — 5 criteria: Sharpe>=1.0, PF>1.2, WR>=40%, >=2 positive regimes, >=200 trades/fold
- `_simulate_fold(ohlcv, signals, features, symbol, bar_offset) -> list[TradeResult]` — 1-bar fill delay, stale signal abort >30bps [SURV-11]
- `_build_report(all_trades, per_fold_metrics, regime_breakdown) -> BacktestReport`
- `_compute_fold_metrics(trades, fold_index) -> dict`
- `_classify_regimes(df) -> list[str]` — bull/bear/sideways per bar
- `_compute_regime_breakdown(trades, regimes) -> dict[str,dict]`
- `_compute_ema_signals(df, fast, slow) -> pl.Series` — Int8 {-1,0,+1}
- `_compute_atr_features(df) -> pl.DataFrame`
- `_compute_sharpe(returns) -> float`, `_compute_max_drawdown(returns) -> float`
- `_validate_inputs(df, signals, feature_df) -> None`
- `_validate_ohlcv(df) -> None` (static), `_parse_hhmm(hhmm) -> tuple[int,int]` (static)

---

### 10.2 algo_trading/research/audit_store.py

**Dataclass ExperimentRecord:**
- `run_id, created_at, git_commit, config_hash, data_hash, strategy_name, strategy_version, strategy_params, data_params, results, metadata, schema_version=1`
- `to_json_line() -> str`, `@classmethod from_json_line(cls, line) -> ExperimentRecord`

**Functions:**
- `_get_git_commit_hash() -> str | None`
- `_hash_data_config(config: dict) -> str` — 16-char SHA-256

**Class AuditStore:**
- `SCHEMA_VERSION: int = 1`
- `__init__(store_path="audit/experiments.jsonl") -> None`
- `log_experiment(strategy_name, strategy_version, strategy_params, data_params, results, metadata=None) -> ExperimentRecord`
- `load_all(self) -> list[ExperimentRecord]` — skips malformed lines
- `find_by_config_hash(config_hash) -> list[ExperimentRecord]`
- `find_by_strategy(strategy_name) -> list[ExperimentRecord]` — newest-first
- `check_determinism(config_hash) -> bool` — True if all matching runs produced identical results
- `summarize(strategy_name=None) -> dict` — {count, avg_sharpe, best_run_id, strategy_name}

---

### 10.3 algo_trading/research/chain_replay_engine.py

**Dataclass ReplayConfig (frozen=True):**
- `symbol, start_date, end_date, snapshot_interval_seconds=60, expiry_filter=None, data_quality_threshold="PARTIAL", strategy_params={}`
- `config_hash() -> str` — 16-hex SHA-256

**Dataclass ReplayEvent:**
- `event_time, sequence_number, snapshot, session_day, bars_elapsed_today, is_expiry_bar, minutes_to_session_close`
- Property `is_near_close -> bool` — < 30 minutes remaining

**Class ChainReplayEngine:**
- `_QUALITY_ORDER: dict` — {UNUSABLE:0, PARTIAL:1, COMPLETE:2}
- `__init__(data_provider, config: ReplayConfig) -> None`
- `replay(self) -> Iterator[ReplayEvent]` — deterministic; no wall-clock calls
- `get_config_hash(self) -> str`
- `estimate_event_count(self) -> int`
- `_quality_ok(quality) -> bool`, `_extract_event_time(snap)` (static), `_extract_sequence(snap)` (static), `_extract_quality(snap)` (static), `_extract_expiry(snap)` (static)

---

### 10.4 algo_trading/research/fill_simulator.py

**Enum FillMode(str, Enum):**
- `OPTIMISTIC, REALISTIC, PESSIMISTIC, QUEUE_POSITION, TIME_OF_DAY_LIQUIDITY`
- `EXPIRY_DAY_BLOWOUT, EVENT_DAY_BLOWOUT, MARKET_ORDER_THIN_BOOK, CANCELLATION_LATENCY`

**Dataclass SimulatedFill:**
- `order_id, symbol, expiry, strike, option_type, side, requested_qty, filled_qty`
- `fill_price, slippage_bps, fill_mode, timestamp, is_partial, cancelled, notes`

**Dataclass FillSimulatorConfig:**
- `optimistic_slippage_bps=0.0, realistic_slippage_bps=5.0, pessimistic_slippage_bps=15.0`
- `queue_fill_probability=0.65, min_fill_probability=0.30`
- `open_auction_spread_multiplier=4.0, midday_spread_multiplier=1.0, pre_close_spread_multiplier=1.5`
- `expiry_final_hour_spread_multiplier=3.0, otm_expiry_fill_probability_cap=0.50`
- `event_day_spread_multiplier=3.5, event_day_otm_fill_prob_cap=0.40`
- `low_oi_threshold_lots=500, thin_book_extra_slippage_bps=20.0`
- `cancel_fill_race_probability=0.05, min_oi_lots_for_normal_fill=100`

**Class FillSimulator:**
- `__init__(config=None, mode=FillMode.REALISTIC) -> None`
- `simulate_fill(order_id, symbol, expiry, strike, option_type, side, qty_lots, quote, event_time, is_expiry_day=False, is_event_day=False, mode=None) -> SimulatedFill | None`
- `_fill_optimistic(...)`: fills at mid price
- `_fill_realistic(...)`: ask+half_spread (buy) or bid-half_spread (sell)
- `_fill_pessimistic(...)`: full ask/bid + market-impact bps
- `_fill_queue_position(...)`: LIMIT queue model; None if miss
- `_fill_time_of_day(...)`: time-of-day spread multiplier
- `_fill_expiry_blowout(...)`: OTM fill-probability cap
- `_fill_event_blowout(...)`: RBI/Budget day blowout
- `_fill_thin_book(...)`: low-OI extra slippage
- `_fill_cancel_race(...)`: cancel→fill race
- `_bps(fill_price, mid) -> float` (static), `_make_rng(event_time, strike, salt) -> random.Random` (static), `_get_field(obj, name, default)` (static)

---

### 10.5 algo_trading/research/metrics_engine.py

**Dataclasses:**
- `TradeRecord`: pnl_inr, entry_time, exit_time, holding_days, strategy_name="", notes=""
- `MetricsSuite`: total_pnl_inr, annualized_return_pct, sharpe_ratio, sharpe_ci_95, calmar_ratio, max_drawdown_pct, max_drawdown_inr, avg_drawdown_pct, max_drawdown_duration_days, total_trades, winning_trades, losing_trades, win_rate, avg_win_inr, avg_loss_inr, profit_factor, expectancy_per_trade_inr, sortino_ratio, var_95_inr, cvar_95_inr, max_consecutive_losses, pct_profitable_months, fold_sharpe_std, passes_min_trade_count, passes_sharpe_threshold, passes_profit_factor
  - `summary_str() -> str`

**Functions:**
- `compute_metrics(trades, capital=100_000.0, n_bootstrap=1000, min_sharpe_threshold=0.5) -> MetricsSuite`
- `per_fold_stability(fold_metrics: list[MetricsSuite]) -> dict` — fold_sharpe_mean, fold_sharpe_std, fold_sharpe_min, fold_sharpe_max, max_fold_pnl_contribution_pct, passes_concentration_limit, all_folds_positive
- `_compute_drawdown(equity_curve, trades) -> tuple[float,float,float,float]` — (max_dd_pct, max_dd_inr, avg_dd_pct, max_dd_duration_days)
- `_annualized_sharpe(daily_pnls, capital) -> float`
- `_bootstrap_sharpe_ci(daily_pnls, capital, n=1000, seed=42) -> tuple[float,float]` — 95% block-bootstrap
- `_annualized_return(equity_curve, trades, capital) -> float` — CAGR
- `_aggregate_to_daily(trades) -> np.ndarray`
- `_pct_profitable_months(trades) -> float`
- `_sortino_ratio(daily_pnls, capital) -> float`
- `_var_cvar_95(daily_pnls) -> tuple[float,float]`
- `_max_consecutive_losses(trades) -> int`
- `_empty_metrics(capital, min_sharpe_threshold) -> MetricsSuite`

---

### 10.6 algo_trading/research/slippage_model.py

**Enum SlippageRegime(str, Enum):** NORMAL, HIGH_VIX, EXPIRY_EARLY, EXPIRY_FINAL, EVENT_DAY, OPEN_AUCTION

**Dataclass SlippageTier (frozen=True):**
- `regime, limit_slippage_bps, market_slippage_bps, atm_spread_multiplier, otm_spread_multiplier, fill_probability_limit, notes`

**Dataclass SlippageEstimate:**
- `regime, order_type, is_atm, slippage_bps, spread_multiplier, fill_probability, notes`
- `slippage_inr(premium_inr) -> float`

**Constants:**
- `SLIPPAGE_TIERS: dict[SlippageRegime, SlippageTier]` — maps regime → SlippageTier

**Class SlippageModel:**
- `__init__(event_dates=None) -> None`
- `detect_regime(event_time, is_expiry_day, india_vix=None) -> SlippageRegime` — priority: EVENT_DAY > EXPIRY_FINAL > ... > NORMAL
- `estimate(order_type, is_atm, event_time, is_expiry_day=False, india_vix=None, regime_override=None) -> SlippageEstimate`
- `round_trip_cost_bps(order_type, is_atm, entry_time, exit_time, is_entry_expiry=False, is_exit_expiry=False, india_vix=None) -> float`
- `regime_summary(self) -> None` (static) — prints tier table

---

### 10.7 algo_trading/research/strategy_simulator.py

**Dataclasses:**
- `OrderIntent`: symbol, expiry, strike, option_type, side, qty_lots, order_type="LIMIT", strategy_tag=""
- `OpenPosition`: position_id, order_intent, fill_price, fill_time, qty_lots, lot_size
- `ClosedTrade`: position_id, symbol, expiry, strike, option_type, opening_side, open_price, close_price, open_time, close_time, qty_lots, lot_size, pnl_gross, total_cost, pnl_net, strategy_tag="", fill_mode=""
  - `holding_days() -> float`
- `StrategyState`: open_positions, closed_trades, session_day, bars_today, is_expiry_bar, custom
  - Property `net_delta_lots -> int`

**Class StrategySimulator:**
- `__init__(fill_simulator, cost_model, lot_size_fn, slippage_model=None) -> None`
- `run(strategy_fn, replay_iter, max_open_positions=20) -> tuple[list[ClosedTrade], StrategyState]`
- `_try_close(state, intent, fill, event_time, lot_size) -> bool` — FIFO matching
- `_force_close_expiry_positions(state, event, counter_base) -> None`
- `_get_quote(snapshot, intent) -> Any | None`, `_compute_cost(symbol, side, qty_lots, lot_size, price, trade_date) -> float`

---

### 10.8 algo_trading/research/walk_forward_runner.py

**Dataclasses:**
- `WalkForwardConfig`: train_years=1.0, test_years=0.5, n_folds=5, purge_bars=20, embargo_bars=5, max_feature_lookback_bars=26, bar_interval_minutes=60, min_trades_per_fold=100, min_sharpe_oos=0.5, max_single_fold_pnl_contribution=0.40, holdout_sharpe_min=0.5
  - Property `total_purge_bars -> int`
- `FoldResult`: fold_idx, train/test start/end/purge/embargo, train_metrics, test_metrics, trades_in_fold
  - Property `oos_is_ratio -> float` — test_sharpe/train_sharpe (>= 0.5 for Phase 2)
- `WalkForwardResult`: config, folds, aggregate_metrics, stability_stats, holdout_metrics, passes_min_folds, passes_min_trades, passes_oos_is_ratio, passes_concentration, passes_holdout
  - Property `passes_all -> bool`
  - `print_summary(self) -> None`

**Class WalkForwardRunner:**
- `__init__(config=None) -> None`
- `generate_splits(start_date, end_date) -> list[tuple[date,date,date,date,date,date]]` — (train_start, train_end, purge_start, embargo_end, test_start, test_end)
- `run(strategy_fn, data_provider, start_date, end_date, symbol, capital=100_000.0, holdout_pct=0.15, audit_store=None, strategy_name="unnamed", strategy_version="v1.0", strategy_params=None) -> WalkForwardResult`

---

END OF PART 4.
## 11. F&O SUBSYSTEM

### 11.1 algo_trading/fo/pricing.py

**Class BSMEngine:**
- `_RISK_FREE_RATE: float = 0.065` (RBI repo rate)
- `call_price(S, K, T, r, sigma) -> float`
- `put_price(S, K, T, r, sigma) -> float`
- `greeks(S, K, T, r, sigma, option_type) -> dict` — {delta, gamma, theta, vega, rho}
- `higher_order_greeks(S, K, T, r, sigma) -> dict` — {vanna, volga, charm}
- `implied_volatility(market_price, S, K, T, r, option_type, max_iter=100, tol=1e-6) -> Optional[float]` — Newton-Raphson + bisection fallback
- `chain_iv(contracts: list[OptionContract], S, T, r=0.065) -> list[float]` — batch IV computation
- `@staticmethod _manaster_koehler_initial_guess(market_price, S, K, T, r) -> float` — Manaster-Koehler (1982) initial guess for OTM

---

### 11.2 algo_trading/fo/strategies.py

**Dataclass StrategyLeg:**
- `symbol, strike, option_type, side ("BUY"|"SELL"), qty_lots, entry_premium, lot_size`
- `net_premium(self) -> float` — positive = credit received

**Dataclass StrategyResult:**
- `strategy_name, legs: list[StrategyLeg], net_premium_total, max_profit, max_loss, breakeven_points: list[float]`
- `payoff_at_expiry(underlying_price) -> float`

**Class StrategyBuilder:**
- `__init__(chain: OptionChain, lot_size: int = 65) -> None`
- `short_strangle(otm_distance: float = 100.0, lots: int = 1) -> StrategyResult` — OTM call + OTM put sold; max profit = net premium
- `short_straddle(lots: int = 1) -> StrategyResult` — ATM call + ATM put sold
- `iron_condor(wing_distance: float = 200.0, lots: int = 1) -> StrategyResult` — 4-leg: sell OTM put spread + OTM call spread
- `bull_put_spread(otm_distance: float = 100.0, lots: int = 1) -> StrategyResult` — buy lower put, sell higher put
- `bear_call_spread(otm_distance: float = 100.0, lots: int = 1) -> StrategyResult` — buy higher call, sell lower call
- `long_straddle(lots: int = 1) -> StrategyResult` — ATM call + ATM put bought
- `covered_future_short_call(fut_lots: int = 1, call_distance: float = 100.0) -> StrategyResult` — long future + short OTM call

---

### 11.3 algo_trading/fo/signal_generator.py

**Constants:**
- `_MIN_DTE: int = 7`, `_MAX_DTE: int = 45`
- `_IV_RANK_HIGH_THRESHOLD: float = 0.65`, `_VRP_POSITIVE_THRESHOLD: float = 0.02`
- `_PCR_BULLISH_THRESHOLD: float = 0.8`, `_PCR_BEARISH_THRESHOLD: float = 1.2`

**Dataclass FOSignal:**
- `symbol, timestamp, strategy_name, confidence, dte, iv_rank, vrp, pcr, signal_reason`

**Class FOSignalGenerator:**
- `__init__(symbol: str) -> None`
- `generate(features: FOFeaturesResult, vrp_estimate, dte: int) -> Optional[FOSignal]`
  - DTE gate: [7,45] — returns None outside range
  - High IV rank (>0.65) → short premium strategies (short strangle/straddle by PCR)
  - Low IV rank (<0.35) → long vol (long straddle)
  - Mid IV rank → directional spreads by PCR
  - Confidence = 0.40×IV_rank + 0.35×VRP + 0.25×PCR_signal
- `_compute_pcr_signal(pcr) -> float` — +1.0 bullish, -1.0 bearish, 0.0 neutral
- `_route_high_iv_strategy(pcr, vrp) -> str`
- `_route_low_iv_strategy(pcr) -> str`
- `_route_mid_iv_strategy(pcr, vrp) -> str`

---

### 11.4 algo_trading/fo/portfolio_risk.py

**Dataclass VRPEstimate:**
- `realized_vol: float`, `implied_vol: float`, `vrp: float` (= implied - realized)
- `is_positive -> bool` (property)

**Class PortfolioRiskEngine:**
- `__init__(cost_model: Optional[object] = None) -> None`
- `compute_portfolio_greeks(positions: list[dict]) -> dict` — aggregates delta/gamma/theta/vega across legs
- `vrp_signal(realized_vol_20d: float, atm_iv: float) -> VRPEstimate`
- `delta_hedge_units(net_delta: float, lot_size: int = 65) -> int` — lots needed to neutralize delta
- `compute_pnl_attribution(positions, underlying_move_pct, iv_change) -> dict` — delta, gamma, vega, theta P&L

---

### 11.5 algo_trading/fo/margin.py

**Class SPANMarginEngine:**
- `_SOMC_FLOOR_PCT: float = 0.03` (3% assumed SOMC floor)
- `_EXPOSURE_MARGIN_PCT: float = 0.015` (1.5% assumed)
- `__init__(n_scenarios: int = 16) -> None` — 16-scenario SPAN array
- `compute_span_margin(position: dict, underlying_price: float) -> float` — SPAN scan_risk + SOMC
- `compute_exposure_margin(position: dict, underlying_price: float) -> float` — 1.5% of notional
- `compute_total_margin(positions: list[dict], underlying_price: float) -> float` — scan_risk + exposure

---

### 11.6 algo_trading/fo/sabr.py

**Dataclass SABRParams:**
- `alpha: float` (initial vol), `beta: float = 1.0` (fixed), `rho: float` (correlation), `nu: float` (vol-of-vol)

**Functions:**
- `sabr_vol(F, K, T, params: SABRParams) -> float` — Hagan 2002 Eq.2.17b approximation
- `fit_sabr(strikes, market_vols, F, T, beta=1.0) -> SABRParams` — L-BFGS-B with 3 random restarts; minimizes sum-squared vol error

---

### 11.7 algo_trading/fo/vol_surface.py

**Class SVISurface:**
- `__init__() -> None`
- `fit(strikes, market_vols, F, T) -> dict` — fits SVI parametric surface; no-arbitrage constraints C1-C5
- `vol_at_strike(K, F, T, params: dict) -> float` — interpolates vol at (K,T)
- `check_no_arbitrage(params: dict) -> bool` — validates C1-C5

**Class VolSurfaceRouter:**
- `__init__() -> None`
- `get_vol(strike, F, T, chain: OptionChain) -> float` — routes: SVI for DTE<=21, SABR for DTE>21

---

### 11.8 algo_trading/fo/vix_estimator.py

**Dataclass VIXEstimate:**
- `vix: float`, `near_term_var: float`, `next_term_var: float`, `interpolation_weight: float`, `timestamp: datetime`

**Function:**
- `estimate_india_vix(near_chain: OptionChain, next_chain: OptionChain, risk_free_rate: float = 0.065) -> VIXEstimate` — CBOE 2009 exact formula; VIX² = 100×√(interpolated); near/next-term weighted interpolation toward 30-day constant maturity

---

### 11.9 algo_trading/fo/execution.py

**Class MultiLegExecutor:**
- `_POLL_INTERVAL_SEC: float = 0.5`
- `_EXECUTION_WINDOW_SEC: int = 30`
- `__init__(broker: BrokerBase, risk_manager, alert_sender=None) -> None`
- `execute_strategy(strategy: StrategyResult, underlying_price: float, max_premium_bps: float = 50.0) -> list[Fill]` — polling-based fill detection; short-first leg ordering; auto-aborts on premium breach; returns fills list
- `unwind_strategy(fills: list[Fill], strategy_result: StrategyResult) -> list[Fill]` — sends opposite-leg orders
- `_get_leg_order(leg: StrategyLeg, underlying_price: float) -> Order`
- `_poll_for_fill(client_order_id: str) -> Optional[Fill]`
- `_compute_current_premium(pending_legs: list[StrategyLeg]) -> float`

---

### 11.10 algo_trading/fo/greeks_hedger.py

**Dataclass HedgeRecord:**
- `hedge_id, timestamp, net_delta_before, net_delta_after, hedge_lots, hedge_direction, fill_price, hedge_reason`

**Class DeltaHedger:**
- `_DELTA_THRESHOLD: float = 0.50` — lots threshold for hedging
- `__init__(lot_size: int = 65, delta_threshold: float = 0.50) -> None`
- `needs_hedge(net_delta: float) -> bool` — |net_delta| > threshold
- `compute_hedge_lots(net_delta: float) -> tuple[int,str]` — (lots, "BUY"|"SELL")
- `record_hedge(net_delta_before, net_delta_after, lots, direction, fill_price, reason="") -> HedgeRecord`
- `unwind_hedge(current_hedge_lots: int, direction: str) -> tuple[int,str]` — reverse direction
- `get_hedge_history(self) -> list[HedgeRecord]`

---

### 11.11 algo_trading/fo/expiry.py

**Class ExpiryCalendar:**
- `__init__() -> None`
- `get_nifty_expiry(year: int, month: int) -> date` — last Thursday of month
- `get_banknifty_expiry(year: int, month: int) -> date` — last Thursday of month (monthly)
- `get_weekly_nifty_expiry(reference_date: date) -> date` — nearest Thursday
- `get_expiry_for_dte(symbol: str, min_dte: int = 7, max_dte: int = 45) -> Optional[date]`
- `days_to_expiry(expiry: date, from_date: Optional[date] = None) -> int`
- `is_expiry_day(symbol: str, d: Optional[date] = None) -> bool`
- `_adjust_for_holiday(d: date) -> date` — moves backward if holiday

---

### 11.12 algo_trading/fo/contracts.py

**Constants:**
- `CONTRACT_REGISTRY: dict[str, dict]` — NIFTY50_FUT: {lot_size:65, exchange:"NFO", segment:"FUT"}, BANKNIFTY_FUT: {lot_size:30, exchange:"NFO", segment:"FUT"}

**Functions:**
- `get_lot_size(symbol: str, trade_date: Optional[date] = None) -> int` — date-aware lot size lookup
- `get_contract_spec(symbol: str) -> dict`

---

### 11.13 algo_trading/fo/cost_model.py

**Class FOCostModel:**
- `_STT_OPTIONS_PCT: float = 0.0005` (0.05% on sell side, options)
- `_STT_FUTURES_PCT: float = 0.0001` (0.01% intraday futures)
- `_EXCHANGE_CHARGE_PCT: float = 0.0000530` (0.00053% for F&O)
- `_SEBI_FEE_PER_CRORE: float = 10.0`, `_GST_PCT: float = 0.18`
- `_STAMP_DUTY_BUY_PCT: float = 0.00003`
- `compute_option_cost(side, qty_lots, lot_size, premium, strike, is_delivery=False) -> dict`
- `compute_futures_cost(side, qty_lots, lot_size, price) -> dict`
- `compute_strategy_cost(strategy: StrategyResult) -> dict` — total cost across all legs
- `round_trip_cost(open_legs: list, close_legs: list) -> float` — INR total

---

### 11.14 algo_trading/fo/interfaces.py

**ABCs (5 interfaces):**
- `IBroker` — `place_order(order) -> str`, `cancel_order(id) -> bool`, `get_fills() -> list`
- `IDataProvider` — `get_chain(symbol, expiry) -> OptionChain`, `get_historical_chains(symbol, start, end) -> list`
- `IStorage` — `write_chain(chain) -> None`, `read_chains(symbol, start, end) -> list`, `write_fill(fill) -> None`
- `IMarginProvider` — `compute_margin(positions) -> float`, `get_available_margin() -> float`
- `IVixProvider` — `get_current_vix() -> float`, `get_vix_history(days) -> list[float]`

---

### 11.15 algo_trading/fo/instrument_master/lot_size_history.py

**Constants:**
- `_LOT_SIZE_EPOCHS: dict[str, list[tuple[date, int]]]` — date-aware lot size history:
  - NIFTY: [(2024-01-01, 75), (2026-01-01, 65)]
  - BANKNIFTY: [(2024-01-01, 35), (2026-01-01, 30)]

**Functions:**
- `get_lot_size(symbol: str, trade_date: Optional[date] = None) -> int` — returns lot size valid on trade_date; raises KeyError on unknown symbol

---

### 11.16 algo_trading/fo/instrument_master/strike_generator.py

**Functions:**
- `get_atm_strike(underlying_price: float, strike_interval: float = 50.0) -> float` — round to nearest 50
- `get_otm_strikes(underlying_price, n_strikes=5, strike_interval=50.0, option_type="CE") -> list[float]` — n strikes away from ATM
- `classify_moneyness(strike, underlying_price, option_type) -> str` — "ITM"|"ATM"|"OTM"
- `generate_strike_range(underlying_price, n_strikes_each_side=10, strike_interval=50.0) -> list[float]`

---

## 12. MARKET DATA

### 12.1 algo_trading/market_data/bhav_copy_ingestor.py

**Constants:**
- `_LEGACY_CUTOVER: date` — 2024-07-08 (format switch date)
- Legacy columns: TtlTrfVal; New UDiFF columns: TtlValOfTxsExctd, FinInstrmTp

**Dataclass FOBhavRow:**
- `symbol, expiry, strike, option_type, open, high, low, close, volume, oi, change_in_oi, trade_date`

**Class BhavCopyIngestor:**
- `__init__(output_dir: Path = Path("data/bhavcopy")) -> None`
- `ingest_cm_bhavcopy(file_path: Path) -> pl.DataFrame` — detects legacy vs UDiFF format by date; parses 34-column schema
- `ingest_fo_bhavcopy(file_path: Path) -> list[FOBhavRow]`
- `download_and_ingest(trade_date: date) -> pl.DataFrame` — downloads from NSE, saves Parquet
- `_detect_format(file_path: Path) -> str` — "legacy" or "udiff"
- `_parse_legacy(df: pd.DataFrame) -> pl.DataFrame`
- `_parse_udiff(df: pd.DataFrame) -> pl.DataFrame` — TtlValOfTxsExctd column fix

---

### 12.2 algo_trading/market_data/chain_normalizer.py

**Class ChainNormalizer:**
- `normalize(raw_chain: dict) -> OptionChain` — converts broker-specific dict format to OptionChain
- `normalize_shoonya(raw: dict) -> OptionChain`
- `normalize_truedata(raw: dict) -> OptionChain`
- `_extract_strikes(raw: dict) -> list[OptionContract]`

---

### 12.3 algo_trading/market_data/chain_snapshot.py

**Dataclass OptionQuote:**
- `strike: float`, `option_type: str`, `bid: float`, `ask: float`, `ltp: float`
- `oi: int`, `volume: int`, `iv: float`, `delta: float`, `gamma: float`, `theta: float`, `vega: float`

**Dataclass ChainSnapshot:**
- `symbol: str`, `underlying_price: float`, `timestamp: datetime` (exchange-stamped, use for replay)
- `expiry: date`, `schema_version: int`, `data_quality: str`
- `quotes: list[OptionQuote]`
- `get_atm_quote(option_type) -> Optional[OptionQuote]`
- `get_quote(strike, option_type) -> Optional[OptionQuote]`
- `total_call_oi -> int` (property), `total_put_oi -> int` (property)
- `pcr -> float` (property) — put/call OI ratio

---

### 12.4 algo_trading/market_data/snapshot_archiver.py

**Class SnapshotArchiver:**
- `__init__(output_dir: Path, compression: str = "zstd") -> None`
- `archive(snapshot: ChainSnapshot) -> Path` — writes compressed Parquet partitioned by date/symbol
- `load(symbol: str, date: date) -> list[ChainSnapshot]`
- `load_range(symbol: str, start: date, end: date) -> list[ChainSnapshot]`
- `_path(symbol: str, d: date) -> Path`

---

### 12.5 algo_trading/market_data/staleness_detector.py

**Constants:**
- `_WARNING_THRESHOLD_SEC: int = 30`, `_CRITICAL_THRESHOLD_SEC: int = 60`

**Class StalenessDetector:**
- `__init__(warning_sec=30, critical_sec=60) -> None`
- `check(snapshot: ChainSnapshot) -> str` — returns "OK"|"WARNING"|"CRITICAL"; uses event_timestamp in replay
- `is_stale(snapshot: ChainSnapshot) -> bool` — True if CRITICAL
- `get_staleness_sec(snapshot: ChainSnapshot) -> float`

---

## 13. ORCHESTRATORS

### 13.1 algo_trading/orchestrator/daily_orchestrator.py

**Class DailyOrchestrator:**
- `__init__(self) -> None` — instantiates all subsystems: ShoonyaBroker, LiveSystemState, StartupReconciler, QuestDBStore, TimeBarBuilder, FeatureComputer, EnsembleInferenceEngine, FatFingerGuard, RiskManager, OrderFSM, DualLockRateLimiter, OrderExecutor, TradeLogger, AlertSender, NewsSignalBridge, APScheduler (timezone=Asia/Kolkata)

- `run(self) -> None` — main entry point; calls _startup(), schedules jobs, blocks on scheduler; catches KeyboardInterrupt and exceptions, calls _shutdown()

- `_startup(self) -> None` — connects broker, QuestDB, resolves symbols, reconciles positions, configures ZMQ, loads inference model spec, starts feature computer subprocess, starts inference engine subprocess, validates bar type consistency

- `_schedule_daily_events(self) -> None` — APScheduler CronTrigger jobs:
  - 09:00: _on_pre_market_check()
  - 09:15: _on_session_open()
  - 09:30: _on_trading_enabled()
  - 15:15: _on_force_close() [SURV-8]
  - 15:30: _on_session_close()
  - 15:45: _on_end_of_day()

- `_on_pre_market_check(self) -> None` — checks disk space, QuestDB connectivity
- `_on_session_open(self) -> None` — subscribes WebSocket, starts alert sink subprocess, sends session-start alert
- `_on_trading_enabled(self) -> None` — sets _trading_enabled flag
- `_on_force_close(self) -> None` — [SURV-8] closes all positions at market
- `_on_session_close(self) -> None` — unsubscribes WS, flushes bars
- `_on_end_of_day(self) -> None` — logs daily P&L, sends summary alert, logs out broker

- `_on_tick(self, tick: Tick) -> None` — validates tick (TickValidator), feeds to TimeBarBuilder; if bar completes → _on_bar()
- `_on_bar(self, bar: Bar) -> None` — persists bar, publishes via ZMQ, checks news signal bridge multiplier; if paper mode → simulate fill
- `_on_signal(self, raw: bytes) -> None` — decodes ZMQ signal, runs resolve_signal() hierarchy, checks EventCalendarFilter multiplier, submits via OrderExecutor
- `_on_order_update(self, update: OrderUpdate) -> None` — passes to OrderExecutor.on_broker_update()

- `_simulate_paper_fill(self, signal: TradeSignal) -> None` — immediate LIMIT fill simulation for paper mode

- `_shutdown(self) -> None` — clears _trading_enabled, stops executor, force-closes positions, flushes QuestDB, stops scheduler, sends shutdown alert, closes store

---

### 13.2 algo_trading/orchestrator/fo_orchestrator.py

**Constants:**
- `_FO_ENTRY_CUTOFF_HOUR: int = 14`, `_FO_ENTRY_CUTOFF_MINUTE: int = 0`
- `_FO_FORCE_CLOSE_HOUR: int = 15`, `_FO_FORCE_CLOSE_MINUTE: int = 25`
- `_DELTA_REBALANCE_HOURS: list[int] = [10,11,12,13,14]`
- `_IV_RANK_SCALE: float = 100.0`, `_GARCH_LOW_VOL: float = 0.12`, `_GARCH_MID_VOL: float = 0.20`, `_GARCH_HIGH_VOL: float = 0.28`, `_GARCH_SKIP_VOL: float = 0.35`
- `_VIX_HISTORY_BARS: int = 20`, `_VIX_ZSCORE_TRIGGER: float = 2.5`

**Class FOOrchestrator:**
- `__init__(symbol="NIFTY", days_to_expiry=21) -> None` — instantiates: QuestDBStore, LiveSystemState, ShoonyaBroker, AlertSender, OrderFSM, DualLockRateLimiter, RiskManager, OrderExecutor, TimeBarBuilder, FOCostModel, PortfolioRiskEngine, FOFeatureEngine, FOSignalGenerator, DeltaHedger, GARCHVolForecaster; state: _latest_chain, _latest_fo_features, _active_strategy, _vix_history, _latest_vix, APScheduler, threading.Events, flush_thread

- `run(self) -> None` — _startup(), _schedule_daily_events(), starts scheduler, blocks on _running
- `_startup(self) -> None` — connects broker/store, resolves symbols, resets state
- `_schedule_daily_events(self) -> None` — CronTrigger IST: 09:15 open, 09:30 trading enabled, hourly 10–14 delta rebalance, 15:25 force close, 15:30 close, 15:45 EOD
- `_on_market_open(self) -> None` — clears trading_enabled, subscribes WS + option chain
- `_on_trading_enabled(self) -> None` — sets trading_enabled
- `_on_delta_rebalance_heartbeat(self) -> None` — hourly; calls _attempt_delta_hedge()
- `_on_fo_force_close(self) -> None` — unwinds delta hedge futures, closes all option positions at market
- `_on_market_close(self) -> None` — unsubscribes WS, flushes bars
- `_on_end_of_day(self) -> None` — logs P&L, alerts, logout, clears _running
- `_on_tick(self, tick) -> None`, `_on_bar(self, bar) -> None` — if flat and before cutoff: _generate_and_submit_signal(); always: _attempt_delta_hedge()
- `_on_order_update(self, update) -> None`
- `_generate_and_submit_signal(self) -> None` — FOFeaturesResult from chain → iv_rank scaled → VRPEstimate → FOSignal → GARCH lot sizing → StrategyBuilder → _submit_strategy_legs()
- `_build_strategy(self, fo_signal, builder, chain, lots=1) -> Optional[StrategyResult]`
- `_submit_strategy_legs(self, strategy) -> None`
- `_attempt_delta_hedge(self, force=False) -> None` — portfolio Greeks → needs_hedge() → compute_hedge_lots() → futures TradeSignal → executor; force=True bypasses threshold for VIX z-score spike
- `_is_within_entry_window(self) -> bool`
- `on_chain_snapshot(self, chain) -> None` — stores chain, computes VIX, calls _update_vix_history()
- `@staticmethod _compute_lots_from_garch(garch_vol) -> int` — returns 2 if garch_vol < 0.12, else 1
- `_update_vix_history(self, vix_value) -> None` — rolling window; triggers proactive hedge at z >= 2.5σ
- `@staticmethod _estimate_atm(chain) -> float`
- `_flush_loop(self) -> None` — daemon thread, flushes QuestDB every _FLUSH_INTERVAL_SEC
- `_handle_shutdown_signal(self, signum, frame) -> None`
- `_shutdown(self) -> None` — unwinds hedge, force-closes, flushes, shuts scheduler

---

### 13.3 algo_trading/orchestrator/health_check.py

**Prometheus metrics (all module-level):**
- `TICKS_PER_SECOND: Gauge(labels=["symbol"])` — live ticks/sec per symbol
- `BAR_LATENCY_MS: Histogram(buckets=[1,5,10,25,50,100,250])` — bar processing latency
- `INFERENCE_LATENCY_MS: Histogram(buckets=[1,2,5,10,20,50])` — inference latency
- `QUESTDB_WRITE_ERRORS: Counter` — cumulative QuestDB write errors
- `ORDERS_PLACED: Counter(labels=["symbol"])` — cumulative orders placed
- `ORDERS_REJECTED: Counter(labels=["reason"])` — cumulative rejections by reason
- `KILL_SWITCH_ACTIVE: Gauge` — 1=active, 0=inactive

**Function:**
- `start_metrics_server(port=9090) -> None` — starts Prometheus HTTP scrape endpoint; uses _server_started flag to prevent double-starts

---

### 13.4 health_check.py (root-level)

**Class HealthCheck (HTTP /health endpoint):**
- `__init__(questdb_host, questdb_ilp_port, zmq_heartbeat_file: Path) -> None`
- `get_status(self) -> dict` — {status: "ok"|"degraded"|"critical", checks: {...}, timestamp}
- `_check_questdb(self) -> bool` — TCP probe to ILP port
- `_check_zmq_heartbeat(self) -> bool` — file mtime < 30s
- `_check_disk_space(self) -> bool` — > 2GB free

**Function:**
- `run_health_server(port: int = 8888, once: bool = False) -> None` — HTTP GET /health; --once CLI flag for single check and exit

---

## 14. ALERTS

### 14.1 algo_trading/alerts/alert_sink.py

**Constants:**
- `_MIN_SEND_INTERVAL_SEC: float = 2.0`
- `_MAX_BACKOFF_SEC: float = 60.0`
- `_RECV_TIMEOUT_MS: int = 1000`
- `_FALLBACK_LOG_PATH: Path`

**Functions:**
- `_priority(level: str) -> int` — RISK→4, CRITICAL→3, WARNING→2, INFO→1
- `_write_fallback(payload: dict) -> None` — JSONL fallback
- `_handle_signal(signum, frame) -> None` — SIGTERM/SIGINT handler
- `run_alert_sink() -> None` — main entry point for alert sink subprocess
- `_validate_telegram_token(bot_token) -> bool` — via getMe API
- `_run_loop(port, sender, *, token_valid) -> None` — ZMQ PULL loop; priority drain (RISK first)
- `_drain_only_mode(port) -> None` — drain when Telegram disabled

**Class _TelegramSender:**
- `__init__(bot_token, chat_id) -> None`
- `_init_bot(self) -> None` — lazy-loads telegram.Bot or None
- `send(text: str) -> bool` — exponential backoff
- `_send_via_bot(text) -> bool` — python-telegram-bot async
- `_send_via_requests(text) -> bool` — HTTP fallback

---

### 14.2 algo_trading/alerts/telegram_bot.py

**Constants:**
- `_SNDHWM: int = 200`
- `_default_sender` — module-level singleton

**Class AlertSender:**
- `_COOLDOWN_SECONDS: dict[str,float]` — per-level rate limits
- `__init__(port=None) -> None` — ZMQ PUSH socket (lazy connect)
- `start(self) -> None`, `stop(self) -> None`
- `send_trade_alert(symbol, action, price, quantity, reason="") -> None`
- `send_risk_alert(message) -> None`
- `send_exception_alert(exc, context="") -> None`
- `send_session_alert(event, details="") -> None`
- `_push(payload: dict) -> None` — ZMQ NOBLOCK with per-level deduplication

**Function:**
- `send_alert(message: str, level: str = "INFO") -> None` — fire-and-forget singleton

---

END OF PART 5.
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
