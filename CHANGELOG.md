# Changelog — NSE AlgoTrader

All notable changes to this system are documented here. Risk-affecting changes are
labelled `[RISK]`. Changes that affect compliance are labelled `[COMPLIANCE]`.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-06-22

### Fixed (Critical / Risk-Affecting)

These fixes were confirmed applied during the exhaustive 3-session bug audit
(June 6–22, 2026). All were blocking live trading before this session.

- **[P0-FIX-1] Order identity mapping** — `broker_shoonya.py` now encodes `client_order_id` (not just `strategy_id`) in the `remarks` field. WebSocket ACK/fill/cancel callbacks can now map to internal orders correctly. Previously, fills and rejections could map to wrong orders, corrupting position state, rate limiter slots, and P&L.

- **[P0-FIX-2] Rate limiter terminal release** — Rate limiter state is now released on all terminal order outcomes: FILLED, CANCELLED, REJECTED, TIMEOUT. Previously, rejected and cancelled orders could leak modification counters, causing later orders to be incorrectly blocked.

- **[P0-FIX-3] Cancel without exchange_order_id** — `cancel_order()` now validates that `exchange_order_id` is present before transitioning to CANCEL_PENDING. Previously, a cancel attempt without a confirmed exchange ID could leave the FSM stuck in CANCEL_PENDING.

- **[P0-FIX-FSM-99] FSM idempotency** — Fill-before-ACK and duplicate fill events are now handled correctly. `(FILLED, PARTIAL_FILL)` added to idempotent set. `(SENT, FILL)` and `(SENT, PARTIAL_FILL)` added to valid transitions. CANCELLED+FILL handled as a late-fill callback without state change.

- **[P0-FIX-GARCH] GARCH annualization factor** — `features/statistical.py`: corrected annualization from `252×6.25` to `252×25`. The previous value caused a 2× underestimate of volatility forecasts.

- **[P0-FIX-STARTUP] structlog processors crash** — `logging_setup.py`: `processor=` keyword corrected to `processors=[]`. The previous code would crash on startup before any trading could begin.

- **[P0-FIX-ATRNULL] Triple-barrier ATR null** — `labeling/triple_barrier.py`: guarded against Polars null ATR value becoming Python `None` and causing `TypeError` in `max()`. Labeling would silently fail for the first bars of any series.

- **[P0-FIX-GATE] Edge gate enforcement** — `scripts/validate_edge.py`: Sharpe, profit_factor, and win_rate gate criteria are now actually enforced (raised `AssertionError` on failure). Previously, the gate checks logged results but never blocked a bad strategy from being promoted.

- **[P0-FIX-MARKET-BOUNDARY] Model registry market hours boundary** — `models/model_registry.py`: corrected `<= close_minutes` to `< close_minutes` in `_is_market_open`. The previous boundary allowed retraining at exactly 15:30:00 IST.

- **[P0-FIX-HOLD-SIZE] Signal hierarchy HOLD size** — `execution/signal_hierarchy.py`: HOLD action now correctly returns `size_multiplier=0.0`. Previously, HOLD could receive a non-zero multiplier from ADWIN drift reduction, causing a position to be sized when no trade should occur.

- **[P0-FIX-KYLE-LAMBDA] Kyle lambda ddof** — `features/microstructure.py`: `np.var(ddof=0)` and `np.cov(ddof=1)` mismatch corrected to use `ddof=1` consistently. The previous mismatch produced incorrect price impact estimates.

- **[P0-FIX-CUMVOL] Bar volume cumulative → incremental** — `market_data/candle_store.py`: bar volume now correctly tracks `bar_start_volume` per symbol and uses the delta (current − previous) rather than the cumulative day total. VWAP, OBI, and volume features were materially wrong before this fix.

- **[P0-FIX-RECONCILE-LOCK] Startup reconcile thread safety** — `market_data/system_state.py`: `startup_reconcile_clean = True` assignment moved inside `with self._state._lock` to prevent race condition during startup reconciliation.

- **[P0-FIX-BACKTEST-ATR] Backtest ATR falsy check** — `backtest/runner.py`: `if atrs[i]` (treats 0.0 as missing) corrected to `if atrs[i] is not None`. ATR=0.0 is a valid (degenerate) value; the old check caused it to be treated as missing data.

- **[P0-FIX-REVERSAL] P&L reversal logic** — `market_data/system_state.py`: position crossing zero (long-to-short or short-to-long reversal) now uses correct `crossed_zero` check instead of dead-code `pos.net_quantity == 0` path. Reversals were previously mis-accounting realized P&L.

### Fixed (High Priority)

- **[P1-FIX-PAPER-DATE] Paper trade IST date** — `scripts/paper_trade.py`: `date.today()` replaced with `_ist_now().date()` for session date calculation. On a UTC VPS, `date.today()` returns UTC date which is 5.5 hours behind IST — this caused wrong session dates after midnight IST.

- **[P1-FIX-LATENCY-ILP] QuestDB ILP timestamp wrapper** — `execution/latency_profiler.py`: QuestDB ILP `at=` now uses `TimestampNanos()` wrapper where available (imported with `try/except`). Raw integer nanosecond values were being rejected by some QuestDB versions.

- **[P1-FIX-ALERTSINK-HWM] AlertSink RCVHWM** — `alerts/alert_sink.py`: `RCVHWM` socket option added to `_drain_only_mode`. Without this, the drain mode socket had unbounded receive queue.

- **[P1-FIX-TRUEDATA-DEAD] TrueData dead assignment** — `market_data/truedata_feed.py`: dead `_reconnect_delay` assignment removed. This was a remnant that could shadow the correct reconnect delay value.

- **[P1-FIX-STAGEROUTER] Stage router redundant checks** — `features/stage_router.py`: duplicate validation check, unnecessary `object.__setattr__` on non-frozen dataclass, and redundant NaN check removed. These caused incorrect validation behavior and spurious attribute errors.

- **[P1-FIX-INFERENCE-HASH] Inference SHA-256 hot loop** — `models/inference_engine.py`: SHA-256 feature schema hash cached at init rather than recomputed per inference call. Previously added ~0.5ms latency per inference in the hot path.

- **[P1-FIX-EXPLAINER-FIG] SHAP figure dead code** — `models/explainer.py`: dead `fig, ax = plt.subplots()` before SHAP's own figure creation removed. This caused a matplotlib figure leak on every SHAP analysis run.

- **[P1-FIX-DOWNLOAD-RSTRIP] Download history interval parsing** — `scripts/download_history.py`: `rstrip("mhd")` ambiguity fixed — "1h" and "1d" both produced "1" with rstrip. Now uses the full interval string for correct parsing.

- **[P1-FIX-F0CHAIN-DATETIME] F&O chain naive datetime** — `market_data/fo_chain.py` (or `data/fo_chain.py`): `datetime.now()` replaced with timezone-aware IST datetime. Also added T=0 and sigma=0 guard in `_d1d2()` to prevent division by zero.

- **[P1-FIX-CUMULQTY] Fill deduplication and cumulative→delta** — `market_data/broker_shoonya.py`: `_cumulative_filled` dict tracks cumulative filled quantity per order; per-fill delta computed before updating position. Prevents double-counting on partial fill callbacks.

- **[P1-FIX-SYNTAX] Syntax errors in sentiment and TFI** — `models/sentiment.py` and `features/tfi.py`: `from __future__` annotations import moved to top of file. Both files had `SyntaxError` which prevented import.

### Added

- **[NEW] Fat-finger guard** — `risk/fat_finger.py`: `FatFingerGuard` class with value cap (INR 2L), ADV% cap (2%), price deviation cap (50 bps), and quantity cap (5000). Wired as first check in `submit_signal()` before any broker interaction.

- **[NEW] Portfolio correlation guard** — `risk/manager.py`: `PortfolioCorrelationGuard` added using rolling 60-bar Pearson correlation. Wired as check #8 in `can_trade()`.

- **[NEW] Filtered EMA cross strategy** — `signals/ema_cross_filtered.py`: `FilteredEMACrossStrategy` with ADX filter (Wilder smoothing, O(1)), volume filter, and relative strength filter. Reduces noise trades compared to plain EMA cross.

- **[NEW] Opening range breakout strategy** — `signals/opening_range_breakout.py`: ORB strategy with 2-bar opening range, volume confirmation (1.5×), and 12:00 cutoff.

- **[NEW] Event calendar filter** — `signals/event_filter.py`: `EventCalendarFilter` with expiry-day, VIX spike, and earnings blackout position size multipliers.

- **[NEW] Order audit trail** — `execution/order_executor.py`: `_OrderAuditWriter` writes JSONL to `state/order_audit.log` on all FSM transitions, before QuestDB. Independent of QuestDB availability. [COMPLIANCE]

- **[NEW] EOD reconciliation** — `orchestrator/daily_orchestrator.py`: `_eod_reconcile()` compares local FSM state vs broker positions and order book before session logout.

- **[NEW] Dead-man WebSocket switch** — `market_data/websocket_feed.py`: `StaleTickGuard` class cancels resting entry orders when no ticks received for more than 120 seconds.

- **[NEW] Config checksum at startup** — `orchestrator/daily_orchestrator.py`: SHA-256 of sanitized config logged and written to `state/config_snapshot_*.json` at every session start. [COMPLIANCE]

- **[NEW] NTP clock sync check** — `orchestrator/daily_orchestrator.py`: NTP drift check at startup (pool.ntp.org). Blocks trading if drift > 30 seconds; warns if > 5 seconds.

- **[NEW] Static IP verification** — `execution/order_executor.py` or config startup: external IP checked against `APPROVED_STATIC_IP` at startup in live mode. Blocks if mismatch. [COMPLIANCE]

- **[NEW] OTR tracking** — `execution/rate_limiter.py`: daily orders/trades counters in `DualLockRateLimiter`; alert at OTR threshold 50.0. [COMPLIANCE]

- **[NEW] Max open orders guard** — `execution/order_executor.py`: global cap 20 open orders, per-symbol cap 3. Entries blocked when cap reached.

- **[NEW] Daily notional cap** — `execution/order_executor.py`: 300% of `capital_inr` daily turnover cap tracked per session.

- **[NEW] Duplicate signal / pending order conflict check** — `execution/order_executor.py`: checks for duplicate signals and pending orders for same symbol/side before placement.

- **[NEW] decision_id correlation ID** — `execution/signal_hierarchy.py`, `execution/order_executor.py`, `execution/order_fsm.py`: `decision_id` UUID added to `TradeSignal`, `Order`, all FSM log messages, all audit log writes, and fill logs.

- **[NEW] Daily session report** — `orchestrator/daily_orchestrator.py`: `_generate_session_report()` writes JSON summary and sends Telegram. Includes trade count, P&L, rejected signals, data quality events, reconciliation status.

- **[NEW] Startup warmup policy** — `features/engine.py` (or orchestrator): `_bar_count` dict gates signals until minimum bars received (`min_bars_before_signal`).

- **[NEW] Late-start policy** — `orchestrator/`: detects if starting after 09:30, logs warning, warmup counter handles delayed readiness.

- **[NEW] Exception policy** — `execution/exception_policy.py`: `fail_closed`, `fail_open`, `safe_exit_action` decorators with documented policy per component type.

- **[NEW] Comprehensive Shoonya status parsing** — `market_data/broker_shoonya.py`: `_STATUS_TO_EVENT` map handles AMO, Trigger, Modified statuses in addition to standard statuses. Alerts on unknown status values.

- **[NEW] Order price tick-size rounding** — `execution/order_executor.py`: `_round_to_tick_size()` applied before fat-finger check and Order creation. Prevents broker rejections for off-tick prices.

- **[NEW] Order lifecycle replay tests** — `tests/test_order_lifecycle_replay.py`: 47 tests, 10 scenarios including ACK-in-flight, fill-before-ACK, reject-after-REST, cancel/fill race, duplicate fill, partial-then-cancel, late-fill-after-cancel, WS reconnect.

- **[NEW] Import smoke tests** — `tests/test_imports.py`: 52 passing, 27 optional-skipped. Covers all `algo_trading` modules. Ensures no import breaks on startup.

- **[NEW] P&L ledger property tests** — `tests/test_pnl_ledger.py`: 29 Hypothesis property tests for long add, short add, partial close, full close, reversal scenarios.

- **[NEW] Timezone consistency tests** — `tests/test_timezone_consistency.py`: IST offset, date boundaries, domain type timezone-awareness.

- **[NEW] CI pipeline** — `.github/workflows/ci.yml`: AST parse, ruff, mypy, pytest, safety check.

- **[NEW] F&O package** — `algo_trading/fo/`: 10 files, full F&O infrastructure including BSM pricing, SPAN margin approximation, multi-leg state machine, portfolio Greeks, SVI volatility surface.

- **[NEW] OOF ensemble stacking** — `models/ensemble.py`: fold[-1] holdout fixed, calibration via `CalibratedClassifierCV` (isotonic), `RegimeConditionedMeta` for regime-specific weighting.

- **[NEW] Model stability report** — `scripts/model_stability_report.py`: per-fold Sharpe, calibration curves, ECE, confusion matrices.

- **[NEW] SHAP feature selection** — `scripts/select_features.py`: Spearman rank stability filter for feature selection.

- **[NEW] Paper trading analysis** — `scripts/paper_trading_analysis.py`: JSONL paper log → Sharpe/PF/WR/MaxDD/fill-rate/time-of-day/symbol breakdown.

- **[NEW] Cost model calibration** — `scripts/calibrate_cost_model.py`: compares intended vs actual fill prices, calibrates slippage model coefficients from paper trading data.

- **[NEW] Label distribution check** — `scripts/check_label_distribution.py`: triple-barrier label distribution audit with warning if any class < 15%.

- **[NEW] Data manifest builder** — `scripts/build_data_manifest.py`: generates `data/MANIFEST.json` with source, date range, missing bars, corporate actions applied flag.

- **[NEW] Corporate actions adjuster** — `market_data/corporate_actions.py` (or `data/corporate_actions.py`): `CorporateActionsAdjuster` using NSEPython + CSV fallback for SPLIT/BONUS/RIGHTS/DIV adjustments.

- **[NEW] Health endpoint** — `health_check.py`: stdlib-only HTTP health endpoint (`GET /health`) returning JSON with questdb/zmq/pnl/kill-switch status.

- **[NEW] systemd service files** — `MD_Files/systemd/`: `algo-trading-main.service`, `alert-sink.service`, `inference.service`, `market-open.timer`, `market-open.service`.

- **[NEW] No-modify order policy** — `execution/order_executor.py`: `modify_order()` raises `NotImplementedError` with documented rationale (OTR parity, cancel-only policy).

### Changed

- `README.md` — Full rewrite with quickstart, architecture diagram, risk rules, live trading gates, known limitations.
- `.env.example` — Regenerated from config schema; all config keys present; port corrected to `QUESTDB_ILP_PORT=9000` (HTTP ILP, not legacy TCP 9009).
- `arch-spec.json` — Updated to QuestDB 8.0.4, corrected ZMQ ports, TabPFN removed from ML options.
- `systemd/` service files — `User=root` corrected to `User=trading`. Companion `Wants=` added for service ordering.
- `pyproject.toml` — ruff (E/F/W/I/UP/B/C4/SIM), mypy (disallow_untyped_defs), optional extras (`paper`, `live`, `ml`, `research`, `fo`).

### Notes

- **F&O 15:40 session extension**: Multiple modules reference 15:40 IST as the new F&O close time. This is based on a secondary news source (Economic Times, June 2026) [ASSUMED]. Verify with official NSE circular before using in production F&O rules.

- **NSE Tuesday expiry**: `fo/expiry.py` implements Tuesday expiry rules per NSE circular FAOP68747. Tests and docs in signals/ that reference Thursday expiry are out of date and should be updated before F&O goes live.

- **Shoonya remarks field length**: The encoding of `client_order_id|strategy_id` in the remarks field assumes the field supports sufficient length. This has NOT been formally verified with Shoonya. Verify field length limit before live trading. [ASSUMED]

- **P0 items 4–15 from audit**: Items P0-4 (F&O multi-leg routing), P0-5 (F&O get_ltp), P0-6 (expiry centralization), P0-7 (session calendar), P0-8 (P&L ledger), P0-9 (net P&L edge gate), P0-11 (ML label mapping), P0-12 (feature schema), P0-13 (cumulative volume — partially fixed in candle_store), P0-14 (compliance gate completeness), P0-15 (health endpoint binding) remain open. See `docs/TODO_RISK_REGISTER.md`.
