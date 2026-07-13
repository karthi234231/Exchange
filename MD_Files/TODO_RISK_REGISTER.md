# TODO Risk Register — NSE AlgoTrader

High-priority TODOs that affect live trading safety. This document is the single source
of truth for open risk items. All P0 items must be resolved before live trading. P1 items
must be resolved before extended paper validation.

Last updated: 2026-06-22 | Source: `MD_Files/audit_improv.md`

---

## Priority Legend

- **P0** — Must fix before ANY live trading. System is unsafe without this.
- **P1** — Must fix before serious paper/live validation. Known to cause wrong behavior.
- **P2** — Important reliability or safety improvement. Should be resolved in paper phase.

---

## P0 Items — All Must Pass Before Live

| ID | Description | Module | Blocks Live Trading | Status |
|----|-------------|--------|--------------------|----|
| P0-1 | Fix live broker order identity mapping: `client_order_id` must be in remarks field; WebSocket ACK/fill/cancel mapping must work | `execution/order_executor.py`, `market_data/broker_shoonya.py` | YES | Fixed (June 2026 session) |
| P0-2 | Release rate limiter state on all terminal order outcomes (FILLED, CANCELLED, REJECTED, TIMEOUT) | `execution/rate_limiter.py`, `execution/order_executor.py` | YES | Fixed (June 2026 session) |
| P0-3 | Prevent cancel request when `exchange_order_id` is missing | `execution/order_executor.py`, `execution/order_fsm.py` | YES | Fixed (June 2026 session) |
| P0-4 | Route F&O strategy legs through true multi-leg execution (`fo/execution.py`), not equity-style `TradeSignal` | `fo/orchestrator.py`, `fo/execution.py` | YES (F&O only) | Open |
| P0-5 | Fix missing `get_ltp` dependency in F&O orchestrator hedge path | `fo/orchestrator.py`, `market_data/system_state.py` | YES (F&O only) | Open |
| P0-6 | Centralize expiry rules: one expiry/calendar service; remove hardcoded Thursday weekday constants | `fo/expiry.py`, `signals/event_filter.py`, tests | YES (F&O only) | Open |
| P0-7 | Build effective-dated exchange session calendar with segment-specific sessions | `market_calendar.py`, all modules using hardcoded times | YES | Open — multiple hardcoded session times (15:25, 15:30, 15:40) found across modules |
| P0-8 | Repair realized/unrealized P&L accounting in `LiveSystemState.update_position_from_fill` | `market_data/system_state.py` | YES — kill switch may fire on wrong P&L | Open |
| P0-9 | Use net P&L (after all costs) for backtest edge gate | `backtest/runner.py`, `scripts/validate_edge.py` | YES — strategy may appear profitable but lose money after costs | Open |
| P0-10 | Block model registration when training pipeline uses stubs (ensemble stub, Treelite stub, SHAP stub) | `scripts/train_models.py`, `models/model_registry.py` | YES — fake model artifacts can reach production | Partially fixed — `validate_artifacts()` checks stub SHAP status; ensemble stub itself still unresolved |
| P0-11 | Standardize ML class label mapping: 0=bear, 1=sideways, 2=bull across ensemble, tuner, explainer, inference | `models/ensemble.py`, `models/tuner.py`, `models/explainer.py` | YES — strategy may trade wrong direction | Open |
| P0-12 | Fix live ML feature availability: frozen feature schema, validate stateful models are fitted at startup | `features/engine.py`, `features/statistical.py`, `models/model_registry.py` | YES — live signals may silently disappear | Open |
| P0-13 | Convert cumulative tick volume to incremental volume before bar builders and microstructure features | `market_data/` tick processing, `features/` | YES — VWAP, OBI, volume bars all wrong with cumulative volume | Open |
| P0-14 | Strengthen live trading compliance gate: add hard startup checks for all 9 gates | `algo_trading/config.py` | YES — system may enter live mode without required controls | Partially addressed — config validator exists but some checks incomplete |
| P0-15 | Protect health endpoint: bind to `127.0.0.1` only, not `0.0.0.0` | `health_check.py` | YES — information leakage risk | Open |

---

## P1 Items — Required Before Serious Paper Validation

| ID | Description | Module | Status |
|----|-------------|--------|--------|
| P1-16 | Fix opening range breakout timing: anchor ORB to exchange calendar timestamps, not first 2 bars seen | `signals/` | Open |
| P1-17 | Align paper trading fill simulation with realistic execution (spread, impact, partial fills) | `backtest/runner.py`, paper mode | Open |
| P1-18 | Wire corporate actions into training pipeline (splits/bonuses corrupt historical returns) | `scripts/train_models.py` | Partially addressed — `CorporateActionsAdjuster` called but import may fail silently |
| P1-19 | Fix first bar volume after startup/reconnect (inflated cumulative volume) | `market_data/` | Open |
| P1-20 | Make storage writes durable on flush failure (retry queue / dead-letter) | `market_data/candle_store.py` | Open |
| P1-21 | Make alerting durable for critical events (write to audit log before Telegram) | `alerts/alert_sink.py` | Open |
| P1-22 | Fix RiskManager daily reset and kill-switch reset semantics | `risk/manager.py`, `market_data/system_state.py` | Open |
| P1-23 | Apply portfolio correlation multiplier to actual position sizing | `risk/manager.py` | Open |
| P1-24 | Make fat-finger guard fail closed on internal errors | `execution/order_executor.py`, `risk/manager.py` | Open |
| P1-25 | Remove SQL interpolation risk in ADV/data queries (whitelist symbol format) | `market_data/candle_store.py` or similar | Open |
| P1-26 | Align cost model slippage policy (remove static fallback, use dynamic ATR-based) | `risk/cost_model.py` | Open |
| P1-27 | Resolve F&O margin/headroom enforcement path | `fo/execution.py` | Open (F&O) |
| P1-28 | Reconcile broker/order book on reconnect: fetch order book before resuming | `execution/order_executor.py`, `market_data/system_state.py` | Open |
| P1-29 | Add production-safe manual kill switch (CLI, file flag, Telegram command) | `risk/manager.py` | Partially addressed — sentinel file approach exists |
| P1-30 | Fix dependency and Python version drift (pin 3.11, lockfile, include all imports) | `requirements.txt`, `pyproject.toml` | Open |
| P1-31 | Fix Makefile shell redirection (quote pip specs like `"scipy>=1.12"`) | `Makefile` | Open |
| P1-32 | Fix Docker health check (must not require broker secrets for liveness) | `Dockerfile`, `health_check.py` | Open |
| P1-33 | Add persistent volumes for runtime state in docker-compose | `docker-compose.yml` | Open |
| P1-34 | Split liveness, readiness, and trading readiness health checks | `health_check.py` | Open |
| P1-35 | Fix RL environment dimension drift (derive from frozen feature schema) | `models/rl_env.py` | Open (Stage 4) |
| P1-36 | Fix stage feature schema drift across Stage 4, ML, RL, feature engine | `features/stage_router.py`, `models/` | Open |

---

## P2 Safety Items — Resolve During Paper Validation

The following P2 items have safety implications and should be addressed during the 30-day
paper validation window. They do not block live trading but increase operational risk.

| ID | Description | Module |
|----|-------------|--------|
| P2-52 | Add order event audit trail independent of QuestDB (local immutable file first) | `execution/order_executor.py` |
| P2-54 | Add dead-man switch for broker WebSocket silence (cancel entries on stale data) | `market_data/` |
| P2-56 | Add config checksum to session logs at startup | `orchestrator/daily_orchestrator.py` |
| P2-59 | Add dry-run live broker mode to validate order payloads without submission | `market_data/broker_shoonya.py` |
| P2-60 | Add replay tests for order lifecycle (ACK-before-REST, fill-before-ACK, cancel+fill race) | `tests/` |
| P2-86 | Add clock sync check at startup (block trading if NTP drift > threshold) | `orchestrator/daily_orchestrator.py` |
| P2-99 | Add WebSocket duplicate/out-of-order update handling | `execution/order_fsm.py` |
| P2-100 | Add fill deduplication (use broker trade id where available) | `execution/order_executor.py` |
| P2-126 | Add position reconciliation tolerance policy (hard mismatches vs price rounding) | `market_data/system_state.py` |

---

## Item Lifecycle

Items move through these states:

1. **Open** — Not yet addressed
2. **In Progress** — Being worked on
3. **Fixed** — Implementation complete, tests added
4. **Verified** — Fixed and confirmed working in paper session
5. **Deferred** — Formally deferred with written justification (P2 only)

P0 and P1 items may NOT be marked Deferred without written justification and a
target resolution date.

---

## How to Close a P0 Item

1. Implement the fix in code
2. Add a test that exercises the fixed behavior
3. Run `pytest tests/ -x` — must pass
4. Update this document: change status to `Fixed` with date
5. Add entry to `CHANGELOG.md` under `### Fixed (Critical / Risk-Affecting)`
6. Have the change reviewed (or self-review with 24-hour gap)

A P0 item is not closed until it has a passing test. "I fixed it in code" without a test
does not count as closed.
