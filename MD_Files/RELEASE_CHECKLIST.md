# Release Checklist — NSE AlgoTrader

Pre-release checklist that operators MUST complete before deploying a new version to live
trading. All items must be checked. Any unchecked item blocks the release.

**Version being released:** `__________`
**Release date:** `__________`
**Operator:** `__________`
**Reviewer:** `__________`

---

## 1. Code Quality Gates

- [ ] All pytest tests pass: `pytest tests/ -x` exits 0
- [ ] Lint passes: `ruff check algo_trading/ tests/ scripts/` exits 0
- [ ] Type check passes: `mypy algo_trading/` exits 0
- [ ] No P0 items open in `docs/TODO_RISK_REGISTER.md`
- [ ] All P0 items from `MD_Files/audit_improv.md` have been fixed or formally deferred with written justification
- [ ] `git diff main` reviewed by a second pair of eyes (or self-review with 24-hour gap minimum)
- [ ] Git tag created: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
- [ ] `CHANGELOG.md` updated with all risk-affecting changes

---

## 2. Paper Validation Gates

Paper validation must run for at least 30 calendar days with realistic fills (not demo paper).

- [ ] 30-day paper run completed with `TRADING_MODE=paper`
- [ ] Walk-forward Sharpe ratio **>= 1.0** (from `validate_edge.py`)
- [ ] Profit factor **>= 1.2** (net of all costs: STT, exchange, GST, stamp duty, slippage)
- [ ] Win rate **>= 40%** over the paper period
- [ ] Max drawdown **<= 15%** from peak during paper period
- [ ] Daily loss limit triggered correctly (halted at 2% loss, required manual reset)
- [ ] Kill switch tested in paper today and positions flattened correctly
- [ ] At least 5 complete trade cycles (entry → exit) without crash
- [ ] Force-close at 15:15 IST working correctly
- [ ] Auto-reconnect after WebSocket disconnect tested
- [ ] Paper session report reviewed: no unexplained signal skips or silent errors

---

## 3. Broker Certification Gates

- [ ] Dry-run order payload validated: live order constructed and logged, NOT submitted
- [ ] Strategy ID registered with broker (Shoonya): `STRATEGY_ID` set in `.env`
- [ ] `BROKER_APPROVAL_DATE` set (YYYY-MM-DD format)
- [ ] Order tagging confirmed in broker order reports (strategy tag visible in reports)
- [ ] Shoonya account: algo trading enabled and approved by broker
- [ ] Test order placed and cancelled successfully using live credentials (minimum-size test)
- [ ] Broker order book reconciliation tested: broker state matches local FSM state
- [ ] Remarks field encoding verified: `client_order_id|strategy_id` visible in broker reports

---

## 4. Model Artifact Gates

- [ ] SHAP report present at `models/trained/shap/` (not a stub JSON)
- [ ] `models/trained/sebi_report.json` status is NOT `"STUB"` — real SHAP values present
- [ ] Feature schema hash recorded in `FrozenModelSpec.feature_schema_hash`
- [ ] Feature schema at inference time matches training feature schema hash
- [ ] Treelite `.so` files compiled and present: `models/trained/xgb.so`, `lgb.so`, `cat.so`
- [ ] `FrozenModelSpec` JSON written with `version`, `trained_on`, `feature_schema_hash`
- [ ] `APPROVED_MODEL_VERSION` in `.env` matches `FrozenModelSpec.version`
- [ ] Model edge gate passed on net-of-cost backtest (Sharpe >= 1.0, profit_factor >= 1.2)
- [ ] No stub model registered: `model_registry.validate_artifacts()` passes without error

---

## 5. Compliance Gates

- [ ] `APPROVED_STATIC_IP` set and verified: `curl -s https://checkip.amazonaws.com` returns the registered IP from the VPS
- [ ] SEBI strategy ID registered with broker (separate from Shoonya account strategy tag)
- [ ] `OTR_TRACKING_ENABLED=true` in live `.env`
- [ ] `AUDIT_LOG_PATH` set, file writable, and directory permissions are `700`
- [ ] Audit log rotation configured (5-year retention — see `docs/RETENTION_POLICY.md`)
- [ ] Health endpoint bound to `127.0.0.1` only, NOT `0.0.0.0`
- [ ] No secrets appear in `logs/algo_trading.jsonl` (run log scanner)
- [ ] `QUESTDB_PG_PASSWORD` is NOT the default `"quest"`

---

## 6. Config Review

- [ ] Config checksum logged at startup (review `state/config_snapshot_*.json`)
- [ ] `.env` reviewed line-by-line for live mode settings
- [ ] `TRADING_MODE=live` confirmed
- [ ] `ENABLE_LIVE_TRADING=true` confirmed (dual opt-in required)
- [ ] `BROKER_DRY_RUN=false` (no dry-run in live)
- [ ] No test credentials in `.env` (e.g., no `TEST_` prefixed user IDs)
- [ ] `CAPITAL_INR` set to actual trading capital (not a test value)
- [ ] `MAX_DAILY_LOSS_PCT`, `MAX_POSITION_PCT`, `MAX_GROSS_EXPOSURE_PCT`, `MAX_DRAWDOWN_PCT` reviewed and correct
- [ ] `SYMBOLS` list reviewed — only symbols with adequate liquidity and ADV

---

## 7. Rollback Plan

- [ ] Previous version tagged in git: `git tag -l | grep vX.Y` shows prior tag
- [ ] Previous model artifacts backed up to `state/model_backup_YYYYMMDD/`
- [ ] Rollback command documented and tested (see `docs/ROLLBACK_PROCEDURE.md`)
- [ ] Operator knows how to stop the live service: `systemctl stop algo-trading-market-open.timer`
- [ ] Previous `.env` backed up as `.env.rollback_YYYYMMDD`

---

## 8. Kill Switch Test

- [ ] Kill switch manually triggered in paper mode **today** (same day as deployment)
- [ ] Verified: all open paper positions flattened after kill switch
- [ ] Verified: no new entry orders placed after kill switch triggered
- [ ] Verified: kill switch sentinel file written to `state/` correctly
- [ ] `KILL_SWITCH_TESTED=true` set in `.env`
- [ ] Kill switch cleared correctly before next session (sentinel file removed)

---

## Sign-Off

| Gate | Status | Signed by | Date |
|------|--------|-----------|------|
| Code Quality | PASS / FAIL | | |
| Paper Validation | PASS / FAIL | | |
| Broker Certification | PASS / FAIL | | |
| Model Artifacts | PASS / FAIL | | |
| Compliance | PASS / FAIL | | |
| Config Review | PASS / FAIL | | |
| Rollback Plan | PASS / FAIL | | |
| Kill Switch Test | PASS / FAIL | | |

**Release approved:** YES / NO

> Do NOT set `ENABLE_LIVE_TRADING=true` until all 8 gates show PASS.
