# Incident Runbook — NSE AlgoTrader

Operational runbooks for common live trading incidents. Each entry follows the format:
**Detection → Severity → Immediate Response → Recovery Steps → Post-Incident**

Keep this document open during live sessions.

---

## RB-1 — Broker WebSocket Disconnect

### Detection
- Log message: `broker.websocket.disconnected` or `shoonya.ws.error`
- No new ticks for more than 30 seconds during market hours (09:15–15:30 IST)
- Telegram alert: `[CRITICAL] WebSocket disconnect`
- `state/live_state.json` heartbeat stale by more than 30 seconds

### Severity
**HIGH** — Trading halted automatically. No new signals. Existing positions unprotected if stop orders are app-managed.

### Immediate Response

1. Check supervisor/WebSocket reconnect thread status:
   ```bash
   tail -f logs/algo_trading.jsonl | python -m json.tool | grep -i "reconnect\|websocket"
   ```

2. The system should auto-reconnect within 10–30 seconds. If it does not:
   ```bash
   sudo systemctl restart algo-trading-orchestrator.service
   ```

3. If resting entry orders are present, cancel them manually via Shoonya portal or:
   - Entry orders should be cancelled automatically by the dead-man switch (data stale > 120s)
   - Verify cancellations in broker order book

4. Do NOT submit new entry orders until WebSocket is confirmed reconnected and ticks are flowing.

### Recovery Steps

1. Confirm reconnect in logs: `broker.websocket.connected` event
2. After reconnect, `StartupReconciler` runs automatically — wait for `reconcile.ok` log
3. If reconciliation finds a mismatch: halt entries until operator confirms state (see RB-6)
4. Verify tick flow resumed: check `state/live_state.json` heartbeat is updating
5. Allow 2 full bars (30 minutes) of clean data before resuming entry signals

### Post-Incident

- [ ] Log disconnect duration in `state/incident_log.md`
- [ ] Check if any orders were sent during the disconnect (zombie order check)
- [ ] Review broker reconnect behavior — was it automatic or manual?
- [ ] If disconnect lasted > 30 minutes during market hours: treat as a lost session

---

## RB-2 — Order Stuck in SENT State

### Detection
- Log message: `order.state.timeout` or `fsm.sent_timeout`
- Order ID in `SENT` state for more than 10 seconds without ACK
- `state/live_state.json` shows order with `state: "SENT"` and stale `state_updated_at`

### Severity
**HIGH** — Broker may have the order and be processing it, OR the submission may have silently failed. Position state is uncertain.

### Immediate Response

1. Do NOT submit another order for the same symbol/side until resolved.

2. Check broker order book manually (Shoonya portal or API):
   ```bash
   # If broker API is accessible:
   # GET order book and look for the order by client_order_id
   ```

3. If the order is NOT in broker order book:
   - Order did not reach broker. Local state shows SENT but broker has nothing.
   - Transition order to TIMEOUT in local state (manual override if needed).
   - Clear the active order from `state/live_state.json`.

4. If the order IS in broker order book:
   - Wait for broker WebSocket ACK (may be delayed).
   - If ACK does not arrive within 30 seconds: cancel via broker portal.

### Recovery Steps

1. Cancel the stale order via broker portal if it is in OPEN/PENDING state.
2. Wait for cancel confirmation (CANCELLED status in broker order book).
3. Perform reconciliation: compare local state to broker state.
4. If position was partially filled before cancellation: update `live_state.json` accordingly.
5. Resume trading only after reconciliation is clean.

### Post-Incident

- [ ] Record the order ID, symbol, and timeline in `state/incident_log.md`
- [ ] Investigate: was this a network issue, a broker API timeout, or a code bug?
- [ ] Review `logs/algo_trading.jsonl` for the order submission event and broker response

---

## RB-3 — Kill Switch Triggered

### Detection
- Log message: `risk.kill_switch.triggered`
- Telegram alert: `[CRITICAL] Kill switch activated`
- `state/kill_switch.sentinel` file exists
- No new orders being placed

### Severity
**CRITICAL** — All trading halted. Existing positions are being flattened.

### Immediate Response

1. Do NOT clear the kill switch immediately. First understand why it triggered.

2. Check which limit was breached:
   ```bash
   tail -200 logs/algo_trading.jsonl | python -m json.tool | grep -i "kill\|limit\|drawdown\|loss"
   ```

   Common triggers:
   - `daily_loss_limit_hit`: daily P&L fell below -2% of capital
   - `drawdown_limit_hit`: drawdown from peak exceeded 10%
   - `manual_kill`: operator triggered via sentinel file
   - `data_quality_halt`: too many bad ticks

3. Verify all positions are flat:
   ```bash
   cat state/live_state.json | python -m json.tool | grep -A 20 '"positions"'
   ```

4. Verify all open orders are cancelled:
   ```bash
   cat state/live_state.json | python -m json.tool | grep -A 20 '"pending_orders"'
   ```

### Recovery Steps

1. Review the P&L and positions that triggered the kill switch.
2. If data_quality triggered it: investigate the bad tick source before resuming.
3. If daily loss limit triggered it: that day's trading is OVER. Do NOT attempt to trade again today.
4. Clear kill switch only after reviewing and understanding the trigger:
   ```bash
   rm -f state/kill_switch.sentinel
   ```
5. For daily loss trigger: kill switch resets automatically at next session start.
   Do NOT manually clear until the next trading day.

### Post-Incident

- [ ] Document trigger reason in `state/incident_log.md`
- [ ] If triggered 2 days in a row: pause live trading and investigate strategy edge
- [ ] If triggered 3 consecutive days: mandatory rollback review (see `docs/ROLLBACK_PROCEDURE.md`)

---

## RB-4 — QuestDB Down

### Detection
- Log message: `questdb.flush.error` or `storage.write_failed`
- Dead-letter file being written: `data/dead_letter.jsonl` is growing
- QuestDB HTTP endpoint not responding: `curl -s http://localhost:9000/status` fails

### Severity
**MEDIUM** — Trading continues. Data is accumulating in the dead-letter file. Audit log completeness is at risk.

### Immediate Response

1. Confirm QuestDB is down:
   ```bash
   curl -s http://localhost:9000/status
   docker ps | grep questdb
   ```

2. Check if order audit log is still writing (this is independent of QuestDB):
   ```bash
   tail -5 state/order_audit.log
   ```
   If audit log is writing: compliance data is safe. Trading can continue.

3. Do NOT halt trading solely because QuestDB is down (data writes go to dead-letter file).

4. Restart QuestDB:
   ```bash
   docker restart questdb
   # OR with compose:
   docker compose restart questdb
   ```

5. Monitor restart:
   ```bash
   docker logs questdb --tail 50
   curl -s http://localhost:9000/status
   ```

### Recovery Steps

1. After QuestDB restarts, the dead-letter file will be replayed automatically on next flush cycle.
2. Verify dead-letter replay:
   ```bash
   wc -l data/dead_letter.jsonl  # should decrease as replay proceeds
   ```
3. If QuestDB cannot restart within 15 minutes: trading may continue but the session report will be incomplete.
4. Alert operator to manually replay dead-letter file after session.

### Post-Incident

- [ ] Review QuestDB logs for crash reason: `docker logs questdb`
- [ ] Check disk space: `df -h` — QuestDB WAL can fill disk silently
- [ ] Backup dead-letter file contents before replay: `cp data/dead_letter.jsonl data/dead_letter_YYYYMMDD.jsonl.bak`
- [ ] Verify all data was replayed: compare trade count from broker order book vs QuestDB

---

## RB-5 — Data Feed Stale (No Ticks > 120 Seconds)

### Detection
- Log message: `data_quality.stale_feed` or `tick_monitor.stale`
- No tick received for any symbol for more than 120 seconds during market hours
- Telegram alert: `[HIGH] Data feed stale — no ticks for 120s`

### Severity
**HIGH** — Trading paused on affected symbols. Existing positions unprotected.

### Immediate Response

1. Cancel all resting entry orders immediately (automatic if dead-man switch is active):
   ```bash
   # Verify entry orders cancelled in logs:
   grep "stale_feed\|entry_cancel" logs/algo_trading.jsonl | tail -20
   ```

2. Maintain existing exit orders (stop-loss orders should NOT be cancelled during stale feed).

3. Do NOT submit new entry orders.

4. Alert operator via Telegram (manual if system Telegram is also affected):
   - Message: `[HIGH] Data feed stale. All entries cancelled. Exits maintained. Monitoring reconnect.`

5. Check WebSocket connection status:
   ```bash
   tail -50 logs/algo_trading.jsonl | python -m json.tool | grep -i "websocket\|tick\|feed"
   ```

### Recovery Steps

1. Wait for automatic WebSocket reconnect (typically 10–30 seconds).
2. After reconnect, verify ticks are flowing again:
   ```bash
   tail -f logs/algo_trading.jsonl | python -m json.tool | grep "tick_received"
   ```
3. Allow 2 full 15-minute bars of clean tick data before resuming entries.
4. If feed does not recover within 10 minutes: restart the orchestrator:
   ```bash
   sudo systemctl restart algo-trading-orchestrator.service
   ```

### Post-Incident

- [ ] Record stale period duration in `state/incident_log.md`
- [ ] Were any positions affected during the stale period? Check broker for fills.
- [ ] If stale > 30 minutes during market hours: treat as lost session; run EOD reconciliation

---

## RB-6 — Wrong or Unknown Broker Position

### Detection
- `StartupReconciler` logs: `reconcile.mismatch` on startup
- Log message: `reconcile.unknown_position` — position in broker book not in local state
- Broker position quantity does not match `state/live_state.json`

### Severity
**CRITICAL** — System integrity is uncertain. Do NOT allow automated trading with unknown exposure.

### Immediate Response

1. **Halt all new entry orders immediately.** This is automatic when reconciliation detects a mismatch.

2. Alert operator:
   ```bash
   # Check reconciliation mismatch details
   grep "reconcile\|mismatch\|unknown_position" logs/algo_trading.jsonl | tail -30 | python -m json.tool
   ```

3. Do NOT auto-liquidate any position you do not understand. An unknown position may be a legitimate position from a prior session that was not saved to state.

4. Manually review broker position book (Shoonya portal):
   - What symbol is it?
   - What quantity and side?
   - When was it opened?

### Recovery Steps

1. Compare broker position book with `state/live_state.json` line by line.

2. For each discrepancy, one of these is true:
   - **Local state has position; broker does not** → position was likely closed by broker (auto-square, broker error). Update local state to flat.
   - **Broker has position; local state does not** → unknown position. Do NOT auto-liquidate.
   - **Both have position but different quantities** → partial fill not recorded. Update local state.

3. For unknown positions (broker has, local state does not):
   - Decide manually: is this a position I want to hold or close?
   - If closing: close manually via Shoonya portal. Record the manual close in `state/incident_log.md`.
   - Then update `state/live_state.json` to reflect the correct state.

4. After manual reconciliation, create a clean reconcile:
   ```bash
   # Edit live_state.json to reflect actual broker state (manual step)
   # Then remove the reconcile halt sentinel if one was created
   rm -f state/reconcile_halt.sentinel
   ```

5. Run one full paper session to confirm state is clean before resuming live.

### Post-Incident

- [ ] Document every position discrepancy in `state/incident_log.md`
- [ ] Investigate root cause: was it a reconnect that missed a fill? A crash during order placement?
- [ ] Add a test case for the specific failure mode if the root cause is found

---

## RB-7 — Model Artifact Missing

### Detection
- Startup log: `model_registry.artifact_missing` or `inference_engine.load_failed`
- `FrozenModelSpec` validation fails: missing `.so` file, missing SHAP report, or stub SHAP status
- Log: `models/trained/sebi_report.json status == STUB`

### Severity
**CRITICAL** — Live trading with an incomplete or missing model is blocked. The system should refuse to start in live mode.

### Immediate Response

1. **Do NOT override the artifact check.** The system is protecting you from trading with an unvalidated model.

2. Identify which artifact is missing:
   ```bash
   python -c "
   from pathlib import Path
   artifacts = ['xgb.so', 'lgb.so', 'cat.so', 'frozen_model_spec.json', 'sebi_report.json']
   for a in artifacts:
       p = Path('models/trained') / a
       print(f'{a}: {\"OK\" if p.exists() and p.stat().st_size > 0 else \"MISSING/EMPTY\"}')"
   ```

3. Check if SHAP report is a stub:
   ```bash
   python -c "
   import json
   r = json.load(open('models/trained/sebi_report.json'))
   print('Status:', r.get('status'))"
   ```

### Recovery Steps

1. **Roll back to the last known-good model version** (see `docs/ROLLBACK_PROCEDURE.md`):
   ```bash
   cp -r state/model_backup_YYYYMMDD/* models/trained/
   ```

2. Verify rolled-back artifacts:
   ```bash
   ls -la models/trained/
   # Verify sizes are non-zero
   # Verify sebi_report.json status != STUB
   ```

3. Update `APPROVED_MODEL_VERSION` in `.env` to match the rolled-back version.

4. **Do NOT trade in live mode until real model artifacts are present.** Paper mode can be used with a stub model for pipeline testing only.

5. Rebuild the model artifacts by running the full training pipeline on a non-trading day (Sunday):
   ```bash
   python -m algo_trading.scripts.train_models --symbols RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK
   ```

6. Verify the resulting SHAP report is NOT a stub before deploying.

### Post-Incident

- [ ] Document which artifact was missing and why
- [ ] Add artifact validation to the pre-session health check
- [ ] Schedule model retraining for the next Sunday and verify artifacts before next week's live session

---

## Incident Log Template

Append to `state/incident_log.md` after every incident:

```markdown
## INCIDENT — YYYY-MM-DD HH:MM IST

- **Runbook:** RB-N (incident type)
- **Severity:** CRITICAL / HIGH / MEDIUM
- **Detection time:** HH:MM IST
- **Resolution time:** HH:MM IST
- **Total duration:** X minutes
- **P&L impact:** approx INR X
- **Positions affected:** (list or "none")
- **Root cause:** (brief description)
- **Action taken:** (what was done)
- **Follow-up required:** (Y/N — what and by when)
```
