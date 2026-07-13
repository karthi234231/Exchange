# Rollback Procedure — NSE AlgoTrader

Step-by-step procedure for reverting to a previous working version during or after a live
trading incident. Follow this document exactly. Do not improvise under pressure.

---

## 1. When to Roll Back

Roll back immediately if ANY of the following is true:

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Consecutive losing days | 3 or more | Stop and investigate before next session |
| Kill switch triggered | Any time | Mandatory review before resuming |
| Broker reconciliation mismatch | Any quantity mismatch | Halt entries, alert, manual review |
| Model artifact missing at startup | Any required artifact | Block startup, roll back model |
| SHAP report is a stub | Status == "STUB" | Block live ML inference |
| Feature schema hash mismatch | Any mismatch | Block inference, do not trade |
| Daily loss >= 2% three days in a row | 3 consecutive days | Full rollback and strategy review |
| Unexplained position in broker book | Any unknown position | Halt entries immediately |

**When in doubt, stop trading. Missing one day's opportunity costs nothing. Trading with a broken system can cost the entire account.**

---

## 2. Pre-Rollback: Stop Current Session

### Step 1 — Trigger kill switch

```bash
# Option A: Sentinel file (fastest — takes effect within next bar)
touch /home/trader/EXCHANGE/state/kill_switch.sentinel

# Option B: systemd stop (stops all new session timers)
sudo systemctl stop algo-trading-market-open.timer
sudo systemctl stop algo-trading-market-open.service
```

### Step 2 — Wait for force-flat to complete

The system flattens all positions at the next bar close. If market hours, wait until 15:15 IST
(force-close time). If urgent:

```bash
# Monitor logs for "force_close.complete" or "kill_switch.positions_flat"
tail -f /home/trader/EXCHANGE/logs/algo_trading.jsonl | python -m json.tool | grep -i "flat\|kill\|close"
```

### Step 3 — Verify positions are flat

```bash
# Check local state file
cat /home/trader/EXCHANGE/state/live_state.json | python -m json.tool

# Manually verify via broker app or Shoonya portal
# state/live_state.json should show positions: {}
```

Do NOT proceed to rollback steps while any position is open.

---

## 3. Rollback Steps

### Step 4 — Identify previous working tag

```bash
git -C /home/trader/EXCHANGE tag -l --sort=-version:refname | head -10
# Pick the last known-good tag, e.g., v1.2.3
```

### Step 5 — Stop all services

```bash
sudo systemctl stop algo-trading-market-open.timer
sudo systemctl stop algo-trading-feature-engine.service
sudo systemctl stop algo-trading-inference.service
sudo systemctl stop algo-trading-orchestrator.service
```

### Step 6 — Checkout previous tag

```bash
cd /home/trader/EXCHANGE
git fetch --tags
git checkout v1.2.3    # replace with actual previous tag
```

If you need to keep local changes:

```bash
git stash
git checkout v1.2.3
```

### Step 7 — Restore previous model artifacts

```bash
# Model artifacts are tagged with the release they were trained for
cp -r /home/trader/EXCHANGE/state/model_backup_YYYYMMDD/* \
       /home/trader/EXCHANGE/models/trained/
# Verify the artifact directory
ls -la /home/trader/EXCHANGE/models/trained/
```

### Step 8 — Restore previous environment config

```bash
cp /home/trader/EXCHANGE/.env.rollback_YYYYMMDD /home/trader/EXCHANGE/.env
# Review the restored .env before starting
cat /home/trader/EXCHANGE/.env | grep -v PASSWORD | grep -v SECRET | grep -v TOKEN
```

### Step 9 — Reinstall dependencies for the rolled-back version

```bash
cd /home/trader/EXCHANGE
source .venv/bin/activate
pip install -e ".[live]" --quiet
# Or if using requirements.txt:
pip install -r requirements.txt --quiet
```

### Step 10 — Validate model artifacts exist

```bash
ls -la /home/trader/EXCHANGE/models/trained/
# Must see: xgb.so, lgb.so, cat.so, frozen_model_spec.json, shap/
# Check that sebi_report.json status is NOT "STUB"
python -c "
import json
r = json.load(open('models/trained/sebi_report.json'))
print('SHAP status:', r.get('status', 'UNKNOWN'))
"
```

### Step 11 — Run paper validation before live

```bash
# Never go straight from rollback to live
# Run paper mode for at least one full session

# Update .env temporarily
sed -i 's/TRADING_MODE=live/TRADING_MODE=paper/' /home/trader/EXCHANGE/.env
sed -i 's/ENABLE_LIVE_TRADING=true/ENABLE_LIVE_TRADING=false/' /home/trader/EXCHANGE/.env

# Start paper session
sudo systemctl start algo-trading-orchestrator.service
```

Monitor the paper session for one full trading day. Only after a clean paper session:

```bash
# Restore live settings
sed -i 's/TRADING_MODE=paper/TRADING_MODE=live/' /home/trader/EXCHANGE/.env
sed -i 's/ENABLE_LIVE_TRADING=false/ENABLE_LIVE_TRADING=true/' /home/trader/EXCHANGE/.env
```

### Step 12 — Restart services for live

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now algo-trading-market-open.timer
sudo systemctl status algo-trading-orchestrator.service
```

---

## 4. Post-Rollback

### Step 13 — Update incident log

Create entry in `state/incident_log.md` (or append to existing):

```
## INCIDENT — YYYY-MM-DD HH:MM IST

- **Trigger:** (what caused the rollback)
- **Rolled back from:** vX.Y.Z
- **Rolled back to:** vA.B.C
- **Positions at time of rollback:** (list open positions or "flat")
- **P&L impact:** (estimated)
- **Operator:** (name)
- **Time to resolution:** (HH:MM to HH:MM IST)
```

### Step 14 — Notify team / stakeholders

Send Telegram alert manually if the system is down:

```
[ROLLBACK] AlgoTrader rolled back from vX.Y.Z to vA.B.C at HH:MM IST.
Reason: <brief description>
Status: Paper validation in progress before next live session.
ETA to live: <date>
```

### Step 15 — Root cause analysis before next deployment

Do NOT redeploy a new version until the root cause of the rollback is understood and fixed.
Use the audit log to replay what happened:

```bash
# Review order events around the incident time
grep '"YYYY-MM-DDTHH:' /home/trader/EXCHANGE/state/order_audit.log | python -m json.tool | less
```

Mandatory before next deployment:
- [ ] Root cause documented in `docs/TODO_RISK_REGISTER.md`
- [ ] Fix implemented and tested
- [ ] Full release checklist completed again from scratch

---

## Command Quick Reference

| Action | Command |
|--------|---------|
| Stop all timers | `sudo systemctl stop algo-trading-market-open.timer` |
| Trigger kill switch | `touch state/kill_switch.sentinel` |
| Check git tags | `git tag -l --sort=-version:refname` |
| Checkout tag | `git checkout v1.2.3` |
| Reinstall deps | `pip install -e ".[live]"` |
| Start services | `sudo systemctl start algo-trading-orchestrator.service` |
| Watch logs | `tail -f logs/algo_trading.jsonl \| python -m json.tool` |
| Check positions | `cat state/live_state.json \| python -m json.tool` |
| Clear kill switch | `rm -f state/kill_switch.sentinel` |
