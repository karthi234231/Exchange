# Operational Runbook

Emergency procedures and daily checklists for live trading operations.

**Read this before going live. Practice every procedure in paper mode first.**

Last updated: 2026-06-06

> Note: Some broker-specific UI paths below (marked `[TODO]`) need to be filled in once Shoonya account is confirmed active and tested. Screenshot these paths during paper trading setup.

---

## Start-of-Day Checklist

Run this every morning before market open (before 09:00 IST):

- [ ] VPN/internet connection stable
- [ ] Laptop plugged in (no battery-only trading)
- [ ] Shoonya session token fresh (login completed today)
- [ ] QuestDB running (`docker ps` shows questdb container up)
- [ ] System clock correct (check against IST — time drift causes order rejection)
- [ ] `python -m exchange --mode paper --dry-run` passes without errors
- [ ] Kill switch reset from previous day if it fired
- [ ] Previous day's reconciliation clean (no open positions overnight)
- [ ] MLflow / model registry shows correct model version loaded
- [ ] Telegram alert test received

If any check fails: **do not start trading that day**.

---

## Normal Session Flow

```
09:00  Run start-of-day checklist
09:10  System startup — startup reconciler runs
09:14  Confirm WebSocket connected, ticks flowing
09:15  Market opens — system active
09:15-09:30  Pre-open bar filtered (no entries — negative expectancy)
09:30  Normal trading begins
15:00  System stops new entries (no_new_trades_after = 15:00)
15:15  Force-close all positions — system issues close orders
15:20  Confirm all positions flat (check broker app)
15:30  Market closes
15:35  Review daily summary in QuestDB / MLflow
15:40  Run reconciliation check — confirm broker matches system state
```

---

## Emergency Procedures

### E1: Broker API Unreachable — Positions Open

**Symptoms:** WebSocket dropped, API calls timing out, orders not reaching broker.

**Steps:**
1. Open Shoonya mobile app or web portal immediately
2. Navigate to: `[TODO: exact path — e.g., Portfolio > Positions]`
3. Manually close all open positions via market order in broker app
4. Do NOT wait for API to recover before closing positions
5. After positions are flat, stop the trading system
6. Log the incident with timestamps in a text file
7. Do not restart until API is confirmed stable

**Contact:** Shoonya support — `[TODO: phone number and support URL]`

---

### E2: Internet Outage — Positions Open

**Symptoms:** No network connectivity.

**Steps:**
1. Immediately switch to mobile hotspot
2. Open Shoonya mobile app on phone (separate connection)
3. Close all open positions via market order in broker app
4. Check if trading system reconnected on hotspot
5. If system reconnected and positions already closed: reconcile state
6. If system did not reconnect: leave it stopped, positions are flat

**Prevention:** Keep mobile hotspot ready. Test hotspot failover before going live.

---

### E3: Laptop Crash / Sleep / Power Failure — Positions Open

**Symptoms:** System not running, positions unknown.

**Steps:**
1. Reboot laptop immediately
2. Open Shoonya app/web — check current positions before restarting system
3. If positions still open and market is still open: close manually via broker app
4. Once positions are flat: restart system in reconciliation mode
5. System startup reconciler will detect discrepancy and halt for review
6. Review QuestDB orders table for last known state
7. Manually confirm positions are flat before allowing new trades

---

### E4: Kill Switch Fired Automatically

**Symptoms:** System logged "KILL SWITCH TRIGGERED", stopped sending orders.

**Steps:**
1. Check reason: `SELECT * FROM risk_state ORDER BY timestamp DESC LIMIT 10` in QuestDB
2. Confirm current positions via broker app
3. Close any open positions manually if system did not auto-close them
4. Do NOT reset kill switch without understanding why it fired
5. Review the day's PnL — kill switch fired for a reason
6. Reset only if: positions are flat AND you understand and accept the cause
7. Reset command: `[TODO: exact CLI command once implemented]`

---

### E5: Order Stuck — No Fill, No Cancel Confirmation

**Symptoms:** Order in PENDING or CANCEL_PENDING state for > 2 minutes.

**Steps:**
1. Check order status in Shoonya app: `[TODO: path — e.g., Orders > Order Book]`
2. If order shows as open/pending in broker app: cancel manually via app
3. If order shows as filled in broker: reconcile system state (update QuestDB)
4. If order shows as rejected in broker: check rejection reason, log it
5. After resolving: check that position in system matches broker position

---

### E6: Telegram Alerts Stopped

**Symptoms:** No alert messages for > 15 minutes during active session.

**Steps:**
1. Check if bot token is still valid: send test message via curl
2. Check system logs for Telegram errors
3. Do not stop trading — alerts are monitoring only, not execution
4. Investigate and fix after market close

---

### E7: QuestDB Unavailable

**Symptoms:** System cannot write ticks or read signals.

**Steps:**
1. Check Docker: `docker ps` — is questdb container running?
2. If container crashed: `docker start questdb`
3. If container is running but unresponsive: `docker restart questdb` (takes ~30 sec)
4. While QuestDB is down: halt trading — data integrity is compromised
5. After restart: run startup reconciler before resuming

---

## End-of-Day Checklist

- [ ] All positions flat (confirmed in broker app, not just system)
- [ ] System stopped cleanly (no error on shutdown)
- [ ] Daily PnL logged in QuestDB `pnl_daily` table
- [ ] Slippage: estimated vs actual noted
- [ ] Any incidents documented in incident log
- [ ] QuestDB backup triggered (or confirmed scheduled backup ran)
- [ ] Model retraining scheduled if weekly refit day

---

## Incident Log Template

Keep a simple text file `incidents/YYYY-MM-DD.txt` for any non-routine event:

```
Date: 2026-XX-XX
Time: HH:MM IST
Incident: [brief description]
Positions affected: [symbols + qty]
Action taken: [what you did]
Outcome: [how it resolved]
Prevention: [what to do differently next time]
```

---

## Important Contact Information

| Contact | Details |
|---|---|
| Shoonya support | `[TODO: fill in]` |
| Shoonya emergency trading desk | `[TODO: fill in]` |
| NSE investor helpline | 1800 22 0058 |
| SEBI SCORES portal | https://scores.sebi.gov.in |

---

## Things to Practice Before Going Live

1. Manually close a position via Shoonya mobile app (practice this once in paper mode)
2. Trigger the kill switch manually and confirm it halts correctly
3. Simulate a laptop crash: close lid, confirm positions are safe
4. Test internet failover: disconnect WiFi, switch to mobile hotspot
5. Restart system mid-session and confirm reconciler works
