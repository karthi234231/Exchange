# Risk Register

Open risks for the NSE AlgoTrader project.
Last updated: 2026-06-06

Converted from IMPROVEMENT_RESEARCH_FEEDBACK.md which used `FIXED` to mean "decided in docs" — status labels corrected here.

---

## Status Labels

| Label | Meaning |
|---|---|
| `OPEN` | Identified, no action taken |
| `DOC_DECIDED` | Decision recorded in docs — no code yet |
| `IMPLEMENTED` | Code written (file + function exists) |
| `TESTED` | Implementation has passing tests |
| `LIVE_VALIDATED` | Validated in paper or live trading |
| `ACCEPTED` | Risk accepted with documented rationale |

---

## P0 — Blocks runnable project or safe execution

| ID | Risk | Status | Mitigation |
|---|---|---|---|
| R-01 | No runnable Python code exists | OPEN | Build Stage 1 scaffold as first milestone |
| R-02 | Free data sources cannot provide 3-year 15-min intraday history | DOC_DECIDED | Two-tier data plan in DATA_COLLECTION_GUIDE.md — paid source required for ML validation |
| R-03 | Stage definitions contradict across files (broker, feed, stage number) | DOC_DECIDED | arch-spec.json + DECISIONS.md are now canonical |
| R-04 | No order state machine — duplicate/zombie orders possible | OPEN | Build execution/order_fsm.py before any broker order code |
| R-05 | Live trading commands documented before safety gates exist | DOC_DECIDED | `ENABLE_LIVE_TRADING=false` by default in .env.example; live mode hard-disabled in code |
| R-06 | TabPFN banned in production but still in _part6_aidev.md | DOC_DECIDED | Stale banner added to _part6_aidev.md; TabPFN banned in DECISIONS.md |
| R-07 | `chmod 600` instruction is Unix-only; project runs on Windows | OPEN | Use Windows ACL — document in security section before credentials are introduced |

---

## P1 — Major design flaw, financial risk, or compliance risk

| ID | Risk | Status | Mitigation |
|---|---|---|---|
| R-10 | Shoonya session tokens expire daily — exit orders can fail | DOC_DECIDED | SessionManager + heartbeat required before live; noted in arch-spec.json auth field |
| R-11 | Static slippage (0.05%) creates delusional Sharpe | DOC_DECIDED | Dynamic slippage model required — tiered by liquidity/time-of-day/volatility |
| R-12 | OBI averaged over 15-min is near-random | DOC_DECIDED | OBI only valid tick-by-tick or per volume bar; banned at 15-min scale |
| R-13 | Walk-forward purge window missing feature lookback | OPEN | purge = max_holding + max_feature_lookback_bars — must be implemented and tested |
| R-14 | Training on OHLCV time bars, executing on L2 volume bars — model collapse | DOC_DECIDED | TRAIN_BAR_TYPE == LIVE_BAR_TYPE constraint in config |
| R-15 | Paper trading gate is inconsistent across docs (2 weeks vs 4 weeks vs 3 months) | DOC_DECIDED | Single gate: 3 months / 40 sessions / Sharpe >= 1.0 — in DECISIONS.md |
| R-16 | SQLite used in setup guide despite QuestDB-only constraint | DOC_DECIDED | SETUP_GUIDE_COMPLETE.md labelled planned-only; QuestDB is canonical |
| R-17 | Stale limit order timeout is 30 min (2 candles) — executes dead alpha | OPEN | Reduce to 15-30 seconds; implement order_manager adjust loop |
| R-18 | Python GIL stalls tick feed during ML inference | DOC_DECIDED | ZeroMQ multiprocess isolation (Process A ingestion, B inference, C execution) |
| R-19 | SEBI compliance claims overstated — SHAP described as "mandatory" | DOC_DECIDED | SHAP is chosen explainability artifact, not the only option; compliance_matrix.md needed |
| R-20 | No runbook for manual close during broker API outage | OPEN | Create docs/RUNBOOK.md with broker app screenshots and emergency procedures |
| R-21 | Corporate action adjustments missing — splits corrupt backtests | OPEN | Build data/corporate_actions.py; store raw + adjusted prices separately |
| R-22 | Model registry absent — live inference can use wrong feature version | OPEN | Implement feature schema hash before any saved model |
| R-23 | Latency claims asserted without measurement | OPEN | Add latency_profiler.py; measure p50/p95/p99 per pipeline stage |
| R-24 | Kaggle GPU training produces artifacts that may not reproduce on CPU | DOC_DECIDED | Require Docker/lockfile parity; test model on local CPU before accepting |
| R-25 | AngelOne TOTP still referenced in some docs as primary | DOC_DECIDED | _part6_aidev.md marked stale; DECISIONS.md is authoritative |

---

## P2 — Quality, maintainability, or validation gaps

| ID | Risk | Status | Mitigation |
|---|---|---|---|
| R-30 | No tests exist for any module | OPEN | No module is complete until tests exist — rule added to build protocol |
| R-31 | Setup guide commands cannot run — files don't exist | DOC_DECIDED | SETUP_GUIDE_COMPLETE.md labelled "Planned Build Guide" |
| R-32 | Many `FIXED` statuses in IMPROVEMENT_RESEARCH_FEEDBACK.md meant "fixed in docs only" | DOC_DECIDED | Status labels corrected to DOC_DECIDED/IMPLEMENTED/TESTED in this register |
| R-33 | Windows vs Linux deployment unresolved | OPEN | Dev: Windows 11; Production: Linux VPS; use cross-platform code in Stage 1 |
| R-34 | Market calendar and NSE holiday handling not first-class | OPEN | Build market_calendar.py before any scheduled trading logic |
| R-35 | Documentation volume too high — contradictions buried | DOC_DECIDED | Research docs in MD_Files; canonical decisions in DECISIONS.md + arch-spec.json |
| R-36 | OneDrive sync may corrupt or expose state files | OPEN | Move data/, state/, models/ outside OneDrive sync path for production |

---

## Resolved (Evidence Required to Close)

An item can only move here after `TESTED` or `LIVE_VALIDATED` status is reached with evidence (test file path + passing result).

No items resolved yet — no code exists.
