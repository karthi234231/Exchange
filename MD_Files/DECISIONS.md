# Current Canonical Decisions

Single source of truth for all active architecture decisions.
Last updated: 2026-06-06

If any other file contradicts a decision here, this file wins.
Update this file whenever a decision changes — do not just update CLAUDE.md or arch-spec.json alone.

---

## Broker

| Decision | Value | Reason |
|---|---|---|
| Primary execution broker | **Shoonya (Finvasia)** | Zero brokerage, no per-request TOTP |
| Dev/docs reference broker | AngelOne SmartAPI | Best documented API for initial code reference |
| AngelOne in production | **FORBIDDEN** | TOTP mid-day expiry, data throttled 1-3/sec |

---

## Current Stage

| Item | Value |
|---|---|
| Current stage | **Stage 1 — planned (no code yet)** |
| Stage 0 status | Blocked — no valid multi-year intraday data source confirmed |
| Paper trading status | Not started — no runnable code exists |
| Live trading | FORBIDDEN |

---

## Data Feed

| Stage | Feed | Rules |
|---|---|---|
| Stage 1 | Shoonya WebSocket (throttled) | EMA cross paper trade only |
| Stage 2A | Shoonya live feed | OHLCV/volume bars only — no L2 microstructure |
| Stage 2B+ | **TrueData** (unthrottled L2) | Required before Hawkes/VPIN/TFI — historical+live parity must be proven first |

---

## Data Storage

| Data type | Storage | Rule |
|---|---|---|
| Ticks, bars, features, signals, orders, fills, P&L | **QuestDB** | No SQLite/PostgreSQL for any of these |
| Local dev mocks only | SQLite allowed as `DevSqliteStore` only | Must be clearly named, never in production path |

---

## Bar Type

| Stage | Bar type |
|---|---|
| Stage 1 (EMA paper trade) | 15-min time bars — acceptable simplification |
| Stage 2A+ ML training | Volume bars — must match live execution bar type |
| Rule | `TRAIN_BAR_TYPE == LIVE_BAR_TYPE` — enforced by config invariant |

---

## ML Models

| Decision | Value |
|---|---|
| Allowed in Stage 1 | **None** — EMA cross only |
| Allowed in Stage 2A | XGBoost + LightGBM + CatBoost baseline only |
| TabPFN | **BANNED from production** — experimentally in research/notebooks only |
| RL (PPO) | **NOT NOW** — research only, unstable on financial data |
| Deep models (KAN, Mamba, TSMixer, FinBERT) | **NOT NOW** — promote out only after deterministic baseline profitable months |
| Model serialization | joblib for training; Treelite compiled .so for live inference |

---

## Compliance & Security

| Decision | Value |
|---|---|
| SHAP | Chosen explainability artifact — not the only SEBI-mandated method |
| Strategy ID | Required field in every live order |
| Live mode gate | Hard-disabled in code until all README.md gates pass |
| `.env` protection | Keep outside OneDrive sync if possible; Windows ACL — do not use Unix chmod |
| Secrets in git | **NEVER** — .gitignore covers .env and all credential files |

---

## Go-Live Gate (single consolidated gate — replaces all prior conflicting versions)

ALL of the following must be true before live mode is unlocked:

- [ ] 3 calendar months paper trading completed
- [ ] At least 40 trading sessions logged
- [ ] Paper Sharpe >= 1.0 after full India cost model
- [ ] Zero unreconciled broker state events
- [ ] Zero duplicate orders in paper log
- [ ] Kill switch tested manually (forced trigger, positions closed)
- [ ] Manual close runbook documented and tested
- [ ] Startup reconciler tested after simulated restart
- [ ] Order FSM tested for every state transition
- [ ] SEBI Strategy ID registered and configured
- [ ] Static IP registered if required by broker
- [ ] `ENABLE_LIVE_TRADING=true` set explicitly in .env
- [ ] Backtest vs paper slippage gap documented

---

## Not Now List (freeze until deterministic baseline is profitable)

- RL position sizing (PPO)
- LLM/FinBERT sentiment
- Deep time-series (KAN, Mamba, TSMixer, Chronos)
- Hawkes process features
- Trade Flow Imbalance (TFI)
- F&O strategies

Items may only be promoted out of this list after evidence from live/paper trading.
