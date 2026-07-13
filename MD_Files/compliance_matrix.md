# Compliance Matrix

Tracks every regulatory and broker compliance requirement for live trading.

**Rule:** Live mode must NOT be enabled until every row below has status `IMPLEMENTED` or `VERIFIED`.

Last updated: 2026-06-06

---

## Status Labels

| Label | Meaning |
|---|---|
| `OPEN` | Requirement identified, nothing done |
| `RESEARCHED` | Official source found and read |
| `DESIGNED` | Implementation approach decided |
| `IMPLEMENTED` | Code written |
| `TESTED` | Tests pass |
| `VERIFIED` | Confirmed with broker/SEBI/NSE directly |

---

## SEBI Retail Algo Requirements

| Requirement | Official Source | Broker Field | Code Module | Status | Owner | Last Checked |
|---|---|---|---|---|---|---|
| Strategy must be registered with broker before live | SEBI circular Feb 2025 | Broker-specific registration flow | `config.STRATEGY_ID` guard in `main.py` | OPEN | — | — |
| Unique Strategy ID tagged on every order | SEBI circular Feb 2025 | Shoonya: `remarks` field | `execution/order_executor.py` | OPEN | — | — |
| Explainability documentation for ML strategy | SEBI circular Feb 2025 | — | `models/explainer.py` (SHAP chosen as evidence method) | OPEN | — | — |
| Algo must be approved by exchange-empanelled auditor | SEBI circular Feb 2025 | Via broker registration | Documentation only | OPEN | — | — |
| Static IP registration (if required by broker) | Broker-specific | Broker account settings | `.env` `STATIC_IP_CONFIRMED` flag | OPEN | — | — |

> SEBI circular: https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html
>
> Note: SHAP is this project's chosen explainability artifact. Whether SEBI mandates SHAP specifically vs. other explainability methods is unconfirmed — verify with broker before going live.

---

## Broker Compliance: Shoonya (Finvasia)

| Requirement | Details | Code Module | Status | Verified With Broker |
|---|---|---|---|---|
| Account KYC complete | Required before any order | Manual — not code | OPEN | No |
| Segment activation (NSE Equity, F&O if needed) | Required before trading that segment | Manual — not code | OPEN | No |
| API access enabled on account | Required for API trading | Manual — broker portal | OPEN | No |
| Strategy ID field mapping | Shoonya uses `remarks` field for strategy tag | `execution/order_executor.py` | OPEN | No |
| Rate limit compliance | 10 orders/sec max | `execution/rate_limiter.py` | OPEN | No |
| Auto square-off time awareness | Broker may auto-square at 15:15 or 15:20 | `risk/manager.py` force_close_time | OPEN | No |

---

## Order Tagging Requirements

Every live order must carry:

| Field | Purpose | Implementation | Status |
|---|---|---|---|
| `client_order_id` | Idempotency and reconciliation | `execution/order_fsm.py` | OPEN |
| `strategy_id` / `remarks` | SEBI compliance tag | `execution/order_executor.py` | OPEN |
| `product_type` | MIS (intraday) vs CNC (delivery) | `config.py` | OPEN |
| `exchange_segment` | NSE / BSE | `config.py` | OPEN |
| `validity` | DAY / IOC / GTD | `execution/order_executor.py` | OPEN |

---

## Audit Logging Requirements

| Requirement | Implementation | Status |
|---|---|---|
| Every order transition logged with timestamp | QuestDB `orders` table via `order_fsm.py` | OPEN |
| Every fill logged with slippage estimate vs actual | QuestDB `fills` table | OPEN |
| Every risk event logged (kill switch, circuit break) | QuestDB `risk_state` table | OPEN |
| SHAP explainability report generated per model | `models/explainer.py` | OPEN |
| Model version logged with every signal | `models/model_registry.py` | OPEN |
| Session logs retained for audit | structlog → rotating file | OPEN |

---

## Pre-Live Compliance Checklist

All of the following must be confirmed before `ENABLE_LIVE_TRADING=true`:

- [ ] SEBI Strategy ID obtained and stored in `.env`
- [ ] Strategy registration date stored in `.env` as `BROKER_APPROVAL_DATE`
- [ ] Static IP registered with broker (if required)
- [ ] `remarks` / strategy_id field tested on a paper order that reaches broker
- [ ] SHAP explainability report generated for the production model
- [ ] Rate limiter tested — cannot exceed 10 orders/sec
- [ ] Auto square-off time confirmed with broker and coded into force_close_time
- [ ] All rows in this matrix at `IMPLEMENTED` or `VERIFIED` status
