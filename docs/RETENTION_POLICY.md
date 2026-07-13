# Retention Policy — NSE AlgoTrader

Defines how long each artifact class is retained, why, and how it is archived.
Compliance with SEBI PFUTP §22 and the SEBI retail algo circular (Feb 4, 2025)
requires specific minimum retention for trade and audit records.

---

## Retention Schedule

| Artifact | Path | Retention Period | Rationale |
|----------|------|-----------------|-----------|
| Order audit log | `state/order_audit.log` | **5 years** | SEBI PFUTP §22 compliance requirement |
| QuestDB trade/fill/signal tables | `data/questdb/` | **5 years** | SEBI compliance — trade records |
| QuestDB raw tick data | `data/questdb/ticks/` | **90 days** | Research use only; large volume |
| 15-min OHLCV bars | `data/questdb/candles/` | **5 years** | Model training data reproducibility |
| Model artifacts (`.so`, `FrozenModelSpec`) | `models/trained/` | **2 years** | Reproducibility and regulatory audit |
| Paper trade logs | `logs/paper_*.jsonl` | **1 year** | Validation records and edge proof |
| Config snapshots | `state/config_snapshot_*.json` | **1 year** | Incident investigation and reproducibility |
| SHAP reports | `models/trained/shap/` | **2 years** | SEBI explainability — model audit trail |
| Session reports (EOD summaries) | `logs/session_report_*.json` | **1 year** | Performance monitoring and reconciliation |
| Dead-letter files (unwritten data) | `data/dead_letter.jsonl` | **90 days** | QuestDB failure recovery |
| State files (`live_state.json`) | `state/live_state.json` | **90 days** | Incident investigation |
| Reconcile sentinels | `state/*.sentinel` | **90 days** | Audit trail for halt events |
| Incident log | `state/incident_log.md` | **5 years** | Regulatory and operational history |
| Telegram alert log | `logs/alerts.jsonl` | **1 year** | Operator notification audit |

---

## Archive and Compression Policy

### Compression Rules

- All log files (`.jsonl`, `.log`) older than **30 days** are gzip-compressed.
- Compressed files retain the original name with `.gz` suffix: `order_audit.2026-01-15.log.gz`
- QuestDB data is stored in its native binary format (already compressed internally).
- Model artifacts are NOT compressed — they must remain importable.

```bash
# Compression cron (run daily at 01:00 IST)
find /home/trader/EXCHANGE/logs/ -name "*.jsonl" -mtime +30 ! -name "*.gz" -exec gzip {} \;
find /home/trader/EXCHANGE/state/ -name "order_audit.*.log" -mtime +30 ! -name "*.gz" -exec gzip {} \;
```

### Archival Rules

- Files older than 1 year are moved to a `archive/` subdirectory.
- Files in `archive/` are never auto-deleted — deletion requires manual operator approval.
- SEBI-required records (audit logs, trade logs, 5-year retention) must be on write-once or
  append-only storage (or have an immutable backup copy).

```bash
# Annual archive (run on January 1 each year)
ARCHIVE_YEAR=$(($(date +%Y) - 1))
mkdir -p /home/trader/EXCHANGE/archive/${ARCHIVE_YEAR}
find /home/trader/EXCHANGE/logs/ -name "*${ARCHIVE_YEAR}*" -exec mv {} archive/${ARCHIVE_YEAR}/ \;
find /home/trader/EXCHANGE/state/ -name "*${ARCHIVE_YEAR}*" -exec mv {} archive/${ARCHIVE_YEAR}/ \;
```

### Deletion Rules

Automated deletion is permitted only for the following:

| Artifact | Delete after |
|----------|-------------|
| Raw tick data | 90 days |
| Dead-letter files | 90 days (after confirmed replay to QuestDB) |
| State backup snapshots | 90 days |
| Temporary paper trade session files | 30 days |

**NEVER auto-delete:**
- Anything in `state/order_audit.*` (5-year SEBI requirement)
- Anything in `archive/` (requires manual review)
- Model artifacts within the 2-year window
- SHAP reports within the 2-year window

---

## Disk Space Estimates

These are rough estimates for a single-strategy system on 5 Nifty 50 symbols.
Scale linearly for more symbols or strategies.

| Artifact | Estimated Size | Period | Annual Rate |
|----------|---------------|--------|-------------|
| Order audit log | ~5 MB/month | — | ~60 MB/year |
| QuestDB tick data | ~200 MB/month | — | ~2.4 GB/year |
| QuestDB OHLCV bars (15-min) | ~50 MB/month | — | ~600 MB/year |
| QuestDB trade/fill tables | ~10 MB/month | — | ~120 MB/year |
| Model artifacts (all versions) | ~500 MB/version | 1 per week | ~26 GB/year |
| SHAP reports | ~20 MB/version | 1 per week | ~1 GB/year |
| Session logs (compressed) | ~2 MB/day | — | ~730 MB/year |
| Config snapshots | ~50 KB/day | — | ~18 MB/year |
| Dead-letter files | ~5 MB/day max | — | ~450 MB/rolling 90-day |

**Total estimated disk usage (5-year retention):**
- Compliance-required (5 years): ~15 GB
- Model/research (2 years): ~60 GB
- Operational (1 year): ~10 GB
- **Recommended VPS disk minimum for 5-year compliance: 100 GB SSD**

---

## Compliance Notes

**SEBI PFUTP §22 (Prevention of Fraudulent and Unfair Trade Practices):**
Algo trading systems must maintain records of all orders, modifications, cancellations, fills,
and rejections for a minimum of 5 years. Records must be accessible for regulatory inspection
within 24 hours. [ASSUMED — verify exact wording with your compliance officer before live trading]

**SEBI Retail Algo Circular (February 4, 2025):**
Algo strategies must be documented, tagged with broker-assigned strategy IDs, and the system
must maintain an audit trail demonstrating compliance with approved strategy behavior.
[Source: https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html]

**Data Residency:**
[ASSUMED] Trade and audit records for Indian securities should be stored in India or with
providers compliant with Indian data localisation requirements. Verify with your broker and
a compliance advisor before using foreign-hosted cloud storage for SEBI-required records.
