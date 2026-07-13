# Backup Policy — NSE AlgoTrader

Defines what to back up, when, where, and how to restore. Compliance with SEBI PFUTP
§22 requires that audit and trade records be retained for 5 years.

---

## Artifact Backup Schedule

### QuestDB Time-Series Data

| Setting | Value |
|---------|-------|
| Backup time | 00:00 IST daily (after market close, before market re-open) |
| Retention | 5 years for trade/fill/signal tables; 90 days for raw tick data |
| Method | QuestDB BACKUP SQL command or filesystem snapshot of `data/questdb/` |
| Destination | Primary: local external drive. Secondary: S3 (or similar cloud) |

```bash
# Daily backup script (run via cron at 00:00 IST)
# QuestDB backup via HTTP API
curl -s -X GET "http://localhost:9000/exp?query=BACKUP+DATABASE" > /dev/null
# OR filesystem snapshot (simpler, requires QuestDB to be stopped briefly or WAL-safe)
rsync -a --delete /home/trader/EXCHANGE/data/questdb/ /mnt/backup/questdb/$(date +%Y%m%d)/
```

### Audit Logs (`state/order_audit.log`)

| Setting | Value |
|---------|-------|
| Backup frequency | Real-time rotation (daily log files) + nightly backup |
| Retention | 5 years (SEBI PFUTP §22 requirement — non-negotiable) |
| Rotation | Daily at 00:00 IST — `order_audit.YYYY-MM-DD.log` |
| Compression | gzip after 30 days |
| Destination | Primary: local external drive. Secondary: S3. Both must be immutable (write-once) |

```bash
# Nightly rotation and backup (via cron at 00:05 IST)
LOG_DATE=$(date -d "yesterday" +%Y-%m-%d)
# Rotate
mv /home/trader/EXCHANGE/state/order_audit.log \
   /home/trader/EXCHANGE/state/order_audit.${LOG_DATE}.log
# Compress logs older than 30 days
find /home/trader/EXCHANGE/state/ -name "order_audit.*.log" -mtime +30 -exec gzip {} \;
# Sync to backup
rsync -a /home/trader/EXCHANGE/state/order_audit.* /mnt/backup/audit/
aws s3 sync /home/trader/EXCHANGE/state/order_audit.* s3://your-bucket/audit/
```

### Model Artifacts (`models/trained/`)

| Setting | Value |
|---------|-------|
| Backup trigger | After every training run (automated at end of train_models.py) |
| Schedule | Sunday nights after market close (standard retraining window) |
| Retention | 2 years (reproducibility and audit trail) |
| Method | Copy entire `models/trained/` directory with version tag |
| Destination | Primary: `state/model_backup_YYYYMMDD/`. Secondary: S3 |

```bash
# Post-training backup (called from train_models.py or Makefile)
BACKUP_DIR="state/model_backup_$(date +%Y%m%d)"
mkdir -p "${BACKUP_DIR}"
cp -r models/trained/ "${BACKUP_DIR}/"
aws s3 sync "${BACKUP_DIR}/" "s3://your-bucket/models/${BACKUP_DIR}/"
```

### Config Snapshots (`state/config_snapshot_*.json`)

| Setting | Value |
|---------|-------|
| Creation | Automatically at startup (logged to `state/config_snapshot_YYYYMMDD_HHMMSS.json`) |
| Backup frequency | Weekly (every Sunday night) |
| Retention | Last 52 weeks (1 year) |
| Destination | Primary: local external drive. Secondary: S3 |

```bash
# Weekly config snapshot backup
find /home/trader/EXCHANGE/state/ -name "config_snapshot_*.json" -newer /tmp/last_config_backup \
  | xargs -I{} cp {} /mnt/backup/configs/
# Prune snapshots older than 52 weeks
find /home/trader/EXCHANGE/state/ -name "config_snapshot_*.json" -mtime +365 -delete
```

### Dead-Letter Files (`data/dead_letter.jsonl`)

| Setting | Value |
|---------|-------|
| Backup frequency | Daily at 00:10 IST |
| Retention | 90 days |
| Method | Copy and date-stamp; replay to QuestDB before deleting |
| Destination | Primary: local external drive. Secondary: S3 |

```bash
# Dead-letter backup (cron at 00:10 IST)
if [ -s /home/trader/EXCHANGE/data/dead_letter.jsonl ]; then
  cp /home/trader/EXCHANGE/data/dead_letter.jsonl \
     /mnt/backup/dead_letter/dead_letter_$(date +%Y%m%d).jsonl
  aws s3 cp /home/trader/EXCHANGE/data/dead_letter.jsonl \
            s3://your-bucket/dead_letter/dead_letter_$(date +%Y%m%d).jsonl
fi
# Prune dead-letter backups older than 90 days
find /mnt/backup/dead_letter/ -mtime +90 -delete
```

### State Files (`state/live_state.json`, `state/reconcile_halt.sentinel`)

| Setting | Value |
|---------|-------|
| Backup frequency | Every 6 hours (00:00, 06:00, 12:00, 18:00 IST) |
| Retention | 90 days |
| Method | Copy with timestamp |
| Destination | Primary: local external drive |

```bash
# 6-hourly state backup
STATE_BACKUP="/mnt/backup/state/state_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "${STATE_BACKUP}" \
    /home/trader/EXCHANGE/state/live_state.json \
    /home/trader/EXCHANGE/state/*.sentinel 2>/dev/null
```

---

## Backup Destinations

### Primary — Local External Drive

- Mount point: `/mnt/backup/`
- Recommended: encrypted external SSD, minimum 500 GB
- Must be physically separate from the VPS (different physical location)
- For VPS deployments: attach a block storage volume (e.g., AWS EBS, DigitalOcean Block Storage)

```bash
# Verify backup drive is mounted
df -h /mnt/backup/
# If not mounted:
sudo mount /dev/sdb1 /mnt/backup/  # adjust device as needed
```

### Secondary — Cloud Storage (S3 or Compatible)

- Service: AWS S3, DigitalOcean Spaces, Backblaze B2, or any S3-compatible provider
- Bucket: `algo-trading-backup-YOURNAME` (use a private bucket with versioning enabled)
- Region: Use Indian or Singapore region for data residency
- Access: IAM user with write-only permissions to the backup bucket [ASSUMED — exact permissions depend on provider]

```bash
# Install and configure AWS CLI
pip install awscli
aws configure  # Enter your S3 access key and secret

# Test write access
aws s3 cp /dev/null s3://your-bucket/test.txt && echo "S3 access OK"
```

---

## Restore Procedure

### Restore QuestDB Data

```bash
# Stop trading system
sudo systemctl stop algo-trading-orchestrator.service

# Stop QuestDB
docker stop questdb

# Restore QuestDB data directory
rsync -a --delete /mnt/backup/questdb/YYYYMMDD/ /home/trader/EXCHANGE/data/questdb/

# Restart QuestDB
docker start questdb

# Verify QuestDB
curl -s "http://localhost:9000/exec?query=SHOW+TABLES"
```

### Restore Audit Logs

```bash
# Audit logs are plain files — copy back and decompress if gzipped
gunzip /mnt/backup/audit/order_audit.YYYY-MM-DD.log.gz
cp /mnt/backup/audit/order_audit.YYYY-MM-DD.log /home/trader/EXCHANGE/state/
```

### Restore Model Artifacts

```bash
# See ROLLBACK_PROCEDURE.md Step 7
cp -r /mnt/backup/models/state_model_backup_YYYYMMDD/ /home/trader/EXCHANGE/models/trained/
```

### Restore Dead-Letter Data to QuestDB

```bash
# After restoring QuestDB, replay unwritten dead-letter data
# This is handled automatically by the system on next startup
# OR manually:
python -m algo_trading.scripts.replay_dead_letter --file /mnt/backup/dead_letter/dead_letter_YYYYMMDD.jsonl
```

---

## Backup Health Check

Run weekly to confirm backups are working:

```bash
# Check backup freshness
echo "=== Backup Health Check ==="
echo "QuestDB:"
ls -lt /mnt/backup/questdb/ | head -3
echo "Audit logs:"
ls -lt /mnt/backup/audit/ | head -3
echo "Models:"
ls -lt /mnt/backup/models/ | head -3
echo "S3 sync:"
aws s3 ls s3://your-bucket/audit/ | tail -3
```

A backup is considered stale if the most recent entry is older than 2 days for daily backups
or older than 2 weeks for weekly backups. Alert operator immediately if stale.
