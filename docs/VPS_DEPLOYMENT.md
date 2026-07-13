# VPS Deployment Guide — NSE AlgoTrader

## Why a VPS

| Connection | Ping to NSE (Mumbai) | Result |
|---|---|---|
| Home broadband (Hyderabad/Bangalore) | 150–250 ms | Acceptable for EMA cross (15-min bars) |
| Home broadband (Delhi/Kolkata) | 200–350 ms | Borderline — latency spikes hurt LIMIT fills |
| Mumbai VPS (DigitalOcean/Hetzner/Linode) | 15–35 ms | 10× improvement — mandatory for sub-5-min strategies |

**Rule:** Run on home machine through Stage 1 paper trading. If latency profiler shows `signal→broker_ack > 250 ms` consistently → move to VPS before Stage 2.

---

## Recommended VPS Specs

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 40 GB SSD | 80 GB SSD |
| Region | **Mumbai** | Mumbai |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Monthly cost | ~₹700–900 | ~₹1,400–1,800 |

**Providers ranked by Mumbai latency:**
1. **DigitalOcean Mumbai** — most reliable, straightforward pricing
2. **Hetzner Mumbai (FSN)** — cheapest per spec
3. **Linode Mumbai** — similar to DigitalOcean
4. **AWS ap-south-1 (Mumbai)** — overkill; spot instances for training runs only

**SEBI compliance note:** SEBI circulars require registering your static public IP with your broker. Whether it must be an Indian IP is broker policy — confirm with Shoonya before buying VPS. DigitalOcean and Hetzner Mumbai both provide Indian IPs.

---

## Initial VPS Setup

```bash
# 1. Connect
ssh root@<your-vps-ip>

# 2. Create non-root user
adduser trader
usermod -aG sudo trader

# 3. Harden SSH
nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
# Set: PermitRootLogin no
systemctl restart ssh

# 4. Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow from <your-home-ip> to any port 22   # SSH from home only
ufw allow 9000/tcp   # QuestDB web console (restrict to your IP in prod)
ufw enable

# 5. Install Python 3.11
apt update && apt install -y python3.11 python3.11-venv python3.11-dev git build-essential

# 6. Install Docker (for QuestDB)
curl -fsSL https://get.docker.com | sh
usermod -aG docker trader

# 7. Switch to trader user
su - trader

# 8. Clone repo
git clone https://github.com/your-username/EXCHANGE.git
cd EXCHANGE/EXCHANGE

# 9. Create venv and install deps
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Shoonya (not on PyPI):
pip install git+https://github.com/Shoonya-Dev/ShoonyaApi-py.git

# 10. Copy .env
nano .env    # add all keys — chmod 600 automatically on nano save
chmod 600 .env

# 11. Start QuestDB
docker run -d \
  --name questdb \
  --restart always \
  -p 9000:9000 \
  -p 9009:9009 \
  -v questdb_data:/root/.questdb \
  questdb/questdb
```

---

## Systemd Services

Three separate services: main trading process, alert sink, and health check.
Each restarts automatically on failure with a 10-second backoff.

### Install

```bash
# Copy service files
sudo cp docs/systemd/algo-trading-main.service    /etc/systemd/system/
sudo cp docs/systemd/algo-trading-alert-sink.service /etc/systemd/system/
sudo cp docs/systemd/algo-trading-health.service  /etc/systemd/system/

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable algo-trading-main
sudo systemctl enable algo-trading-alert-sink
sudo systemctl enable algo-trading-health

# Start
sudo systemctl start algo-trading-alert-sink   # start alert sink first
sudo systemctl start algo-trading-health
sudo systemctl start algo-trading-main         # main last
```

### Check status

```bash
sudo systemctl status algo-trading-main
journalctl -u algo-trading-main -f --lines=50   # follow live logs
```

---

## Daily Restart Cron (Mandatory)

Shoonya sessions expire daily. The system must re-authenticate before market open.

```bash
crontab -e   # as user 'trader'
```

Add these lines:

```cron
# NSE AlgoTrader — daily restart before market pre-open (8:55 AM IST = 3:25 UTC)
# Runs only on weekdays (Mon-Fri = 1-5)
25 3 * * 1-5 /home/trader/bin/restart_trading.sh >> /home/trader/logs/restart.log 2>&1

# Health check ping (every 5 min during market hours, weekdays only)
# Sends result to /tmp/health_last — monitor this file
*/5 3-10 * * 1-5 curl -s http://localhost:8080/health > /tmp/health_last 2>&1
```

Create `/home/trader/bin/restart_trading.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting daily restart..."

# Graceful stop — allow up to 30s for open orders to settle
systemctl stop algo-trading-main
sleep 10
systemctl stop algo-trading-alert-sink
sleep 2

# Start alert sink first (main needs it ready)
systemctl start algo-trading-alert-sink
sleep 3

# Start main
systemctl start algo-trading-main

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Restart complete."
```

```bash
chmod +x /home/trader/bin/restart_trading.sh
mkdir -p /home/trader/logs
```

---

## Monitoring

### Manual health check

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
# Or using Makefile:
make health
```

### Watch mode (refreshes every 5 seconds)

```bash
make monitor
# Equivalent to: watch -n 5 python health_check.py --once
```

### Uptime Robot (free external monitoring)

1. Create account at uptimerobot.com
2. Add HTTP monitor: `http://<your-vps-ip>:8080/health`
3. Alert on status code != 200 (critical status returns 503)
4. Set interval: 5 minutes
5. Add Telegram/email alert

This gives you external monitoring that works even if your VPS's systemd is broken.

---

## Log Management

Logs go to systemd journal. Rotate to prevent disk fill:

```bash
# /etc/systemd/journald.conf
[Journal]
SystemMaxUse=2G
SystemMaxFileSize=100M
MaxRetentionSec=30day
```

QuestDB trade logs are the audit trail (required 5 years under SEBI). Back up QuestDB volume weekly:

```bash
# Weekly backup cron (Sunday 2 AM IST = Saturday 20:30 UTC)
30 20 * * 0 docker run --rm \
  -v questdb_data:/data \
  -v /home/trader/backups:/backups \
  alpine tar czf /backups/questdb_$(date +%Y%m%d).tar.gz /data
```

---

## Performance Tuning

### Python process priority

```bash
# Run main.py with slightly elevated priority (not real-time — no SEBI issues)
# Set in the systemd service file: Nice=-5
```

### Network buffer tuning (reduces tail latency)

Add to `/etc/sysctl.conf`:

```
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
```

Apply: `sudo sysctl -p`

### Measure actual latency from VPS

```python
# Run this from the VPS after deploying:
import time, socket

def ping_shoonya_api(n=100):
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        with socket.create_connection(("api.shoonya.com", 443), timeout=5):
            pass
        times.append((time.monotonic() - t0) * 1000)
    times.sort()
    print(f"Shoonya API ping: p50={times[n//2]:.1f}ms p95={times[int(n*0.95)]:.1f}ms")

ping_shoonya_api()
# VPS target: p50 < 30ms, p95 < 60ms
# Home target: p50 < 200ms, p95 < 400ms
```

---

## Security Checklist

- [ ] SSH key-only auth (no passwords)
- [ ] Root login disabled
- [ ] UFW firewall enabled, SSH restricted to home IP
- [ ] `.env` has `chmod 600`
- [ ] `git check-ignore -v .env` confirms it is in `.gitignore`
- [ ] QuestDB port 9000 NOT exposed to internet (use SSH tunnel or restrict to your IP)
- [ ] Health check port 8080 restricted (or behind nginx with basic auth)
- [ ] Rotate API keys immediately if VPS is ever compromised
- [ ] Broker notified of VPS static IP (SEBI requirement)
