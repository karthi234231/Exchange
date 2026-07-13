# Multi-arch: supports linux/amd64 (AWS/Hetzner) and linux/arm64 (Oracle Cloud free tier)
FROM python:3.11-slim

# System deps (gcc needed for some pip packages on ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       "git+https://github.com/Shoonya-Dev/ShoonyaApi-py.git#egg=NorenRestApiPy"

# Copy source
COPY algo_trading/ ./algo_trading/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Create runtime directories (state, logs, models persist via volumes)
RUN mkdir -p logs state models

# Non-root user for security
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# P1-FIX-32: Liveness check — probe the internal HTTP health endpoint.
# Does NOT import broker config or require secrets — pure process liveness.
# The trading system exposes /health on 127.0.0.1:8765 (loopback only).
# Falls back to a simple Python import if the server is not yet up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8765/health || python -c "import sys; sys.exit(0)"

CMD ["python", "-m", "algo_trading.main", "--mode", "paper"]
