# NSE AlgoTrader — Common Commands
# Usage: make <target>
# Windows: use `nmake` or run commands manually from the recipes

PYTHON = python
MODULE = algo_trading

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-stage3:
	pip install "scipy>=1.12" "numba>=0.59" "msgpack>=1.0"

install-stage4:
	pip install "gymnasium>=0.29" "stable-baselines3>=2.3"
	# For FinBERT sentiment (heavy, optional):
	# pip install "transformers>=4.40" "torch>=2.2"

install-dev:
	pip install pytest>=8.0 pytest-asyncio>=0.23

# ── Data ──────────────────────────────────────────────────────────────────────

sample-data:
	$(PYTHON) -m $(MODULE).scripts.generate_sample_data

download-history:
	$(PYTHON) -m $(MODULE).scripts.download_history

download-fo:
	$(PYTHON) -m $(MODULE).scripts.download_fo_history --synthetic

# ── Validation & Training ─────────────────────────────────────────────────────

validate-edge:
	$(PYTHON) -m $(MODULE).scripts.validate_edge

audit-readiness:
	$(PYTHON) scripts/audit_readiness.py

train:
	$(PYTHON) -m $(MODULE).scripts.train_models

train-force:
	$(PYTHON) -m $(MODULE).scripts.train_models --force

train-rl:
	$(PYTHON) -m $(MODULE).scripts.train_rl

# ── Live Session Runners ──────────────────────────────────────────────────────

run:                ## Run equity daily orchestrator (NIFTY + BANKNIFTY)
	$(PYTHON) -m $(MODULE).orchestrator.daily_orchestrator

run-fo:             ## Run F&O orchestrator — NIFTY 21 DTE (dry-run safe to test)
	$(PYTHON) -m $(MODULE).scripts.run_fo --symbol NIFTY --dte 21

run-fo-banknifty:   ## Run F&O orchestrator — BANKNIFTY 7 DTE
	$(PYTHON) -m $(MODULE).scripts.run_fo --symbol BANKNIFTY --dte 7

run-fo-dry:         ## Dry-run: print F&O config, no broker connections
	$(PYTHON) -m $(MODULE).scripts.run_fo --symbol NIFTY --dte 21 --dry-run

# ── Paper Trading ─────────────────────────────────────────────────────────────

paper:
	$(PYTHON) -m $(MODULE).scripts.paper_trade

paper-30d:
	$(PYTHON) -m $(MODULE).scripts.paper_trade --duration-days 30

# ── Infrastructure ────────────────────────────────────────────────────────────

questdb-start:
	docker run -d -p 9000:9000 -p 9009:9009 --name questdb questdb/questdb

questdb-stop:
	docker stop questdb

mlflow-ui:
	mlflow ui --port 5000

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not integration and not slow and not stage3 and not stage4"

test-integration:
	pytest tests/ -v --tb=short -m integration

test-stage3:
	pytest tests/ -v --tb=short -m stage3

test-stage4:
	pytest tests/ -v --tb=short -m stage4

test-cov:
	pytest tests/ --cov=$(MODULE) --cov-report=html --cov-report=term-missing

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	ruff check $(MODULE)/

lint-ruff:          ## Run ruff linter
	ruff check $(MODULE)/

format:             ## Auto-format with ruff
	ruff format $(MODULE)/

format-check:       ## Check formatting without modifying (CI)
	ruff format --check $(MODULE)/

typecheck:
	mypy $(MODULE)/ --ignore-missing-imports

quality: lint-ruff typecheck format-check  ## Run all quality checks

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

# ── Full Pipeline (run in sequence) ───────────────────────────────────────────

pipeline-dev: sample-data validate-edge train
	@echo "Dev pipeline complete. Run 'make paper' to start paper trading."

.PHONY: install install-stage3 install-stage4 install-dev sample-data \
        download-history download-fo validate-edge audit-readiness train train-force \
        train-rl run run-fo run-fo-banknifty run-fo-dry \
        paper paper-30d questdb-start questdb-stop mlflow-ui \
        test test-fast test-integration test-stage3 test-stage4 test-cov \
        lint lint-ruff format format-check typecheck quality \
        pre-commit-install pipeline-dev
