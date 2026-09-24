# Pythia prediction service — common tasks. `make` or `make help` lists them.

PYTHON_BIN ?= python3.14
VENV       := .venv
PY         := $(VENV)/bin/python
PORT       ?= 8000
LOG_LEVEL  ?= INFO

# The provided run-*.sh scripts call a bare `python`; putting the venv first
# on PATH makes them use the project interpreter without editing them.
VENV_PATH  := PATH="$(CURDIR)/$(VENV)/bin:$$PATH"

.DEFAULT_GOAL := help
.PHONY: help install install-dev test check run run-debug e2e e2e-quick run-v1 run-v2 run-v2-optimized clean

help: ## List available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(PY):
	$(PYTHON_BIN) -m venv $(VENV)

install: $(PY) ## Create the venv (Python 3.14) and install requirements
	$(PY) -m pip install -r requirements.txt

install-dev: $(PY) ## Also install the test dependencies (pytest, httpx)
	$(PY) -m pip install -r requirements-dev.txt

test: ## Unit tests: seconds, no library calls, no server
	$(PY) -m pytest

check: test e2e ## Unit tests, then the full end-to-end run

run: ## Start the API on PORT (default 8000), single worker
	PYTHONUNBUFFERED=1 PYTHONPATH=. LOG_LEVEL=$(LOG_LEVEL) \
		$(PY) -m uvicorn pythia_service.api.main:app --workers 1 --port $(PORT)

run-debug: ## Start the API with DEBUG logs (every stage transition)
	$(MAKE) run LOG_LEVEL=DEBUG

e2e: ## End-to-end check, full catalog (~2-3 min)
	$(PY) tests/e2e.py

e2e-quick: ## End-to-end check, 2 happy segments (~1-2 min)
	$(PY) tests/e2e.py --quick

run-v1: ## DS reference script, library v1 -> results-v1.csv
	$(VENV_PATH) ./run-v1.sh

run-v2: ## Sequential baseline, library v2 -> results-v2.csv
	$(VENV_PATH) ./run-v2.sh

run-v2-optimized: ## Part 2 optimized script, library v2
	$(VENV_PATH) ./run-v2-optimized.sh

clean: ## Remove caches and e2e artifacts
	rm -rf .e2e
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -exec rm -rf {} +
