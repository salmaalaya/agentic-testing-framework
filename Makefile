# Convenience targets (Linux/macOS/Git Bash). On Windows PowerShell, run the commands directly.
PYTHON ?= python

.PHONY: install install-optional install-browsers app pipeline pipeline-all report regression-baseline regression-check test lint format check clean

install:            ## Core + dev dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

install-browsers:   ## Chromium for the Web channel (Playwright)
	$(PYTHON) -m playwright install chromium

install-optional:   ## NLI (torch CPU + transformers), Appium client, voice
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(PYTHON) -m pip install -r requirements-optional.txt

app:                ## Start the fictional TechShop chatbot on :8000
	$(PYTHON) -m uvicorn target_app.app:app --reload --port 8000

pipeline:           ## API channel with the curated corpus
	$(PYTHON) run_pipeline.py --skip-generator --no-check

pipeline-all:       ## API + Web + Mobile, then dashboard and PDF report
	$(PYTHON) run_pipeline.py --canal all --skip-generator --no-check --dashboard --pdf

report:             ## Rebuild dashboard + PDF from the last results
	$(PYTHON) scripts/generate_dashboard.py
	$(PYTHON) scripts/generate_pdf_report.py

regression-baseline:
	$(PYTHON) scripts/compare_versions.py --snapshot baseline

regression-check:
	$(PYTHON) scripts/compare_versions.py --compare baseline

test:               ## Deterministic unit tests (no Ollama needed)
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

check:              ## Verify the local environment (Ollama, models, browsers, Android SDK)
	$(PYTHON) scripts/check_setup.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
