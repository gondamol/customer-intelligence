# Customer Intelligence & Decision Analytics
#
#   make setup           create the virtual environment and install
#   make all             generate, assess quality, build, model and score
#   make run             open the application
#   make test            run the test suite
#
# `make all` on 50,000 customers takes a few minutes. `make demo` builds a
# smaller population if you only want to look around.

PY      := .venv/bin/python
UV      := $(shell command -v uv 2>/dev/null || echo "$$HOME/.local/bin/uv")
PIPE    := $(PY) -m customer_intelligence.pipeline_retail
SYNTH   := $(PY) -m customer_intelligence.pipeline

.DEFAULT_GOAL := help
.PHONY: help setup fetch land quality build train all validate-checks run test lint clean distclean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install the package
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PY) -e ".[dev]"
	@echo "Environment ready. Next: make all"

fetch: ## Download the published source data (UCI, IBM) and cache it
	$(PIPE) fetch

land: ## Land the source tables as published, defects intact
	$(PIPE) land

quality: ## Run the data quality assessment
	$(PIPE) quality

build: ## Build the conformed layer, monthly panel and Customer 360
	$(PIPE) build

train: ## Train the models, score the book and apply the decision engine
	$(PIPE) train

all: ## Run the whole pipeline end to end on the real data
	$(PIPE) all

validate-checks: ## Prove the quality checks work, against known injected defects
	$(SYNTH) all --customers 8000

run: ## Open the Streamlit application
	.venv/bin/streamlit run app/Home.py

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Check that every module imports cleanly
	$(PY) -c "import customer_intelligence, customer_intelligence.pipeline; print('imports clean')"

clean: ## Remove generated artefacts, keep the downloaded source data
	rm -rf data/raw/*.parquet data/processed/* models/*.joblib models/*.json

clean-all: clean ## Also drop the cached downloads
	rm -rf data/external
	@echo "Generated artefacts removed. Run make all to rebuild."

distclean: clean ## Also remove the virtual environment
	rm -rf .venv
