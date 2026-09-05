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
PIPE    := $(PY) -m customer_intelligence.pipeline

.DEFAULT_GOAL := help
.PHONY: help setup generate quality build train all demo run test lint clean distclean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install the package
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PY) -e ".[dev]"
	@echo "Environment ready. Next: make all"

generate: ## Generate the synthetic source data and inject defects
	$(PIPE) generate

quality: ## Run the data quality assessment
	$(PIPE) quality

build: ## Build the conformed layer, monthly panel and Customer 360
	$(PIPE) build

train: ## Train the models, score the book and apply the decision engine
	$(PIPE) train

all: ## Run the whole pipeline end to end (50,000 customers)
	$(PIPE) all

demo: ## Same pipeline on a smaller population, for a quick look
	$(PIPE) all --customers 8000

run: ## Open the Streamlit application
	.venv/bin/streamlit run app/Home.py

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Check that every module imports cleanly
	$(PY) -c "import customer_intelligence, customer_intelligence.pipeline; print('imports clean')"

clean: ## Remove generated data and models, keep the environment
	rm -rf data/raw/*.parquet data/processed/* models/*
	@echo "Generated artefacts removed. Run make all to rebuild."

distclean: clean ## Also remove the virtual environment
	rm -rf .venv
