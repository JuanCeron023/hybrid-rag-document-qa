# DocSage — build, test, and dev tasks.
# Run `make help` for the full list.

VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Backend (Python / FastAPI) ------------------------------------------

$(VENV): ## Create the virtualenv
	python3 -m venv $(VENV)

.PHONY: install
install: $(VENV) ## Install backend dependencies (+ pytest)
	$(PIP) install -q -r requirements.txt pytest

.PHONY: run
run: ## Run the FastAPI server
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: test
test: ## Run tests
	$(PY) -m pytest -q

## --- Frontend (React + TypeScript) ---------------------------------------

.PHONY: frontend-install
frontend-install: ## Install frontend dependencies
	cd frontend && npm install

.PHONY: frontend-build
frontend-build: ## Build the React frontend into web/dist
	cd frontend && npm run build

.PHONY: frontend-dev
frontend-dev: ## Run the Vite dev server (proxies API to the backend)
	cd frontend && npm run dev

## --- Aggregate ------------------------------------------------------------

.PHONY: check
check: install test ## Install deps and run tests

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t docsage .

.PHONY: up
up: ## Start the stack with docker compose
	docker compose up --build

.PHONY: clean
clean: ## Remove venv, caches, and frontend artifacts
	rm -rf $(VENV) .pytest_cache **/__pycache__ web/dist frontend/node_modules
