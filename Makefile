# Makefile — Developer shortcuts for the Enterprise AI Assistant project.
#
# WHAT IS A MAKEFILE?
#   A Makefile is a file of commands (called "targets") that you run with `make`.
#   Instead of remembering long docker/kubectl/pytest commands, you run:
#     make test      → runs all tests
#     make build     → builds all Docker images
#     make deploy    → deploys to Kubernetes
#     make clean     → stops everything and removes containers
#
# WHY USE MAKE?
#   - Shorter commands: `make test` vs `cd backend && pytest tests/ -v`
#   - Documented: new team members run `make help` to see all commands
#   - Consistent: everyone uses the same command regardless of OS
#   - Dependencies: `make deploy` automatically builds first
#
# HOW TO RUN:
#   make help          → list all available targets
#   make up            → start local infra (Postgres, Qdrant, Redis, etc.)
#   make test          → run all tests
#   make build         → build all Docker images
#   make monitoring    → start Prometheus + Grafana

.PHONY: help up down test test-backend test-langgraph test-frontend \
        build build-backend build-langgraph build-frontend \
        deploy deploy-local rollback lint lint-python lint-frontend \
        monitoring monitoring-down logs clean

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Enterprise AI Assistant — Developer Commands"
	@echo "============================================="
	@echo ""
	@echo "LOCAL DEVELOPMENT:"
	@echo "  make up              Start infra (Postgres, Redis, Qdrant, Neo4j, MinIO)"
	@echo "  make down            Stop all infra containers"
	@echo "  make up-full         Start complete stack (infra + apps)"
	@echo "  make langfuse-up     Start Langfuse (LLM observability)"
	@echo "  make monitoring      Start Prometheus + Grafana + Alertmanager"
	@echo "  make monitoring-down Stop monitoring stack"
	@echo ""
	@echo "RUNNING APPS (requires infra to be up):"
	@echo "  make backend         Run FastAPI backend on :8000"
	@echo "  make worker          Run Celery worker"
	@echo "  make langgraph       Run LangGraph agent on :7860"
	@echo "  make frontend        Run Angular frontend on :4200"
	@echo ""
	@echo "TESTING:"
	@echo "  make test            Run ALL tests"
	@echo "  make test-backend    Run backend pytest tests"
	@echo "  make test-langgraph  Run langgraph agent pytest tests"
	@echo "  make test-frontend   Run Angular tests"
	@echo "  make load-test       Run k6 load test"
	@echo ""
	@echo "CODE QUALITY:"
	@echo "  make lint            Lint all Python + TypeScript code"
	@echo "  make lint-python     Lint Python only (ruff)"
	@echo "  make lint-frontend   Lint Angular only (eslint)"
	@echo ""
	@echo "DOCKER:"
	@echo "  make build           Build all Docker images"
	@echo "  make build-backend   Build FastAPI backend image"
	@echo "  make build-langgraph Build LangGraph agent image"
	@echo "  make build-frontend  Build Angular frontend image"
	@echo ""
	@echo "KUBERNETES:"
	@echo "  make deploy-local    Apply all K8s manifests to local cluster (minikube)"
	@echo "  make rollback        Rollback all deployments to previous version"
	@echo "  make k8s-status      Show K8s pod/deployment status"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make logs-backend    Tail backend logs"
	@echo "  make logs-langgraph  Tail LangGraph agent logs"
	@echo "  make clean           Stop everything, remove containers and networks"
	@echo ""

# ── Local development ─────────────────────────────────────────────────────────
up:
	@echo "Starting local infra..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Infra ready. Ports: Postgres:5433 Redis:6379 Qdrant:6333 Neo4j:7474"

down:
	@echo "Stopping local infra..."
	docker-compose -f docker-compose.dev.yml down

up-full:
	@echo "Starting full stack..."
	docker-compose up -d --build

langfuse-up:
	@echo "Starting Langfuse..."
	cd langgraph-agent && docker-compose -f docker-compose.langfuse.yml up -d
	@echo "✅ Langfuse ready at http://localhost:3000"

monitoring:
	@echo "Starting monitoring stack..."
	docker-compose -f monitoring/docker-compose.monitoring.yml up -d
	@echo "✅ Prometheus:   http://localhost:9090"
	@echo "✅ Grafana:      http://localhost:3001 (admin/admin)"
	@echo "✅ Alertmanager: http://localhost:9093"

monitoring-down:
	docker-compose -f monitoring/docker-compose.monitoring.yml down

# ── App runners ───────────────────────────────────────────────────────────────
backend:
	cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	cd backend && source venv/bin/activate && celery -A app.workers.document_worker.celery_app worker --loglevel=info --concurrency=2 --queues=document_processing

langgraph:
	cd langgraph-agent && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

frontend:
	cd frontend && npm start

# ── Testing ───────────────────────────────────────────────────────────────────
test: test-backend test-langgraph

test-backend:
	@echo "Running backend tests..."
	cd backend && python -m pytest tests/ -v --tb=short

test-langgraph:
	@echo "Running LangGraph agent tests..."
	cd langgraph-agent && python -m pytest tests/ -v --tb=short

test-frontend:
	@echo "Running Angular tests..."
	cd frontend && ng test --watch=false --browsers=ChromeHeadless

test-coverage:
	@echo "Running tests with coverage..."
	cd backend && python -m pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo "Coverage report: backend/htmlcov/index.html"

load-test:
	@echo "Running k6 load test..."
	@echo "Install k6 first: https://k6.io/docs/getting-started/installation/"
	k6 run --vus 10 --duration 1m scripts/load-test-k6.js

# ── Code quality ──────────────────────────────────────────────────────────────
lint: lint-python lint-frontend

lint-python:
	@echo "Linting Python with ruff..."
	cd backend && ruff check app/ && ruff format app/ --check
	cd langgraph-agent && ruff check app/ && ruff format app/ --check
	@echo "✅ Python lint passed"

lint-frontend:
	@echo "Linting Angular with ESLint..."
	cd frontend && npx ng lint

format:
	@echo "Formatting Python code with ruff..."
	cd backend && ruff format app/
	cd langgraph-agent && ruff format app/

# ── Docker ───────────────────────────────────────────────────────────────────
build: build-backend build-langgraph build-frontend

build-backend:
	@echo "Building backend image..."
	docker build -t ai-backend:latest ./backend
	@echo "✅ ai-backend:latest built"

build-langgraph:
	@echo "Building LangGraph agent image..."
	docker build -t ai-langgraph-agent:latest ./langgraph-agent
	@echo "✅ ai-langgraph-agent:latest built"

build-frontend:
	@echo "Building frontend image..."
	docker build -t ai-frontend:latest ./frontend
	@echo "✅ ai-frontend:latest built"

build-worker:
	@echo "Building worker image..."
	docker build -t ai-worker:latest -f worker/Dockerfile .
	@echo "✅ ai-worker:latest built"

# ── Kubernetes ───────────────────────────────────────────────────────────────
deploy-local:
	@echo "Deploying to local Kubernetes (minikube)..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/postgres/
	kubectl apply -f k8s/redis/
	kubectl apply -f k8s/qdrant/
	kubectl apply -f k8s/neo4j/
	kubectl apply -f k8s/minio/
	kubectl apply -f k8s/backend/
	kubectl apply -f k8s/worker/
	kubectl apply -f k8s/langgraph-agent/
	kubectl apply -f k8s/frontend/
	kubectl apply -f k8s/ingress.yaml
	@echo "✅ All K8s manifests applied"
	@make k8s-status

k8s-status:
	@echo "=== Kubernetes Status ==="
	kubectl get pods -n ai-assistant
	@echo ""
	kubectl get deployments -n ai-assistant
	@echo ""
	kubectl get services -n ai-assistant

rollback:
	@echo "Rolling back all deployments..."
	kubectl rollout undo deployment/backend -n ai-assistant
	kubectl rollout undo deployment/langgraph-agent -n ai-assistant
	kubectl rollout undo deployment/worker -n ai-assistant
	kubectl rollout undo deployment/frontend -n ai-assistant
	@echo "✅ Rollback initiated"
	@make k8s-status

# ── Logs ─────────────────────────────────────────────────────────────────────
logs-backend:
	docker logs ai-backend -f --tail=100

logs-langgraph:
	docker logs ai-langgraph-agent -f --tail=100

logs-worker:
	docker logs ai-worker -f --tail=100

logs-k8s-backend:
	kubectl logs -n ai-assistant -l app=backend -f --tail=100

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Stopping all containers..."
	docker-compose down --remove-orphans || true
	docker-compose -f docker-compose.dev.yml down --remove-orphans || true
	docker-compose -f monitoring/docker-compose.monitoring.yml down --remove-orphans || true
	cd langgraph-agent && docker-compose -f docker-compose.langfuse.yml down --remove-orphans || true
	@echo "✅ All containers stopped"

clean-all: clean
	@echo "Removing Docker images and volumes..."
	docker rmi ai-backend:latest ai-langgraph-agent:latest ai-frontend:latest ai-worker:latest 2>/dev/null || true
	docker system prune -f
	@echo "✅ Cleanup complete"

# ── DB utilities ──────────────────────────────────────────────────────────────
apply-schema:
	@echo "Applying LangGraph memory schema..."
	PGPASSWORD=dev_password psql -h localhost -p 5433 -U ai_user -d ai_assistant \
	  -f langgraph-agent/app/database/schema.sql
	@echo "✅ Schema applied"

seed-neo4j:
	cd backend && python scripts/seed-neo4j.py
