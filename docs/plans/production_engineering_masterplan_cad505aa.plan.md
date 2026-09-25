---
name: Production Engineering Masterplan
overview: A 16-point production engineering implementation across CI/CD, Kubernetes, monitoring, security, testing, and deployment strategy—building on top of the existing dual-backend (FastAPI :8000 + LangGraph :7860), Angular frontend, multi-DB stack, and raw K8s manifests without replacing anything already working.
todos:
  - id: section2-testing
    content: "Create full pytest test suites: backend/tests/ (unit + integration + RAG quality) and langgraph-agent/tests/ (unit + integration); create scripts/load-test-k6.js"
    status: in_progress
  - id: section3-auth
    content: "Implement real JWT auth: backend/app/core/auth.py, backend/app/api/auth.py (login/refresh routes), protect /api/chat and /api/documents with Depends(get_current_user)"
    status: pending
  - id: section4-langgraph-docker
    content: Create langgraph-agent/Dockerfile (python:3.11-slim, non-root, healthcheck); add langgraph-agent service to docker-compose.yml
    status: pending
  - id: section5-k8s-helm
    content: Create k8s/langgraph-agent/ manifests (deployment, service, hpa, configmap); update k8s/ingress.yaml; create helm/ai-assistant/ chart wrapping all k8s resources
    status: pending
  - id: section6-monitoring
    content: "Create monitoring/ stack: prometheus.yml, alertmanager.yml, grafana dashboards, docker-compose.monitoring.yml; add prometheus_fastapi_instrumentator to both FastAPI apps"
    status: pending
  - id: section7-cicd
    content: Create .github/workflows/ci.yml (lint+test+scan+build+push) and cd.yml (kubectl apply + smoke); create Makefile with dev shortcuts
    status: pending
  - id: section10-guardrails
    content: "Create backend/app/core/guardrails.py and langgraph-agent/app/core/guardrails.py: input length, prompt injection regex, integrate into query_validator node"
    status: pending
  - id: section11-checkpointer
    content: Swap MemorySaver for AsyncPostgresSaver in langgraph-agent/app/graph/workflow.py and update main.py lifespan
    status: pending
  - id: section12-deploy-strategies
    content: Create scripts/blue-green-deploy.sh, canary-deploy.sh, rollback.sh, verify-deployment.sh with full comments
    status: pending
  - id: section14-security
    content: Add .github/workflows/security.yml (Bandit + Trivy + pip-audit); create k8s/network-policy.yaml; tighten CORS on langgraph-agent
    status: pending
  - id: section16-html-guide
    content: Create docs/hands-on/06-production-engineering.html — single self-contained 16-tab interactive guide covering all sections with commands, diagrams, exercises, and the 20-step final hands-on
    status: pending
isProject: false
---

# Production Engineering Masterplan — Enterprise AI Assistant

## Codebase State (from analysis)

### Already Implemented (do not duplicate)
- FastAPI backend :8000 with 9-agent RAG, Celery, Postgres/Qdrant/Neo4j/Redis/MinIO
- LangGraph agent :7860 with 7-node graph, HITL, MCP, memory layer, Langfuse SDK
- Angular 19 frontend :4200 with SSE streaming, document management
- Dockerfiles for backend, frontend, worker
- `docker-compose.yml` (full stack), `docker-compose.dev.yml` (infra), `docker-compose.langfuse.yml`
- Raw K8s manifests: `k8s/` — namespace, configmap, secret, ingress, backend/worker/frontend deployments+HPAs, PostgreSQL/Redis/Neo4j/Qdrant/MinIO StatefulSets
- Langfuse self-hosted v3 (LLM observability) — SDK in both apps
- Extensive docs: `docs/hands-on/01–05`, architecture HTML, interview guides

### Critical Gaps (what we build)
- No CI/CD at all (`.github/workflows/` absent)
- No Prometheus/Grafana/Alertmanager stack
- No automated pytest suites (`langgraph-agent/tests/` empty; no `backend/tests/`)
- No JWT/OAuth implementation (config exists, code is a placeholder)
- No Dockerfile or K8s manifests for `langgraph-agent`
- No Helm charts
- No LLM guardrails / prompt injection protection
- No container security scanning or SAST
- `langgraph-agent` uses in-memory `MemorySaver` (not durable across restarts)

---

## Gap Analysis Table

| Area | Implemented | Missing | Priority |
|------|-------------|---------|----------|
| Auth | JWT config + stub `get_current_user()` | Actual JWT encode/decode, login route, protected endpoints | High |
| Testing | Smoke + load scripts, stale Angular specs | pytest unit/integration/API suites; k6 load tests | High |
| CI/CD | Nothing | GitHub Actions pipeline (lint→test→build→scan→push→deploy) | High |
| Langgraph Docker/K8s | None | `langgraph-agent/Dockerfile`, `k8s/langgraph-agent/` manifests | High |
| Prometheus/Grafana | None | FastAPI metrics endpoint, Prometheus scrape, Grafana dashboards, Alertmanager | High |
| LLM Guardrails | None | Input length/injection guard in both backends | Medium |
| Durable checkpointer | MemorySaver (volatile) | PostgreSQL-backed `AsyncPostgresSaver` for LangGraph | Medium |
| Helm charts | None | `helm/ai-assistant/` chart wrapping existing k8s YAMLs | Medium |
| Deployment strategies | Rolling only (K8s default) | Blue/green + canary patterns documented + implemented | Medium |
| Security scanning | Basic CORS, rate limit, upload checks | Bandit SAST, Trivy container scan, dependabot in CI | Medium |
| Secrets management | `.env` + K8s secret template | Production vault pattern (K8s Secrets + sealed-secrets or external-secrets) | Low |
| Alembic migrations | Scaffold only | Actual revision for existing models | Low |

---

## Implementation Plan (16 Sections)

### Section 1 — Gap Analysis & Architecture Review
Deliverable: enriched `docs/architecture/gap-analysis.md` + updated `system-design.html`

### Section 2 — Automated Testing (pytest + k6)

**Files to create:**
- `backend/tests/conftest.py` — pytest fixtures (async DB, mock OpenAI, mock Qdrant)
- `backend/tests/unit/test_chunker.py`, `test_embedder.py`, `test_agents.py`
- `backend/tests/integration/test_chat_api.py`, `test_document_api.py`, `test_health.py`
- `backend/tests/test_rag_quality.py` — retrieval precision/recall checks
- `langgraph-agent/tests/conftest.py`
- `langgraph-agent/tests/unit/test_nodes.py`, `test_memory.py`, `test_langfuse.py`
- `langgraph-agent/tests/integration/test_chat_flow.py`
- `scripts/load-test-k6.js` — k6 script (replaces basic `load-test.py`)

### Section 3 — JWT Authentication

**Files to create/modify:**
- `backend/app/core/auth.py` — `create_access_token()`, `verify_token()`, `get_current_user()` (real implementation using `python-jose`)
- `backend/app/api/auth.py` — `POST /api/auth/login`, `POST /api/auth/refresh`
- Modify `backend/app/api/chat.py` — add `Depends(get_current_user)` to protected routes
- Modify `backend/app/core/config.py` — ensure `SECRET_KEY`, `JWT_ALGORITHM` wired

### Section 4 — Dockerfile for LangGraph Agent

**Files to create:**
- `langgraph-agent/Dockerfile` — `python:3.11-slim`, non-root user, healthcheck `/api/health`
- Update `docker-compose.yml` — add `langgraph-agent` service (port 7860, depends on postgres/qdrant/redis)

### Section 5 — Kubernetes Manifests for LangGraph Agent + Helm

**Files to create:**
- `k8s/langgraph-agent/deployment.yaml` — with liveness/readiness/startup probes
- `k8s/langgraph-agent/service.yaml`
- `k8s/langgraph-agent/hpa.yaml` — scale on CPU + custom Langfuse metrics
- `k8s/langgraph-agent/configmap.yaml`
- Update `k8s/ingress.yaml` — add `/lg-api/` → langgraph-agent route
- `helm/ai-assistant/Chart.yaml`, `values.yaml`, `templates/` wrapping all `k8s/` resources

### Section 6 — Prometheus + Grafana + Alertmanager

**Files to create:**
- `monitoring/prometheus.yml` — scrape configs for backend :9090, langgraph :9091, postgres-exporter, redis-exporter, node-exporter
- `monitoring/alertmanager.yml` — alert routing rules (email/Slack)
- `monitoring/alerts/ai-assistant.yml` — alert rules (error rate, latency, pod restarts, LLM cost)
- `monitoring/grafana/dashboards/ai-assistant.json` — pre-built dashboard JSON
- `monitoring/grafana/provisioning/` — datasources.yml, dashboards.yml
- `monitoring/docker-compose.monitoring.yml` — prometheus, grafana, alertmanager, node-exporter, redis-exporter, postgres-exporter
- Modify `backend/app/main.py` — add `prometheus_fastapi_instrumentator`
- Modify `langgraph-agent/app/main.py` — same

### Section 7 — CI/CD Pipeline (GitHub Actions)

**Files to create:**
- `.github/workflows/ci.yml` — lint (ruff, eslint) → pytest → k6 smoke → Trivy scan → Docker build+push → notify
- `.github/workflows/cd.yml` — triggered on `main` merge → `kubectl apply` + smoke test → Slack notify
- `.github/workflows/security.yml` — weekly Bandit + Trivy + dependency audit
- `Makefile` — developer shortcuts (`make test`, `make build`, `make deploy`, `make lint`)

### Section 8 — CI Testing (pass/fail examples)

Deliverable: Section in HTML guide showing:
- Green pipeline screenshot path + what each stage output looks like
- Intentional test failure (assert wrong status code), lint failure (unused import), Docker build failure (bad dependency) — with expected CI output

### Section 9 — CD Testing & Deployment Verification

Deliverable: `scripts/verify-deployment.sh` — checks pods, readiness probes, API health, DB connectivity, RAG ping, Redis ping, LLM reachability

### Section 10 — LLM Guardrails + Prompt Injection Protection

**Files to create:**
- `backend/app/core/guardrails.py` — input length limit, injection pattern regex, PII check stub, `validate_query()` function
- `langgraph-agent/app/core/guardrails.py` — same, used in `query_validator` node
- Modify `query_validator.py` — wrap LLM call with guardrail pre-check

### Section 11 — Durable LangGraph Checkpointer

**Files to modify:**
- `langgraph-agent/app/graph/workflow.py` — swap `MemorySaver` for `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`
- `langgraph-agent/app/main.py` — async setup of checkpointer in lifespan

### Section 12 — Deployment Strategies Documentation + Scripts

**Files to create:**
- `scripts/blue-green-deploy.sh` — two-namespace pattern using existing K8s manifests
- `scripts/canary-deploy.sh` — weighted Ingress rule using nginx `canary` annotations
- `scripts/rollback.sh` — `kubectl rollout undo deployment`
- Intentional broken deployment + rollback exercise in HTML guide

### Section 13 — Scaling + Load Testing

**Files to create/modify:**
- `scripts/load-test-k6.js` — 5 stages (1→10→100→500→1000 VUs), chat + upload + RAG scenarios
- `k8s/backend/hpa.yaml` — update to scale 2→20 replicas on CPU + requests/s
- `k8s/langgraph-agent/hpa.yaml` — same

### Section 14 — Security Hardening

**Files to create/modify:**
- `.github/workflows/security.yml` — Bandit SAST + Trivy image scan + pip-audit
- `backend/app/core/guardrails.py` — (same as Section 10)
- `k8s/network-policy.yaml` — deny all ingress except allowed service-to-service
- `langgraph-agent/app/main.py` — CORS tightened from `*` to explicit origins

### Section 15 — Final Architecture Diagram

Deliverable: Updated `docs/architecture/system-design.html` with full post-implementation flow diagram

### Section 16 — Final Hands-On HTML Guide

**File to create:**
- `docs/hands-on/06-production-engineering.html` — single self-contained file (~150KB), 16 interactive tabs covering all sections above with:
  - Architecture diagrams (ASCII + Mermaid-like SVG)
  - Step-by-step exercises (all 20 steps from the user's final exercise)
  - Copy-paste commands with expected outputs
  - CI/CD pipeline visualization
  - Kubernetes resource explanations
  - Grafana dashboard walkthrough
  - Load test results interpretation
  - Security checklist

---

## File Summary (new files being created)

**Testing (8 files):**
- `backend/tests/conftest.py`, `unit/test_*.py` (3), `integration/test_*.py` (2), `test_rag_quality.py`
- `langgraph-agent/tests/conftest.py`, `unit/test_*.py` (3), `integration/test_chat_flow.py`
- `scripts/load-test-k6.js`

**Auth (2 files):**
- `backend/app/core/auth.py`, `backend/app/api/auth.py`

**Docker/K8s (6 files):**
- `langgraph-agent/Dockerfile`
- `k8s/langgraph-agent/deployment.yaml`, `service.yaml`, `hpa.yaml`, `configmap.yaml`
- `k8s/network-policy.yaml`

**Helm (4 files):**
- `helm/ai-assistant/Chart.yaml`, `values.yaml`, `templates/_helpers.tpl`, `templates/*.yaml`

**Monitoring (8 files):**
- `monitoring/prometheus.yml`, `alertmanager.yml`, `alerts/ai-assistant.yml`
- `monitoring/grafana/dashboards/ai-assistant.json`
- `monitoring/grafana/provisioning/datasources.yml`, `dashboards.yml`
- `monitoring/docker-compose.monitoring.yml`

**CI/CD (4 files):**
- `.github/workflows/ci.yml`, `cd.yml`, `security.yml`
- `Makefile`

**Guardrails/Security (2 files):**
- `backend/app/core/guardrails.py`
- `langgraph-agent/app/core/guardrails.py`

**Scripts (3 files):**
- `scripts/blue-green-deploy.sh`, `scripts/canary-deploy.sh`, `scripts/rollback.sh`, `scripts/verify-deployment.sh`

**HTML Guide (1 file):**
- `docs/hands-on/06-production-engineering.html`

**Modified files:**
- `backend/app/main.py` — Prometheus instrumentator
- `backend/app/api/chat.py` — JWT protection
- `langgraph-agent/app/main.py` — Prometheus + durable checkpointer + CORS tightening
- `langgraph-agent/app/graph/workflow.py` — AsyncPostgresSaver
- `langgraph-agent/app/agents/query_validator.py` — guardrail integration
- `docker-compose.yml` — add langgraph-agent service
- `k8s/ingress.yaml` — add langgraph-agent route

---

## Port Map After Implementation

| Service | Port |
|---------|------|
| Angular frontend | 4200 |
| FastAPI backend | 8000 |
| LangGraph agent | 7860 |
| Prometheus | 9090 |
| Grafana | 3001 |
| Alertmanager | 9093 |
| Langfuse | 3000 |
| PostgreSQL | 5433 |
| Qdrant | 6333 |
| Redis | 6379 |
| Neo4j Browser | 7474 |
| MinIO Console | 9001 |
