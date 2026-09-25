# 00 — Test the System (Hands-on Bring-up)

This guide gets the Enterprise AI Assistant running and proves each layer works.

**Machine note:** this environment uses `docker-compose` (hyphen), not `docker compose`.

---

## 1. Prerequisites

| Tool | Check |
|------|--------|
| Docker Engine | `docker info` |
| Docker Compose v2 | `docker-compose version` |
| Python 3.11+ | `python3 --version` |
| Node 20+ (UI) | `node --version` |
| OpenAI API key | real key in `.env` |

---

## 2. One-time setup

```bash
cd enterprise-ai-assistant

# .env is already created for local use with passwords aligned to compose (dev_password).
# REQUIRED: open .env and set a real OPENAI_API_KEY (placeholder will not call OpenAI).

# Optional helper:
./scripts/setup-local.sh
```

Password alignment (important):

- Compose defaults: `POSTGRES_PASSWORD=dev_password`, `NEO4J_PASSWORD=dev_password`
- Your `.env` must match when running apps on the host against Docker infra

Fixture for RAG demos:

- [`docs/fixtures/sample-rag-fact.txt`](../fixtures/sample-rag-fact.txt) — contains unique facts (Aurora revenue **47.3 million USD**, officer **Priya Natarajan**)

---

## 3. Path A — Infra only (recommended for learning)

```bash
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml ps
```

Starts: Postgres, Redis, Neo4j, Qdrant, MinIO.

Then on the host:

```bash
# Terminal 1 — API
cd backend
source venv/bin/activate   # or: python3 -m venv venv && pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Celery worker (required for document processing)
cd backend && source venv/bin/activate
celery -A app.workers.document_worker.celery_app worker \
  --loglevel=info --concurrency=2 --queues=document_processing

# Terminal 3 — Angular UI
cd frontend && npm install && npm start
```

---

## 4. Path B — Full Docker stack

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:4200 |
| API docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333/dashboard |
| Neo4j Browser | http://localhost:7474 (user `neo4j` / password from `.env`) |
| MinIO console | http://localhost:9001 |

If image pull fails with `TLS handshake timeout`, Docker Hub is unreachable — retry later or pull from an internal registry. Verification scripts still document the exact checks.

---

## 5. Verification checklist (in order)

### Automated

```bash
./scripts/verify-stack.sh
./scripts/verify-stack.sh --file docs/fixtures/sample-rag-fact.txt
# or:
python3 scripts/test-pipeline.py --file docs/fixtures/sample-rag-fact.txt
```

### Manual curls

```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/health/live | jq
curl -s http://localhost:8000/health/ready | jq
curl -s http://localhost:8000/api/eval/health | jq
```

Interpret `/api/eval/health`:

- `postgres`, `redis`, `qdrant`, `neo4j`, `minio` should be `"ok"` (or `"not_configured"` only if you intentionally skipped that service)

### What must be up for which feature

| Feature | Required |
|---------|----------|
| Simple chat (LLM only) | Backend + real `OPENAI_API_KEY` |
| RAG over uploads | + Redis, Celery worker, MinIO, Qdrant, Postgres |
| GraphRAG / Cypher | + Neo4j + successful entity extraction on ingest |

---

## 6. One RAG round-trip (UI)

1. Open http://localhost:4200 → **Documents**
2. Upload `docs/fixtures/sample-rag-fact.txt`
3. Wait until status is **complete** (poll or refresh)
4. Watch worker: `docker-compose logs -f worker` (full stack) or Celery terminal
5. Open **Chat** and ask:  
   `What was Project Aurora Q3 revenue?`
6. Expect answer containing **47.3** and citations
7. Open debug panel: intent ≈ `doc_question`, `retrieved_chunks_count` > 0, similarity scores

### API-only round-trip

```bash
# Upload
curl -s -F "file=@docs/fixtures/sample-rag-fact.txt" \
  http://localhost:8000/api/documents/upload | jq

# Poll (replace DOCUMENT_ID)
curl -s http://localhost:8000/api/documents/DOCUMENT_ID | jq

# Chat
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "hands-on",
    "session_id": "session-rag-1",
    "message": "What was Project Aurora Q3 revenue?",
    "document_ids": [],
    "stream": false
  }' | jq
```

---

## 7. Optional load test

```bash
pip install aiohttp
python3 scripts/load-test.py --url http://localhost:8000 --users 5 --duration 30
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Backend not reachable | Start uvicorn or `docker-compose up` |
| Upload stays `pending` | Celery worker not running / Redis down |
| Chat echo / “OpenAI not configured” | Set real `OPENAI_API_KEY` and restart backend |
| Empty retrieval | Wait for `complete`; check Qdrant collection `chunks` |
| Neo4j auth failed | Password must be `dev_password` (or match compose) |
| `docker compose` unknown | Use `docker-compose` |
| Image pull TLS timeout | Network/registry issue — retry or use mirror |

---

## 9. Status of bring-up on this machine (session note)

- `.env` created with compose-aligned passwords
- Fixture `docs/fixtures/sample-rag-fact.txt` added
- `scripts/verify-stack.sh` added
- Infra image pull from Docker Hub hit TLS timeout at time of writing — re-run section 3–6 when registry is reachable

Next: read [01-document-ingestion-e2e.md](01-document-ingestion-e2e.md).
