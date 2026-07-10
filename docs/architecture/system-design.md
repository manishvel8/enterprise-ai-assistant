# System Design — Enterprise AI Chat Assistant

## Design Principles

1. **Decouple heavy work from the API** — Document processing is async (Celery), so uploading a 100-page PDF never blocks the chat API
2. **Three knowledge stores serve different query types** — Vector DB for semantic similarity, Neo4j for relationship traversal, PostgreSQL for structured metadata
3. **Agents are pure functions** — Each agent takes the shared AgentState and returns an updated AgentState. No hidden side effects
4. **Observe everything** — Every LLM call, retrieval call, and agent transition is traced in Langfuse
5. **Security by default** — No hardcoded secrets, Cypher queries are validated before execution, file uploads are validated before storage

---

## Architecture Decision Records (ADRs)

### ADR-001: Why FastAPI over Django/Flask?

**Decision:** Use FastAPI as the backend framework.

**Rationale:**
- Native async support — essential for streaming SSE responses and concurrent agent steps
- Pydantic integration — request/response validation is automatic from type hints
- Built-in OpenAPI docs — `/docs` endpoint available immediately for testing
- Performance — FastAPI is 2–3x faster than Flask/Django for async workloads

**Trade-off:** FastAPI has a smaller ecosystem than Django. We accept this because we are not building a traditional web app.

---

### ADR-002: Why Qdrant over Pinecone/Weaviate?

**Decision:** Use Qdrant as the vector database.

**Rationale:**
- Fully open-source, can run locally and in production without cloud dependency
- Excellent performance on HNSW index
- Metadata filtering before vector search — reduces search space, improves accuracy
- Docker image available — consistent local and production deployment

**Trade-off:** Qdrant does not have a managed cloud tier as mature as Pinecone. For production at scale, consider Pinecone or Weaviate Cloud.

---

### ADR-003: Why Neo4j for the graph layer?

**Decision:** Use Neo4j as the graph database.

**Rationale:**
- Native graph storage — no ORM or joins needed for relationship traversal
- Cypher query language is expressive and readable
- Free Community Edition available for local development
- GraphRAG pattern requires rich relationship traversal that relational DBs cannot express naturally

**Trade-off:** Neo4j Community Edition has no clustering support. For high-availability production, use Neo4j Enterprise or AuraDB.

---

### ADR-004: Why Celery over FastAPI BackgroundTasks?

**Decision:** Use Celery with Redis for document processing, not FastAPI's built-in BackgroundTasks.

**Rationale:**
- FastAPI BackgroundTasks run in the same process — if the server crashes, the task is lost
- Celery tasks persist in Redis — they survive server restarts
- Celery supports task retry with exponential backoff
- Worker pods scale independently from API pods — heavy document processing doesn't compete with chat API resources
- Task status can be polled by the frontend

**Trade-off:** Adds Redis and Celery as dependencies. Acceptable because Redis is also used for caching.

---

### ADR-005: Why LangGraph over a custom agent loop?

**Decision:** Use LangGraph for agent orchestration (introduced in Milestone 22).

**Rationale:**
- Provides a stateful graph with typed nodes and edges
- Supports conditional routing (Router Agent decides which agents to call)
- Supports loops (Critic Agent can route back to Answer Agent)
- Built-in streaming support
- Compatible with LangChain components (retrievers, prompt templates, output parsers)

**Trade-off:** LangGraph adds abstraction complexity. We build the agents manually in earlier milestones to understand the patterns before adding LangGraph.

---

## Scaling Strategy

### Stateless Services (scale horizontally)

| Service | Min Pods | Max Pods | Scale Trigger |
|---------|----------|----------|---------------|
| Backend (FastAPI) | 3 | 20 | CPU > 70% |
| Worker (Celery) | 2 | 10 | Redis queue depth > 5 tasks |
| Frontend (Nginx) | 2 | 5 | CPU > 80% |

### Stateful Services (scale vertically or with replicas)

| Service | Strategy |
|---------|----------|
| PostgreSQL | Single primary for writes, read replica for analytics |
| Neo4j | Community: single node. Enterprise: causal clustering |
| Qdrant | Single node; horizontal sharding available in cluster mode |
| Redis | Single node for dev; Redis Sentinel or Cluster for production HA |

---

## Request Latency Budget

Target: P95 < 3 seconds for a typical RAG response

| Stage | Target Latency |
|-------|---------------|
| Routing (GPT-4o mini) | ~200ms |
| Query rewrite (GPT-4o mini) | ~300ms |
| Vector search (Qdrant) | ~50ms |
| Context building | ~10ms |
| Answer generation (GPT-4o) | ~1500ms (streaming starts at ~500ms) |
| Critic validation (GPT-4o mini) | ~300ms |
| **Total (streaming start)** | **~1000ms** |
| **Total (full response)** | **~2500ms** |

Cache hit path (embedding cache + retrieval cache): < 1 second

---

## Security Architecture

```
Internet → Ingress (TLS termination)
                ↓
          Angular (SPA, no secrets)
                ↓
          FastAPI (Auth middleware)
            ↓         ↓
      Validated    Rejected
      requests     requests (401/403)
          ↓
    Rate limiter
          ↓
    Agent workflow
          ↓
    Databases (internal network only, not exposed to internet)
```

**Key security controls:**
- API keys stored in Kubernetes Secrets, injected as env vars, never logged
- File uploads: type validation, size limit, filename sanitization
- Cypher queries: allowlist of permitted keywords, deny-list of destructive operations, read-only DB user
- All database connections are internal to the Kubernetes cluster
- CORS: only allow known frontend origins
- Rate limiting: per-user and global

---

## Data Flow Diagrams

### Chat Request (Happy Path)

```
Angular                FastAPI               Agents              External
   |                      |                     |                    |
   |--POST /chat--------->|                     |                    |
   |                      |--create AgentState->|                    |
   |                      |                     |--Router----------->|OpenAI (classify)
   |                      |                     |<--intent-----------|
   |                      |                     |--Rewrite---------->|OpenAI (rewrite)
   |                      |                     |<--rewritten_query--|
   |                      |                     |--Retriever-------->|Qdrant (search)
   |                      |                     |<--chunks-----------|
   |                      |                     |--ContextBuilder    |
   |                      |                     |--Answer----------->|OpenAI (generate)
   |<---SSE stream--------|<--stream chunks-----|                    |
   |                      |                     |--Critic----------->|OpenAI (validate)
   |                      |                     |--Memory (save)     |
   |<---citations---------|                     |                    |
   |                      |--Langfuse trace---->|                    |Langfuse
```

### Document Upload (Happy Path)

```
Angular             FastAPI                Redis             Celery Worker
   |                   |                     |                    |
   |--POST /upload/--->|                     |                    |
   |                   |--validate file      |                    |
   |                   |--save to MinIO      |                    |
   |                   |--insert PostgreSQL  |                    |
   |                   |--enqueue task------>|                    |
   |<--202 Accepted----|                     |                    |
   |                   |                     |--dequeue task----->|
   |                   |                     |                    |--parse
   |                   |                     |                    |--normalize
   |                   |                     |                    |--chunk
   |                   |                     |                    |--embed (OpenAI)
   |                   |                     |                    |--store vectors (Qdrant)
   |                   |                     |                    |--store graph (Neo4j)
   |                   |                     |                    |--update PostgreSQL status
   |--GET /documents/status/{id}------------>|                    |
   |<--{"status": "complete"}----------------|                    |
```
