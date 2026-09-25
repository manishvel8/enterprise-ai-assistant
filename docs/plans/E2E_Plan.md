---
name: Enterprise AI Chat Assistant
overview: Build a complete enterprise-grade AI Chat Assistant with RAG, GraphRAG, agentic workflow, Neo4j, OpenAI, Langfuse, Docker, and Kubernetes — implemented milestone by milestone for learning and AI Engineer interview preparation. Milestone 1 establishes the full architecture, folder structure, component map, and deployment strategy before any code is written.
todos:
  - id: m1-arch
    content: "Milestone 1: Review and confirm architecture, folder structure, and component map"
    status: completed
  - id: m2-skeleton
    content: "Milestone 2: Create Angular frontend and FastAPI backend skeleton"
    status: completed
  - id: m3-chat-ui
    content: "Milestone 3: Build simple chat UI and basic backend chat API"
    status: completed
  - id: m4-openai
    content: "Milestone 4: Integrate OpenAI API for simple chat"
    status: completed
  - id: m5-dockerfiles
    content: "Milestone 5: Create Dockerfiles for frontend, backend, and worker"
    status: completed
  - id: m6-compose
    content: "Milestone 6: Create docker-compose.yml for full local dev stack"
    status: completed
  - id: m7-upload
    content: "Milestone 7: Add document upload screen and backend upload API"
    status: completed
  - id: m8-postgres
    content: "Milestone 8: Add PostgreSQL metadata storage with SQLAlchemy"
    status: completed
  - id: m9-celery
    content: "Milestone 9: Add Redis and Celery worker for background processing"
    status: completed
  - id: m10-parsers
    content: "Milestone 10: Implement document parsers (PDF, DOCX, PPTX, CSV, Excel, Image, Audio, Video)"
    status: completed
  - id: m11-normalize
    content: "Milestone 11: Normalize all parsed content into common content_blocks schema"
    status: completed
  - id: m12-chunker
    content: "Milestone 12: Implement metadata-aware chunking with overlap"
    status: completed
  - id: m13-embed
    content: "Milestone 13: Generate embeddings using OpenAI text-embedding-3-small"
    status: completed
  - id: m14-vectordb
    content: "Milestone 14: Store embeddings in Qdrant vector store"
    status: completed
  - id: m15-retrieval
    content: "Milestone 15: Implement vector retrieval with similarity search"
    status: completed
  - id: m16-rag
    content: "Milestone 16: Implement RAG answer generation with source citations"
    status: completed
  - id: m17-neo4j-schema
    content: "Milestone 17: Define Neo4j graph schema and create constraints/indexes"
    status: completed
  - id: m18-entity-extract
    content: "Milestone 18: Extract entities and relationships from chunks using LLM"
    status: completed
  - id: m19-neo4j-store
    content: "Milestone 19: Store graph nodes and relationships in Neo4j"
    status: completed
  - id: m20-cypher
    content: "Milestone 20: Implement safe Cypher query execution with validation"
    status: completed
  - id: m21-graphrag
    content: "Milestone 21: Implement GraphRAG combining vector retrieval and graph traversal"
    status: completed
  - id: m22-agents
    content: "Milestone 22: Build full agentic workflow with all 9 agents"
    status: completed
  - id: m23-langfuse
    content: "Milestone 23: Integrate Langfuse tracing and monitoring for full pipeline"
    status: completed
  - id: m24-streaming
    content: "Milestone 24: Add SSE streaming response to frontend"
    status: completed
  - id: m25-scaling
    content: "Milestone 25: Add caching, rate limiting, concurrency control, and retry logic"
    status: completed
  - id: m26-k8s
    content: "Milestone 26: Create all Kubernetes manifests (Namespace, ConfigMap, Secret, Deployments, StatefulSets, PVCs, Ingress, HPAs)"
    status: completed
  - id: m27-deploy
    content: "Milestone 27: Deploy to Kubernetes using kubectl and verify all pods/services"
    status: completed
  - id: m28-scale-test
    content: "Milestone 28: Test horizontal scaling with multiple concurrent users"
    status: completed
  - id: m29-eval
    content: "Milestone 29: Add evaluation dashboard and debug panel to frontend"
    status: completed
  - id: m30-readme
    content: "Milestone 30: Write final README and AI Engineer interview preparation guide"
    status: completed
isProject: false
---

# Milestone 1 — Architecture, Folder Structure, and System Design

## Goal

Understand the complete system before writing a single line of code. By the end of Milestone 1 you will be able to explain every component, every data flow, and every deployment artifact to an interviewer.

---

## 1. End-to-End Architecture

```mermaid
flowchart TD
    User["Browser / User"]
    Angular["Angular Frontend\n(Nginx container)"]
    Ingress["Kubernetes Ingress\n/ Load Balancer"]
    FastAPI["FastAPI Backend\n(Python container)"]
    Celery["Celery Worker\n(Python container)"]
    Redis["Redis\n(Queue + Cache)"]
    Postgres["PostgreSQL\n(Metadata DB)"]
    VectorDB["Qdrant / pgvector\n(Vector Store)"]
    Neo4j["Neo4j\n(Graph DB)"]
    ObjectStore["Object Storage\n(MinIO / S3)"]
    OpenAI["OpenAI API\n(Chat + Embeddings)"]
    Langfuse["Langfuse\n(Observability)"]

    User -->|HTTPS| Ingress
    Ingress --> Angular
    Angular -->|REST / SSE| FastAPI
    FastAPI -->|Enqueue job| Redis
    FastAPI -->|Read/Write metadata| Postgres
    FastAPI -->|Vector search| VectorDB
    FastAPI -->|Cypher queries| Neo4j
    FastAPI -->|OpenAI calls| OpenAI
    FastAPI -->|Traces + spans| Langfuse
    Redis -->|Job pickup| Celery
    Celery -->|Store file| ObjectStore
    Celery -->|Write metadata| Postgres
    Celery -->|Store embeddings| VectorDB
    Celery -->|Store graph nodes| Neo4j
    Celery -->|Embed call| OpenAI
```

---

## 2. Agentic Workflow (per chat request)

```mermaid
flowchart TD
    Q["User Query"]
    Router["Router Agent\nClassify intent"]
    Rewrite["Query Rewrite Agent\nClarify ambiguous query"]
    Retriever["Retriever Agent\nVector search → top chunks"]
    Cypher["Cypher Agent\nGenerate + execute Cypher"]
    CtxBuilder["Context Builder Agent\nMerge chunks + graph results"]
    Answer["Answer Agent\nOpenAI GPT call"]
    Critic["Critic Agent\nValidate grounding + citations"]
    Memory["Memory Agent\nUpdate session history"]
    Response["Streaming Response + Citations"]

    Q --> Router
    Router -->|"doc_question"| Rewrite
    Router -->|"graph_question"| Cypher
    Router -->|"general_chat"| Answer
    Rewrite --> Retriever
    Retriever --> CtxBuilder
    Cypher --> CtxBuilder
    CtxBuilder --> Answer
    Answer --> Critic
    Critic -->|valid| Memory
    Memory --> Response
    Critic -->|invalid| Answer
```

---

## 3. Document Processing Pipeline

```mermaid
flowchart TD
    Upload["File Upload\n(Angular → FastAPI)"]
    Queue["Redis Queue\n(Celery task)"]
    Parse["Parser\nPDF/DOCX/PPTX/CSV/Excel/\nImage OCR/Audio STT/Video"]
    Normalize["Normalizer\nCommon content_blocks schema"]
    Chunk["Chunker\n500-1000 tokens, overlap, metadata-aware"]
    Embed["Embedder\nOpenAI text-embedding-3-small"]
    Store["Store\nPostgres + VectorDB + Neo4j + ObjectStore"]

    Upload --> Queue
    Queue --> Parse
    Parse --> Normalize
    Normalize --> Chunk
    Chunk --> Embed
    Embed --> Store
```

---

## 4. Shared Agent State Schema

Every agent reads from and writes to a single Python dataclass / TypedDict:

```python
{
  "user_id": str,
  "session_id": str,
  "user_query": str,
  "intent": str | None,            # set by Router
  "rewritten_query": str | None,   # set by QueryRewrite
  "retrieved_chunks": list,        # set by Retriever
  "cypher_query": str | None,      # set by Cypher
  "cypher_results": list,          # set by Cypher
  "final_context": str | None,     # set by ContextBuilder
  "draft_answer": str | None,      # set by Answer
  "validated_answer": str | None,  # set by Critic
  "citations": list,
  "latency_ms": float | None,
  "token_usage": dict | None,
  "cost": float | None,
  "langfuse_trace_id": str | None
}
```

---

## 5. Neo4j Graph Schema

```
(:Document)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
(:Chunk)-[:BELONGS_TO_TOPIC]->(:Topic)
(:Decision)-[:SUPPORTED_BY]->(:Chunk)
(:Action)-[:LINKED_TO]->(:Decision)
```

Node labels: `Document`, `Chunk`, `Entity`, `Topic`, `Person`, `Organization`, `Metric`, `Decision`, `Action`

---

## 6. Folder Structure

```
enterprise-ai-assistant/
│
├── frontend/                        # Angular app
│   ├── src/
│   │   ├── app/
│   │   │   ├── chat/                # Chat page + sidebar
│   │   │   ├── documents/           # Upload + library screens
│   │   │   ├── debug-panel/         # Debug overlay component
│   │   │   └── shared/              # Services, models, interceptors
│   │   └── environments/
│   ├── Dockerfile
│   └── nginx.conf
│
├── backend/                         # FastAPI app
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py              # /chat endpoints
│   │   │   ├── documents.py         # /documents endpoints
│   │   │   └── health.py
│   │   ├── agents/
│   │   │   ├── router_agent.py
│   │   │   ├── rewrite_agent.py
│   │   │   ├── retriever_agent.py
│   │   │   ├── cypher_agent.py
│   │   │   ├── context_builder_agent.py
│   │   │   ├── answer_agent.py
│   │   │   ├── critic_agent.py
│   │   │   ├── memory_agent.py
│   │   │   └── tool_agent.py
│   │   ├── parsers/
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── pptx_parser.py
│   │   │   ├── excel_parser.py
│   │   │   ├── csv_parser.py
│   │   │   ├── image_parser.py      # OCR
│   │   │   ├── audio_parser.py      # Whisper STT
│   │   │   └── video_parser.py      # ffmpeg + audio
│   │   ├── pipeline/
│   │   │   ├── normalizer.py        # → common content_blocks
│   │   │   ├── chunker.py           # metadata-aware chunking
│   │   │   ├── embedder.py          # OpenAI embeddings
│   │   │   └── ingestion.py         # orchestrates full pipeline
│   │   ├── db/
│   │   │   ├── postgres.py          # SQLAlchemy models + session
│   │   │   ├── vector_store.py      # Qdrant / pgvector client
│   │   │   └── neo4j_client.py      # Neo4j driver + Cypher helpers
│   │   ├── services/
│   │   │   ├── openai_service.py    # chat, embed, summary calls
│   │   │   ├── langfuse_service.py  # trace/span helpers
│   │   │   ├── cache_service.py     # Redis caching
│   │   │   └── storage_service.py   # MinIO / S3 upload
│   │   ├── workers/
│   │   │   └── document_worker.py   # Celery tasks
│   │   ├── models/                  # Pydantic request/response schemas
│   │   │   ├── chat.py
│   │   │   ├── document.py
│   │   │   └── agent_state.py
│   │   ├── core/
│   │   │   ├── config.py            # env-based settings
│   │   │   ├── security.py          # CORS, auth placeholder
│   │   │   └── rate_limiter.py
│   │   └── main.py                  # FastAPI app entry point
│   ├── requirements.txt
│   └── Dockerfile
│
├── worker/
│   └── Dockerfile                   # Same base as backend, runs Celery
│
├── docker-compose.yml               # Full local dev stack
│
└── k8s/                             # Kubernetes manifests
    ├── namespace.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── frontend/
    │   ├── deployment.yaml
    │   └── service.yaml
    ├── backend/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── hpa.yaml
    ├── worker/
    │   ├── deployment.yaml
    │   └── hpa.yaml
    ├── redis/
    │   ├── deployment.yaml
    │   └── service.yaml
    ├── postgres/
    │   ├── statefulset.yaml
    │   ├── service.yaml
    │   └── pvc.yaml
    ├── neo4j/
    │   ├── statefulset.yaml
    │   ├── service.yaml
    │   └── pvc.yaml
    └── ingress.yaml
```

---

## 7. Tech Stack — Component Responsibility Map

- **Angular** — UI: chat window, document upload, library, history sidebar, source citation panel, debug panel
- **FastAPI** — REST API, SSE streaming, orchestration, rate limiting, auth
- **OpenAI API** — `gpt-4o` for chat/agents, `text-embedding-3-small` for embeddings, `whisper-1` for audio transcription
- **LangChain / LangGraph** — agent orchestration framework (introduced in Milestone 22)
- **Qdrant** (or pgvector) — vector store for chunk embeddings
- **PostgreSQL** — metadata: documents, chunks, sessions, users
- **Neo4j** — knowledge graph: entities, relationships, topics
- **Redis** — Celery task queue + embedding/retrieval cache
- **Celery** — async document processing worker
- **MinIO** — local S3-compatible object storage for original files
- **Langfuse** — LLM observability: traces, spans, token usage, cost, latency, user feedback
- **Docker** — container images for all services
- **Kubernetes** — production deployment: pods, services, HPA, StatefulSets, Ingress

---

## 8. Kubernetes Concepts (Milestone 1 Primer)

- **Image** — a frozen, portable snapshot of your app and all its dependencies
- **Container** — a running instance of an image, isolated process
- **Pod** — the smallest Kubernetes unit; wraps one or more containers
- **Deployment** — declares how many Pod replicas to run and handles rolling updates (used for stateless: frontend, backend, worker, Redis)
- **StatefulSet** — like Deployment but gives each Pod a stable identity and storage (used for PostgreSQL, Neo4j)
- **Service** — stable DNS name + load balancer in front of a set of Pods
- **Ingress** — HTTP router that maps public hostnames/paths to Services (replaces the Load Balancer at the edge)
- **ConfigMap** — stores non-secret config (URLs, flags) as key-value pairs mounted as env vars
- **Secret** — stores sensitive config (API keys, passwords) base64-encoded, referenced by Pods
- **PersistentVolumeClaim (PVC)** — requests durable disk storage that survives Pod restarts (databases need this)
- **HorizontalPodAutoscaler (HPA)** — automatically increases or decreases Pod replicas based on CPU/memory/custom metrics

---

## 9. Local vs. Production Deployment Strategy

**Local (Docker Compose)**
- Single `docker-compose.yml` starts: frontend, backend, worker, Redis, PostgreSQL, Neo4j, Qdrant, MinIO, Langfuse
- All on one machine, shared Docker network
- Command: `docker compose up --build`

**Production (Kubernetes)**
- Each service runs as isolated Pods in the `ai-assistant` Namespace
- Backend and Worker scale horizontally via HPA
- Databases use StatefulSets with PVCs for durable storage
- Ingress (nginx-ingress-controller) routes `api.*` → backend Service, `app.*` → frontend Service
- Secrets injected via Kubernetes Secret objects (never in source code)
- Command: `kubectl apply -f k8s/`

---

## 10. Milestone Roadmap (30 milestones total)

- **M1** — Architecture + folder structure (this milestone)
- **M2-M3** — Angular + FastAPI skeleton, simple chat UI
- **M4** — OpenAI chat integration
- **M5-M6** — Dockerfiles + Docker Compose
- **M7-M9** — Document upload, PostgreSQL metadata, Redis + Celery
- **M10-M12** — Multi-format parsers, normalizer, chunker
- **M13-M15** — Embeddings, vector store, vector retrieval
- **M16** — RAG answer generation with citations
- **M17-M21** — Neo4j schema, entity extraction, Cypher agent, GraphRAG
- **M22** — Full agentic workflow (all 9 agents)
- **M23** — Langfuse observability
- **M24** — Streaming responses
- **M25** — Caching, rate limiting, concurrency control, retry
- **M26-M28** — Kubernetes manifests, kubectl deploy, scaling tests
- **M29** — Evaluation dashboard + debug panel
- **M30** — README + interview preparation guide

---

## 11. Interview Explanation for Milestone 1

> "I designed the system using a microservices pattern where the frontend, backend API, and document processing worker are completely separate containers. Document ingestion is decoupled from the API using a Redis-backed Celery queue so that uploading a 50-page PDF never blocks the chat API. The knowledge is stored in three complementary stores: PostgreSQL for structured metadata, Qdrant for dense vector similarity search, and Neo4j for entity relationship traversal. Every LLM call is traced in Langfuse so I can observe token usage, latency, and cost in production. On Kubernetes, the stateless backend and worker scale horizontally via HPA while the databases use StatefulSets with PVCs to guarantee durable storage."

---

## 12. Prerequisites to Install Before Milestone 2

- Node.js 20+ and Angular CLI 17+
- Python 3.11+
- Docker Desktop (or Docker Engine + Compose plugin)
- A code editor (VS Code / Cursor)
- OpenAI API key (set in `.env`, never committed)
- Langfuse account (free tier) or self-hosted Langfuse
