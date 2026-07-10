# Enterprise AI Chat Assistant

A complete, production-grade AI application built milestone by milestone for learning and AI Engineer interview preparation.

**Stack:** Angular · FastAPI · OpenAI · LangChain/LangGraph · Qdrant · Neo4j · PostgreSQL · Redis · Celery · MinIO · Langfuse · Docker · Kubernetes

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [End-to-End Architecture](#2-end-to-end-architecture)
3. [Agentic Workflow](#3-agentic-workflow)
4. [Document Processing Pipeline](#4-document-processing-pipeline)
5. [Folder Structure Explained](#5-folder-structure-explained)
6. [Tech Stack — Every Component Explained](#6-tech-stack--every-component-explained)
7. [Neo4j Graph Schema](#7-neo4j-graph-schema)
8. [Shared Agent State](#8-shared-agent-state)
9. [Docker and Kubernetes Concepts](#9-docker-and-kubernetes-concepts)
10. [Local Setup with Docker Compose](#10-local-setup-with-docker-compose)
11. [Production Deployment with Kubernetes](#11-production-deployment-with-kubernetes)
12. [Milestone Roadmap](#12-milestone-roadmap)
13. [Interview Preparation](#13-interview-preparation)
14. [Prerequisites](#14-prerequisites)

---

## 1. Project Goal

Build a ChatGPT-like enterprise application where users can:

- Upload documents in any format (PDF, DOCX, PPTX, Excel, CSV, TXT, images, audio, video)
- Ask questions and get answers grounded in uploaded documents
- See source citations for every answer
- Query knowledge relationships using Neo4j graph traversal
- Use an intelligent agentic workflow that routes, retrieves, reasons, validates, and responds
- Observe every LLM call in Langfuse (token usage, cost, latency, traces)
- Deploy on Kubernetes with automatic horizontal scaling

---

## 2. End-to-End Architecture

```
Browser / User
      |
      | HTTPS
      v
Kubernetes Ingress (nginx)
      |
      |--- /          --> Angular Frontend (Nginx container)
      |--- /api/*     --> FastAPI Backend (Python container)
                              |
                    +---------+-----------+
                    |         |           |
                    v         v           v
                PostgreSQL  Qdrant     Neo4j
                (metadata) (vectors)  (graph)
                    |
                    v
                  Redis  <--------> Celery Worker
                (queue)             (document processing)
                                         |
                                    +---------+
                                    |         |
                                   MinIO   OpenAI
                                 (storage) (embed)
                    |
                    v
                 OpenAI API     Langfuse
                (chat/embed)   (observability)
```

### Data Flow: Chat Request

1. User types a question in Angular
2. Angular sends POST `/api/chat` to FastAPI
3. FastAPI creates an **Agent State** object and starts the agentic workflow
4. **Router Agent** classifies the intent (doc question, graph question, general chat)
5. **Query Rewrite Agent** clarifies ambiguous questions
6. **Retriever Agent** runs vector search in Qdrant → returns top-k chunks
7. **Cypher Agent** (if graph intent) generates and runs a safe Cypher query on Neo4j
8. **Context Builder Agent** merges chunks + graph results into a single context block
9. **Answer Agent** calls OpenAI GPT-4o with the context → generates grounded answer
10. **Critic Agent** validates that the answer is grounded (no hallucinations)
11. **Memory Agent** stores conversation turn in session history
12. FastAPI streams the response back using SSE (Server-Sent Events)
13. Langfuse records every step: intent, chunks, Cypher query, OpenAI tokens, cost, latency

### Data Flow: Document Upload

1. User selects a file in Angular
2. Angular sends POST `/api/documents/upload`
3. FastAPI saves the file to MinIO and enqueues a Celery task in Redis
4. FastAPI returns `202 Accepted` immediately (non-blocking)
5. Celery Worker picks up the task and runs the ingestion pipeline:
   - **Parser** → extracts raw text/content from the file
   - **Normalizer** → converts to common `content_blocks` schema
   - **Chunker** → splits into 500–1000 token chunks with metadata
   - **Embedder** → calls OpenAI `text-embedding-3-small` for each chunk
   - **Storer** → writes metadata to PostgreSQL, vectors to Qdrant, graph nodes to Neo4j

---

## 3. Agentic Workflow

```
User Query
    |
    v
Router Agent ──────────────────────────────────┐
    |                                           |
    | doc_question / summary                    | general_chat
    v                                           v
Query Rewrite Agent                         Answer Agent
    |                                           |
    v                                           v
Retriever Agent                           Critic Agent
    |                                           |
    |                                           v
    +──────────────> Context Builder      Memory Agent
                     Agent <──────────────────--|
                         |                      |
                         | graph_question        v
                         +──> Cypher Agent   Response
                              (Neo4j)
```

### When Each Agent Is Called

| Intent | Agents Invoked |
|--------|---------------|
| `general_chat` | Router → Answer → Critic → Memory |
| `doc_question` | Router → Rewrite → Retriever → ContextBuilder → Answer → Critic → Memory |
| `graph_question` | Router → Cypher → ContextBuilder → Answer → Critic → Memory |
| `summary` | Router → Rewrite → Retriever → ContextBuilder → Answer → Critic → Memory |
| `task` | Router → Tool Agent → Answer → Memory |

---

## 4. Document Processing Pipeline

```
File Upload (Angular)
        |
        v
FastAPI: validate, save to MinIO, enqueue Celery task
        |
        v
Celery Worker picks up task
        |
        v
Parser (format-specific)
    PDF:   PyMuPDF / pdfplumber → pages, tables, OCR for scanned
    DOCX:  python-docx → paragraphs, headings, tables
    PPTX:  python-pptx → slides, titles, notes
    Excel: openpyxl → sheets, rows, columns
    CSV:   pandas → rows, columns
    Image: Tesseract OCR → text
    Audio: OpenAI Whisper → transcript
    Video: ffmpeg → audio → Whisper → transcript + frame OCR
        |
        v
Normalizer
    → common content_blocks schema (document_id, blocks with text, metadata)
        |
        v
Chunker
    → 500–1000 tokens per chunk
    → 10–20% overlap between chunks
    → metadata preserved: page, slide, sheet, timestamp, section title
        |
        v
Embedder
    → OpenAI text-embedding-3-small
    → 1536-dimension vectors
        |
        v
Storage (all in parallel)
    → PostgreSQL: document + chunk metadata
    → Qdrant:     chunk vectors
    → Neo4j:      Document, Chunk, Entity, Topic nodes + relationships
    → MinIO:      original file already stored
```

### Normalized Document Schema

```json
{
  "document_id": "doc_abc123",
  "file_name": "annual_report.pdf",
  "file_type": "pdf",
  "content_blocks": [
    {
      "block_id": "block_001",
      "type": "text",
      "text": "Revenue grew 23% year-over-year driven by cloud segment.",
      "page_number": 4,
      "slide_number": null,
      "sheet_name": null,
      "timestamp": null,
      "metadata": { "section_heading": "Financial Highlights" }
    }
  ]
}
```

### Chunk Schema

```json
{
  "chunk_id": "chunk_xyz789",
  "document_id": "doc_abc123",
  "file_name": "annual_report.pdf",
  "source_type": "pdf",
  "page_number": 4,
  "slide_number": null,
  "sheet_name": null,
  "timestamp_start": null,
  "timestamp_end": null,
  "section_title": "Financial Highlights",
  "chunk_text": "Revenue grew 23% year-over-year driven by cloud segment.",
  "embedding_id": "qdrant_point_id_here",
  "metadata": {}
}
```

---

## 5. Folder Structure Explained

```
enterprise-ai-assistant/
│
├── .env.example               ← Copy to .env, fill in your keys
├── .gitignore                 ← Prevents secrets and build artifacts from being committed
├── README.md                  ← This file
├── docker-compose.yml         ← Starts all services locally with one command
│
├── frontend/                  ← Angular application
│   ├── src/
│   │   ├── app/
│   │   │   ├── chat/          ← Chat page: message list, input box, history sidebar
│   │   │   ├── documents/     ← Upload screen + document library
│   │   │   ├── debug-panel/   ← Shows intent, chunks, Cypher, tokens, cost, latency
│   │   │   └── shared/
│   │   │       ├── services/  ← API service, auth service, SSE service
│   │   │       ├── models/    ← TypeScript interfaces: Message, Document, AgentTrace
│   │   │       └── interceptors/ ← HTTP interceptor for auth headers
│   │   └── environments/      ← environment.ts (dev), environment.prod.ts
│   ├── Dockerfile             ← Multi-stage: ng build → Nginx serve
│   └── nginx.conf             ← Nginx config: SPA routing + /api proxy
│
├── backend/                   ← FastAPI application
│   ├── app/
│   │   ├── main.py            ← FastAPI app creation, router registration, startup events
│   │   │
│   │   ├── api/               ← HTTP route handlers
│   │   │   ├── chat.py        ← POST /chat, GET /chat/history
│   │   │   ├── documents.py   ← POST /documents/upload, GET /documents, DELETE /documents/{id}
│   │   │   └── health.py      ← GET /health (liveness + readiness probes for K8s)
│   │   │
│   │   ├── agents/            ← One file per agent, each is a pure function
│   │   │   ├── router_agent.py        ← Classifies intent using GPT
│   │   │   ├── rewrite_agent.py       ← Rewrites query for better retrieval
│   │   │   ├── retriever_agent.py     ← Calls Qdrant vector search
│   │   │   ├── cypher_agent.py        ← Generates + validates + runs Cypher queries
│   │   │   ├── context_builder_agent.py ← Merges chunks + graph results
│   │   │   ├── answer_agent.py        ← Calls OpenAI GPT with context
│   │   │   ├── critic_agent.py        ← Validates grounding, detects hallucination
│   │   │   ├── memory_agent.py        ← Reads/writes session conversation history
│   │   │   └── tool_agent.py          ← Calls external tools (search, calculator, etc.)
│   │   │
│   │   ├── parsers/           ← One parser per file type
│   │   │   ├── pdf_parser.py          ← PyMuPDF + OCR fallback
│   │   │   ├── docx_parser.py         ← python-docx
│   │   │   ├── pptx_parser.py         ← python-pptx
│   │   │   ├── excel_parser.py        ← openpyxl
│   │   │   ├── csv_parser.py          ← pandas
│   │   │   ├── image_parser.py        ← Tesseract OCR
│   │   │   ├── audio_parser.py        ← OpenAI Whisper API
│   │   │   └── video_parser.py        ← ffmpeg → audio → Whisper + frame OCR
│   │   │
│   │   ├── pipeline/          ← Document ingestion pipeline stages
│   │   │   ├── normalizer.py          ← Converts parser output → content_blocks schema
│   │   │   ├── chunker.py             ← Splits content_blocks into chunks with overlap
│   │   │   ├── embedder.py            ← Calls OpenAI embeddings API per chunk
│   │   │   └── ingestion.py           ← Orchestrates: parse→normalize→chunk→embed→store
│   │   │
│   │   ├── db/                ← Database clients
│   │   │   ├── postgres.py            ← SQLAlchemy engine, session factory, ORM models
│   │   │   ├── vector_store.py        ← Qdrant client: upsert, search, delete
│   │   │   └── neo4j_client.py        ← Neo4j driver: run_query, safe_cypher, close
│   │   │
│   │   ├── services/          ← Business logic / external integrations
│   │   │   ├── openai_service.py      ← Wrappers for chat, embed, whisper calls
│   │   │   ├── langfuse_service.py    ← Trace, span, generation helpers
│   │   │   ├── cache_service.py       ← Redis get/set/delete for embedding + retrieval cache
│   │   │   └── storage_service.py     ← MinIO upload, download, delete
│   │   │
│   │   ├── workers/           ← Celery async task definitions
│   │   │   └── document_worker.py     ← process_document() Celery task
│   │   │
│   │   ├── models/            ← Pydantic schemas for request/response validation
│   │   │   ├── chat.py                ← ChatRequest, ChatResponse, MessageRole
│   │   │   ├── document.py            ← UploadResponse, DocumentMetadata, ChunkSchema
│   │   │   └── agent_state.py         ← AgentState TypedDict (shared across all agents)
│   │   │
│   │   └── core/              ← Cross-cutting concerns
│   │       ├── config.py              ← Pydantic BaseSettings reads from .env
│   │       ├── security.py            ← CORS setup, auth placeholder, file validation
│   │       └── rate_limiter.py        ← Per-user + global concurrency limits
│   │
│   ├── requirements.txt       ← All Python dependencies with pinned versions
│   └── Dockerfile             ← Python image → install deps → run uvicorn
│
├── worker/
│   └── Dockerfile             ← Same base image as backend, runs: celery -A app.workers.document_worker worker
│
├── k8s/                       ← Kubernetes production manifests
│   ├── namespace.yaml         ← Isolates all resources in 'ai-assistant' namespace
│   ├── configmap.yaml         ← Non-secret config: URLs, feature flags
│   ├── secret.yaml            ← API keys, DB passwords (base64 encoded)
│   ├── ingress.yaml           ← Routes public traffic to frontend/backend Services
│   ├── frontend/
│   │   ├── deployment.yaml    ← 2 replicas of Angular/Nginx container
│   │   └── service.yaml       ← ClusterIP service for frontend pods
│   ├── backend/
│   │   ├── deployment.yaml    ← 3 replicas of FastAPI container
│   │   ├── service.yaml       ← ClusterIP service for backend pods
│   │   └── hpa.yaml           ← Scale 3→20 pods based on CPU usage
│   ├── worker/
│   │   ├── deployment.yaml    ← 2 replicas of Celery worker container
│   │   └── hpa.yaml           ← Scale 2→10 pods based on queue depth
│   ├── redis/
│   │   ├── deployment.yaml    ← Single Redis pod (stateless config, data is ephemeral)
│   │   └── service.yaml
│   ├── postgres/
│   │   ├── statefulset.yaml   ← Single PostgreSQL pod with stable identity
│   │   ├── service.yaml       ← Headless service for stable DNS
│   │   └── pvc.yaml           ← 20Gi persistent disk for database files
│   ├── neo4j/
│   │   ├── statefulset.yaml   ← Single Neo4j pod
│   │   ├── service.yaml
│   │   └── pvc.yaml           ← 20Gi persistent disk for graph data
│   └── qdrant/
│       ├── statefulset.yaml   ← Single Qdrant pod
│       ├── service.yaml
│       └── pvc.yaml           ← 10Gi persistent disk for vector index
│
├── docs/
│   ├── architecture/
│   │   ├── system-design.md   ← Deep-dive architecture decisions
│   │   └── data-flows.md      ← Step-by-step data flow for every operation
│   └── interviews/
│       └── ai-engineer-prep.md ← Interview questions + answers for every milestone
│
└── scripts/
    ├── setup-local.sh         ← One-shot script to set up local dev environment
    ├── seed-neo4j.py          ← Create Neo4j constraints and indexes
    └── test-pipeline.py       ← End-to-end test of document processing pipeline
```

---

## 6. Tech Stack — Every Component Explained

### Angular (Frontend)

**What it is:** A TypeScript web framework by Google for building single-page applications.

**Why we use it:** Provides a component-based architecture, built-in HTTP client, routing, and reactive forms — perfect for a complex multi-screen chat application.

**What it does in this project:**
- Chat screen: message list, input, history sidebar, source citation panel
- Documents screen: drag-and-drop upload, library with status indicators
- Debug panel: shows intent, retrieved chunks, Cypher query, tokens, cost, latency
- SSE client: connects to the FastAPI streaming endpoint to display responses word-by-word

---

### FastAPI (Backend)

**What it is:** A modern Python web framework for building APIs, based on Python type hints.

**Why we use it:** Async by default, automatic API documentation, Pydantic validation, native streaming support via SSE.

**What it does in this project:**
- Handles all REST API calls from Angular
- Orchestrates the agentic workflow for every chat request
- Validates uploaded files and enqueues them to Redis/Celery
- Streams chat responses back to the frontend using SSE
- Exposes `/health` endpoint for Kubernetes liveness and readiness probes

---

### OpenAI API

**What it is:** A cloud API providing access to GPT models, embedding models, and Whisper.

**Models used:**
| Model | Purpose |
|-------|---------|
| `gpt-4o` | Chat, agent reasoning, entity extraction, answer generation, critic validation |
| `text-embedding-3-small` | Converting text chunks into 1536-dim vectors |
| `whisper-1` | Transcribing audio/video files to text |

**Why we use it:** Best-in-class quality for reasoning and embeddings; widely used in production AI systems.

---

### LangChain / LangGraph (Agent Orchestration)

**What it is:** Python libraries for building LLM applications and agent workflows.

**Why we use it (introduced in Milestone 22):**
- LangChain provides prompt templates, output parsers, and retrieval chains
- LangGraph provides a stateful graph for multi-agent orchestration with loops and conditionals

**What it does in this project:**
- Defines the agent graph: Router → Rewrite → Retriever → ContextBuilder → Answer → Critic → Memory
- Manages the shared `AgentState` across all agents
- Handles retries, loops (Critic can send back to Answer), and conditional edges

---

### Qdrant (Vector Database)

**What it is:** An open-source vector database optimized for similarity search.

**Why we use it:** Fast approximate nearest-neighbor search on high-dimensional vectors; supports metadata filtering; easy Docker deployment.

**What it does in this project:**
- Stores one vector per document chunk (1536 dimensions from OpenAI)
- Receives a query vector from FastAPI and returns the top-k most similar chunks
- Supports metadata filters: filter by `document_id`, `source_type`, `page_number`, etc.

**Alternative:** pgvector (PostgreSQL extension) — simpler setup but slower for large collections.

---

### PostgreSQL (Relational Metadata Database)

**What it is:** A battle-tested open-source relational database.

**Why we use it:** Reliable ACID transactions; perfect for storing structured metadata about documents, chunks, users, and sessions.

**Tables in this project:**
| Table | Purpose |
|-------|---------|
| `documents` | File name, type, upload time, status, MinIO path |
| `chunks` | Chunk text, page, slide, sheet, embedding ID, document FK |
| `sessions` | User sessions, conversation history |
| `users` | User accounts (added in later milestones) |

---

### Neo4j (Graph Database)

**What it is:** A native graph database that stores nodes and relationships; queries using Cypher language.

**Why we use it:** When a user asks "Which decisions were supported by the revenue analysis?" — a graph traversal query is far more natural and efficient than a relational join or keyword search.

**What it does in this project:**
- Stores entities (Person, Organization, Metric, Decision, Action) extracted from document chunks
- Stores relationships between entities (`RELATED_TO`, `MENTIONS`, `SUPPORTED_BY`)
- Answers graph-style questions via the Cypher Agent

**Example Cypher query:**
```cypher
MATCH (d:Decision)-[:SUPPORTED_BY]->(c:Chunk)-[:MENTIONS]->(m:Metric)
WHERE m.name CONTAINS 'revenue'
RETURN d.text, c.chunk_id, m.name
LIMIT 10
```

---

### Redis

**What it is:** An in-memory key-value store used as a message broker and cache.

**Two roles in this project:**

1. **Celery Broker:** FastAPI pushes document processing jobs onto a Redis queue; Celery workers pull jobs off the queue. This decouples document upload from the API response time.

2. **Cache:** Stores embedding vectors and retrieval results so repeated queries don't re-call OpenAI or Qdrant.

---

### Celery (Async Worker)

**What it is:** A distributed task queue library for Python.

**Why we use it:** Document processing (parsing, chunking, embedding) can take 30–300 seconds. We cannot block the HTTP request. Celery runs this work in the background.

**What it does:**
- Picks up `process_document` tasks from the Redis queue
- Runs the full parse → normalize → chunk → embed → store pipeline
- Updates document status in PostgreSQL (pending → processing → complete / failed)

---

### MinIO (Object Storage)

**What it is:** An open-source, S3-compatible object storage server.

**Why we use it:** Original uploaded files need to be stored durably. We do not store binary files in PostgreSQL. MinIO works locally and is API-compatible with AWS S3, so the same code runs in production.

---

### Langfuse (LLM Observability)

**What it is:** An open-source platform for tracing, monitoring, and evaluating LLM applications.

**Why we use it:** Every LLM call has token costs, latency, and correctness risks. Langfuse makes all of this observable.

**What it tracks in this project:**
- Every chat request creates a top-level **trace**
- Each agent step creates a **span** inside the trace
- Each OpenAI API call creates a **generation** with token counts, cost, latency
- Retrieved chunks and Cypher results are logged as **observations**
- User feedback (thumbs up/down) is logged as **scores**

---

### Docker

**What it is:** A platform for packaging applications into portable containers.

**Images we build:**
| Image | Base | What runs |
|-------|------|-----------|
| `ai-frontend` | `node:20` → `nginx:alpine` | Angular SPA served by Nginx |
| `ai-backend` | `python:3.11-slim` | FastAPI via uvicorn |
| `ai-worker` | same as backend | Celery worker |

**Why separate the backend and worker images?** They run different processes but share the same Python code. Using a shared base image saves build time.

---

### Kubernetes

**What it is:** An open-source container orchestration system.

**Why we use it:** Kubernetes automatically restarts crashed containers, distributes load across multiple pods, scales services up and down, and manages configuration and secrets.

---

## 7. Neo4j Graph Schema

```
Nodes:
  (:Document  {document_id, file_name, file_type, upload_time})
  (:Chunk     {chunk_id, document_id, text, page_number, section_title})
  (:Entity    {entity_id, name, type})
  (:Topic     {topic_id, name})
  (:Person    {name, title, organization})
  (:Organization {name, industry})
  (:Metric    {name, value, unit, period})
  (:Decision  {decision_id, text, date})
  (:Action    {action_id, text, owner, due_date})

Relationships:
  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Chunk)-[:MENTIONS]->(:Person)
  (:Chunk)-[:MENTIONS]->(:Organization)
  (:Chunk)-[:MENTIONS]->(:Metric)
  (:Entity)-[:RELATED_TO]->(:Entity)
  (:Chunk)-[:BELONGS_TO_TOPIC]->(:Topic)
  (:Decision)-[:SUPPORTED_BY]->(:Chunk)
  (:Action)-[:LINKED_TO]->(:Decision)
```

### Sample Cypher Queries

```cypher
-- Find all chunks that mention a specific organization
MATCH (c:Chunk)-[:MENTIONS]->(o:Organization {name: "Acme Corp"})
RETURN c.text, c.page_number

-- Find decisions and their supporting evidence
MATCH (d:Decision)-[:SUPPORTED_BY]->(c:Chunk)
RETURN d.text, c.text, c.page_number

-- Find related entities starting from a metric
MATCH (m:Metric {name: "Revenue"})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN m.name, e.name, e.type

-- Traverse from document to all entities
MATCH (doc:Document {document_id: "doc_123"})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN doc.file_name, e.name, e.type
```

---

## 8. Shared Agent State

Every agent in the agentic workflow reads from and writes to a single shared state object. This is defined as a Python TypedDict:

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    user_id: str
    session_id: str
    user_query: str

    # Set by Router Agent
    intent: Optional[str]           # "doc_question" | "graph_question" | "general_chat" | "summary" | "task"

    # Set by Query Rewrite Agent
    rewritten_query: Optional[str]

    # Set by Retriever Agent
    retrieved_chunks: list          # List of ChunkSchema objects with similarity scores

    # Set by Cypher Agent
    cypher_query: Optional[str]
    cypher_results: list

    # Set by Context Builder Agent
    final_context: Optional[str]

    # Set by Answer Agent
    draft_answer: Optional[str]

    # Set by Critic Agent
    validated_answer: Optional[str]
    is_grounded: Optional[bool]

    # Set by Memory Agent
    citations: list                 # List of source references
    conversation_history: list

    # Observability (filled throughout the workflow)
    latency_ms: Optional[float]
    token_usage: Optional[dict]
    cost: Optional[float]
    langfuse_trace_id: Optional[str]
    errors: list
```

---

## 9. Docker and Kubernetes Concepts

### Docker Concepts

**What is a Docker Image?**
A Docker image is a read-only snapshot of your application code, runtime, libraries, and configuration. Think of it as a ZIP file of your entire application environment.

```
Dockerfile → docker build → Image → docker run → Container
```

**What is a Container?**
A container is a running instance of a Docker image. It is an isolated process that shares the host OS kernel but has its own filesystem, network, and process space. You can run multiple containers from the same image.

**Why containers?**
- "Works on my machine" problem is eliminated — the container IS the environment
- Fast startup (seconds, not minutes)
- Predictable, reproducible deployments

### Kubernetes Concepts

**What is a Pod?**
The smallest deployable unit in Kubernetes. A Pod wraps one or more containers that share a network and storage. In this project, each Pod typically contains one container (e.g., one FastAPI container).

**What is a Deployment?**
A Deployment tells Kubernetes: "I want 3 replicas of this Pod always running. If one crashes, restart it. When I push a new image, replace them one at a time (rolling update)."
- Used for: frontend, backend, worker, Redis (stateless services)

**What is a StatefulSet?**
Like a Deployment, but each Pod gets a stable, predictable identity (e.g., `postgres-0`) and its own persistent volume. This is essential for databases because:
- The same Pod always reconnects to the same disk
- Pods start and stop in order
- Used for: PostgreSQL, Neo4j, Qdrant

**What is a Service?**
A Service gives a stable network address to a group of Pods. Pods come and go (they get new IP addresses), but the Service address never changes. Other services call the Service name, not individual Pod IPs.

```
backend-service:8000 → load balances across → [backend-pod-1, backend-pod-2, backend-pod-3]
```

**What is Ingress?**
Ingress is an HTTP router that sits at the edge of the cluster and routes incoming requests to the correct Service based on hostname or URL path. It replaces a cloud load balancer for routing.

```
app.company.com       → frontend-service:80
app.company.com/api/* → backend-service:8000
```

**What is a ConfigMap?**
A ConfigMap stores non-sensitive configuration as key-value pairs. Pods can read ConfigMap values as environment variables or mounted files. Example: database host, feature flag values, log levels.

**What is a Secret?**
A Secret is like a ConfigMap but for sensitive values (API keys, passwords). Values are base64-encoded and stored separately from ConfigMaps. In production, use a secrets manager (Vault, AWS Secrets Manager) instead of Kubernetes Secrets.

**What is a PersistentVolumeClaim (PVC)?**
A PVC is a request for storage. You say "I need 20Gi of disk space" and Kubernetes provisions it from the available storage (cloud disk, NFS, local disk). The database Pod mounts this volume — data survives Pod restarts.

**What is a HorizontalPodAutoscaler (HPA)?**
HPA watches CPU/memory usage and automatically adjusts the number of Pod replicas.

```
backend HPA: min=3, max=20, target CPU=70%
→ If CPU > 70%: scale up (add more backend pods)
→ If CPU < 70%: scale down (remove idle pods)
```

---

## 10. Local Setup with Docker Compose

### Prerequisites

```bash
# Install Node.js 20+
node --version   # should be v20.x or higher

# Install Angular CLI
npm install -g @angular/cli@17

# Install Python 3.11+
python3 --version   # should be 3.11.x or higher

# Install Docker Engine + Compose plugin
docker --version         # 24.x or higher
docker compose version   # 2.x or higher
```

### Step-by-Step Local Setup

```bash
# 1. Clone the project
git clone <repo-url>
cd enterprise-ai-assistant

# 2. Create your environment file
cp .env.example .env
# Edit .env and add your OpenAI API key and other credentials

# 3. Start all services with Docker Compose
docker compose up --build

# This starts:
#   frontend  → http://localhost:4200
#   backend   → http://localhost:8000
#   worker    → (background, no UI)
#   redis     → localhost:6379
#   postgres  → localhost:5432
#   neo4j     → http://localhost:7474 (browser UI)
#   qdrant    → http://localhost:6333 (dashboard UI)
#   minio     → http://localhost:9001 (MinIO console)
#   langfuse  → http://localhost:3000

# 4. Verify services
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "1.0.0"}

# 5. Stop all services
docker compose down

# Stop and remove all data volumes (fresh start)
docker compose down -v
```

### Useful Docker Commands

```bash
# View logs for a specific service
docker compose logs -f backend

# Restart one service without rebuilding
docker compose restart worker

# Open a shell inside a running container
docker compose exec backend bash

# Check which containers are running
docker compose ps
```

---

## 11. Production Deployment with Kubernetes

### Prerequisites

```bash
# kubectl (Kubernetes CLI)
kubectl version --client

# minikube (for local Kubernetes cluster testing)
minikube version

# Or use a cloud cluster (GKE, EKS, AKS)
```

### Deploy to Kubernetes

```bash
# 1. Build and push Docker images
docker build -t your-registry/ai-frontend:v1 ./frontend
docker build -t your-registry/ai-backend:v1 ./backend
docker build -t your-registry/ai-worker:v1 ./worker
docker push your-registry/ai-frontend:v1
docker push your-registry/ai-backend:v1
docker push your-registry/ai-worker:v1

# 2. Create the namespace first
kubectl apply -f k8s/namespace.yaml

# 3. Apply ConfigMap and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 4. Deploy databases (stateful, must come first)
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/neo4j/
kubectl apply -f k8s/qdrant/
kubectl apply -f k8s/redis/

# 5. Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ai-assistant --timeout=120s
kubectl wait --for=condition=ready pod -l app=neo4j -n ai-assistant --timeout=120s

# 6. Deploy application services
kubectl apply -f k8s/backend/
kubectl apply -f k8s/worker/
kubectl apply -f k8s/frontend/

# 7. Deploy ingress
kubectl apply -f k8s/ingress.yaml

# 8. Apply HPAs
kubectl apply -f k8s/backend/hpa.yaml
kubectl apply -f k8s/worker/hpa.yaml
```

### Verify Deployment

```bash
# Check all pods are Running
kubectl get pods -n ai-assistant

# Check all services
kubectl get services -n ai-assistant

# Check ingress
kubectl get ingress -n ai-assistant

# View logs for a pod
kubectl logs -f deployment/backend -n ai-assistant

# Describe a pod (useful for debugging CrashLoopBackOff)
kubectl describe pod <pod-name> -n ai-assistant

# Check HPA status
kubectl get hpa -n ai-assistant

# Restart a deployment (rolling restart)
kubectl rollout restart deployment/backend -n ai-assistant

# Update image version
kubectl set image deployment/backend backend=your-registry/ai-backend:v2 -n ai-assistant

# Watch rollout progress
kubectl rollout status deployment/backend -n ai-assistant
```

### Debugging Failed Pods

```bash
# If a pod shows CrashLoopBackOff:
kubectl describe pod <pod-name> -n ai-assistant   # look at Events section
kubectl logs <pod-name> -n ai-assistant           # see crash output
kubectl logs <pod-name> --previous -n ai-assistant  # see logs from crashed container

# Common causes:
# 1. Missing environment variable → check ConfigMap and Secret
# 2. Database not ready → check postgres/neo4j pod status
# 3. Wrong image name → check deployment.yaml image field
# 4. OOMKilled → increase memory limits in deployment.yaml
```

---

## 12. Milestone Roadmap

| Milestone | Focus | What You Build |
|-----------|-------|----------------|
| M1 | Architecture | Folder structure, README, system design |
| M2 | Skeleton | Angular app + FastAPI app stubs |
| M3 | Chat UI | Chat screen, message list, backend echo API |
| M4 | OpenAI | Real GPT-4o chat integration |
| M5 | Dockerfiles | Dockerfile for frontend, backend, worker |
| M6 | Docker Compose | Full local dev stack with one command |
| M7 | Upload | Document upload UI + upload API |
| M8 | PostgreSQL | SQLAlchemy models, document + chunk tables |
| M9 | Celery | Redis queue, Celery worker, task status tracking |
| M10 | Parsers | PDF, DOCX, PPTX, CSV, Excel, Image, Audio, Video parsers |
| M11 | Normalizer | Common content_blocks schema |
| M12 | Chunker | Token-aware chunking with metadata and overlap |
| M13 | Embedder | OpenAI embeddings per chunk |
| M14 | Qdrant | Store and retrieve vectors |
| M15 | Retrieval | Top-k vector search with metadata filtering |
| M16 | RAG | Full RAG: retrieve → generate → cite |
| M17 | Neo4j Schema | Constraints, indexes, node/relationship design |
| M18 | Entity Extraction | LLM extracts entities + relationships from chunks |
| M19 | Neo4j Storage | Store graph data from extracted entities |
| M20 | Cypher Agent | Safe Cypher generation and execution |
| M21 | GraphRAG | Combine vector + graph retrieval |
| M22 | All Agents | LangGraph workflow with all 9 agents |
| M23 | Langfuse | Full observability: traces, spans, scores |
| M24 | Streaming | SSE streaming response to Angular |
| M25 | Scale | Caching, rate limiting, concurrency control |
| M26 | K8s Manifests | All YAML files for production deployment |
| M27 | K8s Deploy | kubectl deploy, verify, debug |
| M28 | Scale Test | Load test with multiple users |
| M29 | Eval + Debug | Evaluation metrics, debug panel |
| M30 | Final | README, interview prep guide |

---

## 13. Interview Preparation

### System Design Interview Answer

> "I designed this as a three-tier microservices application. The Angular frontend communicates with a FastAPI backend through an Nginx Ingress. The backend orchestrates a multi-agent LLM workflow using LangGraph, where a Router Agent classifies the query intent and hands off to specialized agents for retrieval, graph querying, answer generation, and validation.
>
> For knowledge storage, I use three complementary databases: PostgreSQL for structured document metadata, Qdrant for dense vector similarity search on chunk embeddings, and Neo4j for entity relationship traversal. This combination lets me answer questions that require semantic similarity, keyword matching, AND relationship reasoning.
>
> Document ingestion is completely decoupled from the API. When a user uploads a file, FastAPI immediately returns 202 Accepted and enqueues the work to a Redis-backed Celery queue. A separate worker pod handles the heavy lifting: parsing, chunking, embedding, and storing. This keeps API response times fast regardless of document size.
>
> On Kubernetes, stateless services (backend, worker, frontend) use Deployments with Horizontal Pod Autoscalers. Databases use StatefulSets with PersistentVolumeClaims so data survives pod restarts. Every LLM call is traced in Langfuse, giving me full visibility into token usage, cost, latency, and hallucination risk."

### Key Interview Questions by Component

**RAG:**
- "How do you choose chunk size?" → 500–1000 tokens, depends on model context window and retrieval precision requirements
- "How do you prevent hallucination?" → Critic Agent validates that every claim in the answer is supported by a retrieved chunk
- "What if no relevant chunks are found?" → Return "I could not find this in the uploaded documents" instead of hallucinating

**Vector Search:**
- "What is cosine similarity?" → Measures the angle between two embedding vectors. Closer angle = more semantically similar text
- "Why not use keyword search (BM25)?" → Keyword search fails on synonyms and paraphrases. Vector search understands meaning, not just words
- "What is ANN (Approximate Nearest Neighbor)?" → Exact search over millions of vectors is too slow. ANN algorithms (HNSW in Qdrant) trade tiny accuracy loss for massive speed gains

**Neo4j / GraphRAG:**
- "When would you use GraphRAG over pure RAG?" → When the answer requires traversing relationships between entities (e.g., "What decisions were influenced by the revenue decline?")
- "How do you prevent unsafe Cypher queries?" → Allowlist of permitted Cypher clauses, query validation before execution, read-only Neo4j user

**Agentic Workflow:**
- "What is the difference between a chain and an agent?" → A chain has fixed steps. An agent decides which steps to take based on the input
- "How do you handle agent loops?" → Set a max iteration count. The Critic Agent can return the answer to the Answer Agent, but only N times before we break the loop

**Kubernetes:**
- "Why StatefulSet for databases?" → Databases need stable Pod identity, ordered startup, and persistent storage — things Deployments don't provide
- "How does HPA know when to scale?" → It queries the Kubernetes Metrics Server for CPU/memory. Custom metrics (queue depth) require the Custom Metrics API

---

## 14. Prerequisites

Install these before starting Milestone 2:

```bash
# Node.js 20+ (for Angular)
https://nodejs.org/

# Angular CLI 17+
npm install -g @angular/cli@17

# Python 3.11+
https://www.python.org/downloads/

# Docker Desktop or Docker Engine
https://docs.docker.com/get-docker/

# kubectl
https://kubernetes.io/docs/tasks/tools/

# minikube (local Kubernetes)
https://minikube.sigs.k8s.io/docs/start/
```

**Accounts needed:**
- OpenAI API key: https://platform.openai.com/api-keys
- Langfuse (free tier): https://cloud.langfuse.com

---

*Built milestone by milestone for AI Engineer learning and interview preparation.*
