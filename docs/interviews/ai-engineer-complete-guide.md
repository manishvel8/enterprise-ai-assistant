# AI Engineer Interview Preparation Guide
## Enterprise AI Chat Assistant — Complete Technical Reference

---

## How to Use This Guide

For each system you built, you should be able to:
1. **Explain** it in one sentence to a non-technical interviewer
2. **Detail** the implementation to a senior engineer
3. **Justify** the technology choices (why X instead of Y)
4. **Discuss** failure modes and how you handle them

---

## Part 1: RAG (Retrieval Augmented Generation)

### One-sentence explanation
> "RAG retrieves the most relevant passages from a document store and injects them into the LLM prompt, so the model answers from your documents instead of its training data."

### How you implemented it
1. **Chunking**: Split documents into 500-1000 token windows with 100-token overlap
2. **Embedding**: Used OpenAI `text-embedding-3-small` (1536-dim vectors)
3. **Storage**: Stored vectors in Qdrant, metadata in PostgreSQL
4. **Retrieval**: Embedded the query, cosine similarity search, top-5 chunks
5. **Generation**: Passed context + query to GPT-4o with citation instructions
6. **Validation**: Critic Agent verified the answer was grounded in context

### Why 500-1000 tokens?
- Too small (< 100 tokens): context is incomplete, poor answers
- Too large (> 1500 tokens): retrieves too broadly, injects irrelevant content
- 500-1000 tokens: ~1-2 paragraphs, captures a complete thought

### Why overlap?
An answer-critical sentence might straddle a chunk boundary. With 100-token overlap, it appears in both chunk N and chunk N+1, ensuring retrieval finds it.

### What can go wrong?
- Wrong chunks retrieved → lower similarity threshold
- Answer not grounded → increase Critic strictness
- High latency → cache embeddings, use smaller model for embedding
- High cost → reduce top-k, compress context, use gpt-4o-mini for non-critical steps

---

## Part 2: Vector Databases (Qdrant)

### One-sentence explanation
> "Qdrant stores 1536-dimensional vector embeddings and finds the nearest neighbors to a query vector using cosine similarity, enabling semantic search."

### Key concepts
- **Vector**: list of 1536 floats representing semantic meaning
- **Cosine similarity**: measures the angle between two vectors (1.0 = identical, 0.0 = unrelated)
- **HNSW index**: Hierarchical Navigable Small World — Qdrant's ANN index
- **Payload**: arbitrary JSON metadata stored alongside the vector (chunk_id, page_number, etc.)
- **Filter**: search within a subset (e.g., only chunks from document_id=X)

### Why Qdrant over pgvector?
| Feature | Qdrant | pgvector |
|---------|--------|----------|
| Dedicated vector DB | ✅ | ❌ (PostgreSQL extension) |
| Filtering | Rich | Basic |
| Scalability | Horizontal | Vertical |
| Management | Separate service | Same as Postgres |
| Use case | Production vector search | Simple embeddings in existing DB |

### HNSW vs FLAT index
- FLAT: exact search, O(n), slow for large collections
- HNSW: approximate nearest neighbor, O(log n), slight accuracy trade-off

---

## Part 3: Neo4j and GraphRAG

### One-sentence explanation
> "Neo4j stores entities and their relationships as a graph, enabling traversal queries that find connections that vector similarity search cannot."

### Graph schema you designed
```
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO {relation_type}]->(:Entity)
(:Chunk)-[:BELONGS_TO_TOPIC]->(:Topic)
(:Decision)-[:SUPPORTED_BY]->(:Chunk)
```

### Why a graph over relational tables?
- Find all entities related to "OpenAI" within 2 hops: trivial in graph, complex join in SQL
- Graphs are native for relationship traversal
- Neo4j's HNSW-backed full-text search + relationship traversal in one query

### GraphRAG flow you implemented
1. Vector search retrieves chunks
2. Extract entity names from those chunks (Neo4j lookup)
3. Traverse 2-hop neighbors of those entities
4. Embed related entity names, retrieve more chunks
5. Merge all context → richer answer

### Cypher query safety
- Whitelist: query must start with `MATCH` or `WITH`
- Blacklist: reject `DELETE`, `MERGE`, `CREATE`, `DETACH`, `DROP`
- Force `LIMIT` on all queries
- Execute in read-only session

---

## Part 4: Agentic Workflow

### One-sentence explanation
> "A multi-agent system where specialized agents (Router, Retriever, Cypher, ContextBuilder, Answer, Critic, Memory) each handle one step, passing a shared AgentState through the pipeline."

### The 8 agents you built
| Agent | Input | Output |
|-------|-------|--------|
| Router | user_query | intent (doc_question / graph_question / general_chat) |
| QueryRewrite | user_query, history | rewritten_query |
| Retriever | rewritten_query | retrieved_chunks (via Qdrant) |
| CypherAgent | rewritten_query | cypher_query, cypher_results (via Neo4j) |
| ContextBuilder | chunks, graph_results | final_context |
| Answer | final_context, query | draft_answer |
| Critic | draft_answer, context | validated_answer, is_grounded |
| Memory | validated_answer | updated conversation_history in DB |

### Why a Critic Agent?
LLMs hallucinate. For enterprise applications (legal, medical, financial), a hallucinated answer can cause real damage. The Critic verifies the answer cites only from the provided context. Retry loop allows self-correction.

### Shared state pattern (AgentState TypedDict)
```python
state = {
    "user_query": "...",
    "intent": None,           # → Router fills this
    "rewritten_query": None,  # → QueryRewrite fills this
    "retrieved_chunks": [],   # → Retriever fills this
    ...
}
```
Each agent receives the full state, reads its inputs, writes its outputs. This decouples agents — each can be tested independently.

### LangGraph (Milestone 22 evolution)
In production, use LangGraph to define agents as graph nodes with conditional edges. This enables:
- Visual graph debugging
- Parallel agent execution
- Checkpointing (resume from any step)
- Human-in-the-loop pauses

---

## Part 5: Document Processing Pipeline

### One-sentence explanation
> "Uploaded files are queued via Redis+Celery, parsed into ContentBlocks, normalized, chunked, embedded, and stored in three databases: PostgreSQL (metadata), Qdrant (vectors), Neo4j (entities)."

### Parser coverage
| Format | Library | Key challenge |
|--------|---------|---------------|
| PDF | PyMuPDF | Scanned PDFs need OCR |
| DOCX | python-docx | Preserve heading hierarchy |
| PPTX | python-pptx | Speaker notes, slide titles |
| Excel | openpyxl | Multi-sheet, header rows |
| CSV | pandas | Large files, encoding |
| Image | pytesseract | Accuracy on handwriting |
| Audio | OpenAI Whisper | 25MB limit, segmentation |
| Video | ffmpeg + Whisper | Extract audio first |

### Why content_blocks schema?
- Normalizes diverse formats into a single structure
- Parser doesn't need to know about chunking
- Chunker doesn't need to know about source format
- Easy to add new formats: implement the interface, everything downstream works

### Async processing (Celery + Redis)
- FastAPI returns `202 Accepted` in < 100ms
- Celery worker picks up task from Redis queue
- Worker processes file (may take 30-300 seconds)
- Worker updates PostgreSQL status → COMPLETE
- Frontend polls GET /api/documents/{id} for status

---

## Part 6: Observability (Langfuse)

### One-sentence explanation
> "Langfuse captures every LLM call as a trace+span hierarchy, showing token usage, latency, cost, and citations for every user request."

### Trace hierarchy for one request
```
Trace: chat_request (user_id, session_id, query)
  ├── Span: router_agent (10ms)
  ├── Span: rewrite_agent (300ms, 150 tokens)
  ├── Span: retriever_agent (50ms)
  │     └── Generation: embed_query (50ms, 10 tokens)
  ├── Span: answer_agent (2000ms)
  │     └── Generation: rag_completion (2000ms, 1500 tokens, $0.004)
  └── Span: critic_agent (1500ms)
        └── Generation: evaluate_answer (1500ms, 800 tokens, $0.002)
```

### What you monitor
- **Latency**: p50/p95/p99 for each agent
- **Cost**: per-query and per-user
- **Faithfulness**: Critic score (0-1)
- **User feedback**: thumbs up/down
- **Error rate**: failed agent steps

---

## Part 7: Docker and Kubernetes

### Docker — key concepts used
- **Multi-stage build**: `builder` image compiles Angular, `nginx` image serves static files
- **Non-root user**: `USER appuser` reduces attack surface
- **HEALTHCHECK**: Docker restarts unhealthy containers
- **.dockerignore**: excludes node_modules, __pycache__, venv (keeps images small)
- **Layer caching**: `COPY requirements.txt` before `COPY .` — dependencies only reinstall when requirements change

### Kubernetes — resources you created
| Resource | Type | Purpose |
|----------|------|---------|
| Namespace | N/A | Logical isolation |
| ConfigMap | N/A | Non-sensitive config (DB URLs, ports) |
| Secret | N/A | Sensitive config (API keys, passwords) |
| Deployment | frontend, backend, worker, redis | Stateless apps |
| StatefulSet | postgres, neo4j, qdrant, minio | Stateful apps needing stable storage |
| Service | all | Stable DNS name for each app |
| Ingress | N/A | HTTP routing, TLS termination |
| HPA | backend, worker | Auto-scale based on CPU/memory |
| PVC | postgres, neo4j, qdrant, minio | Persistent storage |

### Why StatefulSet for databases?
- Pods get stable names: `postgres-0`, `postgres-1`
- Each pod gets its own PVC: `postgres-data-0`, `postgres-data-1`
- Ordered start/stop: master (postgres-0) starts before replicas

### HPA scaling
- Backend scales 2→10 pods when CPU > 70%
- Worker scales 1→8 pods when CPU > 80%
- ScaleUp: add 2 pods max per 60s (prevent thrashing)
- ScaleDown: remove 1 pod per 120s with 5-min stabilization

---

## Part 8: Common Interview Questions

**Q: What is RAG and why is it better than fine-tuning?**
> RAG retrieves external knowledge at inference time. Fine-tuning bakes knowledge into weights at training time. RAG is cheaper (no retraining), updatable (add documents without retraining), and more interpretable (you can show which source each claim comes from).

**Q: How do you prevent LLM hallucinations?**
> Three layers: (1) RAG grounds the answer in retrieved documents, (2) the Critic Agent validates that claims are present in the context, (3) citations allow users to verify sources themselves.

**Q: How do you handle large documents?**
> Upload to MinIO (object storage), queue in Redis, Celery worker downloads and processes asynchronously. Chunking splits into 500-1000 token windows. For audio/video, ffmpeg extracts audio, Whisper transcribes in 10-minute segments.

**Q: How do you scale this system?**
> Backend and worker are stateless, scale horizontally via HPA. Databases use StatefulSets with PVCs. Redis is the only single-point of failure in the simple setup — use Redis Cluster for production HA. Qdrant and Neo4j support clustering in enterprise versions.

**Q: How do you monitor costs?**
> Every OpenAI call is estimated with tiktoken (prompt tokens × price + completion tokens × price). Results are traced in Langfuse as `cost_usd`. Alert if per-query cost exceeds $0.10.

**Q: How do you test the RAG quality?**
> RAG evaluation metrics: (1) Retrieval Precision = relevant chunks in top-k, (2) Answer Faithfulness = fraction of claims grounded in context (Critic score), (3) Answer Relevance = does the answer address the question. Use RAGAS library for automated evaluation.

**Q: Explain the Critic Agent pattern.**
> The Critic is a second LLM call that reviews the first answer. It's the "check your work" step. Given the context and the answer, it assesses whether every claim in the answer is supported by the context. If not, it returns feedback, and the Answer Agent tries again (up to 2 retries). This reduces hallucination at the cost of 1-2 extra LLM calls.

**Q: Why use TypedDict for AgentState instead of a Pydantic model?**
> LangGraph (the framework we're moving toward) requires TypedDict for graph state. TypedDict is also more performant than Pydantic since it doesn't validate on every field access — useful when state is read thousands of times per request.

---

## Part 9: System Design Interview Answer

> "I designed a microservices system with three storage tiers: PostgreSQL for structured metadata, Qdrant for vector embeddings, and Neo4j for entity relationship graphs. Document processing is decoupled from the API using Redis-backed Celery queues so a 50-page PDF upload never blocks the chat response. The core query path uses an 8-agent agentic workflow: the Router classifies intent, the Retriever finds relevant chunks via cosine similarity, the Cypher Agent generates Neo4j graph queries for relationship-heavy questions, the Context Builder merges all retrieved information, the Answer Agent generates a GPT-4o response, and the Critic validates it's grounded in evidence before returning it. Every LLM call is traced in Langfuse for latency, cost, and quality monitoring. The entire system is containerized with Docker and deployed to Kubernetes, where stateless services (backend, worker) scale horizontally via HPA while stateful services (databases) use StatefulSets with PVCs."

---

*This guide covers all 30 milestones of the Enterprise AI Chat Assistant project.*
