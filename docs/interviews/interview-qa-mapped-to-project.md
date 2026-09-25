# Interview Q1–Q81 Mapped to Enterprise AI Assistant

Use **this project** as your primary story. Be honest about what is implemented vs designed.

**60-second pitch (memorize):**

> Users upload enterprise documents; Celery workers parse, chunk at 800 tokens with 100 overlap, embed with text-embedding-3-small into Qdrant, and store metadata in Postgres. On chat, a Router classifies intent; for document questions we rewrite the query, retrieve top-5 cosine-similar chunks, generate with GPT-4o, and a Critic checks faithfulness before Memory persists the turn. Graph questions also run Cypher against Neo4j. The stack is Dockerized and Kubernetes-ready.

Hands-on docs: [`docs/hands-on/`](../hands-on/).

---

## C1 — Intro / experience (Q1–Q8, Q71–Q72)

### Q1. Introduce yourself and technical experience
**Say:** Backend / AI engineer focused on GenAI systems — FastAPI, RAG, vector DBs, Neo4j GraphRAG, multi-agent workflows, Docker/K8s. Flagship project: Enterprise AI Chat Assistant (this repo).

### Q2. Total experience
Answer with your real years. Tie last N years to Python, APIs, and GenAI.

### Q3. Experience in GenAI / Agentic AI / DS / ML
**Say:** Hands-on GenAI: RAG pipelines, embeddings, OpenAI APIs, agentic routing/retrieval/critic loops. Agentic AI: multi-agent workflow with shared state. Classic ML: mention only projects you actually did (e.g. tic-tac-toe Q-learning separately).

### Q4–Q8. Use case, problem, approach, role
| Question | Answer from this project |
|----------|---------------------------|
| Use case | ChatGPT-like enterprise assistant over private docs |
| Problem | Employees need grounded answers from PDFs/office/media with citations — not generic LLM knowledge |
| Approach | RAG + GraphRAG + 8-agent workflow + async ingestion |
| Role | Architecture, ingestion pipeline, retrieval, agents, APIs, Docker/K8s manifests |

### Q71–Q72. How many GenAI projects / years
Answer honestly; position this as your deepest end-to-end GenAI system.

---

## C2 — Security / guardrails (Q9–Q12)

### Q9–Q12. Protect company data before LLM; security; guardrails
**Implemented in this project:**
- Original files stay in **MinIO** (your infra); only retrieved **chunk text** goes to OpenAI for answers/embeddings
- Secrets in `.env` / K8s Secrets — never in source
- Rate limiting (`rate_limiter.py`) — 30 chat/min
- Cypher allowlist — blocks DELETE/CREATE/MERGE/DETACH
- Critic agent — grounding / faithfulness guardrail
- CORS + upload validation (type/size)

**Designed / say if asked for more:**
- PII redaction before LLM (not fully implemented — say you’d add a pre-LLM scrubber)
- VPC / private endpoints for OpenAI if enterprise policy requires
- Tenant isolation by `document_ids` / user ACL

**Hands-on:** Show `k8s/secret.yaml` vs `configmap.yaml`; show `validate_cypher` rejecting writes.

---

## C3 — RAG core (Q13–Q30) — highest priority

### Q13. How did you build the RAG application?
Ingest → chunk → embed → Qdrant; query → embed → top-k → prompt with context → GPT-4o → citations → Critic.

### Q14. Frameworks?
FastAPI, OpenAI Python SDK, Celery, Redis, SQLAlchemy, Qdrant client, Neo4j driver, Angular. Custom agents (LangGraph-ready). Not mandatory LangChain for chunking.

### Q15. Complete RAG workflow?
See [02-query-rag-agents-e2e.md](../hands-on/02-query-rag-agents-e2e.md).

### Q16. Why RAG?
Private, updatable knowledge; citations; cheaper than fine-tuning; no retrain when docs change.

### Q17. Other approaches?
Fine-tuning, stuffing whole docs in context, pure SQL/search. Tradeoffs: fine-tune = stale & costly; long context = $ and noise; keyword search = misses paraphrases.

### Q18. Disadvantages of RAG?
Bad chunks → bad answers; retrieval misses; latency/cost of embed+LLM; chunk boundary loss; needs good eval.

### Q19. Process PDFs / websites?
PDFs: `pdf_parser.py` (PyMuPDF) → ContentBlocks. Websites (design for multi-source Q): crawl/scrape → same normalize→chunk→embed path.

### Q20–Q24. Chunking parameters / who did it / LangChain function?
| Param | Value |
|-------|-------|
| Max | **800 tokens** |
| Overlap | **100 tokens** |
| Min | **50** |
| Tokenizer | tiktoken `cl100k_base` |
| Style | Metadata-aware, heading-aware |

File: `backend/app/pipeline/chunker.py` — **you** (this project) implemented it. Not LangChain `RecursiveCharacterTextSplitter` (mention as alternative).

### Q25–Q26. Embeddings / model?
`embed_chunks` / `embed_query` in `embedder.py` → OpenAI **`text-embedding-3-small`**.

### Q27. Mechanism behind embeddings?
Neural net maps text → dense vector; training makes semantically similar texts close in vector space.

### Q28–Q29. Embedding dimension / 1536?
Length of the vector (features). 1536 is the default size for `text-embedding-3-small` in this project — capacity for nuanced similarity.

### Q30. Change 1536 → 1600?
Breaks Qdrant collection (size mismatch). Must recreate collection and re-embed all docs with a model that outputs 1600 dims. Keep query and doc embeddings identical.

---

## C4 — Vector + graph (Q31–Q38, Q69–Q70)

### Q31. Is Neo4j vector or graph?
**Graph database** (property graph). (Neo4j also has vector indexes in newer versions — we use it for relationships; Qdrant for vectors.)

### Q32–Q34. Store in graph? Combine? Which graph DB?
Yes — Document/Chunk/Entity/Topic in Neo4j. Combine via GraphRAG: vector retrieve → Neo4j neighbors → enrich → answer. DB: **Neo4j**.

### Q35–Q36. Which vector DB?
**Qdrant**, collection `chunks`, Cosine, 1536-d.

### Q37. Other vector DBs?
Pinecone, Weaviate, Chroma, Milvus, pgvector, Elasticsearch kNN.

### Q38. Other graph DBs?
Amazon Neptune, ArangoDB, TigerGraph, JanusGraph.

### Q69–Q70. Integrate vector DB with LLM; after storing, how does LLM use it?
Backend embeds query → Qdrant returns chunk texts → texts go into the **prompt** → LLM generates. LLM does not read vectors directly.

---

## C5 — Multi-source + agents (Q39–Q51)

### Q39–Q43. Four sources: website, folder, Postgres, Excel
| Source | Approach in this architecture |
|--------|-------------------------------|
| Website | Scraper/crawler job → HTML→text → same chunk/embed pipeline |
| Local folder | Batch upload to MinIO → Celery fan-out |
| PostgreSQL | **SQL tool agent** for structured queries + optional embed of text columns |
| Excel | `excel_parser.py` → ContentBlocks → same pipeline |

### Q44. Identify which source?
**Router Agent** classifies intent; optionally a source-router with tools per store (vector vs SQL vs graph).

### Q45–Q49. Agentic design / integrate agents / routing / communication
- Agents: Router, Rewrite, Retriever, Cypher, Context, Answer, Critic, Memory (+ Tool agents per source)
- Main orchestrator: `workflow.py` with shared **`AgentState` TypedDict**
- Routing: Router sets `intent`; conditional branches
- Communication: in-process shared state (not message bus); production could use LangGraph / queues

### Q50. Sequential or parallel?
Mostly sequential; for `graph_question` we run Cypher + Retriever in the same turn (can parallelize with asyncio). Critic loop is sequential retries.

### Q51. Reduce response time?
Cache embeddings/retrieval (Redis), smaller model for Router/Critic, parallel I/O, lower top_k, SSE streaming, skip Critic for low-risk intents, HPA scale workers/API.

---

## C6 — MCP / memory / stateful (Q52–Q68)

### Q52–Q54. What is MCP? How useful?
**Model Context Protocol** — open standard for connecting AI apps to tools/data (resources, prompts, tools) with a consistent host↔server interface. Useful so agents discover tools without custom glue per SaaS. **Not implemented in this repo** — map conceptually to our retriever/Cypher as “tools.”

### Q55–Q57. Short-term vs long-term memory
| Type | In this project |
|------|-----------------|
| Short-term | Last N turns in `conversation_history` / AgentState (session) |
| Long-term | Postgres `MessageModel` / sessions; optional future: vector memory of facts |

Difference: short-term = current conversation window; long-term = durable across sessions.

### Q58–Q60. Initialize memory / frameworks / LangChain memory
Load history at workflow start (`_load_conversation_history`). Frameworks: custom state, LangGraph checkpointers, LangChain ConversationBufferMemory / summary memory. We use **custom AgentState + Postgres** (LangGraph-ready).

### Q61–Q64. Mix GPT-3.5/4 + Ada/Gemini with memory
Yes — configure **chat model** and **embedding model** independently in settings. Memory is session-scoped, not model-scoped. Constraint: all vectors in one Qdrant collection must use the **same** embedding model/dimension.

### Q65–Q68. Stateful vs stateless; is LLM stateful?
- Agentic **application**: stateful (AgentState + DB)
- Individual LLM API call: **stateless** (each call needs full context)
- Stateful = retains state across requests; stateless = does not

---

## C7 — Tools, Python, ML side (Q73–Q80)

### Q73. Decorators in Python?
Functions wrapping functions. Examples: `@celery_app.task`, FastAPI `@router.post`, `@app.get`.

### Q74–Q75. Tools / toolkits; how agents use them?
Tools = callable capabilities (search, Cypher, SQL). Agents decide when to call them. Here: Retriever and Cypher act as tools; Critic validates outputs.

### Q76–Q78. ML / ensemble
Answer from your ML background. Ensembles: bagging (Random Forest), boosting (XGBoost/LightGBM), stacking. Don’t force-fit into this RAG project.

### Q79–Q80. Tic-tac-toe / Q-learning
Answer from that project only: Q-table, reward for win/draw/loss, ε-greedy exploration — unrelated to this RAG repo.

### Q81. Questions for interviewer?
Examples: How do you evaluate RAG in production? Do you prefer managed vector DBs or self-hosted? What’s the latency SLO for agentic workflows?

---

## Quick number card (keep in pocket)

| Topic | Number |
|-------|--------|
| Chunk size | 800 tokens |
| Overlap | 100 tokens |
| Embedding dims | 1536 |
| Embedding model | text-embedding-3-small |
| Chat model | gpt-4o |
| top_k | 5 |
| Similarity threshold | 0.5 |
| Critic retries | 2 |
| Chat rate limit | 30/min |

---

## Honesty notes (interview integrity)

1. Entity → Neo4j is now wired in the Celery worker; if Neo4j/OpenAI unavailable, step is skipped and vector RAG still works.
2. MCP is conceptual knowledge — not coded here.
3. Multi-source website crawl / SQL agent are **design answers** extending this architecture.
4. LangChain memory modules — we use custom state; know the names.
