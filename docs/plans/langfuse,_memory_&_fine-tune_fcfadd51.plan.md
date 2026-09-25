---
name: Langfuse, Memory & Fine-Tune
overview: Extend the existing `langgraph-agent` app with full Langfuse observability (self-hosted Docker), multi-user conversation history with PostgreSQL, context window management, semantic memory retrieval, a fine-tuning pipeline (simulated), and one comprehensive interactive HTML guide covering all 20 parts.
todos:
  - id: langfuse-docker
    content: Create docker-compose.langfuse.yml for self-hosted Langfuse v3 (Postgres + ClickHouse), update .env and config.py with Langfuse + DB vars
    status: in_progress
  - id: langfuse-client
    content: "Create app/observability/langfuse_client.py: Langfuse SDK wrapper with create_trace, add_span, add_generation, add_event, score_trace, flush"
    status: pending
  - id: db-schema
    content: Create app/database/schema.sql (users, conversations, sessions, messages, memory with pgvector), and app/database/db_client.py (psycopg2 pool)
    status: pending
  - id: memory-layer
    content: Create app/memory/conversation_manager.py (CRUD), app/memory/context_manager.py (sliding window, summarization, semantic retrieval), app/memory/memory_retrieval.py (embed + pgvector search)
    status: pending
  - id: instrument-nodes
    content: "Wrap all 7 agent nodes with Langfuse spans/generations: query_validator, router, rag_retriever (with child spans), web_searcher, human_gate (events), response_generator (generation), quality_checker (score)"
    status: pending
  - id: update-state-and-main
    content: Add user_id, conversation_id, session_id, langfuse_trace_id to AgentState; update main.py /api/chat to create trace, load history, retrieve memory, save messages, flush Langfuse
    status: pending
  - id: finetuning
    content: Create app/fine_tuning/dataset_builder.py, pipeline.py (full OpenAI fine-tune code, simulated run), evaluation.py (base vs fine-tuned comparison with Langfuse experiment tags)
    status: pending
  - id: update-requirements
    content: Add langfuse>=2.0, psycopg2-binary, pgvector to requirements.txt
    status: pending
  - id: html-guide
    content: "Create docs/hands-on/05-observability-memory-finetuning.html: 15-tab interactive guide with simulated Langfuse trace explorer, multi-user conversation demo, context window visualizer, fine-tuning pipeline visual, and complete request flow animation"
    status: pending
isProject: false
---

# Langfuse Observability, Conversation Memory & Fine-Tuning Plan

## What We Are Building On

The existing `langgraph-agent/` already has:
- 7-node LangGraph workflow (`query_validator → router → rag_retriever/web_searcher/human_gate → response_generator → quality_checker`)
- FastAPI with SSE streaming at port 7860
- `AgentState` TypedDict shared across all nodes
- Sync OpenAI + Qdrant + Tavily clients

## New Directory Structure

```
langgraph-agent/
├── app/
│   ├── observability/
│   │   └── langfuse_client.py       # Langfuse SDK wrapper (new)
│   ├── memory/
│   │   ├── conversation_manager.py  # conversation CRUD (new)
│   │   ├── context_manager.py       # context window strategies (new)
│   │   └── memory_retrieval.py      # semantic memory search (new)
│   ├── database/
│   │   ├── schema.sql               # PostgreSQL DDL (new)
│   │   └── db_client.py             # psycopg2 connection pool (new)
│   └── fine_tuning/
│       ├── dataset_builder.py       # dataset formatting (new)
│       ├── pipeline.py              # fine-tuning pipeline (new)
│       └── evaluation.py            # base vs fine-tuned comparison (new)
├── docker-compose.langfuse.yml      # self-hosted Langfuse (new)
└── requirements.txt                 # add langfuse, psycopg2-binary, pgvector
docs/
└── hands-on/
    └── 05-observability-memory-finetuning.html  # master HTML guide (new)
```

---

## Task Breakdown

### 1. Langfuse Docker Setup (`docker-compose.langfuse.yml`)
- Langfuse v3 with its own Postgres + ClickHouse via official Docker Compose
- Environment variables: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST=http://localhost:3000`
- Add to `.env` and `app/config.py`

### 2. `app/observability/langfuse_client.py`
Core wrapper exposing:
```python
langfuse = Langfuse(public_key=..., secret_key=..., host=...)

def create_trace(request_id, user_id, session_id, query, tags)
def add_span(trace, name, input, metadata)
def add_generation(span, model, input_tokens, output_tokens, cost)
def add_event(trace, name, input, output)
def score_trace(trace_id, name, value, comment)
def flush()
```
Each LangGraph node will call `add_span()` with node name, inputs, outputs, latency, and errors.

### 3. Instrument All 7 Agent Nodes
Edit each agent file to wrap execution in a Langfuse span:
- `query_validator.py` → span `query_validation`, metadata: sanitized_query, is_valid
- `router.py` → span `routing`, metadata: route, reasoning
- `rag_retriever.py` → parent span `rag_retrieval` with child spans: `embedding` (generation with tokens), `vector_search` (metadata: top_k, scores), optional `reranking`
- `web_searcher.py` → span `web_search`, metadata: result count, scores
- `human_gate.py` → event `human_interrupt`, event `human_response`
- `response_generator.py` → span `response_generation` + generation `llm_call` (model, input/output tokens, cost, latency)
- `quality_checker.py` → span `quality_check` + generation `llm_evaluation` + score on trace

### 4. `app/main.py` — Trace Lifecycle
- On every `/api/chat` request: create Langfuse trace with `request_id`, `user_id` (from header/body), `session_id`, `conversation_id`
- Pass `trace` into `AgentState` (or thread-local) so all nodes can attach spans
- On stream completion: call `langfuse.flush()` and set trace status

### 5. PostgreSQL Schema (`app/database/schema.sql`)

```sql
-- users, conversations, sessions, messages, memory tables
CREATE TABLE users (id UUID PRIMARY KEY, ...);
CREATE TABLE conversations (id UUID, user_id UUID REFERENCES users, ...);
CREATE TABLE sessions (id UUID, conversation_id UUID REFERENCES conversations, ...);
CREATE TABLE messages (id UUID, conversation_id UUID, session_id UUID,
  role TEXT, content TEXT, token_count INT, embedding VECTOR(1536), created_at TIMESTAMPTZ);
CREATE TABLE memory (id UUID, user_id UUID, memory_type TEXT,
  content TEXT, embedding VECTOR(1536), created_at TIMESTAMPTZ);
CREATE INDEX ON messages USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON memory USING ivfflat (embedding vector_cosine_ops);
```
Uses `pgvector` extension for semantic search.

### 6. `app/memory/conversation_manager.py`
- `create_user(name)`, `get_or_create_conversation(user_id, title)`
- `create_session(conversation_id)`, `save_message(conversation_id, session_id, role, content)`
- `get_recent_messages(conversation_id, limit=20)`, `get_token_count(messages)`

### 7. `app/memory/context_manager.py`
Implements all context window strategies:
```python
def sliding_window(messages, max_tokens=8000)         # last N messages within token budget
def summarize_old_messages(messages, llm)             # LLM summarizes messages > threshold
def semantic_retrieve(query, conversation_id, top_k)  # vector search over message embeddings
def build_context(messages, strategy, query, ...)     # master function
```

### 8. `app/memory/memory_retrieval.py`
- `store_memory(user_id, content, memory_type)` — embeds and stores in `memory` table
- `retrieve_relevant_memory(user_id, query, top_k=5)` — pgvector cosine search
- `inject_memory_into_prompt(base_prompt, memories)` — appends retrieved memories

### 9. `/api/chat` Updated Request Flow
```
POST /api/chat
  body: {query, user_id, conversation_id?, thread_id?}
  1. Get/create user → conversation → session
  2. Load recent messages (sliding window / semantic)
  3. Retrieve long-term memory
  4. Build context prompt
  5. Create Langfuse trace
  6. Run LangGraph with context-augmented state
  7. Save assistant message + update memory
  8. Flush Langfuse
  9. Return SSE stream
```

### 10. `app/fine_tuning/dataset_builder.py`
- `build_jsonl_dataset(examples)` — formats messages as OpenAI fine-tune JSONL
- `split_dataset(data, train=0.8, val=0.1, test=0.1)`
- `validate_dataset(data)` — checks format, min examples, no leakage
- `estimate_cost(data)` — token count × cost/token

### 11. `app/fine_tuning/pipeline.py`
```python
# Simulated pipeline (no real API call charged) with real code structure
def prepare_dataset() → train/val/test splits
def upload_dataset(path) → file_id        # real OpenAI API call
def start_finetuning(file_id, model)      # real OpenAI API call
def poll_status(job_id)                   # check until complete
def deploy(model_id)                      # update config to use fine-tuned model
```

### 12. `app/fine_tuning/evaluation.py`
- `evaluate_model(model_name, test_cases)` — runs same prompts on base + fine-tuned
- `compute_metrics(responses)` — accuracy, format adherence, latency, token count, cost
- `compare_experiments(base_results, ft_results)` — side-by-side diff
- Logs both experiments to Langfuse with `tags=["experiment-A"]` vs `tags=["experiment-B"]`

### 13. `docs/hands-on/05-observability-memory-finetuning.html`
One self-contained HTML file (~3,000 lines) with 15 tabbed sections:

- **Architecture** — full system diagram (Mermaid-style ASCII in HTML)
- **Langfuse** — what/why Langfuse, hierarchy definitions (Trace/Span/Generation/Event)
- **Traces** — simulated interactive trace explorer (expandable parent-child nodes with latency, tokens, cost per span)
- **Metrics** — dashboard examples (daily requests, P95 latency, cost/user, error rate)
- **RAG Observability** — retrieval metrics, empty-retrieval detection
- **Agent Observability** — node execution sequence, router decision, tool calls
- **Fine-Tuning** — training pipeline visual + dataset format + base vs fine-tuned comparison table
- **Model Evaluation** — side-by-side output with accuracy/hallucination/latency/cost metrics
- **Conversation History** — multi-user simulator (switch User 1/2/3, separate conversations, message history)
- **Sessions** — session ID lifecycle, session vs conversation distinction
- **Context Window** — visual token budget bar (system=2K, messages=20K, docs=40K, memory=5K, query=1K, remaining), truncation/summarization/semantic demo
- **Memory** — short-term vs long-term, Redis vs PostgreSQL, memory retrieval flow
- **Multi-User Architecture** — architecture evolution for 10 → 1K → 100K → 1M users
- **Database** — SQL schema viewer with explanation per table
- **Complete Request Flow** — animated step-by-step flow: `New Message → Load History → Retrieve Memory → Build Context → LangGraph → Langfuse → Save → Response`

---

## Key Files Changed

- [`langgraph-agent/requirements.txt`](enterprise-ai-assistant/langgraph-agent/requirements.txt) — add `langfuse>=2.0`, `psycopg2-binary`, `pgvector`
- [`langgraph-agent/.env`](enterprise-ai-assistant/langgraph-agent/.env) — add Langfuse + Postgres vars
- [`langgraph-agent/app/config.py`](enterprise-ai-assistant/langgraph-agent/app/config.py) — add Langfuse + DB settings
- [`langgraph-agent/app/graph/state.py`](enterprise-ai-assistant/langgraph-agent/app/graph/state.py) — add `user_id`, `conversation_id`, `session_id`, `langfuse_trace_id` fields
- Each of the 7 agent node files — wrap in Langfuse spans
- [`langgraph-agent/app/main.py`](enterprise-ai-assistant/langgraph-agent/app/main.py) — trace lifecycle + conversation endpoints

---

## Langfuse Trace Hierarchy (designed for this app)

```
Trace: request_<uuid>  [user=user_123, session=session_9]
├── Span: query_validation  (30ms)
│     metadata: {is_valid, sanitized_query}
├── Span: routing  (20ms)
│     metadata: {route="rag", reasoning}
├── Span: rag_retrieval  (350ms)
│     ├── Generation: embedding  (70ms, 8 input tokens, cost=$0.0001)
│     ├── Span: vector_search  (120ms, top_k=5, scores=[0.82, 0.79, ...])
│     └── Span: reranking  (160ms, 3 chunks selected)
├── Span: response_generation  (1800ms)
│     └── Generation: llm_call  (1700ms, gpt-4o, in=1420 out=320 tokens, cost=$0.011)
└── Span: quality_check  (400ms)
      └── Generation: llm_evaluation  (380ms)
      Score: quality_score=0.92
```

---

## Running Steps (after implementation)

```bash
# 1. Start Langfuse (self-hosted Docker)
cd langgraph-agent && docker compose -f docker-compose.langfuse.yml up -d
# Open http://localhost:3000 → create project → copy keys → paste in .env

# 2. Start PostgreSQL (existing Docker stack)
docker compose up -d postgres

# 3. Apply schema
psql -h localhost -p 5433 -U ai_user -d ai_assistant -f app/database/schema.sql

# 4. Activate venv and install new deps
source venv/bin/activate && pip install langfuse psycopg2-binary pgvector

# 5. Start LangGraph app
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 6. Open HTML guide
open docs/hands-on/05-observability-memory-finetuning.html
```
