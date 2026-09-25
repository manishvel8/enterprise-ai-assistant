# Rehearsal Script — RAG (C3) + Multi-Agent (C5)

Practice out loud. Time yourself. Use the whiteboard sketches below.

Related: [interview-qa-mapped-to-project.md](../interviews/interview-qa-mapped-to-project.md)

---

## Day drill — C3 RAG (15 minutes)

Speak these answers without looking, then check the number card.

1. **What is RAG?** (30s)  
   Retrieve relevant private chunks → put them in the prompt → LLM answers with citations.

2. **Why not fine-tune?** (30s)  
   Docs change often; RAG updates by re-ingest; cheaper; auditable citations.

3. **Chunking in your project?** (45s)  
   Custom tiktoken chunker: **800 max / 100 overlap / heading-aware** in `chunker.py`.

4. **Embeddings?** (45s)  
   `text-embedding-3-small`, **1536** dims; same model for docs and queries; cosine search in Qdrant.

5. **What if dimension changes?** (20s)  
   Must recreate Qdrant collection and re-embed everything.

6. **End-to-end after user asks a question?** (60s)  
   Router → Rewrite → embed query → top-5 Qdrant → Context → GPT-4o → Critic → Memory → answer+citations.

**Self-score:** If you miss any number (800/100/1536/top_k=5), repeat until automatic.

---

## Day drill — C5 Multi-agent + 4 sources (20 minutes)

### Spoken architecture (90s)

> “I’d put a Router agent in front. Document-like questions go to a Retriever over Qdrant. Relationship questions go to a Cypher agent on Neo4j. Structured DB questions go to a SQL tool agent against Postgres. Excel and website content are ingested into the same chunk/embed pipeline via Celery. Agents share an AgentState object; the orchestrator sequences them and can run Cypher and vector retrieval in parallel. We cut latency with Redis caches, smaller models for routing, streaming, and HPA.”

### Whiteboard — 4-source RAG (draw this)

```text
                    +------------------+
   User Question -> |   Router Agent   |
                    +--------+---------+
         +-----------+-------+------+------------+
         |           |              |            |
         v           v              v            v
   Retriever     Cypher/SQL    Folder/Web     Excel path
   (Qdrant)      Tool Agents   Ingest batch   excel_parser
         |           |              |            |
         +-----+-----+------+-------+------------+
               v
        Context Builder
               v
         Answer Agent (LLM)
               v
          Critic Agent
               v
          Memory / Response
```

### Whiteboard — ingest (draw this)

```text
File -> API -> MinIO
            -> Postgres PENDING
            -> Redis -> Celery Worker
                         |-> parse/normalize
                         |-> chunk 800/100
                         |-> embed 1536
                         |-> Qdrant + Postgres
                         |-> Neo4j entities
                         |-> COMPLETE
```

---

## Mock interview pairs (ask yourself)

| Prompt | Must include |
|--------|----------------|
| “Walk me through RAG” | ingest + retrieve + generate + cite + critic |
| “Chunking params?” | 800 / 100 / tiktoken / why overlap |
| “Vector vs graph?” | Qdrant similarity vs Neo4j traversal; GraphRAG combo |
| “Four data sources?” | Router + tool per source; Excel parser; SQL tool |
| “How do agents talk?” | Shared AgentState; orchestrator in workflow.py |
| “Stateful or not?” | LLM call stateless; app stateful |
| “How reduce latency?” | cache, parallel, smaller router model, stream, HPA |
| “What is MCP?” | protocol for tools/resources; not in this repo |

---

## Checklist before next interview

- [ ] Ran [00-test-the-system.md](../hands-on/00-test-the-system.md) health checks
- [ ] Uploaded fixture and answered Aurora revenue question live
- [ ] Opened Qdrant point payload once
- [ ] Explained Critic in one sentence
- [ ] Drew 4-source whiteboard from memory
- [ ] Said pitch under 60 seconds
- [ ] Practiced honesty: MCP not coded; multi-source SQL/web = design extension

---

## Timed run (record yourself)

| Block | Time | Content |
|-------|------|---------|
| Pitch | 0:00–1:00 | System pitch |
| RAG deep dive | 1:00–4:00 | Chunk → embed → retrieve → generate |
| Agents | 4:00–7:00 | Router through Memory |
| Security | 7:00–8:00 | Infra + Cypher allowlist + Critic |
| Q&A stress | 8:00–12:00 | Random Q13–Q70 from mapped doc |
