# AI Engineer Interview Preparation Guide

This guide maps every concept in the project to real interview questions. Study this alongside the code.

---

## Section 1: RAG (Retrieval-Augmented Generation)

### What is RAG?

RAG is a pattern where instead of asking an LLM to answer from its training data, you first retrieve relevant context from your own documents, then pass that context to the LLM as part of the prompt. This grounds the answer in your data.

**Without RAG:**
```
User: "What was our Q3 revenue?"
LLM: "I don't have access to your company's financial data." (or worse: hallucinates a number)
```

**With RAG:**
```
User: "What was our Q3 revenue?"
Retriever: finds chunk → "Q3 revenue was $4.2B, up 18% YoY"
LLM: "Based on your Q3 report, revenue was $4.2 billion, an 18% increase year-over-year."
```

### Common RAG Interview Questions

**Q: How do you choose chunk size?**

A: Chunk size is a trade-off between recall and precision. Smaller chunks (200–300 tokens) have higher precision — the retrieved text is more focused — but may miss context. Larger chunks (800–1000 tokens) capture more context but may dilute the relevance signal. We use 500–1000 tokens with 10–20% overlap so that answers that span a chunk boundary are still captured.

**Q: What is embedding overlap and why do you use it?**

A: When you split text into chunks, important information can fall right at the boundary between two chunks. Overlap means we repeat the last N tokens of chunk N at the start of chunk N+1. This ensures boundary-crossing content is captured in at least one chunk.

**Q: How do you handle multi-hop questions?**

A: A multi-hop question requires combining information from multiple chunks. Example: "Which person mentioned in the executive summary also appears in the risk section?" This is where GraphRAG helps — we can traverse the graph from the chunk to the entity and then to other chunks that mention the same entity.

**Q: What is reranking and when do you use it?**

A: Vector search returns the top-k most similar chunks by cosine distance. But similarity doesn't always equal relevance for the specific question. Reranking uses a cross-encoder model (like Cohere Rerank or a local BERT model) to score each retrieved chunk against the query for actual relevance. We apply reranking only when the initial retrieval confidence is low or when the user query is complex.

**Q: How do you prevent hallucination?**

A: Four layers:
1. Context-grounded prompt: "Answer ONLY based on the following context. If the answer is not in the context, say 'I could not find this in the documents.'"
2. Critic Agent: validates that every claim in the draft answer can be traced to a retrieved chunk
3. Citations: force the LLM to cite the source chunk ID for every claim
4. Langfuse monitoring: flag responses where the answer content diverges significantly from retrieved context

---

## Section 2: Vector Databases and Embeddings

**Q: What is a vector embedding?**

A: An embedding is a numerical representation of text (or any data) as a list of floating-point numbers (a vector). Text with similar meaning produces vectors that are geometrically close. OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors.

**Q: What is cosine similarity?**

A: Cosine similarity measures the angle between two vectors. A value of 1.0 means identical direction (very similar meaning). A value of 0.0 means perpendicular (unrelated). A value of -1.0 means opposite. For text retrieval, we want chunks with cosine similarity > 0.75 to the query vector.

```python
# Formula
cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))
```

**Q: Why not use keyword (BM25) search instead of vector search?**

A: Keyword search matches on exact word overlap. It fails when the user uses different words to express the same concept. Vector search understands semantic meaning — "revenue growth" and "sales increase" will have similar embeddings even though they share no words.

**Q: What is HNSW?**

A: Hierarchical Navigable Small World — the indexing algorithm used by Qdrant (and most modern vector databases). It builds a layered graph structure that allows approximate nearest-neighbor search in O(log n) time instead of O(n). The trade-off is a small accuracy loss (1–2%) for a massive speed gain.

**Q: What is metadata filtering in vector search?**

A: When you want to search only within a specific document, time range, or document type, you add a filter condition alongside the vector query. Example: "Find the top 5 chunks most similar to this query, but ONLY from document_id = 'doc_abc123'". This is much more efficient than post-filtering all results.

---

## Section 3: Neo4j and GraphRAG

**Q: When would you use GraphRAG over pure RAG?**

A: When the answer requires traversing relationships between entities. Examples:
- "Which employees are involved in the projects mentioned in the board report?" (Person → Project → Document)
- "What decisions were supported by the revenue analysis?" (Decision → SUPPORTED_BY → Chunk → Metric)
- "Find all risks linked to our partnership with Acme Corp" (Organization → Chunk → Risk)

Pure vector search cannot answer these because it doesn't understand the relationships between entities across documents.

**Q: What is Cypher?**

A: Cypher is Neo4j's query language. It uses a visual, ASCII-art syntax to describe graph patterns.

```cypher
-- "Find all chunks that mention a person who works at Acme Corp"
MATCH (p:Person)-[:WORKS_AT]->(o:Organization {name: "Acme Corp"})
      <-[:MENTIONS]-(c:Chunk)
RETURN p.name, c.text, c.page_number
```

**Q: How do you prevent unsafe Cypher queries generated by an LLM?**

A: Three layers of protection:
1. **Allowlist**: Only permit `MATCH`, `RETURN`, `WHERE`, `WITH`, `LIMIT`, `ORDER BY`
2. **Deny-list**: Block `CREATE`, `DELETE`, `MERGE`, `DROP`, `SET`, `REMOVE`
3. **Read-only user**: The Neo4j user used by the application only has read permissions — even if a destructive query gets through, it will fail at the database level

---

## Section 4: Agentic Workflows

**Q: What is the difference between a chain and an agent?**

A: 
- **Chain**: Fixed sequence of steps. Every request goes through the same steps in the same order. No decision-making.
- **Agent**: Dynamic. The LLM decides which tools/steps to call based on the input. Can handle novel situations a chain cannot.

This project uses a hybrid: the top-level workflow is a LangGraph graph (structured routing), but each agent uses LLM reasoning to make decisions within its scope.

**Q: What is the role of the Router Agent?**

A: The Router Agent uses a fast LLM call (GPT-4o mini) to classify the user's query into one of:
- `doc_question` — needs document retrieval
- `graph_question` — needs Neo4j graph traversal
- `general_chat` — conversational, no retrieval needed
- `summary` — summarize an entire document
- `task` — external tool needed

This classification determines which agents are called next. Routing first saves cost — we don't run expensive retrieval for simple conversational messages.

**Q: Why have a separate Critic Agent?**

A: The Answer Agent is optimized to produce a helpful, fluent answer. It may occasionally be too confident or include details not strictly supported by the context. The Critic Agent is a separate LLM call with a different prompt, focused on verification:
1. Is every claim traceable to a retrieved chunk?
2. Are citations present?
3. Does the answer include anything not in the context?

Having a separate critic prevents the same LLM from grading its own output.

**Q: How do you handle the case where the Critic Agent rejects the answer?**

A: The workflow loops back to the Answer Agent with the Critic's feedback. The Answer Agent revises the answer. We cap this loop at 3 iterations to prevent infinite loops. If validation still fails after 3 attempts, we return the best answer with a caveat.

---

## Section 5: Langfuse Observability

**Q: Why do you need observability for LLM applications?**

A: Unlike traditional APIs where latency and errors are the main concerns, LLM applications have additional failure modes:
- **Hallucination**: The model states something confidently that isn't in the context
- **Cost drift**: A change in prompt length can triple token costs
- **Quality regression**: A new model version might be faster but less accurate for your use case
- **Context window overflow**: Adding more documents to context might push other important content out

Langfuse gives visibility into all of these in production.

**Q: What is a trace, span, and generation in Langfuse?**

A: 
- **Trace**: The top-level container for one user request (one chat message = one trace)
- **Span**: A time-bounded step within the trace (Router Agent span, Retriever Agent span)
- **Generation**: A specific LLM API call within a span, with input/output/tokens/cost recorded

```
Trace: chat_request_abc123
  └── Span: router_agent (200ms)
        └── Generation: gpt-4o-mini, tokens: 150, cost: $0.0001
  └── Span: retriever_agent (50ms)
  └── Span: answer_agent (1500ms)
        └── Generation: gpt-4o, tokens: 2300, cost: $0.023
  └── Span: critic_agent (300ms)
        └── Generation: gpt-4o-mini, tokens: 800, cost: $0.0005
Total: $0.024, 2050ms
```

---

## Section 6: Docker and Kubernetes

**Q: Explain the difference between a Docker image and a container.**

A: An image is a read-only blueprint. A container is a running instance of that blueprint. You can run many containers from the same image. Analogy: image = class definition, container = object instance.

**Q: Why does this project have three separate Docker images?**

A: The frontend, backend, and worker serve different purposes and scale independently:
- Frontend has no Python dependencies — a small Nginx image is sufficient
- Backend needs a full Python environment but is optimized for low latency
- Worker needs the same Python environment as backend but is optimized for throughput and can crash/restart without affecting the API

Separate images mean we can scale workers independently when there's a backlog of documents to process.

**Q: Why use StatefulSet for PostgreSQL and Neo4j instead of Deployment?**

A: A Deployment can move a pod to any node and give it any name. For databases, this would lose the connection to the persistent volume. StatefulSet guarantees:
1. Each pod gets a stable, predictable name (postgres-0)
2. Each pod always reconnects to the same PersistentVolumeClaim
3. Pods start and stop in a predictable order

**Q: Explain how HPA works for the backend.**

A: HPA continuously queries the Kubernetes Metrics Server for CPU and memory usage of the backend deployment. When average CPU across all backend pods exceeds 70%, HPA adds more backend pods (up to the configured maximum of 20). When CPU drops below 70% and pods are idle, HPA removes them (down to the minimum of 3). This ensures the backend can handle traffic spikes without over-provisioning at idle.

**Q: What are liveness and readiness probes?**

A:
- **Liveness probe**: "Is this container alive?" — If it fails, Kubernetes restarts the container. We expose `GET /health` for this.
- **Readiness probe**: "Is this container ready to receive traffic?" — If it fails, Kubernetes removes the pod from the Service's load balancer but does NOT restart it. This is used during startup (waiting for DB connection) or during graceful shutdown.

---

## Section 7: Cost and Performance Optimization

**Q: How do you control OpenAI API costs?**

A: Six strategies:
1. **Embedding cache**: Store embeddings in Redis. If the same text is embedded again (e.g., popular document re-uploaded), return the cached vector instead of calling the API
2. **Retrieval cache**: Store top-k results for popular queries. If the same question is asked again within 1 hour, return cached chunks
3. **Model routing**: Use GPT-4o mini for routing and validation (cheap, fast), GPT-4o only for final answer generation (expensive, high quality)
4. **Prompt caching**: OpenAI caches repeated prompt prefixes. We structure prompts so the system prompt and retrieved context come first, user query last
5. **Top-k optimization**: Don't retrieve 20 chunks if 5 is sufficient. More chunks = larger prompt = higher token cost
6. **Reranking gating**: Only apply reranking (which adds latency and cost) when initial retrieval confidence is below threshold

**Q: How would you handle 1000 concurrent users?**

A: 
1. HPA scales backend pods from 3 to 20 based on CPU
2. Each backend pod handles async requests with uvicorn workers
3. Redis rate limiting prevents any single user from overwhelming the OpenAI API
4. Global semaphore limits concurrent OpenAI API calls to prevent hitting rate limits
5. Retrieval cache serves repeated queries without touching the LLM
6. Streaming responses reduce perceived latency — users see text appearing within 500ms even if the full response takes 3 seconds

---

## Section 8: Common Failure Scenarios and How to Handle Them

| Failure | Detection | Recovery |
|---------|-----------|---------|
| OpenAI API timeout | Circuit breaker | Retry with exponential backoff (max 3 retries) |
| Qdrant returns empty results | Check retrieved_chunks length | Return "no relevant documents found" with no hallucination |
| Neo4j Cypher error | Try/except on query execution | Return empty graph results, log error to Langfuse |
| Celery worker crash | Task in Redis stays in queue | Celery acks task only after completion; if worker crashes, task is re-queued |
| PostgreSQL connection pool exhausted | SQLAlchemy pool overflow error | Increase pool size, add connection timeout, queue requests |
| Document too large for context window | Token count exceeds limit | Truncate context, prioritize highest-similarity chunks, log warning |
| User uploads malicious file | File type validation fails | Return 400, log attempt, do not process file |
