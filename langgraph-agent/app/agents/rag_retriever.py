"""
agents/rag_retriever.py — NODE 3: RAG Retriever

ROLE:
  Embeds the validated query and searches Qdrant for the most
  relevant document chunks. Returns them as RetrievedDoc objects.

FLOW:
  validated_query
    → text-embedding-3-small → 1536-dim vector
    → Qdrant cosine search (top_k=5, threshold=0.25)
    → list[RetrievedDoc] sorted by similarity score

WHY EMBEDDING THE QUERY?
  We can't search vectors with text directly. We convert the query to the
  same vector space as our stored chunk embeddings, then measure cosine
  similarity. Chunks close in meaning will be geometrically close.

OUTPUT KEYS:
  retrieved_documents — list of matching chunks
  execution_trace     — appended TraceEntry
"""

import logging
import time

from app.config import settings
from app.graph.state import AgentState, RetrievedDoc, TraceEntry
from app.services import llm_service, qdrant_service
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


async def rag_retriever_node(state: AgentState) -> dict:
    start = time.time()
    query = state.get("validated_query") or state["original_query"]
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))
    via = "MCP" if settings.use_mcp_tools else "direct"

    trace.append(
        TraceEntry(
            node="rag_retriever",
            status="running",
            started_at=start,
            input_summary=f'Searching Qdrant ({via}) for: "{query[:80]}"',
            output_summary="",
        )
    )

    # ── LANGFUSE: parent span for entire RAG retrieval ────────────────────────
    with tracer.span(
        trace_id,
        name="rag_retrieval",
        input={"query": query, "via": via},
        metadata={"node": "rag_retriever", "top_k": 5, "threshold": 0.25},
    ) as rag_span:
        try:
            if settings.use_mcp_tools:
                from app.services import mcp_client
                raw = await mcp_client.search_documents_via_mcp(query, top_k=5)
                docs: list[RetrievedDoc] = [
                    RetrievedDoc(
                        chunk_id=d.get("chunk_id", ""),
                        document_id=d.get("document_id", ""),
                        file_name=d.get("file_name", "unknown"),
                        chunk_text=d.get("chunk_text", ""),
                        score=float(d.get("score") or 0),
                        page_number=d.get("page_number"),
                        section_title=d.get("section_title"),
                    )
                    for d in raw
                    if isinstance(d, dict)
                ]
            else:
                # ── LANGFUSE: child generation for embedding call ──────────────
                embed_start = time.time()
                with tracer.generation(
                    trace_id,
                    name="query_embedding",
                    model=settings.openai_embedding_model,
                    input={"text": query, "model": settings.openai_embedding_model},
                    parent_span_id=rag_span.id,
                ) as embed_gen:
                    query_vector = await llm_service.embed(query)
                    embed_ms = (time.time() - embed_start) * 1000
                    # Embedding: 1 input text → 1 vector (token count ≈ query length)
                    query_tokens = len(query.split())
                    tracer.update_generation_tokens(embed_gen, query_tokens, 0)
                    embed_gen.update(
                        output={"vector_dim": len(query_vector)},
                        metadata={"duration_ms": round(embed_ms, 1)},
                    )

                # ── LANGFUSE: child span for vector search ────────────────────
                search_start = time.time()
                with tracer.span(
                    trace_id,
                    name="vector_search",
                    input={"top_k": 5, "threshold": 0.25},
                    parent_span_id=rag_span.id,
                ) as search_span:
                    docs = await qdrant_service.search(
                        query_vector=query_vector,
                        top_k=5,
                        score_threshold=0.25,
                    )
                    search_ms = (time.time() - search_start) * 1000
                    scores = [d.get("score", 0) if isinstance(d, dict) else getattr(d, "score", 0) for d in docs]
                    search_span.update(
                        output={"chunk_count": len(docs), "scores": scores[:5]},
                        metadata={
                            "duration_ms": round(search_ms, 1),
                            "collection": settings.qdrant_collection,
                        },
                    )

            duration = (time.time() - start) * 1000
            trace[-1] = TraceEntry(
                node="rag_retriever",
                status="completed",
                started_at=start,
                completed_at=time.time(),
                duration_ms=round(duration, 1),
                input_summary=f'Query: "{query[:80]}" via={via}',
                output_summary=(
                    f"Retrieved {len(docs)} chunks"
                    + (
                        f' | top score={docs[0]["score"]:.3f} from {docs[0]["file_name"]}'
                        if docs
                        else " | no matching chunks found"
                    )
                ),
            )

            # Update parent RAG span with summary
            rag_span.update(
                output={
                    "chunk_count": len(docs),
                    "top_scores": [
                        round(d.get("score", 0) if isinstance(d, dict) else getattr(d, "score", 0), 3)
                        for d in docs[:3]
                    ],
                    "sources": list({
                        d.get("file_name", "") if isinstance(d, dict) else getattr(d, "file_name", "")
                        for d in docs
                    }),
                },
                metadata={"duration_ms": round(duration, 1), "via": via},
            )

            if not docs:
                logger.warning("RAG retrieved 0 chunks for query: %s", query[:60])
                errors.append(
                    "RAG found no matching chunks above threshold 0.25. "
                    "The answer may be based on general knowledge."
                )
                # ── LANGFUSE: event for empty retrieval (useful for debugging) ─
                tracer.event(
                    trace_id,
                    name="empty_retrieval",
                    input={"query": query},
                    output={"chunk_count": 0, "threshold": 0.25},
                    level="WARNING",
                )

            return {
                "retrieved_documents": docs,
                "execution_trace": trace,
                "errors": errors,
            }

        except Exception as e:
            logger.error("rag_retriever failed: %s", e)
            errors.append(f"rag_retriever: {e}")
            trace[-1] = TraceEntry(
                node="rag_retriever",
                status="failed",
                started_at=start,
                completed_at=time.time(),
                input_summary=f'Query: "{query[:80]}"',
                output_summary="",
                error=str(e),
            )
            rag_span.update(level="ERROR", status_message=str(e))
            return {
                "retrieved_documents": [],
                "execution_trace": trace,
                "errors": errors,
            }
