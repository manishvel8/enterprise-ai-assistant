"""
agents/retriever_agent.py — Retriever Agent: Semantic vector search + optional GraphRAG enrichment.

Role: Called after QueryRewrite for "doc_question" intent.
  Embeds the rewritten query and retrieves the most relevant chunks.
  Optionally enriches results with graph traversal (GraphRAG).

Two modes:
  1. vector_only: Pure semantic similarity search (faster, cheaper)
  2. graphrag: Vector search + Neo4j entity graph traversal (richer context)

The mode is determined by:
  - Whether Neo4j is configured
  - The query complexity (graph_question → graphrag, doc_question → vector_only)
"""

import logging
from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)


async def run_retriever_agent(state: AgentState) -> AgentState:
    """
    Retriever Agent: find relevant document chunks.

    Reads from state: rewritten_query, user_query, document_ids, intent
    Writes to state: retrieved_chunks, debug_info.retriever
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    document_ids = state.get("document_ids") or None
    intent = state.get("intent", "doc_question")

    logger.info(f"Retriever Agent: searching for: {query[:80]}")

    use_graphrag = intent == "graph_question"

    if use_graphrag:
        chunks, graph_debug = await _graphrag_retrieve(query, document_ids)
        state["debug_info"]["retriever"] = {
            "mode": "graphrag",
            **graph_debug,
        }
    else:
        chunks = await _vector_retrieve(query, document_ids)
        state["debug_info"]["retriever"] = {
            "mode": "vector",
            "chunks_found": len(chunks),
        }

    state["retrieved_chunks"] = chunks
    logger.info(f"Retriever Agent: found {len(chunks)} chunks")

    return state


async def _vector_retrieve(query: str, document_ids):
    """Pure vector similarity search."""
    from app.db.retriever import retrieve_chunks
    return await retrieve_chunks(query=query, top_k=5, document_ids=document_ids)


async def _graphrag_retrieve(query: str, document_ids):
    """Vector search enriched with Neo4j graph traversal."""
    from app.pipeline.graphrag import graphrag_retrieve
    chunks, debug = await graphrag_retrieve(query=query, top_k=5, document_ids=document_ids)
    return chunks, debug
