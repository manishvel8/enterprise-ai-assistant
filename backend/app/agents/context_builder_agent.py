"""
agents/context_builder_agent.py — Context Builder Agent: Merge and format all retrieved context.

Role: Runs after Retriever Agent and/or Cypher Agent.
  Combines vector-retrieved chunks + Neo4j Cypher results into a single
  coherent context string that the Answer Agent will use in its prompt.

Why a separate agent?
  The Answer Agent's prompt has a limited size. The Context Builder:
  1. Deduplicates overlapping chunks (same content, different retrievals)
  2. Re-ranks by relevance score (highest first)
  3. Truncates to fit within MAX_CONTEXT_CHARS
  4. Formats the context with numbered source citations
  5. Includes graph results separately from chunk results

This separation keeps the Answer Agent focused on generation, not formatting.
"""

import logging
from typing import Any, Dict

from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 15_000   # ~4000 tokens, leaves room for prompt + completion


async def run_context_builder_agent(state: AgentState) -> AgentState:
    """
    Context Builder Agent: assemble the final context for the Answer Agent.

    Reads from state: retrieved_chunks, cypher_results
    Writes to state: final_context
    """
    from app.pipeline.graphrag import build_graphrag_context

    chunks = state.get("retrieved_chunks", [])
    cypher_results = state.get("cypher_results", [])

    logger.info(
        f"Context Builder: merging {len(chunks)} chunks + {len(cypher_results)} graph results"
    )

    # Sort chunks by similarity score (highest first)
    if chunks:
        try:
            chunks = sorted(
                chunks,
                key=lambda c: c.get("similarity_score", 0) if isinstance(c, dict) else c.similarity_score,
                reverse=True,
            )
        except Exception:
            pass   # sorting is best-effort

    final_context = build_graphrag_context(chunks, cypher_results)

    # Truncate to fit within limit
    if len(final_context) > MAX_CONTEXT_CHARS:
        final_context = final_context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated due to length]"

    state["final_context"] = final_context
    logger.info(f"Context Builder: {len(final_context)} chars of context assembled")
    state["debug_info"]["context_builder"] = {
        "context_chars": len(final_context),
        "chunk_count": len(chunks),
        "graph_results": len(cypher_results),
    }

    return state
