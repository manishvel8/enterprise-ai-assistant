"""
agent_state.py — Shared state object passed between all agents.

Every agent in the workflow:
  1. Receives the current AgentState
  2. Reads from it (its inputs)
  3. Writes its outputs back to it
  4. Returns the updated AgentState

This single shared state is the "memory" of the agentic workflow for one request.

Why TypedDict?
  TypedDict gives us type checking without runtime overhead.
  LangGraph expects graph state to be a TypedDict.
"""

from typing import TypedDict, Optional, List, Dict, Any


class RetrievedChunk(TypedDict):
    """A document chunk returned by vector search, with similarity score."""
    chunk_id: str
    document_id: str
    file_name: str
    source_type: str
    page_number: Optional[int]
    slide_number: Optional[int]
    sheet_name: Optional[str]
    timestamp_start: Optional[float]
    timestamp_end: Optional[float]
    section_title: Optional[str]
    chunk_text: str
    similarity_score: float
    metadata: Dict[str, Any]


class Citation(TypedDict):
    """A source citation attached to the answer."""
    chunk_id: str
    document_id: str
    file_name: str
    page_number: Optional[int]
    section_title: Optional[str]
    excerpt: str                    # short snippet from the chunk


class AgentState(TypedDict):
    """
    The shared state object for the entire agentic workflow.

    Passed from agent to agent. Each agent reads what it needs
    and writes its results back into the same object.
    """

    # --- Input (set by the API handler before the workflow starts) ---
    user_id: str
    session_id: str
    user_query: str
    document_ids: List[str]         # optional: restrict retrieval to specific docs

    # --- Router Agent output ---
    intent: Optional[str]
    # Possible values:
    #   "doc_question"   — needs document retrieval via vector search
    #   "graph_question" — needs Neo4j graph traversal
    #   "general_chat"   — no retrieval needed, pure conversation
    #   "summary"        — summarize an entire document
    #   "task"           — needs external tool call

    # --- Query Rewrite Agent output ---
    rewritten_query: Optional[str]

    # --- Retriever Agent output ---
    retrieved_chunks: List[RetrievedChunk]

    # --- Cypher Agent output ---
    cypher_query: Optional[str]
    cypher_results: List[Dict[str, Any]]

    # --- Context Builder Agent output ---
    final_context: Optional[str]

    # --- Answer Agent output ---
    draft_answer: Optional[str]

    # --- Critic Agent output ---
    validated_answer: Optional[str]
    is_grounded: Optional[bool]
    critic_feedback: Optional[str]  # Critic's explanation if answer is not grounded
    critic_iterations: int          # How many times we've looped through critic

    # --- Memory Agent output ---
    citations: List[Citation]
    conversation_history: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]

    # --- Observability (filled throughout the workflow) ---
    latency_ms: Optional[float]
    token_usage: Optional[Dict[str, int]]   # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    cost_usd: Optional[float]
    langfuse_trace_id: Optional[str]
    errors: List[str]
    debug_info: Dict[str, Any]      # arbitrary debug info for the debug panel
