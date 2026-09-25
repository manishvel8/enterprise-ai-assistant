"""
graph/state.py — Shared state that flows through every LangGraph node.

CONCEPT: LangGraph maintains ONE state object that all nodes read from
and write to. When a node returns a dict, LangGraph merges those keys
back into the state. Every node sees the LATEST version of the state.

Think of it as a shared whiteboard: each agent reads what it needs,
does its work, and writes its result back onto the board.

HOW FIELDS FLOW:
  original_query → [query_validator] → validated_query, validation_result
  validated_query → [router] → route, route_reasoning, human_question
  route → conditional edge → [rag_retriever | web_searcher | human_gate]
  retrieved_docs / web_results / human_input → [response_generator] → generated_response
  generated_response → [quality_checker] → quality_result, revision_feedback
  quality_result → conditional edge → END or back to response_generator
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict, NotRequired


# ─────────────────────────────── Sub-models ───────────────────────────────────

class ValidationResult(TypedDict):
    """Output of query_validator node."""
    is_valid: bool
    reason: str               # why valid or invalid
    sanitized_query: str      # cleaned/normalised version of the query
    domain_relevant: bool     # is it within our knowledge domain?


class RouteDecision(TypedDict):
    """Output of router node."""
    route: str                # "rag" | "web_search" | "needs_human"
    reasoning: str            # why this route was chosen
    human_question: str       # question to ask the human (only for needs_human)


class RetrievedDoc(TypedDict):
    """One document chunk returned from Qdrant."""
    chunk_id: str
    document_id: str
    file_name: str
    chunk_text: str
    score: float              # cosine similarity score
    page_number: Optional[int]
    section_title: Optional[str]


class WebResult(TypedDict):
    """One result returned from Tavily web search."""
    title: str
    url: str
    content: str              # snippet / summary
    score: float              # relevance score from Tavily


class QualityResult(TypedDict):
    """Output of quality_checker node."""
    verdict: str              # "PASS" | "REVISE"
    score: float              # 0.0–1.0  (0 = terrible, 1 = perfect)
    is_grounded: bool         # does answer come from evidence?
    is_complete: bool         # does it fully answer the question?
    issues: list[str]         # list of problems found (empty if PASS)
    revision_feedback: str    # concrete instructions for improvement


class TraceEntry(TypedDict):
    """One row in the execution trace panel on the UI."""
    node: str
    status: str               # "running" | "completed" | "failed" | "interrupted"
    started_at: float         # unix timestamp
    completed_at: NotRequired[float]
    duration_ms: NotRequired[float]
    input_summary: str        # short human-readable description
    output_summary: str
    error: NotRequired[str]


class Source(TypedDict):
    """Citation attached to the final answer."""
    kind: str                 # "document" | "web"
    title: str
    reference: str            # page/URL
    excerpt: str              # short quote from the source


# ─────────────────────────────── Main State ───────────────────────────────────

class AgentState(TypedDict):
    """
    The single shared state object for the entire LangGraph workflow.

    LangGraph passes this to every node. Each node returns a DICT containing
    ONLY the keys it wants to update. LangGraph merges them automatically.

    Example after query_validator runs:
      {
        "original_query": "What is Bhanu Teja's experience?",
        "thread_id": "abc123",
        "validated_query": "What is Bhanu Teja's professional experience?",
        "validation_result": {"is_valid": True, "reason": "...", ...},
        ...all other fields still at their initial values...
      }
    """

    # ── Input ──────────────────────────────────────────────────────
    original_query: str       # raw text the user typed
    thread_id: str            # unique ID for this conversation thread

    # ── Validation (filled by query_validator) ─────────────────────
    validated_query: str
    validation_result: Optional[ValidationResult]

    # ── Routing (filled by router) ─────────────────────────────────
    route: str                # "rag" | "web_search" | "needs_human" | ""
    route_reasoning: str

    # ── RAG evidence (filled by rag_retriever) ─────────────────────
    retrieved_documents: list[RetrievedDoc]

    # ── Web evidence (filled by web_searcher) ──────────────────────
    web_results: list[WebResult]

    # ── Human gate (filled by human_gate) ──────────────────────────
    human_question: str       # question posed to the human
    human_input: str          # answer received from the human

    # ── Combined evidence (filled by response_generator) ───────────
    evidence: str             # formatted context string fed to LLM
    sources: list[Source]     # citations
    generated_response: str   # draft answer

    # ── Quality (filled by quality_checker) ────────────────────────
    quality_result: Optional[QualityResult]
    retry_count: int          # how many revisions so far
    max_retries: int          # ceiling — prevents infinite loop

    # ── Final output ───────────────────────────────────────────────
    final_answer: str         # same as generated_response after PASS
    is_complete: bool

    # ── Observability ──────────────────────────────────────────────
    execution_trace: list[TraceEntry]
    errors: list[str]

    # ── Multi-user / Conversation tracking ─────────────────────────
    # These IDs are set by main.py before the graph starts.
    # Every node can read them to attach Langfuse spans and save messages.
    user_id: str                 # who is asking (e.g. "user_123")
    conversation_id: str         # which conversation this belongs to
    session_id: str              # which browser session / visit
    langfuse_trace_id: str       # the Langfuse trace ID (= request_id)
    conversation_context: str    # pre-built conversation history for LLM
    model_name: str              # which LLM model is being used


def initial_state(
    query: str,
    thread_id: str,
    max_retries: int = 3,
    user_id: str = "anonymous",
    conversation_id: str = "",
    session_id: str = "",
    langfuse_trace_id: str = "",
    conversation_context: str = "",
    model_name: str = "gpt-4o",
) -> AgentState:
    """
    Build the blank initial state before the first node runs.
    Every field is set to a safe default so nodes never see KeyError.

    NEW PARAMS (multi-user support):
      user_id            — user identifier (from auth or request body)
      conversation_id    — which conversation this request belongs to
      session_id         — which browser session this is
      langfuse_trace_id  — Langfuse trace ID created in main.py
      conversation_context — pre-built system prompt with history + memory
      model_name         — LLM model to use
    """
    return AgentState(
        original_query=query,
        thread_id=thread_id,
        validated_query="",
        validation_result=None,
        route="",
        route_reasoning="",
        retrieved_documents=[],
        web_results=[],
        human_question="",
        human_input="",
        evidence="",
        sources=[],
        generated_response="",
        quality_result=None,
        retry_count=0,
        max_retries=max_retries,
        final_answer="",
        is_complete=False,
        execution_trace=[],
        errors=[],
        # Multi-user fields
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        langfuse_trace_id=langfuse_trace_id,
        conversation_context=conversation_context,
        model_name=model_name,
    )
