"""
graph/edges.py — Conditional edge functions for the LangGraph StateGraph.

CONCEPT:
  In LangGraph, a conditional edge reads the current state and returns
  a string that maps to the NEXT node to visit.

  builder.add_conditional_edges(
      "source_node",
      edge_function,           ← defined here
      {"return_value": "node_name", ...}
  )

WHY SEPARATE FILE?
  Edge logic can get complex (retry counts, error states, etc.).
  Keeping it here makes workflow.py clean and readable.
  Each function is independently testable.

EDGE MAP (what calls what):
  query_validator → [route_after_validation]
    "invalid" → END
    "valid"   → router

  router → [route_after_router]
    "rag"          → rag_retriever
    "web_search"   → web_searcher
    "needs_human"  → human_gate

  rag_retriever → response_generator (direct edge, no condition)
  web_searcher  → response_generator (direct edge, no condition)
  human_gate    → [route_after_human]
    "rag"        → rag_retriever
    "web_search" → web_searcher

  response_generator → quality_checker (direct edge, no condition)

  quality_checker → [route_after_quality]
    "PASS"   → END
    "REVISE" → response_generator
"""

import logging
from langgraph.graph import END
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_after_validation(state: AgentState) -> str:
    """
    After query_validator:
      - If invalid (e.g. gibberish, injection) → END immediately
      - Otherwise → router
    """
    val = state.get("validation_result")
    if val is None:
        # Validator errored but we still set is_valid=True in the fallback
        logger.debug("edge: validation_result is None → continue to router")
        return "valid"

    is_valid = val.get("is_valid", True) if isinstance(val, dict) else getattr(val, "is_valid", True)

    if not is_valid:
        logger.info("edge: query is invalid → END")
        return "invalid"

    logger.debug("edge: query is valid → router")
    return "valid"


def route_after_router(state: AgentState) -> str:
    """
    After router:
      - "rag"         → rag_retriever
      - "web_search"  → web_searcher
      - "needs_human" → human_gate
    """
    route = state.get("route", "rag")
    logger.info("edge: router decided route=%s", route)

    if route == "web_search":
        return "web_search"
    if route == "needs_human":
        return "needs_human"
    return "rag"  # default


def route_after_human(state: AgentState) -> str:
    """
    After human_gate (post-resume):
    The human answered, and human_gate_node updated state["route"].
    Re-route using the updated value.
    """
    route = state.get("route", "rag")
    logger.info("edge: post-human route=%s", route)
    if route == "web_search":
        return "web_search"
    return "rag"


def route_after_quality(state: AgentState) -> str:
    """
    After quality_checker:
      - verdict = "PASS"   → END (return answer to user)
      - verdict = "REVISE" → response_generator (try again)

    Safety net: if retry_count >= max_retries, always go to END.
    (quality_checker already forces PASS in this case, but double-check here.)
    """
    quality = state.get("quality_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # Hard ceiling — never loop more than max_retries times
    if retry_count >= max_retries:
        logger.info("edge: max retries reached → END")
        return "end"

    if quality is None:
        logger.warning("edge: quality_result is None → END")
        return "end"

    verdict = quality.get("verdict", "PASS") if isinstance(quality, dict) else getattr(quality, "verdict", "PASS")

    if verdict == "REVISE":
        logger.info("edge: verdict=REVISE retry=%d → response_generator", retry_count)
        return "revise"

    logger.info("edge: verdict=PASS → END")
    return "end"
