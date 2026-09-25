"""
agents/workflow.py — The full agentic workflow orchestrator.

This module ties all 9 agents together into a single pipeline.
It implements the flow described in the architecture diagram:

  Router → QueryRewrite → Retriever → ContextBuilder → Answer → Critic → Memory
      ↗ (graph_question) CypherAgent → ContextBuilder
      ↗ (general_chat)              → Answer → Critic → Memory

Architecture pattern: LangGraph-style but custom implementation.
  Each agent is an async function that receives AgentState and returns updated AgentState.
  The workflow function orchestrates the agent calls in the right order.

Why not use LangGraph yet?
  LangGraph is introduced in Milestone 22+ as per the plan.
  This custom implementation teaches the concepts without framework magic.
  When we add LangGraph, we can migrate agents as Graph nodes.

Retry loop (Critic):
  If the Critic rejects the answer, we retry the Answer Agent with the critic feedback.
  Max 2 retries before returning a "could not verify" message.
"""

import logging
import time
from typing import Optional

from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

MAX_CRITIC_RETRIES = 2


async def run_agentic_workflow(
    user_id: str,
    session_id: str,
    user_query: str,
    document_ids: Optional[list] = None,
) -> AgentState:
    """
    Run the complete 9-agent workflow for a user query.

    Args:
        user_id: User identifier
        session_id: Conversation session identifier
        user_query: The raw user query
        document_ids: Optional list to scope retrieval to specific documents

    Returns:
        Final AgentState with validated_answer, citations, and debug_info.
    """
    start_time = time.time()

    # Initialize AgentState
    state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "user_query": user_query,
        "document_ids": document_ids or [],
        "intent": None,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "cypher_query": None,
        "cypher_results": [],
        "final_context": None,
        "draft_answer": None,
        "validated_answer": None,
        "is_grounded": None,
        "critic_feedback": None,
        "critic_iterations": 0,
        "citations": [],
        "conversation_history": [],
        "latency_ms": None,
        "token_usage": None,
        "cost_usd": None,
        "langfuse_trace_id": None,
        "errors": [],
        "debug_info": {},
    }

    # Load conversation history from PostgreSQL
    state = await _load_conversation_history(state)

    # Start Langfuse trace
    from app.services.langfuse_service import LangfuseTrace
    trace = LangfuseTrace(user_id=user_id, session_id=session_id, query=user_query)
    trace.__enter__()

    # ── Agent 1: Router ──────────────────────────────────────────────────────
    from app.agents.router_agent import run_router_agent
    state = await run_router_agent(state)
    intent = state.get("intent", "general_chat")

    # ── Agent 2: QueryRewrite (skip for general_chat) ────────────────────────
    if intent in ("doc_question", "graph_question", "summary"):
        from app.agents.rewrite_agent import run_rewrite_agent
        state = await run_rewrite_agent(state)

    # ── Agent 3: Retriever or Cypher ─────────────────────────────────────────
    if intent == "doc_question" or intent == "summary":
        from app.agents.retriever_agent import run_retriever_agent
        state = await run_retriever_agent(state)

    elif intent == "graph_question":
        # Run both: graph traversal (Cypher) AND vector retrieval
        from app.agents.cypher_agent import run_cypher_agent
        from app.agents.retriever_agent import run_retriever_agent
        state = await run_cypher_agent(state)
        state = await run_retriever_agent(state)

    # ── Agent 4: Context Builder (skip for general_chat) ─────────────────────
    if intent != "general_chat":
        from app.agents.context_builder_agent import run_context_builder_agent
        state = await run_context_builder_agent(state)

    # ── Agent 5+6+7: Answer → Critic loop ────────────────────────────────────
    from app.agents.answer_agent import run_answer_agent
    from app.agents.critic_agent import run_critic_agent

    for attempt in range(MAX_CRITIC_RETRIES + 1):
        state = await run_answer_agent(state)
        state = await run_critic_agent(state)

        if state.get("is_grounded", True):
            break

        if attempt < MAX_CRITIC_RETRIES:
            # Inject critic feedback so Answer Agent can improve
            feedback = state.get("critic_feedback", "")
            state["user_query"] = f"{user_query}\n\n[Note: Previous answer was not grounded. {feedback}]"
            logger.info(f"Critic retry {attempt + 1}/{MAX_CRITIC_RETRIES}")

    # If all retries exhausted and still not grounded
    if not state.get("validated_answer"):
        state["validated_answer"] = (
            "I was unable to generate a verified answer from the available documents. "
            "Please try rephrasing your question or uploading more relevant documents."
        )
        state["citations"] = []

    # ── Agent 8: Memory ──────────────────────────────────────────────────────
    from app.agents.memory_agent import run_memory_agent
    state = await run_memory_agent(state)

    # Calculate latency
    state["latency_ms"] = (time.time() - start_time) * 1000

    # Finish Langfuse trace
    trace.set_output(state)
    state["langfuse_trace_id"] = trace.trace_id
    trace.__exit__(None, None, None)

    logger.info(
        f"Workflow complete: intent={intent}, "
        f"latency={state['latency_ms']:.0f}ms, "
        f"answer_length={len(state.get('validated_answer', ''))}"
    )

    return state


async def _load_conversation_history(state: AgentState) -> AgentState:
    """Load recent conversation history from PostgreSQL for context."""
    try:
        from app.db.postgres import get_session_factory, get_messages_for_session
        session_factory = get_session_factory()
        async with session_factory() as db:
            messages = await get_messages_for_session(db, state["session_id"], limit=20)
            state["conversation_history"] = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
    except Exception as e:
        logger.warning(f"Could not load conversation history: {e}")
    return state
