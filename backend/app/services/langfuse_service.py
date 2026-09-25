"""
services/langfuse_service.py — Langfuse LLM observability integration.

What is Langfuse?
  Langfuse is an open-source LLM observability platform (like Datadog for LLM apps).
  It tracks:
  - Every LLM call (prompt, completion, token usage, latency, cost)
  - Full traces (one trace = one user request through all agents)
  - Spans (one span = one agent or sub-step within a trace)
  - User feedback (thumbs up/down on responses)
  - Evaluation metrics (RAG faithfulness, answer relevance)

Why Langfuse?
  Without observability, you're flying blind:
  - Which queries are slow? (latency > 5s = bad UX)
  - Which queries cost the most? (cost per query optimization)
  - Which answers are hallucinating? (faithfulness monitoring)
  - Which document types have the worst retrieval? (RAG quality)

Trace structure for one user request:
  Trace: "chat_request" (user_id, session_id, query)
    ├── Span: "router_agent" (intent classification)
    ├── Span: "rewrite_agent" (query rewriting)
    ├── Span: "retriever_agent" (vector search)
    │     └── Generation: "embed_query" (embedding API call)
    ├── Span: "cypher_agent" (Cypher generation)
    │     └── Generation: "generate_cypher" (GPT-4o call)
    ├── Span: "context_builder_agent"
    ├── Span: "answer_agent"
    │     └── Generation: "rag_completion" (GPT-4o call + tokens)
    ├── Span: "critic_agent"
    │     └── Generation: "evaluate_answer" (GPT-4o call)
    └── Span: "memory_agent"

Self-hosted vs cloud:
  - Cloud: langfuse.com (free tier: 50k observations/month)
  - Self-hosted: runs as a Docker container (see docker-compose.yml)
  - Set LANGFUSE_HOST to switch between them
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_langfuse_client = None


def get_langfuse_client():
    """
    Get or create the Langfuse client singleton.
    Returns None if Langfuse is not configured or package not installed.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse
        from app.core.config import settings

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.info("Langfuse not configured (LANGFUSE_PUBLIC_KEY not set)")
            return None

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or "https://cloud.langfuse.com",
        )
        logger.info(f"Langfuse connected: {settings.langfuse_host or 'cloud.langfuse.com'}")
        return _langfuse_client

    except ImportError:
        logger.warning("langfuse package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"Langfuse initialization failed: {e}")
        return None


class LangfuseTrace:
    """
    Context manager for a Langfuse trace (one user request).

    Usage:
        async with LangfuseTrace(user_id, session_id, query) as trace:
            state = await run_agentic_workflow(...)
            trace.set_output(state)
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        query: str,
        name: str = "chat_request",
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.query = query
        self.name = name
        self._trace = None
        self._client = get_langfuse_client()

    def __enter__(self):
        if self._client:
            try:
                self._trace = self._client.trace(
                    name=self.name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    input={"query": self.query},
                    metadata={"session_id": self.session_id},
                )
            except Exception as e:
                logger.warning(f"Langfuse trace creation failed: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._trace:
            try:
                if exc_type:
                    self._trace.update(
                        status_message=f"Error: {exc_val}",
                        level="ERROR",
                    )
                self._client.flush()
            except Exception as e:
                logger.warning(f"Langfuse trace flush failed: {e}")
        return False   # don't suppress exceptions

    def set_output(self, state: Dict[str, Any]) -> None:
        """Record the final output of the workflow."""
        if not self._trace:
            return
        try:
            self._trace.update(
                output={"answer": state.get("validated_answer", "")[:500]},
                usage={
                    "input": state.get("token_usage", {}).get("prompt_tokens", 0),
                    "output": state.get("token_usage", {}).get("completion_tokens", 0),
                    "total": state.get("token_usage", {}).get("total_tokens", 0),
                },
                metadata={
                    "intent": state.get("intent"),
                    "latency_ms": state.get("latency_ms"),
                    "cost_usd": state.get("cost_usd"),
                    "citation_count": len(state.get("citations", [])),
                    "critic_iterations": state.get("critic_iterations", 0),
                    "errors": state.get("errors", []),
                },
            )
        except Exception as e:
            logger.warning(f"Langfuse trace update failed: {e}")

    @property
    def trace_id(self) -> Optional[str]:
        """Return the Langfuse trace ID for linking in the debug panel."""
        if self._trace:
            return self._trace.id
        return None


def create_span(
    trace_id: Optional[str],
    name: str,
    input_data: Optional[Dict] = None,
) -> Any:
    """
    Create a Langfuse span within an existing trace.

    Use for tracking individual agent steps.
    Returns None if Langfuse is not available.
    """
    client = get_langfuse_client()
    if not client or not trace_id:
        return None

    try:
        return client.span(
            trace_id=trace_id,
            name=name,
            input=input_data or {},
        )
    except Exception as e:
        logger.warning(f"Langfuse span creation failed: {e}")
        return None


def record_generation(
    trace_id: Optional[str],
    name: str,
    model: str,
    prompt: str,
    completion: str,
    usage: Optional[Dict] = None,
    cost_usd: Optional[float] = None,
) -> None:
    """
    Record a single LLM API call (generation) in Langfuse.

    Called by the Answer Agent, Critic Agent, Router Agent, etc.
    """
    client = get_langfuse_client()
    if not client or not trace_id:
        return

    try:
        client.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            model_parameters={"temperature": 0.3},
            input=prompt[:2000],
            output=completion[:2000],
            usage={
                "input": usage.get("prompt_tokens", 0) if usage else 0,
                "output": usage.get("completion_tokens", 0) if usage else 0,
                "total": usage.get("total_tokens", 0) if usage else 0,
            },
            metadata={"cost_usd": cost_usd},
        )
    except Exception as e:
        logger.warning(f"Langfuse generation record failed: {e}")


def record_score(
    trace_id: Optional[str],
    name: str,
    value: float,
    comment: Optional[str] = None,
) -> None:
    """
    Record an evaluation score for a trace (e.g., faithfulness, relevance).

    Used by the Critic Agent to record grounding quality.
    """
    client = get_langfuse_client()
    if not client or not trace_id:
        return

    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as e:
        logger.warning(f"Langfuse score record failed: {e}")
