"""
agents/human_gate.py — NODE 5: Human Gate (Human-in-the-Loop)

After the human chooses docs vs web, we also rewrite validated_query into a
retrieval-friendly search string. Otherwise the graph keeps searching the
meta question ("Should I use docs or internet?") against resumes / the web
and returns weak "context does not provide..." answers.
"""

import logging
import time

from langgraph.types import interrupt

from app.graph.state import AgentState, TraceEntry
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


def _classify_route(human_response: str) -> str:
    resp = human_response.lower().strip()
    # Docs first (so "search uploaded documents" does not match bare "search")
    if any(
        w in resp
        for w in (
            "document",
            "documents",
            "resume",
            "cv",
            "file",
            "uploaded",
            "rag",
            "my docs",
            "local",
        )
    ):
        return "rag"
    if any(
        w in resp
        for w in ("web", "internet", "online", "live web", "tavily", "google")
    ):
        return "web_search"
    return "rag"


def _rewrite_query_for_route(original: str, route: str, human_response: str) -> str:
    """
    Turn a meta / ambiguous user question into something retrievable.
    """
    o = (original or "").strip()
    lower = o.lower()
    is_docs_vs_web = any(
        k in lower
        for k in (
            "document",
            "documents",
            "internet",
            "web",
            "or the",
            "vs",
        )
    ) and ("or" in lower or "vs" in lower or "should i" in lower)

    if route == "web_search":
        if is_docs_vs_web:
            return (
                "When should AI agents use private documents / RAG versus "
                "live web search? Best practices for document QA vs internet research"
            )
        return o or human_response

    # rag
    if is_docs_vs_web:
        return (
            "RAG pipelines document question answering AI agents "
            "using uploaded documents and private knowledge"
        )
    return o or human_response


def human_gate_node(state: AgentState) -> dict:
    """
    Pause execution and request human input.

    IMPORTANT: This node is SYNC (not async).
    interrupt() needs sync graph.stream()/invoke() in LangGraph 1.2.x.

    LANGFUSE: Uses events (not spans) because:
      - The interrupt() pause can last seconds or minutes (human response time)
      - We don't want a "span" that's open for an unknown duration
      - Instead: event "human_interrupt" before + event "human_response" after
    """
    start = time.time()
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))
    original = state.get("validated_query") or state.get("original_query") or ""
    human_question = state.get("human_question") or (
        "I need a bit more context. Should I search your uploaded documents, "
        "or look this up on the web?"
    )

    trace.append(
        TraceEntry(
            node="human_gate",
            status="interrupted",
            started_at=start,
            input_summary=f'Waiting for human: "{human_question[:80]}"',
            output_summary="",
        )
    )

    logger.info("human_gate: interrupting — question=%s", human_question[:80])

    # ── LANGFUSE: log the interruption as a point-in-time event ──────────────
    tracer.event(
        trace_id,
        name="human_interrupt",
        input={"question": human_question, "original_query": original},
        metadata={"node": "human_gate", "status": "waiting_for_human"},
    )

    human_response: str = interrupt(
        {
            "question": human_question,
            "thread_id": state.get("thread_id", ""),
            "node": "human_gate",
        }
    )

    logger.info("human_gate: resumed — response=%s", str(human_response)[:80])

    new_route = _classify_route(str(human_response))
    rewritten = _rewrite_query_for_route(original, new_route, str(human_response))

    duration = (time.time() - start) * 1000
    trace[-1] = TraceEntry(
        node="human_gate",
        status="completed",
        started_at=start,
        completed_at=time.time(),
        duration_ms=round(duration, 1),
        input_summary=f'Question: "{human_question[:80]}"',
        output_summary=(
            f'Human said: "{str(human_response)[:60]}" → route={new_route} | '
            f'search="{rewritten[:50]}"'
        ),
    )

    # ── LANGFUSE: log the human response as a second event ───────────────────
    tracer.event(
        trace_id,
        name="human_response",
        input={"question": human_question},
        output={
            "human_response": str(human_response)[:200],
            "classified_route": new_route,
            "rewritten_query": rewritten,
        },
        metadata={"duration_ms": round(duration, 1)},
    )

    return {
        "human_input": str(human_response),
        "route": new_route,
        # Downstream RAG / web search use validated_query
        "validated_query": rewritten,
        "execution_trace": trace,
        "errors": errors,
    }
