"""
agents/web_searcher.py — NODE 4: Web Searcher

ROLE:
  Fetches live information from the web using Tavily.
  Called when the router decides the query needs current/recent data
  that isn't in our Qdrant document collection.

FLOW:
  validated_query → Tavily API → list[WebResult]

WHAT TAVILY RETURNS:
  - title: page title
  - url: source URL
  - content: cleaned text snippet (no HTML, no ads)
  - score: Tavily's relevance score (higher = more relevant)

OUTPUT KEYS:
  web_results     — list of WebResult objects
  execution_trace — appended TraceEntry
"""

import logging
import time

from app.config import settings
from app.graph.state import AgentState, TraceEntry, WebResult
from app.services import tavily_service
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


async def web_searcher_node(state: AgentState) -> dict:
    start = time.time()
    query = state.get("validated_query") or state["original_query"]
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))
    via = "MCP" if settings.use_mcp_tools else "direct"

    trace.append(
        TraceEntry(
            node="web_searcher",
            status="running",
            started_at=start,
            input_summary=f'Searching web ({via}) for: "{query[:80]}"',
            output_summary="",
        )
    )

    # ── LANGFUSE: span for the web search node ───────────────────────────────
    with tracer.span(
        trace_id,
        name="web_search",
        input={"query": query, "via": via},
        metadata={"node": "web_searcher", "max_results": 5},
    ) as node_span:
        try:
            if settings.use_mcp_tools:
                from app.services import mcp_client
                raw = await mcp_client.web_search_via_mcp(query, max_results=5)
                results: list[WebResult] = [
                    WebResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        content=r.get("content", ""),
                        score=float(r.get("score") or 0),
                    )
                    for r in raw
                    if isinstance(r, dict)
                ]
            else:
                results = await tavily_service.search(query, max_results=5)

            duration = (time.time() - start) * 1000
            trace[-1] = TraceEntry(
                node="web_searcher",
                status="completed",
                started_at=start,
                completed_at=time.time(),
                duration_ms=round(duration, 1),
                input_summary=f'Query: "{query[:80]}" via={via}',
                output_summary=(
                    f"Got {len(results)} web results"
                    + (
                        f' | top: "{results[0]["title"][:50]}"'
                        if results
                        else ""
                    )
                ),
            )
            node_span.update(
                output={
                    "result_count": len(results),
                    "top_titles": [r.get("title", "") if isinstance(r, dict) else r["title"] for r in results[:3]],
                    "scores": [round(r.get("score", 0) if isinstance(r, dict) else r["score"], 3) for r in results[:3]],
                },
                metadata={"duration_ms": round(duration, 1)},
            )

            return {
                "web_results": results,
                "execution_trace": trace,
                "errors": errors,
            }

        except Exception as e:
            logger.error("web_searcher failed: %s", e)
            errors.append(f"web_searcher: {e}")
            trace[-1] = TraceEntry(
                node="web_searcher",
                status="failed",
                started_at=start,
                completed_at=time.time(),
                input_summary=f'Query: "{query[:80]}"',
                output_summary="",
                error=str(e),
            )
            node_span.update(level="ERROR", status_message=str(e))
            return {
                "web_results": [],
                "execution_trace": trace,
                "errors": errors,
            }
