"""
agents/router.py — NODE 2: Router

ROLE:
  Reads the validated query and decides WHICH retrieval path to take.

ROUTES:
  "rag"          → retrieve from Qdrant vector store (uploaded documents)
  "web_search"   → search the live web via Tavily
  "needs_human"  → pause and ask the human for clarification

WHEN TO USE EACH ROUTE:
  rag           — query is clearly about uploaded documents (resumes, CVs,
                  internal docs, specific files mentioned in the system)
  web_search    — query needs live/current info (news, stock prices, recent
                  events) or broad factual knowledge not in our docs
  needs_human   — query is genuinely ambiguous: could go either way and the
                  wrong answer would mislead the user

WHY A SEPARATE ROUTING NODE?
  Mixing routing logic into the retriever makes it hard to test.
  A dedicated router is easy to prompt-engineer and debug independently.

OUTPUT KEYS:
  route           — "rag" | "web_search" | "needs_human"
  route_reasoning — why this route was chosen
  human_question  — what to ask the human (only for needs_human)
  execution_trace — appended TraceEntry
"""

import logging
import re
import time
from pydantic import BaseModel, Field

from app.graph.state import AgentState, TraceEntry
from app.services import llm_service
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


# ─── Heuristics (fast, no LLM needed) ───────────────────────────────────────

_DOC_HINTS = re.compile(
    r"\b(resume|cv|curriculum.vitae|document|pdf|uploaded|experience|skill|"
    r"qualification|role|job|position|candidate|applicant|bhanu|teja|employee|"
    r"profile|summary|achievement|education|degree|certification)\b",
    re.IGNORECASE,
)

_WEB_HINTS = re.compile(
    r"\b(latest|current|today|news|stock|price|weather|recent|2024|2025|2026|"
    r"live|real.time|trending|who won|score|update)\b",
    re.IGNORECASE,
)

_GREET_ONLY = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay|great|nice|"
    r"good morning|good afternoon|good evening)[\s!.?]*$",
    re.IGNORECASE,
)

# Explicit "docs OR web?" questions → always ask the human
_FORCE_HUMAN = re.compile(
    r"\b((documents?|docs|resume|uploaded).{0,40}(or|vs).{0,20}(internet|web|online)|"
    r"(internet|web|online).{0,40}(or|vs).{0,20}(documents?|docs|resume|uploaded)|"
    r"should i (use|search|look).{0,30}(documents?|internet|web))\b",
    re.IGNORECASE,
)


def _heuristic_route(query: str) -> tuple[str, str] | tuple[None, str]:
    """
    Fast pre-filter.
    Returns (route, human_question) or (None, "") to fall through to LLM.
    """
    if _FORCE_HUMAN.search(query):
        return (
            "needs_human",
            "Should I search your uploaded documents, or look this up on the live web?",
        )
    if _GREET_ONLY.match(query.strip()):
        return "web_search", ""
    if _DOC_HINTS.search(query) and not _WEB_HINTS.search(query):
        return "rag", ""
    if _WEB_HINTS.search(query) and not _DOC_HINTS.search(query):
        return "web_search", ""
    return None, ""


# ─── Structured output schema ────────────────────────────────────────────────

class RouteSchema(BaseModel):
    route: str = Field(
        description='One of: "rag", "web_search", "needs_human"'
    )
    reasoning: str = Field(
        description="One or two sentences explaining why you chose this route"
    )
    human_question: str = Field(
        default="",
        description="If route=needs_human: the exact clarifying question to ask. Empty otherwise.",
    )


SYSTEM_PROMPT = """\
You are a routing agent for an AI assistant. Classify the query into exactly one route:

"rag"         — The query asks about specific documents, resumes, CVs, uploaded files,
                or internal company knowledge. Use this when document-specific facts are needed.

"web_search"  — The query requires live/recent information, broad factual knowledge,
                or is a general knowledge question not tied to specific uploaded documents.

"needs_human" — The query is genuinely ambiguous between rag and web_search AND choosing
                the wrong one would give a misleading answer. Use this SPARINGLY.
                If in doubt, pick rag or web_search — don't make the human wait.

Rules:
- Never invent a fourth route
- If query mentions a person's name in context of a resume/CV → rag
- If query asks about live data (prices, news, scores) → web_search
- needs_human is a LAST RESORT — only for truly ambiguous 50/50 cases
"""


# ─── Node function ───────────────────────────────────────────────────────────

async def router_node(state: AgentState) -> dict:
    start = time.time()
    query = state.get("validated_query") or state["original_query"]
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    trace.append(
        TraceEntry(
            node="router",
            status="running",
            started_at=start,
            input_summary=f'Query: "{query[:80]}"',
            output_summary="",
        )
    )

    # Fast heuristic check first
    heuristic, human_q = _heuristic_route(query)
    if heuristic:
        logger.info("router: heuristic → %s", heuristic)
        duration = (time.time() - start) * 1000
        trace[-1] = TraceEntry(
            node="router",
            status="completed",
            started_at=start,
            completed_at=time.time(),
            duration_ms=round(duration, 1),
            input_summary=f'Query: "{query[:80]}"',
            output_summary=f"route={heuristic} (heuristic)",
        )
        # ── LANGFUSE: log heuristic decision as an event (no LLM call) ───────
        tracer.event(
            trace_id,
            name="routing_decision",
            input={"query": query},
            output={"route": heuristic, "method": "heuristic"},
            metadata={"duration_ms": round(duration, 1)},
        )
        return {
            "route": heuristic,
            "route_reasoning": f"Heuristic pattern matched '{heuristic}'",
            "human_question": human_q,
            "execution_trace": trace,
            "errors": errors,
        }

    # Fall through to LLM routing
    # ── LANGFUSE: span wraps the LLM routing call ─────────────────────────────
    with tracer.span(
        trace_id,
        name="routing",
        input={"query": query, "method": "llm"},
        metadata={"node": "router"},
    ) as node_span:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Route this query: {query}"},
            ]
            with tracer.generation(
                trace_id,
                name="router_llm_call",
                model=state.get("model_name", "gpt-4o"),
                input=messages,
                parent_span_id=node_span.id,
            ) as gen_ctx:
                result, in_tok, out_tok = await llm_service.structured_chat_with_usage(
                    messages=messages,
                    response_format=RouteSchema,
                )
                tracer.update_generation_tokens(gen_ctx, in_tok, out_tok)
                gen_ctx.update(output=result.model_dump())

            # Safety: only allow valid routes
            valid_routes = {"rag", "web_search", "needs_human"}
            route = result.route if result.route in valid_routes else "rag"

            duration = (time.time() - start) * 1000
            trace[-1] = TraceEntry(
                node="router",
                status="completed",
                started_at=start,
                completed_at=time.time(),
                duration_ms=round(duration, 1),
                input_summary=f'Query: "{query[:80]}"',
                output_summary=f"route={route} | {result.reasoning[:60]}",
            )

            node_span.update(
                output={"route": route, "reasoning": result.reasoning},
                metadata={"duration_ms": round(duration, 1)},
            )
            logger.info("router: LLM → route=%s", route)

            return {
                "route": route,
                "route_reasoning": result.reasoning,
                "human_question": result.human_question,
                "execution_trace": trace,
                "errors": errors,
            }

        except Exception as e:
            logger.error("router failed: %s", e)
            errors.append(f"router: {e}")
            trace[-1] = TraceEntry(
                node="router",
                status="failed",
                started_at=start,
                completed_at=time.time(),
                input_summary=f'Query: "{query[:80]}"',
                output_summary="",
                error=str(e),
            )
            node_span.update(level="ERROR", status_message=str(e))
            return {
                "route": "rag",  # safe default
                "route_reasoning": "Fallback to rag due to routing error",
                "human_question": "",
                "execution_trace": trace,
                "errors": errors,
            }
