"""
app/observability/langfuse_client.py  —  Langfuse SDK wrapper

─────────────────────────────────────────────────────────────────────────────
WHAT IS LANGFUSE?
  Langfuse is an open-source LLM observability platform. Every time your
  application handles a user request, Langfuse records a "Trace" — a tree
  of nested observations (spans, generations, events) that shows exactly
  what happened, in what order, how long each step took, and how much it cost.

HIERARCHY (Langfuse concepts):
  ┌─────────────────────────────────────────────────────────┐
  │  Session  (one user's browser/chat session)             │
  │  └── Trace  (one user request end-to-end)               │
  │        ├── Span        (any unit of work: node, tool)   │
  │        │     └── Span  (nested child work)              │
  │        ├── Generation  (any LLM API call)               │
  │        └── Event       (a discrete point-in-time event) │
  └─────────────────────────────────────────────────────────┘

DEFINITIONS:
  Trace      — The top-level container for one complete request.
               Attached to a user_id, session_id, and metadata.
               Example: "User asked: What is RAG?"

  Span       — A unit of work with a start and end time (duration).
               Used for: LangGraph nodes, RAG retrieval, tool calls.
               Example: "rag_retrieval took 350ms"

  Generation — A special Span specifically for LLM API calls.
               Tracks: model, input tokens, output tokens, cost, latency.
               Example: "gpt-4o returned 320 tokens in 1.7s at $0.011"

  Event      — A point-in-time occurrence with no duration.
               Used for: human-in-the-loop interruptions, errors, decisions.
               Example: "Human chose: search uploaded documents"

  Session    — Groups multiple Traces from the same user interaction.
               Example: All requests within one browser tab session.

  User       — Groups all Traces/Sessions from one user.
               Example: Everything user_123 has ever done.

  Observation — The generic term for Span + Generation + Event (they all
                inherit from Observation in Langfuse's data model).

TRACE HIERARCHY FOR THIS APPLICATION:
  Trace: request_<uuid>  [user=user_123, session=session_9]
  ├── Span: query_validation  (30ms)
  │     metadata: {is_valid, sanitized_query}
  ├── Span: routing  (20ms)
  │     metadata: {route="rag", reasoning}
  ├── Span: rag_retrieval  (350ms)
  │     ├── Generation: embedding  (70ms, 8 tokens, $0.0001)
  │     ├── Span: vector_search  (120ms, top_k=5, scores=[0.82, 0.79])
  │     └── Span: reranking  (160ms, 3 chunks selected)
  ├── Span: response_generation  (1800ms)
  │     └── Generation: llm_call  (1700ms, gpt-4o, 1420→320 tokens, $0.011)
  └── Span: quality_check  (400ms)
        └── Generation: llm_evaluation  (380ms)
        Score: quality_score=0.92

HOW TO USE IN A NODE:
  from app.observability.langfuse_client import get_tracer

  tracer = get_tracer()

  def my_node(state: AgentState) -> dict:
      trace_id = state.get("langfuse_trace_id")
      with tracer.span(trace_id, "my_node", input={"query": ...}) as span:
          result = do_work()
          span.update(output={"result": result})
      return {"result": result}
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ── Lazy import so the app still starts when langfuse is not installed ─────────
try:
    from langfuse import Langfuse
    from langfuse.model import ModelUsage
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning(
        "langfuse package not installed. Run: pip install langfuse>=2.0  "
        "All tracing calls will be no-ops."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SpanContext — thin wrapper so node code doesn't import langfuse directly
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpanContext:
    """
    Returned by LangfuseTracer.span() context manager.
    Node code calls span.update(...) to add outputs and metadata.

    WHY WRAP?
      If Langfuse is unavailable or disabled, this is a no-op object.
      Node code never crashes — tracing simply does nothing.
    """
    _span: Any = field(default=None, repr=False)     # the real langfuse span/generation
    _tracer: Any = field(default=None, repr=False)   # back-reference to LangfuseTracer
    trace_id: str = ""
    name: str = ""
    start_time: float = field(default_factory=time.time)

    def update(
        self,
        output: Any = None,
        metadata: Optional[dict] = None,
        level: str = "DEFAULT",        # "DEFAULT" | "DEBUG" | "WARNING" | "ERROR"
        status_message: Optional[str] = None,
    ) -> None:
        """Update the span with output data. Call this at the end of node work."""
        if self._span is None:
            return
        try:
            kwargs: dict = {}
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if status_message is not None:
                kwargs["status_message"] = status_message
            kwargs["level"] = level
            self._span.update(**kwargs)
        except Exception as exc:
            logger.debug("Langfuse span.update failed (non-critical): %s", exc)

    def end(self, end_time: Optional[float] = None) -> None:
        """Explicitly close the span (context manager calls this automatically)."""
        if self._span is None:
            return
        try:
            self._span.end(end_time=end_time)
        except Exception as exc:
            logger.debug("Langfuse span.end failed (non-critical): %s", exc)

    @property
    def id(self) -> Optional[str]:
        """Return the underlying Langfuse observation ID (for nesting children)."""
        if self._span is None:
            return None
        try:
            return str(self._span.id)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# LangfuseTracer — the main class used throughout the application
# ─────────────────────────────────────────────────────────────────────────────

class LangfuseTracer:
    """
    Singleton wrapper around the Langfuse SDK.

    All LangGraph nodes import this via `get_tracer()` and call:
      tracer.span(...)        → create/attach a span
      tracer.generation(...)  → create/attach an LLM generation
      tracer.event(...)       → log a discrete event
      tracer.score(...)       → score a trace (quality, relevance, etc.)
      tracer.flush()          → force-send buffered data before request ends

    Internally, active traces are stored in a dict keyed by trace_id so that
    any node can attach to the correct trace without passing objects around.
    """

    def __init__(self) -> None:
        self._client: Optional[Any] = None       # Langfuse SDK client
        self._traces: dict[str, Any] = {}        # trace_id → Langfuse trace object
        self._enabled: bool = False
        self._initialized: bool = False

    def init(self, public_key: str, secret_key: str, host: str, enabled: bool = True) -> None:
        """
        Call once at application startup (in main.py lifespan).

        WHY LAZY?
          Langfuse SDK connects to the server only when the first trace is
          created. This init call just stores credentials.
        """
        self._enabled = enabled and _LANGFUSE_AVAILABLE

        if not self._enabled:
            logger.info("Langfuse tracing disabled (enabled=%s, available=%s).", enabled, _LANGFUSE_AVAILABLE)
            self._initialized = True
            return

        if not public_key or public_key.startswith("pk-lf-REPLACE"):
            logger.warning(
                "Langfuse PUBLIC_KEY not configured. "
                "Set LANGFUSE_PUBLIC_KEY in .env after creating a project at http://localhost:3000"
            )
            self._enabled = False
            self._initialized = True
            return

        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=False,               # set True to see SDK logs
            )
            self._initialized = True
            logger.info("Langfuse client initialised → %s", host)
        except Exception as exc:
            logger.error("Failed to initialise Langfuse: %s", exc)
            self._enabled = False
            self._initialized = True

    # ── Trace management ──────────────────────────────────────────────────────

    def create_trace(
        self,
        request_id: str,
        user_id: str = "anonymous",
        session_id: str = "",
        conversation_id: str = "",
        query: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new top-level Trace for one user request.

        RETURNS the trace_id (same as request_id) so it can be stored in
        AgentState["langfuse_trace_id"] and threaded through every node.

        WHAT IS A TRACE?
          Think of it as the "folder" for everything that happened during
          one request. All spans, generations, and events go inside this folder.
        """
        if not self._enabled or self._client is None:
            return request_id

        try:
            trace = self._client.trace(
                id=request_id,                          # unique ID for this request
                name="agent_request",                   # human-readable name in UI
                user_id=user_id,                        # for per-user analysis
                session_id=session_id or request_id,    # groups related traces
                input={"query": query},                 # what the user asked
                tags=tags or ["langgraph", "production"],
                metadata={
                    "conversation_id": conversation_id,
                    "app": "langgraph-agent",
                    **(metadata or {}),
                },
            )
            self._traces[request_id] = trace
            logger.debug("Langfuse trace created: %s (user=%s)", request_id, user_id)
        except Exception as exc:
            logger.debug("Langfuse create_trace failed (non-critical): %s", exc)

        return request_id

    def finalize_trace(
        self,
        trace_id: str,
        output: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the top-level trace with the final output.
        Call this once the LangGraph workflow finishes (or fails).
        """
        if not self._enabled or trace_id not in self._traces:
            return
        try:
            trace = self._traces[trace_id]
            trace.update(
                output=output,
                metadata={
                    "success": success,
                    "error": error,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse finalize_trace failed (non-critical): %s", exc)

    # ── Span (any unit of work) ───────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        trace_id: str,
        name: str,
        input: Any = None,
        metadata: Optional[dict] = None,
        parent_span_id: Optional[str] = None,
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager that creates a Langfuse Span.

        USAGE IN A NODE:
          with tracer.span(trace_id, "rag_retrieval", input={"query": q}) as sp:
              docs = retrieve(q)
              sp.update(output={"chunk_count": len(docs)})

        WHY CONTEXT MANAGER?
          Automatically records start_time and end_time (duration).
          Even if the node throws an exception, the span is still closed.
        """
        ctx = SpanContext(trace_id=trace_id, name=name, start_time=time.time())

        if not self._enabled or trace_id not in self._traces:
            yield ctx
            return

        try:
            trace = self._traces[trace_id]
            span_kwargs: dict = {
                "name": name,
                "input": input,
                "metadata": metadata or {},
            }
            if parent_span_id:
                span_kwargs["parent_observation_id"] = parent_span_id

            lf_span = trace.span(**span_kwargs)
            ctx._span = lf_span
            ctx._tracer = self
        except Exception as exc:
            logger.debug("Langfuse span create failed (non-critical): %s", exc)

        try:
            yield ctx
        except Exception as node_exc:
            # Node threw — record the error on the span, then re-raise
            try:
                if ctx._span:
                    ctx._span.update(
                        level="ERROR",
                        status_message=str(node_exc),
                    )
            except Exception:
                pass
            raise
        finally:
            try:
                if ctx._span:
                    ctx._span.end()
            except Exception as exc:
                logger.debug("Langfuse span end failed (non-critical): %s", exc)

    # ── Generation (LLM API call) ─────────────────────────────────────────────

    @contextmanager
    def generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: Any = None,
        metadata: Optional[dict] = None,
        parent_span_id: Optional[str] = None,
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager for LLM API calls. Like span() but also tracks:
          - model name and provider
          - input / output tokens
          - cost (auto-calculated by Langfuse for known models)
          - latency

        USAGE:
          with tracer.generation(trace_id, "llm_call", model="gpt-4o",
                                 input=prompt, parent_span_id=span.id) as gen:
              response = llm.chat(prompt)
              gen.update(
                  output=response.content,
                  metadata={"input_tokens": 1420, "output_tokens": 320}
              )

        WHY SEPARATE FROM span()?
          Langfuse renders Generations differently in the UI. They show
          model name, token counts, and cost in a dedicated table.
        """
        ctx = SpanContext(trace_id=trace_id, name=name, start_time=time.time())

        if not self._enabled or trace_id not in self._traces:
            yield ctx
            return

        try:
            trace = self._traces[trace_id]
            gen_kwargs: dict = {
                "name": name,
                "model": model,
                "input": input,
                "metadata": metadata or {},
            }
            if parent_span_id:
                gen_kwargs["parent_observation_id"] = parent_span_id

            lf_gen = trace.generation(**gen_kwargs)
            ctx._span = lf_gen
            ctx._tracer = self
        except Exception as exc:
            logger.debug("Langfuse generation create failed (non-critical): %s", exc)

        try:
            yield ctx
        except Exception as node_exc:
            try:
                if ctx._span:
                    ctx._span.update(level="ERROR", status_message=str(node_exc))
            except Exception:
                pass
            raise
        finally:
            try:
                if ctx._span:
                    ctx._span.end()
            except Exception as exc:
                logger.debug("Langfuse generation end failed (non-critical): %s", exc)

    def update_generation_tokens(
        self,
        span_ctx: SpanContext,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o",
    ) -> None:
        """
        Record token usage on a Generation span after the LLM call completes.

        COST CALCULATION:
          Langfuse automatically calculates cost if the model name is a
          known OpenAI model (gpt-4o, gpt-3.5-turbo, etc.).
          For custom models, set usage.input and usage.output manually.
        """
        if span_ctx._span is None:
            return
        try:
            span_ctx._span.update(
                usage={
                    "input": input_tokens,       # prompt tokens
                    "output": output_tokens,     # completion tokens
                    "total": input_tokens + output_tokens,
                    "unit": "TOKENS",
                },
            )
        except Exception as exc:
            logger.debug("Langfuse update_generation_tokens failed: %s", exc)

    # ── Event (point-in-time) ─────────────────────────────────────────────────

    def event(
        self,
        trace_id: str,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict] = None,
        level: str = "DEFAULT",
        parent_span_id: Optional[str] = None,
    ) -> None:
        """
        Log a discrete point-in-time event (no duration).

        USAGE:
          tracer.event(trace_id, "human_interrupt",
                       input={"question": "Use docs or web?"},
                       metadata={"route": "needs_human"})

        WHEN TO USE EVENT vs SPAN?
          Span  → work that takes time (node execution, API call)
          Event → something that happened instantaneously (decision, error, user input)
        """
        if not self._enabled or trace_id not in self._traces:
            return
        try:
            trace = self._traces[trace_id]
            ev_kwargs: dict = {
                "name": name,
                "input": input,
                "output": output,
                "level": level,
                "metadata": metadata or {},
            }
            if parent_span_id:
                ev_kwargs["parent_observation_id"] = parent_span_id
            trace.event(**ev_kwargs)
        except Exception as exc:
            logger.debug("Langfuse event failed (non-critical): %s", exc)

    # ── Scores ────────────────────────────────────────────────────────────────

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",    # "NUMERIC" | "BOOLEAN" | "CATEGORICAL"
    ) -> None:
        """
        Attach a numeric score to a trace.

        USAGE:
          tracer.score(trace_id, "quality_score", 0.92,
                       comment="Grounded and complete")
          tracer.score(trace_id, "user_feedback", 1.0,
                       data_type="BOOLEAN", comment="thumbs up")

        WHY SCORES?
          Scores let you filter and sort traces in the Langfuse dashboard.
          E.g. "show me all traces where quality_score < 0.5"
          You can also define evaluation pipelines that auto-score traces.
        """
        if not self._enabled or self._client is None:
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type=data_type,
            )
        except Exception as exc:
            logger.debug("Langfuse score failed (non-critical): %s", exc)

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """
        Force-send all buffered events to the Langfuse server.

        WHY FLUSH?
          The Langfuse SDK batches events for performance. In a long-running
          FastAPI server, events are eventually auto-sent. But for short-lived
          requests (or tests), you must call flush() to ensure data is not lost.

        WHEN TO CALL?
          At the end of every /api/chat request, after the final SSE event.
        """
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush failed (non-critical): %s", exc)

    def cleanup_trace(self, trace_id: str) -> None:
        """Remove trace from local dict after request is done (memory management)."""
        self._traces.pop(trace_id, None)

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────

_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """
    Return the global LangfuseTracer instance.

    PATTERN:
      The tracer is created once at module load time and initialised once
      at startup (main.py lifespan). All nodes call get_tracer() to get
      the same singleton — no need to pass it through function arguments.

    USAGE:
      from app.observability.langfuse_client import get_tracer
      tracer = get_tracer()
    """
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


def init_tracer(public_key: str, secret_key: str, host: str, enabled: bool = True) -> LangfuseTracer:
    """
    Initialise the global tracer. Call this once in main.py @asynccontextmanager lifespan.

    EXAMPLE (main.py):
      from app.observability.langfuse_client import init_tracer, get_tracer

      @asynccontextmanager
      async def lifespan(app: FastAPI):
          init_tracer(
              public_key=settings.langfuse_public_key,
              secret_key=settings.langfuse_secret_key,
              host=settings.langfuse_host,
              enabled=settings.langfuse_enabled,
          )
          yield
          get_tracer().flush()  # flush on shutdown
    """
    tracer = get_tracer()
    tracer.init(public_key, secret_key, host, enabled)
    return tracer
