"""
app/main.py — FastAPI application with SSE streaming endpoints.

ENDPOINTS:

  POST /api/chat
    Body: {"query": "...", "thread_id": "optional"}
    Returns: text/event-stream  (Server-Sent Events)
    Events:
      node_update  — a node just completed, here is its state delta
      interrupt    — graph paused at human_gate, waiting for input
      final        — graph finished, here is the complete state
      error        — something went wrong
      done         — stream is closing

  POST /api/chat/resume
    Body: {"thread_id": "...", "human_input": "..."}
    Returns: text/event-stream  (same event format as /api/chat)

  GET /api/thread/{thread_id}
    Returns: current snapshot of graph state for this thread

  GET /api/graph/schema
    Returns: node + edge list for UI visualisation

  GET /api/health
    Returns: health status of all services

  GET /
    Returns: static/index.html (the UI)

HOW SSE STREAMING WORKS:
  Server-Sent Events is a one-way push channel from server to browser.
  The client opens a long-lived HTTP connection; the server sends events
  in the format: "data: {...json...}\\n\\n"
  The browser's EventSource API parses these automatically.

  We use async generators: each `yield` sends one SSE event.
  StreamingResponse wraps the generator and keeps the connection open.

HOW INTERRUPT/RESUME WORKS:
  1. /api/chat starts graph.astream()
  2. When human_gate runs interrupt(), LangGraph yields a special event:
       {"__interrupt__": (Interrupt(value={...}),)}
  3. We detect this, send it as an SSE "interrupt" event, and stop streaming
  4. Client shows the human gate panel
  5. Client POST /api/chat/resume → we call graph.astream(Command(resume=...), config)
  6. Graph resumes from exactly where interrupt() was called
  7. Streaming continues normally
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from app.config import settings
from app.graph.state import initial_state
from app.graph.workflow import get_graph, get_graph_schema

# Prometheus metrics (optional — gracefully skipped if not installed)
try:
    from prometheus_fastapi_instrumentator import Instrumentator as PrometheusInstrumentator
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
from app.services import qdrant_service
from app.services import mcp_client

# ── Observability: Langfuse ────────────────────────────────────────────────────
from app.observability.langfuse_client import init_tracer, get_tracer

# ── Database + Memory ─────────────────────────────────────────────────────────
from app.database.db_client import init_db, get_db
from app.memory import conversation_manager as conv_mgr
from app.memory import context_manager as ctx_mgr
from app.memory import memory_retrieval as mem_ret

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Initialize Langfuse and PostgreSQL connection pool here so they are
    available for every request without re-connecting every time.
    """
    # 1. Initialize Langfuse tracing
    init_tracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        enabled=settings.langfuse_enabled,
    )

    # 2. Initialize PostgreSQL connection pool for conversation history
    init_db(dsn=settings.conversation_db_dsn, min_conn=2, max_conn=10)

    # 3. Warm up the LangGraph (compile it)
    get_graph()
    logger.info("LangGraph compiled and ready.")

    yield  # ← application runs here

    # Shutdown: flush any remaining Langfuse events
    get_tracer().flush()
    get_db().close()
    logger.info("Langfuse flushed, DB pool closed.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LangGraph Agentic RAG",
    description="7-node LangGraph workflow with RAG, web search, HITL, Langfuse, and conversation memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tightened from ["*"] to explicit origins.
    # ["*"] allows any website to call this API (CSRF risk in production).
    allow_origins=[
        "http://localhost:4200",   # Angular dev
        "http://localhost:7860",   # self
        "http://localhost:3000",   # Langfuse
        "http://localhost:80",     # Nginx
        "http://localhost",
    ],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Prometheus /metrics endpoint
if _PROMETHEUS_AVAILABLE:
    PrometheusInstrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled — GET /metrics")

# Serve static files (the UI)
import os
_STATIC = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_STATIC):
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Extended request model for multi-user conversation support.

    FIELDS:
      query           — the user's message
      thread_id       — LangGraph thread for interrupt/resume (auto-generated if empty)
      user_id         — who is sending (from auth, or use "anonymous" for demo)
      conversation_id — which conversation to continue (empty = create new)
      session_id      — browser session ID (empty = create new)

    EXAMPLE:
      {
        "query": "What are Neha's skills?",
        "user_id": "user_demo_1",
        "conversation_id": "optional-existing-conv-uuid",
        "session_id": "optional-existing-session-uuid"
      }
    """
    query: str
    thread_id: str = ""
    user_id: str = "user_demo_1"         # default to demo user
    conversation_id: str = ""            # empty = auto-create
    session_id: str = ""                 # empty = auto-create


class ResumeRequest(BaseModel):
    thread_id: str
    human_input: str
    user_id: str = "user_demo_1"
    conversation_id: str = ""
    session_id: str = ""


class McpCallRequest(BaseModel):
    """Call one MCP tool by name (hands-on testing)."""
    name: str
    arguments: dict = {}


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse(event_type: str, data: Any) -> str:
    """Format one Server-Sent Event frame."""
    payload = json.dumps({"type": event_type, **data} if isinstance(data, dict) else {"type": event_type, "data": data})
    return f"data: {payload}\n\n"


def _sanitise(obj: Any) -> Any:
    """Make an object JSON-serialisable (strip non-serialisable values)."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ── Streaming generator ────────────────────────────────────────────────────────

async def _stream_graph(
    input_or_command: Any,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream graph execution as SSE events.

    Works for both initial invocations and Command(resume=...) invocations.

    WHY sync graph.stream() instead of astream()?
      LangGraph 1.2.x: interrupt() needs the sync runnable config context.
      graph.astream() + interrupt() raises:
        "Called get_config outside of a runnable context"
      We run sync .stream() in a worker thread and forward events over a queue
      so FastAPI can still SSE-stream node updates live.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    yield sse("started", {"thread_id": thread_id, "timestamp": time.time()})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _worker() -> None:
        try:
            for event in graph.stream(
                input_or_command,
                config=config,
                stream_mode="updates",
            ):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put({"__error__": e}), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    worker_fut = loop.run_in_executor(None, _worker)
    interrupted = False

    try:
        while True:
            event = await queue.get()
            if event is _DONE:
                break

            if isinstance(event, dict) and "__error__" in event:
                err = event["__error__"]
                logger.error("Graph stream error: %s", err, exc_info=True)
                yield sse("error", {"message": str(err), "thread_id": thread_id})
                return

            # ── Interrupt detected ──────────────────────────────────────────
            if isinstance(event, dict) and "__interrupt__" in event:
                interrupts = event["__interrupt__"]
                interrupt_value = {}
                if interrupts:
                    first = interrupts[0]
                    interrupt_value = getattr(first, "value", first)
                logger.info("Graph interrupted: thread_id=%s", thread_id)
                yield sse("interrupt", {
                    "thread_id": thread_id,
                    "interrupt_value": _sanitise(interrupt_value),
                    "timestamp": time.time(),
                })
                interrupted = True
                return  # Stream ends here; client must call /resume

            # ── Normal node update ─────────────────────────────────────────
            if not isinstance(event, dict):
                continue
            for node_name, updates in event.items():
                if node_name.startswith("_"):
                    continue
                if not isinstance(updates, dict):
                    continue

                trace = updates.get("execution_trace", [])
                last_trace = trace[-1] if trace else {}

                yield sse("node_update", {
                    "node": node_name,
                    "timestamp": time.time(),
                    "trace": _sanitise(last_trace),
                    "updates": _sanitise({
                        k: v for k, v in updates.items()
                        if k not in ("execution_trace", "errors")
                        and v
                    }),
                })

    except Exception as e:
        logger.error("Graph stream error: %s", e, exc_info=True)
        yield sse("error", {"message": str(e), "thread_id": thread_id})
        return
    finally:
        await worker_fut

    if interrupted:
        return

    # ── Fetch final state from checkpointer ──────────────────────────────────
    try:
        snapshot = await asyncio.to_thread(graph.get_state, config)
        if snapshot and snapshot.values:
            vals = snapshot.values
            yield sse("final", {
                "thread_id": thread_id,
                "answer": vals.get("final_answer") or vals.get("generated_response", ""),
                "sources": _sanitise(vals.get("sources", [])),
                "route": vals.get("route", ""),
                "quality_score": (
                    vals.get("quality_result", {}).get("score")
                    if isinstance(vals.get("quality_result"), dict)
                    else None
                ),
                "retry_count": vals.get("retry_count", 0),
                "is_complete": vals.get("is_complete", False),
                "errors": vals.get("errors", []),
                "execution_trace": _sanitise(vals.get("execution_trace", [])),
                "validation": _sanitise(vals.get("validation_result")),
            })
    except Exception as e:
        logger.error("Failed to fetch final state: %s", e)

    yield sse("done", {"thread_id": thread_id})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the UI."""
    index = os.path.join(_STATIC, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "LangGraph Agentic RAG API", "docs": "/docs"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Start a new conversation thread with full Langfuse tracing + conversation history.

    COMPLETE REQUEST FLOW:
      1. Get/create user record (PostgreSQL)
      2. Get/create conversation (PostgreSQL)
      3. Create/get session (PostgreSQL)
      4. Save user message (PostgreSQL)
      5. Load recent messages (sliding window)
      6. Retrieve long-term memory (pgvector semantic search)
      7. Build system prompt with history + memory
      8. Create Langfuse trace (observability)
      9. Run LangGraph workflow (7 nodes, all traced to Langfuse)
     10. Save assistant response (PostgreSQL)
     11. Flush Langfuse events
     12. Return SSE stream to client
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    tracer = get_tracer()

    logger.info(
        "New chat: request_id=%s user=%s query=%s",
        request_id, request.user_id, request.query[:60]
    )

    # ── STEP 1-3: User / Conversation / Session management ───────────────────
    try:
        user = conv_mgr.get_or_create_user(
            external_id=request.user_id,
            name=request.user_id.replace("_", " ").title(),
        )
        user_uuid = str(user.get("id", request.user_id))

        # Get or create conversation
        if request.conversation_id:
            conv = conv_mgr.get_conversation(request.conversation_id) or {}
        else:
            # Auto-title from first 60 chars of query
            title = request.query[:60].strip() + ("..." if len(request.query) > 60 else "")
            conv = conv_mgr.create_conversation(user_uuid, title=title)
        conv_id = str(conv.get("id", request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"))

        # Get or create session
        if request.session_id:
            session = {"id": request.session_id, "thread_id": thread_id}
        else:
            session = conv_mgr.create_session(
                conversation_id=conv_id,
                user_id=user_uuid,
                thread_id=thread_id,
            )
        session_id = str(session.get("id", f"sess_{uuid.uuid4().hex[:12]}"))

        # ── STEP 4: Save the user message immediately ─────────────────────────
        user_msg = conv_mgr.save_message(
            conversation_id=conv_id,
            session_id=session_id,
            user_id=user_uuid,
            role="user",
            content=request.query,
            langfuse_trace_id=request_id,
        )

        # ── STEP 5: Load conversation history (sliding window strategy) ───────
        all_messages = conv_mgr.get_recent_messages(conv_id, limit=50)
        context_result = ctx_mgr.build_context(
            messages=all_messages[:-1],  # exclude the message we just saved
            query=request.query,
            conversation_id=conv_id,
            strategy="hybrid",
            max_tokens=20_000,
            keep_recent_n=15,
        )

        # ── STEP 6: Retrieve long-term memory ─────────────────────────────────
        memories = mem_ret.retrieve_relevant_memory(
            user_id=user_uuid,
            query=request.query,
            top_k=5,
        )

        # ── STEP 7: Build context-aware system prompt ─────────────────────────
        base_system_prompt = (
            "You are an expert AI assistant. "
            "Answer questions about uploaded documents or provide general knowledge. "
            "Always cite your sources."
        )
        system_prompt = ctx_mgr.build_system_prompt_with_context(
            base_system_prompt=base_system_prompt,
            conversation_summary=context_result.get("summary", ""),
            long_term_memories=memories,
        )

    except Exception as db_exc:
        logger.warning("DB/memory setup failed (continuing without history): %s", db_exc)
        user_uuid = request.user_id
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        system_prompt = "You are an expert AI assistant."

    # ── STEP 8: Create Langfuse trace ─────────────────────────────────────────
    tracer.create_trace(
        request_id=request_id,
        user_id=user_uuid,
        session_id=session_id,
        conversation_id=conv_id,
        query=request.query,
        tags=["langgraph", "production", f"route-tbd"],
        metadata={
            "thread_id": thread_id,
            "has_history": bool(context_result.get("messages")) if "context_result" in dir() else False,
            "memory_count": len(memories) if "memories" in dir() else 0,
            "context_strategy": "hybrid",
        },
    )

    # ── STEP 9: Build initial state for LangGraph ─────────────────────────────
    state = initial_state(
        query=request.query,
        thread_id=thread_id,
        max_retries=settings.max_quality_retries,
        user_id=user_uuid,
        conversation_id=conv_id,
        session_id=session_id,
        langfuse_trace_id=request_id,
        conversation_context=system_prompt,
        model_name=settings.openai_chat_model,
    )

    async def _stream_with_save():
        """
        Wrap the graph stream to save the response and flush Langfuse after completion.
        """
        final_answer = ""
        try:
            async for chunk in _stream_graph(state, thread_id):
                yield chunk
                # Extract final answer from the SSE event for saving
                if '"type": "final"' in chunk or '"type":"final"' in chunk:
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        final_answer = data.get("answer", "")
                    except Exception:
                        pass
        finally:
            # ── STEP 10: Save assistant response ─────────────────────────────
            if final_answer:
                try:
                    conv_mgr.save_message(
                        conversation_id=conv_id,
                        session_id=session_id,
                        user_id=user_uuid,
                        role="assistant",
                        content=final_answer,
                        model_used=settings.openai_chat_model,
                        langfuse_trace_id=request_id,
                    )
                    # ── Finalize Langfuse trace ───────────────────────────────
                    tracer.finalize_trace(
                        trace_id=request_id,
                        output={"answer": final_answer[:500]},
                        success=True,
                    )
                except Exception as save_exc:
                    logger.warning("Failed to save response to DB: %s", save_exc)
            # ── STEP 11: Flush Langfuse ───────────────────────────────────────
            tracer.flush()
            tracer.cleanup_trace(request_id)

    return StreamingResponse(
        _stream_with_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
            "X-Request-ID": request_id,
            "X-Conversation-ID": conv_id,
            "X-Session-ID": session_id,
        },
    )


@app.post("/api/chat/resume")
async def chat_resume(request: ResumeRequest):
    """
    Resume a paused conversation after human gate.

    The client provides the thread_id (from the interrupt event) and
    the human's input. The graph resumes from after the interrupt() call.
    """
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    if not request.human_input.strip():
        raise HTTPException(status_code=400, detail="human_input cannot be empty")

    logger.info(
        "Resuming: thread_id=%s human_input=%s",
        request.thread_id, request.human_input[:60],
    )

    # Command(resume=value) tells LangGraph to continue from the interrupt()
    # call and return `value` as the result of interrupt()
    command = Command(resume=request.human_input)

    return StreamingResponse(
        _stream_graph(command, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/thread/{thread_id}")
async def get_thread(thread_id: str):
    """
    Inspect current state for a thread (useful for debugging).
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise HTTPException(status_code=404, detail="Thread not found or no state yet")
        return {
            "thread_id": thread_id,
            "state": _sanitise(snapshot.values),
            "next": list(snapshot.next),
            "tasks": [str(t) for t in snapshot.tasks],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/schema")
async def graph_schema():
    """Return the graph topology for the UI visualisation."""
    return get_graph_schema()


@app.get("/api/health")
async def health():
    """Check all service connections."""
    results: dict = {"status": "ok", "services": {}}

    # Qdrant
    try:
        info = await qdrant_service.collection_info()
        results["services"]["qdrant"] = info
    except Exception as e:
        results["services"]["qdrant"] = {"error": str(e)}
        results["status"] = "degraded"

    # LLM (just check config)
    results["services"]["llm"] = {
        "model": settings.openai_chat_model,
        "base_url": settings.openai_api_base or "openai-cloud",
        "configured": bool(settings.openai_api_key),
    }

    # Tavily
    results["services"]["tavily"] = {
        "configured": bool(settings.tavily_api_key),
    }

    # MCP
    results["services"]["mcp"] = {
        "use_mcp_tools": settings.use_mcp_tools,
        "server_module": "mcp_server.server",
        "tools_endpoint": "/api/mcp/tools",
        "call_endpoint": "/api/mcp/call",
    }

    return results


@app.get("/api/mcp/tools")
async def mcp_list_tools():
    """
    Hands-on: list tools exposed by our MCP server (discovery).

    Under the hood this spawns `python -m mcp_server.server` over stdio,
    initializes an MCP session, and calls tools/list.
    """
    try:
        tools = await mcp_client.list_tools()
        return {
            "ok": True,
            "count": len(tools),
            "tools": tools,
            "hint": "Call a tool with POST /api/mcp/call {\"name\":\"echo_demo\",\"arguments\":{\"message\":\"hi\"}}",
        }
    except Exception as e:
        logger.error("MCP list_tools failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"MCP list_tools failed: {e}")


@app.post("/api/mcp/call")
async def mcp_call_tool(request: McpCallRequest):
    """
    Hands-on: call one MCP tool.

    Examples:
      {"name":"echo_demo","arguments":{"message":"hello MCP"}}
      {"name":"qdrant_info","arguments":{}}
      {"name":"search_documents","arguments":{"query":"Neha skills","top_k":3}}
      {"name":"web_search","arguments":{"query":"AI trends 2026","max_results":3}}
    """
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="tool name is required")
    try:
        result = await mcp_client.call_tool(request.name, request.arguments or {})
        return result
    except Exception as e:
        logger.error("MCP call_tool failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"MCP call_tool failed: {e}")


# ── Conversation History API ──────────────────────────────────────────────────

@app.get("/api/users/{user_id}/conversations")
async def list_conversations(user_id: str):
    """
    Return all conversations for a user (for sidebar display).

    EXAMPLE:
      GET /api/users/user_demo_1/conversations
    """
    try:
        user = conv_mgr.get_or_create_user(external_id=user_id)
        conversations = conv_mgr.list_user_conversations(str(user["id"]), limit=20)
        return {"user_id": user_id, "conversations": conversations}
    except Exception as e:
        logger.error("list_conversations failed: %s", e)
        return {"user_id": user_id, "conversations": [], "error": str(e)}


@app.get("/api/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, limit: int = 20):
    """
    Return recent messages for a conversation.

    EXAMPLE:
      GET /api/conversations/{uuid}/messages?limit=20
    """
    try:
        messages = conv_mgr.get_recent_messages(conversation_id, limit=limit)
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "messages": messages,
        }
    except Exception as e:
        logger.error("get_messages failed: %s", e)
        return {"conversation_id": conversation_id, "messages": [], "error": str(e)}


@app.post("/api/users/{user_id}/memory")
async def store_user_memory(user_id: str, body: dict):
    """
    Manually store a long-term memory for a user.

    EXAMPLE:
      POST /api/users/user_demo_1/memory
      Body: {"content": "User prefers Python examples", "memory_type": "preference"}
    """
    try:
        user = conv_mgr.get_or_create_user(external_id=user_id)
        from app.services import llm_service
        mem = mem_ret.store_memory(
            user_id=str(user["id"]),
            content=body.get("content", ""),
            memory_type=body.get("memory_type", "fact"),
            importance=float(body.get("importance", 0.5)),
            embed_fn=lambda t: asyncio.get_event_loop().run_until_complete(llm_service.embed(t)),
        )
        return {"ok": True, "memory": mem}
    except Exception as e:
        logger.error("store_memory failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/memory")
async def get_user_memory(user_id: str):
    """
    Return all long-term memories for a user.
    """
    try:
        user = conv_mgr.get_or_create_user(external_id=user_id)
        memories = mem_ret.get_user_memories_by_importance(str(user["id"]), limit=20)
        return {"user_id": user_id, "memories": memories}
    except Exception as e:
        logger.error("get_memory failed: %s", e)
        return {"user_id": user_id, "memories": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level="info",
    )
