"""
graph/workflow.py — Builds and compiles the LangGraph StateGraph.

CONCEPT:
  StateGraph is the heart of the application. It defines:
    - Which nodes exist (the agents)
    - How they connect (edges)
    - Which connections are conditional (routing decisions)
    - Where execution starts (entry point)
    - How state is persisted (checkpointer)

  Once compiled, the graph can be invoked like a function:
    result = await graph.ainvoke(initial_state, config)
  Or streamed:
    async for event in graph.astream(initial_state, config):
        ...

THE GRAPH TOPOLOGY:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  START → query_validator ──[invalid]──→ END     │
  │              │                                  │
  │           [valid]                               │
  │              ↓                                  │
  │           router ──────[rag]──────→ rag_retriever ─┐
  │              │                                  │  │
  │        [web_search]──→ web_searcher ────────────┤  │
  │              │                                  │  │
  │        [needs_human]──→ human_gate              │  │
  │                            │                    │  │
  │                       [rag/web] ────────────────┘  │
  │                                                     │
  │                        response_generator ←─────────┘
  │                              │              ↑
  │                        quality_checker      │
  │                         │         │         │
  │                       [PASS]   [REVISE]─────┘
  │                         │
  │                        END
  │                                                 │
  └─────────────────────────────────────────────────┘

CHECKPOINTER:
  MemorySaver stores the complete graph state in memory keyed by thread_id.
  This is what enables interrupt/resume — the state is preserved between
  HTTP requests so the graph can be paused and resumed later.

  For production, swap MemorySaver with SqliteSaver or PostgresSaver.
"""

import asyncio
import inspect
import logging
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ── Durable checkpointer (PostgreSQL) ────────────────────────────────────────
# WHY REPLACE MEMORYSAVER?
#   MemorySaver stores state in Python memory on a single process.
#   Problems:
#     1. Pod restart = ALL conversation state lost
#     2. Multiple K8s replicas cannot share in-memory state
#        (pod-1 starts a conversation, pod-2 gets the resume request → state not found)
#   Solution: PostgresSaver stores state in PostgreSQL
#     - Survives restarts
#     - Any replica can access any conversation's state
#     - Scales horizontally
#
# HOW IT WORKS:
#   LangGraph calls checkpointer.put() after every node execution.
#   Stores: {thread_id: {node_name: {state_dict}}}
#   On resume: checkpointer.get(thread_id) returns the saved state.
#
# INSTALL:
#   pip install langgraph-checkpoint-postgres
#
# USAGE:
#   Set USE_POSTGRES_CHECKPOINTER=true in .env to enable.
#   Falls back to MemorySaver if the library is not installed or DB is unavailable.
#
# SCHEMA:
#   LangGraph automatically creates these tables in your Postgres DB:
#     checkpoints      — stores full state snapshots
#     checkpoint_writes — stores individual write operations
#   Run after enabling:
#     from langgraph.checkpoint.postgres import PostgresSaver
#     PostgresSaver.create_tables(conn)
import os as _os
_USE_POSTGRES_CHECKPOINTER = _os.environ.get("USE_POSTGRES_CHECKPOINTER", "false").lower() == "true"

from app.graph.state import AgentState
from app.graph.edges import (
    route_after_validation,
    route_after_router,
    route_after_human,
    route_after_quality,
)
from app.agents.query_validator import query_validator_node
from app.agents.router import router_node
from app.agents.rag_retriever import rag_retriever_node
from app.agents.web_searcher import web_searcher_node
from app.agents.human_gate import human_gate_node
from app.agents.response_generator import response_generator_node
from app.agents.quality_checker import quality_checker_node

logger = logging.getLogger(__name__)

# ── Checkpointer selection ────────────────────────────────────────────────────
# Uses PostgresSaver when USE_POSTGRES_CHECKPOINTER=true (production).
# Falls back to MemorySaver for development (no extra setup needed).
def _create_checkpointer():
    """
    Create the appropriate checkpointer based on configuration.

    Returns MemorySaver for dev, AsyncPostgresSaver for production.
    Gracefully falls back to MemorySaver if Postgres library is unavailable.
    """
    if not _USE_POSTGRES_CHECKPOINTER:
        logger.info("Using MemorySaver checkpointer (dev mode). "
                    "Set USE_POSTGRES_CHECKPOINTER=true for durable persistence.")
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from app.config import settings

        # Build connection string for LangGraph checkpointer
        # Note: uses same DB as conversation history but different tables
        dsn = settings.conversation_db_dsn.replace("+asyncpg", "").replace("+psycopg2", "")
        conn = __import__("psycopg", fromlist=["connect"]).connect(dsn, autocommit=True)

        # Create the tables if they don't exist
        PostgresSaver.create_tables(conn)
        checkpointer = PostgresSaver(conn)

        logger.info("Using PostgresSaver checkpointer (durable, production-ready). "
                    "Conversation state persists across restarts and replicas.")
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed. "
            "Falling back to MemorySaver. "
            "Install: pip install langgraph-checkpoint-postgres psycopg[binary]"
        )
        return MemorySaver()
    except Exception as e:
        logger.warning(
            f"Failed to initialize PostgresSaver ({e}). "
            "Falling back to MemorySaver."
        )
        return MemorySaver()


checkpointer = _create_checkpointer()

# ── Cached compiled graph ─────────────────────────────────────────────────────
_graph = None


def _as_sync(fn):
    """
    Wrap an async node so it can run under sync graph.stream().

    Why: interrupt() only works reliably with sync .stream()/.invoke() in
    LangGraph 1.2.x. Our LLM/Qdrant/Tavily nodes are async, so we bridge them
    with a dedicated event loop and cancel leftover tasks before close
    (avoids noisy 'Event loop is closed' from httpx cleanup).
    """
    if not inspect.iscoroutinefunction(fn):
        return fn

    def sync_fn(state):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(fn(state))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    sync_fn.__name__ = getattr(fn, "__name__", "sync_fn")
    sync_fn.__doc__ = getattr(fn, "__doc__", "")
    return sync_fn


def build_graph():
    """
    Construct and compile the StateGraph.

    Called once at startup. Returns a compiled CompiledStateGraph
    that can be used for both .ainvoke() and .astream().
    """
    builder = StateGraph(AgentState)

    # ── Add all 7 nodes ───────────────────────────────────────────────────────
    # Async agents are wrapped for sync stream (required for human-in-the-loop).
    builder.add_node("query_validator", _as_sync(query_validator_node))
    builder.add_node("router", _as_sync(router_node))
    builder.add_node("rag_retriever", _as_sync(rag_retriever_node))
    builder.add_node("web_searcher", _as_sync(web_searcher_node))
    builder.add_node("human_gate", human_gate_node)  # sync — uses interrupt()
    builder.add_node("response_generator", _as_sync(response_generator_node))
    builder.add_node("quality_checker", _as_sync(quality_checker_node))

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "query_validator")

    # ── query_validator → router OR END ───────────────────────────────────────
    builder.add_conditional_edges(
        "query_validator",
        route_after_validation,
        {
            "valid": "router",
            "invalid": END,
        },
    )

    # ── router → retrieval nodes ───────────────────────────────────────────────
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag_retriever",
            "web_search": "web_searcher",
            "needs_human": "human_gate",
        },
    )

    # ── human_gate → retrieval (re-routes based on human answer) ──────────────
    builder.add_conditional_edges(
        "human_gate",
        route_after_human,
        {
            "rag": "rag_retriever",
            "web_search": "web_searcher",
        },
    )

    # ── Direct edges: retrieval → generation ──────────────────────────────────
    builder.add_edge("rag_retriever", "response_generator")
    builder.add_edge("web_searcher", "response_generator")

    # ── response_generator → quality_checker ──────────────────────────────────
    builder.add_edge("response_generator", "quality_checker")

    # ── quality_checker → END or revise loop ──────────────────────────────────
    builder.add_conditional_edges(
        "quality_checker",
        route_after_quality,
        {
            "end": END,
            "revise": "response_generator",  # retry loop
        },
    )

    # ── Compile with checkpointer ─────────────────────────────────────────────
    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully")
    return compiled


def get_graph():
    """Return the singleton compiled graph. Build it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_graph_schema() -> dict:
    """
    Return a JSON-serialisable description of the graph topology
    for the UI to render the visualisation.
    """
    return {
        "nodes": [
            {"id": "query_validator", "label": "Query Validator", "order": 1,
             "description": "Validates and sanitises the user query"},
            {"id": "router", "label": "Router", "order": 2,
             "description": "Decides: RAG, Web Search, or Human?"},
            {"id": "rag_retriever", "label": "RAG Retriever", "order": 3,
             "description": "Embeds query → searches Qdrant vector store"},
            {"id": "web_searcher", "label": "Web Searcher", "order": 3,
             "description": "Searches the live web via Tavily API"},
            {"id": "human_gate", "label": "Human Gate", "order": 3,
             "description": "Pauses and asks a human clarifying question"},
            {"id": "response_generator", "label": "Response Generator", "order": 4,
             "description": "Generates answer from evidence using GPT-4o"},
            {"id": "quality_checker", "label": "Quality Checker", "order": 5,
             "description": "Scores answer; PASS → done, REVISE → retry"},
        ],
        "edges": [
            {"from": "START", "to": "query_validator", "label": ""},
            {"from": "query_validator", "to": "router", "label": "valid"},
            {"from": "query_validator", "to": "END", "label": "invalid"},
            {"from": "router", "to": "rag_retriever", "label": "rag"},
            {"from": "router", "to": "web_searcher", "label": "web_search"},
            {"from": "router", "to": "human_gate", "label": "needs_human"},
            {"from": "human_gate", "to": "rag_retriever", "label": "rag"},
            {"from": "human_gate", "to": "web_searcher", "label": "web_search"},
            {"from": "rag_retriever", "to": "response_generator", "label": ""},
            {"from": "web_searcher", "to": "response_generator", "label": ""},
            {"from": "response_generator", "to": "quality_checker", "label": ""},
            {"from": "quality_checker", "to": "END", "label": "PASS"},
            {"from": "quality_checker", "to": "response_generator", "label": "REVISE"},
        ],
    }
