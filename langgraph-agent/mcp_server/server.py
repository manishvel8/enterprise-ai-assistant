"""
mcp_server/server.py — Model Context Protocol SERVER for this project.

WHAT IS THIS?
  An MCP server is a small program that EXPOSES tools (functions) over a
  standard protocol. Any MCP client (Cursor, Claude Desktop, our FastAPI
  mcp_client, etc.) can discover and call those tools without importing
  our Python modules directly.

TOOLS EXPOSED:
  1. search_documents  — RAG over Qdrant (same collection as LangGraph)
  2. web_search        — Tavily live web search
  3. qdrant_info       — collection health / point count
  4. echo_demo         — trivial tool so beginners can test MCP with no deps

HOW TO RUN (stdio — how Cursor talks to MCP):
  cd langgraph-agent
  source venv/bin/activate
  python -m mcp_server.server

HOW TO RUN (HTTP — easy curl testing):
  python -m mcp_server.server --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make `app.*` importable when this file is launched as a subprocess
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_server")

# Server identity — clients see this name when listing MCP servers
mcp = MCPServer(
    "enterprise-rag-mcp",
    instructions=(
        "Tools for the Enterprise AI / LangGraph project: "
        "search uploaded documents in Qdrant, search the live web via Tavily, "
        "and check vector-store health."
    ),
)


def _run_async(coro):
    """
    Run an async coroutine from a sync MCP tool handler.

    Prefer sync service APIs when possible. If a coro is passed, run it on a
    fresh loop and cancel leftover tasks so httpx does not log
    'Event loop is closed' on cleanup.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@mcp.tool(
    name="echo_demo",
    description="Beginner demo tool. Echoes your message back. No external deps.",
)
def echo_demo(message: str) -> str:
    """Return the same message with an MCP prefix (for first-time testing)."""
    return json.dumps({"ok": True, "echo": message, "via": "MCP echo_demo"})


@mcp.tool(
    name="qdrant_info",
    description="Return Qdrant collection name, point count, and status.",
)
def qdrant_info() -> str:
    """Health/stats for the shared `chunks` collection."""
    from app.services import qdrant_service

    info = qdrant_service.collection_info_sync()
    return json.dumps(info, indent=2)


@mcp.tool(
    name="search_documents",
    description=(
        "Semantic search over uploaded PDFs in Qdrant (RAG). "
        "Use for questions about resumes, CVs, or internal documents."
    ),
)
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Embed the query and search Qdrant. Returns JSON list of chunks.
    """
    from app.services import llm_service, qdrant_service

    # sync OpenAI embed via async wrapper on a clean loop
    vector = _run_async(llm_service.embed(query))
    docs = qdrant_service.search_sync(
        query_vector=vector,
        top_k=top_k,
        score_threshold=0.25,
    )
    return json.dumps(
        {
            "query": query,
            "count": len(docs),
            "chunks": docs,
        },
        indent=2,
    )


@mcp.tool(
    name="web_search",
    description=(
        "Live web search via Tavily. Use for latest news, trends, "
        "or facts not in uploaded documents."
    ),
)
def web_search(query: str, max_results: int = 5) -> str:
    """Call Tavily and return JSON results."""
    from app.services import tavily_service

    results = tavily_service.search_sync(query, max_results=max_results)
    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "results": results,
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Enterprise RAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="stdio = Cursor/Claude Desktop; streamable-http = curl/browser testing",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    logger.info(
        "Starting MCP server 'enterprise-rag-mcp' transport=%s",
        args.transport,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
