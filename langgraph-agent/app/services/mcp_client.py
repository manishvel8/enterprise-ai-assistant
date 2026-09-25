"""
services/mcp_client.py — MCP CLIENT that talks to our MCP server.

CONCEPT:
  MCP separates:
    - SERVER  = owns tools (search_documents, web_search, ...)
    - CLIENT  = discovers tools + calls them over a transport

  This module is the CLIENT used by:
    1. FastAPI endpoints  GET /api/mcp/tools , POST /api/mcp/call
    2. LangGraph nodes when USE_MCP_TOOLS=true

TRANSPORT USED HERE: stdio
  We spawn `python -m mcp_server.server` as a child process and speak
  MCP JSON-RPC over stdin/stdout. This is the same pattern Cursor uses.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import settings

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]  # langgraph-agent/


def _server_params() -> StdioServerParameters:
    """How to start our MCP server as a subprocess."""
    python = sys.executable  # same venv as the API
    env = os.environ.copy()
    # Ensure child sees the same .env-driven settings via environment
    if settings.openai_api_key:
        env["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_api_base:
        env["OPENAI_API_BASE"] = settings.openai_api_base
    if settings.tavily_api_key:
        env["TAVILY_API_KEY"] = settings.tavily_api_key
    if settings.qdrant_url:
        env["QDRANT_URL"] = settings.qdrant_url
    if settings.qdrant_collection:
        env["QDRANT_COLLECTION"] = settings.qdrant_collection

    return StdioServerParameters(
        command=python,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        cwd=str(_ROOT),
        env=env,
    )


@asynccontextmanager
async def mcp_session():
    """
    Open a short-lived MCP session (spawn server → initialize → yield → close).

    For demos/learning this is fine. Production would keep a long-lived pool.
    """
    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools() -> list[dict[str, Any]]:
    """Ask the MCP server what tools it exposes (discovery)."""
    async with mcp_session() as session:
        result = await session.list_tools()
        tools = []
        for t in result.tools:
            tools.append(
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
            )
        return tools


async def call_tool(name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Call one MCP tool by name.

    Returns a normalised dict:
      { "tool": name, "ok": bool, "text": str, "raw": ... }
    """
    arguments = arguments or {}
    async with mcp_session() as session:
        result = await session.call_tool(name, arguments)

        # MCP content is a list of content blocks (usually text)
        texts: list[str] = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(text)
            else:
                texts.append(str(block))

        combined = "\n".join(texts)
        parsed: Any = combined
        try:
            parsed = json.loads(combined)
        except Exception:
            pass

        is_error = bool(getattr(result, "isError", False))
        return {
            "tool": name,
            "ok": not is_error,
            "arguments": arguments,
            "text": combined,
            "data": parsed,
        }


async def search_documents_via_mcp(query: str, top_k: int = 5) -> list[dict]:
    """Convenience wrapper used by LangGraph when USE_MCP_TOOLS=true."""
    out = await call_tool("search_documents", {"query": query, "top_k": top_k})
    data = out.get("data") or {}
    if isinstance(data, dict):
        return data.get("chunks") or []
    return []


async def web_search_via_mcp(query: str, max_results: int = 5) -> list[dict]:
    """Convenience wrapper used by LangGraph when USE_MCP_TOOLS=true."""
    out = await call_tool("web_search", {"query": query, "max_results": max_results})
    data = out.get("data") or {}
    if isinstance(data, dict):
        return data.get("results") or []
    return []
