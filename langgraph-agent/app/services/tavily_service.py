"""
services/tavily_service.py — Tavily web search.

CONCEPT:
  When the router decides "web_search", web_searcher calls search().
  Tavily returns clean snippets + URLs (no HTML scraping).

WHY SYNC CLIENT?
  LangGraph human-in-the-loop uses sync graph.stream() and wraps async
  nodes with asyncio.run() in a worker thread. AsyncTavilyClient then
  often hangs / errors with "Event loop is closed" on cleanup.
  Sync TavilyClient is reliable in that path.
"""

import logging
from app.config import settings
from app.graph.state import WebResult

logger = logging.getLogger(__name__)


def search_sync(query: str, max_results: int = 5) -> list[WebResult]:
    """Synchronous Tavily search (preferred for LangGraph sync stream)."""
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — web search skipped")
        return [
            WebResult(
                title="[Tavily not configured]",
                url="",
                content="Set TAVILY_API_KEY in .env to enable live web search.",
                score=0.0,
            )
        ]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # faster + fewer timeouts than advanced
            include_answer=True,
            include_raw_content=False,
        )

        results: list[WebResult] = []
        # Optional short answer from Tavily itself
        if response.get("answer"):
            results.append(
                WebResult(
                    title="Tavily summary",
                    url="",
                    content=response["answer"],
                    score=1.0,
                )
            )

        for r in response.get("results", []):
            results.append(
                WebResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=round(r.get("score", 0.0), 4),
                )
            )

        logger.info("Tavily returned %d results for: %s", len(results), query[:60])
        return results[: max_results + 1]

    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return [
            WebResult(
                title="[Tavily error]",
                url="",
                content=f"Web search failed: {e}",
                score=0.0,
            )
        ]


async def search(query: str, max_results: int = 5) -> list[WebResult]:
    """Async wrapper — runs sync client (safe under asyncio.run wrappers)."""
    return search_sync(query, max_results=max_results)
