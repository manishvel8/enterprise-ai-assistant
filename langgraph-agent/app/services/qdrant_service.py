"""
services/qdrant_service.py — Qdrant vector search.

Uses the SYNC QdrantClient so short-lived asyncio.run() wrappers
(LangGraph HITL + MCP server tools) do not log:
  RuntimeError: Event loop is closed
during httpx/async client cleanup.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.graph.state import RetrievedDoc

logger = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    """Return a singleton sync Qdrant client."""
    global _client
    if _client is None:
        kwargs: dict = {
            "url": settings.qdrant_url,
            "check_compatibility": False,
        }
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def search_sync(
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.25,
    document_ids: Optional[list[str]] = None,
) -> list[RetrievedDoc]:
    """Sync similarity search (preferred implementation)."""
    client = get_qdrant()

    search_filter = None
    if document_ids:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=doc_id),
                )
                for doc_id in document_ids
            ]
        )

    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True,
        )
        results = response.points
    except Exception as e:
        logger.error("Qdrant search failed: %s", e)
        return []

    docs: list[RetrievedDoc] = []
    for hit in results:
        payload = hit.payload or {}
        docs.append(
            RetrievedDoc(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                file_name=payload.get("file_name", "unknown"),
                chunk_text=payload.get("chunk_text", ""),
                score=round(hit.score or 0.0, 4),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
            )
        )

    logger.info("Qdrant returned %d chunks (threshold=%.2f)", len(docs), score_threshold)
    return docs


async def search(
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.25,
    document_ids: Optional[list[str]] = None,
) -> list[RetrievedDoc]:
    """Async-compatible wrapper around sync search."""
    return search_sync(
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
        document_ids=document_ids,
    )


def collection_info_sync() -> dict:
    try:
        client = get_qdrant()
        info = client.get_collection(settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}


async def collection_info() -> dict:
    return collection_info_sync()
