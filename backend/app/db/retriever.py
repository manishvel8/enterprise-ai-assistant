"""
db/retriever.py — Semantic similarity search using Qdrant.

How retrieval works:
  1. Embed the query with the same model used for chunks (text-embedding-3-small)
  2. Search Qdrant for the k most similar vectors using cosine similarity
  3. Optionally filter by document_id (if user wants to search specific docs)
  4. Return matched chunks with similarity scores as RetrievedChunk objects

Why top-k = 5?
  Retrieving 5 chunks gives the LLM enough context to answer most questions
  without exceeding the context window or increasing costs significantly.
  This is configurable per request.

Score threshold:
  We filter out chunks with similarity below the threshold to avoid injecting
  irrelevant context. Default is 0.25 (tuned for text-embedding-3-small on this stack).

Cache:
  Retrieval results are cached by (query_hash + document_ids) in Redis
  for 5 minutes. The same user asking slight variants of the same question
  will get the cached result rather than re-embedding and re-searching.
"""

import logging
from typing import List, Optional

from app.models.agent_state import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
# text-embedding-3-small cosine scores on this corpus often land ~0.3–0.55;
# 0.5 was filtering out all hits (demo resume topped out ~0.48).
DEFAULT_SCORE_THRESHOLD = 0.25
COLLECTION_NAME = "chunks"


async def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    document_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Find the most semantically similar chunks to `query`.

    Args:
        query: The user's search query (or rewritten query from QueryRewrite agent)
        top_k: Number of results to return
        score_threshold: Minimum cosine similarity score (0–1)
        document_ids: If provided, only search chunks from these documents

    Returns:
        List of RetrievedChunk objects, sorted by similarity (highest first).
        Returns empty list if Qdrant is unavailable or no matches found.
    """
    from app.services.cache_service import get_cached_retrieval, cache_retrieval
    from app.core.config import settings

    # Check retrieval cache
    doc_ids_list = document_ids or []
    cached = get_cached_retrieval(query, doc_ids_list, top_k)
    if cached:
        logger.info(f"Retrieval cache hit for query: {query[:50]}...")
        return [RetrievedChunk(**c) for c in cached]

    # Embed the query
    from app.pipeline.embedder import embed_query
    query_vector = await embed_query(query)

    # Search Qdrant
    results = _search_qdrant(
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
        document_ids=document_ids,
    )

    # Cache results for 5 minutes
    if results:
        cache_retrieval(query, doc_ids_list, top_k, [dict(r) for r in results], ttl=300)

    logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
    return results


def _search_qdrant(
    query_vector: List[float],
    top_k: int,
    score_threshold: float,
    document_ids: Optional[List[str]],
) -> List[RetrievedChunk]:
    """
    Execute the Qdrant search and convert results to RetrievedChunk objects.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
        from app.core.config import settings

        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=10,
        )

        # Build optional filter for specific documents
        search_filter = None
        if document_ids:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids),
                    )
                ]
            )

        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True,
        )

        chunks = []
        for hit in search_results:
            payload = hit.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                file_name=payload.get("file_name", ""),
                source_type=payload.get("source_type", ""),
                page_number=payload.get("page_number"),
                slide_number=payload.get("slide_number"),
                sheet_name=payload.get("sheet_name"),
                timestamp_start=payload.get("timestamp_start"),
                timestamp_end=None,
                section_title=payload.get("section_title"),
                chunk_text=payload.get("chunk_text", ""),
                similarity_score=hit.score,
                metadata={
                    "chunk_index": payload.get("chunk_index"),
                    "token_count": payload.get("token_count"),
                },
            ))

        return chunks

    except ImportError:
        logger.error("qdrant-client not installed. Run: pip install qdrant-client")
        return []
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []
