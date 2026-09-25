"""
pipeline/embedder.py — Generate embeddings for document chunks using OpenAI.

Model: text-embedding-3-small
  - 1536-dimensional vectors
  - Cost: $0.02 per 1M tokens (~$0.00002 per chunk)
  - Excellent for semantic search, document retrieval, clustering

Why embeddings?
  Text search (LIKE '%keyword%') can only find exact word matches.
  Embeddings encode semantic meaning as a vector in 1536-dimensional space.
  Similar meanings are close together (high cosine similarity) even if the
  exact words differ. This is what powers "semantic search" / RAG retrieval.

Batching strategy:
  The OpenAI embedding API accepts up to 2048 inputs per request.
  We batch chunks to minimize API calls (and thus latency + cost).
  Default batch size: 100 chunks per API call.

Caching:
  Embeddings are cached in Redis (via cache_service) so re-processing
  the same chunk doesn't re-call the API. Cache key = SHA-256 of chunk text.
"""

import logging
import asyncio
import hashlib
from typing import List, Optional

from app.models.document import ChunkSchema

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100   # chunks per API call


async def embed_chunks(chunks: List[ChunkSchema]) -> List[List[float]]:
    """
    Generate embeddings for a list of chunks.

    Returns:
        List of embedding vectors (same order as input chunks).
        Each vector is a list of 1536 floats.
    """
    if not chunks:
        return []

    from app.services.openai_service import get_openai_client
    from app.services.cache_service import get_cached_embedding, cache_embedding
    from app.core.config import settings

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — returning zero vectors")
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in chunks]

    client = get_openai_client()
    all_embeddings: List[Optional[List[float]]] = [None] * len(chunks)

    # Separate cached vs uncached chunks
    to_embed_indices = []
    for i, chunk in enumerate(chunks):
        text_hash = _text_hash(chunk.chunk_text)
        cached = get_cached_embedding(text_hash)
        if cached:
            all_embeddings[i] = cached
            logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
        else:
            to_embed_indices.append(i)

    logger.info(
        f"Embedding {len(to_embed_indices)} chunks "
        f"({len(chunks) - len(to_embed_indices)} cache hits)"
    )

    if not to_embed_indices:
        return all_embeddings   # type: ignore

    # Batch API calls
    for batch_start in range(0, len(to_embed_indices), BATCH_SIZE):
        batch_indices = to_embed_indices[batch_start:batch_start + BATCH_SIZE]
        batch_texts = [chunks[i].chunk_text for i in batch_indices]

        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
                dimensions=EMBEDDING_DIMENSIONS,
            )

            for local_idx, embedding_obj in enumerate(response.data):
                chunk_idx = batch_indices[local_idx]
                vector = embedding_obj.embedding
                all_embeddings[chunk_idx] = vector

                # Cache the embedding
                text_hash = _text_hash(chunks[chunk_idx].chunk_text)
                cache_embedding(text_hash, vector)

            logger.info(
                f"Batch {batch_start // BATCH_SIZE + 1}: embedded {len(batch_indices)} chunks "
                f"({response.usage.total_tokens} tokens)"
            )

        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
            # Fill with zero vectors for failed batch
            for chunk_idx in batch_indices:
                all_embeddings[chunk_idx] = [0.0] * EMBEDDING_DIMENSIONS

    # Ensure no None values remain
    for i, emb in enumerate(all_embeddings):
        if emb is None:
            all_embeddings[i] = [0.0] * EMBEDDING_DIMENSIONS

    return all_embeddings   # type: ignore


def embed_chunks_sync(chunks: List[ChunkSchema]) -> List[List[float]]:
    """
    Synchronous wrapper for embed_chunks.
    Used by Celery workers (synchronous context).
    """
    return asyncio.run(embed_chunks(chunks))


async def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for similarity search.

    The query embedding is compared against chunk embeddings
    in Qdrant to find semantically similar chunks.

    Returns:
        A 1536-dimensional embedding vector.
    """
    from app.services.openai_service import get_openai_client
    from app.services.cache_service import get_cached_embedding, cache_embedding
    from app.core.config import settings

    if not settings.openai_api_key:
        return [0.0] * EMBEDDING_DIMENSIONS

    # Check cache
    text_hash = _text_hash(query)
    cached = get_cached_embedding(text_hash)
    if cached:
        return cached

    client = get_openai_client()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
        dimensions=EMBEDDING_DIMENSIONS,
    )
    vector = response.data[0].embedding

    # Cache for 1 hour (queries are often repeated)
    cache_embedding(text_hash, vector, ttl=3600)

    return vector


def _text_hash(text: str) -> str:
    """SHA-256 hash of text for use as cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
