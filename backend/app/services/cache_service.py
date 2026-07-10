"""
cache_service.py — Redis caching for embeddings, retrieval results, and query results.

Why cache?
  The most expensive operations in this application are:
  1. Calling OpenAI embeddings API (costs money, adds latency)
  2. Running vector search in Qdrant (fast, but still a network call)
  3. Calling OpenAI chat completions (costs money, adds latency)

  If the same question or the same text is processed again, we should
  return the cached result instead of paying for another API call.

Cache layers:
  1. Embedding cache: hash(text) → embedding vector
     - Identical chunks from duplicate uploads don't need re-embedding
     - Same user query doesn't need re-embedding for retrieval

  2. Retrieval cache: hash(query_embedding + filters) → top-k chunks
     - Same question asked twice returns cached chunks
     - Cache TTL: 1 hour (chunks don't change often)

  3. Rate limiting: per-user and global counters
     - Implemented as Redis incrementing counters with TTL

TTL (Time To Live):
  - Embeddings: 24 hours (embeddings don't change unless model changes)
  - Retrieval: 1 hour (new documents might change the top results)
  - Session data: 30 days
"""

import json
import hashlib
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

_redis_client = None


def get_redis_client():
    """Get or create the Redis client. Returns None if Redis is unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        from app.core.config import settings
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=False,  # we store JSON bytes
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        _redis_client = client
        logger.info(f"Redis client connected: {settings.redis_host}:{settings.redis_port}")
        return _redis_client
    except ImportError:
        logger.warning("redis package not installed. Caching disabled.")
        return None
    except Exception as e:
        logger.warning(f"Redis not available: {e}. Caching disabled.")
        return None


def _cache_key(prefix: str, content: str) -> str:
    """
    Generate a deterministic cache key from a prefix and content.

    Uses SHA-256 hash of the content to create a fixed-length key
    regardless of how long the content is.

    Example:
        _cache_key("embed", "What is revenue?")
        → "embed:a3f8b2c1d4e5..."
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return f"{prefix}:{content_hash}"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────

def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Return cached embedding vector if available.

    Args:
        text: The text that was embedded.

    Returns:
        List of 1536 floats, or None if not in cache.
    """
    client = get_redis_client()
    if not client:
        return None

    key = _cache_key("embed", text)
    try:
        cached = client.get(key)
        if cached:
            embedding = json.loads(cached)
            logger.debug(f"Embedding cache HIT: {key[:20]}...")
            return embedding
    except Exception as e:
        logger.warning(f"Redis get failed: {e}")
    return None


def cache_embedding(text: str, embedding: List[float], ttl: int = 86400) -> None:
    """
    Store an embedding vector in cache.

    Args:
        text: The original text that was embedded.
        embedding: The 1536-dim vector from OpenAI.
        ttl: Time to live in seconds (default: 24 hours).
    """
    client = get_redis_client()
    if not client:
        return

    key = _cache_key("embed", text)
    try:
        client.setex(key, ttl, json.dumps(embedding))
        logger.debug(f"Embedding cached: {key[:20]}...")
    except Exception as e:
        logger.warning(f"Redis set failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Cache
# ─────────────────────────────────────────────────────────────────────────────

def get_cached_retrieval(query: str, document_ids: List[str], top_k: int) -> Optional[List[dict]]:
    """
    Return cached retrieval results if available.

    Cache key includes query + document_ids + top_k so that
    different filter combinations have separate cache entries.
    """
    client = get_redis_client()
    if not client:
        return None

    cache_content = f"{query}|{sorted(document_ids)}|{top_k}"
    key = _cache_key("retrieval", cache_content)
    try:
        cached = client.get(key)
        if cached:
            result = json.loads(cached)
            logger.debug(f"Retrieval cache HIT: query='{query[:30]}...'")
            return result
    except Exception as e:
        logger.warning(f"Redis retrieval get failed: {e}")
    return None


def cache_retrieval(query: str, document_ids: List[str], top_k: int, chunks: List[dict], ttl: int = 3600) -> None:
    """
    Store retrieval results in cache.

    Args:
        ttl: Time to live in seconds (default: 1 hour).
             Short TTL because new uploads change retrieval results.
    """
    client = get_redis_client()
    if not client:
        return

    cache_content = f"{query}|{sorted(document_ids)}|{top_k}"
    key = _cache_key("retrieval", cache_content)
    try:
        client.setex(key, ttl, json.dumps(chunks))
        logger.debug(f"Retrieval cached: query='{query[:30]}...'")
    except Exception as e:
        logger.warning(f"Redis retrieval set failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────

def check_rate_limit(user_id: str, limit_per_minute: int) -> bool:
    """
    Check if a user has exceeded their rate limit.

    Uses a sliding window counter with 60-second TTL.

    Returns:
        True if the user is within rate limit (allowed).
        False if the user has exceeded the limit (block).
    """
    client = get_redis_client()
    if not client:
        return True  # Allow if Redis unavailable (don't block on infra issues)

    key = f"rate_limit:{user_id}"
    try:
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)   # reset counter every 60 seconds
        results = pipe.execute()
        current_count = results[0]
        allowed = current_count <= limit_per_minute
        if not allowed:
            logger.warning(f"Rate limit exceeded: user={user_id}, count={current_count}/{limit_per_minute}")
        return allowed
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        return True  # Allow on error


# ─────────────────────────────────────────────────────────────────────────────
# General Cache Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_cache(key: str, value: Any, ttl: int = 3600) -> None:
    """Store any JSON-serializable value in cache."""
    client = get_redis_client()
    if not client:
        return
    try:
        client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache set failed for key {key}: {e}")


def get_cache(key: str) -> Optional[Any]:
    """Retrieve a value from cache. Returns None if not found or expired."""
    client = get_redis_client()
    if not client:
        return None
    try:
        cached = client.get(key)
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Cache get failed for key {key}: {e}")
        return None


def delete_cache(key: str) -> None:
    """Delete a cache entry."""
    client = get_redis_client()
    if not client:
        return
    try:
        client.delete(key)
    except Exception as e:
        logger.warning(f"Cache delete failed for key {key}: {e}")
