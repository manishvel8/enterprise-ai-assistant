"""
core/rate_limiter.py — Redis-backed rate limiting for API endpoints.

Rate limiting prevents:
  1. API abuse (a single user sending 1000 requests/minute)
  2. OpenAI cost explosion (uncapped requests = uncapped spend)
  3. DoS attacks (even accidental load from buggy frontends)

Algorithm: Sliding window counter
  For each (user_id, endpoint), track how many requests were made
  in the last N seconds using a Redis sorted set or simple TTL counter.

  Simple TTL approach (used here):
  - Key: rate_limit:{user_id}:{endpoint}
  - Value: request count
  - TTL: window_seconds
  - On each request: INCR the key. If it didn't exist, EXPIRE it.
  - If count > limit, reject with 429 Too Many Requests.

Default limits:
  - Chat: 30 requests per minute per user
  - Document upload: 10 uploads per minute per user
  - Health: no limit

In production, use a dedicated rate-limiting service or FastAPI middleware
like `slowapi` (wraps limits library) for more sophisticated algorithms.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Rate limit configuration: (requests, window_seconds)
RATE_LIMITS = {
    "chat": (30, 60),          # 30 requests per minute
    "upload": (10, 60),        # 10 uploads per minute
    "embed": (100, 60),        # 100 embedding requests per minute
}


def check_rate_limit(user_id: str, endpoint: str = "chat") -> tuple[bool, int]:
    """
    Check if the user has exceeded the rate limit for this endpoint.

    Args:
        user_id: The user identifier
        endpoint: The endpoint being rate-limited

    Returns:
        (allowed: bool, remaining: int)
        allowed=True means the request can proceed.
        remaining is the number of requests left in the current window.
    """
    from app.services.cache_service import get_redis_client

    client = get_redis_client()
    if not client:
        # Redis not available — allow all requests (graceful degradation)
        return True, 999

    limit, window = RATE_LIMITS.get(endpoint, (100, 60))
    key = f"rate_limit:{user_id}:{endpoint}"

    try:
        # Atomic increment
        current = client.incr(key)

        # Set TTL only on the first request in a window
        if current == 1:
            client.expire(key, window)

        remaining = max(0, limit - current)
        allowed = current <= limit

        if not allowed:
            logger.warning(
                f"Rate limit exceeded: user={user_id}, endpoint={endpoint}, "
                f"count={current}, limit={limit}"
            )

        return allowed, remaining

    except Exception as e:
        logger.warning(f"Rate limit check failed (allowing request): {e}")
        return True, 999


def get_rate_limit_headers(user_id: str, endpoint: str = "chat") -> dict:
    """
    Return standard rate limit HTTP headers for a response.

    Standard headers (RFC 6585):
        X-RateLimit-Limit: 30
        X-RateLimit-Remaining: 25
        X-RateLimit-Reset: 1720000060
    """
    from app.services.cache_service import get_redis_client
    import time

    client = get_redis_client()
    limit, window = RATE_LIMITS.get(endpoint, (100, 60))

    if not client:
        return {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(limit),
        }

    try:
        key = f"rate_limit:{user_id}:{endpoint}"
        current = int(client.get(key) or 0)
        ttl = client.ttl(key)
        remaining = max(0, limit - current)
        reset_time = int(time.time()) + max(ttl, 0)

        return {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }
    except Exception:
        return {"X-RateLimit-Limit": str(limit)}
