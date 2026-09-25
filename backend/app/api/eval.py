"""
api/eval.py — Evaluation and observability endpoints for the debug panel.

These endpoints power the evaluation dashboard in the Angular frontend:
  GET /api/eval/stats       — Overall system stats (documents, chunks, vectors)
  GET /api/eval/recent      — Recent queries with latency and cost
  POST /api/eval/feedback   — Submit user feedback (thumbs up/down) on a response
  GET /api/eval/health      — Detailed health check for all services
"""

import logging
from fastapi import APIRouter
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/eval", tags=["Evaluation"])
logger = logging.getLogger(__name__)


class UserFeedback(BaseModel):
    """User feedback on an AI response."""
    trace_id: Optional[str] = None
    session_id: str
    message_id: Optional[str] = None
    score: int   # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = None


@router.get("/stats", summary="System statistics for the evaluation dashboard")
async def get_stats():
    """
    Return overall system statistics.

    Includes: document count, chunk count, vector count, conversation count.
    Used by the debug panel to show system health at a glance.
    """
    stats = {
        "documents": {"total": 0, "processing": 0, "complete": 0, "failed": 0},
        "chunks": {"total": 0},
        "vectors": None,
        "conversations": {"total": 0},
        "services": {},
    }

    # PostgreSQL stats
    try:
        from app.db.postgres import get_session_factory
        from sqlalchemy import text

        session_factory = get_session_factory()
        async with session_factory() as db:
            # Document counts by status
            result = await db.execute(text(
                "SELECT status, COUNT(*) as count FROM documents GROUP BY status"
            ))
            for row in result:
                stats["documents"][row.status] = row.count
                stats["documents"]["total"] += row.count

            # Chunk count
            result = await db.execute(text("SELECT COUNT(*) FROM chunks"))
            stats["chunks"]["total"] = result.scalar() or 0

            # Session count
            result = await db.execute(text("SELECT COUNT(*) FROM sessions"))
            stats["conversations"]["total"] = result.scalar() or 0

        stats["services"]["postgres"] = "ok"
    except Exception as e:
        stats["services"]["postgres"] = str(e)

    # Qdrant vector count
    try:
        from app.db.vector_store import get_collection_info
        info = get_collection_info()
        if info:
            stats["vectors"] = {"count": info.get("vectors_count", 0)}
            stats["services"]["qdrant"] = "ok"
    except Exception as e:
        stats["services"]["qdrant"] = str(e)

    # Redis status
    try:
        from app.services.cache_service import get_redis_client
        client = get_redis_client()
        if client:
            client.ping()
            stats["services"]["redis"] = "ok"
        else:
            stats["services"]["redis"] = "not_configured"
    except Exception as e:
        stats["services"]["redis"] = str(e)

    # Neo4j status
    try:
        from app.db.neo4j_client import health_check
        stats["services"]["neo4j"] = "ok" if health_check() else "unavailable"
    except Exception as e:
        stats["services"]["neo4j"] = str(e)

    return stats


@router.get("/recent", summary="Recent query history with metrics")
async def get_recent_queries(limit: int = 20):
    """
    Return recent chat queries with latency, cost, and quality metrics.

    Used by the debug panel to show query performance trends.
    """
    try:
        from app.db.postgres import get_session_factory, MessageModel
        from sqlalchemy import select, desc

        session_factory = get_session_factory()
        async with session_factory() as db:
            result = await db.execute(
                select(MessageModel)
                .where(MessageModel.role == "user")
                .order_by(desc(MessageModel.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()

            return {
                "queries": [
                    {
                        "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                        "session_id": m.session_id,
                        "timestamp": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in messages
                ]
            }
    except Exception as e:
        logger.error(f"Failed to get recent queries: {e}")
        return {"queries": []}


@router.post("/feedback", summary="Submit user feedback on a response")
async def submit_feedback(feedback: UserFeedback):
    """
    Submit thumbs up/down feedback for a response.

    Stores the feedback in Langfuse as a score.
    Used to measure user satisfaction and identify problem areas.
    """
    from app.services.langfuse_service import record_score

    record_score(
        trace_id=feedback.trace_id,
        name="user_feedback",
        value=float(feedback.score),
        comment=feedback.comment,
    )

    logger.info(
        f"Feedback received: session={feedback.session_id}, "
        f"score={feedback.score}, trace={feedback.trace_id}"
    )

    return {"status": "ok", "message": "Feedback recorded"}


@router.get("/health", summary="Detailed health check for all services")
async def detailed_health():
    """
    Detailed health check for all infrastructure components.

    Returns status for each service: ok, degraded, unavailable.
    Used by the debug panel health indicator.
    """
    health = {}

    services = {
        "postgres": _check_postgres,
        "redis": _check_redis,
        "qdrant": _check_qdrant,
        "neo4j": _check_neo4j,
        "minio": _check_minio,
    }

    for service_name, check_fn in services.items():
        try:
            health[service_name] = await check_fn()
        except Exception as e:
            health[service_name] = {"status": "error", "error": str(e)}

    all_ok = all(s.get("status") == "ok" for s in health.values())
    return {
        "overall": "ok" if all_ok else "degraded",
        "services": health,
    }


async def _check_postgres():
    try:
        from app.db.postgres import get_session_factory
        from sqlalchemy import text
        session_factory = get_session_factory()
        async with session_factory() as db:
            await db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


async def _check_redis():
    try:
        from app.services.cache_service import get_redis_client
        client = get_redis_client()
        if client:
            client.ping()
            return {"status": "ok"}
        return {"status": "not_configured"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


async def _check_qdrant():
    try:
        from app.db.vector_store import get_collection_info
        info = get_collection_info()
        if info:
            return {"status": "ok", "vectors_count": info.get("vectors_count")}
        return {"status": "collection_missing"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


async def _check_neo4j():
    try:
        from app.db.neo4j_client import health_check
        ok = health_check()
        return {"status": "ok" if ok else "unavailable"}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


async def _check_minio():
    try:
        from app.services.storage_service import get_minio_client
        client = get_minio_client()
        if client:
            return {"status": "ok"}
        return {"status": "not_configured"}
    except Exception as e:
        return {"status": "not_configured", "error": str(e)}
