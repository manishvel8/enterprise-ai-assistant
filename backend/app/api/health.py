"""
health.py — Health check endpoint.

Kubernetes uses two probes to check if a pod is healthy:

  /health/live  (liveness probe):
    "Is the application process alive?"
    If this fails, Kubernetes RESTARTS the pod.
    Should fail only if the app is deadlocked or crashed.

  /health/ready (readiness probe):
    "Is the application ready to serve traffic?"
    If this fails, Kubernetes removes the pod from the load balancer
    but does NOT restart it. Use this to signal "I'm still starting up"
    or "I'm draining requests, about to shut down".

We return information about connected dependencies so operators can
quickly see which service is unhealthy in a dashboard.
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

# Track when the application started (for uptime reporting)
_start_time = time.time()


class DependencyStatus(BaseModel):
    status: str                     # "up" | "down" | "unknown"
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                     # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: float
    dependencies: Dict[str, DependencyStatus]


@router.get("/live", summary="Liveness probe")
async def liveness():
    """
    Returns 200 as long as the FastAPI process is running.
    Kubernetes restarts the pod if this returns non-200 or times out.
    """
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness():
    """
    Checks that key dependencies are reachable.
    Returns 200 only when all required dependencies are up.

    Currently (Milestone 2): only checks that the app started correctly.
    Later milestones will add PostgreSQL, Redis, Qdrant, Neo4j checks.
    """
    uptime = time.time() - _start_time
    dependencies: Dict[str, DependencyStatus] = {}

    # --- Future: add real dependency checks here ---
    # Example:
    # try:
    #     await postgres_pool.fetchval("SELECT 1")
    #     dependencies["postgres"] = DependencyStatus(status="up")
    # except Exception as e:
    #     dependencies["postgres"] = DependencyStatus(status="down", error=str(e))

    dependencies["app"] = DependencyStatus(status="up", latency_ms=0.0)

    # Check Redis (Milestone 9+)
    try:
        import redis as redis_lib
        import time
        r = redis_lib.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, socket_timeout=2)
        t0 = time.time()
        r.ping()
        latency = (time.time() - t0) * 1000
        r.close()
        dependencies["redis"] = DependencyStatus(status="up", latency_ms=round(latency, 2))
    except Exception as e:
        dependencies["redis"] = DependencyStatus(status="down", error=str(e)[:100])

    all_up = all(d.status == "up" for d in dependencies.values())
    overall = "healthy" if all_up else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        uptime_seconds=round(uptime, 2),
        dependencies=dependencies,
    )


@router.get("", summary="Combined health check (for quick testing)")
async def health():
    """Quick health check — returns app name, version, and status."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }
