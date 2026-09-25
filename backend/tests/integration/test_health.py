"""
tests/integration/test_health.py — Integration tests for health check endpoints.

WHAT WE TEST:
  GET /health          → 200 + body has status/version
  GET /health/live     → 200 (liveness probe — used by Kubernetes)
  GET /health/ready    → 200 or 503 (readiness probe — used by Kubernetes)

WHY TEST HEALTH ENDPOINTS?
  Kubernetes uses these probes to decide whether to route traffic to a pod.
  If the liveness probe returns 500, Kubernetes restarts the container.
  If the readiness probe returns 503, Kubernetes stops routing traffic to it.
  A broken health endpoint can take down the entire application in production.

KUBERNETES CONTEXT:
  livenessProbe  → if fails: restart the container
  readinessProbe → if fails: remove pod from Service endpoints (stop traffic)
  startupProbe   → gives slow-starting containers more time before liveness kicks in
"""

import pytest


@pytest.mark.asyncio
class TestHealthEndpoints:

    async def test_root_health_returns_200(self, async_client):
        """
        GET /health must return 200.
        This is the most basic sanity check — if this fails, nothing else matters.
        """
        response = await async_client.get("/health")
        assert response.status_code == 200, (
            f"Health endpoint returned {response.status_code}. "
            "Check that the /health router is registered in main.py"
        )

    async def test_root_health_has_status_field(self, async_client):
        """Health response must include a 'status' field for monitoring tools."""
        response = await async_client.get("/health")
        body = response.json()
        assert "status" in body, (
            f"Health response missing 'status' field. Got: {body}"
        )

    async def test_root_health_has_version_field(self, async_client):
        """Health response should include app version for deployment verification."""
        response = await async_client.get("/health")
        body = response.json()
        assert "version" in body, f"Health response missing 'version'. Got: {body}"

    async def test_liveness_probe_returns_200(self, async_client):
        """
        GET /health/live → always 200 (means 'the process is alive').
        Kubernetes liveness probes MUST get 200 to avoid pod restarts.
        """
        response = await async_client.get("/health/live")
        assert response.status_code == 200, (
            f"Liveness probe returned {response.status_code}. "
            "This would cause Kubernetes to restart the pod in a crash loop."
        )

    async def test_readiness_probe_returns_valid_code(self, async_client):
        """
        GET /health/ready → 200 (all deps up) or 503 (some dep down).
        Both are valid responses. What is NOT valid is a 500 (unhandled exception).
        """
        response = await async_client.get("/health/ready")
        assert response.status_code in (200, 503), (
            f"Readiness probe returned unexpected {response.status_code}. "
            "Should be 200 (ready) or 503 (dependencies unavailable)."
        )

    async def test_health_response_includes_service_name(self, async_client):
        """Application name should be present so monitoring tools can label it."""
        response = await async_client.get("/health")
        body = response.json()
        # Accept either 'service', 'app', or 'name' field
        has_name = any(k in body for k in ("service", "app", "name", "app_name"))
        assert has_name, f"Health response should include app name. Got: {body}"
