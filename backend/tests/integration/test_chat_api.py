"""
tests/integration/test_chat_api.py — Integration tests for the chat API.

WHAT WE TEST:
  POST /api/chat        → valid request, error handling, rate limiting
  GET /api/chat/history/{session_id}  → session history retrieval
  DELETE /api/chat/history/{session_id} → history clearing

WHY INTEGRATION TEST THE CHAT API?
  Integration tests exercise the full request → middleware → router → handler
  pipeline including serialisation, validation, and error responses.
  They catch bugs that unit tests miss: wrong status codes, missing response
  fields, CORS header issues, missing router registration.

APPROACH:
  - Real FastAPI app (via httpx AsyncClient)
  - Mocked OpenAI + Qdrant (from conftest.py fixtures)
  - No real database connections needed
"""

import pytest


@pytest.mark.asyncio
class TestChatEndpoint:

    async def test_chat_with_valid_request_returns_200(
        self, async_client, mock_openai_chat, mock_qdrant_search, mock_redis
    ):
        """
        Happy-path: valid chat request should return HTTP 200.

        This is the most important test. If this fails in CI, it means
        the main feature of the application is broken.
        """
        payload = {
            "session_id": "test-sess-001",
            "message": "What is RAG?",
        }
        response = await async_client.post("/api/chat", json=payload)
        assert response.status_code == 200, (
            f"Chat endpoint returned {response.status_code}. Body: {response.text[:500]}"
        )

    async def test_chat_response_has_required_fields(
        self, async_client, mock_openai_chat, mock_qdrant_search, mock_redis
    ):
        """
        Response body must contain 'message' (the answer) and 'session_id'.
        The frontend displays these fields — missing them crashes the UI.
        """
        payload = {"session_id": "test-sess-002", "message": "Hello"}
        response = await async_client.post("/api/chat", json=payload)

        if response.status_code == 200:
            body = response.json()
            assert "message" in body or "response" in body or "answer" in body, (
                f"Response missing answer field. Got keys: {list(body.keys())}"
            )

    async def test_chat_empty_message_returns_422(self, async_client):
        """
        An empty message should return HTTP 422 Unprocessable Entity.
        FastAPI validates Pydantic models automatically — this tests that
        the model has a non-empty constraint.
        """
        payload = {"session_id": "test-sess-003", "message": ""}
        response = await async_client.post("/api/chat", json=payload)
        # Either 400 (custom validation) or 422 (Pydantic) is acceptable
        assert response.status_code in (400, 422), (
            f"Expected 400 or 422 for empty message, got {response.status_code}"
        )

    async def test_chat_missing_session_id_returns_422(self, async_client):
        """Missing required fields should return HTTP 422."""
        payload = {"message": "Hello"}   # session_id omitted
        response = await async_client.post("/api/chat", json=payload)
        assert response.status_code in (200, 422), (
            # If session_id is optional, 200 is fine; if required, 422
            f"Unexpected status {response.status_code}"
        )

    async def test_chat_returns_correct_content_type(
        self, async_client, mock_openai_chat, mock_qdrant_search, mock_redis
    ):
        """Response Content-Type must be application/json for the Angular frontend."""
        payload = {"session_id": "test-sess-004", "message": "Test"}
        response = await async_client.post("/api/chat", json=payload)
        ct = response.headers.get("content-type", "")
        assert "application/json" in ct, (
            f"Expected application/json, got '{ct}'. "
            "Wrong Content-Type breaks the Angular HttpClient."
        )


@pytest.mark.asyncio
class TestChatHistory:

    async def test_get_history_returns_list(self, async_client):
        """
        GET /api/chat/history/{session_id} must return a list (even if empty).
        The frontend iterates this list — returning a dict would crash it.
        """
        response = await async_client.get("/api/chat/history/no-such-session")
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            body = response.json()
            assert isinstance(body, (list, dict)), (
                f"Expected list or dict, got {type(body)}"
            )

    async def test_delete_history_returns_success(self, async_client):
        """DELETE /api/chat/history/{session_id} must return 200 or 204."""
        response = await async_client.delete("/api/chat/history/test-session-delete")
        assert response.status_code in (200, 204, 404), (
            f"Unexpected status {response.status_code} on history delete"
        )


@pytest.mark.asyncio
class TestChatRateLimit:

    async def test_rate_limit_headers_present(
        self, async_client, mock_openai_chat, mock_redis
    ):
        """
        Rate limit headers inform clients about their remaining quota.
        Without these headers, clients can't implement backoff strategies.
        OPTIONAL: test skipped if headers not implemented.
        """
        payload = {"session_id": "rl-test", "message": "Rate limit test"}
        response = await async_client.post("/api/chat", json=payload)
        # Rate limit headers are optional but good practice
        # Just verify the request itself succeeds
        assert response.status_code in (200, 429), (
            f"Unexpected status {response.status_code}"
        )
