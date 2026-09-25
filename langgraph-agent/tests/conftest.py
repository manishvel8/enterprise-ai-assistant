"""
langgraph-agent/tests/conftest.py — Shared fixtures for LangGraph agent tests.

STRUCTURE:
  - async_client     → httpx AsyncClient for the LangGraph FastAPI app (:7860)
  - mock_llm_service → replaces OpenAI calls in all 7 LangGraph nodes
  - mock_qdrant      → replaces Qdrant vector search
  - base_state       → default AgentState for node-level tests
  - mock_langfuse    → disables Langfuse SDK (no-op tracer in tests)
  - mock_db          → replaces psycopg2 pool (no real Postgres needed)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_langfuse():
    """
    Disable Langfuse in tests.
    Without this, tests would try to send traces to http://localhost:3000
    which is not available in CI, causing connection errors and slow tests.
    """
    with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
        yield


@pytest.fixture
def mock_llm_service():
    """
    Replace all LLM calls with fast deterministic responses.
    Each node that calls llm_service.chat / structured_chat gets a mock.
    """
    chat_mock = AsyncMock(return_value="This is a test LLM response.")

    structured_mock = AsyncMock()
    structured_mock.return_value = MagicMock(
        model_dump=lambda: {"is_valid": True, "reason": "test", "sanitized_query": "test query", "domain_relevant": True}
    )

    with (
        patch("app.services.llm_service.chat", new=chat_mock),
        patch("app.services.llm_service.structured_chat", new=structured_mock),
        patch("app.services.llm_service.chat_with_usage", new=AsyncMock(
            return_value=("Test response", {"prompt_tokens": 10, "completion_tokens": 5})
        )),
    ):
        yield {"chat": chat_mock, "structured": structured_mock}


@pytest.fixture
def mock_qdrant_service():
    """Replace Qdrant search with deterministic fake results."""
    fake_results = [
        {
            "text": "The candidate has 5 years of Python and FastAPI experience.",
            "score": 0.92,
            "document_id": "doc-001",
            "source_name": "resume.pdf",
            "chunk_index": 0,
        }
    ]
    with patch("app.services.qdrant_service.search_similar", new_callable=AsyncMock,
               return_value=fake_results):
        yield fake_results


@pytest.fixture
def mock_tavily_service():
    """Replace web search with deterministic fake results."""
    fake_results = [
        {
            "title": "LangGraph Tutorial",
            "url": "https://example.com/langgraph",
            "content": "LangGraph is a framework for building stateful AI agents.",
            "score": 0.88,
        }
    ]
    with patch("app.services.tavily_service.search", new_callable=AsyncMock,
               return_value=fake_results):
        yield fake_results


@pytest.fixture
def mock_db_pool():
    """Replace psycopg2 pool so memory/conversation tests work without Postgres."""
    mock_pool = MagicMock()
    mock_pool.execute = MagicMock(return_value=None)
    mock_pool.fetchone = MagicMock(return_value=None)
    mock_pool.fetchall = MagicMock(return_value=[])

    with patch("app.database.db_client.get_pool", return_value=mock_pool):
        yield mock_pool


@pytest.fixture
def base_agent_state() -> dict:
    """
    Minimal valid AgentState for testing individual nodes.
    All required fields are present with sensible defaults.
    """
    return {
        "user_query": "What are the candidate's Python skills?",
        "validated_query": None,
        "validation_result": None,
        "route_decision": None,
        "retrieved_chunks": [],
        "web_results": [],
        "human_feedback": None,
        "generated_response": None,
        "quality_score": None,
        "quality_passed": None,
        "quality_retry_count": 0,
        "execution_trace": [],
        "errors": [],
        # Multi-user fields
        "user_id": "test-user-001",
        "conversation_id": None,
        "session_id": "test-session-001",
        "langfuse_trace_id": None,
        "conversation_context": [],
        "model_name": "gpt-4o",
    }


@pytest.fixture(scope="session")
async def async_client():
    """httpx AsyncClient for the LangGraph FastAPI app."""
    import httpx

    # Patch DB/LLM before importing app
    with (
        patch("app.database.db_client.init_pool", new_callable=AsyncMock),
        patch("app.services.qdrant_service.init_qdrant", new_callable=AsyncMock),
        patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False),
    ):
        from app.main import app
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client
