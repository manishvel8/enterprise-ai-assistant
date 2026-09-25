"""
conftest.py — Shared pytest fixtures for the backend test suite.

WHY conftest.py?
  pytest automatically discovers this file and makes all fixtures defined here
  available to every test in this directory and subdirectories without any import.

FIXTURE STRATEGY:
  - async_client      → httpx AsyncClient wired to the FastAPI app (no real network)
  - mock_openai       → replaces real OpenAI calls so tests are cheap and fast
  - mock_qdrant       → replaces real Qdrant so no vector DB needed in CI
  - mock_redis        → replaces real Redis for rate-limit / cache tests
  - sample_pdf_bytes  → tiny valid PDF for upload tests
"""

import io
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# ─── Async test support ───────────────────────────────────────────────────────
# pytest-asyncio scoped to "session" avoids re-creating event loops for each test.
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─── App import ───────────────────────────────────────────────────────────────
# We patch expensive startup (DB connections) before importing the app so that
# tests can run without real infrastructure.

@pytest.fixture(scope="session")
def mock_db_connections():
    """
    Patch all startup connections so the FastAPI app can be imported
    in CI without Postgres/Qdrant/Neo4j/Redis being available.
    """
    patches = [
        patch("app.db.postgres.init_db", new_callable=AsyncMock),
        patch("app.db.vector_store.init_qdrant", new_callable=AsyncMock),
        patch("app.db.neo4j_client.init_neo4j", new_callable=AsyncMock),
        patch("app.services.cache.init_redis", new_callable=AsyncMock),
    ]
    mocks = [p.start() for p in patches]
    yield mocks
    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
async def async_client(mock_db_connections) -> AsyncGenerator:
    """
    httpx AsyncClient pointed at the FastAPI app.

    WHY httpx over requests?
      The FastAPI app uses async endpoints. httpx supports async I/O natively
      and can talk to the ASGI app directly (no real HTTP server needed).

    HOW IT WORKS:
      httpx.AsyncClient(app=app, base_url="http://test") creates an in-process
      HTTP client. Requests are routed through FastAPI's ASGI interface — same
      middleware, same routing, same dependency injection — but zero network.
    """
    import httpx
    from app.main import app

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ─── OpenAI mock ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_chat():
    """
    Replace OpenAI chat completions with a fast deterministic response.

    WHY MOCK LLM CALLS?
    - Real LLM calls cost money
    - They are slow (1–5s each)
    - They are non-deterministic (different output each time = flaky tests)
    - CI runners may not have API key access
    """
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "This is a test answer from the mock LLM."
    fake_response.usage.prompt_tokens = 50
    fake_response.usage.completion_tokens = 20
    fake_response.usage.total_tokens = 70
    fake_response.model = "gpt-4o"

    with patch("app.services.openai_service.client.chat.completions.create",
               return_value=fake_response):
        yield fake_response


@pytest.fixture
def mock_openai_embed():
    """Return a deterministic 1536-d unit vector for embedding calls."""
    fake_embed = MagicMock()
    fake_embed.data = [MagicMock()]
    fake_embed.data[0].embedding = [0.01] * 1536

    with patch("app.services.openai_service.client.embeddings.create",
               return_value=fake_embed):
        yield fake_embed


# ─── Qdrant mock ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qdrant_search():
    """Return fake search results from Qdrant without hitting the real DB."""
    from qdrant_client.models import ScoredPoint

    fake_point = MagicMock(spec=ScoredPoint)
    fake_point.score = 0.92
    fake_point.payload = {
        "text": "This is a sample chunk retrieved from Qdrant.",
        "document_id": "doc-test-001",
        "source_name": "test_document.pdf",
        "chunk_index": 0,
        "section_title": "Introduction",
    }

    with patch("app.db.vector_store.search_similar_chunks",
               new_callable=AsyncMock,
               return_value=[fake_point]):
        yield [fake_point]


# ─── Redis mock ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """Mock Redis so rate-limit and cache tests work without Redis."""
    mock_r = AsyncMock()
    mock_r.get.return_value = None          # cache miss by default
    mock_r.set.return_value = True
    mock_r.incr.return_value = 1            # always under rate limit
    mock_r.expire.return_value = True

    with patch("app.services.cache.get_redis_client", return_value=mock_r):
        yield mock_r


# ─── Sample file fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Minimal valid PDF binary.
    PDF format requires specific magic bytes (%PDF-) and a cross-reference table.
    This tiny PDF has one page with the text 'Hello World'.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 720 Td (Hello World) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000058 00000 n\n"
        b"0000000115 00000 n\n0000000266 00000 n\n"
        b"0000000360 00000 n\n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
    )


@pytest.fixture
def sample_chat_request() -> dict:
    """Standard chat request payload used across multiple tests."""
    return {
        "session_id": "test-session-001",
        "user_id": "test-user-001",
        "message": "What are the key skills in the uploaded resume?",
    }


@pytest.fixture
def auth_headers() -> dict:
    """
    JWT authorization header for tests that hit protected endpoints.
    Generates a real token using the same create_access_token function.
    """
    try:
        from app.core.auth import create_access_token
        token = create_access_token({"sub": "test-user-001", "email": "test@example.com"})
        return {"Authorization": f"Bearer {token}"}
    except ImportError:
        # auth not yet implemented — return empty headers
        return {}
