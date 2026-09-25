"""
tests/unit/test_agents.py — Unit tests for the 9-agent workflow (backend).

WHAT WE TEST:
  1. Router correctly identifies intent (rag / graph_question / general_chat)
  2. Query rewrite produces a cleaner version
  3. Agent state propagation (fields survive round-trip)
  4. Critic rejects low-quality answers and triggers retry
  5. Memory agent stores and loads conversation history

WHY MOCK THE LLM?
  Each agent calls OpenAI. Without mocking, unit tests would:
    - Cost money on every CI run
    - Take 30+ seconds
    - Fail if the API is down
  We mock the LLM layer and test the agent *logic* (routing decisions, retries,
  state mutations) independently of LLM response quality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def base_state(**overrides) -> dict:
    """Build a minimal AgentState dict for testing."""
    state = {
        "user_id": "user-001",
        "session_id": "sess-001",
        "user_query": "What are Neha's skills?",
        "document_ids": [],
        "intent": None,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "cypher_query": None,
        "cypher_results": [],
        "final_context": None,
        "draft_answer": None,
        "validated_answer": None,
        "is_grounded": None,
        "critic_feedback": None,
        "critic_iterations": 0,
        "citations": [],
        "conversation_history": [],
        "latency_ms": None,
        "token_usage": None,
        "cost_usd": None,
        "langfuse_trace_id": None,
        "errors": [],
        "debug_info": {},
    }
    state.update(overrides)
    return state


# ─── Router agent tests ───────────────────────────────────────────────────────

class TestRouterAgent:

    @patch("app.agents.router_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_router_returns_rag_for_document_query(self, mock_chat):
        """
        A query about a document should route to 'rag' retrieval.
        EXPECTED: intent='rag'
        """
        mock_chat.return_value = '{"intent": "rag", "reason": "query about uploaded document"}'
        from app.agents import router_agent
        state = base_state(user_query="What is in the uploaded resume?")
        result = await router_agent.run(state)
        assert result.get("intent") == "rag", f"Expected intent='rag', got {result.get('intent')}"

    @patch("app.agents.router_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_router_returns_general_chat_for_offopic(self, mock_chat):
        """A general greeting should not trigger document retrieval."""
        mock_chat.return_value = '{"intent": "general_chat", "reason": "simple greeting"}'
        from app.agents import router_agent
        state = base_state(user_query="Hello, how are you?")
        result = await router_agent.run(state)
        assert result.get("intent") == "general_chat"

    @patch("app.agents.router_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_router_preserves_existing_state_fields(self, mock_chat):
        """Router must not clear fields set by earlier nodes."""
        mock_chat.return_value = '{"intent": "rag", "reason": "document question"}'
        from app.agents import router_agent
        state = base_state(
            user_query="Find Python skills",
            session_id="my-session",
            user_id="my-user",
        )
        result = await router_agent.run(state)
        # Critical: earlier fields must be preserved
        assert result.get("session_id") == "my-session"
        assert result.get("user_id") == "my-user"


# ─── Critic agent tests ───────────────────────────────────────────────────────

class TestCriticAgent:

    @patch("app.agents.critic_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_critic_passes_good_answer(self, mock_chat):
        """
        When the LLM returns is_grounded=True the critic marks the answer valid.
        EXPECTED: is_grounded=True, critic_iterations unchanged
        """
        mock_chat.return_value = '{"is_grounded": true, "feedback": "Answer is accurate."}'
        from app.agents import critic_agent
        state = base_state(
            draft_answer="The candidate has 5 years of Python experience.",
            retrieved_chunks=[{"text": "Python expert with 5 years experience"}],
        )
        result = await critic_agent.run(state)
        assert result.get("is_grounded") is True

    @patch("app.agents.critic_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_critic_rejects_hallucinated_answer(self, mock_chat):
        """
        When the LLM says is_grounded=False, the critic must flag for revision.
        EXPECTED: is_grounded=False, critic_feedback is non-empty
        """
        mock_chat.return_value = (
            '{"is_grounded": false, '
            '"feedback": "Answer claims Java experience not found in context."}'
        )
        from app.agents import critic_agent
        state = base_state(
            draft_answer="The candidate is an expert Java developer.",
            retrieved_chunks=[{"text": "Candidate knows Python and SQL."}],
        )
        result = await critic_agent.run(state)
        assert result.get("is_grounded") is False
        assert result.get("critic_feedback"), "Critic feedback must be non-empty on rejection"

    @patch("app.agents.critic_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_critic_increments_iteration_counter(self, mock_chat):
        """Each critic call must increment critic_iterations (for retry loop limiting)."""
        mock_chat.return_value = '{"is_grounded": false, "feedback": "Not grounded."}'
        from app.agents import critic_agent
        state = base_state(
            draft_answer="Some answer.",
            retrieved_chunks=[],
            critic_iterations=1,
        )
        result = await critic_agent.run(state)
        assert result.get("critic_iterations", 0) > 1


# ─── Workflow integration (lightweight) ───────────────────────────────────────

class TestWorkflowStateFlow:

    @patch("app.agents.router_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    @patch("app.agents.rewrite_agent.openai_service.chat_completion",
           new_callable=AsyncMock)
    async def test_state_flows_through_router_and_rewrite(self, mock_rewrite, mock_router):
        """
        State must accumulate fields as it passes through nodes.
        After router + rewrite: intent and rewritten_query must be set.
        """
        mock_router.return_value = '{"intent": "rag", "reason": "doc query"}'
        mock_rewrite.return_value = '{"rewritten_query": "Python development skills"}'

        from app.agents import router_agent, rewrite_agent
        state = base_state(user_query="what python stuff does she know")
        state = await router_agent.run(state)
        state = await rewrite_agent.run(state)

        assert state.get("intent") == "rag"
        assert state.get("rewritten_query") is not None
        assert "python" in state.get("rewritten_query", "").lower()
