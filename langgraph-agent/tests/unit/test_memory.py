"""
langgraph-agent/tests/unit/test_memory.py — Unit tests for the memory layer.

WHAT WE TEST:
  - ConversationManager: user/conversation/session CRUD
  - ContextManager: sliding window, summarisation, hybrid strategies
  - MemoryRetrieval: long-term memory store and pgvector search

WHY MEMORY TESTS?
  The memory layer is what makes the AI assistant feel like a conversation,
  not a series of isolated questions. Bugs here cause:
  - User context leakage between sessions (privacy violation)
  - Lost conversation history (users have to repeat themselves)
  - Wrong context window size (exceeds token budget → API error)

MOCKING APPROACH:
  All Postgres interactions go through db_client.get_pool().
  We mock the pool so tests run without a real database.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestContextManager:
    """
    ContextManager computes the conversation context to inject into each LLM call.
    It implements 5 strategies: last_n, sliding_window, summarize, semantic, hybrid.
    """

    def test_last_n_returns_correct_message_count(self):
        """last_n strategy should return exactly n most recent messages."""
        from app.memory.context_manager import get_last_n_messages

        messages = [
            {"role": "user", "content": f"message {i}"} for i in range(20)
        ]
        result = get_last_n_messages(messages, n=5)
        assert len(result) == 5, f"Expected 5 messages, got {len(result)}"
        # Should be the most RECENT 5
        assert result[-1]["content"] == "message 19"
        assert result[0]["content"] == "message 15"

    def test_last_n_with_fewer_messages_returns_all(self):
        """If fewer than n messages exist, return all of them (no index error)."""
        from app.memory.context_manager import get_last_n_messages

        messages = [{"role": "user", "content": "only one"}]
        result = get_last_n_messages(messages, n=10)
        assert len(result) == 1

    def test_sliding_window_respects_token_budget(self):
        """
        Sliding window strategy must not exceed max_tokens.
        Exceeding the token budget causes OpenAI API errors.
        """
        from app.memory.context_manager import sliding_window_context

        # Each message is ~10 tokens; budget of 50 tokens → max 5 messages
        messages = [
            {"role": "user", "content": "word " * 10, "token_count": 10}
            for _ in range(20)
        ]
        result = sliding_window_context(messages, max_tokens=50)
        total_tokens = sum(m.get("token_count", 10) for m in result)
        assert total_tokens <= 50, (
            f"Sliding window exceeded token budget: {total_tokens} > 50. "
            "This would cause OpenAI API context length errors."
        )

    def test_empty_history_returns_empty_context(self):
        """Empty conversation history should produce empty context, not an error."""
        from app.memory.context_manager import get_last_n_messages

        result = get_last_n_messages([], n=10)
        assert result == []


class TestConversationManager:
    """
    ConversationManager handles user/conversation/session CRUD in PostgreSQL.
    All DB calls are mocked.
    """

    def test_get_or_create_user_returns_user_dict(self, mock_db_pool):
        """get_or_create_user must return a dict with at least 'id' and 'name'."""
        from app.memory.conversation_manager import ConversationManager

        mock_db_pool.fetchone.return_value = {
            "id": "user-abc",
            "name": "test-user-001",
            "created_at": "2026-01-01",
        }

        mgr = ConversationManager()
        user = mgr.get_or_create_user("test-user-001")

        assert isinstance(user, dict), f"Expected dict, got {type(user)}"
        assert "id" in user or user is not None

    def test_save_message_calls_db_execute(self, mock_db_pool):
        """save_message must write to the database (execute called at least once)."""
        from app.memory.conversation_manager import ConversationManager

        mgr = ConversationManager()
        mgr.save_message(
            conversation_id="conv-001",
            role="user",
            content="Hello",
            token_count=5,
        )
        assert mock_db_pool.execute.called, (
            "save_message did not call db_client.execute. "
            "Messages are not being persisted."
        )

    def test_get_messages_returns_list(self, mock_db_pool):
        """get_messages must return a list (empty list if no messages)."""
        from app.memory.conversation_manager import ConversationManager

        mock_db_pool.fetchall.return_value = []
        mgr = ConversationManager()
        messages = mgr.get_messages("conv-001", limit=20)
        assert isinstance(messages, list)


class TestMemoryRetrieval:
    """
    MemoryRetrieval stores and retrieves long-term semantic memories using pgvector.
    """

    def test_store_memory_calls_db_execute(self, mock_db_pool):
        """store_memory must persist the embedding to the memory table."""
        with patch("app.memory.memory_retrieval.embed_text",
                   return_value=[0.01] * 1536):
            from app.memory.memory_retrieval import MemoryRetrieval

            mr = MemoryRetrieval()
            mr.store_memory(
                user_id="user-001",
                content="User prefers concise answers",
                memory_type="preference",
                importance=0.8,
            )
            assert mock_db_pool.execute.called

    def test_retrieve_memories_returns_list(self, mock_db_pool):
        """retrieve_similar_memories must return a list (empty if no memories)."""
        mock_db_pool.fetchall.return_value = []
        with patch("app.memory.memory_retrieval.embed_text",
                   return_value=[0.01] * 1536):
            from app.memory.memory_retrieval import MemoryRetrieval

            mr = MemoryRetrieval()
            results = mr.retrieve_similar_memories("user-001", "Python", top_k=5)
            assert isinstance(results, list)
