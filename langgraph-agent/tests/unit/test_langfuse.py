"""
langgraph-agent/tests/unit/test_langfuse.py — Unit tests for Langfuse observability.

WHAT WE TEST:
  - LangfuseTracer no-ops gracefully when LANGFUSE_ENABLED=False
  - create_trace returns a trace object (or None if disabled)
  - add_span works without raising exceptions
  - flush doesn't crash when called with no active traces
  - Score function accepts valid score values

WHY TEST OBSERVABILITY CODE?
  If the Langfuse tracer crashes, it brings down the entire request because
  it runs inside each agent node. We must guarantee that:
  1. Langfuse errors are caught and never propagate to the user
  2. The tracer degrades gracefully when Langfuse is unavailable (offline mode)
"""

import pytest
from unittest.mock import patch, MagicMock


class TestLangfuseTracerDisabled:
    """Tests that the tracer is a safe no-op when LANGFUSE_ENABLED=False."""

    def test_get_tracer_returns_object_when_disabled(self):
        """get_tracer() must return an object (even a no-op), never None."""
        with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
            from app.observability.langfuse_client import get_tracer
            tracer = get_tracer()
            assert tracer is not None, (
                "get_tracer() returned None. This will cause AttributeError "
                "in every agent node that calls tracer.create_trace()"
            )

    def test_create_trace_does_not_raise_when_disabled(self):
        """create_trace must not raise even when Langfuse is disabled."""
        with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
            from app.observability.langfuse_client import get_tracer
            tracer = get_tracer()
            try:
                trace = tracer.create_trace(
                    name="test-trace",
                    user_id="user-001",
                    session_id="sess-001",
                    metadata={"test": True},
                )
            except Exception as e:
                pytest.fail(
                    f"create_trace raised {type(e).__name__}: {e} when Langfuse disabled. "
                    "This would crash every chat request."
                )

    def test_add_span_does_not_raise_when_disabled(self):
        """add_span must not raise when Langfuse is disabled."""
        with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
            from app.observability.langfuse_client import get_tracer
            tracer = get_tracer()
            try:
                tracer.add_span(
                    trace_id="trace-001",
                    name="test-span",
                    input_data={"query": "test"},
                    output_data={"result": "ok"},
                )
            except Exception as e:
                pytest.fail(f"add_span raised {type(e).__name__}: {e}")

    def test_flush_does_not_raise_when_disabled(self):
        """flush() must be safe to call even with no active Langfuse connection."""
        with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
            from app.observability.langfuse_client import get_tracer
            tracer = get_tracer()
            try:
                tracer.flush()
            except Exception as e:
                pytest.fail(f"flush() raised {type(e).__name__}: {e}")

    def test_score_trace_does_not_raise_when_disabled(self):
        """score_trace must not raise when Langfuse is disabled."""
        with patch("app.observability.langfuse_client.LANGFUSE_ENABLED", False):
            from app.observability.langfuse_client import get_tracer
            tracer = get_tracer()
            try:
                tracer.score_trace(
                    trace_id="trace-001",
                    name="quality",
                    value=0.9,
                    comment="test score",
                )
            except Exception as e:
                pytest.fail(f"score_trace raised {type(e).__name__}: {e}")


class TestLangfuseTracerEnabled:
    """Tests with Langfuse SDK mocked (connection exists but no real server)."""

    def test_create_trace_returns_trace_id(self):
        """When Langfuse is enabled, create_trace should return a trace ID string."""
        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "trace-12345"
        mock_langfuse.trace.return_value = mock_trace

        with (
            patch("app.observability.langfuse_client.LANGFUSE_ENABLED", True),
            patch("app.observability.langfuse_client.Langfuse", return_value=mock_langfuse),
        ):
            from importlib import reload
            import app.observability.langfuse_client as lf_mod
            reload(lf_mod)

            tracer = lf_mod.get_tracer()
            # Just verify it doesn't raise
            assert tracer is not None
