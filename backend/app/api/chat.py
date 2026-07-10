"""
chat.py — Chat API endpoints.

Milestone 4: Real OpenAI GPT-4o integration with conversation history.
Milestone 16: RAG-grounded responses with source citations.
Milestone 22: Full 9-agent agentic workflow.

POST /api/chat                         — Send a message, get a response
GET  /api/chat/history/{session_id}    — Get conversation history
DELETE /api/chat/history/{session_id}  — Clear conversation history
GET  /api/chat/sessions                — List active sessions (debug)
"""

import time
import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException, status

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatHistoryItem,
    MessageRole,
    DebugInfo,
)
from app.core.config import settings

router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store — replaced by PostgreSQL in Milestone 8.
# ─────────────────────────────────────────────────────────────────────────────
_sessions: Dict[str, List[Dict]] = {}


def _get_history(session_id: str) -> List[Dict]:
    """Return conversation history for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def _history_to_messages(history: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert stored history to OpenAI messages format.

    OpenAI expects: [{"role": "user"|"assistant", "content": "..."}]
    We use the last 10 exchanges (20 messages) to keep prompt size manageable.

    Why limit history?
    - Each message in history adds to the prompt token count
    - More tokens = higher cost + slower response
    - Very old context is usually irrelevant
    """
    recent = history[-20:]                          # last 10 turns
    return [{"role": m["role"], "content": m["content"]} for m in recent]


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="""
Send a message and receive an AI-generated response.

**Milestone 4 behaviour:** Real GPT-4o responses with conversation history.
**Milestone 16+:** Adds RAG retrieval and source citations.
**Milestone 22+:** Full 9-agent agentic workflow.

Requires OPENAI_API_KEY in .env file.
    """,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat handler.

    Flow (Milestone 4):
    1. Load session history
    2. Add user message to history
    3. Convert history to OpenAI messages format
    4. Call OpenAI GPT-4o
    5. Save assistant response to history
    6. Return ChatResponse with answer + debug info

    Flow (Milestone 22+):
    Replace step 3-4 with the full agentic workflow graph.
    """
    from datetime import datetime, timezone
    start_time = time.time()

    history = _get_history(request.session_id)

    # Save user message to history
    timestamp = datetime.now(timezone.utc).isoformat()
    history.append({"role": "user", "content": request.message, "timestamp": timestamp})

    # --- Try OpenAI, fall back to echo if not configured ---
    openai_available = bool(settings.openai_api_key and settings.openai_api_key != "sk-your-openai-key-here")

    if openai_available:
        answer, token_usage, cost_usd = await _call_openai(history)
    else:
        answer = _fallback_response(request.message)
        token_usage = None
        cost_usd = None

    # Save assistant response to history
    history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    latency_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        citations=[],
        debug=DebugInfo(
            intent="general_chat",
            retrieved_chunks_count=0,
            similarity_scores=[],
            cypher_results_count=0,
            token_usage=token_usage,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2),
        ) if settings.enable_debug_panel else None,
    )


async def _call_openai(history: List[Dict]) -> tuple:
    """
    Call OpenAI and return (answer, token_usage, cost_usd).

    Handles common OpenAI errors and returns a user-friendly message.
    """
    from app.services.openai_service import chat_completion
    from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

    messages = _history_to_messages(history[:-1])  # exclude the message we just appended (user's turn)
    # Add the latest user message
    messages.append({"role": "user", "content": history[-1]["content"]})

    try:
        result = await chat_completion(messages=messages)
        return (
            result["content"],
            result["usage"],
            result["cost_usd"],
        )
    except RateLimitError:
        logger.warning("OpenAI rate limit hit")
        return (
            "I'm receiving too many requests right now. Please try again in a moment.",
            None,
            None,
        )
    except APITimeoutError:
        logger.warning("OpenAI timeout")
        return (
            "The AI took too long to respond. Please try again.",
            None,
            None,
        )
    except APIConnectionError:
        logger.error("OpenAI connection error")
        return (
            "Could not connect to the AI service. Check your internet connection.",
            None,
            None,
        )
    except APIStatusError as e:
        logger.error(f"OpenAI API error: {e.status_code} {e.message}")
        if e.status_code == 401:
            return (
                "Invalid OpenAI API key. Set OPENAI_API_KEY in your .env file.",
                None,
                None,
            )
        return (
            f"OpenAI API error: {e.message}",
            None,
            None,
        )
    except ValueError as e:
        logger.error(f"OpenAI config error: {e}")
        return (
            str(e),
            None,
            None,
        )


def _fallback_response(message: str) -> str:
    """
    Response when OpenAI is not configured.
    Guides the user to add their API key.
    """
    return (
        f"[Milestone 4 — OpenAI not configured]\n\n"
        f"You asked: \"{message}\"\n\n"
        f"To enable real AI responses:\n"
        f"1. Get an OpenAI API key: https://platform.openai.com/api-keys\n"
        f"2. Open your .env file\n"
        f"3. Set: OPENAI_API_KEY=sk-your-key-here\n"
        f"4. Restart the backend server"
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get conversation history",
)
async def get_history(session_id: str) -> ChatHistoryResponse:
    """Returns all messages in a session with timestamps."""
    history = _get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatHistoryItem(
                session_id=session_id,
                role=MessageRole(msg["role"]),
                content=msg["content"],
                timestamp=msg.get("timestamp"),
            )
            for msg in history
        ],
    )


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
)
async def clear_history(session_id: str):
    """Clears all messages for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"message": f"History for session '{session_id}' cleared."}


@router.get("/sessions", summary="List active sessions (debug)")
async def list_sessions():
    """Returns all active session IDs and message counts."""
    return {
        "session_count": len(_sessions),
        "sessions": [
            {"session_id": sid, "message_count": len(msgs)}
            for sid, msgs in _sessions.items()
        ],
    }
