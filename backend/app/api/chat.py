"""
chat.py — Chat API endpoints.

Milestone 3: Full chat handler with conversation history.
Milestone 4: OpenAI GPT integration replaces the echo response.
Milestone 22: Full agentic workflow.

POST /api/chat                         — Send a message, get a response
GET  /api/chat/history/{session_id}    — Get conversation history
DELETE /api/chat/history/{session_id}  — Clear conversation history
"""

import time
import uuid
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict

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

# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store — replaced by PostgreSQL in Milestone 8.
#
# Structure:
#   _sessions = {
#       "session_abc123": [
#           {"role": "user", "content": "Hello", "timestamp": "2024-01-01T12:00:00"},
#           {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T12:00:01"},
#       ]
#   }
# ─────────────────────────────────────────────────────────────────────────────
_sessions: Dict[str, List[Dict]] = {}


def _get_session_history(session_id: str) -> List[Dict]:
    """Return the conversation history for a session, creating it if it doesn't exist."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def _build_context_from_history(history: List[Dict]) -> str:
    """
    Build a human-readable conversation context from history.

    Used in Milestone 3 to give the echo response access to prior messages.
    In Milestone 4+, this is replaced by OpenAI's messages array format.
    """
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:    # use last 6 messages for context (3 turns)
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="""
Send a message and receive an AI response.

**Current milestone (M3):** Intelligent echo response with conversation context awareness.
**Milestone 4:** Real OpenAI GPT-4o response.
**Milestone 16+:** RAG-grounded response with source citations.
**Milestone 22+:** Full 9-agent workflow response.
    """,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat handler.

    Receives a user message, updates session history, generates a response,
    saves the response to history, and returns it.
    """
    start_time = time.time()

    # --- Load session history ---
    history = _get_session_history(request.session_id)

    # --- Save the user's message to history ---
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    history.append({
        "role": "user",
        "content": request.message,
        "timestamp": timestamp,
    })

    # --- Generate response ---
    # Milestone 3: Smart echo that references conversation context.
    # Milestone 4: Replace this block with OpenAI API call.
    response_text = _generate_echo_response(request.message, history[:-1])

    # --- Save assistant response to history ---
    history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    latency_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        session_id=request.session_id,
        answer=response_text,
        citations=[],
        debug=DebugInfo(
            intent="general_chat",
            retrieved_chunks_count=0,
            similarity_scores=[],
            cypher_results_count=0,
            latency_ms=round(latency_ms, 2),
        ) if settings.enable_debug_panel else None,
    )


def _generate_echo_response(user_message: str, prior_history: List[Dict]) -> str:
    """
    Milestone 3 response generator.

    Creates a context-aware echo response that demonstrates:
    - The message was received
    - The system knows conversation history length
    - The next milestone will add real OpenAI integration

    Replace this entire function in Milestone 4.
    """
    turn_count = len([h for h in prior_history if h["role"] == "user"])

    if turn_count == 0:
        context_note = "This is the start of our conversation."
    elif turn_count == 1:
        context_note = "This is turn 2 of our conversation."
    else:
        context_note = f"We've exchanged {turn_count} messages so far."

    # Check if user is asking for a specific feature that's coming
    lower_msg = user_message.lower()
    if any(word in lower_msg for word in ["document", "pdf", "upload", "file"]):
        feature_note = "\n\nDocument upload and RAG retrieval is coming in Milestone 7-16."
    elif any(word in lower_msg for word in ["graph", "neo4j", "relationship", "entity"]):
        feature_note = "\n\nNeo4j GraphRAG integration is coming in Milestone 17-21."
    elif any(word in lower_msg for word in ["agent", "workflow", "route"]):
        feature_note = "\n\nThe full 9-agent workflow is coming in Milestone 22."
    else:
        feature_note = "\n\nOpenAI GPT-4o integration is coming in Milestone 4."

    return (
        f"[Milestone 3 Echo] You asked: \"{user_message}\"\n\n"
        f"{context_note}{feature_note}"
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get conversation history",
)
async def get_history(session_id: str) -> ChatHistoryResponse:
    """
    Returns all messages in a session in chronological order.

    The Angular frontend calls this when switching between sessions
    to load the previous conversation.

    Milestone 3: In-memory store.
    Milestone 8+: PostgreSQL.
    """
    history = _get_session_history(session_id)
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
    """
    Clears all messages for a session.

    Called when the user clicks "New Chat" and the frontend wants to
    clear the server-side history for the old session.
    """
    if session_id in _sessions:
        del _sessions[session_id]
    return {"message": f"History for session '{session_id}' cleared."}


@router.get(
    "/sessions",
    summary="List all active sessions (debug only)",
)
async def list_sessions():
    """
    Returns all session IDs currently in memory.
    Useful for debugging during development.
    """
    return {
        "session_count": len(_sessions),
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(msgs),
            }
            for sid, msgs in _sessions.items()
        ],
    }
