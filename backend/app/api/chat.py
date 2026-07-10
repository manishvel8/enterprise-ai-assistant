"""
chat.py — Chat API endpoints.

Milestone 2: Skeleton only — returns a hardcoded echo response.
Milestone 3: Adds real message handling.
Milestone 4: Integrates OpenAI GPT.
Milestone 22: Integrates full agentic workflow.

POST /api/chat        — Send a message, get a response
GET  /api/chat/history/{session_id} — Get conversation history
"""

import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

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

# --- In-memory session store (Milestone 2 placeholder) ---
# This will be replaced by PostgreSQL in Milestone 8.
# Key: session_id, Value: list of message dicts
_sessions: dict = {}


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="""
Send a message to the AI assistant.

**Milestone 2 behaviour:** Returns an echo response (repeats your message back).
**Milestone 3+:** Connects to real chat logic.
**Milestone 4+:** Connects to OpenAI GPT.
    """,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Workflow (full — Milestone 22):
    1. Create AgentState from the request
    2. Run Router Agent → determine intent
    3. Run Query Rewrite Agent
    4. Run Retriever Agent → get relevant chunks
    5. Run Cypher Agent (if graph intent)
    6. Run Context Builder Agent
    7. Run Answer Agent → call OpenAI
    8. Run Critic Agent → validate answer
    9. Run Memory Agent → save to session history
    10. Return ChatResponse with answer + citations + debug info

    Current behaviour (Milestone 2): Echo the user's message.
    """
    start_time = time.time()

    # --- Save user message to session history ---
    if request.session_id not in _sessions:
        _sessions[request.session_id] = []

    _sessions[request.session_id].append({
        "role": MessageRole.USER,
        "content": request.message,
    })

    # --- Milestone 2: Echo response ---
    # This will be replaced in Milestone 3 with real logic.
    echo_answer = (
        f"[ECHO - Milestone 2] You said: \"{request.message}\"\n\n"
        f"This is a skeleton response. OpenAI integration comes in Milestone 4."
    )

    # --- Save assistant response to session history ---
    _sessions[request.session_id].append({
        "role": MessageRole.ASSISTANT,
        "content": echo_answer,
    })

    latency_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        session_id=request.session_id,
        answer=echo_answer,
        citations=[],
        debug=DebugInfo(
            intent="general_chat",
            latency_ms=round(latency_ms, 2),
        ) if settings.enable_debug_panel else None,
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get conversation history for a session",
)
async def get_history(session_id: str) -> ChatHistoryResponse:
    """
    Returns all messages in a session.

    Milestone 2: Reads from in-memory store.
    Milestone 8+: Reads from PostgreSQL.
    """
    messages = _sessions.get(session_id, [])
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatHistoryItem(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
            )
            for msg in messages
        ],
    )


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history for a session",
)
async def clear_history(session_id: str):
    """Clear all messages for a session. Useful for 'new conversation' button."""
    _sessions.pop(session_id, None)
    return {"message": f"History for session '{session_id}' cleared."}
