"""
chat.py — Pydantic models for the chat API.

These are the request and response shapes validated by FastAPI automatically.
If the request body doesn't match ChatRequest, FastAPI returns 422 with details.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    """Who sent the message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """One message in the conversation."""
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """
    Body for POST /api/chat

    Example:
        {
          "user_id": "user_123",
          "session_id": "session_abc",
          "message": "What was the revenue in Q3?",
          "document_ids": ["doc_001", "doc_002"]
        }
    """
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Conversation session identifier")
    message: str = Field(..., min_length=1, max_length=4000, description="User's message")
    document_ids: List[str] = Field(default=[], description="Restrict retrieval to these documents")
    stream: bool = Field(default=True, description="If true, stream response via SSE")


class SourceCitation(BaseModel):
    """A source cited in the answer."""
    chunk_id: str
    document_id: str
    file_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    excerpt: str = Field(..., description="Short text snippet from the source")


class DebugInfo(BaseModel):
    """Debug metadata returned alongside the answer for the debug panel."""
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None
    retrieved_chunks_count: int = 0
    similarity_scores: List[float] = []
    cypher_query: Optional[str] = None
    cypher_results_count: int = 0
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    langfuse_trace_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
    Response for POST /api/chat (non-streaming mode).

    Streaming mode sends SSE chunks instead.
    """
    session_id: str
    answer: str
    citations: List[SourceCitation] = []
    debug: Optional[DebugInfo] = None


class ChatHistoryItem(BaseModel):
    """One turn in the conversation history."""
    session_id: str
    role: MessageRole
    content: str
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Response for GET /api/chat/history/{session_id}"""
    session_id: str
    messages: List[ChatHistoryItem] = []
