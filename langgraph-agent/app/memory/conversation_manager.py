"""
app/memory/conversation_manager.py  —  Conversation CRUD operations

─────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE DOES:
  Manages the lifecycle of users, conversations, sessions, and messages
  in PostgreSQL. Every time a user sends a message, this module:
    1. Gets or creates the user record
    2. Gets or creates the conversation
    3. Creates a new session (or continues the current one)
    4. Saves the user message
    5. Later saves the assistant response

ID RELATIONSHIPS:
  user_id  →  conversation_id  →  session_id  →  message_id

  user_123
  ├── conversation_001 ("RAG discussion")
  │     ├── session_A  (Monday 9am-10am)
  │     │     ├── message_1 (user: "What is RAG?")
  │     │     ├── message_2 (assistant: "RAG means...")
  │     │     └── message_3 (user: "How does chunking work?")
  │     └── session_B  (Tuesday 2pm-3pm)
  │           ├── message_4 (user: "Remind me what RAG is")
  │           └── message_5 (assistant: "Based on our conversation...")
  └── conversation_002 ("Fine-tuning help")
        └── session_C  (Wednesday)

MULTI-USER PRODUCTION NOTES:
  For 10 users:      Single Postgres instance is fine
  For 1,000 users:   Add connection pooling (PgBouncer)
  For 100K users:    Read replicas for SELECT queries
  For 1M+ users:     Partition messages table by user_id or created_at
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import tiktoken

from app.database.db_client import get_db

logger = logging.getLogger(__name__)

# ── Tokenizer for counting tokens ─────────────────────────────────────────────
# cl100k_base is used by GPT-4, GPT-3.5-turbo, and text-embedding-3-small
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base tokenizer."""
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────────
# User operations
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_user(
    external_id: str,
    name: str = "Anonymous",
    email: Optional[str] = None,
) -> dict:
    """
    Get an existing user by external_id, or create a new one.

    WHY external_id?
      Your auth system (Keycloak, Auth0, etc.) has its own user IDs.
      We store those as external_id and use our internal UUID for all
      database relationships. This decouples auth from data storage.

    IDEMPOTENT: Safe to call on every request (uses INSERT ON CONFLICT).
    """
    db = get_db()

    if not db.is_enabled:
        # Return a fake user dict when DB is not available (e.g. in tests)
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, external_id)),
            "external_id": external_id,
            "name": name,
            "email": email,
        }

    row = db.execute_returning(
        """
        INSERT INTO users (external_id, name, email)
        VALUES (%s, %s, %s)
        ON CONFLICT (external_id) DO UPDATE
          SET name = EXCLUDED.name,
              last_seen_at = NOW()
        RETURNING *
        """,
        (external_id, name, email),
    )
    return dict(row) if row else {}


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Fetch a user by their internal UUID."""
    db = get_db()
    row = db.fetchone("SELECT * FROM users WHERE id = %s", (user_id,))
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation operations
# ─────────────────────────────────────────────────────────────────────────────

def create_conversation(
    user_id: str,
    title: str = "New Conversation",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create a new conversation for a user.

    WHEN TO CREATE A NEW CONVERSATION vs CONTINUE EXISTING:
      New conversation: user starts a fresh topic, or clicks "New Chat"
      Continue existing: user sends a follow-up in the same chat window

    The title is usually set to the first user message (truncated to 60 chars).
    """
    db = get_db()

    if not db.is_enabled:
        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": title,
            "message_count": 0,
        }

    row = db.execute_returning(
        """
        INSERT INTO conversations (user_id, title, metadata)
        VALUES (%s, %s, %s)
        RETURNING *
        """,
        (user_id, title[:200], metadata or {}),
    )
    return dict(row) if row else {}


def get_conversation(conversation_id: str) -> Optional[dict]:
    """Fetch a conversation by ID."""
    db = get_db()
    row = db.fetchone(
        "SELECT * FROM conversations WHERE id = %s",
        (conversation_id,)
    )
    return dict(row) if row else None


def list_user_conversations(user_id: str, limit: int = 20) -> list[dict]:
    """
    Return a user's conversations sorted by most recently updated.
    Used to populate the conversation sidebar (like ChatGPT's left panel).
    """
    db = get_db()
    rows = db.fetchall(
        """
        SELECT id, title, message_count, total_tokens, is_archived,
               created_at, updated_at
        FROM conversations
        WHERE user_id = %s AND is_archived = FALSE
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    return [dict(r) for r in rows]


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Update the auto-generated title (usually after first AI response)."""
    db = get_db()
    db.execute(
        "UPDATE conversations SET title = %s WHERE id = %s",
        (title[:200], conversation_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session operations
# ─────────────────────────────────────────────────────────────────────────────

def create_session(
    conversation_id: str,
    user_id: str,
    thread_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create a new session (one browser visit / reconnection).

    thread_id is the LangGraph thread ID used for interrupt/resume.
    Storing it here allows resuming interrupted conversations.
    """
    db = get_db()

    if not db.is_enabled:
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "thread_id": thread_id or str(uuid.uuid4()),
        }

    row = db.execute_returning(
        """
        INSERT INTO sessions (conversation_id, user_id, thread_id, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING *
        """,
        (conversation_id, user_id, thread_id, metadata or {}),
    )
    return dict(row) if row else {}


def end_session(session_id: str) -> None:
    """Mark a session as ended (user closed the tab or timed out)."""
    db = get_db()
    db.execute(
        """
        UPDATE sessions
        SET is_active = FALSE, ended_at = NOW()
        WHERE id = %s
        """,
        (session_id,),
    )


def get_active_session(conversation_id: str) -> Optional[dict]:
    """Return the most recent active session for a conversation."""
    db = get_db()
    row = db.fetchone(
        """
        SELECT * FROM sessions
        WHERE conversation_id = %s AND is_active = TRUE
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (conversation_id,),
    )
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Message operations
# ─────────────────────────────────────────────────────────────────────────────

def save_message(
    conversation_id: str,
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    model_used: Optional[str] = None,
    latency_ms: Optional[int] = None,
    langfuse_trace_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Save one message (user or assistant) to the database.

    CALLED TWICE per request:
      1. Save user message BEFORE calling LLM (role="user")
      2. Save assistant response AFTER LLM returns (role="assistant")

    token_count is pre-computed here using tiktoken.
    The embedding is NOT populated here — it's done asynchronously
    (see memory_retrieval.py) to avoid slowing down the main request.
    """
    db = get_db()
    token_count = count_tokens(content)

    if not db.is_enabled:
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "token_count": token_count,
        }

    row = db.execute_returning(
        """
        INSERT INTO messages
            (conversation_id, session_id, user_id, role, content,
             token_count, model_used, latency_ms, langfuse_trace_id, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        (
            conversation_id, session_id, user_id, role, content,
            token_count, model_used, latency_ms, langfuse_trace_id,
            metadata or {},
        ),
    )
    # Also increment session message_count
    db.execute(
        "UPDATE sessions SET message_count = message_count + 1 WHERE id = %s",
        (session_id,),
    )
    return dict(row) if row else {}


def get_recent_messages(
    conversation_id: str,
    limit: int = 20,
    include_system: bool = False,
) -> list[dict]:
    """
    Fetch the last N messages from a conversation, oldest first.

    WHY OLDEST FIRST?
      When building the LLM prompt, we pass messages in chronological order:
        system_prompt → [older messages] → [recent messages] → current query
      Fetching in DESC order (newest first) then reversing gives oldest-first.

    PERFORMANCE:
      The index idx_messages_conversation_created handles this query efficiently
      even for millions of messages.
    """
    db = get_db()

    role_filter = "" if include_system else "AND role != 'system'"

    rows = db.fetchall(
        f"""
        SELECT id, role, content, token_count, created_at, metadata
        FROM messages
        WHERE conversation_id = %s {role_filter}
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (conversation_id, limit),
    )
    # Reverse to get chronological order (oldest first)
    return [dict(r) for r in reversed(rows)]


def get_conversation_token_count(conversation_id: str) -> int:
    """Return total tokens used in a conversation (from conversations table cache)."""
    db = get_db()
    row = db.fetchone(
        "SELECT total_tokens FROM conversations WHERE id = %s",
        (conversation_id,),
    )
    return row["total_tokens"] if row else 0


def format_messages_for_llm(messages: list[dict]) -> list[dict]:
    """
    Convert database message rows into the format expected by the LLM API.

    INPUT  (from DB): [{"role": "user", "content": "...", "token_count": 15, ...}]
    OUTPUT (for LLM): [{"role": "user", "content": "..."}]

    The LLM only needs role and content. All other fields are stripped.
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in ("user", "assistant", "system")
    ]
