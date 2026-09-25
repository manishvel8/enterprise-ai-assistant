"""
app/memory/memory_retrieval.py  —  Long-term Semantic Memory

─────────────────────────────────────────────────────────────────────────────
SHORT-TERM vs LONG-TERM MEMORY:

  SHORT-TERM MEMORY (conversation history):
    - The recent message transcript from this conversation
    - Stored in messages table (session-scoped)
    - Loaded verbatim on every request (or via sliding window)
    - Limited by context window (~20-50 messages)
    - Stored in: PostgreSQL (messages table) + optional Redis cache

  LONG-TERM MEMORY (user knowledge):
    - Distilled facts about the user that persist across conversations
    - Examples: "User prefers Python", "User works at TechCorp on RAG projects"
    - Stored in: PostgreSQL memory table (with 1536-d embeddings)
    - Retrieved SEMANTICALLY (find memories relevant to current query)
    - NOT loaded verbatim — only relevant ones injected into system prompt

MEMORY FLOW:
  User sends query
  → embed query
  → search memory table for similar memories (pgvector cosine)
  → inject top 5 relevant memories into system prompt
  → LLM responds with memory context

MEMORY CREATION:
  Option 1 (explicit): "Remember that I prefer Python"
  Option 2 (extract):  After each conversation, LLM extracts facts
  Option 3 (manual):   Admin sets user preferences

EXAMPLE QUERY:
  User says: "Give me the same style dashboard we discussed last week."
  → embed("same style dashboard discussed last week")
  → cosine search: finds memory "User likes dark theme, grid layout dashboards"
  → inject into prompt: "Long-term memory: User prefers dark grid dashboards"
  → LLM generates the right style without re-asking
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Store memory
# ─────────────────────────────────────────────────────────────────────────────

def store_memory(
    user_id: str,
    content: str,
    memory_type: str = "fact",     # "preference" | "fact" | "instruction" | "summary" | "episodic"
    importance: float = 0.5,
    source_msg_id: Optional[str] = None,
    embed_fn=None,                 # callable: (text) → list[float]
    expires_at: Optional[datetime] = None,
) -> Optional[dict]:
    """
    Create a new long-term memory for a user.

    USAGE:
      store_memory(
          user_id="user_123",
          content="User prefers Python examples over pseudocode",
          memory_type="preference",
          importance=0.8,
          embed_fn=embed_text,
      )

    EMBEDDING GENERATION:
      The content is embedded into a 1536-d vector for semantic retrieval.
      If no embed_fn is provided, the memory is stored without an embedding
      (it will be retrievable by user_id filter but not by semantic search).

    IMPORTANCE SCORE:
      0.9-1.0 = critical (always include: user's name, job, explicit instructions)
      0.6-0.8 = important (preferences, past decisions)
      0.3-0.5 = informational (topics discussed)
      0.0-0.2 = low priority (casual mentions)
    """
    from app.database.db_client import get_db
    db = get_db()

    # Generate embedding
    embedding = None
    if embed_fn is not None:
        try:
            embedding = embed_fn(content)
        except Exception as exc:
            logger.warning("Failed to embed memory content: %s", exc)

    if not db.is_enabled:
        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
        }

    # Format embedding for PostgreSQL vector type
    embed_param = None
    if embedding is not None:
        embed_param = "[" + ",".join(str(x) for x in embedding) + "]"

    row = db.execute_returning(
        """
        INSERT INTO memory
            (user_id, content, memory_type, embedding, importance, source_msg_id, expires_at)
        VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
        RETURNING *
        """,
        (user_id, content, memory_type, embed_param, importance, source_msg_id, expires_at),
    )
    logger.debug("Stored memory for user %s: %.60s...", user_id, content)
    return dict(row) if row else None


def update_message_embedding(message_id: str, embedding: list[float]) -> None:
    """
    Update the embedding column of a saved message.

    WHY SEPARATE?
      Embedding generation takes 50-100ms (API call).
      We don't want to block the response while we embed each message.
      Instead: save message immediately, embed asynchronously.

      In production: use a Celery task or background thread for this.
    """
    from app.database.db_client import get_db
    db = get_db()

    if not db.is_enabled:
        return

    embed_str = "[" + ",".join(str(x) for x in embedding) + "]"
    db.execute(
        "UPDATE messages SET embedding = %s::vector WHERE id = %s",
        (embed_str, message_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve memory
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_relevant_memory(
    user_id: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.35,
    memory_types: Optional[list[str]] = None,
    embed_fn=None,
) -> list[dict]:
    """
    Find the most relevant long-term memories for the current query.

    EXAMPLE:
      query = "Give me the same style dashboard we discussed last week"
      → embed query
      → cosine search in memory table filtered by user_id
      → returns: [
            "User likes dark-themed grid dashboards",
            "User asked about dashboard design on 2026-09-14",
         ]

    HOW COSINE SIMILARITY WORKS:
      Each memory is a point in 1536-dimensional space.
      The query is also embedded to a point in that space.
      Cosine distance (0-2) measures angle between vectors.
      Cosine similarity = 1 - distance (closer = more similar).
      threshold=0.35 means: "only return if similarity > 35%"

    PRODUCTION NOTES:
      - The ivfflat index in schema.sql makes this fast even for millions of rows
      - For higher recall, use hnsw index instead of ivfflat (but slower inserts)
      - Consider caching popular memory sets in Redis (TTL = 5 minutes)
    """
    from app.database.db_client import get_db
    db = get_db()

    if not db.is_enabled or embed_fn is None:
        return get_user_memories_by_importance(user_id, limit=top_k)

    try:
        query_embedding = embed_fn(query)
    except Exception as exc:
        logger.warning("Failed to embed query for memory retrieval: %s", exc)
        return get_user_memories_by_importance(user_id, limit=top_k)

    # Build the filter condition
    type_filter = ""
    params_extra: tuple = ()
    if memory_types:
        placeholders = ",".join(["%s"] * len(memory_types))
        type_filter = f"AND memory_type IN ({placeholders})"
        params_extra = tuple(memory_types)

    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    from app.database.db_client import get_db as _get_db
    _db = _get_db()
    if not _db.is_enabled:
        return []

    query_sql = f"""
        SELECT id, content, memory_type, importance, access_count, created_at,
               1 - (embedding <=> %s::vector) AS similarity
        FROM memory
        WHERE user_id = %s
          AND (expires_at IS NULL OR expires_at > NOW())
          AND embedding IS NOT NULL
          AND 1 - (embedding <=> %s::vector) > %s
          {type_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = (vec_str, user_id, vec_str, threshold) + params_extra + (vec_str, top_k)
    rows = _db.fetchall(query_sql, params)

    # Increment access_count for retrieved memories
    if rows:
        ids = [str(r["id"]) for r in rows]
        try:
            _db.execute(
                "UPDATE memory SET access_count = access_count + 1 WHERE id = ANY(%s::uuid[])",
                (ids,),
            )
        except Exception:
            pass

    logger.debug(
        "retrieve_relevant_memory: found %d memories for user %s (query: %.40s...)",
        len(rows), user_id, query
    )
    return [dict(r) for r in rows]


def get_user_memories_by_importance(
    user_id: str,
    limit: int = 10,
    memory_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Return top memories for a user ordered by importance (no semantic search).
    Used as fallback when embedding is unavailable, or for system prompt injection.

    USAGE:
      Always inject the top 3 "instruction" type memories into every request
      (e.g. "Always respond in bullet points" set by the user).
    """
    from app.database.db_client import get_db
    db = get_db()

    if not db.is_enabled:
        return []

    type_clause = ""
    params: tuple = (user_id,)
    if memory_types:
        placeholders = ",".join(["%s"] * len(memory_types))
        type_clause = f"AND memory_type IN ({placeholders})"
        params += tuple(memory_types)

    params += (limit,)

    rows = db.fetchall(
        f"""
        SELECT id, content, memory_type, importance, created_at
        FROM memory
        WHERE user_id = %s
          AND (expires_at IS NULL OR expires_at > NOW())
          {type_clause}
        ORDER BY importance DESC, created_at DESC
        LIMIT %s
        """,
        params,
    )
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Memory extraction (automatic from conversation)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_store_memories(
    user_id: str,
    messages: list[dict],
    llm_chat_fn=None,
    embed_fn=None,
) -> list[dict]:
    """
    Use an LLM to extract memorable facts from a conversation and store them.

    WHEN TO CALL:
      After every N messages, or at the end of a session.

    WHAT IT EXTRACTS:
      - User preferences ("I prefer Python", "I like dark UI")
      - User facts ("I'm a senior engineer at TechCorp")
      - Decisions made ("We decided to use Qdrant for our vector DB")
      - Instructions ("Always give code examples")

    EXAMPLE:
      Input messages:
        User: "I'm building a RAG system and I prefer Python code"
        AI:   "Here's a Python RAG example..."
        User: "Actually I work with TypeScript mostly"
      Extracted memories:
        - "User is building a RAG system" (fact)
        - "User prefers TypeScript" (preference, overrides Python)

    PRODUCTION NOTE:
      Run this as a background Celery task to avoid blocking the response.
    """
    if llm_chat_fn is None or not messages:
        return []

    # Format conversation for extraction
    convo = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in messages[-20:]   # only look at recent messages
    )

    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Extract memorable facts from this conversation. "
                "Output JSON array: [{\"content\": \"...\", \"type\": \"preference|fact|instruction\", \"importance\": 0.0-1.0}]. "
                "Only include concrete, durable facts worth remembering. Max 5 items. "
                "If nothing memorable, return []."
            ),
        },
        {"role": "user", "content": convo},
    ]

    try:
        raw = llm_chat_fn(extraction_prompt)
        import json
        # Extract JSON from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        items = json.loads(raw[start:end])
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)
        return []

    stored = []
    for item in items[:5]:
        if not isinstance(item, dict) or "content" not in item:
            continue
        mem = store_memory(
            user_id=user_id,
            content=item.get("content", ""),
            memory_type=item.get("type", "fact"),
            importance=float(item.get("importance", 0.5)),
            embed_fn=embed_fn,
        )
        if mem:
            stored.append(mem)

    logger.info("Extracted and stored %d memories for user %s", len(stored), user_id)
    return stored


def inject_memory_into_prompt(
    base_prompt: str,
    memories: list[dict],
    max_memories: int = 8,
) -> str:
    """
    Append relevant memories to the system prompt.

    INPUT:
      base_prompt = "You are a helpful AI assistant..."
      memories = [
          {"content": "User prefers Python", "memory_type": "preference"},
          {"content": "User is building a RAG system", "memory_type": "fact"},
      ]

    OUTPUT:
      "You are a helpful AI assistant...

       Long-term memory about this user:
       - [Preference] User prefers Python
       - [Fact] User is building a RAG system"
    """
    if not memories:
        return base_prompt

    lines = []
    for m in memories[:max_memories]:
        mtype = m.get("memory_type", "fact").capitalize()
        content = m.get("content", "").strip()
        if content:
            lines.append(f"- [{mtype}] {content}")

    if not lines:
        return base_prompt

    memory_block = "\n\nLong-term memory about this user:\n" + "\n".join(lines)
    return base_prompt + memory_block
