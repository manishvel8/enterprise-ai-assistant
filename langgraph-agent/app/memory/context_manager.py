"""
app/memory/context_manager.py  —  Context Window Management Strategies

─────────────────────────────────────────────────────────────────────────────
THE PROBLEM:
  GPT-4o has a 128K token context window.
  A conversation with 10,000 messages would have ~500,000 tokens.
  We cannot send all messages to the LLM on every request.

  Context window breakdown (typical):
    System prompt:         2,000 tokens
    Recent messages:      20,000 tokens
    Retrieved documents:  40,000 tokens
    Long-term memory:      5,000 tokens
    User query:            1,000 tokens
    Reserved for output:  60,000 tokens
    ─────────────────────────────────
    TOTAL:               128,000 tokens  ←  context window limit

  The challenge: intelligently select WHICH messages to include.

STRATEGIES IMPLEMENTED HERE:
  1. last_n_messages      — Simple: always take the last N messages
  2. sliding_window       — Token-budget-aware: take as many as fit
  3. summarize_old        — Compress old messages into a summary
  4. semantic_retrieve    — Fetch messages MOST RELEVANT to current query
  5. hybrid               — Combine: summary + recent + semantic (PRODUCTION)

WHICH TO USE IN PRODUCTION?
  For most applications: hybrid strategy
    - Recent 20 messages (immediate context)
    - Summarize messages 21-100
    - Semantic search for relevant older messages
    - Long-term memory from memory table

CONTEXT vs STORAGE LIMIT:
  Storage limit:       Can store unlimited messages (disk is cheap ~$0.10/GB)
  Context window limit: Can only SEND 128K tokens per request to LLM
  → You can STORE 10 years of history but only SEND the relevant parts
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")

# ── Model context window sizes ─────────────────────────────────────────────────
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o":              128_000,
    "gpt-4o-mini":         128_000,
    "gpt-4-turbo":         128_000,
    "gpt-4":                 8_192,
    "gpt-3.5-turbo":        16_385,
    "claude-3-5-sonnet":   200_000,
    "claude-3-opus":       200_000,
}

# ── Default token budgets per section ─────────────────────────────────────────
DEFAULT_BUDGETS = {
    "system_prompt":        2_000,
    "recent_messages":     20_000,
    "retrieved_documents": 40_000,
    "long_term_memory":     5_000,
    "user_query":           1_000,
    "output_reserved":     60_000,   # space left for the LLM to generate a response
}


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (matches GPT-4/3.5 tokenizer)."""
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return len(text) // 4


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens across a list of message dicts."""
    return sum(count_tokens(m.get("content", "")) for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: Last N Messages (simplest)
# ─────────────────────────────────────────────────────────────────────────────

def last_n_messages(messages: list[dict], n: int = 20) -> list[dict]:
    """
    Take the last N messages regardless of token count.

    WHEN TO USE:
      Quick development, simple applications, short conversations.

    RISK:
      If the last 20 messages are very long (code blocks, documents),
      this can still exceed the context window.

    EXAMPLE:
      Input: 1000 messages
      Output: last 20 messages
    """
    return messages[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Sliding Window (token-budget-aware)
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window(
    messages: list[dict],
    max_tokens: int = 20_000,
    min_messages: int = 3,       # always include at least 3 messages
) -> list[dict]:
    """
    Include as many recent messages as fit within the token budget.

    HOW IT WORKS:
      Start from the most recent message and keep adding older messages
      until we'd exceed the token budget. Always include at least
      min_messages to avoid cutting off critical context.

    EXAMPLE:
      max_tokens = 8000
      Messages (newest first): [user:3000, ai:3000, user:3000, ai:3000, ...]
      Result: last 2 pairs (6000 tokens < 8000 budget)

    WHY BETTER THAN last_n?
      Adapts to message length. Short messages → more history.
      Long messages (code, docs) → fewer messages but still fits budget.
    """
    if not messages:
        return []

    selected = []
    total_tokens = 0

    # Work backwards from most recent
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg.get("content", ""))
        if total_tokens + msg_tokens > max_tokens and len(selected) >= min_messages:
            break
        selected.insert(0, msg)     # prepend to maintain chronological order
        total_tokens += msg_tokens

    logger.debug(
        "sliding_window: %d/%d messages, %d tokens (budget: %d)",
        len(selected), len(messages), total_tokens, max_tokens
    )
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Summarization (compress old messages)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_old_messages(
    messages: list[dict],
    keep_recent_n: int = 20,
    llm_chat_fn=None,          # callable: (messages) → str
    existing_summary: str = "",
) -> tuple[str, list[dict]]:
    """
    Summarize old messages and keep only recent ones.

    RETURNS:
      (summary_text, recent_messages)

    HOW IT WORKS:
      Split messages into:
        old_messages   = messages[:-keep_recent_n]   ← summarize these
        recent_messages = messages[-keep_recent_n:]  ← keep these verbatim

      The summary is prepended to the prompt as a system message:
        "Previously in this conversation: [summary]"
        [recent messages follow]

    EXAMPLE PROMPT STRUCTURE:
      System: "You are a helpful assistant.
               Previously: User was asking about RAG. We discussed chunking..."
      User: "What is RAG?"         ← message[-20]
      AI: "RAG means..."           ← message[-19]
      ...
      User: "Now explain chunking" ← message[-1] (current)

    WHEN TO TRIGGER SUMMARIZATION:
      When total_tokens > 60% of context window.
      Or when message count > 100.
    """
    if len(messages) <= keep_recent_n:
        # Nothing to summarize
        return existing_summary, messages

    old_messages = messages[:-keep_recent_n]
    recent_messages = messages[-keep_recent_n:]

    if llm_chat_fn is None:
        # No LLM available for summarization — fallback to keyword summary
        topics = set()
        for m in old_messages:
            words = m.get("content", "").split()[:5]
            topics.update(w.lower() for w in words if len(w) > 4)
        summary = existing_summary + f"\n[Summary of {len(old_messages)} older messages covering: {', '.join(list(topics)[:10])}]"
        return summary.strip(), recent_messages

    # Build the summarization prompt
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in old_messages
    )
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Summarize the key information, facts, user preferences, "
                "and decisions from the following conversation into 3-5 sentences. "
                "Focus on what would be useful context for future messages."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous summary (if any): {existing_summary}\n\n"
                f"Conversation to summarize:\n{convo_text}"
            ),
        },
    ]

    try:
        new_summary = llm_chat_fn(summary_prompt)
        logger.info(
            "Summarized %d messages into %d tokens",
            len(old_messages), count_tokens(new_summary)
        )
        return new_summary, recent_messages
    except Exception as exc:
        logger.warning("Summarization failed: %s — using keyword fallback", exc)
        summary = f"[Could not summarize {len(old_messages)} older messages]"
        return summary, recent_messages


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4: Semantic Retrieval (most relevant messages)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_retrieve_messages(
    query: str,
    conversation_id: str,
    top_k: int = 5,
    threshold: float = 0.4,
    embed_fn=None,             # callable: (text) → list[float]
) -> list[dict]:
    """
    Find the most semantically relevant past messages for the current query.

    HOW IT WORKS:
      1. Embed the current query into a 1536-d vector
      2. Do cosine similarity search against all stored message embeddings
      3. Return top_k most similar messages

    EXAMPLE:
      Current query: "Explain chunking again but relate it to my previous RAG question"
      → Finds: message where user asked "What is RAG?" (high similarity)
      → Finds: message where AI explained chunking (high similarity)
      → Injects those into context even though they were 50 messages ago

    WHEN TO USE:
      When you have old conversations (>100 messages) and need to retrieve
      relevant historical context without loading all messages.

    PRODUCTION NOTE:
      Message embeddings are generated asynchronously after each message is saved
      (see memory_retrieval.py). The ivfflat index on messages.embedding handles
      millions of vectors efficiently.
    """
    from app.database.db_client import get_db

    db = get_db()
    if not db.is_enabled or embed_fn is None:
        return []

    try:
        query_embedding = embed_fn(query)
        rows = db.execute_vector_search(
            table="messages",
            embedding_col="embedding",
            query_embedding=query_embedding,
            filter_col="conversation_id",
            filter_val=conversation_id,
            top_k=top_k,
            threshold=threshold,
            extra_cols="id, role, content, token_count, created_at, similarity",
        )
        logger.debug(
            "semantic_retrieve_messages: found %d relevant messages (query: %.40s...)",
            len(rows), query
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Semantic message retrieval failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5: Hybrid (RECOMMENDED for production)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    messages: list[dict],
    query: str,
    conversation_id: str,
    strategy: str = "hybrid",
    max_tokens: int = 25_000,
    keep_recent_n: int = 20,
    llm_chat_fn=None,
    embed_fn=None,
    existing_summary: str = "",
    long_term_memories: Optional[list[dict]] = None,
) -> dict:
    """
    Master function: builds the full context for the LLM prompt.

    STRATEGIES:
      "last_n"   → simple last N messages
      "sliding"  → token-budget sliding window
      "summary"  → summarize old + keep recent
      "semantic" → semantic retrieval of relevant messages
      "hybrid"   → summary + recent + semantic + memory (RECOMMENDED)

    RETURNS a dict:
      {
          "messages":     [...],    # the message list to send to LLM
          "summary":      "...",    # conversation summary (if any)
          "token_count":  1234,     # total tokens used
          "strategy_used": "hybrid",
          "truncated":    True,     # was any truncation applied?
      }

    HYBRID STRATEGY (recommended):
      1. Keep recent 20 messages verbatim
      2. Summarize older messages (if >20)
      3. Semantic search for relevant messages from far past
      4. Inject long-term memory into system prompt
      5. Check total fits within budget; truncate if needed

    EXAMPLE PROMPT BUILT BY HYBRID:
      [system]: "You are a helpful AI.
                 Long-term memory: User prefers Python examples.
                 Conversation summary: User was learning about RAG and chunking..."
      [user]:  "What is RAG?"         ← from semantic retrieval (50 msgs ago)
      [ai]:    "RAG means..."
      [user]:  "Explain chunking"     ← recent message
      [ai]:    "Chunking works by..."
      [user]:  "Explain it again"     ← current query
    """
    result = {
        "messages": [],
        "summary": existing_summary,
        "token_count": 0,
        "strategy_used": strategy,
        "truncated": False,
        "semantic_messages": [],
    }

    if not messages:
        return result

    if strategy == "last_n":
        selected = last_n_messages(messages, n=keep_recent_n)
        result["messages"] = selected
        result["token_count"] = count_messages_tokens(selected)
        return result

    elif strategy == "sliding":
        selected = sliding_window(messages, max_tokens=max_tokens)
        result["messages"] = selected
        result["token_count"] = count_messages_tokens(selected)
        if len(selected) < len(messages):
            result["truncated"] = True
        return result

    elif strategy == "summary":
        summary, recent = summarize_old_messages(
            messages,
            keep_recent_n=keep_recent_n,
            llm_chat_fn=llm_chat_fn,
            existing_summary=existing_summary,
        )
        result["messages"] = recent
        result["summary"] = summary
        result["token_count"] = count_messages_tokens(recent) + count_tokens(summary)
        result["truncated"] = len(recent) < len(messages)
        return result

    elif strategy == "semantic":
        if embed_fn:
            semantic_msgs = semantic_retrieve_messages(
                query=query,
                conversation_id=conversation_id,
                top_k=5,
                embed_fn=embed_fn,
            )
            result["messages"] = semantic_msgs
            result["semantic_messages"] = semantic_msgs
            result["token_count"] = count_messages_tokens(semantic_msgs)
        else:
            recent = last_n_messages(messages, n=keep_recent_n)
            result["messages"] = recent
            result["token_count"] = count_messages_tokens(recent)
        return result

    else:  # hybrid (PRODUCTION DEFAULT)
        # Step 1: Get recent messages (verbatim)
        recent_budget = max_tokens // 2
        recent = sliding_window(messages, max_tokens=recent_budget, min_messages=5)

        # Step 2: Summarize anything older than recent
        old_messages = [m for m in messages if m not in recent]
        summary = existing_summary
        if old_messages and llm_chat_fn:
            summary, _ = summarize_old_messages(
                messages=old_messages,
                keep_recent_n=0,   # summarize everything old
                llm_chat_fn=llm_chat_fn,
                existing_summary=existing_summary,
            )

        # Step 3: Semantic retrieval for relevant historical messages
        semantic_msgs = []
        if embed_fn and old_messages:
            semantic_msgs = semantic_retrieve_messages(
                query=query,
                conversation_id=conversation_id,
                top_k=3,
                embed_fn=embed_fn,
            )
            # Deduplicate: remove semantic matches already in recent
            recent_ids = {m.get("id") for m in recent}
            semantic_msgs = [m for m in semantic_msgs if m.get("id") not in recent_ids]

        # Step 4: Combine semantic + recent (in chronological order)
        combined = semantic_msgs + recent
        # Sort by created_at if available
        combined.sort(key=lambda m: m.get("created_at") or "", reverse=False)

        # Step 5: Check total token budget
        total = count_messages_tokens(combined) + count_tokens(summary)
        if total > max_tokens:
            # Emergency fallback: just use sliding window
            combined = sliding_window(combined, max_tokens=max_tokens)
            result["truncated"] = True

        result["messages"] = combined
        result["summary"] = summary
        result["semantic_messages"] = semantic_msgs
        result["token_count"] = count_messages_tokens(combined) + count_tokens(summary)
        result["truncated"] = result["truncated"] or (len(combined) < len(messages))
        return result


def build_system_prompt_with_context(
    base_system_prompt: str,
    conversation_summary: str = "",
    long_term_memories: Optional[list[dict]] = None,
) -> str:
    """
    Build the full system prompt including memory and conversation summary.

    STRUCTURE:
      [base instructions]
      [long-term memory section]
      [conversation summary section]

    EXAMPLE OUTPUT:
      "You are a helpful AI assistant specializing in ML engineering.

       Long-term memory about this user:
       - User prefers Python code examples over pseudocode
       - User is a senior backend engineer at TechCorp
       - User often asks about RAG and vector databases

       Previous conversation summary:
       Earlier we discussed RAG (Retrieval-Augmented Generation) and
       the user learned about chunking strategies for PDF documents."
    """
    parts = [base_system_prompt.strip()]

    # Add long-term memories
    if long_term_memories:
        memory_lines = []
        for m in long_term_memories[:10]:   # max 10 memories in prompt
            memory_lines.append(f"- {m.get('content', '')}")
        if memory_lines:
            parts.append(
                "\nLong-term memory about this user:\n" + "\n".join(memory_lines)
            )

    # Add conversation summary
    if conversation_summary and len(conversation_summary) > 20:
        parts.append(
            f"\nPrevious conversation summary:\n{conversation_summary}"
        )

    return "\n".join(parts)
