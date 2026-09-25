"""
agents/memory_agent.py — Memory Agent: Update conversation history and session state.

Role: Final agent in the workflow.
  After the Critic validates the answer, the Memory Agent:
  1. Appends the current Q&A to the conversation history
  2. Saves the session message to PostgreSQL
  3. Prepares the final response (validated_answer + citations)

Why a Memory Agent?
  Conversation continuity requires remembering past messages.
  Without memory, every query is answered in isolation — the user can't say
  "Expand on your previous answer" or "What did you say about the CEO?"

  The Memory Agent maintains:
  - In-state history: List of {role, content} dicts passed to OpenAI
  - Database history: Messages stored in PostgreSQL for retrieval across sessions

Session persistence:
  The conversation_history in AgentState is used for the current request.
  It's populated at the start of each request from PostgreSQL.
  The Memory Agent appends to it and saves back.
"""

import logging
from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20   # Keep last 20 messages in state (10 turns)


async def run_memory_agent(state: AgentState) -> AgentState:
    """
    Memory Agent: save the conversation turn and update history.

    Reads from state: user_query, validated_answer/draft_answer, session_id, user_id
    Writes to state: conversation_history (updated)
    """
    query = state.get("user_query", "")
    answer = state.get("validated_answer") or state.get("draft_answer", "")
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")

    if not answer:
        logger.warning("Memory Agent: no answer to save")
        return state

    # Append to in-state conversation history
    history = state.get("conversation_history", [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    # Trim to keep only the last N messages
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    state["conversation_history"] = history

    # Persist to PostgreSQL
    await _save_message_to_db(session_id, user_id, query, answer)

    logger.info(f"Memory Agent: saved turn, history length = {len(history)}")
    state["debug_info"]["memory"] = {
        "history_length": len(history),
        "session_id": session_id,
    }

    return state


async def _save_message_to_db(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Save the Q&A pair to PostgreSQL message history."""
    try:
        from app.db.postgres import get_session_factory, insert_message, upsert_session
        session_factory = get_session_factory()
        async with session_factory() as db:
            await upsert_session(db, session_id, user_id or "anonymous")
            await insert_message(db, session_id, "user", user_message)
            await insert_message(db, session_id, "assistant", assistant_message)
            await db.commit()

    except Exception as e:
        logger.warning(f"Memory Agent: failed to save to PostgreSQL (non-fatal): {e}")
