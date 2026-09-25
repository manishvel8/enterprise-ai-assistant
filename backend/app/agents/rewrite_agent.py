"""
agents/rewrite_agent.py — Query Rewrite Agent: Improve ambiguous or conversational queries.

Role: Runs after Router Agent for "doc_question" and "graph_question" intents.
  Converts the raw user query into a more specific, retrieval-optimized form.

Why rewrite queries?
  User queries are often:
  - Too vague: "tell me about the risks" → "What are the financial and operational risks described in the document?"
  - Conversational: "what about the CEO?" → "Who is the CEO and what is their background?"
  - Pronoun-dependent: "What did she say about this?" → "What did [person from previous message] say about [topic]?"

  Rewritten queries produce better embeddings that match document chunks more precisely.

Conversation context:
  We pass the last 4 messages of conversation history so the agent can
  resolve pronouns like "she", "it", "this", "that" correctly.
"""

import logging
from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """Given the conversation history and the latest user query,
rewrite the query to be more specific and suitable for document retrieval.

Rules:
1. Resolve all pronouns (she, he, it, they, this, that) using conversation context
2. Make implicit topics explicit
3. Keep the rewritten query concise (1-2 sentences)
4. If the query is already clear and specific, return it unchanged

Conversation history:
{history}

Original query: {query}

Rewritten query (no explanation, just the query):"""


async def run_rewrite_agent(state: AgentState) -> AgentState:
    """
    Query Rewrite Agent: rewrite the user's query for better retrieval.

    Reads from state: user_query, conversation_history
    Writes to state: rewritten_query
    """
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    logger.info(f"Rewrite Agent: original query: {query[:80]}")

    rewritten = await _rewrite_query(query, history)
    state["rewritten_query"] = rewritten

    logger.info(f"Rewrite Agent: rewritten query: {rewritten[:80]}")
    state["debug_info"]["rewrite"] = {
        "original": query,
        "rewritten": rewritten,
    }

    return state


async def _rewrite_query(query: str, history: list) -> str:
    """Rewrite the query using GPT-4o with conversation context."""
    from app.services.openai_service import get_openai_client
    from app.core.config import settings

    if not settings.openai_api_key:
        return query

    # Format last 4 messages for context
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-4:]
    ) if history else "No previous messages"

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a query rewriting assistant. Return only the rewritten query.",
                },
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(history=history_text, query=query),
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}. Using original query.")
        return query
