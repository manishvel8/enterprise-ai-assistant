"""
agents/router_agent.py — Router Agent: Classify user intent and route to the right sub-workflow.

Role: First agent in the workflow.
  Receives the raw user query and determines which path to take:
  - "doc_question"   → QueryRewrite → Retriever → ContextBuilder → Answer → Critic
  - "graph_question" → CypherAgent → ContextBuilder → Answer → Critic
  - "general_chat"   → Answer directly (no retrieval)
  - "summary"        → special summarization flow
"""

import logging
import re
from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

VALID_INTENTS = ["doc_question", "graph_question", "general_chat", "summary"]

# Prefer retrieval unless the user is clearly just chatting.
_DOC_HINTS = re.compile(
    r"\b("
    r"resume|cv|document|pdf|docx|uploaded|file|policy|report|"
    r"experience|skill|education|project|company|role|job|"
    r"who is|what is|summar|according to|from the|in the "
    r")\b",
    re.IGNORECASE,
)
_CHAT_ONLY = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good morning|good evening|how are you)[\s!.?]*$",
    re.IGNORECASE,
)

ROUTER_PROMPT = """Classify the user's question into exactly one of these intents:

- doc_question: ANY factual question that might be answered from uploaded documents
  (people, roles, skills, experience, policies, products, numbers, "what does X say").
  When unsure between doc_question and general_chat, choose doc_question.
- graph_question: Explicit relationship questions ("who worked with", "connected to", "related to").
- general_chat: ONLY greetings, thanks, or pure small-talk with no factual ask.
- summary: Explicit request to summarize a document or topic.

Respond with only the intent label (one word). No explanation.

User question: {query}
"""


async def run_router_agent(state: AgentState) -> AgentState:
    """
    Router Agent: classify user query intent.

    Reads from state: user_query
    Writes to state: intent
    """
    query = state.get("user_query", "")
    logger.info(f"Router Agent: classifying query: {query[:80]}")

    intent = await _classify_intent(query)
    intent = _apply_heuristics(query, intent)
    state["intent"] = intent

    logger.info(f"Router Agent: intent = {intent}")
    state["debug_info"]["router"] = {"intent": intent}

    return state


def _apply_heuristics(query: str, intent: str) -> str:
    """Correct common LLM misclassifications for RAG demos."""
    q = (query or "").strip()
    if _CHAT_ONLY.match(q):
        return "general_chat"
    if intent == "general_chat" and _DOC_HINTS.search(q):
        logger.info("Router heuristic: overriding general_chat → doc_question")
        return "doc_question"
    if intent not in VALID_INTENTS:
        return "doc_question"
    return intent


async def _classify_intent(query: str) -> str:
    """Use GPT-4o to classify the query intent. Falls back to 'doc_question' on error."""
    from app.services.openai_service import get_openai_client
    from app.core.config import settings

    if not settings.openai_api_key:
        return "doc_question"

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a document RAG system. "
                        "Prefer doc_question over general_chat whenever documents might help. "
                        "Reply with only the intent label."
                    ),
                },
                {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        intent = response.choices[0].message.content.strip().lower().replace("`", "")
        # Models sometimes return "Intent: doc_question"
        for valid in VALID_INTENTS:
            if valid in intent:
                return valid
        return "doc_question"

    except Exception as e:
        logger.error(f"Router Agent classification failed: {e}")
        return "doc_question"
