"""
agents/answer_agent.py — Answer Agent: Generate the final answer using GPT-4o.

Role: Core generation agent — called after ContextBuilder (or directly for general_chat).
  Takes the assembled context and conversation history and produces an answer.

For doc_question / graph_question:
  - Uses the full RAG prompt with context + citations
  - Must ground answers in the provided context

For general_chat:
  - Uses conversation history only (no context)
  - Can answer from general knowledge

Streaming:
  The Answer Agent supports streaming (Milestone 24).
  When streaming=True, it yields tokens incrementally.
  The FastAPI endpoint uses SSE to push tokens to the frontend.
"""

import logging
from app.models.agent_state import AgentState, Citation

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = """You are an enterprise AI assistant with access to a document knowledge base via a separate retrieval path.
Be helpful, professional, and concise.
For greetings and small-talk, reply briefly.
Do NOT claim that you cannot access uploaded documents — document questions are handled by retrieval before this step.
If you truly have no context for a factual ask, say the user should ask about content from their uploaded files."""


RAG_SYSTEM_PROMPT = """You are an enterprise AI assistant that answers questions based on provided document context.

Rules:
1. Answer ONLY based on the provided context. Do not use knowledge outside of it.
2. If the context doesn't contain enough information, clearly say so.
3. Cite your sources using [1], [2], etc. referring to the numbered context sections.
4. Be precise and include specific numbers, names, and details from the context.
5. Keep answers concise and well-structured."""


async def run_answer_agent(state: AgentState) -> AgentState:
    """
    Answer Agent: generate the response to the user's query.

    Reads from state: user_query, final_context, intent, conversation_history, retrieved_chunks
    Writes to state: draft_answer, citations
    """
    from app.pipeline.rag import generate_rag_answer, _build_context

    query = state.get("user_query", "")
    intent = state.get("intent", "general_chat")
    context = state.get("final_context")
    history = state.get("conversation_history", [])
    chunks = state.get("retrieved_chunks", [])

    logger.info(f"Answer Agent: generating answer for intent={intent}")

    if intent == "general_chat" or not context:
        # No RAG context — general conversation
        answer = await _general_chat_answer(query, history)
        state["citations"] = []
    else:
        # RAG answer with citations
        answer, citations = await generate_rag_answer(
            query=query,
            chunks=chunks,
            conversation_history=history,
        )
        state["citations"] = citations

    state["draft_answer"] = answer
    logger.info(f"Answer Agent: generated {len(answer)} char answer")
    state["debug_info"]["answer"] = {
        "answer_length": len(answer),
        "citation_count": len(state.get("citations", [])),
        "intent": intent,
    }

    return state


async def _general_chat_answer(query: str, history: list) -> str:
    """Generate a conversational response without document context."""
    from app.services.openai_service import chat_completion

    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    try:
        result = await chat_completion(messages)
        return result["content"] if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"I encountered an error generating the answer: {e}"
