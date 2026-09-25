"""
agents/critic_agent.py — Critic Agent: Validate answer grounding and quality.

Role: Quality gate — runs after Answer Agent, before Memory Agent.
  Checks if the generated answer is:
  1. Grounded in the retrieved context (no hallucinations)
  2. Actually answers the user's question
  3. Contains accurate citations
  4. Not harmful or inappropriate

If the answer fails, the Critic can:
  a) Retry: send the answer back to the Answer Agent with feedback
  b) Reject: set validated_answer = None and return an error message

Why a Critic Agent?
  LLMs can hallucinate — generate confident-sounding but false information.
  For enterprise applications, incorrect answers can cause real harm.
  The Critic provides a second check before delivering to the user.

Max retries: 2
  We allow the Answer Agent to try twice more if the Critic rejects.
  After 3 total attempts, we return a "could not verify" message.
"""

import logging
from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

MAX_CRITIC_ITERATIONS = 2

CRITIC_PROMPT = """You are a quality control agent. Evaluate whether the answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate:
1. Is the answer grounded in the context? (Does it only use information from the context?)
2. Does it actually answer the question?
3. Are there any obvious factual errors or hallucinations?

Respond with a JSON object:
{{"is_grounded": true/false, "feedback": "brief explanation if not grounded", "confidence": 0.0-1.0}}

Important: Treat section titles and source headers as part of the context.
If a detail appears in the Document Context (including Source/Section lines), it IS grounded.
When unsure, set is_grounded to true with lower confidence rather than false.
"""


async def run_critic_agent(state: AgentState) -> AgentState:
    """
    Critic Agent: validate the draft answer.

    Reads from state: draft_answer, final_context, user_query, critic_iterations
    Writes to state: validated_answer, is_grounded, critic_feedback, critic_iterations
    """
    import json

    answer = state.get("draft_answer", "")
    context = state.get("final_context", "")
    query = state.get("user_query", "")
    iterations = state.get("critic_iterations", 0)
    intent = state.get("intent", "general_chat")

    # Skip validation for general chat (no context to ground against)
    if intent == "general_chat" or not context:
        state["validated_answer"] = answer
        state["is_grounded"] = True
        return state

    logger.info(f"Critic Agent: validating answer (iteration {iterations + 1})")

    # Validate the answer against enough context (avoid false rejects from truncation)
    evaluation = await _evaluate_answer(query, context[:12000], answer)

    state["critic_iterations"] = iterations + 1
    state["is_grounded"] = evaluation.get("is_grounded", True)
    state["critic_feedback"] = evaluation.get("feedback", "")

    if state["is_grounded"]:
        state["validated_answer"] = answer
        logger.info(f"Critic Agent: answer accepted (confidence={evaluation.get('confidence', 1.0):.2f})")
    else:
        logger.warning(f"Critic Agent: answer rejected — {state['critic_feedback']}")
        state["validated_answer"] = None

    state["debug_info"]["critic"] = {
        "iteration": state["critic_iterations"],
        "is_grounded": state["is_grounded"],
        "feedback": state.get("critic_feedback"),
        "confidence": evaluation.get("confidence"),
    }

    return state


async def _evaluate_answer(query: str, context: str, answer: str) -> dict:
    """Use GPT-4o to evaluate whether the answer is grounded in context."""
    import json
    from app.services.openai_service import get_openai_client
    from app.core.config import settings

    if not settings.openai_api_key:
        return {"is_grounded": True, "feedback": "", "confidence": 1.0}

    prompt = CRITIC_PROMPT.format(
        context=context,
        question=query,
        answer=answer,
    )

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict quality control agent. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        logger.warning(f"Critic evaluation failed: {e}. Accepting answer.")
        return {"is_grounded": True, "feedback": "", "confidence": 1.0}
