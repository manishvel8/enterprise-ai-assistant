"""
openai_service.py — Wrapper for all OpenAI API calls.

Why a wrapper instead of calling OpenAI directly?
  1. One place to configure the API key, model names, and retry logic
  2. Easy to mock in tests
  3. One place to add logging and Langfuse tracing later
  4. Easy to swap models (e.g., GPT-4o → GPT-4o mini for cheaper routing)
  5. One place to enforce token limits and handle rate limit errors

OpenAI SDK concepts:
  - AsyncOpenAI: async version of the client — required for FastAPI async endpoints
  - chat.completions.create: the main chat endpoint
  - messages: array of {role, content} dicts
    - "system": instructions for the assistant
    - "user": user's message
    - "assistant": previous assistant replies (for conversation memory)
  - stream=True: returns a generator of chunks instead of waiting for full response
  - usage: token counts returned in the response metadata
"""

import logging
from typing import List, Dict, AsyncGenerator, Optional
from openai import AsyncOpenAI
from openai import APIStatusError, RateLimitError, APITimeoutError, APIConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Single shared AsyncOpenAI client — created once at module import time.
# AsyncOpenAI manages a connection pool internally.
_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """
    Get or create the shared OpenAI async client.

    Why lazy initialization?
    - Avoids import-time errors if OPENAI_API_KEY is not set
    - Allows tests to mock the client before the first call
    """
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
        kwargs = {"api_key": settings.openai_api_key}
        # Support OpenAI-compatible gateways (Azure-style proxies, LiteLLM, vLLM, etc.)
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base.rstrip("/")
            logger.info(f"OpenAI client using custom base_url: {kwargs['base_url']}")
        _client = AsyncOpenAI(**kwargs)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — the personality and constraints of the AI assistant.
#
# Milestone 4: Basic system prompt.
# Milestone 16+: Updated to include retrieved context (RAG prompt).
# Milestone 22+: Each agent has its own specialized system prompt.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Enterprise AI Assistant. You help users analyze documents, answer questions, and extract insights.

Your capabilities grow with each milestone:
- Milestone 4 (current): General conversation using your training knowledge
- Milestone 16: Answering questions grounded in uploaded documents (RAG)
- Milestone 21: Traversing knowledge relationships via Neo4j GraphRAG
- Milestone 22: Multi-agent reasoning workflow

Guidelines:
- Be concise and precise
- If you don't know something, say so clearly
- Format code and data clearly using markdown
- Do not hallucinate facts about the user's documents until RAG is integrated"""


async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    system_prompt: Optional[str] = None,
) -> Dict:
    """
    Call the OpenAI chat completions API.

    Args:
        messages: Conversation history as [{role, content}] dicts.
                  Do NOT include the system message — this function adds it.
        model: Override the default model from settings (for cheap routing).
        temperature: 0.0 = deterministic, 1.0 = creative. Use 0.3 for factual tasks.
        max_tokens: Maximum tokens in the response.
        system_prompt: Override the default system prompt.

    Returns:
        {
            "content": str,            # The assistant's response text
            "model": str,              # Model that was used
            "usage": {                 # Token counts
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            },
            "cost_usd": float,         # Approximate cost in USD
        }

    Raises:
        RateLimitError: OpenAI rate limit hit — caller should retry with backoff
        APITimeoutError: Request timed out — caller should retry
        APIConnectionError: Network error — caller should retry
        APIStatusError: Other API error (invalid key, model not found, etc.)
    """
    client = get_openai_client()
    model = model or settings.openai_chat_model
    prompt = system_prompt or SYSTEM_PROMPT

    # Build the full messages array: system + conversation history
    full_messages = [{"role": "system", "content": prompt}] + messages

    logger.info(f"OpenAI chat call: model={model}, messages={len(full_messages)}, max_tokens={max_tokens}")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        cost = _estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        logger.info(
            f"OpenAI response: tokens={usage.total_tokens}, "
            f"cost=${cost:.5f}, finish={response.choices[0].finish_reason}"
        )

        return {
            "content": content,
            "model": model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "cost_usd": cost,
            "finish_reason": response.choices[0].finish_reason,
        }

    except RateLimitError as e:
        logger.error(f"OpenAI rate limit: {e}")
        raise
    except APITimeoutError as e:
        logger.error(f"OpenAI timeout: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise
    except APIStatusError as e:
        logger.error(f"OpenAI API error {e.status_code}: {e.message}")
        raise


async def chat_completion_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of chat_completion.

    Yields text chunks as they arrive from OpenAI instead of waiting
    for the complete response. This is what makes responses appear
    word-by-word in the frontend.

    Used in Milestone 24 when streaming SSE is added to the Angular frontend.
    """
    client = get_openai_client()
    model = model or settings.openai_chat_model
    prompt = system_prompt or SYSTEM_PROMPT

    full_messages = [{"role": "system", "content": prompt}] + messages

    async with client.chat.completions.stream(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


async def create_embedding(text: str) -> List[float]:
    """
    Create a vector embedding for the given text.

    Used in Milestone 13+ for embedding document chunks.
    The embedding is a list of 1536 floats (for text-embedding-3-small).

    Args:
        text: The text to embed. Should be a document chunk (500-1000 tokens).

    Returns:
        List of 1536 floats representing the semantic meaning of the text.
    """
    client = get_openai_client()

    # Truncate text if needed — embedding model has 8192 token limit
    # For safety, truncate at 6000 chars (~1500 tokens)
    if len(text) > 6000:
        text = text[:6000]
        logger.warning("Text truncated to 6000 chars for embedding")

    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an OpenAI API call in USD.

    Pricing as of 2024 (update when OpenAI changes prices):
    Source: https://openai.com/api/pricing/

    Note: This is an estimate. Actual costs may vary.
    """
    # Prices per 1M tokens
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    }

    rates = pricing.get(model, {"input": 2.50, "output": 10.00})
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 6)
