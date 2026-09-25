"""
services/llm_service.py — Thin wrapper around the OpenAI client.

WHY SYNC OpenAI (not AsyncOpenAI)?
  LangGraph human-in-the-loop uses sync graph.stream() and wraps async
  nodes with a short-lived event loop. AsyncOpenAI/httpx then logs:
    RuntimeError: Event loop is closed
  when cleaning up after the loop is closed — even though the LLM call
  already succeeded. Sync OpenAI avoids that noise entirely.
"""

import logging
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def get_llm() -> OpenAI:
    """Return a configured OpenAI client (works with internal gateways)."""
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
    return OpenAI(**kwargs)


async def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Plain text chat completion (async signature, sync HTTP under the hood)."""
    client = get_llm()
    m = model or settings.openai_chat_model
    logger.debug("LLM call: model=%s, messages=%d", m, len(messages))

    response = client.chat.completions.create(
        model=m,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def structured_chat(
    messages: list[dict],
    response_format: Type[T],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> T:
    """Structured output chat completion using Pydantic."""
    client = get_llm()
    m = model or settings.openai_chat_model
    logger.debug("Structured LLM call: model=%s, schema=%s", m, response_format.__name__)

    try:
        response = client.beta.chat.completions.parse(
            model=m,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned null structured output")
        return parsed
    except Exception as e:
        logger.error("Structured chat failed: %s", e)
        raise


async def chat_with_usage(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, int, int]:
    """
    Like chat() but also returns (content, input_tokens, output_tokens).
    Used by nodes that need to report token consumption to Langfuse.

    USAGE:
      content, in_tok, out_tok = await llm_service.chat_with_usage(messages)
      tracer.update_generation_tokens(gen_ctx, in_tok, out_tok)
    """
    client = get_llm()
    m = model or settings.openai_chat_model
    response = client.chat.completions.create(
        model=m,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return content, input_tokens, output_tokens


async def structured_chat_with_usage(
    messages: list[dict],
    response_format: Type[T],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> tuple[T, int, int]:
    """
    Like structured_chat() but also returns (parsed_obj, input_tokens, output_tokens).
    """
    client = get_llm()
    m = model or settings.openai_chat_model
    try:
        response = client.beta.chat.completions.parse(
            model=m,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned null structured output")
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return parsed, input_tokens, output_tokens
    except Exception as e:
        logger.error("Structured chat with usage failed: %s", e)
        raise


async def embed(text: str) -> list[float]:
    """Embed a single string. Returns a 1536-dim vector."""
    client = get_llm()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding
