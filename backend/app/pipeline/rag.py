"""
pipeline/rag.py — RAG answer generation with source citations.

RAG = Retrieval Augmented Generation

The idea:
  1. Embed the user's query (semantic search vector)
  2. Retrieve top-k semantically similar chunks from Qdrant
  3. Build a context string from the retrieved chunks
  4. Pass the context + query to GPT-4o to generate a grounded answer
  5. Return the answer with citations linking back to source chunks

Why RAG instead of fine-tuning?
  Fine-tuning bakes knowledge into model weights at training time.
  You can't update it without expensive retraining.
  RAG retrieves fresh knowledge at inference time from your document store.
  You can add new documents and they're immediately searchable.

Citation format:
  Each citation points to the exact chunk that the answer used.
  The UI displays citations as clickable footnotes: [1], [2], etc.

Context window management:
  We limit context to the top-k chunks (default 5) and truncate long chunks
  to stay within the GPT-4o context window limit.
"""

import logging
from typing import List, Tuple, Optional

from app.models.agent_state import RetrievedChunk, Citation

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000   # ~3000 tokens, leaves room for prompt + response


RAG_SYSTEM_PROMPT = """You are an enterprise AI assistant that answers questions based on provided document context.

Rules:
1. Answer ONLY based on the provided context. Do not use knowledge outside of it.
2. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this."
3. Always cite your sources using [1], [2], etc. referring to the numbered context sections.
4. Be precise and concise. Do not pad with unnecessary phrases.
5. For technical questions, include specific numbers, names, and details from the context.
"""


async def generate_rag_answer(
    query: str,
    chunks: List[RetrievedChunk],
    conversation_history: Optional[List[dict]] = None,
) -> Tuple[str, List[Citation]]:
    """
    Generate an answer using retrieved chunks as context.

    Args:
        query: The user's original (or rewritten) query
        chunks: Retrieved chunks from vector search
        conversation_history: Previous messages in the conversation

    Returns:
        (answer_text, citations) tuple
    """
    from app.services.openai_service import get_openai_client, chat_completion
    from app.core.config import settings

    if not chunks:
        return (
            "I couldn't find relevant information in the uploaded documents to answer your question.",
            [],
        )

    # Build numbered context from chunks
    context, citations = _build_context(chunks)

    # Build prompt messages
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    # Include recent conversation history (last 6 messages = 3 turns)
    if conversation_history:
        messages.extend(conversation_history[-6:])

    # Add the RAG context + user query
    messages.append({
        "role": "user",
        "content": f"""Context from documents:

{context}

---
Question: {query}

Please answer based on the context above and cite your sources."""
    })

    try:
        result = await chat_completion(messages)
        answer = result["content"] if isinstance(result, dict) else result
        logger.info(f"RAG answer generated: {len(answer)} chars, {len(citations)} citations")
        return answer, citations

    except Exception as e:
        logger.error(f"RAG answer generation failed: {e}")
        return f"I encountered an error generating the answer: {e}", []


def _chunk_get(chunk, key, default=None):
    """Read a field from a RetrievedChunk TypedDict or object."""
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _build_context(chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
    """
    Build a numbered context string from retrieved chunks.

    Format:
        [1] Source: filename.pdf, Page 3, Section: Introduction
        Text content here...

        [2] Source: report.docx, Page 7
        Another chunk here...

    Also builds the Citation list for the response.
    """
    context_parts = []
    citations = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        file_name = _chunk_get(chunk, "file_name", "") or "unknown"
        page_number = _chunk_get(chunk, "page_number")
        slide_number = _chunk_get(chunk, "slide_number")
        sheet_name = _chunk_get(chunk, "sheet_name")
        section_title = _chunk_get(chunk, "section_title")
        chunk_text = _chunk_get(chunk, "chunk_text", "") or ""

        source_parts = [file_name]
        if page_number:
            source_parts.append(f"Page {page_number}")
        elif slide_number:
            source_parts.append(f"Slide {slide_number}")
        elif sheet_name:
            source_parts.append(f"Sheet: {sheet_name}")
        if section_title:
            source_parts.append(f"Section: {section_title}")

        source_desc = ", ".join(source_parts)

        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break

        if len(chunk_text) > remaining:
            chunk_text = chunk_text[:remaining] + "..."

        context_part = f"[{i}] Source: {source_desc}\n{chunk_text}"
        context_parts.append(context_part)
        total_chars += len(context_part)

        excerpt = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text

        citations.append(Citation(
            chunk_id=_chunk_get(chunk, "chunk_id", ""),
            document_id=_chunk_get(chunk, "document_id", ""),
            file_name=file_name,
            page_number=page_number,
            section_title=section_title,
            excerpt=excerpt,
        ))

    context = "\n\n".join(context_parts)
    return context, citations
