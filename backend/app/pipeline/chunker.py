"""
pipeline/chunker.py — Metadata-aware chunking with token-based overlap.

Why chunk?
  LLM context windows are limited. A 50-page PDF might be 100,000 tokens,
  but GPT-4o can only see 128,000 tokens at once and embedding models are
  limited to 8192 tokens per call. We split documents into chunks of 500-1000
  tokens so we can embed each chunk separately and retrieve only the most
  relevant chunks for each user query.

Why overlap?
  A sentence that answers the user's question might span a chunk boundary.
  If chunk 1 ends at token 500 and chunk 2 starts at token 501, context
  that bridges both sides is lost. With 100-token overlap, chunk 2 starts at
  token 401 — so relevant sentences at the boundary appear in both chunks.

Why metadata-aware?
  Headings provide important context for the text below them. If we chunk
  naively by token count, a heading might end up in chunk N but the paragraph
  it describes is in chunk N+1 — losing the context. The chunker:
  1. Always starts a new chunk at a heading block
  2. Carries the most recent heading as `section_title` metadata on each chunk

Algorithm:
  1. Iterate ContentBlocks
  2. Track current heading (carried forward until next heading)
  3. Accumulate text until token count reaches CHUNK_MAX_TOKENS
  4. When limit is reached, emit a chunk and retain last OVERLAP_TOKENS
     tokens at the start of the next chunk

Token counting:
  We use tiktoken (same tokenizer as OpenAI models) for accurate token counts.
  cl100k_base encoding is used by GPT-4 and text-embedding-3-small.
"""

import logging
import uuid
from typing import List, Optional
from app.models.document import NormalizedDocument, ChunkSchema, ContentBlock

logger = logging.getLogger(__name__)

# ── Chunking config ───────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 800       # Target chunk size (stay below embedding model limit)
CHUNK_MIN_TOKENS = 50        # Don't emit a chunk smaller than this (avoids noise)
OVERLAP_TOKENS = 100         # Tokens to carry over from previous chunk
# ─────────────────────────────────────────────────────────────────────


def chunk_document(doc: NormalizedDocument) -> List[ChunkSchema]:
    """
    Split a NormalizedDocument into ChunkSchema objects.

    Each chunk includes:
    - chunk_text: the text content (500-1000 tokens)
    - section_title: the most recent heading (for context)
    - page_number / slide_number / sheet_name: source location
    - chunk_index: position within the document

    Returns:
        List of ChunkSchema ready for embedding and storage.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning("tiktoken not installed, falling back to word-count approximation")
        enc = None

    def count_tokens(text: str) -> int:
        if enc:
            return len(enc.encode(text))
        return len(text.split())   # ~0.75 tokens per word approximation

    def decode_tokens(token_ids: list) -> str:
        if enc:
            return enc.decode(token_ids)
        return " ".join(token_ids)  # fallback: tokens are words

    def encode_text(text: str) -> list:
        if enc:
            return enc.encode(text)
        return text.split()   # fallback: tokens are words

    chunks: List[ChunkSchema] = []
    chunk_index = 0

    current_heading: Optional[str] = None
    current_tokens: list = []    # token IDs (or words if fallback)
    current_meta: dict = {}      # metadata from current block accumulation

    def flush_chunk(overlap: bool = True):
        """Emit the current accumulated tokens as a ChunkSchema."""
        nonlocal current_tokens, chunk_index

        text = decode_tokens(current_tokens).strip()
        token_count = count_tokens(text)

        if token_count < CHUNK_MIN_TOKENS:
            # Too small to be useful on its own — keep accumulating
            return False

        chunk_id = f"{doc.document_id}_chunk_{chunk_index}"
        chunks.append(ChunkSchema(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            file_name=doc.file_name,
            source_type=doc.file_type,
            page_number=current_meta.get("page_number"),
            slide_number=current_meta.get("slide_number"),
            sheet_name=current_meta.get("sheet_name"),
            timestamp_start=current_meta.get("timestamp"),
            section_title=current_heading,
            chunk_text=text,
            metadata={
                "chunk_index": chunk_index,
                "token_count": token_count,
                **current_meta.get("extra", {}),
            },
        ))
        chunk_index += 1

        # Carry over overlap tokens into the next chunk
        if overlap and len(current_tokens) > OVERLAP_TOKENS:
            current_tokens = current_tokens[-OVERLAP_TOKENS:]
        else:
            current_tokens = []

        return True

    for block in doc.content_blocks:
        is_heading = block.type in ("heading", "title", "slide_title")

        if is_heading:
            # Flush current chunk before starting a new section
            if current_tokens:
                flush_chunk(overlap=False)   # no overlap at heading boundaries
            current_heading = block.text
            # Update metadata for the new section
            current_meta = _extract_meta(block)
            # Don't add heading text to current_tokens — it becomes section_title
            continue

        # Update metadata when block has position info
        if not current_meta or block.page_number is not None:
            current_meta = _extract_meta(block)

        # Encode block text to tokens
        block_tokens = encode_text(block.text)

        # Process block tokens, flushing when chunk size is reached
        i = 0
        while i < len(block_tokens):
            remaining_capacity = CHUNK_MAX_TOKENS - len(current_tokens)
            if remaining_capacity <= 0:
                # Flush current chunk
                flush_chunk(overlap=True)
                remaining_capacity = CHUNK_MAX_TOKENS

            take = min(remaining_capacity, len(block_tokens) - i)
            current_tokens.extend(block_tokens[i:i + take])
            i += take

            if len(current_tokens) >= CHUNK_MAX_TOKENS:
                flush_chunk(overlap=True)

    # Flush any remaining tokens
    if current_tokens:
        flush_chunk(overlap=False)

    logger.info(
        f"Chunked '{doc.file_name}': {len(doc.content_blocks)} blocks → {len(chunks)} chunks "
        f"[{CHUNK_MAX_TOKENS} max tokens, {OVERLAP_TOKENS} overlap]"
    )

    return chunks


def _extract_meta(block: ContentBlock) -> dict:
    """Extract position metadata from a ContentBlock."""
    return {
        "page_number": block.page_number,
        "slide_number": block.slide_number,
        "sheet_name": block.sheet_name,
        "timestamp": block.timestamp,
        "extra": {k: v for k, v in block.metadata.items() if k not in ("source",)},
    }
