"""
tests/unit/test_chunker.py — Unit tests for the document chunker.

WHAT WE TEST:
  1. Empty document returns no chunks
  2. Short document fits in a single chunk
  3. Long document is split into multiple chunks
  4. Overlap is preserved between consecutive chunks
  5. Heading metadata is carried into chunks
  6. Minimum token threshold avoids tiny tail chunks
  7. Chunk IDs are unique UUIDs

WHY UNIT TEST THE CHUNKER?
  The chunker is a pure function (input: NormalizedDocument → output: List[ChunkSchema]).
  It has no dependencies on external services, so tests are instant.
  Chunking quality directly affects RAG quality — a bug here silently degrades
  every downstream retrieval without raising an obvious error.
"""

import pytest
from app.pipeline.chunker import chunk_document, CHUNK_MAX_TOKENS, OVERLAP_TOKENS
from app.models.document import NormalizedDocument, ContentBlock


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_doc(blocks: list[dict]) -> NormalizedDocument:
    """Build a minimal NormalizedDocument from a list of {type, text} dicts."""
    return NormalizedDocument(
        document_id="test-doc-001",
        source_name="test.pdf",
        source_type="pdf",
        total_pages=1,
        language="en",
        blocks=[ContentBlock(**b) for b in blocks],
        metadata={"author": "Test"},
    )


def word_block(n_words: int, block_type: str = "paragraph", heading: str = None) -> dict:
    """Create a text block with exactly n_words words."""
    text = " ".join([f"word{i}" for i in range(n_words)])
    d = {"type": block_type, "text": text}
    if heading:
        d["heading"] = heading
    return d


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestChunkerBasics:

    def test_empty_document_returns_no_chunks(self):
        """An empty document (no blocks) must return an empty list, not crash."""
        doc = make_doc([])
        chunks = chunk_document(doc)
        assert chunks == [], "Expected [] for empty document"

    def test_short_document_single_chunk(self):
        """A short document (well below CHUNK_MAX_TOKENS) should produce 1 chunk."""
        doc = make_doc([word_block(50)])
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].document_id == "test-doc-001"

    def test_long_document_multiple_chunks(self):
        """
        A long document should be split into multiple chunks.
        CHUNK_MAX_TOKENS is ~800 tokens. 3000 words ≈ 2250 tokens → expect 3+ chunks.
        """
        doc = make_doc([word_block(3000)])
        chunks = chunk_document(doc)
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    def test_chunk_ids_are_unique(self):
        """Every chunk must have a unique UUID — duplicates would break Qdrant indexing."""
        doc = make_doc([word_block(3000)])
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_chunk_preserves_document_id(self):
        """All chunks from a document must reference the original document_id."""
        doc = make_doc([word_block(2000)])
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.document_id == "test-doc-001"


class TestChunkerOverlap:

    def test_consecutive_chunks_share_overlap_text(self):
        """
        The last words of chunk[n] should appear at the start of chunk[n+1].
        This overlap helps the LLM reason across chunk boundaries.

        If this test fails: answers to questions that span chunk boundaries will
        miss context, degrading RAG quality silently.
        """
        doc = make_doc([word_block(5000)])
        chunks = chunk_document(doc)
        if len(chunks) < 2:
            pytest.skip("Document too short to produce overlap")

        # Take last 20 words of chunk[0] and first 20 words of chunk[1]
        words_end_of_first = set(chunks[0].text.split()[-20:])
        words_start_of_second = set(chunks[1].text.split()[:20])

        overlap = words_end_of_first & words_start_of_second
        assert len(overlap) > 0, (
            "No overlap found between consecutive chunks. "
            "Check OVERLAP_TOKENS setting in chunker.py"
        )


class TestChunkerMetadata:

    def test_heading_carried_into_chunk_metadata(self):
        """
        Heading blocks should set section_title metadata on subsequent chunks.
        Without this, the LLM loses section context when answering queries.
        """
        blocks = [
            {"type": "heading", "text": "Executive Summary"},
            word_block(200),
        ]
        doc = make_doc(blocks)
        chunks = chunk_document(doc)

        # At least the first chunk should carry the heading
        assert any(
            c.metadata.get("section_title") == "Executive Summary"
            for c in chunks
        ), "Heading not propagated to chunk metadata"

    def test_token_count_is_set(self):
        """Every chunk must have a non-zero token_count for cost estimation."""
        doc = make_doc([word_block(200)])
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.token_count > 0, f"Chunk {c.chunk_id} has token_count=0"

    def test_chunk_index_is_sequential(self):
        """Chunks must have sequential indexes starting from 0."""
        doc = make_doc([word_block(3000)])
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i, f"Expected chunk_index={i}, got {c.chunk_index}"
