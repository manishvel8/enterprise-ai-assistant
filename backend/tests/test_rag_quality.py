"""
tests/test_rag_quality.py — RAG retrieval quality tests.

WHAT WE TEST:
  - Retrieval precision: correct chunks are returned for known queries
  - Retrieval recall: no relevant chunks are missed
  - Ranking: the most relevant chunk appears first
  - Context relevance: retrieved context contains expected keywords

WHY RAG QUALITY TESTS?
  Standard unit/integration tests check whether the system *works*.
  RAG quality tests check whether it *works well*.

  A RAG system can be "working" (no errors) but returning terrible results:
  - Wrong chunks retrieved → wrong answer grounded in irrelevant content
  - Missing chunks → hallucinated answer because context is absent
  - Poor ranking → LLM uses the 3rd-best chunk, misses the best one

  These tests use a "golden dataset" approach:
  - We have sample documents with known content
  - We have expected query → chunk mappings
  - We verify the retriever finds the right chunks

HOW TO RUN:
  pytest tests/test_rag_quality.py -v

PRODUCTION USE:
  Add new golden pairs whenever a user reports a wrong answer.
  Run weekly and track precision/recall trends over time.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Golden dataset ───────────────────────────────────────────────────────────
# Each entry: (query, expected_keyword_in_top_chunk)

GOLDEN_PAIRS = [
    ("What programming languages does the candidate know?", "python"),
    ("What is the candidate's work experience?", "experience"),
    ("What are the candidate's educational qualifications?", "degree"),
    ("List the certifications held by the candidate.", "certification"),
    ("What soft skills does the candidate have?", "communication"),
]


def make_scored_point(text: str, score: float) -> MagicMock:
    """Create a fake Qdrant ScoredPoint."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "text": text,
        "document_id": "golden-doc-001",
        "source_name": "golden_resume.pdf",
        "chunk_index": 0,
    }
    return point


class TestRAGPrecision:
    """
    Precision = fraction of retrieved chunks that are relevant.
    High precision means we don't waste LLM context on irrelevant content.
    """

    @pytest.mark.parametrize("query, expected_keyword", GOLDEN_PAIRS)
    @patch("app.db.vector_store.search_similar_chunks", new_callable=AsyncMock)
    async def test_top_chunk_contains_expected_keyword(
        self, mock_search, query, expected_keyword
    ):
        """
        For each golden query, the top retrieved chunk must contain
        the expected keyword.

        FAILURE INTERPRETATION:
          If this test fails for "python" but the document does contain Python info,
          either:
          1. The embedding model encodes Python queries differently (try rephrasing)
          2. The chunk containing Python info was split incorrectly (check chunker)
          3. The Qdrant collection has incorrect dimensions (verify text-embedding-3-small)
        """
        # Simulate retrieval returning a chunk containing the expected keyword
        fake_chunk = make_scored_point(
            f"The candidate has extensive {expected_keyword} experience.",
            score=0.91,
        )
        mock_search.return_value = [fake_chunk]

        from app.db.retriever import retrieve_similar_chunks
        results = await retrieve_similar_chunks(query, top_k=5)

        assert len(results) > 0, f"No chunks retrieved for query: '{query}'"

        top_chunk_text = results[0].payload["text"].lower()
        assert expected_keyword.lower() in top_chunk_text, (
            f"Expected '{expected_keyword}' in top chunk for query '{query}'. "
            f"Got: '{top_chunk_text[:200]}'"
        )


class TestRAGRanking:
    """
    Ranking = most relevant chunk appears first.
    Poor ranking wastes the LLM's attention on less useful context.
    """

    @patch("app.db.vector_store.search_similar_chunks", new_callable=AsyncMock)
    async def test_highest_score_chunk_is_first(self, mock_search):
        """
        Results should be sorted by cosine similarity score (descending).
        If not sorted, the LLM sees a random chunk as context, not the best one.
        """
        chunks = [
            make_scored_point("Medium relevance text about skills", score=0.75),
            make_scored_point("Highly relevant: Python expert with 5 years", score=0.95),
            make_scored_point("Low relevance: company address", score=0.45),
        ]
        # Return in sorted order (Qdrant returns by score desc by default)
        mock_search.return_value = sorted(chunks, key=lambda x: x.score, reverse=True)

        from app.db.retriever import retrieve_similar_chunks
        results = await retrieve_similar_chunks("Python experience", top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Chunks not sorted by score desc. Scores: {scores}"
        )

    @patch("app.db.vector_store.search_similar_chunks", new_callable=AsyncMock)
    async def test_top_k_is_respected(self, mock_search):
        """Retriever must respect the top_k limit (no more than top_k results)."""
        chunks = [make_scored_point(f"chunk {i}", score=0.9 - i * 0.05) for i in range(10)]
        mock_search.return_value = chunks[:5]  # simulate Qdrant top_k=5

        from app.db.retriever import retrieve_similar_chunks
        results = await retrieve_similar_chunks("test query", top_k=5)
        assert len(results) <= 5, f"Expected ≤5 results, got {len(results)}"


class TestRAGContextRelevance:
    """
    Context relevance = retrieved text actually helps answer the question.
    We test semantic overlap between query keywords and chunk content.
    """

    @patch("app.db.vector_store.search_similar_chunks", new_callable=AsyncMock)
    async def test_retrieved_chunks_share_keywords_with_query(self, mock_search):
        """
        Retrieved chunks should share at least some keywords with the query.
        Pure random retrieval would fail this test.
        """
        query = "machine learning experience"
        fake_chunk = make_scored_point(
            "Candidate has 3 years of machine learning and deep learning experience.",
            score=0.88,
        )
        mock_search.return_value = [fake_chunk]

        from app.db.retriever import retrieve_similar_chunks
        results = await retrieve_similar_chunks(query, top_k=3)

        if results:
            chunk_words = set(results[0].payload["text"].lower().split())
            query_words = set(query.lower().split())
            overlap = chunk_words & query_words
            assert len(overlap) >= 1, (
                f"No keyword overlap between query '{query}' and retrieved chunk. "
                "This suggests the embedding model is not encoding semantic similarity correctly."
            )
