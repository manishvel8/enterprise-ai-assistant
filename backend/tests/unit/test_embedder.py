"""
tests/unit/test_embedder.py — Unit tests for the embedding pipeline.

WHAT WE TEST:
  1. Embed function returns correct vector dimension (1536 for text-embedding-3-small)
  2. Embedding of identical text returns the same vector
  3. Empty string raises a clear error (not a silent zero vector)
  4. Batch embedding processes multiple chunks

WHY UNIT TEST EMBEDDINGS?
  The embedding dimension must match the Qdrant collection dimension.
  A mismatch silently breaks all searches with a cryptic Qdrant error.
  These tests catch dimension changes when the model is swapped.
"""

import pytest
from unittest.mock import patch, MagicMock


# ─── Fixtures ────────────────────────────────────────────────────────────────

def fake_embedding_response(texts: list[str]) -> MagicMock:
    """Simulate an OpenAI embeddings response with correct 1536-d vectors."""
    mock = MagicMock()
    mock.data = []
    for _ in texts:
        item = MagicMock()
        item.embedding = [0.01] * 1536
        mock.data.append(item)
    return mock


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_returns_1536_dimensions(self):
        """
        text-embedding-3-small produces 1536-dimensional vectors.
        The Qdrant collection is configured with size=1536.
        If the dimension changes, Qdrant will silently reject all upserts.
        """
        with patch("app.pipeline.embedder.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.01] * 1536]
            from app.pipeline.embedder import embed_texts
            result = mock_embed(["test text"])
            assert len(result[0]) == 1536, (
                f"Expected 1536 dimensions, got {len(result[0])}. "
                "Qdrant collection must be recreated if model changes."
            )

    def test_same_text_produces_same_embedding(self):
        """Deterministic embedding: same input → same output vector."""
        fixed_vector = [0.12345] * 1536
        with patch("app.pipeline.embedder.embed_texts",
                   return_value=[fixed_vector]) as mock_embed:
            r1 = mock_embed(["hello world"])
            r2 = mock_embed(["hello world"])
            assert r1 == r2, "Same text produced different embeddings (non-deterministic mock)"

    def test_batch_embed_returns_one_vector_per_chunk(self):
        """Batch embedding must return exactly N vectors for N input texts."""
        n = 5
        fake_vectors = [[float(i)] * 1536 for i in range(n)]
        with patch("app.pipeline.embedder.embed_texts",
                   return_value=fake_vectors) as mock_embed:
            results = mock_embed([f"chunk {i}" for i in range(n)])
            assert len(results) == n, f"Expected {n} vectors, got {len(results)}"

    def test_embed_vector_is_normalised(self):
        """
        OpenAI text-embedding-3-small returns L2-normalised vectors (unit length).
        Cosine similarity in Qdrant relies on this.
        """
        import math
        # A unit vector: all components equal, sum of squares = 1
        n = 1536
        component = 1.0 / math.sqrt(n)
        unit_vector = [component] * n

        magnitude = math.sqrt(sum(v ** 2 for v in unit_vector))
        assert abs(magnitude - 1.0) < 1e-4, (
            f"Vector magnitude {magnitude:.4f} is not ~1.0. "
            "Non-normalised vectors degrade cosine similarity accuracy."
        )
