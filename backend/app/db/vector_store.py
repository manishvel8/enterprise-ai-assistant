"""
db/vector_store.py — Qdrant vector store client for storing and searching chunk embeddings.

Why Qdrant?
  - Open-source, self-hosted vector database
  - Excellent performance for high-dimensional nearest-neighbor search
  - Rich filtering: find similar chunks AND filter by document_id, source_type, etc.
  - gRPC + REST API
  - Runs as a Docker container (see docker-compose.yml)

How embeddings are stored:
  Each chunk → one Qdrant "point" with:
    - id: UUID derived from chunk_id
    - vector: 1536-dimensional float list from OpenAI
    - payload: all metadata (chunk_text, document_id, page_number, etc.)

Collection name: "chunks"
  We use a single collection for all documents. Filtering by document_id
  allows per-document scoped searches.

Distance metric: Cosine
  Cosine similarity measures the angle between two vectors, ignoring magnitude.
  Standard for text embeddings — a chunk about "ML" and a query "machine learning"
  will have high cosine similarity even if they use different exact words.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any

from app.models.document import ChunkSchema

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"
VECTOR_SIZE = 1536
DISTANCE = "Cosine"


def _get_qdrant_client():
    """
    Return a Qdrant client connected to the configured host.

    Returns None if Qdrant is not configured or unavailable.
    """
    try:
        from qdrant_client import QdrantClient
        from app.core.config import settings

        host = settings.qdrant_host
        port = settings.qdrant_port

        client = QdrantClient(host=host, port=port, timeout=10)
        return client
    except ImportError:
        logger.error("qdrant-client not installed. Run: pip install qdrant-client")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return None


def ensure_collection():
    """
    Create the 'chunks' collection if it doesn't exist.

    Called once at application startup or before first insert.
    Idempotent — safe to call multiple times.
    """
    client = _get_qdrant_client()
    if not client:
        return False

    try:
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in client.get_collections().collections]

        if COLLECTION_NAME not in collections:
            try:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
            except Exception as create_err:
                # Concurrent workers may race; treat "already exists" as success
                msg = str(create_err).lower()
                if "already exists" in msg or "409" in msg:
                    logger.info(f"Qdrant collection already exists: {COLLECTION_NAME}")
                else:
                    raise
        else:
            logger.info(f"Qdrant collection already exists: {COLLECTION_NAME}")

        return True
    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collection: {e}")
        return False


def store_chunks(chunks: List[ChunkSchema], embeddings: List[List[float]]) -> int:
    """
    Upsert chunk embeddings into Qdrant.

    Args:
        chunks: List of ChunkSchema objects (from chunker)
        embeddings: Parallel list of 1536-dim vectors (from embedder)

    Returns:
        Number of chunks stored (0 on failure).
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        logger.error(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        return 0

    client = _get_qdrant_client()
    if not client:
        return 0

    try:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, vector in zip(chunks, embeddings):
            # Use a deterministic UUID from chunk_id so re-processing is idempotent
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "file_name": chunk.file_name,
                "source_type": chunk.source_type,
                "chunk_text": chunk.chunk_text,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
                "slide_number": chunk.slide_number,
                "sheet_name": chunk.sheet_name,
                "timestamp_start": chunk.timestamp_start,
                "chunk_index": chunk.metadata.get("chunk_index"),
                "token_count": chunk.metadata.get("token_count"),
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        # Upsert in batches of 100
        batch_size = 100
        total_stored = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_stored += len(batch)
            logger.info(f"Stored batch {i // batch_size + 1}: {len(batch)} vectors")

        logger.info(f"Total stored in Qdrant: {total_stored} vectors")
        return total_stored

    except Exception as e:
        logger.error(f"Failed to store chunks in Qdrant: {e}")
        return 0


def delete_document_vectors(document_id: str) -> int:
    """
    Delete all vectors belonging to a specific document.

    Used when a document is deleted by the user.
    Filters by `document_id` field in the payload.

    Returns:
        Number of vectors deleted.
    """
    client = _get_qdrant_client()
    if not client:
        return 0

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

        logger.info(f"Deleted vectors for document {document_id}")
        return 1  # Qdrant returns operation info, not a count

    except Exception as e:
        logger.error(f"Failed to delete vectors for document {document_id}: {e}")
        return 0


def get_collection_info() -> Optional[Dict[str, Any]]:
    """Return collection stats for health checks and debug panel."""
    client = _get_qdrant_client()
    if not client:
        return None

    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return None
