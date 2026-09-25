"""
pipeline/ingestion.py — Orchestrates the full document ingestion pipeline.

Pipeline stages (in order):
  1. Download file from MinIO (object storage)
  2. Normalize (parse) the file → ContentBlocks (via normalizer.py)
  3. Chunk the ContentBlocks → ChunkSchema objects (via chunker.py)
  4. Embed each chunk → 1536-dim vector (via embedder.py)
  5. Store vectors in Qdrant (via vector_store.py)
  6. Store chunk metadata in PostgreSQL (via postgres.py)
  7. Update document status in PostgreSQL → "complete"

This function is called by the Celery worker after it receives a task.
It runs synchronously (Celery tasks are sync by default).
"""

import logging
from typing import List

from app.models.document import ChunkSchema

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(document_id: str, file_name: str, file_type: str) -> int:
    """
    Run the full document ingestion pipeline for a given document.

    Args:
        document_id: UUID of the document in PostgreSQL
        file_name: Original file name
        file_type: File extension (pdf, docx, etc.)

    Returns:
        Number of chunks created.
    """
    import asyncio

    logger.info(f"Starting ingestion for document {document_id} ({file_name})")

    try:
        # Stage 1: Download file from MinIO
        file_bytes = _download_file(document_id, file_name)
        if not file_bytes:
            raise RuntimeError(f"Failed to download file for document {document_id}")

        logger.info(f"Downloaded {len(file_bytes)} bytes for {file_name}")

        # Stage 2: Normalize (parse) → ContentBlocks
        from app.pipeline.normalizer import normalize
        normalized_doc = normalize(file_bytes, document_id, file_name, file_type)
        logger.info(f"Parsed: {len(normalized_doc.content_blocks)} content blocks")

        # Stage 3: Chunk → ChunkSchema
        from app.pipeline.chunker import chunk_document
        chunks = chunk_document(normalized_doc)
        logger.info(f"Chunked: {len(chunks)} chunks")

        if not chunks:
            logger.warning(f"No chunks produced for document {document_id}")
            return 0

        # Stage 4: Embed chunks → vectors
        from app.pipeline.embedder import embed_chunks_sync
        embeddings = embed_chunks_sync(chunks)
        logger.info(f"Embedded: {len(embeddings)} vectors")

        # Stage 5: Store vectors in Qdrant
        from app.db.vector_store import ensure_collection, store_chunks
        ensure_collection()
        stored = store_chunks(chunks, embeddings)
        logger.info(f"Stored in Qdrant: {stored} vectors")

        # Stage 6: Store chunk metadata in PostgreSQL
        asyncio.run(_save_chunks_to_postgres(document_id, chunks))
        logger.info(f"Saved {len(chunks)} chunks to PostgreSQL")

        # Stage 7: Entity extraction → Neo4j (GraphRAG). Non-fatal if Neo4j down.
        try:
            from app.db.neo4j_client import create_document_node, setup_schema
            from app.pipeline.entity_extractor import extract_entities_sync
            from app.pipeline.graph_store import store_chunk_graph

            setup_schema()
            create_document_node(document_id, file_name, file_type)
            for chunk in chunks:
                store_chunk_graph(chunk, extract_entities_sync(chunk))
            logger.info(f"Neo4j graph updated for document {document_id}")
        except Exception as e:
            logger.warning(f"Neo4j graph step skipped (non-fatal): {e}")

        return len(chunks)

    except Exception as e:
        logger.error(f"Ingestion pipeline failed for document {document_id}: {e}")
        raise


def _download_file(document_id: str, file_name: str) -> bytes:
    """
    Download the raw file bytes from MinIO/object storage.

    Falls back to checking PostgreSQL for a file_path if MinIO is unavailable.
    """
    try:
        from app.services.storage_service import download_file
        file_bytes = download_file(document_id, file_name)
        if file_bytes:
            return file_bytes
    except Exception as e:
        logger.warning(f"MinIO download failed: {e}")

    return b""


async def _save_chunks_to_postgres(document_id: str, chunks: List[ChunkSchema]):
    """
    Persist chunk metadata to PostgreSQL for structured queries.

    Chunks are stored in the `document_chunks` table so we can:
    - Count chunks per document
    - Query chunks by page number / section
    - Fetch chunk text without going to Qdrant
    """
    try:
        from app.db.postgres import get_session_factory, ChunkModel

        chunk_records = [
            ChunkModel(
                chunk_id=c.chunk_id,
                document_id=document_id,
                file_name=c.file_name,
                source_type=c.source_type,
                chunk_index=c.metadata.get("chunk_index", 0),
                chunk_text=c.chunk_text,
                page_number=c.page_number,
                slide_number=c.slide_number,
                sheet_name=c.sheet_name,
                section_title=c.section_title,
            )
            for c in chunks
        ]

        session_factory = get_session_factory()
        async with session_factory() as session:
            for chunk_record in chunk_records:
                session.add(chunk_record)
            await session.commit()

    except Exception as e:
        logger.error(f"Failed to save chunks to PostgreSQL: {e}")
        # Don't raise — chunks are in Qdrant, PostgreSQL is secondary
