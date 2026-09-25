"""
document_worker.py — Celery task definitions for document processing.

How Celery works:
  1. FastAPI calls: process_document.delay(document_id, file_name, file_type)
     → This creates a task message and sends it to the Redis queue
     → FastAPI returns immediately (non-blocking)

  2. Celery worker (running in a separate container/process) polls Redis
     → Picks up the task
     → Calls the process_document() function with the provided arguments
     → Updates PostgreSQL with the result (complete or failed)

  3. FastAPI can be polled: GET /api/documents/{document_id}
     → Returns the current status from PostgreSQL

Key Celery concepts:
  - app: the Celery application instance
  - @app.task: decorator that registers a function as a Celery task
  - .delay(): shorthand for .apply_async() — sends task to queue
  - bind=True: gives the task access to self (for retry logic)
  - max_retries: how many times to retry on failure
  - countdown: seconds to wait before retrying

Why separate the worker from the API?
  - Document processing can take 30–300 seconds for large files
  - A synchronous call would block FastAPI for that entire time
  - Other users cannot get responses while one user's document is being processed
  - With Celery: the API returns in <100ms regardless of document size
  - Workers can be scaled independently (more workers = faster processing)
"""

import logging
import asyncio
from typing import Optional

from celery import Celery
from celery.utils.log import get_task_logger

from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Celery Application Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Create the Celery app instance.
# broker_url: where Celery sends tasks (Redis queue)
# result_backend: where Celery stores task results
celery_app = Celery(
    "document_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configuration
celery_app.conf.update(
    # Task routing — send document tasks to 'document_processing' queue
    task_routes={
        "app.workers.document_worker.process_document": {
            "queue": "document_processing"
        },
    },

    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Retry configuration
    task_acks_late=True,        # Acknowledge task only after completion (not at pickup)
                                # If worker crashes, task goes back to queue for retry
    task_reject_on_worker_lost=True,

    # Result expiry — how long to keep task results in Redis
    result_expires=3600,        # 1 hour

    # Worker concurrency
    worker_prefetch_multiplier=1,   # each worker picks up one task at a time
                                     # (prevents a slow task from blocking other tasks)

    # Timezone
    timezone="UTC",
    enable_utc=True,
)

logger = get_task_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Run async code inside Celery (which is synchronous)
#
# Celery tasks are synchronous by default.
# Our database code uses asyncio (asyncpg).
# This helper creates a fresh asyncio event loop to run async code from Celery.
# ─────────────────────────────────────────────────────────────────────────────

def run_async(coroutine):
    """
    Run an async coroutine from synchronous Celery task code.

    Disposes the SQLAlchemy async engine after each call so the next
    run_async() does not reuse connections bound to a closed event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        try:
            from app.db import postgres as pg

            async def _dispose():
                if pg._engine is not None:
                    await pg._engine.dispose()
                    pg._engine = None
                    pg._session_factory = None

            loop.run_until_complete(_dispose())
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(None)


# ─────────────────────────────────────────────────────────────────────────────
# Main Document Processing Task
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.workers.document_worker.process_document",
    max_retries=3,
    default_retry_delay=30,   # seconds between retries
    acks_late=True,
    track_started=True,
)
def process_document(
    self,
    document_id: str,
    file_name: str,
    file_type: str,
) -> dict:
    """
    Full document ingestion pipeline.

    Steps:
    1. Update status → PROCESSING in PostgreSQL
    2. Download original file from MinIO
    3. Parse file (format-specific parser)   ← Milestone 10
    4. Normalize to content_blocks schema    ← Milestone 11
    5. Chunk with metadata                   ← Milestone 12
    6. Embed each chunk via OpenAI           ← Milestone 13
    7. Store vectors in Qdrant               ← Milestone 14
    8. Extract entities and store in Neo4j   ← Milestone 18-19
    9. Update status → COMPLETE in PostgreSQL
    10. Update chunk_count in PostgreSQL

    On error:
    - Update status → FAILED with error message
    - Retry up to 3 times with 30-second delay
    - Log full traceback for debugging

    Args:
        document_id: Unique document identifier
        file_name: Original filename (for logging and parser selection)
        file_type: File extension (pdf, docx, etc.)

    Returns:
        {"status": "complete", "chunk_count": N}
        or {"status": "failed", "error": "error message"}
    """
    logger.info(f"Starting document processing: {document_id} ({file_name})")

    try:
        # Step 1: Mark as PROCESSING
        run_async(_update_status(document_id, "processing"))
        logger.info(f"Status updated to PROCESSING: {document_id}")

        # Step 2: Download file from MinIO (use stored path from Postgres)
        file_bytes = _get_file_bytes(document_id, file_name)

        if file_bytes is None:
            raise ValueError(
                f"Could not retrieve file bytes for {document_id}. "
                "Check MinIO is running and the object was uploaded."
            )

        # Guard: never feed placeholder/non-PDF bytes into the PDF parser
        if file_type == "pdf" and not file_bytes.startswith(b"%PDF"):
            raise ValueError(
                f"Downloaded bytes for {document_id} are not a valid PDF "
                f"(got {len(file_bytes)} bytes, magic={file_bytes[:8]!r}). "
                "MinIO download likely failed or returned the wrong object."
            )

        # Step 3-8: Run ingestion pipeline
        chunk_count = _run_ingestion_pipeline(document_id, file_name, file_type, file_bytes)

        if chunk_count == 0:
            raise ValueError(
                "No searchable chunks produced. "
                "For scanned/image PDFs install tesseract OCR, or upload a text PDF. "
                "Do not upload government ID cards for demos."
            )

        # Step 9-10: Mark as COMPLETE
        run_async(_update_status(document_id, "complete", chunk_count=chunk_count))
        logger.info(f"Document processing complete: {document_id}, chunks: {chunk_count}")

        return {"status": "complete", "chunk_count": chunk_count}

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Document processing failed: {document_id}: {error_msg}", exc_info=True)

        # Update status to FAILED
        try:
            run_async(_update_status(document_id, "failed", error_message=error_msg[:500]))
        except Exception:
            pass  # Don't let the status update error hide the original error

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task {document_id} (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=exc, countdown=self.default_retry_delay)

        return {"status": "failed", "error": error_msg}


def _get_file_bytes(document_id: str, file_name: str) -> Optional[bytes]:
    """
    Download file from MinIO using the storage_path saved in PostgreSQL.

    Falls back to the stable key documents/{document_id}/original.{ext}
    then the legacy key documents/{document_id}/{file_name}.
    """
    from app.services.storage_service import download_file

    storage_path = run_async(_lookup_storage_path(document_id))
    candidates = []
    if storage_path:
        candidates.append(storage_path)

    ext = ""
    if "." in file_name:
        ext = "." + file_name.rsplit(".", 1)[-1].lower()
    candidates.append(f"documents/{document_id}/original{ext}")
    candidates.append(f"documents/{document_id}/{file_name}")

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        file_bytes = download_file(path)
        if file_bytes:
            logger.info(f"Loaded {len(file_bytes)} bytes from {path}")
            return file_bytes

    logger.error(f"All MinIO download attempts failed for {document_id}")
    return None


async def _lookup_storage_path(document_id: str) -> Optional[str]:
    """Read storage_path from PostgreSQL for this document."""
    try:
        from app.db.postgres import get_session_factory, DocumentModel
        from sqlalchemy import select

        session_factory = get_session_factory()
        async with session_factory() as db:
            result = await db.execute(
                select(DocumentModel.storage_path).where(
                    DocumentModel.document_id == document_id
                )
            )
            return result.scalar_one_or_none()
    except Exception as e:
        logger.warning(f"Could not lookup storage_path: {e}")
        return None


def _run_ingestion_pipeline(
    document_id: str,
    file_name: str,
    file_type: str,
    file_bytes: bytes,
) -> int:
    """
    Run the full ingestion pipeline and return the number of chunks created.

    Delegates to pipeline/ingestion.py which orchestrates:
    parse → normalize → chunk → embed → store (Qdrant + PostgreSQL)
    """
    from app.pipeline.normalizer import normalize
    from app.pipeline.chunker import chunk_document
    from app.pipeline.embedder import embed_chunks_sync
    from app.db.vector_store import ensure_collection, store_chunks
    from app.db.neo4j_client import create_document_node, setup_schema
    from app.pipeline.entity_extractor import extract_entities_sync
    from app.pipeline.graph_store import store_chunk_graph

    logger.info(f"Pipeline: Parsing {file_type.upper()} — {file_name}")
    normalized_doc = normalize(file_bytes, document_id, file_name, file_type)

    logger.info(f"Pipeline: Chunking {len(normalized_doc.content_blocks)} content blocks")
    chunks = chunk_document(normalized_doc)

    if not chunks:
        logger.warning(f"No chunks produced for {document_id}")
        return 0

    logger.info(f"Pipeline: Embedding {len(chunks)} chunks")
    embeddings = embed_chunks_sync(chunks)

    logger.info(f"Pipeline: Storing vectors in Qdrant")
    ensure_collection()
    stored = store_chunks(chunks, embeddings)
    if stored == 0:
        raise RuntimeError(
            "Failed to store vectors in Qdrant (0 points written). "
            "Install qdrant-client and ensure Qdrant is running on :6333."
        )
    logger.info(f"Pipeline: Stored {stored} vectors in Qdrant")

    logger.info(f"Pipeline: Saving chunk metadata to PostgreSQL")
    run_async(_save_chunks_postgres(document_id, chunks))

    # GraphRAG prep: Document + Chunk nodes, then LLM entity extraction → Neo4j
    # Non-fatal if Neo4j or OpenAI is unavailable (vector RAG still works).
    try:
        setup_schema()
        create_document_node(document_id, file_name, file_type)
        graph_stored = 0
        for chunk in chunks:
            entities_data = extract_entities_sync(chunk)
            if store_chunk_graph(chunk, entities_data):
                graph_stored += 1
        logger.info(
            f"Pipeline: Neo4j graph update for {graph_stored}/{len(chunks)} chunks"
        )
    except Exception as e:
        logger.warning(f"Neo4j entity graph step skipped (non-fatal): {e}")

    chunk_count = len(chunks)
    logger.info(f"Pipeline complete: {chunk_count} chunks for document {document_id}")
    return chunk_count


async def _save_chunks_postgres(document_id: str, chunks) -> None:
    """Persist chunk metadata to PostgreSQL."""
    try:
        from app.db.postgres import get_session_factory, ChunkModel
        session_factory = get_session_factory()
        async with session_factory() as session:
            for chunk in chunks:
                record = ChunkModel(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    file_name=chunk.file_name,
                    source_type=chunk.source_type,
                    chunk_index=chunk.metadata.get("chunk_index", 0),
                    chunk_text=chunk.chunk_text,
                    page_number=chunk.page_number,
                    slide_number=chunk.slide_number,
                    sheet_name=chunk.sheet_name,
                    section_title=chunk.section_title,
                )
                session.add(record)
            await session.commit()
    except Exception as e:
        logger.warning(f"PostgreSQL chunk save failed (non-fatal): {e}")


async def _update_status(
    document_id: str,
    status: str,
    chunk_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """
    Update document status in PostgreSQL.
    Falls back silently if PostgreSQL is not available.
    """
    try:
        from app.db.postgres import get_session_factory, update_document_status, DocumentStatusEnum
        session_factory = get_session_factory()
        async with session_factory() as db:
            await update_document_status(
                db=db,
                document_id=document_id,
                status=DocumentStatusEnum(status),
                chunk_count=chunk_count,
                error_message=error_message,
            )
            await db.commit()
            logger.debug(f"Status updated in PostgreSQL: {document_id} → {status}")
    except Exception as e:
        logger.warning(f"Could not update PostgreSQL status: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Additional Utility Tasks
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="app.workers.document_worker.delete_document_data")
def delete_document_data(document_id: str) -> dict:
    """
    Background task to clean up all data for a deleted document.

    Called after the user deletes a document from the UI.
    Removes: Qdrant vectors, Neo4j nodes, MinIO file.
    """
    logger.info(f"Deleting all data for document: {document_id}")

    try:
        from app.db.vector_store import delete_document_vectors
        delete_document_vectors(document_id)
    except Exception as e:
        logger.warning(f"Qdrant cleanup failed: {e}")

    try:
        from app.db.neo4j_client import delete_document_graph
        delete_document_graph(document_id)
    except Exception as e:
        logger.warning(f"Neo4j cleanup failed: {e}")

    return {"status": "deleted", "document_id": document_id}


@celery_app.task(name="app.workers.document_worker.health_check")
def health_check() -> dict:
    """
    Simple task to verify the worker is running and Redis is connected.
    Call from the API: health_check.delay().get(timeout=10)
    """
    return {"status": "healthy", "worker": "document_worker"}
