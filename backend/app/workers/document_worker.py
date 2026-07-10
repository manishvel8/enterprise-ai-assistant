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
    """Run an async coroutine from synchronous Celery task code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


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

        # Step 2: Download file from MinIO
        from app.services.storage_service import download_file
        file_bytes = _get_file_bytes(document_id, file_name)

        if file_bytes is None:
            raise ValueError(f"Could not retrieve file bytes for {document_id}")

        # Step 3-8: Run ingestion pipeline
        chunk_count = _run_ingestion_pipeline(document_id, file_name, file_type, file_bytes)

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
    Download file from MinIO, or return placeholder bytes for development.

    When MinIO is not configured (no minio package), we return a placeholder
    so the rest of the pipeline can still be tested.
    """
    from app.services.storage_service import download_file
    storage_path = f"documents/{document_id}/{file_name}"
    file_bytes = download_file(storage_path)

    if file_bytes is None:
        # Development fallback — MinIO not available
        logger.warning(f"MinIO not available. Using placeholder bytes for {document_id}")
        return b"[PLACEHOLDER DOCUMENT CONTENT - MinIO not configured]"

    return file_bytes


def _run_ingestion_pipeline(
    document_id: str,
    file_name: str,
    file_type: str,
    file_bytes: bytes,
) -> int:
    """
    Run the full ingestion pipeline and return the number of chunks created.

    Milestone 9: Stub — logs each step and returns 0.
    Milestone 10+: Each step is replaced with real implementation.
    """
    logger.info(f"Pipeline step 1/6: Parsing {file_type.upper()} file: {file_name}")
    # Milestone 10: raw_content = parse(file_bytes, file_type)
    raw_content = {"text": "[Document content - parser not yet implemented]", "page_count": 1}

    logger.info(f"Pipeline step 2/6: Normalizing to content_blocks schema")
    # Milestone 11: normalized = normalize(raw_content, document_id, file_name, file_type)
    normalized = {"document_id": document_id, "content_blocks": [{"text": raw_content["text"]}]}

    logger.info(f"Pipeline step 3/6: Chunking with metadata")
    # Milestone 12: chunks = chunk(normalized)
    chunks = [{"chunk_id": f"{document_id}_chunk_0", "text": raw_content["text"]}]

    logger.info(f"Pipeline step 4/6: Generating embeddings")
    # Milestone 13: embedded_chunks = embed(chunks)
    # (calls OpenAI text-embedding-3-small for each chunk)

    logger.info(f"Pipeline step 5/6: Storing vectors in Qdrant")
    # Milestone 14: store_vectors(embedded_chunks)

    logger.info(f"Pipeline step 6/6: Extracting and storing graph entities")
    # Milestone 18-19: extract_and_store_entities(chunks)

    chunk_count = len(chunks)
    logger.info(f"Pipeline complete: {chunk_count} chunks for document {document_id}")
    return chunk_count


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

    # TODO Milestone 14: Delete from Qdrant
    # TODO Milestone 19: Delete from Neo4j
    # TODO Milestone 7: Delete from MinIO (already done in API for now)

    return {"status": "deleted", "document_id": document_id}


@celery_app.task(name="app.workers.document_worker.health_check")
def health_check() -> dict:
    """
    Simple task to verify the worker is running and Redis is connected.
    Call from the API: health_check.delay().get(timeout=10)
    """
    return {"status": "healthy", "worker": "document_worker"}
