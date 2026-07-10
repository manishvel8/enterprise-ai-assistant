"""
documents.py — Document upload and library API endpoints.

Milestone 7: File validation + MinIO storage.
Milestone 8: PostgreSQL metadata persistence.
Milestone 9: Celery queue for background processing.
Milestone 10-12: Full ingestion pipeline in the worker.

POST   /api/documents/upload       — Upload + store in MinIO
GET    /api/documents              — List all documents
GET    /api/documents/{id}         — Get document status
GET    /api/documents/{id}/url     — Get download URL from MinIO
DELETE /api/documents/{id}         — Delete document + MinIO file
"""

import uuid
import mimetypes
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks

from app.models.document import (
    UploadResponse,
    DocumentMetadata,
    DocumentListResponse,
    DocumentStatus,
)
from app.services import storage_service
from app.core.config import settings

router = APIRouter(prefix="/api/documents", tags=["Documents"])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory document store — replaced by PostgreSQL in Milestone 8.
# ─────────────────────────────────────────────────────────────────────────────
_documents: dict[str, DocumentMetadata] = {}
_storage_paths: dict[str, str] = {}    # document_id → MinIO path


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for AI processing",
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a document to be processed by the AI pipeline.

    Steps (Milestone 7):
    1. Validate filename and extension
    2. Read file bytes and check size
    3. Detect MIME type from content
    4. Generate unique document_id
    5. Upload to MinIO object storage
    6. Store metadata in memory
    7. Return 202 Accepted with document_id

    Steps added in later milestones:
    8. (M8) Save metadata to PostgreSQL
    9. (M9) Enqueue Celery processing task
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    # Sanitize filename — prevent path traversal attacks
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename != file.filename.replace("/", "").replace("\\", ""):
        safe_filename = file.filename.replace("/", "_").replace("\\", "_")

    # Check extension
    suffix = Path(safe_filename).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Allowed types: {settings.allowed_extensions}"
            ),
        )

    # Read file content and check size
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB. "
                   f"File is {len(content) / 1024 / 1024:.1f}MB.",
        )

    # Detect MIME type from content (safer than trusting file extension)
    content_type = _detect_mime_type(content, safe_filename)
    logger.info(f"Upload received: {safe_filename} ({len(content)} bytes, {content_type})")

    # Generate unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Upload to MinIO
    storage_path = storage_service.upload_file(
        document_id=document_id,
        file_bytes=content,
        file_name=safe_filename,
        content_type=content_type,
    )

    if storage_path:
        _storage_paths[document_id] = storage_path
        logger.info(f"Stored at: {storage_path}")
    else:
        logger.warning(f"MinIO storage failed for {document_id}. Proceeding without file storage.")

    # Store metadata in memory
    metadata = DocumentMetadata(
        document_id=document_id,
        file_name=safe_filename,
        file_type=suffix,
        file_size_bytes=len(content),
        status=DocumentStatus.PENDING,
        chunk_count=0,
    )
    _documents[document_id] = metadata

    # --- Save to PostgreSQL (Milestone 8) ---
    await _persist_document(document_id, safe_filename, suffix, len(content), storage_path)

    # --- Enqueue Celery processing task (Milestone 9) ---
    _enqueue_processing(document_id, safe_filename, suffix)

    return UploadResponse(
        document_id=document_id,
        file_name=safe_filename,
        file_type=suffix,
        status=DocumentStatus.PENDING,
        message=(
            "Document uploaded and stored. "
            "Background processing pipeline connects in Milestone 9. "
            f"document_id: {document_id}"
        ),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents() -> DocumentListResponse:
    """
    Return all documents with their current processing status.
    Reads from PostgreSQL if available, falls back to in-memory.
    """
    try:
        from app.db.postgres import get_session_factory, list_all_documents, DocumentStatusEnum
        session_factory = get_session_factory()
        async with session_factory() as db:
            db_docs = await list_all_documents(db)
            docs = [
                DocumentMetadata(
                    document_id=d.document_id,
                    file_name=d.file_name,
                    file_type=d.file_type,
                    file_size_bytes=d.file_size_bytes,
                    status=DocumentStatus(d.status.value),
                    chunk_count=d.chunk_count or 0,
                    error_message=d.error_message,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                    updated_at=d.updated_at.isoformat() if d.updated_at else None,
                )
                for d in db_docs
            ]
            return DocumentListResponse(documents=docs, total=len(docs))
    except Exception:
        # Fall back to in-memory store
        docs = sorted(_documents.values(), key=lambda d: d.document_id, reverse=True)
        return DocumentListResponse(documents=list(docs), total=len(docs))


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document processing status",
)
async def get_document(document_id: str) -> DocumentMetadata:
    """
    Poll this endpoint to track document processing progress.
    Status flow: pending → processing → complete | failed
    """
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


@router.get(
    "/{document_id}/url",
    summary="Get a pre-signed download URL for the original file",
)
async def get_download_url(document_id: str, expires_seconds: int = 3600):
    """
    Generate a time-limited URL to download the original file directly from MinIO.

    This avoids routing large file downloads through the API server.
    """
    if document_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    storage_path = _storage_paths.get(document_id)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original file not found in storage.",
        )

    url = storage_service.get_presigned_url(storage_path, expires_seconds)
    if not url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage is not available.",
        )

    return {"download_url": url, "expires_seconds": expires_seconds}


@router.delete(
    "/{document_id}",
    summary="Delete a document and all its data",
)
async def delete_document(document_id: str):
    """
    Delete a document and clean up all associated data.

    Milestone 7: Deletes from memory + MinIO.
    Later milestones: Also deletes from PostgreSQL, Qdrant vectors, Neo4j nodes.
    """
    if document_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # Delete from MinIO
    storage_path = _storage_paths.pop(document_id, None)
    if storage_path:
        storage_service.delete_file(storage_path)

    # Delete from memory
    del _documents[document_id]

    # TODO Milestone 8: Delete from PostgreSQL
    # TODO Milestone 14: Delete vectors from Qdrant
    # TODO Milestone 19: Delete nodes from Neo4j

    logger.info(f"Deleted document: {document_id}")
    return {"message": f"Document '{document_id}' deleted successfully."}


def _enqueue_processing(document_id: str, file_name: str, file_type: str) -> None:
    """
    Enqueue a Celery document processing task.

    Falls back silently if Redis is not available — the document stays
    in PENDING status and can be retried when the worker starts.
    """
    try:
        from app.workers.document_worker import process_document
        task = process_document.delay(document_id, file_name, file_type)
        logger.info(f"Enqueued processing task: {task.id} for document {document_id}")
    except Exception as e:
        logger.warning(f"Could not enqueue Celery task (Redis may not be running): {e}")
        logger.info(f"Document {document_id} will remain PENDING until worker picks it up")


async def _persist_document(
    document_id: str,
    file_name: str,
    file_type: str,
    file_size_bytes: int,
    storage_path: Optional[str],
) -> None:
    """
    Persist document metadata to PostgreSQL if available.
    Falls back silently to in-memory only if DB is not connected.
    """
    try:
        from app.db.postgres import get_session_factory, DocumentModel, DocumentStatusEnum
        from sqlalchemy.exc import OperationalError
        session_factory = get_session_factory()
        async with session_factory() as db:
            doc = DocumentModel(
                document_id=document_id,
                file_name=file_name,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                storage_path=storage_path,
                status=DocumentStatusEnum.PENDING,
                chunk_count=0,
            )
            db.add(doc)
            await db.commit()
            logger.info(f"Document {document_id} saved to PostgreSQL")
    except Exception as e:
        logger.warning(f"PostgreSQL not available — using in-memory store only: {e}")


def _detect_mime_type(content: bytes, filename: str) -> str:
    """
    Detect MIME type from file content and filename.

    We first try to detect from the actual bytes (magic number check),
    then fall back to filename extension.

    Why not trust the extension alone?
    A user could rename a .exe file to .pdf and try to upload it.
    Content-based detection is more reliable.
    """
    # Try python-magic (reads magic bytes from file header)
    try:
        import magic
        detected = magic.from_buffer(content[:1024], mime=True)
        if detected and detected != "application/octet-stream":
            return detected
    except (ImportError, Exception):
        pass

    # Fallback: guess from filename extension
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"
