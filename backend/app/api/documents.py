"""
documents.py — Document upload and library API endpoints.

Milestone 2: Skeleton only — accepts uploads but doesn't process them.
Milestone 7: Saves to MinIO object storage.
Milestone 8: Saves metadata to PostgreSQL.
Milestone 9: Enqueues Celery processing task.
Milestone 10+: Celery worker parses, chunks, embeds, and stores.

POST   /api/documents/upload      — Upload a new document
GET    /api/documents             — List all documents
GET    /api/documents/{id}        — Get document status
DELETE /api/documents/{id}        — Delete a document
"""

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pathlib import Path

from app.models.document import (
    UploadResponse,
    DocumentMetadata,
    DocumentListResponse,
    DocumentStatus,
)
from app.core.config import settings

router = APIRouter(prefix="/api/documents", tags=["Documents"])

# --- In-memory document store (Milestone 2 placeholder) ---
# This will be replaced by PostgreSQL in Milestone 8.
_documents: dict = {}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing",
    description="""
Upload a document to be processed by the AI pipeline.

Returns immediately with a document_id. Processing happens in the background.
Poll GET /api/documents/{id} to check status.

**Supported formats:** PDF, DOCX, PPTX, XLSX, CSV, TXT, PNG, JPG, MP3, MP4, WAV

**Milestone 2:** Accepts file, validates, stores metadata in memory only.
**Milestone 7+:** Saves file to MinIO object storage.
**Milestone 8+:** Saves metadata to PostgreSQL.
**Milestone 9+:** Enqueues Celery processing task.
    """,
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a document upload.

    Validation:
    - File extension must be in allowed list
    - File must not be empty
    - File size must not exceed max_upload_size_mb

    After validation (Milestone 2):
    - Generate a document_id
    - Store metadata in memory
    - Return 202 Accepted with document_id

    After Milestone 9:
    - Also save file to MinIO
    - Save metadata to PostgreSQL
    - Enqueue Celery task for background processing
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    suffix = Path(file.filename).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{suffix}' is not supported. Allowed: {settings.allowed_extensions}",
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
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB.",
        )

    # Generate a unique document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Store metadata in memory (replaced by PostgreSQL in Milestone 8)
    _documents[document_id] = DocumentMetadata(
        document_id=document_id,
        file_name=file.filename,
        file_type=suffix,
        file_size_bytes=len(content),
        status=DocumentStatus.PENDING,
        chunk_count=0,
    )

    # TODO Milestone 7: Save file to MinIO
    # storage_service.upload(document_id, content, file.filename)

    # TODO Milestone 8: Save metadata to PostgreSQL
    # await postgres.insert_document(_documents[document_id])

    # TODO Milestone 9: Enqueue Celery task
    # from app.workers.document_worker import process_document
    # process_document.delay(document_id, file.filename, suffix)

    return UploadResponse(
        document_id=document_id,
        file_name=file.filename,
        file_type=suffix,
        status=DocumentStatus.PENDING,
        message="Document accepted. Processing pipeline is not yet connected (Milestone 9).",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents() -> DocumentListResponse:
    """Return all documents with their current processing status."""
    docs = list(_documents.values())
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document processing status",
)
async def get_document(document_id: str) -> DocumentMetadata:
    """
    Poll this endpoint to check if a document has finished processing.

    Expected status flow: pending → processing → complete (or failed)
    """
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


@router.delete(
    "/{document_id}",
    summary="Delete a document",
)
async def delete_document(document_id: str):
    """
    Delete a document and all its associated data.

    Milestone 2: Removes from in-memory store.
    Later milestones: Also removes from PostgreSQL, Qdrant, Neo4j, and MinIO.
    """
    if document_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    del _documents[document_id]
    return {"message": f"Document '{document_id}' deleted."}
