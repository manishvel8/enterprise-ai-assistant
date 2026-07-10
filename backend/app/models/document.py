"""
document.py — Pydantic models for document upload and retrieval API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class DocumentStatus(str, Enum):
    """Lifecycle status of an uploaded document."""
    PENDING = "pending"         # Uploaded, waiting in queue
    PROCESSING = "processing"   # Celery worker is working on it
    COMPLETE = "complete"       # Fully processed, available for retrieval
    FAILED = "failed"           # Processing failed


class UploadResponse(BaseModel):
    """
    Response for POST /api/documents/upload

    Returns immediately (202 Accepted) with a document_id to poll.
    The actual processing happens asynchronously in the Celery worker.
    """
    document_id: str
    file_name: str
    file_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    message: str = "Document uploaded. Processing will begin shortly."


class DocumentMetadata(BaseModel):
    """Full document metadata as stored in PostgreSQL."""
    document_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    status: DocumentStatus
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for GET /api/documents"""
    documents: List[DocumentMetadata]
    total: int


class ChunkSchema(BaseModel):
    """
    The normalized unit of document content that gets embedded and stored.

    Every parser ultimately produces a list of these, regardless of file type.
    """
    chunk_id: str
    document_id: str
    file_name: str
    source_type: str                # "pdf", "docx", "audio", etc.
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    timestamp_start: Optional[float] = None    # seconds (for audio/video)
    timestamp_end: Optional[float] = None
    section_title: Optional[str] = None
    chunk_text: str
    embedding_id: Optional[str] = None         # set after storing in Qdrant
    metadata: Dict[str, Any] = {}


class ContentBlock(BaseModel):
    """
    One block of content extracted from a document by a parser.

    Before chunking, the document is represented as a list of ContentBlocks.
    After chunking, each block (or group of blocks) becomes a ChunkSchema.
    """
    block_id: str
    type: str                       # "text", "table", "heading", "image_ocr", etc.
    text: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = {}


class NormalizedDocument(BaseModel):
    """
    The common document schema produced by every parser.

    Regardless of whether the source was a PDF, PPTX, or video,
    the output is always this structure. The chunker operates on this.
    """
    document_id: str
    file_name: str
    file_type: str
    content_blocks: List[ContentBlock]
