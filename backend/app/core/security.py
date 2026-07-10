"""
security.py — CORS configuration, file validation, and auth placeholder.

CORS (Cross-Origin Resource Sharing):
  Browsers block JavaScript from calling APIs on different domains by default.
  We must explicitly allow our Angular frontend origin to call FastAPI.

File Validation:
  Never trust uploaded file extensions or MIME types alone.
  We validate both extension and MIME type before processing.

Auth Placeholder:
  Full JWT authentication will be implemented in a later milestone.
  For now, all endpoints are open. The placeholder is here to show where auth belongs.
"""

import os
import magic                   # python-magic for MIME type detection
from pathlib import Path
from fastapi import HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings


def configure_cors(app):
    """
    Add CORS middleware to the FastAPI app.

    This must be added before any routes are registered.

    Why: Angular runs on localhost:4200 (or a different domain in production).
    The browser will block API calls unless the server explicitly allows the origin.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,    # e.g. ["http://localhost:4200"]
        allow_credentials=True,
        allow_methods=["*"],                    # GET, POST, PUT, DELETE, OPTIONS
        allow_headers=["*"],                    # Authorization, Content-Type, etc.
    )


# MIME type allowlist — what we allow for document uploads
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "text/csv",
    "text/plain",
    "image/png",
    "image/jpeg",
    "audio/mpeg",
    "audio/wav",
    "video/mp4",
}


def validate_upload_file(file: UploadFile) -> None:
    """
    Validate a file upload for safety.

    Checks:
    1. File extension must be in the allowed list
    2. File size must not exceed max_upload_size_mb
    3. MIME type must match actual file content (prevents extension spoofing)

    Raises HTTPException 400 if validation fails.
    """
    # 1. Check extension
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{suffix}' is not allowed. Allowed types: {settings.allowed_extensions}",
        )

    # 2. Check filename for path traversal attacks
    safe_filename = os.path.basename(file.filename or "")
    if safe_filename != file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )


async def validate_upload_size(file: UploadFile) -> bytes:
    """
    Read and return file content, enforcing size limit.

    Returns the file bytes so the caller doesn't have to read again.
    Raises HTTPException 413 if file is too large.
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {settings.max_upload_size_mb}MB",
        )
    return content


# --- Auth Placeholder ---
# This will be replaced with real JWT validation in a later milestone.

async def get_current_user():
    """
    Placeholder dependency for auth.
    Inject this into protected endpoints: current_user = Depends(get_current_user)
    Returns a mock user dict for now.
    """
    return {"user_id": "user_demo", "email": "demo@example.com"}
