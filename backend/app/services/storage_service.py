"""
storage_service.py — Object storage for uploaded files (MinIO / S3).

Why object storage?
  - Uploaded files can be large (100MB+ for videos)
  - Databases are not designed to store binary files (BLOBs are slow, wasteful)
  - Object storage is designed for exactly this: cheap, durable binary storage
  - MinIO is API-compatible with AWS S3 — same code runs locally and in the cloud

Flow:
  1. User uploads file → FastAPI receives bytes
  2. FastAPI calls storage_service.upload() → saves to MinIO
  3. FastAPI saves the MinIO path to PostgreSQL (metadata)
  4. Celery worker calls storage_service.download() → reads bytes from MinIO
  5. Worker processes the bytes through the ingestion pipeline

Why use MinIO locally instead of S3?
  - No AWS account needed for development
  - Free, runs in Docker
  - Same boto3/minio API so switching to S3 is a one-line config change
"""

import io
import logging
from pathlib import Path
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

_minio_client = None
_minio_init_attempted = False

# Real on-disk fallback when MinIO client/package is unavailable.
# Path layout: backend/.local_uploads/{document_id}/original.ext
_LOCAL_UPLOAD_ROOT = Path(__file__).resolve().parents[2] / ".local_uploads"


def get_minio_client():
    """
    Get or create the MinIO client.

    Returns None if MinIO is not configured or not reachable.
    Upload/download then use a real local-disk fallback.
    """
    global _minio_client, _minio_init_attempted
    if _minio_client is not None:
        return _minio_client
    if _minio_init_attempted and _minio_client is None:
        return None
    _minio_init_attempted = True

    try:
        from minio import Minio
        client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )
        _ensure_bucket_exists(client)
        _minio_client = client
        logger.info(f"MinIO client initialized: {settings.minio_endpoint}")
        return _minio_client
    except ImportError:
        logger.error(
            "minio package not installed. Run: pip install minio. "
            "Using local disk fallback under backend/.local_uploads/"
        )
        return None
    except Exception as e:
        logger.warning(f"MinIO not available: {e}. Using local disk fallback.")
        return None


def _ensure_bucket_exists(client) -> None:
    """Create the storage bucket if it doesn't exist."""
    bucket = settings.minio_bucket_name
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info(f"Created MinIO bucket: {bucket}")
    except Exception as e:
        logger.error(f"Failed to ensure bucket exists: {e}")
        raise


def _local_object_path(document_id: str, file_name: str) -> Path:
    ext = ""
    if "." in file_name:
        ext = "." + file_name.rsplit(".", 1)[-1].lower()
    return _LOCAL_UPLOAD_ROOT / document_id / f"original{ext}"


def _upload_local(document_id: str, file_bytes: bytes, file_name: str) -> str:
    """Write bytes to disk and return a local:// storage path the worker can read."""
    path = _local_object_path(document_id, file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    storage_path = f"local://{document_id}/{path.name}"
    logger.warning(
        f"Stored file on local disk (MinIO unavailable): {path} ({len(file_bytes)} bytes)"
    )
    return storage_path


def _download_local(storage_path: str) -> Optional[bytes]:
    """Read bytes from local://{document_id}/original.ext fallback storage."""
    # local://{document_id}/{filename}
    without_scheme = storage_path[len("local://") :]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        logger.error(f"Invalid local:// path: {storage_path}")
        return None
    document_id, file_name = parts
    path = _LOCAL_UPLOAD_ROOT / document_id / file_name
    if not path.is_file():
        # Also try original.* if legacy path used display name
        candidates = list((_LOCAL_UPLOAD_ROOT / document_id).glob("original.*")) if (
            _LOCAL_UPLOAD_ROOT / document_id
        ).is_dir() else []
        if candidates:
            path = candidates[0]
        else:
            logger.error(f"Local file missing: {path}")
            return None
    data = path.read_bytes()
    logger.info(f"Loaded local file {path} ({len(data)} bytes)")
    return data


def upload_file(document_id: str, file_bytes: bytes, file_name: str, content_type: str = "application/octet-stream") -> Optional[str]:
    """
    Upload a file to MinIO object storage.

    Object key uses a sanitized name (document_id + extension) so spaces,
    apostrophes, and unicode in original filenames never break downloads.
    The original display name is kept only in Postgres metadata.

    Falls back to real local disk under backend/.local_uploads/ if MinIO is down.
    """
    # Stable object key: documents/{doc_id}/original.ext
    ext = ""
    if "." in file_name:
        ext = "." + file_name.rsplit(".", 1)[-1].lower()
    object_name = f"documents/{document_id}/original{ext}"

    client = get_minio_client()
    if not client:
        return _upload_local(document_id, file_bytes, file_name)

    try:
        client.put_object(
            bucket_name=settings.minio_bucket_name,
            object_name=object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
        logger.info(f"Uploaded file to MinIO: {object_name} ({len(file_bytes)} bytes)")
        return object_name
    except Exception as e:
        logger.error(f"MinIO upload failed: {e}. Falling back to local disk.")
        return _upload_local(document_id, file_bytes, file_name)


def download_file(storage_path: str) -> Optional[bytes]:
    """
    Download a file from MinIO object storage (or local:// disk fallback).

    Called by the Celery worker to get the raw file bytes for processing.
    """
    if storage_path.startswith("local://"):
        return _download_local(storage_path)

    client = get_minio_client()
    if not client:
        logger.warning("MinIO not available. Cannot download MinIO object.")
        return None

    try:
        response = client.get_object(settings.minio_bucket_name, storage_path)
        data = response.read()
        response.close()
        response.release_conn()
        logger.info(f"Downloaded file from MinIO: {storage_path} ({len(data)} bytes)")
        return data
    except Exception as e:
        logger.error(f"MinIO download failed for '{storage_path}': {e}")
        return None


def delete_file(storage_path: str) -> bool:
    """
    Delete a file from MinIO (or local disk fallback).

    Returns True if deletion succeeded, False otherwise.
    """
    if storage_path.startswith("local://"):
        without_scheme = storage_path[len("local://") :]
        parts = without_scheme.split("/", 1)
        if len(parts) == 2:
            path = _LOCAL_UPLOAD_ROOT / parts[0] / parts[1]
            if path.is_file():
                path.unlink()
            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        return True

    client = get_minio_client()
    if not client:
        return False

    try:
        client.remove_object(settings.minio_bucket_name, storage_path)
        logger.info(f"Deleted file from MinIO: {storage_path}")
        return True
    except Exception as e:
        logger.error(f"MinIO delete failed: {e}")
        return False


def get_presigned_url(storage_path: str, expires_seconds: int = 3600) -> Optional[str]:
    """
    Generate a time-limited URL for direct file download.

    Used by the frontend to let users download their original uploaded files
    without routing through the API server (saves bandwidth).
    """
    if storage_path.startswith("local://"):
        return None

    client = get_minio_client()
    if not client:
        return None

    try:
        from datetime import timedelta
        url = client.presigned_get_object(
            settings.minio_bucket_name,
            storage_path,
            expires=timedelta(seconds=expires_seconds),
        )
        return url
    except Exception as e:
        logger.error(f"Presigned URL generation failed: {e}")
        return None
