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
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

_minio_client = None


def get_minio_client():
    """
    Get or create the MinIO client.

    Returns None if MinIO is not configured or not reachable.
    The upload functions fall back to local disk if MinIO is unavailable.
    """
    global _minio_client
    if _minio_client is not None:
        return _minio_client

    try:
        from minio import Minio
        _minio_client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )
        _ensure_bucket_exists(_minio_client)
        logger.info(f"MinIO client initialized: {settings.minio_endpoint}")
        return _minio_client
    except ImportError:
        logger.warning("minio package not installed. File storage disabled.")
        return None
    except Exception as e:
        logger.warning(f"MinIO not available: {e}. File storage disabled.")
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


def upload_file(document_id: str, file_bytes: bytes, file_name: str, content_type: str = "application/octet-stream") -> Optional[str]:
    """
    Upload a file to MinIO object storage.

    Args:
        document_id: Unique ID — used as part of the storage key
        file_bytes: Raw file content
        file_name: Original filename (for content-type detection)
        content_type: MIME type of the file

    Returns:
        Storage path (e.g., "documents/doc_abc123/report.pdf")
        or None if upload failed.

    Why use document_id as prefix?
        - All files for one document are grouped in one "folder"
        - Easy to delete all files for a document in one operation
        - Avoids filename collisions (two users uploading "report.pdf")
    """
    client = get_minio_client()
    if not client:
        logger.warning(f"MinIO not available. File '{file_name}' not stored.")
        return f"local://{document_id}/{file_name}"  # fake path for development

    storage_path = f"documents/{document_id}/{file_name}"

    try:
        client.put_object(
            bucket_name=settings.minio_bucket_name,
            object_name=storage_path,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
        logger.info(f"Uploaded file to MinIO: {storage_path} ({len(file_bytes)} bytes)")
        return storage_path
    except Exception as e:
        logger.error(f"MinIO upload failed: {e}")
        return None


def download_file(storage_path: str) -> Optional[bytes]:
    """
    Download a file from MinIO object storage.

    Called by the Celery worker to get the raw file bytes for processing.

    Args:
        storage_path: The path returned by upload_file()

    Returns:
        File bytes, or None if download failed.
    """
    if storage_path.startswith("local://"):
        # Development fallback — file wasn't actually stored
        logger.warning(f"Cannot download local:// path: {storage_path}")
        return None

    client = get_minio_client()
    if not client:
        logger.warning("MinIO not available. Cannot download file.")
        return None

    try:
        response = client.get_object(settings.minio_bucket_name, storage_path)
        data = response.read()
        response.close()
        response.release_conn()
        logger.info(f"Downloaded file from MinIO: {storage_path} ({len(data)} bytes)")
        return data
    except Exception as e:
        logger.error(f"MinIO download failed: {e}")
        return None


def delete_file(storage_path: str) -> bool:
    """
    Delete a file from MinIO.

    Called when a document is deleted from the system.

    Returns True if deletion succeeded, False otherwise.
    """
    if storage_path.startswith("local://"):
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

    Args:
        storage_path: MinIO object path
        expires_seconds: How long the URL is valid (default: 1 hour)

    Returns:
        Pre-signed URL string, or None if generation failed.
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
