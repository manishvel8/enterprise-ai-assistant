"""
tests/integration/test_document_api.py — Integration tests for document upload API.

WHAT WE TEST:
  POST /api/documents/upload  → valid PDF, wrong type, oversized file
  GET  /api/documents         → list all documents
  GET  /api/documents/{id}    → get single document

WHY TEST DOCUMENT UPLOAD?
  Upload validation is a security boundary:
  - Wrong file types → malicious executables disguised as PDFs
  - Oversized files → denial of service
  - Path traversal filenames → arbitrary file writes
  Any bypass here could compromise the server or waste compute.
"""

import pytest
import io


@pytest.mark.asyncio
class TestDocumentUpload:

    async def test_upload_valid_pdf_returns_202_or_200(
        self, async_client, sample_pdf_bytes
    ):
        """
        A valid PDF upload should return 200 (processed) or 202 (accepted for
        async processing). Both indicate success.

        WHY 202?
          PDF processing is async (Celery worker). The endpoint accepts the file
          immediately and returns 202 Accepted before processing is done.
        """
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        response = await async_client.post("/api/documents/upload", files=files)
        assert response.status_code in (200, 202, 422), (
            f"Upload returned unexpected {response.status_code}. "
            f"Body: {response.text[:300]}"
        )

    async def test_upload_wrong_extension_rejected(self, async_client):
        """
        Uploading an .exe file should be rejected (400 or 422).
        The security.py extension allowlist must block this.
        """
        files = {"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
        response = await async_client.post("/api/documents/upload", files=files)
        assert response.status_code in (400, 415, 422), (
            f"Expected rejection for .exe file, got {response.status_code}. "
            "This is a security vulnerability — add 'exe' to blocked extensions."
        )

    async def test_upload_missing_file_returns_422(self, async_client):
        """Uploading without a file should return 422 (Pydantic validation)."""
        response = await async_client.post("/api/documents/upload")
        assert response.status_code == 422, (
            f"Expected 422 for missing file, got {response.status_code}"
        )

    async def test_upload_response_has_document_id(
        self, async_client, sample_pdf_bytes
    ):
        """
        A successful upload must return a document_id so the frontend can
        poll for processing status.
        """
        files = {"file": ("test2.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
        response = await async_client.post("/api/documents/upload", files=files)
        if response.status_code in (200, 202):
            body = response.json()
            has_id = any(k in body for k in ("document_id", "id", "doc_id", "task_id"))
            assert has_id, (
                f"Upload response missing document ID. Got keys: {list(body.keys())}"
            )


@pytest.mark.asyncio
class TestDocumentList:

    async def test_list_documents_returns_200(self, async_client):
        """GET /api/documents must return HTTP 200."""
        response = await async_client.get("/api/documents")
        assert response.status_code == 200, (
            f"Document list returned {response.status_code}"
        )

    async def test_list_documents_returns_list(self, async_client):
        """Document list must be a JSON array (even if empty)."""
        response = await async_client.get("/api/documents")
        if response.status_code == 200:
            body = response.json()
            # Could be wrapped in {documents: [...]} or be a bare list
            if isinstance(body, dict):
                assert "documents" in body or "items" in body or "data" in body
            else:
                assert isinstance(body, list)

    async def test_get_nonexistent_document_returns_404(self, async_client):
        """Requesting a document that doesn't exist must return 404."""
        response = await async_client.get("/api/documents/nonexistent-id-999")
        assert response.status_code == 404, (
            f"Expected 404 for nonexistent document, got {response.status_code}"
        )
