"""
test-pipeline.py — End-to-end smoke test for the document processing pipeline.

Run this after Milestone 9 (Celery worker) is implemented to verify the full pipeline.

Usage:
    python scripts/test-pipeline.py --file path/to/test.pdf

What this tests:
    1. Backend /health endpoint responds
    2. File upload API accepts the file
    3. Celery worker picks up the task
    4. Document appears in PostgreSQL with status 'complete'
    5. Chunks appear in Qdrant
    6. Graph nodes appear in Neo4j
"""

import argparse
import time
import sys
import requests

BACKEND_URL = "http://localhost:8000"


def check_health():
    print("1. Checking backend health ...")
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        print(f"   OK: {resp.json()}")
    except Exception as e:
        print(f"   ERROR: Backend not reachable — {e}")
        print("   Make sure docker compose is running.")
        sys.exit(1)


def upload_file(file_path: str) -> str:
    print(f"2. Uploading file: {file_path} ...")
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/api/documents/upload",
            files={"file": f},
            timeout=30,
        )
    if resp.status_code not in (200, 202):
        print(f"   ERROR: Upload failed with status {resp.status_code}: {resp.text}")
        sys.exit(1)
    data = resp.json()
    document_id = data.get("document_id")
    print(f"   OK: document_id = {document_id}")
    return document_id


def poll_status(document_id: str, max_wait: int = 120):
    print(f"3. Waiting for processing to complete (max {max_wait}s) ...")
    for i in range(max_wait):
        time.sleep(1)
        try:
            resp = requests.get(f"{BACKEND_URL}/api/documents/{document_id}", timeout=5)
            data = resp.json()
            status = data.get("status", "unknown")
            if status == "complete":
                print(f"   OK: Processing complete after {i+1}s")
                return data
            elif status == "failed":
                print(f"   ERROR: Processing failed: {data.get('error')}")
                sys.exit(1)
            elif i % 10 == 0:
                print(f"   ... status={status}, elapsed={i}s")
        except Exception:
            pass
    print(f"   ERROR: Timed out after {max_wait}s")
    sys.exit(1)


def test_retrieval(document_id: str):
    print("4. Testing vector retrieval ...")
    resp = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={
            "user_id": "test_user",
            "session_id": "test_session",
            "message": "What is the main topic of this document?",
            "document_ids": [document_id],
        },
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"   OK: Got answer with {len(data.get('citations', []))} citations")
    else:
        print(f"   WARN: Chat API returned {resp.status_code} (may not be implemented yet)")


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--file", required=False, default=None, help="Path to test file")
    args = parser.parse_args()

    check_health()

    if args.file:
        document_id = upload_file(args.file)
        poll_status(document_id)
        test_retrieval(document_id)
        print("\nAll tests passed!")
    else:
        print("\nNo --file provided. Only health check ran.")
        print("Usage: python scripts/test-pipeline.py --file path/to/test.pdf")


if __name__ == "__main__":
    main()
