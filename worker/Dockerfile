# =============================================================================
# Worker Dockerfile — Celery document processing worker
# =============================================================================
#
# The worker uses the SAME Python dependencies and application code as the backend.
# The only difference from the backend is the CMD.
#
# BUILD CONTEXT: This Dockerfile is built from the project ROOT (enterprise-ai-assistant/)
# so it can access both worker/ and backend/ directories.
#
# docker-compose.yml sets: context: . (project root), dockerfile: worker/Dockerfile
#
# Why share code with backend?
#   - DRY: same app/ code, same requirements.txt
#   - The Celery task (document_worker.py) lives in backend/app/workers/
#   - Worker needs the same parsers, embedders, DB clients as the backend
#
# Build (via docker compose):
#   docker compose build worker
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="AI Engineer Project"
LABEL description="Enterprise AI Chat Assistant — Celery Document Processing Worker"

# ─── System dependencies ──────────────────────────────────────────────────────
# Same as backend, plus tools needed by document parsers (added as milestones progress)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    # Uncomment in Milestone 10:
    # ffmpeg            (for video audio extraction)
    # tesseract-ocr     (for image OCR)
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
# Copy from backend's requirements.txt (build context is project root)
COPY backend/requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ─── Application source ───────────────────────────────────────────────────────
# Copy the full backend application (worker needs all app modules)
COPY backend/app/ ./app/

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 \
    CMD celery -A app.workers.document_worker inspect ping || exit 1

# ─── Start Celery worker ──────────────────────────────────────────────────────
#
# How Celery works:
#   1. Worker starts and connects to Redis (the broker)
#   2. Worker listens for tasks on the 'document_processing' queue
#   3. FastAPI enqueues a task: process_document.delay(document_id)
#   4. Worker picks up the task and runs process_document()
#   5. Worker writes results to Redis (result backend) and updates PostgreSQL
#
# --concurrency=4:
#   How many tasks to process simultaneously per worker pod.
#   For I/O-bound tasks (calling OpenAI, writing to DB): higher concurrency is fine.
#   For CPU-bound tasks (parsing large PDFs): lower concurrency prevents OOM.
#
# --queues=document_processing:
#   Only process tasks from this queue.
#   Allows routing urgent tasks to a fast queue and bulk tasks to a slow queue.
CMD ["celery", \
     "-A", "app.workers.document_worker", \
     "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--queues=document_processing"]
