# =============================================================================
# Backend Dockerfile — FastAPI application
# =============================================================================
#
# Uses a single-stage build (no multi-stage needed for Python).
# We use python:3.11-slim as the base — it's small but includes pip.
#
# Why python:3.11-slim instead of python:3.11?
#   python:3.11 includes many development tools we don't need in production.
#   python:3.11-slim is ~150MB vs ~900MB for the full image.
#
# Build:    docker build -t ai-backend:v1 ./backend
# Run:      docker run -p 8000:8000 --env-file .env ai-backend:v1
# Dev run:  docker run -p 8000:8000 --env-file .env -v $(pwd)/app:/app/app ai-backend:v1
# =============================================================================

FROM python:3.11-slim

# Metadata — visible in Docker Hub and docker inspect
LABEL maintainer="AI Engineer Project"
LABEL description="Enterprise AI Chat Assistant — FastAPI Backend"

# ─── System dependencies ──────────────────────────────────────────────────────
# Install only what's needed for Python packages that have C extensions.
# libmagic1: for python-magic (file type detection)
# libpq-dev: for psycopg2 (PostgreSQL driver) — added when needed
# We run as a non-root user for security.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements.txt FIRST (before source code).
# Docker caches this layer. If requirements.txt hasn't changed,
# pip install is skipped on the next build — faster CI/CD rebuilds.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't cache pip's download cache (saves ~50MB in image)
# --upgrade pip: get the latest pip before installing
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ─── Application source ───────────────────────────────────────────────────────
# Copy source code AFTER installing dependencies.
# This way, code changes don't invalidate the pip install layer.
COPY app/ ./app/

# ─── Non-root user ────────────────────────────────────────────────────────────
# Running as root in a container is a security risk.
# Create a dedicated user for the application process.
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# ─── Port ─────────────────────────────────────────────────────────────────────
# EXPOSE documents which port the container listens on.
# It does NOT publish the port — that's done at runtime with -p 8000:8000.
EXPOSE 8000

# ─── Health check ─────────────────────────────────────────────────────────────
# Docker will run this command periodically to check if the container is healthy.
# If it fails 3 times, Docker marks the container as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Start command ────────────────────────────────────────────────────────────
# uvicorn: the ASGI server that runs FastAPI
# --host 0.0.0.0: listen on all interfaces (not just localhost inside the container)
# --port 8000: which port to listen on
# --workers 1: single worker per container (K8s scales pods, not workers)
# --log-level info: log level
#
# For development with hot reload, override this command:
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
