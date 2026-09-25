# ─────────────────────────────────────────────────────────────────────────────
# langgraph-agent/Dockerfile — Container image for the LangGraph Agent service
#
# WHY A DOCKERFILE?
#   Without Docker, the LangGraph agent only runs on your laptop.
#   With Docker, it runs identically everywhere: local, CI/CD, Kubernetes.
#   "Works on my machine" becomes "works in a container" = reproducible.
#
# BUILD:
#   cd enterprise-ai-assistant/langgraph-agent
#   docker build -t langgraph-agent:dev .
#
# RUN LOCALLY:
#   docker run -p 7860:7860 --env-file .env langgraph-agent:dev
#
# MULTI-STAGE BUILD EXPLANATION:
#   Stage 1 (builder): Install dependencies — the heavy step (~500MB)
#   Stage 2 (runtime): Copy only the installed packages, not build tools
#   Result: Final image is smaller and has fewer attack surfaces
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# python:3.11-slim = Debian with Python but without development headers.
# We install gcc/build-essentials here for packages that need compilation (psycopg2).
FROM python:3.11-slim AS builder

# Set working directory for build
WORKDIR /build

# Install system dependencies needed to compile Python packages.
# psycopg2 needs libpq-dev (PostgreSQL client library headers).
# build-essential provides gcc and g++ for C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching):
# If requirements.txt doesn't change, this layer is cached.
# Only code changes don't re-run pip install.
COPY requirements.txt .

# Install into a prefix directory (not system Python).
# --no-cache-dir: don't cache pip downloads (keeps image smaller).
# --prefix=/install: installs packages to /install (copied to stage 2).
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only RUNTIME system libraries (no compilers).
# libpq5 = PostgreSQL client runtime (for psycopg2 at runtime).
# curl = needed for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder stage.
# /install/lib/python3.11/site-packages → /usr/local/lib/python3.11/site-packages
COPY --from=builder /install /usr/local

# ── Non-root user (security best practice) ────────────────────────────────────
# Running as root inside a container is a security risk.
# If an attacker exploits the app, they'd have root access to the container.
# A non-root user limits the blast radius.
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# Copy application code
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup mcp_server/ ./mcp_server/
COPY --chown=appuser:appgroup static/ ./static/

# Create data directory (fine-tuning datasets, etc.)
RUN mkdir -p /app/data && chown appuser:appgroup /app/data

# Switch to non-root user
USER appuser

# ── Environment variables ──────────────────────────────────────────────────────
# Default values (overridden by docker-compose environment: or K8s ConfigMap/Secret)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# ── Expose port ───────────────────────────────────────────────────────────────
# EXPOSE documents the port; it doesn't publish it (docker run -p 7860:7860 does that).
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Docker health checks are used by docker-compose depends_on and Kubernetes probes.
# --interval=30s: check every 30 seconds
# --timeout=10s: fail if no response in 10 seconds
# --start-period=40s: don't count failures during the first 40s (startup time)
# --retries=3: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# ── Startup command ───────────────────────────────────────────────────────────
# Use exec form (array) not shell form ("string"):
# - Exec form: PID 1 = uvicorn → receives SIGTERM gracefully
# - Shell form: PID 1 = /bin/sh → uvicorn is a child process, may not get SIGTERM
#
# --host 0.0.0.0: listen on all interfaces (required in containers)
# --port 7860: match the EXPOSE above
# --workers 1: LangGraph uses in-process state; multiple workers would
#              cause different workers to handle different parts of the same conversation.
#              Use K8s horizontal scaling (multiple pods) instead of multiple workers.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
