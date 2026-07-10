"""
main.py — FastAPI application entry point.

This file:
1. Creates the FastAPI app instance
2. Registers CORS middleware
3. Registers all API routers
4. Defines startup and shutdown lifecycle events
5. Is the entry point for uvicorn: uvicorn app.main:app --reload

Why lifespan events?
  startup: Initialize database connection pools, load ML models, warm up caches
  shutdown: Gracefully close connections, flush pending traces to Langfuse
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import health, chat, documents

# Configure application-level logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Code before 'yield' runs at startup.
    Code after 'yield' runs at shutdown.

    Milestone 2: Just logs startup/shutdown.
    Milestone 8: Initialize PostgreSQL connection pool.
    Milestone 9: Initialize Redis connection.
    Milestone 14: Initialize Qdrant client.
    Milestone 17: Initialize Neo4j driver.
    Milestone 23: Initialize Langfuse client.
    """
    # --- STARTUP ---
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.backend_debug}")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info(f"OpenAI model: {settings.openai_chat_model}")

    # --- Initialize PostgreSQL ---
    # Milestone 8: Enable when PostgreSQL is running.
    # Uses create_all() to create tables if they don't exist.
    # In production, use Alembic migrations instead.
    postgres_url = settings.postgres_url
    if settings.postgres_password and settings.postgres_password != "":
        try:
            from app.db.postgres import init_db
            await init_db()
            logger.info("PostgreSQL connected and tables verified")
        except Exception as e:
            logger.warning(f"PostgreSQL not available (running without DB): {e}")
    else:
        logger.warning("POSTGRES_PASSWORD not set — running without PostgreSQL")

    # TODO Milestone 9: Initialize Redis
    # from app.services.cache_service import init_redis
    # await init_redis()

    # TODO Milestone 14: Initialize Qdrant
    # from app.db.vector_store import init_qdrant
    # init_qdrant()

    # TODO Milestone 17: Initialize Neo4j
    # from app.db.neo4j_client import init_neo4j
    # init_neo4j()

    # TODO Milestone 23: Initialize Langfuse
    # from app.services.langfuse_service import init_langfuse
    # init_langfuse()

    logger.info("Application startup complete. Accepting requests.")
    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down application ...")

    # Close PostgreSQL connections
    try:
        from app.db.postgres import close_db
        await close_db()
    except Exception as e:
        logger.warning(f"Error closing DB: {e}")

    logger.info("Shutdown complete.")


# Create the FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
Enterprise AI Chat Assistant with RAG, GraphRAG, and Agentic Workflow.

## Features
- Chat with your documents using AI
- RAG: Retrieve relevant chunks and generate grounded answers
- GraphRAG: Traverse entity relationships via Neo4j
- Agentic workflow: Router, Retriever, Cypher, Answer, Critic, Memory agents
- Full observability via Langfuse

## Milestone Progress
This API is being built milestone by milestone. Current milestone: **2 (Skeleton)**
    """,
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI at http://localhost:8000/redoc
    openapi_url="/openapi.json",
)

# --- CORS Middleware ---
# Must be added BEFORE routers so it applies to all routes.
# This allows Angular (localhost:4200) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register API Routers ---
# Each router is defined in its own file under app/api/
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirect hint for users who visit the bare URL."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
