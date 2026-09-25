"""
config.py — Application settings loaded from environment variables.

Pydantic BaseSettings automatically reads values from:
  1. Environment variables (highest priority)
  2. .env file
  3. Default values defined here

Why this pattern?
  - No hardcoded secrets anywhere in the code
  - Same code runs locally (reads .env) and in Kubernetes (reads env vars from Secret/ConfigMap)
  - Type-safe: wrong types raise validation errors at startup, not at runtime
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Load project-root .env whether uvicorn is started from backend/ or repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../enterprise-ai-assistant
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """All application configuration, sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",             # ignore unknown env vars (avoids errors on K8s)
    )

    # --- Application ---
    app_name: str = "Enterprise AI Chat Assistant"
    app_version: str = "1.0.0"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_debug: bool = False
    log_level: str = "INFO"

    # --- CORS ---
    # Comma-separated origins allowed to call the API
    backend_cors_origins: str = "http://localhost:4200"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]

    # --- Security ---
    secret_key: str = "change_me_to_a_long_random_string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # --- OpenAI (or OpenAI-compatible gateway) ---
    openai_api_key: str = ""
    # Custom base URL for internal gateways, e.g. http://host:port/v1
    # Leave empty to use the default OpenAI cloud endpoint.
    openai_api_base: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_whisper_model: str = "whisper-1"

    # --- PostgreSQL (used from Milestone 8 onwards) ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_assistant"
    postgres_user: str = "ai_user"
    postgres_password: str = ""

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Redis (used from Milestone 9 onwards) ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # --- Qdrant (used from Milestone 14 onwards) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "document_chunks"

    # --- Neo4j (used from Milestone 17 onwards) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- MinIO (used from Milestone 7 onwards) ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "ai-documents"
    minio_use_ssl: bool = False

    # --- Langfuse (used from Milestone 23 onwards) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 30
    max_concurrent_openai_calls: int = 10

    # --- Document Processing ---
    max_upload_size_mb: int = 100
    allowed_extensions: str = "pdf,docx,pptx,xlsx,csv,txt,png,jpg,jpeg,mp3,mp4,wav"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    # --- Feature Flags ---
    enable_graph_rag: bool = True
    enable_streaming: bool = True
    enable_debug_panel: bool = True


# Create a single shared settings instance.
# Import this anywhere in the app: from app.core.config import settings
settings = Settings()
