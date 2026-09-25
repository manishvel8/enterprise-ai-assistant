"""
config.py — Centralised settings loaded from .env

Pydantic Settings reads values from:
  1. Real environment variables (highest priority)
  2. The .env file in the langgraph-agent directory
  3. Defaults defined here

Why here and not in individual files?
  One place to change anything. No magic strings scattered across the code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV) if _ENV.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI / gateway
    openai_api_key: str = ""
    openai_api_base: str = ""          # e.g. http://172.25.1.60:30444/v1
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Tavily
    tavily_api_key: str = ""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "chunks"

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "langgraph-agent"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 7860
    max_quality_retries: int = 3

    # MCP — when true, RAG / web nodes call tools via MCP client (stdio)
    # instead of importing services directly. Great for learning the protocol.
    use_mcp_tools: bool = False

    # ── Langfuse ────────────────────────────────────────────────────────────
    # Self-hosted Langfuse for LLM observability. After running docker-compose:
    #   1. Open http://localhost:3000 → create project → copy keys
    #   2. Paste LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY into .env
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_enabled: bool = True          # set False to disable all tracing

    # ── PostgreSQL for Conversation History ─────────────────────────────────
    # Stores users, conversations, sessions, messages, and long-term memory.
    # Uses pgvector extension for semantic similarity search over message history.
    conversation_db_host: str = "localhost"
    conversation_db_port: int = 5433
    conversation_db_name: str = "ai_assistant"
    conversation_db_user: str = "ai_user"
    conversation_db_password: str = "dev_password"

    @property
    def conversation_db_dsn(self) -> str:
        """Build a psycopg2-compatible connection string."""
        return (
            f"host={self.conversation_db_host} "
            f"port={self.conversation_db_port} "
            f"dbname={self.conversation_db_name} "
            f"user={self.conversation_db_user} "
            f"password={self.conversation_db_password}"
        )


settings = Settings()
