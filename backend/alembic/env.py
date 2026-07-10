"""
alembic/env.py — Alembic migration environment.

Configured for async SQLAlchemy (asyncpg driver) with auto-detection
of model changes via target_metadata.

Usage:
  # Generate a new migration from model changes:
  alembic revision --autogenerate -m "add users table"

  # Apply all pending migrations:
  alembic upgrade head

  # Downgrade one migration:
  alembic downgrade -1

  # Show migration history:
  alembic history
"""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import our models so Alembic can detect schema changes
from app.db.postgres import Base

# Read Alembic config
config = context.config

# Configure logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the metadata for autogenerate comparison
target_metadata = Base.metadata

# Override the database URL from environment variable
# This ensures Alembic uses the same .env settings as the application
from app.core.config import settings
config.set_main_option("sqlalchemy.url", settings.postgres_url)


def run_migrations_offline() -> None:
    """
    Run migrations in offline mode (generate SQL without connecting to DB).
    Used for generating SQL scripts to review before running.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine (required for asyncpg)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in online mode (connects to DB and applies changes)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
