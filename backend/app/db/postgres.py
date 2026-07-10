"""
postgres.py — PostgreSQL database setup using SQLAlchemy async ORM.

Why SQLAlchemy?
  - Python's most mature ORM (Object-Relational Mapper)
  - Async support via asyncpg driver — works perfectly with FastAPI async endpoints
  - Alembic for schema migrations (add columns without destroying data)
  - Type-safe models with Python type hints

Why async?
  - FastAPI is async. If we use a synchronous DB call, it BLOCKS the event loop
    while waiting for PostgreSQL — defeating the purpose of async FastAPI.
  - asyncpg + SQLAlchemy async = non-blocking DB queries that run concurrently.

Key concepts:
  - Engine: the database connection pool (created once at startup)
  - Session: a unit of work (like a database transaction)
  - Model: a Python class that maps to a database table
  - AsyncSession: async version of Session (use with 'async with' / 'await')
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, AsyncGenerator
from sqlalchemy import (
    Column, String, Integer, BigInteger, DateTime, Text, ForeignKey, Enum
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Database Engine and Session Factory
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine (connection pool)."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.postgres_url,
            pool_size=10,           # max persistent connections in pool
            max_overflow=20,        # extra connections allowed during spikes
            pool_timeout=30,        # seconds to wait for a connection from pool
            pool_pre_ping=True,     # test connections before using (handles restarts)
            echo=settings.backend_debug,  # log all SQL in debug mode
        )
        logger.info("PostgreSQL engine created")
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,  # don't expire objects after commit (simpler)
            class_=AsyncSession,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in an endpoint:
        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Document))

    The 'async with' / 'yield' pattern ensures:
    - Session is committed if no exception occurred
    - Session is rolled back if an exception occurred
    - Session is always closed (connection returned to pool)
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize the database — create all tables if they don't exist.

    Called once at application startup (in main.py lifespan).

    In production, use Alembic migrations instead of create_all()
    so that column additions don't drop existing data.
    """
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db() -> None:
    """Close the database engine and all connections. Called at shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")


# ─────────────────────────────────────────────────────────────────────────────
# ORM Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Enum Types
# ─────────────────────────────────────────────────────────────────────────────

class DocumentStatusEnum(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models (map to database tables)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentModel(Base):
    """
    Represents an uploaded document in PostgreSQL.

    Table: documents
    One document → many chunks (one-to-many relationship)
    """
    __tablename__ = "documents"

    document_id = Column(String(50), primary_key=True, index=True)
    file_name = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False, default=0)
    storage_path = Column(Text, nullable=True)          # MinIO path
    status = Column(
        Enum(DocumentStatusEnum),
        nullable=False,
        default=DocumentStatusEnum.PENDING,
    )
    chunk_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship: one document has many chunks
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document {self.document_id} {self.file_name} [{self.status}]>"


class ChunkModel(Base):
    """
    Represents one processed chunk of a document.

    Table: chunks
    One chunk belongs to one document.
    The chunk_text is stored here; the embedding vector is in Qdrant.
    The embedding_id links this chunk to its Qdrant point.
    """
    __tablename__ = "chunks"

    chunk_id = Column(String(50), primary_key=True, index=True)
    document_id = Column(
        String(50),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name = Column(String(500), nullable=False)
    source_type = Column(String(20), nullable=False)    # "pdf", "docx", etc.
    page_number = Column(Integer, nullable=True)
    slide_number = Column(Integer, nullable=True)
    sheet_name = Column(String(200), nullable=True)
    timestamp_start = Column(Integer, nullable=True)    # seconds (audio/video)
    timestamp_end = Column(Integer, nullable=True)
    section_title = Column(Text, nullable=True)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(String(100), nullable=True)   # Qdrant point ID
    chunk_index = Column(Integer, nullable=False, default=0)  # position in document
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship: chunk belongs to document
    document = relationship("DocumentModel", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk {self.chunk_id} doc={self.document_id} page={self.page_number}>"


class SessionModel(Base):
    """
    Represents a chat session (a conversation thread).

    Table: sessions
    One session has many messages.
    """
    __tablename__ = "sessions"

    session_id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship: session has many messages
    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan",
                             order_by="MessageModel.created_at")


class MessageModel(Base):
    """
    Represents one message in a chat session.

    Table: messages
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(100),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)           # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship: message belongs to session
    session = relationship("SessionModel", back_populates="messages")


# ─────────────────────────────────────────────────────────────────────────────
# Repository functions — wraps raw SQLAlchemy queries in named functions
#
# Why a repository pattern?
#   - Keeps business logic (API handlers, agents) separate from SQL
#   - Easy to swap the database without changing the API handler
#   - Easy to unit-test (mock the repository)
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy import select, update, delete


async def insert_document(db: AsyncSession, doc: DocumentModel) -> None:
    """Insert a new document record."""
    db.add(doc)
    await db.flush()  # write to DB within the current transaction


async def get_document_by_id(db: AsyncSession, document_id: str) -> Optional[DocumentModel]:
    """Fetch a document by its ID."""
    result = await db.execute(select(DocumentModel).where(DocumentModel.document_id == document_id))
    return result.scalar_one_or_none()


async def list_all_documents(db: AsyncSession) -> List[DocumentModel]:
    """Return all documents ordered by creation date (newest first)."""
    result = await db.execute(
        select(DocumentModel).order_by(DocumentModel.created_at.desc())
    )
    return list(result.scalars().all())


async def update_document_status(
    db: AsyncSession,
    document_id: str,
    status: DocumentStatusEnum,
    chunk_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Update a document's processing status."""
    await db.execute(
        update(DocumentModel)
        .where(DocumentModel.document_id == document_id)
        .values(
            status=status,
            chunk_count=chunk_count,
            error_message=error_message,
            updated_at=datetime.now(timezone.utc),
        )
    )


async def delete_document_by_id(db: AsyncSession, document_id: str) -> None:
    """Delete a document and all its chunks (CASCADE handles chunks)."""
    await db.execute(delete(DocumentModel).where(DocumentModel.document_id == document_id))


async def insert_chunks(db: AsyncSession, chunks: List[ChunkModel]) -> None:
    """Insert multiple chunks in one batch operation (faster than one at a time)."""
    db.add_all(chunks)
    await db.flush()


async def get_chunks_for_document(db: AsyncSession, document_id: str) -> List[ChunkModel]:
    """Return all chunks for a document, ordered by position."""
    result = await db.execute(
        select(ChunkModel)
        .where(ChunkModel.document_id == document_id)
        .order_by(ChunkModel.chunk_index)
    )
    return list(result.scalars().all())


async def upsert_session(db: AsyncSession, session_id: str, user_id: str) -> SessionModel:
    """Get or create a session."""
    existing = await db.get(SessionModel, session_id)
    if existing:
        return existing
    session = SessionModel(session_id=session_id, user_id=user_id)
    db.add(session)
    await db.flush()
    return session


async def insert_message(db: AsyncSession, session_id: str, role: str, content: str) -> MessageModel:
    """Append a message to a session."""
    msg = MessageModel(session_id=session_id, role=role, content=content)
    db.add(msg)
    await db.flush()
    return msg


async def get_messages_for_session(db: AsyncSession, session_id: str, limit: int = 50) -> List[MessageModel]:
    """Return recent messages for a session."""
    result = await db.execute(
        select(MessageModel)
        .where(MessageModel.session_id == session_id)
        .order_by(MessageModel.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars().all())
