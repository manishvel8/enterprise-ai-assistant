"""
app/database/db_client.py  —  PostgreSQL connection pool (psycopg2)

─────────────────────────────────────────────────────────────────────────────
WHY A CONNECTION POOL?
  Opening a new PostgreSQL connection takes ~50-100ms (TCP handshake, auth,
  SSL negotiation). For a chat application handling 100 requests/second,
  that overhead is unacceptable.

  A connection POOL pre-opens N connections and reuses them.
  Each request borrows a connection, uses it, and returns it.
  This reduces connection setup time to <1ms.

POOL SIZING RULES:
  - For 10 concurrent users:     min=2,  max=5
  - For 1,000 concurrent users:  min=5,  max=20
  - For 100K concurrent users:   use PgBouncer (external connection pooler)
  - PostgreSQL itself handles ~200 max connections before performance drops

HOW TO USE:
  from app.database.db_client import get_db, execute, fetchall, fetchone

  # Simple query
  rows = fetchall("SELECT * FROM messages WHERE conversation_id = %s", (conv_id,))

  # Transaction
  with get_db() as conn:
      with conn.cursor() as cur:
          cur.execute("INSERT INTO messages ...", (...))
          conn.commit()
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ── Lazy import — psycopg2 only needed when DB features are used ───────────────
try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    logger.warning(
        "psycopg2 not installed. Run: pip install psycopg2-binary  "
        "Database features (conversation history, memory) will be disabled."
    )

try:
    from pgvector.psycopg2 import register_vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False
    logger.warning(
        "pgvector not installed. Run: pip install pgvector  "
        "Semantic memory search will be disabled."
    )


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseClient — singleton managing the connection pool
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseClient:
    """
    Manages a psycopg2 ThreadedConnectionPool.

    ThreadedConnectionPool is thread-safe: multiple threads can borrow
    connections simultaneously without data corruption.
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._enabled: bool = False

    def init(self, dsn: str, min_conn: int = 2, max_conn: int = 10) -> None:
        """
        Create the connection pool. Call once at application startup.

        DSN format: "host=localhost port=5433 dbname=ai_assistant user=ai_user password=..."
        """
        if not _PSYCOPG2_AVAILABLE:
            return

        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                dsn=dsn,
                cursor_factory=psycopg2.extras.RealDictCursor,  # returns dicts, not tuples
            )
            self._enabled = True
            logger.info("PostgreSQL connection pool created (min=%d, max=%d)", min_conn, max_conn)

            # Test the connection
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()
                    logger.info("PostgreSQL connected: %s", version["version"][:50] if version else "unknown")

                    # Register pgvector type if available
                    if _PGVECTOR_AVAILABLE:
                        register_vector(conn)
                        logger.info("pgvector extension registered.")

        except Exception as exc:
            logger.error("Failed to create DB connection pool: %s", exc)
            logger.warning("Conversation history and memory features will be unavailable.")
            self._enabled = False

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """
        Borrow a connection from the pool, yield it, then return it.

        USAGE:
          with db.connection() as conn:
              with conn.cursor() as cur:
                  cur.execute("SELECT 1")

        IMPORTANT: The connection is NOT automatically committed.
        Call conn.commit() explicitly, or use the execute() helper below.
        """
        if not self._enabled or self._pool is None:
            raise RuntimeError("Database not available. Is PostgreSQL running?")

        conn = self._pool.getconn()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def execute(self, query: str, params: tuple = ()) -> Optional[list[dict]]:
        """
        Execute a write query (INSERT/UPDATE/DELETE) and commit.
        Returns None (no rows to return).

        USAGE:
          db.execute(
              "INSERT INTO users (name) VALUES (%s)",
              ("Manish",)
          )
        """
        if not self._enabled:
            return None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        return None

    def fetchall(self, query: str, params: tuple = ()) -> list[dict]:
        """
        Execute a SELECT and return all rows as list of dicts.

        USAGE:
          rows = db.fetchall(
              "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at",
              (conv_id,)
          )
          for row in rows:
              print(row["content"])   # dict access
        """
        if not self._enabled:
            return []
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall() or []

    def fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        """
        Execute a SELECT and return the first row as a dict, or None.

        USAGE:
          user = db.fetchone(
              "SELECT * FROM users WHERE external_id = %s",
              ("user_123",)
          )
          if user:
              print(user["name"])
        """
        if not self._enabled:
            return None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    def fetchone_value(self, query: str, params: tuple = ()) -> Any:
        """Fetch a single scalar value (e.g. COUNT, UUID)."""
        row = self.fetchone(query, params)
        if row is None:
            return None
        return list(row.values())[0]

    def execute_returning(self, query: str, params: tuple = ()) -> Optional[dict]:
        """
        Execute INSERT ... RETURNING * and return the inserted row.

        USAGE:
          new_user = db.execute_returning(
              "INSERT INTO users (name) VALUES (%s) RETURNING *",
              ("Manish",)
          )
          print(new_user["id"])  # the generated UUID
        """
        if not self._enabled:
            return None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
            conn.commit()
        return row

    def execute_vector_search(
        self,
        table: str,
        embedding_col: str,
        query_embedding: list[float],
        filter_col: str,
        filter_val: Any,
        top_k: int = 5,
        threshold: float = 0.3,
        extra_cols: str = "*",
    ) -> list[dict]:
        """
        Perform a pgvector cosine similarity search.

        USAGE:
          results = db.execute_vector_search(
              table="memory",
              embedding_col="embedding",
              query_embedding=[0.1, 0.2, ...],  # 1536 floats
              filter_col="user_id",
              filter_val=user_id,
              top_k=5,
          )

        HOW IT WORKS:
          pgvector's <=> operator computes cosine distance.
          Distance 0.0 = identical, 2.0 = completely opposite.
          1 - distance = similarity (0.0 to 1.0).
          We filter: WHERE distance < threshold (i.e. similarity > 1-threshold).
        """
        if not self._enabled or not _PGVECTOR_AVAILABLE:
            return []

        # Format embedding as PostgreSQL vector literal
        vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        query = f"""
            SELECT {extra_cols},
                   1 - ({embedding_col} <=> %s::vector) AS similarity
            FROM {table}
            WHERE {filter_col} = %s
              AND {embedding_col} IS NOT NULL
              AND 1 - ({embedding_col} <=> %s::vector) > %s
            ORDER BY {embedding_col} <=> %s::vector
            LIMIT %s
        """
        params = (vec_str, filter_val, vec_str, threshold, vec_str, top_k)
        return self.fetchall(query, params)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        """Close all connections in the pool. Call at application shutdown."""
        if self._pool is not None:
            self._pool.closeall()
            logger.info("Database connection pool closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_db: Optional[DatabaseClient] = None


def get_db() -> DatabaseClient:
    """Return the global DatabaseClient singleton."""
    global _db
    if _db is None:
        _db = DatabaseClient()
    return _db


def init_db(dsn: str, min_conn: int = 2, max_conn: int = 10) -> DatabaseClient:
    """
    Initialise the global DB client. Call once in main.py lifespan.

    EXAMPLE (main.py):
      from app.database.db_client import init_db, get_db

      @asynccontextmanager
      async def lifespan(app: FastAPI):
          init_db(settings.conversation_db_dsn)
          yield
          get_db().close()
    """
    db = get_db()
    db.init(dsn, min_conn, max_conn)
    return db
