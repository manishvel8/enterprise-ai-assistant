"""
db/neo4j_client.py — Neo4j driver, schema setup, and Cypher query helpers.

Graph Schema:
  (:Document {document_id, file_name, file_type, created_at})
      -[:HAS_CHUNK]->
  (:Chunk {chunk_id, document_id, chunk_text_preview, page_number, section_title})
      -[:MENTIONS]->
  (:Entity {name, type, normalized_name})
      -[:RELATED_TO {relation_type}]->
  (:Entity)

  (:Chunk)-[:BELONGS_TO_TOPIC]->(:Topic {name})
  (:Decision {description})-[:SUPPORTED_BY]->(:Chunk)
  (:Action {description})-[:LINKED_TO]->(:Decision)

Node labels:
  Document, Chunk, Entity, Topic, Person, Organization, Metric, Decision, Action

Why a graph database?
  Relational databases (PostgreSQL) are great for tabular data.
  Vector databases (Qdrant) are great for semantic similarity.
  But neither can efficiently traverse relationships like:
    "Which people mentioned in document A are related to decisions in document B?"
    "What metrics are associated with this organization?"
  Neo4j stores these as first-class relationships and traverses them in O(1).

GraphRAG workflow:
  1. Retriever finds chunks via vector search (Qdrant)
  2. For each chunk, query Neo4j for related entities and topics
  3. Neo4j traversal enriches context beyond what vector search alone finds

Why Cypher?
  Cypher is Neo4j's declarative graph query language (like SQL for graphs).
  MATCH (e:Entity)-[:RELATED_TO]->(e2:Entity)
  WHERE e.name = 'OpenAI'
  RETURN e2.name, e2.type
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_driver = None


def get_neo4j_driver():
    """
    Get or create the global Neo4j driver (singleton pattern).

    The driver manages a connection pool. We create it once at startup
    and reuse it for all queries (thread-safe).
    """
    global _driver
    if _driver is not None:
        return _driver

    if not settings.neo4j_uri:
        logger.warning("NEO4J_URI not configured — Neo4j features disabled")
        return None

    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=20,
        )
        # Verify connectivity
        _driver.verify_connectivity()
        logger.info(f"Neo4j connected: {settings.neo4j_uri}")
        return _driver
    except ImportError:
        logger.error("neo4j package not installed. Run: pip install neo4j")
        return None
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return None


def close_neo4j_driver():
    """Close the Neo4j driver (called at app shutdown)."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


@contextmanager
def get_session():
    """Context manager for a Neo4j session."""
    driver = get_neo4j_driver()
    if not driver:
        raise RuntimeError("Neo4j driver not available")

    with driver.session() as session:
        yield session


def run_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a read-only Cypher query and return results as a list of dicts.

    Args:
        query: A Cypher query string
        parameters: Query parameters ($param syntax)

    Returns:
        List of result records as dicts.
        Empty list if Neo4j is unavailable or query returns no results.
    """
    driver = get_neo4j_driver()
    if not driver:
        return []

    try:
        with driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    except Exception as e:
        logger.error(f"Cypher query failed: {e}\nQuery: {query[:200]}")
        return []


def run_cypher_write(query: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
    """
    Execute a write Cypher query (CREATE / MERGE / SET / DELETE).

    Returns True on success, False on failure.
    """
    driver = get_neo4j_driver()
    if not driver:
        return False

    try:
        with driver.session() as session:
            session.run(query, parameters or {})
            return True
    except Exception as e:
        logger.error(f"Cypher write failed: {e}\nQuery: {query[:200]}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Schema Setup — Constraints and Indexes
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Uniqueness constraints (also create indexes automatically)
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.normalized_name, e.type) IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE",

    # Full-text search index on Entity names (for fuzzy search)
    """CREATE FULLTEXT INDEX entity_name_fts IF NOT EXISTS
       FOR (e:Entity|Person|Organization|Metric)
       ON EACH [e.name]""",

    # Range index on chunk page_number for page-scoped queries
    "CREATE INDEX chunk_page IF NOT EXISTS FOR (c:Chunk) ON (c.page_number)",
    "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
]


def setup_schema() -> bool:
    """
    Apply all schema constraints and indexes.

    Should be called once at application startup.
    All queries use IF NOT EXISTS so they are idempotent.

    Returns True if all constraints were applied successfully.
    """
    driver = get_neo4j_driver()
    if not driver:
        logger.warning("Neo4j not available — skipping schema setup")
        return False

    success = True
    for query in SCHEMA_QUERIES:
        try:
            with driver.session() as session:
                session.run(query)
            logger.debug(f"Schema applied: {query[:60]}...")
        except Exception as e:
            logger.error(f"Schema query failed: {e}\nQuery: {query}")
            success = False

    if success:
        logger.info("Neo4j schema setup complete")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Document and Chunk Node Operations
# ─────────────────────────────────────────────────────────────────────────────

def create_document_node(document_id: str, file_name: str, file_type: str) -> bool:
    """Create or update a Document node in Neo4j."""
    return run_cypher_write(
        """
        MERGE (d:Document {document_id: $document_id})
        SET d.file_name = $file_name,
            d.file_type = $file_type,
            d.created_at = datetime()
        """,
        {"document_id": document_id, "file_name": file_name, "file_type": file_type},
    )


def create_chunk_node(
    chunk_id: str,
    document_id: str,
    chunk_text_preview: str,
    page_number: Optional[int] = None,
    section_title: Optional[str] = None,
) -> bool:
    """Create or update a Chunk node and link it to its Document."""
    return run_cypher_write(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.document_id = $document_id,
            c.text_preview = $text_preview,
            c.page_number = $page_number,
            c.section_title = $section_title
        WITH c
        MATCH (d:Document {document_id: $document_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text_preview": chunk_text_preview[:300],
            "page_number": page_number,
            "section_title": section_title,
        },
    )


def delete_document_graph(document_id: str) -> bool:
    """
    Delete all Neo4j nodes and relationships for a document.

    Removes: Document node, all its Chunk nodes, and relationships.
    Orphaned Entity/Topic nodes that were only mentioned in this document
    are also cleaned up.
    """
    return run_cypher_write(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[r]-()
        DELETE r, c, d
        """,
        {"document_id": document_id},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query Helpers for GraphRAG
# ─────────────────────────────────────────────────────────────────────────────

def get_entities_for_chunk(chunk_id: str) -> List[Dict[str, Any]]:
    """
    Get all entities mentioned in a specific chunk.

    Used by GraphRAG to enrich chunk context with entity relationships.
    """
    return run_cypher(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN e.name AS name, e.type AS type, e.normalized_name AS normalized_name
        LIMIT 20
        """,
        {"chunk_id": chunk_id},
    )


def get_related_entities(entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
    """
    Find entities related to a given entity within N hops.

    depth=2 means: entity → related_entity → further_related_entity

    Example:
        get_related_entities("OpenAI") might return:
        - GPT-4 (product)
        - Sam Altman (person)
        - Microsoft (organization)
    """
    return run_cypher(
        """
        MATCH (e:Entity {normalized_name: toLower($entity_name)})
        MATCH path = (e)-[:RELATED_TO*1..{depth}]-(related:Entity)
        RETURN DISTINCT
            related.name AS name,
            related.type AS type,
            length(path) AS distance
        ORDER BY distance ASC
        LIMIT 30
        """.replace("{depth}", str(depth)),
        {"entity_name": entity_name.lower()},
    )


def get_document_graph_summary(document_id: str) -> Dict[str, Any]:
    """
    Get a summary of the graph structure for a document.

    Used by the debug panel to show what the graph looks like.
    """
    results = run_cypher(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (c)-[:BELONGS_TO_TOPIC]->(t:Topic)
        RETURN
            count(DISTINCT c) AS chunk_count,
            count(DISTINCT e) AS entity_count,
            count(DISTINCT t) AS topic_count
        """,
        {"document_id": document_id},
    )

    if results:
        return results[0]
    return {"chunk_count": 0, "entity_count": 0, "topic_count": 0}


def health_check() -> bool:
    """Return True if Neo4j is reachable."""
    driver = get_neo4j_driver()
    if not driver:
        return False
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return True
    except Exception:
        return False
