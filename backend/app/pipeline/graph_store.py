"""
pipeline/graph_store.py — Store extracted entities and relationships in Neo4j.

This module bridges the gap between:
  - entity_extractor.py (which uses LLM to extract entities)
  - neo4j_client.py (which provides raw Cypher execution)

For each document chunk, we:
  1. Create/merge the Chunk node
  2. Create/merge Entity nodes (deduplicated by normalized_name + type)
  3. Create MENTIONS relationships (Chunk → Entity)
  4. Create RELATED_TO relationships (Entity → Entity)
  5. Create/merge Topic nodes and BELONGS_TO_TOPIC relationships
  6. Create Decision and Action nodes if present

Why MERGE instead of CREATE?
  MERGE = CREATE if not exists, otherwise match existing.
  This is idempotent — re-processing a document doesn't create duplicate nodes.
  An "OpenAI" entity in document A and document B become the same node.
  This is what makes the graph useful: cross-document relationships.
"""

import logging
from typing import Any, Dict, List, Optional

from app.models.document import ChunkSchema

logger = logging.getLogger(__name__)


def store_chunk_graph(
    chunk: ChunkSchema,
    entities_data: Optional[Dict[str, Any]],
) -> bool:
    """
    Store a chunk and its extracted entities in Neo4j.

    Args:
        chunk: The chunk metadata (from chunker)
        entities_data: Extracted entities from entity_extractor.py

    Returns:
        True if successful, False if Neo4j is unavailable.
    """
    from app.db.neo4j_client import get_neo4j_driver

    driver = get_neo4j_driver()
    if not driver:
        return False

    if not entities_data:
        return True   # No entities to store, but not a failure

    try:
        with driver.session() as session:
            # Store entities and relationships in a single transaction
            session.execute_write(
                _store_chunk_transaction,
                chunk,
                entities_data,
            )
        return True
    except Exception as e:
        logger.error(f"Failed to store chunk graph for {chunk.chunk_id}: {e}")
        return False


def _store_chunk_transaction(tx, chunk: ChunkSchema, entities_data: Dict[str, Any]):
    """
    Run all graph writes for one chunk inside a single transaction.
    If any query fails, the entire transaction rolls back.
    """
    entities = entities_data.get("entities", [])
    relationships = entities_data.get("relationships", [])
    topics = entities_data.get("topics", [])
    decisions = entities_data.get("decisions", [])

    # 1. Ensure Chunk node exists
    tx.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.document_id = $document_id,
            c.text_preview = $text_preview,
            c.page_number = $page_number,
            c.section_title = $section_title
        """,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text_preview=chunk.chunk_text[:300],
        page_number=chunk.page_number,
        section_title=chunk.section_title,
    )

    # 2. Store entities and MENTIONS relationships
    for entity in entities:
        name = entity.get("name", "").strip()
        entity_type = entity.get("type", "Entity").strip()
        if not name:
            continue

        normalized = name.lower()

        # Create entity node (use type as label for efficient querying)
        label = _sanitize_label(entity_type)
        tx.run(
            f"""
            MERGE (e:{label} {{normalized_name: $normalized_name}})
            SET e.name = $name,
                e.type = $type
            WITH e
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            normalized_name=normalized,
            name=name,
            type=entity_type,
            chunk_id=chunk.chunk_id,
        )

    # 3. Store RELATED_TO relationships between entities
    for rel in relationships:
        from_name = rel.get("from", "").strip().lower()
        to_name = rel.get("to", "").strip().lower()
        relation = rel.get("relation", "RELATED_TO").strip().upper()

        if not from_name or not to_name:
            continue

        # Sanitize relation type (only letters and underscores)
        relation = "".join(c if c.isalnum() or c == "_" else "_" for c in relation)

        tx.run(
            f"""
            MATCH (from {{normalized_name: $from_name}})
            MATCH (to {{normalized_name: $to_name}})
            MERGE (from)-[r:RELATED_TO {{relation_type: $relation}}]->(to)
            """,
            from_name=from_name,
            to_name=to_name,
            relation=relation,
        )

    # 4. Store Topic nodes
    for topic in topics:
        if not topic or not isinstance(topic, str):
            continue
        tx.run(
            """
            MERGE (t:Topic {name: $name})
            WITH t
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:BELONGS_TO_TOPIC]->(t)
            """,
            name=topic.lower().strip(),
            chunk_id=chunk.chunk_id,
        )

    # 5. Store Decision nodes
    import uuid
    for decision_text in decisions:
        if not decision_text or not isinstance(decision_text, str):
            continue
        decision_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.chunk_id}:{decision_text[:100]}"))
        tx.run(
            """
            MERGE (d:Decision {decision_id: $decision_id})
            SET d.description = $description
            WITH d
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (d)-[:SUPPORTED_BY]->(c)
            """,
            decision_id=decision_id,
            description=decision_text[:500],
            chunk_id=chunk.chunk_id,
        )


def store_document_chunks_graph(
    chunks: List[ChunkSchema],
    entities_list: List[Optional[Dict[str, Any]]],
) -> int:
    """
    Store multiple chunks and their entities in Neo4j.

    Args:
        chunks: All chunks from a document
        entities_list: Parallel list of extracted entities (same order as chunks)

    Returns:
        Number of chunks successfully stored.
    """
    stored = 0
    for chunk, entities in zip(chunks, entities_list):
        if store_chunk_graph(chunk, entities):
            stored += 1

    logger.info(f"Stored {stored}/{len(chunks)} chunks in Neo4j graph")
    return stored


def _sanitize_label(label: str) -> str:
    """
    Convert an entity type to a valid Neo4j label.

    Neo4j labels can only contain letters, numbers, and underscores.
    They must start with a letter.

    Examples:
        "Person" → "Person"
        "Organization" → "Organization"
        "AI Model" → "AIModel"
        "123Invalid" → "Entity"
    """
    clean = "".join(c for c in label if c.isalnum())
    if not clean or not clean[0].isalpha():
        return "Entity"
    return clean
