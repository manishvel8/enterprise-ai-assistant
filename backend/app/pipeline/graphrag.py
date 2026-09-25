"""
pipeline/graphrag.py — GraphRAG: Combine vector retrieval and graph traversal.

What is GraphRAG?
  Standard RAG retrieves chunks by semantic similarity.
  GraphRAG additionally traverses the entity relationship graph to find
  related information that semantic search might miss.

  Example:
    Query: "What are the risks associated with our data vendor?"
    - Vector search finds chunks mentioning "data vendor risks"
    - Graph traversal finds: data_vendor → RELATED_TO → contract, SLA, breach_history
    - Combined context: better, more complete answer

How it works:
  1. Vector retrieval: find top-k most similar chunks (existing RAG)
  2. Extract entities from the retrieved chunks
  3. Traverse Neo4j graph: find entities related to those entities
  4. Find chunks that mention those related entities
  5. Add those additional chunks to the context (de-duplicated)
  6. Pass enriched context to the Answer Agent

This is "graph-enriched RAG" — the graph doesn't replace vector search,
it augments it by finding related context that similarity search misses.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from app.models.agent_state import RetrievedChunk

logger = logging.getLogger(__name__)


async def graphrag_retrieve(
    query: str,
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
) -> tuple[List[RetrievedChunk], Dict[str, Any]]:
    """
    GraphRAG retrieval: vector search + graph-enriched context.

    Args:
        query: User query (or rewritten query)
        top_k: Number of initial vector search results
        document_ids: Optional document filter

    Returns:
        (enriched_chunks, debug_info) tuple
        enriched_chunks includes both vector-retrieved and graph-enriched chunks
        debug_info contains entity graph and traversal info for the debug panel
    """
    from app.db.retriever import retrieve_chunks

    debug_info: Dict[str, Any] = {}

    # Step 1: Vector retrieval
    vector_chunks = await retrieve_chunks(
        query=query,
        top_k=top_k,
        document_ids=document_ids,
    )
    logger.info(f"GraphRAG: {len(vector_chunks)} chunks from vector search")
    debug_info["vector_chunks"] = len(vector_chunks)

    if not vector_chunks:
        return [], debug_info

    # Step 2: Extract entity names from retrieved chunks
    entity_names = _extract_entity_names_from_chunks(vector_chunks)
    logger.info(f"GraphRAG: found {len(entity_names)} entities in chunks")
    debug_info["entities_found"] = list(entity_names)

    if not entity_names:
        return vector_chunks, debug_info

    # Step 3: Find related entities via Neo4j graph traversal
    related_entities = _get_related_entity_names(entity_names)
    logger.info(f"GraphRAG: {len(related_entities)} related entities from graph")
    debug_info["related_entities"] = list(related_entities)

    if not related_entities:
        return vector_chunks, debug_info

    # Step 4: Find additional chunks that mention the related entities
    graph_enriched_chunks = await _retrieve_chunks_for_entities(
        related_entities,
        exclude_chunk_ids={c["chunk_id"] if isinstance(c, dict) else c.chunk_id for c in vector_chunks},
        document_ids=document_ids,
    )
    logger.info(f"GraphRAG: {len(graph_enriched_chunks)} additional chunks from graph enrichment")
    debug_info["graph_enriched_chunks"] = len(graph_enriched_chunks)

    # Step 5: Merge results (vector chunks first, graph chunks second)
    all_chunks = vector_chunks + graph_enriched_chunks
    debug_info["total_chunks"] = len(all_chunks)

    return all_chunks, debug_info


def _extract_entity_names_from_chunks(chunks: List[RetrievedChunk]) -> Set[str]:
    """
    Find entities mentioned in a list of retrieved chunks by querying Neo4j.

    Returns a set of entity normalized_names.
    """
    from app.db.neo4j_client import get_entities_for_chunk

    entity_names: Set[str] = set()
    for chunk in chunks:
        chunk_id = chunk["chunk_id"] if isinstance(chunk, dict) else chunk.chunk_id
        entities = get_entities_for_chunk(chunk_id)
        for e in entities:
            if e.get("normalized_name"):
                entity_names.add(e["normalized_name"])

    return entity_names


def _get_related_entity_names(entity_names: Set[str]) -> Set[str]:
    """
    Find entities related to any of the given entity names in the Neo4j graph.

    Traverses up to 2 hops from each entity.
    Returns normalized names of all discovered related entities.
    """
    from app.db.neo4j_client import get_related_entities

    related: Set[str] = set()
    for name in entity_names:
        relatives = get_related_entities(name, depth=2)
        for r in relatives:
            if r.get("name"):
                related.add(r["name"].lower())

    # Remove entities we already have
    return related - entity_names


async def _retrieve_chunks_for_entities(
    entity_names: Set[str],
    exclude_chunk_ids: Set[str],
    document_ids: Optional[List[str]],
) -> List[RetrievedChunk]:
    """
    For each entity name, embed it and retrieve similar chunks.

    This finds chunks that discuss the related entities discovered via graph traversal.
    """
    from app.db.retriever import retrieve_chunks

    graph_chunks: List[RetrievedChunk] = []
    seen_chunk_ids: Set[str] = set(exclude_chunk_ids)

    # Limit to top 3 entities to avoid too many API calls
    for entity_name in list(entity_names)[:3]:
        chunks = await retrieve_chunks(
            query=entity_name,
            top_k=3,
            score_threshold=0.4,   # slightly lower threshold for graph-enriched results
            document_ids=document_ids,
        )

        for chunk in chunks:
            chunk_id = chunk["chunk_id"] if isinstance(chunk, dict) else chunk.chunk_id
            if chunk_id not in seen_chunk_ids:
                graph_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

    return graph_chunks


def build_graphrag_context(
    chunks: List[RetrievedChunk],
    cypher_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build the final context string from both vector chunks and Cypher results.

    Used by the Context Builder Agent to assemble the prompt context.

    Format:
        --- Document Context ---
        [1] Source: file.pdf, Page 3
        chunk text...

        --- Graph Query Results ---
        Entity: OpenAI
        Related: Microsoft (PARTNER_OF), GPT-4 (PRODUCT_OF)
    """
    parts = []

    # Vector/graph-enriched chunk context
    if chunks:
        parts.append("--- Document Context ---")
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = chunk["chunk_text"] if isinstance(chunk, dict) else chunk.chunk_text
            file_name = chunk["file_name"] if isinstance(chunk, dict) else chunk.file_name
            page_number = chunk.get("page_number") if isinstance(chunk, dict) else chunk.page_number

            source = file_name
            if page_number:
                source += f", Page {page_number}"
            section_title = (
                chunk.get("section_title") if isinstance(chunk, dict) else getattr(chunk, "section_title", None)
            )
            if section_title:
                source += f", Section: {section_title}"

            # Keep enough text for grounding checks (resume facts often appear mid-page)
            parts.append(f"[{i}] Source: {source}\n{chunk_text[:4000]}")

    # Cypher graph results
    if cypher_results:
        parts.append("\n--- Graph Query Results ---")
        for row in cypher_results[:10]:   # limit to 10 rows
            parts.append(str(row))

    return "\n\n".join(parts)
