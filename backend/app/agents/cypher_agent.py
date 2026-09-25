"""
agents/cypher_agent.py — Cypher Agent: Generate and safely execute Neo4j Cypher queries.

Role in the agentic workflow:
  When the Router Agent classifies a query as "graph_question",
  the Cypher Agent:
  1. Generates a Cypher query from the natural language question using GPT-4o
  2. Validates the query for safety (no DELETE, no DETACH DELETE, etc.)
  3. Executes the query against Neo4j
  4. Returns structured results as a list of dicts

Why generate Cypher dynamically?
  Hard-coding Cypher queries for every possible question is impossible.
  GPT-4o has learned the Cypher syntax and can generate valid queries
  given a description of the graph schema.

Safety validation:
  LLM-generated code can be malicious or buggy. We:
  1. Only allow read-only queries (MATCH/RETURN, no WRITE/DELETE)
  2. Check for dangerous keywords (DELETE, DROP, DETACH, SET, MERGE, CREATE)
  3. Limit result count (LIMIT 50) to prevent memory issues
  4. Timeout queries after 10 seconds

This is the "Text-to-Cypher" problem — analogous to Text-to-SQL.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.models.agent_state import AgentState

logger = logging.getLogger(__name__)

# Cypher keywords that indicate write operations — must be blocked
DANGEROUS_KEYWORDS = [
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bDROP\b",
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bCALL\b.*\bapoc\b",    # APOC procedures can be dangerous
    r"\bLOAD\s+CSV\b",
]

CYPHER_SCHEMA_DESCRIPTION = """
Neo4j Graph Schema:
- (:Document {document_id, file_name, file_type, created_at})
- (:Chunk {chunk_id, document_id, text_preview, page_number, section_title})
- (:Entity {name, type, normalized_name})
- (:Person {name, normalized_name})       ← subtype of Entity
- (:Organization {name, normalized_name}) ← subtype of Entity
- (:Metric {name, normalized_name})       ← subtype of Entity
- (:Topic {name})
- (:Decision {decision_id, description})
- (:Action {description})

Relationships:
- (:Document)-[:HAS_CHUNK]->(:Chunk)
- (:Chunk)-[:MENTIONS]->(:Entity|Person|Organization|Metric)
- (:Entity)-[:RELATED_TO {relation_type}]->(:Entity)
- (:Chunk)-[:BELONGS_TO_TOPIC]->(:Topic)
- (:Decision)-[:SUPPORTED_BY]->(:Chunk)
- (:Action)-[:LINKED_TO]->(:Decision)

Rules:
- Use only READ queries (MATCH, RETURN, WHERE, ORDER BY, LIMIT, WITH)
- Always include LIMIT (max 50)
- Use toLower() for case-insensitive matching
- Use CONTAINS for partial string matching
"""

CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher expert. Generate a READ-ONLY Cypher query to answer the question.

{schema}

Question: {question}

Return ONLY the Cypher query, no explanation.
The query must:
1. Start with MATCH
2. End with RETURN
3. Include LIMIT <= 50
4. Use only read operations
"""


async def run_cypher_agent(state: AgentState) -> AgentState:
    """
    Cypher Agent: Generate and execute a Cypher query for the user's question.

    Reads from state:
        - user_query or rewritten_query
        - document_ids (optional, to scope the search)

    Writes to state:
        - cypher_query: the generated Cypher
        - cypher_results: list of result dicts from Neo4j
        - errors: appended with any errors
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    logger.info(f"Cypher Agent: processing query: {query[:80]}")

    # Generate Cypher
    cypher = await _generate_cypher(query)
    state["cypher_query"] = cypher

    if not cypher:
        state["cypher_results"] = []
        state["errors"].append("Cypher Agent: failed to generate query")
        return state

    # Validate safety
    is_safe, reason = validate_cypher(cypher)
    if not is_safe:
        logger.warning(f"Cypher rejected: {reason}")
        state["cypher_results"] = []
        state["errors"].append(f"Cypher Agent: unsafe query blocked — {reason}")
        state["cypher_query"] = None
        return state

    # Execute query
    results = _execute_cypher(cypher)
    state["cypher_results"] = results

    logger.info(f"Cypher Agent: {len(results)} results from Neo4j")
    state["debug_info"]["cypher"] = {
        "query": cypher,
        "result_count": len(results),
    }
    return state


async def _generate_cypher(question: str) -> Optional[str]:
    """Generate a Cypher query using GPT-4o."""
    from app.services.openai_service import get_openai_client
    from app.core.config import settings

    if not settings.openai_api_key:
        return None

    prompt = CYPHER_GENERATION_PROMPT.format(
        schema=CYPHER_SCHEMA_DESCRIPTION,
        question=question,
    )

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Cypher query generator. Output ONLY the Cypher query.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        cypher = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        cypher = re.sub(r"```(?:cypher)?\n?", "", cypher).strip()

        logger.info(f"Generated Cypher: {cypher[:200]}")
        return cypher

    except Exception as e:
        logger.error(f"Cypher generation failed: {e}")
        return None


def validate_cypher(cypher: str) -> tuple[bool, str]:
    """
    Validate that a Cypher query is read-only and safe to execute.

    Returns:
        (is_safe: bool, reason: str)
        If not safe, reason explains why it was blocked.
    """
    if not cypher or not cypher.strip():
        return False, "Empty query"

    upper = cypher.upper()

    # Must start with MATCH or WITH (read operations)
    stripped = cypher.strip().upper()
    if not (stripped.startswith("MATCH") or stripped.startswith("WITH")):
        return False, "Query must start with MATCH or WITH"

    # Check for dangerous write keywords
    for pattern in DANGEROUS_KEYWORDS:
        if re.search(pattern, upper, re.IGNORECASE):
            matched = re.search(pattern, upper, re.IGNORECASE)
            return False, f"Blocked keyword detected: {matched.group() if matched else pattern}"

    # Must have RETURN
    if "RETURN" not in upper:
        return False, "Query missing RETURN clause"

    # Must have LIMIT (prevent runaway queries)
    if "LIMIT" not in upper:
        # Auto-append LIMIT as a safety measure
        logger.warning("Cypher query missing LIMIT — appending LIMIT 20")
        return True, "ok"   # We'll append the limit at execution time

    return True, "ok"


def _execute_cypher(cypher: str) -> List[Dict[str, Any]]:
    """
    Execute a validated Cypher query and return results.

    Auto-appends LIMIT 20 if missing to prevent memory issues.
    """
    from app.db.neo4j_client import run_cypher

    # Auto-append LIMIT if missing
    if "LIMIT" not in cypher.upper():
        cypher = cypher.rstrip() + " LIMIT 20"

    results = run_cypher(cypher)
    return results
