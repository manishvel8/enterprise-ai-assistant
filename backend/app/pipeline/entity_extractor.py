"""
pipeline/entity_extractor.py — Extract named entities and relationships from chunk text using GPT-4o.

Why LLM-based extraction instead of spaCy NER?
  - spaCy recognizes standard NER labels: PERSON, ORG, GPE, DATE, etc.
  - LLM-based extraction can identify domain-specific entities: Metric, Decision, Action, Topic
  - LLM understands relationships between entities (not just presence)
  - More flexible — works across industries without fine-tuning

Output format:
  {
    "entities": [
      {"name": "OpenAI", "type": "Organization"},
      {"name": "GPT-4o", "type": "Entity"},
      {"name": "Sam Altman", "type": "Person"}
    ],
    "relationships": [
      {"from": "Sam Altman", "to": "OpenAI", "relation": "CEO_OF"},
      {"from": "OpenAI", "to": "GPT-4o", "relation": "CREATED"}
    ],
    "topics": ["artificial intelligence", "language models"],
    "decisions": ["Deploy GPT-4o to production by Q3 2024"],
    "metrics": ["99.9% uptime SLA", "$20/month pricing"]
  }

Cost optimization:
  - We only extract entities for chunks that contain substantive text (>100 chars)
  - We use GPT-4o-mini (faster, cheaper) for extraction (not GPT-4o)
  - Results are cached so the same chunk isn't re-processed on re-upload
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.models.document import ChunkSchema

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """Extract named entities and relationships from the following text chunk.

Return a JSON object with this structure:
{
  "entities": [{"name": "...", "type": "Person|Organization|Metric|Decision|Action|Topic|Entity"}],
  "relationships": [{"from": "entity_name", "to": "entity_name", "relation": "RELATION_TYPE"}],
  "topics": ["topic1", "topic2"],
  "decisions": ["decision description"],
  "metrics": ["metric or KPI with value"]
}

Rules:
- Only include entities that are explicitly mentioned in the text
- Relation types should be SCREAMING_SNAKE_CASE (e.g. CEO_OF, WORKS_AT, RELATED_TO)
- Topics are high-level themes (1-3 word phrases)
- Decisions are concrete choices or conclusions made in the text
- Metrics are quantitative measurements (include the actual numbers)
- Return valid JSON only — no explanation text

Text chunk:
{chunk_text}"""


async def extract_entities(chunk: ChunkSchema) -> Optional[Dict[str, Any]]:
    """
    Extract entities and relationships from a single chunk.

    Returns:
        Dictionary with entities, relationships, topics, decisions, metrics.
        Returns None if extraction fails or chunk text is too short.
    """
    if len(chunk.chunk_text) < 100:
        return None   # Too short to be worth extracting

    from app.services.openai_service import get_openai_client
    from app.core.config import settings

    if not settings.openai_api_key:
        return None

    prompt = ENTITY_EXTRACTION_PROMPT.replace("{chunk_text}", chunk.chunk_text[:3000])

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",   # cheaper model for extraction
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise entity extraction system. Always return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,       # deterministic extraction
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        logger.debug(
            f"Extracted from chunk {chunk.chunk_id}: "
            f"{len(result.get('entities', []))} entities, "
            f"{len(result.get('relationships', []))} relationships"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Entity extraction returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Entity extraction failed for chunk {chunk.chunk_id}: {e}")
        return None


def extract_entities_sync(chunk: ChunkSchema) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for use in Celery workers."""
    import asyncio
    return asyncio.run(extract_entities(chunk))


async def extract_entities_batch(chunks: List[ChunkSchema]) -> List[Optional[Dict[str, Any]]]:
    """
    Extract entities from multiple chunks concurrently.

    Uses asyncio.gather() for parallel API calls (bounded by OpenAI rate limits).
    """
    import asyncio
    tasks = [extract_entities(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Entity extraction task failed: {r}")
            processed.append(None)
        else:
            processed.append(r)

    return processed
