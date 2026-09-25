"""
langgraph-agent/app/core/guardrails.py — LLM Guardrails for the LangGraph Agent.

Same purpose as backend/app/core/guardrails.py but integrated specifically
into the LangGraph workflow:
  - validate_query() is called in the query_validator_node (Node 1)
  - validate_response() is called in the quality_checker_node (Node 7)

ARCHITECTURE POSITION:
  User → /api/chat →
    query_validator_node →
      [guardrails.validate_query()] ← THIS FILE
      LLM validation call →
      router_node → ... → response_generator_node →
        quality_checker_node →
          [guardrails.validate_response()] ← THIS FILE
          → response returned

DIFFERENCES FROM BACKEND GUARDRAILS:
  The backend guardrails run at the HTTP layer (before the workflow).
  These guardrails run inside the graph (as part of the LangGraph nodes).
  This gives us double protection and fine-grained per-node control.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 3000          # Slightly higher for LangGraph (handles longer queries)
MAX_QUERY_LENGTH_HARD = 10_000
MIN_QUERY_LENGTH = 3
MAX_RESPONSE_LENGTH = 50_000

# ─────────────────────────────────────────────────────────────────────────────
# Prompt injection patterns (same as backend)
# ─────────────────────────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"override\s+your\s+(instructions?|programming|training|rules?|guidelines?)",
    r"you\s+are\s+now\s+(a\s+)?(DAN|jailbreak|evil|unrestricted|unconstrained)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(DAN|jailbreak|evil|unrestricted)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?different\s+(AI|model|assistant)",
    r"switch\s+to\s+(DAN|developer|admin|root|jailbreak)\s+mode",
    r"(repeat|print|reveal|show|tell\s+me|output)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt)",
    r"```\s*(system|user|assistant)\s*\n",
    r"<\|?(im_start|system|endoftext|pad|eos)\|?>",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "api_key": re.compile(r'\b(sk-[a-zA-Z0-9]{20,}|pk-[a-zA-Z0-9]{20,})\b'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailError(Exception):
    def __init__(self, message: str, code: str = "GUARDRAIL_BLOCKED"):
        super().__init__(message)
        self.code = code


@dataclass
class GuardrailResult:
    is_safe: bool
    reason: Optional[str] = None
    sanitized_text: Optional[str] = None
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Validation functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_query(query: str) -> GuardrailResult:
    """
    Validate user query before it enters the LangGraph workflow.

    Called in query_validator_node BEFORE the LLM validation call.
    If this raises GuardrailError, the graph ends immediately (no LLM call).
    This saves cost AND prevents injection.
    """
    if not query or not query.strip():
        raise GuardrailError("Query cannot be empty.", code="EMPTY_QUERY")

    if len(query) < MIN_QUERY_LENGTH:
        raise GuardrailError(f"Query too short (min {MIN_QUERY_LENGTH} chars).", code="QUERY_TOO_SHORT")

    if len(query) > MAX_QUERY_LENGTH_HARD:
        raise GuardrailError(
            f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH_HARD}). "
            "Please shorten your query.",
            code="QUERY_TOO_LONG"
        )

    warnings = []
    if len(query) > MAX_QUERY_LENGTH:
        warnings.append(f"Long query ({len(query)} chars) — may be slow and expensive")

    # Prompt injection check
    for pattern in _COMPILED_INJECTION:
        if pattern.search(query):
            logger.warning(f"Injection attempt in LangGraph query: {query[:100]}")
            raise GuardrailError(
                "Query contains patterns that are not allowed. Please rephrase.",
                code="PROMPT_INJECTION"
            )

    return GuardrailResult(
        is_safe=True,
        sanitized_text=query.strip(),
        warnings=warnings,
    )


def validate_response(response: str) -> GuardrailResult:
    """
    Validate LLM response in quality_checker_node before returning to user.
    Redacts any PII leaked by the LLM.
    """
    if not response:
        return GuardrailResult(is_safe=True, sanitized_text="", warnings=["Empty response"])

    warnings = []
    sanitized = response

    if len(response) > MAX_RESPONSE_LENGTH:
        sanitized = response[:MAX_RESPONSE_LENGTH] + "\n\n[Response truncated]"
        warnings.append("Response truncated")

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(sanitized):
            logger.warning(f"PII ({pii_type}) detected in LangGraph response — redacting")
            sanitized = pattern.sub(f"[{pii_type.upper()} REDACTED]", sanitized)
            warnings.append(f"{pii_type} redacted from response")

    return GuardrailResult(is_safe=True, sanitized_text=sanitized, warnings=warnings)
