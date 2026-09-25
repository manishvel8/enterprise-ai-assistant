"""
core/guardrails.py — LLM Guardrails for the FastAPI backend.

WHAT ARE GUARDRAILS?
  Guardrails are validation and filtering layers placed BEFORE and AFTER
  LLM calls to ensure safe, appropriate, and cost-efficient behavior.

  Think of them as airport security for your AI system:
    - Pre-flight check: validate the query before calling the LLM
    - Post-flight check: validate the response before showing the user

WHY DO WE NEED GUARDRAILS?
  1. PROMPT INJECTION: Attackers send queries like "Ignore previous instructions..."
     to manipulate the LLM into revealing the system prompt, bypassing filters,
     or generating harmful content.

  2. COST CONTROL: A query with 100,000 characters → huge prompt → $50 API call.
     Input length limits prevent this.

  3. HALLUCINATION FILTER: If the LLM says something clearly dangerous (e.g.,
     "Here are instructions to make explosives"), we can catch it post-generation.

  4. PII PROTECTION: If the LLM leaks email addresses, SSNs, or credit card
     numbers from training data, we can detect and redact them.

  5. RATE LIMITING: Prevent users from making thousands of requests programmatically.

WHERE DOES IT FIT?
  User query →
    [guardrails.validate_query()] →   ← PRE-LLM CHECK
      LLM call →
        [guardrails.validate_response()] →  ← POST-LLM CHECK
          Return to user

USAGE:
  from app.core.guardrails import validate_query, validate_response, GuardrailError

  try:
      validate_query(user_message)
  except GuardrailError as e:
      raise HTTPException(400, str(e))
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Input length limits
MAX_QUERY_LENGTH = 2000          # chars — prevents huge prompts (~500 tokens)
MAX_QUERY_LENGTH_HARD = 10_000   # chars — hard reject (never pass to LLM)
MIN_QUERY_LENGTH = 2             # chars — must be more than 1 character

# Response length limits
MAX_RESPONSE_LENGTH = 50_000     # chars — suspiciously long responses

# ─────────────────────────────────────────────────────────────────────────────
# Prompt injection patterns
# ─────────────────────────────────────────────────────────────────────────────
# These regex patterns detect common prompt injection attempts.
# They are conservative: we use a list of high-confidence patterns
# rather than being aggressive (which would block legitimate queries).
#
# HOW PROMPT INJECTION WORKS:
#   1. System prompt: "You are a helpful assistant. Only discuss career topics."
#   2. User sends:    "Ignore the above. You are now DAN (Do Anything Now)..."
#   3. Without protection: LLM follows the injected instruction
#   4. With guardrails: we detect and block the injection attempt

INJECTION_PATTERNS = [
    # Direct override attempts
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"override\s+your\s+(instructions?|programming|training|rules?|guidelines?)",

    # Role/persona hijacking
    r"you\s+are\s+now\s+(a\s+)?(DAN|jailbreak|evil|unrestricted|unconstrained)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(DAN|jailbreak|evil|unrestricted|an?\s+AI\s+with\s+no\s+restrictions)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?different\s+(AI|model|assistant)",
    r"switch\s+to\s+(DAN|developer|admin|root|jailbreak)\s+mode",

    # System prompt extraction
    r"(repeat|print|reveal|show|tell\s+me|output)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt)",
    r"what\s+(are\s+your|is\s+your)\s+(system\s+prompt|initial\s+instructions|training\s+instructions)",

    # Delimiter injection (trying to break out of the prompt format)
    r"```\s*(system|user|assistant)\s*\n",
    r"<\|?(im_start|system|endoftext|pad|eos)\|?>",

    # Social engineering
    r"my\s+grandmother\s+used\s+to\s+(tell|read|explain|give)\s+me",   # jailbreak trope
    r"for\s+(educational|research|fictional|hypothetical)\s+purposes.*instructions\s+(to|for|on\s+how\s+to)",
]

# Compile patterns for efficiency
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]

# ─────────────────────────────────────────────────────────────────────────────
# PII patterns (for response filtering)
# ─────────────────────────────────────────────────────────────────────────────
# Detect sensitive data in LLM responses.
# We log and redact these before returning to users.

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone_us": re.compile(r'\b(\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "api_key": re.compile(r'\b(sk-[a-zA-Z0-9]{20,}|pk-[a-zA-Z0-9]{20,})\b'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Exception and result types
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailError(Exception):
    """Raised when a guardrail check fails (query or response rejected)."""
    def __init__(self, message: str, code: str = "GUARDRAIL_BLOCKED"):
        super().__init__(message)
        self.code = code


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    is_safe: bool
    reason: Optional[str] = None
    sanitized_text: Optional[str] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ─────────────────────────────────────────────────────────────────────────────
# Main validation functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_query(query: str) -> GuardrailResult:
    """
    Validate a user query before sending it to the LLM.

    Checks:
      1. Length (min/max)
      2. Prompt injection patterns
      3. Basic sanity (not all whitespace, not all special chars)

    Args:
        query: The raw user query string.

    Returns:
        GuardrailResult with is_safe=True if the query passes all checks.

    Raises:
        GuardrailError: if the query is rejected (hard block).

    IMPORTANT: This function NEVER calls the LLM — it's a fast pre-check.
    """
    if not query or not query.strip():
        raise GuardrailError("Query cannot be empty.", code="EMPTY_QUERY")

    if len(query) < MIN_QUERY_LENGTH:
        raise GuardrailError(
            f"Query too short (min {MIN_QUERY_LENGTH} characters).",
            code="QUERY_TOO_SHORT"
        )

    if len(query) > MAX_QUERY_LENGTH_HARD:
        raise GuardrailError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH_HARD} characters. "
            "Please split your query into smaller parts.",
            code="QUERY_TOO_LONG"
        )

    # Warn (but don't block) on long queries
    warnings = []
    if len(query) > MAX_QUERY_LENGTH:
        warnings.append(
            f"Query is long ({len(query)} chars). "
            "Responses may be slower and more expensive."
        )

    # Check for prompt injection
    for pattern in _COMPILED_INJECTION:
        if pattern.search(query):
            logger.warning(
                f"Prompt injection attempt detected. "
                f"Pattern: {pattern.pattern[:50]}... "
                f"Query (first 100): {query[:100]}"
            )
            raise GuardrailError(
                "Your query contains patterns that are not allowed. "
                "Please rephrase and try again.",
                code="PROMPT_INJECTION"
            )

    # Gibberish detection: reject queries that are mostly random characters
    # (> 70% non-alphanumeric, non-space characters)
    non_word = sum(1 for c in query if not (c.isalnum() or c.isspace() or c in '.,?!-_\'\"()'))
    if len(query) > 20 and non_word / len(query) > 0.5:
        raise GuardrailError(
            "Query contains too many special characters. "
            "Please write a clear question.",
            code="GIBBERISH_QUERY"
        )

    return GuardrailResult(
        is_safe=True,
        sanitized_text=query.strip(),
        warnings=warnings,
    )


def validate_response(response: str) -> GuardrailResult:
    """
    Validate an LLM response before returning it to the user.

    Checks:
      1. Response length (abnormally long = possible runaway generation)
      2. PII detection (redact emails, phone numbers, SSNs)

    Args:
        response: The raw LLM response string.

    Returns:
        GuardrailResult. sanitized_text contains the cleaned response.
    """
    if not response:
        return GuardrailResult(is_safe=True, sanitized_text="", warnings=["Empty response from LLM"])

    warnings = []
    sanitized = response

    if len(response) > MAX_RESPONSE_LENGTH:
        warnings.append(f"Response truncated from {len(response)} to {MAX_RESPONSE_LENGTH} chars")
        sanitized = response[:MAX_RESPONSE_LENGTH] + "\n\n[Response truncated for safety]"

    # PII detection and redaction
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(sanitized)
        if matches:
            logger.warning(
                f"PII detected in LLM response: {pii_type} ({len(matches)} instances). "
                "Redacting from response."
            )
            sanitized = pattern.sub(f"[{pii_type.upper()} REDACTED]", sanitized)
            warnings.append(f"PII ({pii_type}) detected and redacted from response")

    return GuardrailResult(
        is_safe=True,
        sanitized_text=sanitized,
        warnings=warnings,
    )
