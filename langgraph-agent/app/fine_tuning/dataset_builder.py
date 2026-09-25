"""
app/fine_tuning/dataset_builder.py  —  Fine-Tuning Dataset Preparation

─────────────────────────────────────────────────────────────────────────────
WHEN SHOULD YOU FINE-TUNE?

  DECISION FRAMEWORK:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Problem                      │  Solution                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  Model doesn't know X         │  RAG (add knowledge)            │
  │  Model knows X but formats    │  Prompt engineering             │
  │    wrong                      │                                 │
  │  Model is too slow/expensive  │  Use smaller model              │
  │  Domain-specific STYLE        │  Fine-tuning                    │
  │  Consistent output FORMAT     │  Fine-tuning                    │
  │  Specialized BEHAVIOR         │  Fine-tuning                    │
  │  Specific task ACCURACY       │  Fine-tuning + RAG              │
  └─────────────────────────────────────────────────────────────────┘

  FINE-TUNING IS REQUIRED when:
    1. You need consistent output format (e.g. always JSON, always bullet points)
    2. The model needs domain-specific style (legal language, medical terminology)
    3. The model should behave differently from its default (more terse, more formal)
    4. You have lots of examples of the EXACT style/format you want

  FINE-TUNING IS NOT REQUIRED when:
    1. You just need to add new knowledge (use RAG instead — cheaper, updatable)
    2. A good system prompt can achieve the same result
    3. You have < 50 training examples (too few to learn from)
    4. Your dataset is lower quality than the base model's training data

OPENAI FINE-TUNING FORMAT:
  Each training example is a JSONL (JSON Lines) record with a "messages" key.
  This mirrors the Chat Completions API format exactly.

  {
    "messages": [
      {"role": "system",    "content": "You are a concise HR assistant..."},
      {"role": "user",      "content": "What are Neha's skills?"},
      {"role": "assistant", "content": "Neha has skills in: Python, FastAPI..."}
    ]
  }

DATASET QUALITY RULES:
  ✓ At least 50-100 examples to see improvement
  ✓ 500-1000 examples for significant improvement
  ✓ Consistent style across all examples
  ✓ Correct answers only (garbage in = garbage out)
  ✓ Diverse queries (not 50 variations of the same question)
  ✓ System prompt SAME as what you use in production

  ✗ Avoid: duplicates, wrong answers, inconsistent format
  ✗ Avoid: data leakage (test queries appearing in training)
  ✗ Avoid: very long examples (increase cost, slow convergence)

DATA LEAKAGE:
  If your test set contains queries that also appear in training,
  your evaluation will be artificially high. The model "memorized"
  those exact queries rather than learning general patterns.
  FIX: Always split BEFORE creating examples (see split_dataset below).
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)
_tokenizer = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────────────────────
# Sample domain-specific training data
# ─────────────────────────────────────────────────────────────────────────────

# This is the domain we are fine-tuning for:
# An HR assistant that answers questions about resumes in a structured format.
# The fine-tuned model should ALWAYS respond with bullet points and
# explicitly cite the resume section.

SYSTEM_PROMPT = (
    "You are an expert HR assistant. When answering questions about resumes, "
    "always structure your answer with bullet points. "
    "Always cite the specific resume section (e.g. [Skills], [Experience], [Education]). "
    "Be concise: maximum 5 bullet points per answer. "
    "If information is not in the resume, say 'Not mentioned in resume' rather than guessing."
)

# 20 sample training examples for demonstration
SAMPLE_TRAINING_EXAMPLES = [
    {
        "user": "What programming languages does the candidate know?",
        "assistant": "From [Skills] section:\n• Python (3+ years, primary language)\n• JavaScript/TypeScript (React, Node.js)\n• SQL (PostgreSQL, MySQL)\n• Java (2 years, enterprise applications)",
    },
    {
        "user": "How many years of experience does this candidate have?",
        "assistant": "From [Experience] section:\n• Total: 5 years professional experience\n• Backend Engineering: 3 years (2021-2024)\n• Junior Developer: 2 years (2019-2021)\n• Current role: Senior Backend Engineer at TechCorp",
    },
    {
        "user": "What is the candidate's highest education?",
        "assistant": "From [Education] section:\n• B.Tech in Computer Science, IIT Delhi (2019)\n• CGPA: 8.7/10\n• Specialization: Machine Learning and Data Science",
    },
    {
        "user": "What projects has this person worked on?",
        "assistant": "From [Projects] section:\n• RAG-based Document Q&A System (Python, LangChain, Qdrant)\n• Real-time Analytics Dashboard (React, FastAPI, PostgreSQL)\n• Microservices Migration (Docker, Kubernetes, AWS)\n• E-commerce Recommendation Engine (ML, collaborative filtering)",
    },
    {
        "user": "Does the candidate have cloud experience?",
        "assistant": "From [Skills] and [Experience] sections:\n• AWS: EC2, S3, Lambda, RDS (2 years)\n• GCP: BigQuery, Cloud Run (1 year)\n• Azure: Blob Storage, Functions (basic exposure)\n• Kubernetes and Docker (3 years)",
    },
    {
        "user": "What is the candidate's current salary expectation?",
        "assistant": "Not mentioned in resume\n• Salary expectations are typically discussed in interview\n• Suggest asking directly during screening call",
    },
    {
        "user": "Has this person worked with AI or machine learning?",
        "assistant": "From [Skills] and [Projects] sections:\n• Machine Learning: scikit-learn, TensorFlow (2 years)\n• LLM Applications: LangChain, OpenAI API, RAG pipelines\n• Vector Databases: Qdrant, Pinecone\n• Deployed 2 ML models to production",
    },
    {
        "user": "What are the candidate's soft skills?",
        "assistant": "From [Summary] section:\n• Team leadership: managed a team of 4 engineers\n• Communication: presented technical designs to non-technical stakeholders\n• Problem-solving: led incident response for production outages\nNote: Soft skills are briefly mentioned — recommend verifying in interview",
    },
    {
        "user": "Is this candidate suitable for a senior backend role?",
        "assistant": "Based on [Experience] and [Skills] sections:\n• Meets seniority requirement: 5 years experience\n• Strong backend stack: Python, FastAPI, PostgreSQL, Redis\n• System design exposure: microservices, async processing\n• Leadership: tech lead for 2 projects\n• Recommendation: Strong match for senior backend role",
    },
    {
        "user": "What databases has this person used?",
        "assistant": "From [Skills] and [Experience] sections:\n• Relational: PostgreSQL (primary, 4 years), MySQL (2 years)\n• NoSQL: MongoDB (2 years), Redis (3 years, caching)\n• Vector: Qdrant (1 year, semantic search)\n• Search: Elasticsearch (1 year)",
    },
    {
        "user": "Does the candidate have any certifications?",
        "assistant": "From [Certifications] section:\n• AWS Solutions Architect Associate (2023)\n• Google Professional Data Engineer (2022)\n• No expired or unverified certifications listed",
    },
    {
        "user": "Where is this candidate located?",
        "assistant": "From [Contact] section:\n• Location: Bangalore, Karnataka, India\n• Open to relocation: Not mentioned\n• Remote work: Not specified — recommend confirming",
    },
    {
        "user": "What frameworks does this candidate know?",
        "assistant": "From [Skills] section:\n• Backend: FastAPI, Django, Flask (Python)\n• Frontend: React, Next.js (JavaScript)\n• ML: TensorFlow, PyTorch, LangChain\n• Testing: pytest, Jest\n• Infrastructure: Docker, Kubernetes, Terraform",
    },
    {
        "user": "Has this person published any research or papers?",
        "assistant": "Not mentioned in resume\n• No publications or research papers listed\n• If relevant, suggest asking about open-source contributions instead (GitHub mentioned in resume)",
    },
    {
        "user": "What is the candidate's notice period?",
        "assistant": "Not mentioned in resume\n• Notice period details are not included\n• Standard corporate notice period in India is 30-90 days\n• Recommend confirming in the first screening call",
    },
    {
        "user": "What is the candidate's key achievement?",
        "assistant": "From [Experience] section:\n• Reduced API latency by 60% through async processing and caching\n• Migrated monolith to microservices serving 500K daily users\n• Built RAG system processing 1M documents with 95% retrieval accuracy\n• Led team of 4, delivered project 2 weeks ahead of schedule",
    },
    {
        "user": "Does this candidate have team lead experience?",
        "assistant": "From [Experience] section:\n• Tech Lead at TechCorp (2023-present): 4-engineer team\n• Responsibilities: code reviews, sprint planning, architecture decisions\n• Mentored 2 junior developers\n• Delivered 3 projects on time as lead",
    },
    {
        "user": "What is the candidate's GitHub or portfolio?",
        "assistant": "From [Contact] section:\n• GitHub: github.com/candidate-username\n• LinkedIn: linkedin.com/in/candidate-name\n• Portfolio: Not listed separately\n• Note: Review GitHub repos to verify technical claims",
    },
    {
        "user": "Can this candidate work with microservices?",
        "assistant": "From [Experience] and [Projects] sections:\n• Led microservices migration at TechCorp (2022-2023)\n• Technologies: Docker, Kubernetes, RabbitMQ, API Gateway\n• Experience: service mesh, circuit breakers, distributed tracing\n• Scale: 12 microservices, 500K requests/day",
    },
    {
        "user": "What are the gaps in this candidate's experience?",
        "assistant": "Based on full resume analysis:\n• No mobile development experience\n• Limited frontend experience (React only, no Vue/Angular)\n• No mention of security or compliance experience\n• No international work experience\n• Recommendation: Gaps are typical for a backend specialist — assess based on role fit",
    },
]


def build_training_example(
    user_message: str,
    assistant_message: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """
    Build one training example in OpenAI fine-tuning format.

    FORMAT:
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }

    WHY THIS FORMAT?
      OpenAI's fine-tuning API processes examples in the Chat Completions format.
      The model learns: given (system + user), produce (assistant).
      It adjusts its weights to match the style/format of your assistant messages.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def build_jsonl_dataset(
    examples: Optional[list[dict]] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """
    Build a list of training examples in fine-tuning format.

    USAGE:
      dataset = build_jsonl_dataset()           # use built-in samples
      dataset = build_jsonl_dataset(my_examples) # use your own examples

    INPUT FORMAT (if providing your own):
      [{"user": "...", "assistant": "..."}, ...]
    """
    if examples is None:
        examples = SAMPLE_TRAINING_EXAMPLES

    dataset = []
    for ex in examples:
        if "messages" in ex:
            # Already in fine-tune format
            dataset.append(ex)
        elif "user" in ex and "assistant" in ex:
            dataset.append(build_training_example(
                user_message=ex["user"],
                assistant_message=ex["assistant"],
                system_prompt=system_prompt,
            ))
        else:
            logger.warning("Skipping malformed example: %s", str(ex)[:100])

    return dataset


def split_dataset(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split dataset into train / validation / test sets.

    WHY SPLIT?
      Training set:    What the model learns from
      Validation set:  Monitor for overfitting during training
      Test set:        Final unbiased evaluation (NEVER use for training)

    DATA LEAKAGE WARNING:
      Always split BEFORE any augmentation or preprocessing.
      If you split after augmenting, synthetic copies of test data
      may end up in training (data leakage).

    RETURNS:
      (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test ratios must sum to 1.0")

    rng = random.Random(random_seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    logger.info(
        "Dataset split: total=%d train=%d val=%d test=%d",
        n, len(train), len(val), len(test)
    )
    return train, val, test


def validate_dataset(data: list[dict]) -> dict:
    """
    Validate dataset quality before uploading to OpenAI.

    CHECKS:
      - Minimum 10 examples (OpenAI requires min 10, recommend 50+)
      - All examples have 'messages' key
      - All messages have 'role' and 'content'
      - No empty content fields
      - Token counts within limits (4096 max per example for gpt-3.5-turbo)
      - Duplicate detection

    RETURNS:
      {
        "valid": True/False,
        "errors": [...],
        "warnings": [...],
        "stats": {...}
      }
    """
    errors = []
    warnings = []
    token_counts = []
    seen_user_messages = set()
    duplicates = 0

    if len(data) < 10:
        errors.append(f"Only {len(data)} examples. OpenAI requires minimum 10.")
    if len(data) < 50:
        warnings.append(f"Only {len(data)} examples. Recommend 50-500 for good results.")

    for i, ex in enumerate(data):
        if "messages" not in ex:
            errors.append(f"Example {i}: missing 'messages' key")
            continue

        msgs = ex["messages"]
        roles = [m.get("role") for m in msgs]

        if "user" not in roles:
            errors.append(f"Example {i}: no 'user' message")
        if "assistant" not in roles:
            errors.append(f"Example {i}: no 'assistant' message")

        for m in msgs:
            if not m.get("content", "").strip():
                errors.append(f"Example {i}: empty content in {m.get('role')} message")

        # Token count
        total_tokens = sum(len(_tokenizer.encode(m.get("content", ""))) for m in msgs)
        token_counts.append(total_tokens)
        if total_tokens > 4096:
            warnings.append(f"Example {i}: {total_tokens} tokens (max 4096 recommended)")

        # Duplicate check
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        if user_msg in seen_user_messages:
            duplicates += 1
            warnings.append(f"Example {i}: duplicate user message detected")
        seen_user_messages.add(user_msg)

    stats = {
        "total_examples": len(data),
        "avg_tokens_per_example": round(sum(token_counts) / max(len(token_counts), 1), 1),
        "max_tokens": max(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "total_tokens": sum(token_counts),
        "duplicate_count": duplicates,
        "estimated_training_cost_usd": round(sum(token_counts) / 1_000_000 * 8.0, 4),  # ~$8/M tokens for gpt-3.5-turbo
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def estimate_cost(data: list[dict], model: str = "gpt-3.5-turbo") -> dict:
    """
    Estimate fine-tuning cost before running.

    OPENAI PRICING (2024-2026 approximate):
      gpt-3.5-turbo fine-tune: $8.00 per 1M tokens
      gpt-4o-mini fine-tune:  $25.00 per 1M tokens
      Training tokens = sum of all tokens in all training examples

    EXAMPLE:
      100 examples × 200 tokens each = 20,000 tokens
      20,000 / 1,000,000 × $8.00 = $0.16 total
    """
    cost_per_million = {
        "gpt-3.5-turbo": 8.00,
        "gpt-4o-mini": 25.00,
        "gpt-4o": 100.00,         # approximate
    }

    total_tokens = sum(
        sum(len(_tokenizer.encode(m.get("content", ""))) for m in ex.get("messages", []))
        for ex in data
    )
    rate = cost_per_million.get(model, 8.00)
    cost = (total_tokens / 1_000_000) * rate

    return {
        "model": model,
        "total_examples": len(data),
        "total_tokens": total_tokens,
        "cost_per_million_tokens": rate,
        "estimated_cost_usd": round(cost, 4),
        "note": "Multiply by number of epochs (default 3-10) for total cost",
    }


def save_jsonl(data: list[dict], path: str) -> str:
    """Save dataset to a JSONL file (one JSON per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(data), path)
    return str(p)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_dataset(
    output_dir: str = "data/fine_tuning",
) -> dict:
    """
    Complete dataset preparation pipeline.

    STEPS:
      1. Build training examples from sample data
      2. Validate dataset quality
      3. Split into train/val/test
      4. Estimate cost
      5. Save JSONL files
      6. Return paths and stats

    USAGE:
      from app.fine_tuning.dataset_builder import prepare_dataset
      result = prepare_dataset("data/fine_tuning")
      # result["train_path"] → "data/fine_tuning/train.jsonl"
    """
    logger.info("Building fine-tuning dataset...")

    # Step 1: Build examples
    dataset = build_jsonl_dataset()

    # Step 2: Validate
    validation = validate_dataset(dataset)
    if not validation["valid"]:
        logger.error("Dataset validation failed: %s", validation["errors"])
        return {"ok": False, "errors": validation["errors"]}

    if validation["warnings"]:
        for w in validation["warnings"]:
            logger.warning("Dataset warning: %s", w)

    # Step 3: Split
    train, val, test = split_dataset(dataset)

    # Step 4: Estimate cost
    cost = estimate_cost(train, model="gpt-3.5-turbo")

    # Step 5: Save
    train_path = save_jsonl(train, f"{output_dir}/train.jsonl")
    val_path = save_jsonl(val, f"{output_dir}/val.jsonl")
    test_path = save_jsonl(test, f"{output_dir}/test.jsonl")

    result = {
        "ok": True,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "validation": validation,
        "cost_estimate": cost,
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }

    logger.info("Dataset ready: %s", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    # Quick test: python -m app.fine_tuning.dataset_builder
    result = prepare_dataset()
    print(json.dumps(result, indent=2))
