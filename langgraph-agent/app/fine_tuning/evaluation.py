"""
app/fine_tuning/evaluation.py  —  Base Model vs Fine-Tuned Model Comparison

─────────────────────────────────────────────────────────────────────────────
WHY EVALUATE?
  Fine-tuning costs money and time. Before deploying a fine-tuned model,
  you must prove it actually performs BETTER than the base model.
  Otherwise, you've wasted time and money.

EVALUATION DIMENSIONS:
  1. Format adherence   — Does it use bullet points as trained?
  2. Citation accuracy  — Does it correctly cite resume sections?
  3. Conciseness        — Is it within 5 bullet points?
  4. Hallucination rate — Does it invent facts not in the prompt?
  5. Latency            — How long does each response take?
  6. Token cost         — How many tokens does it use?

LANGFUSE EXPERIMENT STRUCTURE:
  Experiment A: tag="experiment:base-model"         model=gpt-4o
  Experiment B: tag="experiment:fine-tuned-model"   model=ft:gpt-3.5-turbo:..

  In the Langfuse UI:
    Traces → Filter by tag "experiment:base-model" → avg quality score
    Traces → Filter by tag "experiment:fine-tuned-model" → avg quality score
    → Compare the two distributions

A/B TESTING IN PRODUCTION:
  After evaluating, if fine-tuned model is better, you can do gradual rollout:
  50% of users → base model  (control group)
  50% of users → fine-tuned  (treatment group)
  Monitor quality_score in Langfuse for both groups over 1 week.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from typing import Optional

import tiktoken

from app.config import settings
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
_tokenizer = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────────────────────
# Test cases (used for both models with identical inputs)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = (
    "You are an AI assistant that answers questions about resumes."
)

SYSTEM_PROMPT_FINETUNED = (
    "You are an expert HR assistant. When answering questions about resumes, "
    "always structure your answer with bullet points. "
    "Always cite the specific resume section (e.g. [Skills], [Experience], [Education]). "
    "Be concise: maximum 5 bullet points per answer. "
    "If information is not in the resume, say 'Not mentioned in resume' rather than guessing."
)

TEST_CASES = [
    {
        "id": "tc_001",
        "query": "What programming languages does this candidate know?",
        "context": "Skills: Python, JavaScript, SQL, Java. Experience with FastAPI and React.",
        "expected_format": "bullet_points_with_citations",
        "expected_keywords": ["Python", "JavaScript", "Skills"],
    },
    {
        "id": "tc_002",
        "query": "What is the candidate's highest education?",
        "context": "Education: B.Tech Computer Science, IIT Delhi, 2019, CGPA 8.7",
        "expected_format": "bullet_points_with_citations",
        "expected_keywords": ["B.Tech", "Education", "IIT"],
    },
    {
        "id": "tc_003",
        "query": "How many years of experience does this person have?",
        "context": "Experience: 5 years total. Senior Backend Engineer at TechCorp (2021-present), Junior Dev (2019-2021).",
        "expected_format": "bullet_points_with_citations",
        "expected_keywords": ["5 years", "Experience"],
    },
    {
        "id": "tc_004",
        "query": "What is the candidate's salary expectation?",
        "context": "Name: Neha Pukale. Email: neha@example.com. Skills: Python, AWS.",
        "expected_format": "not_mentioned",
        "expected_keywords": ["Not mentioned"],
    },
    {
        "id": "tc_005",
        "query": "Has this person worked with cloud platforms?",
        "context": "Skills: AWS (EC2, S3, Lambda), GCP (BigQuery), Docker, Kubernetes.",
        "expected_format": "bullet_points_with_citations",
        "expected_keywords": ["AWS", "GCP", "Skills"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_format_score(response: str) -> float:
    """
    Check if response uses bullet points as trained.

    Score: 1.0 if bullets present, 0.0 if plain text.
    """
    has_bullets = any(line.strip().startswith(("•", "-", "*", "·")) for line in response.split("\n"))
    has_numbered = any(line.strip()[:2].rstrip(".").isdigit() for line in response.split("\n") if line.strip())
    return 1.0 if (has_bullets or has_numbered) else 0.0


def compute_citation_score(response: str) -> float:
    """
    Check if response cites resume sections like [Skills], [Experience].

    Score: 1.0 if citations present, 0.0 if missing.
    """
    import re
    citations = re.findall(r"\[([A-Za-z ]+)\]", response)
    return 1.0 if citations else 0.0


def compute_conciseness_score(response: str, max_bullets: int = 5) -> float:
    """
    Check if response is within bullet point limit.

    Score: 1.0 if ≤ max_bullets lines, decreases above limit.
    """
    bullet_lines = [
        line for line in response.split("\n")
        if line.strip().startswith(("•", "-", "*", "·")) or
           (line.strip()[:2].rstrip(".").isdigit() and len(line.strip()) > 2)
    ]
    if len(bullet_lines) == 0:
        return 0.5  # no bullets at all
    if len(bullet_lines) <= max_bullets:
        return 1.0
    return max(0.0, 1.0 - (len(bullet_lines) - max_bullets) * 0.2)


def compute_hallucination_score(response: str, context: str) -> float:
    """
    Simple heuristic: does the response mention facts NOT in the context?

    REAL HALLUCINATION DETECTION:
      This is a simplified version. Production approaches:
        1. Use an LLM-as-judge: "Does this answer contain claims not in the context?"
        2. Use NLI (Natural Language Inference) models
        3. Use Langfuse evaluations: define a scoring function and apply to all traces

    SCORE: 1.0 = no hallucination detected, 0.0 = likely hallucination.
    """
    # Extract numbers from response (numbers are often hallucinated)
    import re
    response_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", response))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context))
    extra_numbers = response_numbers - context_numbers

    # If many numbers in response aren't in context, possible hallucination
    if len(extra_numbers) > 3:
        return 0.5  # suspicious but not conclusive
    return 1.0


def compute_keyword_recall(response: str, expected_keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in the response."""
    if not expected_keywords:
        return 1.0
    found = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    return found / len(expected_keywords)


def compute_metrics(
    response: str,
    context: str,
    expected_keywords: list[str],
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    Compute all metrics for one model response.

    METRICS:
      format_score    — uses bullet points?
      citation_score  — cites [Sections]?
      conciseness     — ≤ 5 bullets?
      hallucination   — no invented facts?
      keyword_recall  — mentions expected keywords?
      overall_score   — weighted average
      latency_ms      — API response time
      input_tokens    — prompt tokens (drives cost)
      output_tokens   — completion tokens
      cost_usd        — estimated cost at gpt-3.5-turbo prices
    """
    format_score = compute_format_score(response)
    citation_score = compute_citation_score(response)
    conciseness = compute_conciseness_score(response)
    hallucination = compute_hallucination_score(response, context)
    keyword_recall = compute_keyword_recall(response, expected_keywords)

    # Weighted overall score
    overall = (
        format_score * 0.25 +
        citation_score * 0.30 +
        conciseness * 0.15 +
        hallucination * 0.20 +
        keyword_recall * 0.10
    )

    # Cost calculation (gpt-3.5-turbo pricing)
    cost_usd = (input_tokens / 1_000_000 * 1.50) + (output_tokens / 1_000_000 * 2.00)

    return {
        "format_score": round(format_score, 3),
        "citation_score": round(citation_score, 3),
        "conciseness_score": round(conciseness, 3),
        "hallucination_score": round(hallucination, 3),
        "keyword_recall": round(keyword_recall, 3),
        "overall_score": round(overall, 3),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost_usd, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation on one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    experiment_tag: str,
    test_cases: Optional[list[dict]] = None,
    system_prompt: Optional[str] = None,
    simulate: bool = True,
) -> list[dict]:
    """
    Run all test cases through one model and collect metrics.

    LANGFUSE INTEGRATION:
      Each test case creates a Langfuse trace tagged with experiment_tag.
      This allows side-by-side comparison in the Langfuse UI:
        Traces → filter by tag "experiment:base-model" → avg quality_score

    USAGE:
      base_results = evaluate_model(
          model_name="gpt-4o",
          experiment_tag="experiment:base-model",
          system_prompt=SYSTEM_PROMPT_BASE,
      )
      ft_results = evaluate_model(
          model_name="ft:gpt-3.5-turbo:org:hr-assistant:id",
          experiment_tag="experiment:fine-tuned-model",
          system_prompt=SYSTEM_PROMPT_FINETUNED,
      )
    """
    if test_cases is None:
        test_cases = TEST_CASES
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_BASE

    tracer = get_tracer()
    results = []

    logger.info(
        "Evaluating model: %s (tag=%s, simulate=%s, %d test cases)",
        model_name, experiment_tag, simulate, len(test_cases)
    )

    for tc in test_cases:
        tc_id = tc.get("id", "unknown")
        query = tc.get("query", "")
        context = tc.get("context", "")
        expected_keywords = tc.get("expected_keywords", [])

        prompt = f"Context from resume:\n{context}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Create Langfuse trace for this evaluation run
        import uuid
        trace_id = f"eval_{tc_id}_{uuid.uuid4().hex[:8]}"
        tracer.create_trace(
            request_id=trace_id,
            user_id="evaluator",
            session_id="evaluation_run",
            query=query,
            tags=[experiment_tag, "evaluation", f"model:{model_name[:20]}"],
            metadata={"test_case_id": tc_id, "model": model_name},
        )

        start_time = time.time()

        if simulate:
            # Simulate responses that show the difference between base and fine-tuned
            if "fine-tuned" in experiment_tag:
                # Fine-tuned model: structured, cited, concise (what training taught it)
                response = (
                    f"From [{list(context.split(':')[0].strip().split())[-1]}] section:\n"
                    f"• {query.replace('?', '')} — detailed answer here\n"
                    f"• Key fact 1 from context\n"
                    f"• Key fact 2 from context"
                )
                input_tokens = len(_tokenizer.encode(prompt)) + 150
                output_tokens = len(_tokenizer.encode(response))
                latency_ms = 450.0 + (hash(tc_id) % 200)
            else:
                # Base model: less structured, no citations, verbose
                response = (
                    f"Based on the provided context, the candidate "
                    f"has various relevant qualifications. The information shows "
                    f"that {query.lower().replace('?', '')} which is relevant. "
                    f"Additionally, there are other factors to consider."
                )
                input_tokens = len(_tokenizer.encode(prompt)) + 200
                output_tokens = len(_tokenizer.encode(response))
                latency_ms = 850.0 + (hash(tc_id) % 300)
        else:
            # Real API call
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base or None,
            )
            api_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            response = api_response.choices[0].message.content or ""
            input_tokens = api_response.usage.prompt_tokens
            output_tokens = api_response.usage.completion_tokens
            latency_ms = (time.time() - start_time) * 1000

        # Compute metrics
        metrics = compute_metrics(
            response=response,
            context=context,
            expected_keywords=expected_keywords,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Log quality score to Langfuse
        tracer.score(
            trace_id=trace_id,
            name="quality_score",
            value=metrics["overall_score"],
            comment=f"format={metrics['format_score']} | citation={metrics['citation_score']}",
        )
        tracer.score(
            trace_id=trace_id,
            name="hallucination_score",
            value=metrics["hallucination_score"],
        )

        result = {
            "test_case_id": tc_id,
            "model": model_name,
            "experiment_tag": experiment_tag,
            "query": query,
            "response": response,
            "metrics": metrics,
        }
        results.append(result)
        logger.info(
            "TC %s: overall=%.2f | format=%.2f | citation=%.2f | latency=%.0fms",
            tc_id, metrics["overall_score"], metrics["format_score"],
            metrics["citation_score"], metrics["latency_ms"]
        )

        tracer.flush()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Compare two models
# ─────────────────────────────────────────────────────────────────────────────

def compare_experiments(
    base_results: list[dict],
    ft_results: list[dict],
) -> dict:
    """
    Side-by-side comparison of base model vs fine-tuned model.

    RETURNS a report showing which model is better on each metric
    and whether the improvement is worth the fine-tuning cost.

    INTERPRETATION GUIDE:
      format_score > +0.2:     Fine-tuning significantly improved formatting
      citation_score > +0.2:   Fine-tuning improved source attribution
      latency improvement:     Fine-tuned model (smaller) is faster
      cost improvement:        Fine-tuned model uses fewer tokens
      overall < +0.1:          Fine-tuning may not be worth it
    """

    def avg_metric(results: list[dict], key: str) -> float:
        values = [r["metrics"].get(key, 0) for r in results]
        return round(sum(values) / max(len(values), 1), 3)

    base_metrics = {
        "format_score":        avg_metric(base_results, "format_score"),
        "citation_score":      avg_metric(base_results, "citation_score"),
        "conciseness_score":   avg_metric(base_results, "conciseness_score"),
        "hallucination_score": avg_metric(base_results, "hallucination_score"),
        "keyword_recall":      avg_metric(base_results, "keyword_recall"),
        "overall_score":       avg_metric(base_results, "overall_score"),
        "avg_latency_ms":      avg_metric(base_results, "latency_ms"),
        "avg_total_tokens":    avg_metric(base_results, "total_tokens"),
        "avg_cost_usd":        avg_metric(base_results, "cost_usd"),
    }

    ft_metrics = {
        "format_score":        avg_metric(ft_results, "format_score"),
        "citation_score":      avg_metric(ft_results, "citation_score"),
        "conciseness_score":   avg_metric(ft_results, "conciseness_score"),
        "hallucination_score": avg_metric(ft_results, "hallucination_score"),
        "keyword_recall":      avg_metric(ft_results, "keyword_recall"),
        "overall_score":       avg_metric(ft_results, "overall_score"),
        "avg_latency_ms":      avg_metric(ft_results, "latency_ms"),
        "avg_total_tokens":    avg_metric(ft_results, "total_tokens"),
        "avg_cost_usd":        avg_metric(ft_results, "cost_usd"),
    }

    improvements = {
        k: round(ft_metrics[k] - base_metrics[k], 3)
        for k in base_metrics
    }

    # Recommendation
    overall_improvement = improvements.get("overall_score", 0)
    latency_improvement = improvements.get("avg_latency_ms", 0)

    if overall_improvement > 0.15:
        recommendation = "DEPLOY: Fine-tuned model is significantly better. Recommend full rollout."
    elif overall_improvement > 0.05:
        recommendation = "CONSIDER: Fine-tuned model is slightly better. Do A/B test with 10% traffic."
    elif overall_improvement > -0.05:
        recommendation = "NEUTRAL: Negligible difference. Not worth deploying. Collect more training data."
    else:
        recommendation = "REVERT: Fine-tuned model is worse. Check training data quality and labels."

    return {
        "base_model": base_results[0]["model"] if base_results else "unknown",
        "fine_tuned_model": ft_results[0]["model"] if ft_results else "unknown",
        "test_cases_evaluated": len(base_results),
        "base_metrics": base_metrics,
        "fine_tuned_metrics": ft_metrics,
        "improvements": improvements,
        "recommendation": recommendation,
        "langfuse_comparison": {
            "instructions": (
                "In Langfuse UI: Traces → filter by tag 'experiment:base-model' → "
                "look at quality_score distribution. Then filter by 'experiment:fine-tuned-model'. "
                "Compare the two distributions."
            ),
            "base_tag": "experiment:base-model",
            "ft_tag": "experiment:fine-tuned-model",
            "metric_to_compare": "quality_score",
        },
    }


def run_full_evaluation(
    base_model: str = "gpt-4o",
    fine_tuned_model: str = "ft:gpt-3.5-turbo:org:hr-assistant:SIMULATED_ID",
    simulate: bool = True,
) -> dict:
    """
    Run complete evaluation: base model + fine-tuned model + comparison.

    USAGE:
      from app.fine_tuning.evaluation import run_full_evaluation
      report = run_full_evaluation(simulate=True)
      print(json.dumps(report, indent=2))
    """
    logger.info("=== EVALUATION: %s vs %s ===", base_model, fine_tuned_model)

    # Evaluate base model (Experiment A)
    base_results = evaluate_model(
        model_name=base_model,
        experiment_tag="experiment:base-model",
        system_prompt=SYSTEM_PROMPT_BASE,
        simulate=simulate,
    )

    # Evaluate fine-tuned model (Experiment B)
    ft_results = evaluate_model(
        model_name=fine_tuned_model,
        experiment_tag="experiment:fine-tuned-model",
        system_prompt=SYSTEM_PROMPT_FINETUNED,
        simulate=simulate,
    )

    # Compare
    comparison = compare_experiments(base_results, ft_results)

    return {
        "base_results": base_results,
        "fine_tuned_results": ft_results,
        "comparison": comparison,
    }


if __name__ == "__main__":
    # Quick test: python -m app.fine_tuning.evaluation
    report = run_full_evaluation(simulate=True)
    print(json.dumps(report["comparison"], indent=2))
