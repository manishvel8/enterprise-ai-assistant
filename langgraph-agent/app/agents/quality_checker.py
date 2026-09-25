"""
agents/quality_checker.py — NODE 7: Quality Checker

ROLE:
  Evaluates the generated response against:
    1. Groundedness — does the answer come from the evidence?
    2. Completeness — does it fully answer the question?
    3. Quality — is it well-written and specific?

OUTPUTS:
  verdict = "PASS" → graph goes to END, answer is returned to user
  verdict = "REVISE" → graph loops back to response_generator

RETRY LIMIT:
  retry_count tracks how many times we've sent back to response_generator.
  When retry_count >= max_retries (default 3), we force PASS to prevent
  an infinite loop. The final answer includes a note about quality issues.

WHY A SEPARATE QUALITY NODE?
  Self-evaluation is a proven technique (Constitutional AI, LLM-as-judge).
  The quality checker uses a DIFFERENT prompt than the generator, so it can
  catch issues the generator missed. It's a second opinion.

SCORING:
  score 0.0–0.5: poor → REVISE
  score 0.5–0.7: acceptable → depends on is_grounded + is_complete
  score 0.7–1.0: good → PASS

OUTPUT KEYS:
  quality_result     — QualityResult dict
  retry_count        — incremented if REVISE
  final_answer       — set if PASS
  is_complete        — True if PASS
  execution_trace    — appended TraceEntry
"""

import logging
import time
from pydantic import BaseModel, Field

from app.graph.state import AgentState, QualityResult, TraceEntry
from app.services import llm_service
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


# ─── Structured output schema ────────────────────────────────────────────────

class QualitySchema(BaseModel):
    verdict: str = Field(
        description='"PASS" if the answer is good, "REVISE" if it needs improvement'
    )
    score: float = Field(
        description="Quality score from 0.0 (terrible) to 1.0 (perfect)", ge=0.0, le=1.0
    )
    is_grounded: bool = Field(
        description="True if the answer is clearly supported by the provided context"
    )
    is_complete: bool = Field(
        description="True if the answer fully addresses the user's question"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific problems. Empty list if verdict=PASS.",
    )
    revision_feedback: str = Field(
        default="",
        description="Concrete instructions for improvement. Empty if verdict=PASS.",
    )


SYSTEM_PROMPT = """\
You are a strict quality-control agent. Evaluate the AI assistant's answer.

You will receive:
  - QUESTION: the user's query
  - CONTEXT: the retrieved evidence
  - ANSWER: the generated response

Evaluation criteria:
1. GROUNDED: Every factual claim in the answer must be traceable to the context.
   If the context is empty/insufficient and the answer admits this honestly, is_grounded=True.
2. COMPLETE: The answer fully addresses what was asked. It doesn't leave key parts unanswered.
3. QUALITY: Clear writing, specific facts (not vague), proper citation usage.

Scoring:
  1.0: perfect — grounded, complete, well-written
  0.7: good enough — minor style issues but factually correct
  0.5: acceptable — answers the question but could be better
  0.3: poor — partially answers or has grounding issues
  0.0: terrible — wrong, hallucinated, or empty

Verdict rules:
  PASS if score >= 0.65 OR (is_grounded=True AND is_complete=True AND score >= 0.5)
  REVISE otherwise

When giving revision_feedback, be SPECIFIC:
  Good: "Add the score from Source 1 that shows 0.87 cosine similarity"
  Bad:  "Be more specific"
"""


# ─── Node function ───────────────────────────────────────────────────────────

async def quality_checker_node(state: AgentState) -> dict:
    start = time.time()
    query = state.get("validated_query") or state["original_query"]
    answer = state.get("generated_response", "")
    evidence = state.get("evidence", "")[:8000]  # cap to avoid token limit
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))

    trace.append(
        TraceEntry(
            node="quality_checker",
            status="running",
            started_at=start,
            input_summary=f'Checking answer (retry_count={retry_count})',
            output_summary="",
        )
    )

    # ── Force PASS if max retries reached ─────────────────────────────────────
    if retry_count >= max_retries:
        logger.warning(
            "quality_checker: max retries (%d) reached — forcing PASS", max_retries
        )
        forced_result = QualityResult(
            verdict="PASS",
            score=0.5,
            is_grounded=True,
            is_complete=True,
            issues=["Max revision limit reached — answer may not be perfect"],
            revision_feedback="",
        )
        duration = (time.time() - start) * 1000
        trace[-1] = TraceEntry(
            node="quality_checker",
            status="completed",
            started_at=start,
            completed_at=time.time(),
            duration_ms=round(duration, 1),
            input_summary=f'retry_count={retry_count} >= max={max_retries}',
            output_summary="FORCED PASS (max retries)",
        )
        return {
            "quality_result": forced_result,
            "final_answer": answer,
            "is_complete": True,
            "execution_trace": trace,
            "errors": errors,
        }

    # ── LLM quality evaluation ─────────────────────────────────────────────────
    # ── LANGFUSE: span + generation + score ───────────────────────────────────
    with tracer.span(
        trace_id,
        name="quality_check",
        input={"query": query, "answer_length": len(answer), "retry_count": retry_count},
        metadata={"node": "quality_checker"},
    ) as node_span:
        try:
            eval_prompt = (
                f"QUESTION: {query}\n\n"
                f"CONTEXT (truncated to 8000 chars):\n{evidence}\n\n"
                f"ANSWER:\n{answer}"
            )
            eval_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt},
            ]

            with tracer.generation(
                trace_id,
                name="llm_evaluation",
                model=state.get("model_name", "gpt-4o"),
                input=eval_messages,
                parent_span_id=node_span.id,
                metadata={"temperature": 0.0, "purpose": "quality_evaluation"},
            ) as eval_gen:
                result, in_tok, out_tok = await llm_service.structured_chat_with_usage(
                    messages=eval_messages,
                    response_format=QualitySchema,
                    temperature=0.0,
                )
                tracer.update_generation_tokens(eval_gen, in_tok, out_tok)
                eval_gen.update(output=result.model_dump())

            quality = QualityResult(
                verdict=result.verdict,
                score=result.score,
                is_grounded=result.is_grounded,
                is_complete=result.is_complete,
                issues=result.issues,
                revision_feedback=result.revision_feedback,
            )

            # ── LANGFUSE: attach quality score to the TRACE ──────────────────
            # This is the key metric for "how good was the answer?"
            # In the Langfuse UI: Traces → filter by quality_score < 0.7
            tracer.score(
                trace_id,
                name="quality_score",
                value=result.score,
                comment=f"verdict={result.verdict} | grounded={result.is_grounded} | complete={result.is_complete}",
            )
            if result.issues:
                tracer.score(
                    trace_id,
                    name="has_issues",
                    value=1.0 if result.issues else 0.0,
                    data_type="BOOLEAN",
                    comment="; ".join(result.issues[:3]),
                )

            # Increment retry_count only if REVISE
            new_retry_count = retry_count + 1 if result.verdict == "REVISE" else retry_count

            # Set final_answer if PASS
            final_answer = answer if result.verdict == "PASS" else ""
            is_complete = result.verdict == "PASS"

            duration = (time.time() - start) * 1000
            trace[-1] = TraceEntry(
                node="quality_checker",
                status="completed",
                started_at=start,
                completed_at=time.time(),
                duration_ms=round(duration, 1),
                input_summary=f'Evaluating answer ({len(answer)} chars)',
                output_summary=(
                    f"verdict={result.verdict} | score={result.score:.2f} | "
                    f"grounded={result.is_grounded} | complete={result.is_complete}"
                    + (f" | issues: {'; '.join(result.issues[:2])}" if result.issues else "")
                ),
            )
            node_span.update(
                output={
                    "verdict": result.verdict,
                    "score": result.score,
                    "is_grounded": result.is_grounded,
                    "is_complete": result.is_complete,
                    "issues": result.issues,
                },
                metadata={"duration_ms": round(duration, 1)},
            )

            logger.info(
                "quality_checker: verdict=%s score=%.2f retry=%d",
                result.verdict, result.score, new_retry_count,
            )

            return {
                "quality_result": quality,
                "retry_count": new_retry_count,
                "final_answer": final_answer,
                "is_complete": is_complete,
                "execution_trace": trace,
                "errors": errors,
            }

        except Exception as e:
            logger.error("quality_checker failed: %s", e)
            errors.append(f"quality_checker: {e}")
            # On error, default to PASS so we don't loop forever
            trace[-1] = TraceEntry(
                node="quality_checker",
                status="failed",
                started_at=start,
                completed_at=time.time(),
                input_summary="Quality check failed",
                output_summary="",
                error=str(e),
            )
            node_span.update(level="ERROR", status_message=str(e))
            return {
                "quality_result": QualityResult(
                    verdict="PASS",
                    score=0.5,
                    is_grounded=True,
                    is_complete=True,
                    issues=["Quality check skipped due to error"],
                    revision_feedback="",
                ),
                "retry_count": retry_count,
                "final_answer": answer,
                "is_complete": True,
                "execution_trace": trace,
                "errors": errors,
            }
