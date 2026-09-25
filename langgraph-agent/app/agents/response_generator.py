"""
agents/response_generator.py — NODE 6: Response Generator

ROLE:
  Takes all collected evidence (RAG chunks, web results, or human input)
  and generates a coherent, cited answer using GPT-4o.

FLOW:
  retrieved_documents | web_results | human_input
    → _build_evidence()      — formats into numbered context block
    → _build_prompt()        — constructs the system + user prompt
    → GPT-4o chat completion — generates the answer
    → post-processing        — extracts sources

WHY SEPARATE FROM RETRIEVAL?
  The generator runs AGAIN if the quality_checker returns "REVISE".
  Keeping it separate means we can re-run generation with the same
  evidence but with revision_feedback added to the prompt.
  The evidence doesn't change — only the generation does.

OUTPUT KEYS:
  evidence           — formatted context string (for quality checker)
  sources            — list of citations attached to the answer
  generated_response — the draft answer text
  execution_trace    — appended TraceEntry
"""

import logging
import time

from app.graph.state import AgentState, Source, TraceEntry
from app.services import llm_service
from app.observability.langfuse_client import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()

SYSTEM_PROMPT = """\
You are an expert assistant. Answer the user's ORIGINAL question.

Rules:
1. Prefer facts from the provided context. Cite with [Source N] when you use them.
2. If the human already chose a path (documents vs web), respect that choice and explain clearly.
3. If context is resumes/CVs and the question is advisory (docs vs internet), do NOT say you have no information —
   answer the advisory question using the human's choice + any relevant RAG/agent concepts in the context,
   and briefly note that uploaded files are examples of private document knowledge.
4. Only say "insufficient information" when the user asked for specific facts that are truly missing.
5. Be concise and well-structured. Use bullet points for lists.

If revision feedback is given, address ALL the issues mentioned before answering.
"""


def _build_evidence(state: AgentState) -> tuple[str, list[Source]]:
    """
    Combine all retrieved evidence into a numbered context block.
    Returns (evidence_string, sources_list).
    """
    parts: list[str] = []
    sources: list[Source] = []
    idx = 1

    # RAG chunks
    for doc in state.get("retrieved_documents", []):
        text = doc.get("chunk_text", "") if isinstance(doc, dict) else getattr(doc, "chunk_text", "")
        fname = doc.get("file_name", "doc") if isinstance(doc, dict) else getattr(doc, "file_name", "doc")
        page = doc.get("page_number") if isinstance(doc, dict) else getattr(doc, "page_number", None)
        score = doc.get("score", 0) if isinstance(doc, dict) else getattr(doc, "score", 0)

        parts.append(f"[Source {idx}] {fname} (page {page or '?'}, score={score:.3f})\n{text}")
        sources.append(Source(
            kind="document",
            title=fname,
            reference=f"Page {page or '?'}",
            excerpt=text[:200],
        ))
        idx += 1

    # Web results
    for r in state.get("web_results", []):
        title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
        url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
        content = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")

        parts.append(f"[Source {idx}] {title}\nURL: {url}\n{content}")
        sources.append(Source(
            kind="web",
            title=title,
            reference=url,
            excerpt=content[:200],
        ))
        idx += 1

    # Human clarification
    human_input = state.get("human_input", "")
    if human_input:
        parts.append(f"[Source {idx}] Human clarification\n{human_input}")
        sources.append(Source(
            kind="document",
            title="Human clarification",
            reference="User input",
            excerpt=human_input[:200],
        ))
        idx += 1

    evidence = "\n\n---\n\n".join(parts) if parts else "No context available."
    return evidence, sources


async def response_generator_node(state: AgentState) -> dict:
    start = time.time()
    # Show the user's original question; validated_query may be a search rewrite
    original_q = state.get("original_query") or ""
    search_q = state.get("validated_query") or original_q
    query = original_q or search_q
    trace_id = state.get("langfuse_trace_id", "")        # for Langfuse
    trace = list(state.get("execution_trace", []))
    errors = list(state.get("errors", []))
    retry_count = state.get("retry_count", 0)
    human_input = state.get("human_input", "")
    route = state.get("route", "")

    trace.append(
        TraceEntry(
            node="response_generator",
            status="running",
            started_at=start,
            input_summary=(
                f'Generating answer (attempt {retry_count + 1}) for: "{query[:60]}"'
            ),
            output_summary="",
        )
    )

    # Build evidence context
    evidence, sources = _build_evidence(state)

    # Add revision feedback if this is a retry
    quality_result = state.get("quality_result")
    revision_section = ""
    if quality_result and retry_count > 0:
        feedback = (
            quality_result.get("revision_feedback", "")
            if isinstance(quality_result, dict)
            else getattr(quality_result, "revision_feedback", "")
        )
        if feedback:
            revision_section = f"\n\nREVISION FEEDBACK (address ALL of these):\n{feedback}"

    human_section = ""
    if human_input:
        human_section = (
            f"\n\nHuman chose retrieval path: {route or 'unknown'}\n"
            f"Human clarification: {human_input}\n"
            f"Search query used for retrieval: {search_q}\n"
        )

    user_message = (
        f"Context:\n{evidence}"
        f"{human_section}"
        f"{revision_section}"
        f"\n\nOriginal user question: {query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # ── LANGFUSE: span for the node + generation for the LLM call ─────────────
    with tracer.span(
        trace_id,
        name="response_generation",
        input={"query": query, "retry_attempt": retry_count + 1,
               "evidence_chunks": len(state.get("retrieved_documents", [])) + len(state.get("web_results", []))},
        metadata={"node": "response_generator"},
    ) as node_span:
        try:
            with tracer.generation(
                trace_id,
                name="llm_call",
                model=state.get("model_name", "gpt-4o"),
                input=messages,
                parent_span_id=node_span.id,
                metadata={"temperature": 0.2, "max_tokens": 1500},
            ) as gen_ctx:
                answer, in_tok, out_tok = await llm_service.chat_with_usage(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                )
                # ── THIS IS WHERE COST IS TRACKED ──────────────────────────────
                # Langfuse auto-computes cost for gpt-4o:
                #   input  tokens: $0.005/1K  → 1420 tokens = $0.0071
                #   output tokens: $0.015/1K  → 320 tokens  = $0.0048
                #   total cost:                              = $0.0119
                tracer.update_generation_tokens(gen_ctx, in_tok, out_tok)
                gen_ctx.update(
                    output=answer[:500],   # first 500 chars for preview in UI
                    metadata={"input_tokens": in_tok, "output_tokens": out_tok},
                )

            duration = (time.time() - start) * 1000
            trace[-1] = TraceEntry(
                node="response_generator",
                status="completed",
                started_at=start,
                completed_at=time.time(),
                duration_ms=round(duration, 1),
                input_summary=f'Query: "{query[:60]}" | evidence_chunks={len(state.get("retrieved_documents", [])) + len(state.get("web_results", []))}',
                output_summary=f'Answer: "{answer[:80]}…"',
            )
            node_span.update(
                output={"answer_preview": answer[:200], "answer_length": len(answer)},
                metadata={
                    "duration_ms": round(duration, 1),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "retry_attempt": retry_count + 1,
                },
            )

            return {
                "evidence": evidence,
                "sources": sources,
                "generated_response": answer,
                "execution_trace": trace,
                "errors": errors,
            }

        except Exception as e:
            logger.error("response_generator failed: %s", e)
            errors.append(f"response_generator: {e}")
            trace[-1] = TraceEntry(
                node="response_generator",
                status="failed",
                started_at=start,
                completed_at=time.time(),
                input_summary=f'Query: "{query[:60]}"',
                output_summary="",
                error=str(e),
            )
            node_span.update(level="ERROR", status_message=str(e))
            return {
                "evidence": evidence,
                "sources": sources,
                "generated_response": f"I encountered an error generating the response: {e}",
                "execution_trace": trace,
                "errors": errors,
            }
