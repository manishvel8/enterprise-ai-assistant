# Hands-on Learning Index

Start here if you are preparing for GenAI / RAG interviews using this codebase.

| Doc | Purpose |
|-----|---------|
| [00-test-the-system.md](00-test-the-system.md) | Bring-up, health checks, first RAG round-trip |
| [01-document-ingestion-e2e.md](01-document-ingestion-e2e.md) | Upload → MinIO → Celery → chunk → embed → stores |
| [02-query-rag-agents-e2e.md](02-query-rag-agents-e2e.md) | One chat query through all agents |
| [../interviews/interview-qa-mapped-to-project.md](../interviews/interview-qa-mapped-to-project.md) | Q1–Q81 answers mapped to this repo |
| [../rehearsal/rag-and-agents-drill.md](../rehearsal/rag-and-agents-drill.md) | Spoken drills + whiteboard sketches |
| [../fixtures/sample-rag-fact.txt](../fixtures/sample-rag-fact.txt) | Unique facts for retrieval demos |

Scripts:

- `scripts/verify-stack.sh` — health (+ optional file smoke)
- `scripts/test-pipeline.py` — upload → poll → chat
- `scripts/load-test.py` — concurrent chat load
