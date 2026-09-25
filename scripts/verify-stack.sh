#!/usr/bin/env bash
# verify-stack.sh — Run after docker-compose is up. Checks health and optional RAG round-trip.
#
# Usage:
#   ./scripts/verify-stack.sh
#   ./scripts/verify-stack.sh --file docs/fixtures/sample-rag-fact.txt
#
# Note: use `docker-compose` (hyphen) on this machine if `docker compose` is unavailable.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) FILE="$2"; shift 2 ;;
    --url) BACKEND_URL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=============================================="
echo " Enterprise AI Assistant — Stack Verification"
echo " Backend: $BACKEND_URL"
echo "=============================================="

fail=0

check() {
  local name="$1"
  local url="$2"
  echo ""
  echo "→ $name"
  if curl -sf --max-time 5 "$url" >/tmp/eai_verify.json 2>/dev/null; then
    echo "  OK: $(cat /tmp/eai_verify.json | head -c 200)"
  else
    echo "  FAIL: $url not reachable"
    fail=1
  fi
}

check "GET /health" "$BACKEND_URL/health"
check "GET /health/live" "$BACKEND_URL/health/live"
check "GET /health/ready" "$BACKEND_URL/health/ready"
check "GET /api/eval/health" "$BACKEND_URL/api/eval/health"

if [[ -n "$FILE" ]]; then
  echo ""
  echo "→ Running smoke pipeline with file: $FILE"
  cd "$ROOT"
  python3 scripts/test-pipeline.py --file "$FILE" || fail=1
fi

echo ""
if [[ "$fail" -eq 0 ]]; then
  echo "✅ Verification passed (or health-only OK)."
  echo "Next: open http://localhost:4200 → upload docs/fixtures/sample-rag-fact.txt"
  echo "Ask: What was Project Aurora Q3 revenue?"
else
  echo "❌ Verification failed."
  echo ""
  echo "Bring-up (this host uses docker-compose, not docker compose):"
  echo "  cd $ROOT"
  echo "  # Edit .env and set a real OPENAI_API_KEY"
  echo "  docker-compose -f docker-compose.dev.yml up -d    # infra only"
  echo "  # OR full stack:"
  echo "  docker-compose up --build"
  echo ""
  echo "If image pull fails (TLS timeout), retry when Docker Hub is reachable,"
  echo "or load images from an internal registry."
  exit 1
fi
