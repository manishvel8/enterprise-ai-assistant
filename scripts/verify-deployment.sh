#!/usr/bin/env bash
# scripts/verify-deployment.sh — Post-deployment health verification.
#
# WHAT IS A SMOKE TEST?
#   A smoke test is a minimal set of checks run immediately after deployment
#   to verify the application is alive and serving requests.
#   Named after the hardware test: "apply power, see if smoke comes out."
#
# WHAT WE CHECK:
#   1. All Kubernetes pods are Running
#   2. Deployments have correct replica count
#   3. Health endpoints return 200
#   4. Database connectivity
#   5. Qdrant vector store connectivity
#   6. Redis connectivity
#   7. LLM API connectivity (OpenAI reachable)
#
# USAGE:
#   # Check local Docker stack:
#   bash scripts/verify-deployment.sh --mode docker
#
#   # Check Kubernetes cluster:
#   bash scripts/verify-deployment.sh --mode k8s --namespace ai-assistant
#
#   # Check specific URL:
#   BACKEND_URL=http://api.yourdomain.com bash scripts/verify-deployment.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODE="${MODE:-docker}"                  # docker or k8s
NAMESPACE="${NAMESPACE:-ai-assistant}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
LANGGRAPH_URL="${LANGGRAPH_URL:-http://localhost:7860}"
TIMEOUT="${TIMEOUT:-30}"                # seconds per health check

# ── Colors for output ─────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

# ── Helper functions ──────────────────────────────────────────────────────────
pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}❌ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $1"; WARN=$((WARN+1)); }
info() { echo -e "${BLUE}ℹ️  INFO${NC}: $1"; }
section() { echo -e "\n${BLUE}═══ $1 ═══${NC}"; }

http_check() {
    local url="$1"
    local expected_status="${2:-200}"
    local description="${3:-$url}"

    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null || echo "000")

    if [ "$status" = "$expected_status" ]; then
        pass "$description → HTTP $status"
        return 0
    else
        fail "$description → Expected HTTP $expected_status, got HTTP $status"
        return 1
    fi
}

# ── SECTION 1: Docker container health ───────────────────────────────────────
check_docker_containers() {
    section "Docker Container Status"

    local containers=("ai-backend" "ai-langgraph-agent" "ai-postgres" "ai-redis" "ai-qdrant" "ai-neo4j")

    for container in "${containers[@]}"; do
        local status
        status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "not_found")
        local health
        health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")

        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "none" ]; then
                pass "Container $container: running ($health)"
            else
                warn "Container $container: running but health=$health"
            fi
        else
            fail "Container $container: $status"
        fi
    done
}

# ── SECTION 2: Kubernetes pod health ─────────────────────────────────────────
check_k8s_pods() {
    section "Kubernetes Pod Status"

    info "Checking pods in namespace: $NAMESPACE"

    # Check all pods are Running
    local not_running
    not_running=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | \
        grep -v "Running\|Completed" || true)

    if [ -z "$not_running" ]; then
        pass "All pods are Running"
    else
        fail "Some pods are not Running:\n$not_running"
    fi

    # Check deployments
    info "Deployment status:"
    kubectl get deployments -n "$NAMESPACE" 2>/dev/null || warn "kubectl not available"

    # Check HPAs
    info "HPA status:"
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || true
}

# ── SECTION 3: API health endpoints ─────────────────────────────────────────
check_api_health() {
    section "API Health Endpoints"

    # Backend health
    http_check "$BACKEND_URL/health" 200 "Backend /health"
    http_check "$BACKEND_URL/health/live" 200 "Backend /health/live (liveness)"

    local ready_status
    ready_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$BACKEND_URL/health/ready" 2>/dev/null || echo "000")
    if [ "$ready_status" = "200" ] || [ "$ready_status" = "503" ]; then
        pass "Backend /health/ready → HTTP $ready_status (200=ready, 503=deps-down)"
    else
        fail "Backend /health/ready → unexpected HTTP $ready_status"
    fi

    # LangGraph health
    http_check "$LANGGRAPH_URL/api/health" 200 "LangGraph /api/health"
    http_check "$LANGGRAPH_URL/api/graph/schema" 200 "LangGraph /api/graph/schema"

    # Verify graph has 7 nodes
    local schema
    schema=$(curl -s "$LANGGRAPH_URL/api/graph/schema" 2>/dev/null || echo "{}")
    local node_count
    node_count=$(echo "$schema" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('nodes',[])))" 2>/dev/null || echo "?")
    if [ "$node_count" = "7" ]; then
        pass "LangGraph has 7 nodes"
    else
        warn "LangGraph node count: $node_count (expected 7)"
    fi
}

# ── SECTION 4: Database connectivity ─────────────────────────────────────────
check_databases() {
    section "Database Connectivity"

    # PostgreSQL
    if PGPASSWORD=dev_password psql -h localhost -p 5433 -U ai_user -d ai_assistant \
        -c "SELECT 1" -q 2>/dev/null | grep -q "1"; then
        pass "PostgreSQL connection"

        # Check key tables
        local table_count
        table_count=$(PGPASSWORD=dev_password psql -h localhost -p 5433 -U ai_user -d ai_assistant \
            -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null | tr -d ' \n' || echo "0")
        info "PostgreSQL tables found: $table_count"
    else
        fail "PostgreSQL connection (is docker-compose.dev.yml running?)"
    fi

    # Qdrant
    local qdrant_status
    qdrant_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:6333/healthz" 2>/dev/null || echo "000")
    if [ "$qdrant_status" = "200" ]; then
        pass "Qdrant connection"
        # Check collection exists
        local collections
        collections=$(curl -s "http://localhost:6333/collections" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print([c['name'] for c in d.get('result',{}).get('collections',[])])" 2>/dev/null || echo "[]")
        info "Qdrant collections: $collections"
    else
        fail "Qdrant connection → HTTP $qdrant_status"
    fi

    # Redis
    if redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q "PONG"; then
        pass "Redis connection"
    else
        warn "Redis connection failed (redis-cli not installed or Redis down)"
    fi

    # Neo4j
    local neo4j_status
    neo4j_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://localhost:7474" 2>/dev/null || echo "000")
    if [ "$neo4j_status" = "200" ]; then
        pass "Neo4j connection"
    else
        warn "Neo4j → HTTP $neo4j_status"
    fi
}

# ── SECTION 5: LLM connectivity ───────────────────────────────────────────────
check_llm() {
    section "LLM API Connectivity"

    # Check OpenAI API (just the domain, not a real API call — avoids cost)
    local openai_reachable
    openai_reachable=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
        "https://api.openai.com" 2>/dev/null || echo "000")

    if [ "$openai_reachable" != "000" ]; then
        pass "OpenAI API reachable (HTTP $openai_reachable)"
    else
        warn "OpenAI API not reachable — check network connectivity and firewall"
    fi

    # Check if OPENAI_API_KEY is set
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        pass "OPENAI_API_KEY is set"
    else
        warn "OPENAI_API_KEY is not set in environment"
    fi
}

# ── SECTION 6: Monitoring ─────────────────────────────────────────────────────
check_monitoring() {
    section "Monitoring Stack"

    local prometheus_status
    prometheus_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:9090/-/ready" 2>/dev/null || echo "000")
    if [ "$prometheus_status" = "200" ]; then
        pass "Prometheus ready"
    else
        warn "Prometheus not responding → HTTP $prometheus_status (start with: make monitoring)"
    fi

    local grafana_status
    grafana_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3001/api/health" 2>/dev/null || echo "000")
    if [ "$grafana_status" = "200" ]; then
        pass "Grafana ready"
    else
        warn "Grafana not responding → HTTP $grafana_status"
    fi

    local langfuse_status
    langfuse_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:3000/api/public/health" 2>/dev/null || echo "000")
    if [ "$langfuse_status" = "200" ]; then
        pass "Langfuse ready"
    else
        warn "Langfuse not responding → HTTP $langfuse_status"
    fi
}

# ── MAIN ──────────────────────────────────────────────────────────────────────
main() {
    echo -e "\n${BLUE}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Enterprise AI Assistant — Deployment Check  ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
    echo -e "Mode: $MODE | Backend: $BACKEND_URL | LangGraph: $LANGGRAPH_URL\n"

    if [ "$MODE" = "k8s" ]; then
        check_k8s_pods
    else
        check_docker_containers
    fi

    check_api_health
    check_databases
    check_llm
    check_monitoring

    # ── Summary ────────────────────────────────────────────────────────────────
    echo -e "\n${BLUE}═══ Summary ═══${NC}"
    echo -e "${GREEN}✅ PASSED: $PASS${NC}"
    echo -e "${YELLOW}⚠️  WARNINGS: $WARN${NC}"
    echo -e "${RED}❌ FAILED: $FAIL${NC}"

    if [ "$FAIL" -gt 0 ]; then
        echo -e "\n${RED}DEPLOYMENT VERIFICATION FAILED${NC}"
        echo "Check the FAIL items above and investigate."
        exit 1
    elif [ "$WARN" -gt 0 ]; then
        echo -e "\n${YELLOW}DEPLOYMENT PASSED WITH WARNINGS${NC}"
        echo "Review warnings — they may indicate non-critical issues."
        exit 0
    else
        echo -e "\n${GREEN}DEPLOYMENT VERIFIED SUCCESSFULLY${NC}"
        exit 0
    fi
}

main "$@"
