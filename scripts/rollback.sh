#!/usr/bin/env bash
# scripts/rollback.sh — Emergency rollback for all or individual services.
#
# WHAT IS A ROLLBACK?
#   A rollback reverts a Kubernetes Deployment to its previous version.
#   Kubernetes keeps a history of Deployment revisions (configurable).
#
#   kubectl rollout undo deployment/backend
#     → reverts to the last known-good version
#     → Kubernetes starts rolling the new pods back to the old image
#     → happens in seconds (old image is already pulled)
#
# WHY ROLLBACK?
#   Production deployments sometimes break things:
#     - LLM prompt change causes hallucinations in 5% of responses
#     - New chunk size degrades RAG quality
#     - Database migration has a bug
#
#   Rollback gives you a "get out of jail" button that works in seconds.
#
# USAGE:
#   # Roll back ALL services to previous version:
#   bash scripts/rollback.sh
#
#   # Roll back only one service:
#   bash scripts/rollback.sh --service backend
#
#   # Roll back to a specific revision:
#   bash scripts/rollback.sh --service backend --revision 3
#
#   # Show rollout history before rolling back:
#   bash scripts/rollback.sh --history

set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-assistant}"
SERVICE="${SERVICE:-all}"
REVISION="${REVISION:-0}"   # 0 = previous revision
HISTORY="${HISTORY:-false}"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLUE}[ROLLBACK]${NC} $*"; }
success() { echo -e "${GREEN}[ROLLBACK] ✅ $*${NC}"; }
error() { echo -e "${RED}[ROLLBACK] ❌ $*${NC}"; exit 1; }

SERVICES=("backend" "langgraph-agent" "worker" "frontend")

show_history() {
    for svc in "${SERVICES[@]}"; do
        log "Rollout history for $svc:"
        kubectl rollout history deployment/"$svc" -n "$NAMESPACE" 2>/dev/null || log "  (not deployed)"
        echo ""
    done
}

rollback_service() {
    local svc="$1"
    log "Rolling back deployment/$svc in namespace $NAMESPACE..."

    # Show current image before rollback
    local current_image
    current_image=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")
    log "  Current image: $current_image"

    if [ "$REVISION" -gt 0 ]; then
        kubectl rollout undo deployment/"$svc" \
            -n "$NAMESPACE" \
            --to-revision="$REVISION"
    else
        kubectl rollout undo deployment/"$svc" \
            -n "$NAMESPACE"
    fi

    # Wait for rollback to complete
    kubectl rollout status deployment/"$svc" \
        -n "$NAMESPACE" \
        --timeout=120s \
        || error "Rollback of $svc failed or timed out"

    # Show new image after rollback
    local new_image
    new_image=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")
    success "$svc rolled back → $new_image"
}

verify_rollback() {
    log "Verifying rollback..."
    local all_ok=true

    for svc in "${SERVICES[@]}"; do
        if kubectl get deployment "$svc" -n "$NAMESPACE" &>/dev/null; then
            local ready
            ready=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
                -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            local desired
            desired=$(kubectl get deployment "$svc" -n "$NAMESPACE" \
                -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")

            if [ "$ready" = "$desired" ]; then
                success "$svc: $ready/$desired ready"
            else
                error "$svc: only $ready/$desired ready after rollback"
                all_ok=false
            fi
        fi
    done

    if $all_ok; then
        success "All services healthy after rollback"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service) SERVICE="$2"; shift 2 ;;
            --revision) REVISION="$2"; shift 2 ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            --history) HISTORY="true"; shift ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
    done
}

main() {
    parse_args "$@"

    if [ "$HISTORY" = "true" ]; then
        show_history; exit 0
    fi

    echo -e "\n${BLUE}╔═══════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   ROLLBACK INITIATED               ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════╝${NC}"
    log "Namespace: $NAMESPACE | Service: $SERVICE | Revision: ${REVISION:-previous}"

    if [ "$SERVICE" = "all" ]; then
        for svc in "${SERVICES[@]}"; do
            rollback_service "$svc" || log "  $svc not deployed — skipping"
        done
    else
        rollback_service "$SERVICE"
    fi

    verify_rollback
    success "Rollback complete — run: bash scripts/verify-deployment.sh --mode k8s"
}

main "$@"
