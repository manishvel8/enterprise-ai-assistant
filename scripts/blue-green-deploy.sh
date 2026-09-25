#!/usr/bin/env bash
# scripts/blue-green-deploy.sh — Blue/Green deployment for zero-downtime updates.
#
# WHAT IS BLUE/GREEN DEPLOYMENT?
#   Blue/Green = running two identical production environments (blue and green).
#   At any time, only ONE receives traffic (let's say "blue" = current live).
#
#   Deployment steps:
#   1. Deploy new version to "green" environment (users still on blue)
#   2. Run smoke tests on green (verify it works)
#   3. Switch traffic: update Ingress/Service to point to green
#   4. Blue is now idle (immediate rollback available: switch back to blue)
#   5. After confirming green is stable: delete blue deployment
#
# WHY BLUE/GREEN vs ROLLING UPDATE?
#   Rolling update: replaces pods one by one (some pods run old, some run new code simultaneously)
#     ✅ Less resource usage (no duplicate deployment)
#     ❌ Old and new versions serve traffic simultaneously (API incompatibility risk)
#
#   Blue/Green:
#     ✅ No mixed versions in production (all old OR all new)
#     ✅ Instant rollback (switch selector back to blue)
#     ✅ Test green before switching traffic
#     ❌ Doubles resource usage during the switch
#
# WHEN TO USE BLUE/GREEN:
#   - Breaking API changes (can't run old/new simultaneously)
#   - Database schema migrations (needs coordinated cutover)
#   - When you need instant rollback capability
#
# USAGE:
#   # Deploy new version to green:
#   bash scripts/blue-green-deploy.sh --image ai-backend:v2.0.0 --service backend
#
#   # Roll back to blue (previous version):
#   bash scripts/blue-green-deploy.sh --rollback --service backend

set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-assistant}"
SERVICE="${SERVICE:-backend}"
IMAGE="${IMAGE:-}"
ROLLBACK="${ROLLBACK:-false}"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLUE}[BG-DEPLOY]${NC} $*"; }
success() { echo -e "${GREEN}[BG-DEPLOY] ✅ $*${NC}"; }
error() { echo -e "${RED}[BG-DEPLOY] ❌ $*${NC}"; exit 1; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --image) IMAGE="$2"; shift 2 ;;
            --service) SERVICE="$2"; shift 2 ;;
            --rollback) ROLLBACK="true"; shift ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

get_current_color() {
    # Returns "blue" or "green" — whichever is currently receiving traffic
    kubectl get service "${SERVICE}-service" -n "$NAMESPACE" \
        -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue"
}

get_inactive_color() {
    local current
    current=$(get_current_color)
    if [ "$current" = "blue" ]; then echo "green"; else echo "blue"; fi
}

deploy_to_color() {
    local color="$1"
    local image="$2"

    log "Deploying $image to ${color} environment..."

    # Create a new deployment with the color label
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE}-${color}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE}
    color: ${color}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ${SERVICE}
      color: ${color}
  template:
    metadata:
      labels:
        app: ${SERVICE}
        color: ${color}
    spec:
      containers:
        - name: ${SERVICE}
          image: ${image}
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: ai-assistant-config
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 6
EOF

    log "Waiting for ${color} deployment to be ready..."
    kubectl rollout status deployment/${SERVICE}-${color} \
        -n "$NAMESPACE" \
        --timeout=120s \
        || error "${color} deployment failed health checks"

    success "${color} deployment is ready"
}

switch_traffic() {
    local target_color="$1"
    log "Switching traffic to ${target_color}..."

    # Update the Service selector to point to the new color
    kubectl patch service "${SERVICE}-service" \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"app\":\"${SERVICE}\",\"color\":\"${target_color}\"}}}"

    success "Traffic switched to ${target_color}"
}

run_smoke_tests() {
    local color="$1"
    log "Running smoke tests against ${color}..."

    # Get a pod from the target color
    local pod
    pod=$(kubectl get pods -n "$NAMESPACE" \
        -l "app=${SERVICE},color=${color}" \
        --no-headers \
        -o custom-columns=NAME:.metadata.name | head -1)

    if [ -z "$pod" ]; then
        error "No ${color} pod found for smoke test"
    fi

    # Health check inside the pod
    kubectl exec -n "$NAMESPACE" "$pod" -- \
        curl -f http://localhost:8000/health \
        && success "Smoke test PASSED on ${color}" \
        || error "Smoke test FAILED on ${color}"
}

cleanup_old_deployment() {
    local old_color="$1"
    log "Removing old ${old_color} deployment..."
    kubectl delete deployment "${SERVICE}-${old_color}" -n "$NAMESPACE" \
        --ignore-not-found=true
    success "${old_color} deployment removed"
}

main() {
    parse_args "$@"

    if [ "$ROLLBACK" = "true" ]; then
        local current
        current=$(get_current_color)
        local previous
        previous=$(get_inactive_color)

        log "ROLLBACK: switching from ${current} back to ${previous}"
        switch_traffic "$previous"
        success "Rolled back to ${previous}"
        exit 0
    fi

    if [ -z "$IMAGE" ]; then
        error "Usage: $0 --image <image:tag> --service <service>"
    fi

    local current_color
    current_color=$(get_current_color)
    local new_color
    new_color=$(get_inactive_color)

    log "Blue/Green Deploy Summary:"
    log "  Service:      $SERVICE"
    log "  New image:    $IMAGE"
    log "  Current live: $current_color"
    log "  Deploying to: $new_color"

    # Step 1: Deploy to inactive slot
    deploy_to_color "$new_color" "$IMAGE"

    # Step 2: Smoke test new deployment (before switching traffic)
    run_smoke_tests "$new_color"

    # Step 3: Switch traffic
    switch_traffic "$new_color"

    # Step 4: Cleanup old deployment (after confirming new is live)
    sleep 30   # brief wait for in-flight requests to complete
    cleanup_old_deployment "$current_color"

    success "Blue/Green deployment complete! Traffic on: $new_color"
}

main "$@"
