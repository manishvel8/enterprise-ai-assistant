#!/usr/bin/env bash
# scripts/canary-deploy.sh — Canary deployment (gradual traffic shifting).
#
# WHAT IS CANARY DEPLOYMENT?
#   A canary deployment gradually shifts a small percentage of traffic to a
#   new version before fully rolling it out.
#   Named after "canary in a coal mine" — miners brought canaries to detect
#   toxic gas early (canary dies = gas present). Similarly, canary deployment
#   exposes a small fraction of users to the new version first.
#
#   Traffic split example:
#     Canary 10%:  90% → stable (v1), 10% → canary (v2)
#     Canary 25%:  75% → stable (v1), 25% → canary (v2)
#     Canary 100%: 0%  → stable (v1), 100% → canary (v2) = full rollout
#
# WHY CANARY vs BLUE/GREEN?
#   Canary:     ✅ Gradual exposure, monitors error rate as traffic increases
#               ✅ Much less risk for large user bases
#               ❌ More complex traffic management
#
#   Blue/Green: ✅ Simple instant switch
#               ❌ No gradual exposure (100% users see new version immediately)
#
# HOW WE IMPLEMENT CANARY (nginx-ingress):
#   We use nginx-ingress "canary" annotations to split traffic by weight.
#   Two Ingress resources:
#     1. Stable Ingress  → routes to stable service (old version)
#     2. Canary Ingress  → routes canary-weight% to canary service (new version)
#
# USAGE:
#   bash scripts/canary-deploy.sh --image ai-backend:v2.0.0 --weight 10
#   bash scripts/canary-deploy.sh --promote --weight 50  # increase to 50%
#   bash scripts/canary-deploy.sh --promote --weight 100 # full rollout
#   bash scripts/canary-deploy.sh --rollback              # remove canary

set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-assistant}"
SERVICE="${SERVICE:-backend}"
IMAGE="${IMAGE:-}"
WEIGHT="${WEIGHT:-10}"       # percentage of traffic to canary
PROMOTE="${PROMOTE:-false}"  # increase weight on existing canary
ROLLBACK="${ROLLBACK:-false}"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

log() { echo -e "${BLUE}[CANARY]${NC} $*"; }
success() { echo -e "${GREEN}[CANARY] ✅ $*${NC}"; }
warn() { echo -e "${YELLOW}[CANARY] ⚠️  $*${NC}"; }
error() { echo -e "${RED}[CANARY] ❌ $*${NC}"; exit 1; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --image) IMAGE="$2"; shift 2 ;;
            --service) SERVICE="$2"; shift 2 ;;
            --weight) WEIGHT="$2"; shift 2 ;;
            --promote) PROMOTE="true"; shift ;;
            --rollback) ROLLBACK="true"; shift ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

create_canary_deployment() {
    local image="$1"
    log "Creating canary deployment with image: $image"

    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE}-canary
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE}
    track: canary
spec:
  # Only 1 canary replica — it receives WEIGHT% of traffic.
  # 1 canary + 2 stable = canary gets ~33% if weight=100, but nginx
  # uses the weight annotation (not replica ratio) for precise control.
  replicas: 1
  selector:
    matchLabels:
      app: ${SERVICE}
      track: canary
  template:
    metadata:
      labels:
        app: ${SERVICE}
        track: canary
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
---
# Canary Service — routes to canary pods only
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE}-canary-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ${SERVICE}
    track: canary
  ports:
    - port: 8000
      targetPort: 8000
EOF

    log "Waiting for canary deployment..."
    kubectl rollout status deployment/${SERVICE}-canary \
        -n "$NAMESPACE" --timeout=120s \
        || error "Canary deployment failed"

    success "Canary deployment ready"
}

set_canary_weight() {
    local weight="$1"
    log "Setting canary traffic weight to ${weight}%..."

    # nginx-ingress canary annotations split traffic.
    # nginx.ingress.kubernetes.io/canary-weight: "10" → 10% of traffic
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${SERVICE}-canary-ingress
  namespace: ${NAMESPACE}
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "${weight}"
    # Optionally: route by header (useful for internal testing)
    # nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    # nginx.ingress.kubernetes.io/canary-by-header-value: "true"
spec:
  ingressClassName: nginx
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ${SERVICE}-canary-service
                port:
                  number: 8000
EOF

    success "Canary weight set to ${weight}%"
    warn "Monitor error rates in Grafana before increasing weight"
    log "Check metrics: rate(http_requests_total{track='canary'}[5m])"
}

rollback_canary() {
    log "Rolling back canary (removing canary ingress and deployment)..."
    kubectl delete ingress "${SERVICE}-canary-ingress" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service "${SERVICE}-canary-service" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete deployment "${SERVICE}-canary" -n "$NAMESPACE" --ignore-not-found=true
    success "Canary rolled back — 100% traffic on stable"
}

promote_to_full() {
    log "Promoting canary to full (100%) deployment..."
    # Get canary image
    local canary_image
    canary_image=$(kubectl get deployment "${SERVICE}-canary" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null \
        || echo "$IMAGE")

    # Update the main stable deployment to use canary image
    kubectl set image deployment/${SERVICE} \
        ${SERVICE}=${canary_image} \
        -n "$NAMESPACE"
    kubectl rollout status deployment/${SERVICE} -n "$NAMESPACE" --timeout=180s

    # Remove canary
    rollback_canary
    success "Full promotion complete — all traffic on new version"
}

print_monitoring_hints() {
    log "Monitor canary with these Prometheus queries:"
    echo "  Error rate canary:  rate(http_requests_total{track='canary', status_code=~'5..'}[5m]) / rate(http_requests_total{track='canary'}[5m])"
    echo "  P95 latency canary: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{track='canary'}[5m]))"
    echo "  Error rate stable:  rate(http_requests_total{track='stable', status_code=~'5..'}[5m]) / rate(http_requests_total{track='stable'}[5m])"
    echo ""
    log "Canary runbook:"
    echo "  OK (error rate similar): bash scripts/canary-deploy.sh --promote --weight 50"
    echo "  Problem (error rate up): bash scripts/canary-deploy.sh --rollback"
}

main() {
    parse_args "$@"

    if [ "$ROLLBACK" = "true" ]; then
        rollback_canary; exit 0
    fi

    if [ "$WEIGHT" = "100" ] || [ "$PROMOTE" = "true" ] && [ "$WEIGHT" = "100" ]; then
        promote_to_full; exit 0
    fi

    if [ "$PROMOTE" = "true" ]; then
        set_canary_weight "$WEIGHT"
        print_monitoring_hints
        exit 0
    fi

    if [ -z "$IMAGE" ]; then
        error "Usage: $0 --image <image:tag> [--weight 10] [--service backend]"
    fi

    log "Canary Deployment:"
    log "  Service: $SERVICE  |  Image: $IMAGE  |  Initial weight: ${WEIGHT}%"

    create_canary_deployment "$IMAGE"
    set_canary_weight "$WEIGHT"
    print_monitoring_hints
}

main "$@"
