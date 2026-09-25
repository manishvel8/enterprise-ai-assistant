#!/usr/bin/env bash
# k8s-deploy.sh — Deploy the full application to Kubernetes.
#
# Prerequisites:
#   - kubectl configured with your cluster context
#   - Docker images built and pushed to your registry
#   - Kubernetes cluster running (minikube or cloud)
#
# Usage:
#   chmod +x scripts/k8s-deploy.sh
#   ./scripts/k8s-deploy.sh
#
# Milestone 27: kubectl apply all manifests and verify pod status.

set -euo pipefail

NAMESPACE="ai-assistant"
K8S_DIR="$(dirname "$0")/../k8s"

echo "=========================================="
echo " Enterprise AI Assistant — K8s Deploy"
echo "=========================================="

# ── Step 1: Create namespace ────────────────────────────────────────────────
echo ""
echo "1. Creating namespace..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

# ── Step 2: Apply ConfigMap and Secrets ─────────────────────────────────────
echo ""
echo "2. Applying ConfigMap and Secrets..."
kubectl apply -f "$K8S_DIR/configmap.yaml" -n "$NAMESPACE"

echo "   ⚠️  secret.yaml contains placeholder values."
echo "   Edit k8s/secret.yaml with your real secrets before applying."
echo "   Or use: kubectl create secret generic ai-assistant-secrets --from-env-file=.env"
kubectl apply -f "$K8S_DIR/secret.yaml" -n "$NAMESPACE"

# ── Step 3: Deploy databases (StatefulSets) ──────────────────────────────────
echo ""
echo "3. Deploying databases..."
kubectl apply -f "$K8S_DIR/postgres/statefulset.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/postgres/service.yaml" -n "$NAMESPACE"

kubectl apply -f "$K8S_DIR/neo4j/statefulset.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/neo4j/service.yaml" -n "$NAMESPACE"

kubectl apply -f "$K8S_DIR/qdrant/statefulset.yaml" -n "$NAMESPACE"

kubectl apply -f "$K8S_DIR/minio/statefulset.yaml" -n "$NAMESPACE"

# ── Step 4: Deploy Redis ─────────────────────────────────────────────────────
echo ""
echo "4. Deploying Redis..."
kubectl apply -f "$K8S_DIR/redis/deployment.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/redis/service.yaml" -n "$NAMESPACE"

# ── Step 5: Wait for databases to be ready ───────────────────────────────────
echo ""
echo "5. Waiting for database pods to be ready (may take 60-120 seconds)..."
kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=120s || true
kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=60s || true
kubectl wait --for=condition=ready pod -l app=qdrant -n "$NAMESPACE" --timeout=60s || true

# ── Step 6: Deploy application services ─────────────────────────────────────
echo ""
echo "6. Deploying backend and worker..."
kubectl apply -f "$K8S_DIR/backend/deployment.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/backend/service.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/backend/hpa.yaml" -n "$NAMESPACE"

kubectl apply -f "$K8S_DIR/worker/deployment.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/worker/hpa.yaml" -n "$NAMESPACE"

echo ""
echo "7. Deploying frontend..."
kubectl apply -f "$K8S_DIR/frontend/deployment.yaml" -n "$NAMESPACE"
kubectl apply -f "$K8S_DIR/frontend/service.yaml" -n "$NAMESPACE"

# ── Step 7: Apply Ingress ────────────────────────────────────────────────────
echo ""
echo "8. Applying Ingress..."
kubectl apply -f "$K8S_DIR/ingress.yaml" -n "$NAMESPACE"

# ── Step 8: Wait for all deployments ─────────────────────────────────────────
echo ""
echo "9. Waiting for all deployments to be ready..."
kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout=120s || true
kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout=120s || true
kubectl rollout status deployment/worker -n "$NAMESPACE" --timeout=120s || true

# ── Step 9: Show status ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Deployment Status"
echo "=========================================="
echo ""
echo "Pods:"
kubectl get pods -n "$NAMESPACE"
echo ""
echo "Services:"
kubectl get services -n "$NAMESPACE"
echo ""
echo "Deployments:"
kubectl get deployments -n "$NAMESPACE"
echo ""
echo "StatefulSets:"
kubectl get statefulsets -n "$NAMESPACE"
echo ""
echo "HPA:"
kubectl get hpa -n "$NAMESPACE"
echo ""
echo "Ingress:"
kubectl get ingress -n "$NAMESPACE"

echo ""
echo "=========================================="
echo " ✅ Deployment complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Monitor pods: kubectl get pods -n ai-assistant -w"
echo "  - Check backend logs: kubectl logs -f deployment/backend -n ai-assistant"
echo "  - Port-forward backend: kubectl port-forward svc/backend-service 8000:8000 -n ai-assistant"
echo "  - Port-forward frontend: kubectl port-forward svc/frontend-service 4200:80 -n ai-assistant"
echo "  - Scale backend: kubectl scale deployment/backend --replicas=5 -n ai-assistant"
