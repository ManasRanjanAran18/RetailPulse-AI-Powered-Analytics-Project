# Kubernetes deployment

```bash
# 1. Edit image names (replace YOUR_GITHUB_USERNAME) and the ingress host
# 2. Apply everything (namespace first - files are numbered/alphabetical)
kubectl apply -f k8s/

# 3. Check
kubectl -n retailpulse get pods,svc,ingress,hpa

# Quick local access without an ingress
kubectl -n retailpulse port-forward svc/retailpulse-dashboard 8501:80
```

| File | Purpose |
|---|---|
| `00-namespace.yaml` | `retailpulse` namespace |
| `deployment.yaml` | Dashboard, 2 replicas, health probes, resource limits |
| `service.yaml` | ClusterIP service |
| `ingress.yaml` | HTTP(S) entry point (ingress-nginx) |
| `hpa.yaml` | Autoscale 2-6 pods at 70% CPU |
| `mlflow.yaml` | MLflow tracking server + persistent volume |
| `cronjob-retrain.yaml` | Weekly retraining job |

Images are built and pushed to GHCR by `.github/workflows/publish-image.yml`.
