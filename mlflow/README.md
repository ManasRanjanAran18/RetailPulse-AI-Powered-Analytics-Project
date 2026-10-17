# MLflow

Experiment tracking for RetailPulse (SQLite backend, no server required locally).

```bash
# Log all pipeline runs
python scripts/run_pipeline.py --mlflow

# Browse the UI (http://localhost:5000)
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

# Or use a remote/shared server
export MLFLOW_TRACKING_URI=http://my-mlflow-server:5000
```

The tracking database (`mlflow/mlflow.db`) is git-ignored. Notebooks `07` and `14`
demonstrate parameter sweeps, artifact logging, the Model Registry and champion selection.

Run MLflow + dashboard together: `docker compose -f docker/docker-compose.yml up --build dashboard mlflow`.
