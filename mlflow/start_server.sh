#!/usr/bin/env bash
# Start a local MLflow tracking server using the project's SQLite backend.
set -euo pipefail
cd "$(dirname "$0")/.."
exec mlflow server --host 0.0.0.0 --port "${PORT:-5000}" \
  --backend-store-uri "sqlite:///mlflow/mlflow.db" \
  --default-artifact-root "./mlruns"
