"""Optional MLflow tracking helpers. Everything here is a safe no-op when
MLflow is not installed or tracking is disabled, so the lean deployment image
(which does not ship MLflow) keeps working."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from src.config import ROOT

try:  # pragma: no cover - depends on environment
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "retailpulse")


def is_available() -> bool:
    return mlflow is not None


def setup(enabled: bool = True) -> bool:
    """Configure MLflow. Returns True if tracking is active."""
    if not enabled or mlflow is None:
        return False
    # SQLite is MLflow's recommended local backend (the file store is deprecated).
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return True


@contextmanager
def run(name: str, active: bool, params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None) -> Iterator[Any]:
    """Context manager yielding a tracker with .metrics() / .artifact()."""
    if not active or mlflow is None:
        yield _NullTracker()
        return
    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
        yield _Tracker()


class _NullTracker:
    def metrics(self, d: Dict[str, float]) -> None:
        pass

    def artifact(self, path: str) -> None:
        pass


class _Tracker:
    def metrics(self, d: Dict[str, float]) -> None:
        clean = {k: float(v) for k, v in d.items()
                 if isinstance(v, (int, float)) and v == v}
        if clean:
            mlflow.log_metrics(clean)

    def artifact(self, path: str) -> None:
        mlflow.log_artifact(str(path))
