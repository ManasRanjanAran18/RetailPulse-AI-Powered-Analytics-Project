"""RFM feature engineering + K-Means customer segmentation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import MODELS_DIR, SEED

RFM_COLUMNS = ["recency", "frequency", "monetary"]
SEGMENT_NAMES_4 = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]


def build_rfm(tx: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Recency (days), Frequency (orders), Monetary (revenue) per customer."""
    as_of = pd.Timestamp(as_of) if as_of is not None else tx["order_date"].max() + pd.Timedelta(days=1)
    hist = tx[tx["order_date"] < as_of]
    g = hist.groupby("customer_id").agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    )
    g["recency"] = (as_of - g["last_purchase"]).dt.days
    return g[RFM_COLUMNS].reset_index()


def _make_pipeline(k: int, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def evaluate_k(rfm: pd.DataFrame, ks=range(2, 8), seed: int = SEED) -> pd.DataFrame:
    """Inertia and silhouette for a range of k (for the elbow / silhouette plot)."""
    rows = []
    X = rfm[RFM_COLUMNS]
    for k in ks:
        pipe = _make_pipeline(k, seed).fit(X)
        Xt = pipe[:-1].transform(X)
        labels = pipe.named_steps["kmeans"].labels_
        rows.append({"k": k, "inertia": pipe.named_steps["kmeans"].inertia_,
                     "silhouette": silhouette_score(Xt, labels)})
    return pd.DataFrame(rows)


def _name_clusters(profile: pd.DataFrame, k: int) -> Dict[int, str]:
    """Business names for clusters. For k=4: the two most-recent clusters are
    'active' (higher spend -> Champions, lower -> Loyal Customers) and the two
    least-recent are 'lapsed' (higher spend -> At Risk, lower -> Hibernating)."""
    if k != 4:
        order = profile.sort_values("monetary", ascending=False).index
        return {int(c): f"Segment {i + 1}" for i, c in enumerate(order)}
    by_recency = profile.sort_values("recency").index.tolist()
    active, lapsed = by_recency[:2], by_recency[2:]
    active = sorted(active, key=lambda c: -profile.loc[c, "monetary"])
    lapsed = sorted(lapsed, key=lambda c: -profile.loc[c, "monetary"])
    names = dict(zip(active, ["Champions", "Loyal Customers"]))
    names.update(dict(zip(lapsed, ["At Risk", "Hibernating"])))
    return {int(c): n for c, n in names.items()}


def fit_segmentation(rfm: pd.DataFrame, k: int = 4, seed: int = SEED) -> Dict:
    """Fit K-Means on log-scaled RFM and give clusters business names."""
    X = rfm[RFM_COLUMNS]
    pipe = _make_pipeline(k, seed).fit(X)
    raw_labels = pipe.named_steps["kmeans"].labels_

    tmp = rfm.copy()
    tmp["cluster"] = raw_labels
    prof = tmp.groupby("cluster")[RFM_COLUMNS].mean()
    mapping = _name_clusters(prof, k)

    Xt = pipe[:-1].transform(X)
    return {
        "pipeline": pipe,
        "cluster_to_segment": mapping,
        "k": k,
        "silhouette": float(silhouette_score(Xt, raw_labels)),
    }


def predict_segments(bundle: Dict, rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    clusters = bundle["pipeline"].predict(rfm[RFM_COLUMNS])
    out["cluster"] = clusters
    out["segment"] = [bundle["cluster_to_segment"][int(c)] for c in clusters]
    return out


def segment_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    s = segmented.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1)
    s["revenue_share_pct"] = (100 * s["total_revenue"] / s["total_revenue"].sum()).round(1)
    return s.sort_values("avg_monetary", ascending=False).reset_index()


def save_segmentation(bundle: Dict, path: Path | None = None) -> Path:
    path = Path(path) if path else MODELS_DIR / "segmentation.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_segmentation(path: Path | None = None) -> Dict:
    return joblib.load(Path(path) if path else MODELS_DIR / "segmentation.joblib")
