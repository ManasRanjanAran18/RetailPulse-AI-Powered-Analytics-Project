"""Churn prediction: feature engineering, labelling, training, scoring.

Definition: among customers who were *active* (bought within the last
ACTIVE_WINDOW_DAYS) at a cutoff date, a customer has churned if they make no
purchase in the following CHURN_HORIZON_DAYS. Features use only data strictly
before the cutoff, so there is no label leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import (ACTIVE_WINDOW_DAYS, CHURN_HORIZON_DAYS, LOYALTY_TIERS,
                        MODELS_DIR, REGIONS, SEED)

FEATURE_COLUMNS = [
    "recency", "frequency", "monetary", "avg_order_value", "avg_qty",
    "n_categories", "n_stores", "tenure_days", "avg_gap_days", "recency_ratio",
    "orders_30d", "orders_90d", "orders_prev_90d", "order_trend",
    "loyalty_tier_num", "age",
] + [f"region_{r}" for r in REGIONS]


def build_features(tx: pd.DataFrame, customers: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer-level features computed from transactions strictly before `as_of`."""
    as_of = pd.Timestamp(as_of)
    hist = tx[tx["order_date"] < as_of]
    g = hist.groupby("customer_id")
    f = g.agg(
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        avg_qty=("quantity", "mean"),
        n_categories=("category", "nunique"),
        n_stores=("store", "nunique"),
    )
    f["recency"] = (as_of - f["last_purchase"]).dt.days
    f["tenure_days"] = (as_of - f["first_purchase"]).dt.days
    f["avg_gap_days"] = f["tenure_days"] / f["frequency"].clip(lower=1)
    f["recency_ratio"] = f["recency"] / f["avg_gap_days"].clip(lower=1)

    def window_orders(days_from: int, days_to: int) -> pd.Series:
        lo = as_of - pd.Timedelta(days=days_from)
        hi = as_of - pd.Timedelta(days=days_to)
        m = (hist["order_date"] >= lo) & (hist["order_date"] < hi)
        return hist.loc[m].groupby("customer_id")["order_id"].nunique().reindex(f.index, fill_value=0)

    f["orders_30d"] = window_orders(30, 0)
    f["orders_90d"] = window_orders(90, 0)
    f["orders_prev_90d"] = window_orders(180, 90)
    f["order_trend"] = (f["orders_90d"] + 1) / (f["orders_prev_90d"] + 1)

    cust = customers.set_index("customer_id")
    f["loyalty_tier_num"] = cust["loyalty_tier"].map({t: i for i, t in enumerate(LOYALTY_TIERS)}).reindex(f.index)
    f["age"] = cust["age"].reindex(f.index)
    for r in REGIONS:
        f[f"region_{r}"] = (cust["region"].reindex(f.index) == r).astype(int)

    f["loyalty_tier_num"] = f["loyalty_tier_num"].fillna(0)
    f["age"] = f["age"].fillna(f["age"].median())
    return f[FEATURE_COLUMNS]


def build_labels(tx: pd.DataFrame, index: pd.Index, cutoff: pd.Timestamp,
                 horizon_days: int = CHURN_HORIZON_DAYS) -> pd.Series:
    cutoff = pd.Timestamp(cutoff)
    fut = tx[(tx["order_date"] >= cutoff) &
             (tx["order_date"] < cutoff + pd.Timedelta(days=horizon_days))]
    active_future = set(fut["customer_id"])
    return pd.Series([0 if c in active_future else 1 for c in index],
                     index=index, name="churned")


def make_training_set(tx: pd.DataFrame, customers: pd.DataFrame,
                      horizon_days: int = CHURN_HORIZON_DAYS,
                      active_window: int = ACTIVE_WINDOW_DAYS):
    """Return (X, y, cutoff). Cutoff = last data date - horizon."""
    last_date = tx["order_date"].max()
    cutoff = last_date - pd.Timedelta(days=horizon_days - 1)
    X = build_features(tx, customers, cutoff)
    X = X[X["recency"] <= active_window]
    y = build_labels(tx, X.index, cutoff, horizon_days)
    return X, y, cutoff


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "base_rate": float(y_test.mean()),
        "n_test": int(len(y_test)),
    }


def train_churn_model(tx: pd.DataFrame, customers: pd.DataFrame, seed: int = SEED,
                      params: Optional[dict] = None) -> Dict:
    X, y, cutoff = make_training_set(tx, customers)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    p = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.06,
         "subsample": 0.85, "random_state": seed}
    if params:
        p.update(params)
    model = GradientBoostingClassifier(**p).fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    return {
        "model": model, "features": FEATURE_COLUMNS, "metrics": metrics,
        "importance": importance, "cutoff": str(cutoff.date()),
        "horizon_days": CHURN_HORIZON_DAYS, "params": p,
    }


def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else ("Medium" if p >= 0.3 else "Low")


def score_customers(bundle: Dict, tx: pd.DataFrame, customers: pd.DataFrame,
                    as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Churn probability for every customer with purchase history as of `as_of`."""
    as_of = pd.Timestamp(as_of) if as_of is not None else tx["order_date"].max() + pd.Timedelta(days=1)
    X = build_features(tx, customers, as_of)
    proba = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    out = pd.DataFrame({
        "customer_id": X.index,
        "churn_probability": np.round(proba, 4),
        "recency": X["recency"].to_numpy(),
        "frequency": X["frequency"].to_numpy(),
        "monetary": X["monetary"].round(2).to_numpy(),
        "active": (X["recency"] <= ACTIVE_WINDOW_DAYS).to_numpy(),
    })
    out["risk_band"] = out["churn_probability"].map(risk_band)
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def save_churn_model(bundle: Dict, path: Path | None = None) -> Path:
    path = Path(path) if path else MODELS_DIR / "churn_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_churn_model(path: Path | None = None) -> Dict:
    return joblib.load(Path(path) if path else MODELS_DIR / "churn_model.joblib")
