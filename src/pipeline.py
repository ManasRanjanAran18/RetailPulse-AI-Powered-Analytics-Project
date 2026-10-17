"""End-to-end RetailPulse pipeline:

    generate data (if missing) -> ETL -> segmentation -> churn -> forecasting
    (GBM + Prophet + LSTM hybrid ensemble) -> inventory optimisation -> drift

All artifacts are written to data/processed, models/ and reports/ so the
Streamlit dashboard can serve them without retraining.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from src import tracking
from src.churn.model import (build_features, save_churn_model, score_customers,
                             train_churn_model)
from src.config import (ACTIVE_WINDOW_DAYS, FORECAST_HORIZON, HOLDOUT_DAYS,
                        PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs)
from src.forecasting.ensemble import (allocate_to_products, available_models,
                                      run_forecasting)
from src.ingestion.etl import load_processed, run_etl
from src.ingestion.synthetic import generate_dataset
from src.inventory.optimizer import optimize_inventory
from src.monitoring.drift import drift_report, error_drift
from src.segmentation.rfm import (build_rfm, evaluate_k, fit_segmentation,
                                  predict_segments, save_segmentation,
                                  segment_summary)

log = logging.getLogger("retailpulse.pipeline")


def _step(name: str):
    log.info("▶ %s", name)
    return time.time()


def run_pipeline(regenerate: bool = False, skip_prophet: bool = False,
                 skip_lstm: bool = False, use_mlflow: bool = False) -> Dict:
    ensure_dirs()
    t_start = time.time()
    summary: Dict = {}
    tracking_on = tracking.setup(use_mlflow)

    # ---------------------------------------------------------------- data
    t = _step("Data generation + ETL")
    if regenerate or not (RAW_DIR / "transactions.csv").exists():
        generate_dataset()
    summary["etl"] = run_etl()
    d = load_processed()
    tx, customers, inventory = d["transactions"], d["customers"], d["inventory"]
    log.info("  ETL done in %.1fs: %s", time.time() - t, summary["etl"])

    # -------------------------------------------------------- segmentation
    _step("Customer segmentation")
    rfm = build_rfm(tx)
    evaluate_k(rfm).to_csv(PROCESSED_DIR / "segmentation_k_search.csv", index=False)
    seg_bundle = fit_segmentation(rfm, k=4)
    save_segmentation(seg_bundle)
    segmented = predict_segments(seg_bundle, rfm)
    segmented.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)
    segment_summary(segmented).to_csv(PROCESSED_DIR / "segment_summary.csv", index=False)
    summary["segmentation"] = {"k": 4, "silhouette": round(seg_bundle["silhouette"], 4)}
    with tracking.run("segmentation_kmeans", tracking_on, {"k": 4}) as tr:
        tr.metrics({"silhouette": seg_bundle["silhouette"]})

    # --------------------------------------------------------------- churn
    _step("Churn prediction")
    churn_bundle = train_churn_model(tx, customers)
    save_churn_model(churn_bundle)
    scores = score_customers(churn_bundle, tx, customers)
    scores.to_csv(PROCESSED_DIR / "churn_scores.csv", index=False)
    churn_bundle["importance"].rename("importance").to_csv(
        PROCESSED_DIR / "churn_feature_importance.csv", index_label="feature")
    summary["churn"] = {**churn_bundle["metrics"], "cutoff": churn_bundle["cutoff"]}
    with tracking.run("churn_gbm", tracking_on, churn_bundle["params"]) as tr:
        tr.metrics(churn_bundle["metrics"])

    # ---------------------------------------------------------- forecasting
    _step("Demand forecasting (hybrid ensemble)")
    series = d["daily"].set_index("date")["units"]
    models = available_models(skip_prophet=skip_prophet, skip_lstm=skip_lstm)
    log.info("  models: %s", list(models))
    fc = run_forecasting(series, models, FORECAST_HORIZON, HOLDOUT_DAYS)
    fc["backtest"].to_csv(PROCESSED_DIR / "forecast_backtest.csv", index=False)
    fc["future"].to_csv(PROCESSED_DIR / "forecast_future.csv", index=False)
    fc["metrics"].rename_axis("model").to_csv(PROCESSED_DIR / "forecast_metrics.csv")
    summary["forecast"] = {
        "models": list(models), "weights": fc["weights"].to_dict(),
        "metrics": fc["metrics"].round(4).to_dict(orient="index"),
        "timings_sec": fc["timings"],
    }
    for name, row in fc["metrics"].iterrows():
        with tracking.run(f"forecast_{name.split()[0]}", tracking_on,
                          {"model": name, "holdout_days": HOLDOUT_DAYS}) as tr:
            tr.metrics(row.to_dict())

    # ------------------------------------------------------------ inventory
    _step("Inventory optimisation")
    demand = allocate_to_products(fc["future"], d["product_daily"])
    demand.to_csv(PROCESSED_DIR / "product_demand_forecast.csv", index=False)
    inv = optimize_inventory(inventory, demand)
    inv.to_csv(PROCESSED_DIR / "inventory_plan.csv", index=False)
    summary["inventory"] = inv["status"].value_counts().to_dict()

    # -------------------------------------------------------------- drift
    _step("Drift detection")
    end = tx["order_date"].max()
    ref = tx[tx["order_date"] < tx["order_date"].min() + pd.Timedelta(days=365)]
    cur = tx[tx["order_date"] > end - pd.Timedelta(days=90)]
    drift_tx = drift_report(ref, cur, ["quantity", "unit_price", "revenue"])
    drift_tx.to_csv(PROCESSED_DIR / "drift_transactions.csv", index=False)

    f_ref = build_features(tx, customers, end - pd.Timedelta(days=180))
    f_cur = build_features(tx, customers, end + pd.Timedelta(days=1))
    f_ref, f_cur = (f[f["recency"] <= ACTIVE_WINDOW_DAYS] for f in (f_ref, f_cur))
    drift_feat = drift_report(f_ref, f_cur, ["recency", "frequency", "monetary",
                                             "avg_order_value", "orders_30d", "orders_90d"])
    drift_feat.to_csv(PROCESSED_DIR / "drift_customer_features.csv", index=False)

    bt = fc["backtest"].set_index("date")
    err = error_drift(bt["ensemble"] - bt["actual"])
    err.reset_index().to_csv(PROCESSED_DIR / "drift_forecast_error.csv", index=False)
    summary["drift"] = {
        "transactions_alerts": int((drift_tx["status"] == "ALERT").sum()),
        "feature_alerts": int((drift_feat["status"] == "ALERT").sum()),
        "forecast_error_breaches": int(err["breach"].sum()),
    }

    # ------------------------------------------------------------- summary
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["runtime_sec"] = round(time.time() - t_start, 1)
    with open(REPORTS_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("✔ Pipeline finished in %.1fs", summary["runtime_sec"])
    return summary
