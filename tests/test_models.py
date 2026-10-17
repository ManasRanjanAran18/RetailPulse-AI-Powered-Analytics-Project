import numpy as np
import pandas as pd

from src.churn.model import FEATURE_COLUMNS, build_features, score_customers, train_churn_model
from src.forecasting.ensemble import (allocate_to_products, inverse_error_weights,
                                      regression_metrics, run_forecasting)
from src.forecasting.gbm import forecast_gbm
from src.ingestion.etl import daily_sales, product_daily_units
from src.inventory.optimizer import optimize_inventory
from src.monitoring.drift import drift_report, psi
from src.segmentation.rfm import (build_rfm, fit_segmentation, predict_segments,
                                  segment_summary)


def test_segmentation_names_and_coverage(small_data):
    rfm = build_rfm(small_data["clean"])
    bundle = fit_segmentation(rfm, k=4)
    seg = predict_segments(bundle, rfm)
    assert set(seg["segment"]) == {"Champions", "Loyal Customers", "At Risk", "Hibernating"}
    assert len(seg) == len(rfm)
    assert segment_summary(seg)["customers"].sum() == len(rfm)


def test_churn_features_have_no_future_leakage(small_data):
    tx, cust = small_data["clean"], small_data["customers"]
    cutoff = tx["order_date"].max() - pd.Timedelta(days=90)
    f = build_features(tx, cust, cutoff)
    assert list(f.columns) == FEATURE_COLUMNS
    assert (f["recency"] >= 1).all()          # last purchase strictly before cutoff
    assert f.notna().all().all()


def test_churn_model_beats_chance_and_scores_valid(small_data):
    tx, cust = small_data["clean"], small_data["customers"]
    bundle = train_churn_model(tx, cust)
    assert bundle["metrics"]["roc_auc"] > 0.6
    scores = score_customers(bundle, tx, cust)
    assert scores["churn_probability"].between(0, 1).all()
    assert set(scores["risk_band"]) <= {"Low", "Medium", "High"}


def test_gbm_forecast_shape_and_sanity(small_data):
    s = daily_sales(small_data["clean"]).set_index("date")["units"]
    fc = forecast_gbm(s.iloc[:-30], 30)
    assert len(fc) == 30 and (fc["yhat"] >= 0).all()
    assert (fc["yhat_lower"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["yhat_upper"]).all()
    m = regression_metrics(s.iloc[-30:], fc["yhat"])
    assert m["wape"] < 0.6


def test_ensemble_weights_and_allocation(small_data):
    clean = small_data["clean"]
    s = daily_sales(clean).set_index("date")["units"]
    res = run_forecasting(s, {"gbm": forecast_gbm}, horizon=14, holdout=30)
    assert len(res["future"]) == 14
    assert abs(res["weights"].sum() - 1.0) < 1e-6
    w = inverse_error_weights(pd.DataFrame({"mae": [10.0, 20.0]}, index=["a", "b"]))
    assert w["a"] > w["b"]
    demand = allocate_to_products(res["future"], product_daily_units(clean))
    assert abs(demand["forecast_units_horizon"].sum() - res["future"]["ensemble"].sum()) < 1e-6


def test_inventory_policy(small_data):
    clean = small_data["clean"]
    s = daily_sales(clean).set_index("date")["units"]
    res = run_forecasting(s, {"gbm": forecast_gbm}, horizon=14, holdout=30)
    demand = allocate_to_products(res["future"], product_daily_units(clean))
    plan = optimize_inventory(small_data["inventory"], demand)
    assert (plan["reorder_point"] >= plan["safety_stock"]).all()
    assert (plan["eoq"] > 0).all()
    needs = plan["status"].isin(["Critical", "Reorder"])
    assert (plan.loc[needs, "recommended_order_qty"] > 0).all()
    assert (plan.loc[~needs, "recommended_order_qty"] == 0).all()
    higher = optimize_inventory(small_data["inventory"], demand, service_level=0.999)
    assert (higher["safety_stock"] >= plan["safety_stock"]).all()


def test_drift_detects_shift_but_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25
    ref, cur = pd.DataFrame({"x": a}), pd.DataFrame({"x": rng.normal(1.5, 1, 5000)})
    assert drift_report(ref, cur, ["x"]).loc[0, "status"] == "ALERT"
