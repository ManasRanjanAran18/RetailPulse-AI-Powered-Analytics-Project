"""Inventory optimisation: safety stock, reorder point, EOQ, stock status."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import SERVICE_LEVEL


def optimize_inventory(inventory: pd.DataFrame, demand: pd.DataFrame,
                       service_level: float = SERVICE_LEVEL,
                       overstock_days: int = 60) -> pd.DataFrame:
    """Combine current stock with forecast demand to produce reorder policy.

    demand columns required: product_id, forecast_daily_mean, hist_daily_std
    inventory columns required: product_id, current_stock, lead_time_days,
        unit_cost, holding_cost_rate, order_cost
    """
    df = inventory.merge(demand, on="product_id", how="left")
    z = float(norm.ppf(service_level))
    d = df["forecast_daily_mean"].clip(lower=0.01)
    sigma = df["hist_daily_std"].fillna(0)
    lead = df["lead_time_days"]

    df["safety_stock"] = np.ceil(z * sigma * np.sqrt(lead)).astype(int)
    df["reorder_point"] = np.ceil(d * lead + df["safety_stock"]).astype(int)
    holding = (df["unit_cost"] * df["holding_cost_rate"]).clip(lower=1e-6)
    df["eoq"] = np.ceil(np.sqrt(2 * d * 365 * df["order_cost"] / holding)).astype(int)
    df["days_of_cover"] = (df["current_stock"] / d).round(1)

    status = np.select(
        [df["current_stock"] < df["safety_stock"],
         df["current_stock"] <= df["reorder_point"],
         df["days_of_cover"] > overstock_days],
        ["Critical", "Reorder", "Overstock"],
        default="Healthy",
    )
    df["status"] = status
    order_up_to = df["reorder_point"] + df["eoq"]
    needs_order = df["status"].isin(["Critical", "Reorder"])
    df["recommended_order_qty"] = np.where(
        needs_order, np.ceil(order_up_to - df["current_stock"]).clip(lower=0), 0).astype(int)
    df["excess_units"] = np.where(
        df["status"] == "Overstock", (df["current_stock"] - order_up_to).clip(lower=0), 0).astype(int)
    df["excess_value"] = (df["excess_units"] * df["unit_cost"]).round(2)
    df["order_cost_estimate"] = (df["recommended_order_qty"] * df["unit_cost"]).round(2)
    return df
