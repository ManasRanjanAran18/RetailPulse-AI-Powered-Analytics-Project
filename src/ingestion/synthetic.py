"""Synthetic retail data generator.

Produces four raw CSVs (transactions, customers, products, inventory) with
realistic structure: weekly + yearly seasonality, a Nov/Dec peak, a growth
trend, heterogeneous customer purchase rates, and customers who gradually
disengage and then churn. A small amount of deliberate dirt (duplicates,
returns, missing prices) is injected so the cleaning step has real work to do.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LOYALTY_TIERS, RAW_DIR, REGIONS, SEED, ensure_dirs

CATEGORIES = {
    # category: (min price, max price, mean extra quantity per line)
    "Grocery": (2, 15, 2.5),
    "Electronics": (40, 400, 0.2),
    "Apparel": (10, 90, 0.5),
    "Home & Kitchen": (8, 120, 0.6),
    "Beauty": (5, 60, 0.8),
}
STORES = ["Online", "Store A", "Store B", "Store C"]
SUPPLIERS = ["Acme Supply", "Globex Traders", "Initech Wholesale", "Umbrella Logistics"]


def generate_dataset(
    seed: int = SEED,
    n_customers: int = 1500,
    n_products: int = 40,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir) if out_dir else RAW_DIR
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    # ------------------------------------------------------------ products
    cats = list(CATEGORIES)
    prod_cat = [cats[i % len(cats)] for i in range(n_products)]
    prices = np.array([round(rng.uniform(CATEGORIES[c][0], CATEGORIES[c][1]), 2)
                       for c in prod_cat])
    products = pd.DataFrame({
        "product_id": [f"P{i + 1:03d}" for i in range(n_products)],
        "product_name": [f"{c} Item {i + 1}" for i, c in enumerate(prod_cat)],
        "category": prod_cat,
        "unit_price": prices,
        "unit_cost": np.round(prices * rng.uniform(0.5, 0.75, n_products), 2),
    })
    popularity = rng.pareto(2.0, n_products) + 0.5
    base_w = popularity / popularity.sum()
    holiday_w = base_w * np.where(np.array(prod_cat) == "Electronics", 1.6, 1.0)
    holiday_w = holiday_w / holiday_w.sum()

    # ---------------------------------------------------------- seasonality
    t = np.arange(n_days)
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    weekly = np.where(dow >= 5, 1.25, np.where(dow == 4, 1.1, 1.0))
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 80) / 365.25)
    holiday = np.where(np.isin(month, [11, 12]), 1.35, 1.0)
    trend = 1 + 0.0003 * t
    season = weekly * yearly * holiday * trend

    # ------------------------------------------------------------ customers
    customer_ids = [f"C{i + 1:05d}" for i in range(n_customers)]
    pre_existing = rng.random(n_customers) < 0.6
    signup_idx = np.where(pre_existing, 0, rng.integers(0, n_days - 30, n_customers))
    signup_dates = np.where(
        pre_existing,
        dates[0] - pd.to_timedelta(rng.integers(30, 700, n_customers), unit="D"),
        dates[signup_idx],
    )
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": pd.to_datetime(signup_dates).strftime("%Y-%m-%d"),
        "region": rng.choice(REGIONS, n_customers),
        "loyalty_tier": rng.choice(LOYALTY_TIERS, n_customers, p=[0.55, 0.30, 0.15]),
        "age": np.clip(rng.normal(37, 11, n_customers).round(), 18, 80).astype(int),
    })

    rate = rng.lognormal(mean=np.log(0.042), sigma=0.7, size=n_customers)
    tier_boost = customers["loyalty_tier"].map({"Bronze": 1.0, "Silver": 1.25, "Gold": 1.6}).to_numpy()
    rate = rate * tier_boost
    churner = rng.random(n_customers) < 0.45
    churn_idx = np.maximum((rng.beta(2.5, 1.5, n_customers) * n_days).astype(int), signup_idx + 30)

    rows_date, rows_cust = [], []
    for i in range(n_customers):
        mult = np.zeros(n_days)
        mult[signup_idx[i]:] = 1.0
        if churner[i]:
            c = churn_idx[i]
            if c < n_days:
                mult[c:] = 0.0
            ramp_start = max(c - 60, signup_idx[i])
            if ramp_start < min(c, n_days):
                ramp = np.linspace(1.0, 0.3, c - ramp_start)
                end_r = min(c, n_days)
                mult[ramp_start:end_r] = ramp[: end_r - ramp_start]
        counts = rng.poisson(rate[i] * season * mult)
        nz = np.nonzero(counts)[0]
        if nz.size:
            rows_date.append(np.repeat(nz, counts[nz]))
            rows_cust.append(np.full(counts[nz].sum(), i))

    day_idx = np.concatenate(rows_date)
    cust_idx = np.concatenate(rows_cust)
    order = np.argsort(day_idx, kind="stable")
    day_idx, cust_idx = day_idx[order], cust_idx[order]
    n_rows = len(day_idx)

    # -------------------------------------------------------------- products
    is_holiday = np.isin(month[day_idx], [11, 12])
    prod_idx = np.where(
        is_holiday,
        rng.choice(n_products, n_rows, p=holiday_w),
        rng.choice(n_products, n_rows, p=base_w),
    )
    cat_arr = np.array(prod_cat)[prod_idx]
    extra_lambda = np.array([CATEGORIES[c][2] for c in cat_arr])
    quantity = 1 + rng.poisson(extra_lambda)
    unit_price = np.round(prices[prod_idx] * rng.uniform(0.9, 1.0, n_rows), 2)

    tx = pd.DataFrame({
        "order_id": [f"O{i + 1:07d}" for i in range(n_rows)],
        "order_date": dates[day_idx].strftime("%Y-%m-%d"),
        "customer_id": np.array(customer_ids)[cust_idx],
        "product_id": products["product_id"].to_numpy()[prod_idx],
        "category": cat_arr,
        "quantity": quantity,
        "unit_price": unit_price,
        "store": rng.choice(STORES, n_rows, p=[0.4, 0.25, 0.2, 0.15]),
    })

    # -------------------------------------------------- inject realistic dirt
    n_dup = max(1, int(0.003 * n_rows))
    n_ret = max(1, int(0.004 * n_rows))
    n_nan = max(1, int(0.002 * n_rows))
    ret_rows = rng.choice(n_rows, n_ret, replace=False)
    tx.loc[ret_rows, "quantity"] = -tx.loc[ret_rows, "quantity"]
    nan_rows = rng.choice(n_rows, n_nan, replace=False)
    tx.loc[nan_rows, "unit_price"] = np.nan
    dups = tx.sample(n_dup, random_state=seed)
    tx = pd.concat([tx, dups], ignore_index=True).sort_values(
        ["order_date", "order_id"], kind="stable").reset_index(drop=True)

    # -------------------------------------------------------------- inventory
    clean = tx[tx["quantity"] > 0].drop_duplicates()
    last_date = pd.to_datetime(clean["order_date"]).max()
    recent = clean[pd.to_datetime(clean["order_date"]) > last_date - pd.Timedelta(days=90)]
    daily_units = recent.groupby("product_id")["quantity"].sum().reindex(
        products["product_id"], fill_value=0) / 90.0
    lead = rng.integers(3, 21, n_products)
    cover_days = rng.uniform(2, 75, n_products)
    inventory = pd.DataFrame({
        "product_id": products["product_id"],
        "current_stock": np.maximum((daily_units.to_numpy() * cover_days).round().astype(int), 1),
        "lead_time_days": lead,
        "unit_cost": products["unit_cost"],
        "holding_cost_rate": 0.20,
        "order_cost": np.round(rng.uniform(30, 80, n_products), 2),
        "supplier": rng.choice(SUPPLIERS, n_products),
    })

    tx.to_csv(out_dir / "transactions.csv", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    products.to_csv(out_dir / "products.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    return {"transactions": tx, "customers": customers,
            "products": products, "inventory": inventory}
