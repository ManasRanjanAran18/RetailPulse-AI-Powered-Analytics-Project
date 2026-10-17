"""Automated ETL: load raw CSVs -> validate/clean -> build analytics tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, ensure_dirs


def load_raw(raw_dir: Path = RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, customers, products, inventory) raw tables."""
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "transactions.csv")
    customers = pd.read_csv(raw_dir / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(raw_dir / "products.csv")
    inventory = pd.read_csv(raw_dir / "inventory.csv")
    return tx, customers, products, inventory


def clean_transactions(tx: pd.DataFrame, products: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Validate and clean the transactions table; returns (clean_df, report)."""
    report: Dict[str, int] = {"rows_raw": int(len(tx))}
    df = tx.copy()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["order_date", "customer_id", "product_id"])
    report["rows_dropped_missing_keys"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    missing_price = df["unit_price"].isna()
    report["missing_prices_imputed"] = int(missing_price.sum())
    list_price = products.set_index("product_id")["unit_price"]
    df.loc[missing_price, "unit_price"] = df.loc[missing_price, "product_id"].map(list_price)

    before = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    report["returns_or_invalid_removed"] = int(before - len(df))

    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report


def daily_sales(tx: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue / units / order-lines with gaps filled with zeros."""
    g = tx.groupby("order_date").agg(
        revenue=("revenue", "sum"), units=("quantity", "sum"), orders=("order_id", "nunique"))
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full, fill_value=0)
    g.index.name = "date"
    return g.reset_index()


def product_daily_units(tx: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = dates, columns = product_id, values = units sold."""
    pt = tx.pivot_table(index="order_date", columns="product_id", values="quantity",
                        aggfunc="sum", fill_value=0)
    full = pd.date_range(pt.index.min(), pt.index.max(), freq="D")
    pt = pt.reindex(full, fill_value=0)
    pt.index.name = "date"
    return pt.reset_index()


def run_etl(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> Dict[str, int]:
    """Full ETL run. Writes processed CSVs + a data-quality report."""
    ensure_dirs()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx, customers, products, inventory = load_raw(raw_dir)
    clean, report = clean_transactions(tx, products)

    clean.to_csv(out_dir / "transactions_clean.csv", index=False)
    daily_sales(clean).to_csv(out_dir / "daily_sales.csv", index=False)
    product_daily_units(clean).to_csv(out_dir / "product_daily_units.csv", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    products.to_csv(out_dir / "products.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)

    report["date_min"] = str(clean["order_date"].min().date())
    report["date_max"] = str(clean["order_date"].max().date())
    report["customers"] = int(clean["customer_id"].nunique())
    report["products"] = int(clean["product_id"].nunique())
    with open(out_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def load_processed(out_dir: Path = PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    """Load the processed tables with correct dtypes."""
    out_dir = Path(out_dir)
    return {
        "transactions": pd.read_csv(out_dir / "transactions_clean.csv", parse_dates=["order_date"]),
        "daily": pd.read_csv(out_dir / "daily_sales.csv", parse_dates=["date"]),
        "product_daily": pd.read_csv(out_dir / "product_daily_units.csv", parse_dates=["date"]),
        "customers": pd.read_csv(out_dir / "customers.csv", parse_dates=["signup_date"]),
        "products": pd.read_csv(out_dir / "products.csv"),
        "inventory": pd.read_csv(out_dir / "inventory.csv"),
    }
