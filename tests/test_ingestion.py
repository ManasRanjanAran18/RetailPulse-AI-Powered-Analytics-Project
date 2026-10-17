import pandas as pd

from src.ingestion.etl import daily_sales, product_daily_units


def test_cleaning_removes_dirt(small_data):
    clean, report = small_data["clean"], small_data["report"]
    assert report["duplicates_removed"] > 0
    assert report["returns_or_invalid_removed"] > 0
    assert report["missing_prices_imputed"] > 0
    assert (clean["quantity"] > 0).all()
    assert clean["unit_price"].notna().all()
    assert not clean.duplicated().any()
    assert (clean["revenue"] > 0).all()


def test_daily_sales_is_continuous_and_consistent(small_data):
    clean = small_data["clean"]
    d = daily_sales(clean)
    assert d["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert abs(d["revenue"].sum() - clean["revenue"].sum()) < 1e-6
    assert d["units"].sum() == clean["quantity"].sum()


def test_product_daily_units_totals(small_data):
    clean = small_data["clean"]
    wide = product_daily_units(clean).set_index("date")
    assert wide.to_numpy().sum() == clean["quantity"].sum()
