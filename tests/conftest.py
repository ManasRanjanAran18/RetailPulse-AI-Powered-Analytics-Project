import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.etl import clean_transactions  # noqa: E402
from src.ingestion.synthetic import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    """A small, fast synthetic dataset (does not touch the repo's data/ folder)."""
    out = tmp_path_factory.mktemp("raw")
    raw = generate_dataset(seed=7, n_customers=300, n_products=12,
                           start="2024-01-01", end="2025-06-30", out_dir=out)
    clean, report = clean_transactions(raw["transactions"], raw["products"])
    return {**raw, "clean": clean, "report": report, "raw_dir": out}
