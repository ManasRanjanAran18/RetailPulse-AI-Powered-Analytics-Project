#!/usr/bin/env python
"""CLI: (re)generate the synthetic raw retail dataset into data/raw/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.synthetic import generate_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--products", type=int, default=40)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    d = generate_dataset(seed=a.seed, n_customers=a.customers, n_products=a.products,
                         start=a.start, end=a.end)
    for name, df in d.items():
        print(f"{name:<13} {df.shape}")


if __name__ == "__main__":
    main()
