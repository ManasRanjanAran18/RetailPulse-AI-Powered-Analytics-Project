#!/usr/bin/env python
"""CLI: run the whole RetailPulse pipeline.

Examples
--------
    python scripts/run_pipeline.py                  # full run (all models)
    python scripts/run_pipeline.py --light          # GBM only (fast, no TF/Prophet)
    python scripts/run_pipeline.py --regenerate     # rebuild synthetic data first
    python scripts/run_pipeline.py --mlflow         # also log runs to MLflow
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic raw data")
    ap.add_argument("--skip-prophet", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--light", action="store_true", help="skip Prophet and LSTM")
    ap.add_argument("--mlflow", action="store_true", help="log runs to MLflow (if installed)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run_pipeline(regenerate=args.regenerate,
                 skip_prophet=args.skip_prophet or args.light,
                 skip_lstm=args.skip_lstm or args.light,
                 use_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
