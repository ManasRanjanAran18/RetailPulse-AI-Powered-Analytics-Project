"""Central configuration: paths and project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42

# Forecasting
FORECAST_HORIZON = 30      # days to forecast ahead
HOLDOUT_DAYS = 60          # days held out for backtesting

# Churn
CHURN_HORIZON_DAYS = 90    # "churned" = no purchase in the next N days
ACTIVE_WINDOW_DAYS = 180   # a customer is "active" if they bought in the last N days

# Inventory
SERVICE_LEVEL = 0.95

# Categorical domains (kept fixed so model feature columns are stable)
REGIONS = ["North", "South", "East", "West"]
LOYALTY_TIERS = ["Bronze", "Silver", "Gold"]


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
