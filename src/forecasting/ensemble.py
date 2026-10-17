"""Backtesting + hybrid (weighted) ensemble of GBM, Prophet and LSTM."""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON, HOLDOUT_DAYS
from src.forecasting.gbm import forecast_gbm

log = logging.getLogger(__name__)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-9)),
        "bias": float(np.mean(err)),
    }


def available_models(skip_prophet: bool = False, skip_lstm: bool = False
                     ) -> Dict[str, Callable[[pd.Series, int], pd.DataFrame]]:
    """Return the forecasters that can run in the current environment."""
    models: Dict[str, Callable] = {"gbm": forecast_gbm}
    if not skip_prophet:
        try:
            import prophet  # noqa: F401
            from src.forecasting.prophet_model import forecast_prophet
            models["prophet"] = forecast_prophet
        except Exception as e:  # noqa: BLE001
            log.warning("Prophet unavailable, skipping: %s", e)
    if not skip_lstm:
        try:
            import tensorflow  # noqa: F401
            from src.forecasting.lstm_model import forecast_lstm
            models["lstm"] = forecast_lstm
        except Exception as e:  # noqa: BLE001
            log.warning("TensorFlow unavailable, skipping LSTM: %s", e)
    return models


def inverse_error_weights(metrics: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1/MAE (better models get more say)."""
    inv = 1.0 / metrics["mae"].clip(lower=1e-9)
    return (inv / inv.sum()).round(4)


def blend(frames: Dict[str, pd.DataFrame], weights: pd.Series, col: str = "yhat") -> np.ndarray:
    total = sum(float(weights[n]) for n in frames)
    return sum(frames[n][col].to_numpy() * float(weights[n]) for n in frames) / total


def run_forecasting(series: pd.Series, models: Optional[Dict[str, Callable]] = None,
                    horizon: int = FORECAST_HORIZON, holdout: int = HOLDOUT_DAYS) -> Dict:
    """Backtest every model on the last `holdout` days, derive ensemble weights,
    then refit on all data and forecast `horizon` days ahead."""
    models = models or available_models()
    series = series.astype(float)
    series.index = pd.DatetimeIndex(series.index)
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    # ---- 1) backtest
    back = pd.DataFrame({"actual": test})
    back_frames: Dict[str, pd.DataFrame] = {}
    timings: Dict[str, float] = {}
    for name, fn in models.items():
        t0 = time.time()
        fc = fn(train, holdout)
        timings[name] = round(time.time() - t0, 1)
        back_frames[name] = fc
        back[name] = fc["yhat"].to_numpy()
    rows = {n: regression_metrics(test.to_numpy(), back[n].to_numpy()) for n in models}
    metrics = pd.DataFrame(rows).T
    weights = inverse_error_weights(metrics)
    back["ensemble"] = blend(back_frames, weights)
    metrics.loc["ensemble (hybrid)"] = regression_metrics(test.to_numpy(), back["ensemble"].to_numpy())
    back.index.name = "date"

    # ---- 2) refit on all data, forecast ahead
    fut_frames: Dict[str, pd.DataFrame] = {n: fn(series, horizon) for n, fn in models.items()}
    future = pd.DataFrame({"date": fut_frames[next(iter(fut_frames))]["ds"].to_numpy()})
    for n, fr in fut_frames.items():
        future[n] = fr["yhat"].to_numpy()
    future["ensemble"] = blend(fut_frames, weights)
    band = 1.645 * metrics.loc["ensemble (hybrid)", "rmse"]
    future["ensemble_lower"] = np.maximum(future["ensemble"] - band, 0)
    future["ensemble_upper"] = future["ensemble"] + band

    return {"backtest": back.reset_index(), "future": future, "metrics": metrics,
            "weights": weights, "timings": timings}


def allocate_to_products(future: pd.DataFrame, product_units: pd.DataFrame,
                         lookback: int = 90) -> pd.DataFrame:
    """Top-down allocation of the total-demand forecast to products using each
    product's share of units over the trailing `lookback` days."""
    wide = product_units.set_index("date")
    recent = wide.iloc[-lookback:]
    share = recent.sum() / recent.sum().sum()
    total_units = float(future["ensemble"].sum())
    n_days = len(future)
    out = pd.DataFrame({
        "product_id": share.index,
        "share": share.to_numpy(),
        "forecast_units_horizon": share.to_numpy() * total_units,
    })
    out["forecast_daily_mean"] = out["forecast_units_horizon"] / n_days
    out["hist_daily_mean"] = recent.mean().reindex(out["product_id"]).to_numpy()
    out["hist_daily_std"] = recent.std(ddof=0).reindex(out["product_id"]).to_numpy()
    return out
