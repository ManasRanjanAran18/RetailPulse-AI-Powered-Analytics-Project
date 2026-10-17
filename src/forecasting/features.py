"""Feature engineering for the tree-based (GBM) demand forecaster.

Lag/rolling features are expressed *relative to the trailing 28-day mean*
(the "level"). The model therefore learns the shape of demand (weekday effects,
seasonality, momentum) rather than its absolute size, which keeps recursive
multi-step forecasts stable when demand trends upward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame({
        "dow": index.dayofweek.to_numpy(),
        "is_weekend": (index.dayofweek.to_numpy() >= 5).astype(int),
        "month": index.month.to_numpy(),
        "holiday_season": np.isin(index.month.to_numpy(), [11, 12]).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


def build_feature_frame(series: pd.Series):
    """Return (features, level). Row t only uses information up to t-1."""
    shifted = series.shift(1)
    level = shifted.rolling(28).mean()
    df = pd.DataFrame(index=series.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = series.shift(lag) / level
    df["roll_mean_7"] = shifted.rolling(7).mean() / level
    df["roll_std_7"] = shifted.rolling(7).std() / level
    df["level"] = level
    return df.join(calendar_features(series.index)), level
