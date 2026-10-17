"""Gradient-boosting demand forecaster (level-normalised lags, recursive multi-step)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import SEED
from src.forecasting.features import build_feature_frame


class GBMForecaster:
    """Predicts demand / trailing-28-day-mean, then rescales by the level."""

    def __init__(self, seed: int = SEED, **params):
        p = dict(max_iter=300, learning_rate=0.05, max_depth=4,
                 min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
        p.update(params)
        self.model = HistGradientBoostingRegressor(**p)
        self.series: pd.Series | None = None
        self.resid_std_: float = 0.0

    def fit(self, series: pd.Series) -> "GBMForecaster":
        series = series.astype(float).copy()
        series.index = pd.DatetimeIndex(series.index)
        self.series = series
        X, level = build_feature_frame(series)
        data = X.drop(columns="level").assign(y=series / level).dropna()
        self.columns_ = [c for c in data.columns if c != "y"]
        self.model.fit(data[self.columns_], data["y"])
        fitted = self.model.predict(data[self.columns_]) * level.loc[data.index]
        self.resid_std_ = float((series.loc[data.index] - fitted).std(ddof=0))
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        history = self.series.copy()
        dates, preds = [], []
        for _ in range(horizon):
            next_date = history.index[-1] + pd.Timedelta(days=1)
            history.loc[next_date] = np.nan
            X, level = build_feature_frame(history)
            ratio = float(self.model.predict(X.iloc[[-1]][self.columns_])[0])
            yhat = max(ratio * float(level.iloc[-1]), 0.0)
            history.loc[next_date] = yhat
            dates.append(next_date)
            preds.append(yhat)
        yhat = np.array(preds)
        band = 1.645 * self.resid_std_
        return pd.DataFrame({"ds": dates, "yhat": yhat,
                             "yhat_lower": np.maximum(yhat - band, 0),
                             "yhat_upper": yhat + band})


def forecast_gbm(train: pd.Series, horizon: int) -> pd.DataFrame:
    return GBMForecaster().fit(train).forecast(horizon)
