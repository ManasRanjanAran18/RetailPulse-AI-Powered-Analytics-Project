"""Facebook Prophet forecaster (optional dependency)."""
from __future__ import annotations

import logging

import pandas as pd


def forecast_prophet(train: pd.Series, horizon: int) -> pd.DataFrame:
    from prophet import Prophet  # imported lazily: heavy, optional

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    df = pd.DataFrame({"ds": pd.DatetimeIndex(train.index), "y": train.to_numpy(dtype=float)})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative",
                interval_width=0.90)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future).tail(horizon)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        out[c] = out[c].clip(lower=0)
    return out
