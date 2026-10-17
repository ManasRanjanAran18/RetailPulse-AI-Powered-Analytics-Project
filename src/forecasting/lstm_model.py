"""LSTM forecaster (optional dependency: TensorFlow / Keras)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import SEED


def _windows(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X)[..., None], np.array(y)


def forecast_lstm(train: pd.Series, horizon: int, window: int = 28,
                  epochs: int = 40, seed: int = SEED, verbose: int = 0) -> pd.DataFrame:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf  # imported lazily: heavy, optional

    tf.keras.utils.set_random_seed(seed)
    values = train.to_numpy(dtype="float32")
    mu, sd = float(values.mean()), float(values.std() or 1.0)
    scaled = (values - mu) / sd

    X, y = _windows(scaled, window)
    split = int(len(X) * 0.9)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
              epochs=epochs, batch_size=32, verbose=verbose,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

    resid = y[split:] - model.predict(X[split:], verbose=0).ravel()
    resid_std = float(np.std(resid) * sd)

    hist = list(scaled[-window:])
    preds = []
    for _ in range(horizon):
        x = np.array(hist[-window:], dtype="float32").reshape(1, window, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        hist.append(p)
    yhat = np.maximum(np.array(preds) * sd + mu, 0)
    dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    band = 1.645 * resid_std
    return pd.DataFrame({"ds": dates, "yhat": yhat,
                         "yhat_lower": np.maximum(yhat - band, 0),
                         "yhat_upper": yhat + band})
