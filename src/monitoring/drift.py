"""Data / model drift detection: Population Stability Index + KS test."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using reference-quantile bins.
    <0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status_from_psi(value: float) -> str:
    if value != value:
        return "n/a"
    if value >= PSI_ALERT:
        return "ALERT"
    if value >= PSI_WARN:
        return "WARN"
    return "OK"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        p = psi(reference[c].to_numpy(), current[c].to_numpy())
        ks = stats.ks_2samp(reference[c].dropna(), current[c].dropna())
        rows.append({
            "feature": c, "psi": round(p, 4),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": float(ks.pvalue),
            "status": status_from_psi(p),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def error_drift(errors: pd.Series, window: int = 14, sigma: float = 3.0) -> pd.DataFrame:
    """Flag days where rolling forecast error leaves a +/- sigma band around the
    long-run mean (a simple model-performance drift signal)."""
    abs_err = errors.abs()
    roll = abs_err.rolling(window, min_periods=window).mean()
    mu, sd = abs_err.mean(), abs_err.std(ddof=0)
    out = pd.DataFrame({"abs_error": abs_err, "rolling_error": roll})
    out["upper_limit"] = mu + sigma * sd / np.sqrt(window)
    out["breach"] = out["rolling_error"] > out["upper_limit"]
    return out
