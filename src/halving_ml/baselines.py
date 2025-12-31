"""Baseline predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def majority_class_baseline(y_train: pd.Series, size: int) -> pd.Series:
    pos_rate = float(y_train.mean())
    return pd.Series([pos_rate] * size)


def last_regime_baseline(y: pd.Series, test_idx) -> pd.Series:
    shifted = y.shift(1)
    preds = shifted.loc[test_idx]
    if preds.isna().any():
        preds = preds.fillna(int(y.value_counts().idxmax()))
    return preds.astype(float)


def garch_baseline(returns: pd.Series, test_idx) -> pd.Series:
    series = returns.reset_index(drop=True)
    train_returns = series.iloc[: test_idx[0]].dropna()
    if train_returns.empty:
        return pd.Series([0.0] * len(test_idx), index=test_idx)

    model = arch_model(train_returns * 100, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1, start=len(train_returns), reindex=True, align="target")
    sigma = (forecast.variance.squeeze() ** 0.5) / 100
    sigma = sigma.reindex(range(len(series)))
    preds = sigma.iloc[test_idx]
    preds = preds.ffill().bfill()
    return preds * np.sqrt(365)


__all__ = ["majority_class_baseline", "last_regime_baseline", "garch_baseline"]
