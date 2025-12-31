"""Baseline predictors."""
from __future__ import annotations

from typing import Tuple

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


def _safe_residual(return_value: float, mean_param: float) -> float:
    if np.isnan(return_value):
        return 0.0
    return float(return_value * 100 - mean_param)


def garch_baseline(returns: pd.Series, train_idx, test_idx) -> Tuple[pd.Series, pd.Series]:
    """Fit GARCH(1,1) on train data and roll forward 1-step sigma forecasts.

    Parameters
    ----------
    returns
        Full return series (ordered as in the feature matrix).
    train_idx
        Indices for the training window.
    test_idx
        Indices for the evaluation window.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Annualized conditional volatility for train and test windows.
    """

    series = returns.reset_index(drop=True).astype(float)
    train_returns = series.iloc[train_idx]
    if train_returns.dropna().empty:
        empty_train = pd.Series([np.nan] * len(train_idx), index=train_idx)
        empty_test = pd.Series([np.nan] * len(test_idx), index=test_idx)
        return empty_train, empty_test

    model = arch_model(train_returns * 100, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off")

    fitted_train_sigma = pd.Series(
        (res.conditional_volatility / 100) * np.sqrt(365), index=train_returns.index
    )

    omega = float(res.params.get("omega", 0.0))
    alpha = float(res.params.get("alpha[1]", 0.0))
    beta = float(res.params.get("beta[1]", 0.0))
    mu = float(res.params.get("mu", 0.0))

    last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
    last_resid = float(res.resid.iloc[-1])

    test_returns = series.iloc[test_idx]
    forecasts = []
    for ret in test_returns:
        sigma2_next = omega + alpha * last_resid**2 + beta * last_sigma2
        sigma2_next = max(float(sigma2_next), 0.0)
        forecasts.append(np.sqrt(sigma2_next) / 100 * np.sqrt(365))

        last_resid = _safe_residual(ret, mu)
        last_sigma2 = sigma2_next

    forecast_series = pd.Series(forecasts, index=test_returns.index)
    return fitted_train_sigma, forecast_series


__all__ = ["majority_class_baseline", "last_regime_baseline", "garch_baseline"]
