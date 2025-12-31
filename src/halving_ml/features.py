"""Feature engineering utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _halving_features(dates: pd.Series) -> pd.DataFrame:
    halving_dates = pd.to_datetime(config.HALVING_DATES)
    days_since_last = []
    pre_halving_30d = []
    post_halving_30d = []

    for date in dates:
        past = halving_dates[halving_dates <= date]
        future = halving_dates[halving_dates > date]
        last_halving = past.max() if len(past) else pd.NaT
        next_halving = future.min() if len(future) else pd.NaT

        if pd.isna(last_halving):
            days_since_last.append(np.nan)
        else:
            days_since_last.append((pd.Timestamp(date) - last_halving).days)

        if pd.isna(next_halving):
            pre_halving_30d.append(False)
        else:
            pre_halving_30d.append(0 <= (next_halving - pd.Timestamp(date)).days <= 30)

        if pd.isna(last_halving):
            post_halving_30d.append(False)
        else:
            post_halving_30d.append(0 <= (pd.Timestamp(date) - last_halving).days <= 30)

    return pd.DataFrame(
        {
            "days_since_last_halving": days_since_last,
            "pre_halving_30d": pre_halving_30d,
            "post_halving_30d": post_halving_30d,
        },
        index=dates.index,
    )


def build_features(df: pd.DataFrame, include_halving: bool = True) -> pd.DataFrame:
    prices = df.set_index("Date")["Close"].copy()
    returns = np.log(prices / prices.shift(1))

    base_ret = returns.shift(1)
    base_price = prices.shift(1)

    features = pd.DataFrame(index=prices.index)
    features["r_lag"] = base_ret
    features["price_lag"] = base_price

    def _safe_skew(window_values: pd.Series) -> float:
        """Return a finite skew value even when variance is (near) zero.

        Pandas returns ``NaN`` for skew/kurtosis when the window contains
        constant values (zero variance). That leads to entire columns of
        ``NaN`` for deterministic price paths, which then wipes out all
        rows after a ``dropna()``. Returning ``0.0`` in those cases keeps
        feature matrices usable while preserving alignment behavior.
        """

        values = window_values.dropna()
        if len(values) == 0:
            return np.nan

        if np.isfinite(values.std(ddof=0)) and values.std(ddof=0) < 1e-12:
            return 0.0
        return values.skew()

    def _safe_kurt(window_values: pd.Series) -> float:
        values = window_values.dropna()
        if len(values) == 0:
            return np.nan

        if np.isfinite(values.std(ddof=0)) and values.std(ddof=0) < 1e-12:
            return 0.0
        return values.kurt()

    for window in config.FEATURE_WINDOWS:
        rolling = base_ret.rolling(window)
        features[f"ret_mean_{window}"] = rolling.mean()
        features[f"ret_std_{window}"] = rolling.std()
        features[f"ret_skew_{window}"] = rolling.apply(_safe_skew, raw=False)
        features[f"ret_kurt_{window}"] = rolling.apply(_safe_kurt, raw=False)

    features["ret_sum_5"] = base_ret.rolling(5).sum()
    sma10 = base_price.rolling(10).mean()
    sma50 = base_price.rolling(50).mean()
    features["sma10_sma50_ratio"] = sma10 / sma50

    rolling_peak = base_price.expanding().max()
    features["drawdown"] = base_price / rolling_peak - 1

    halving_df = _halving_features(features.index.to_series())
    if include_halving:
        features = pd.concat([features, halving_df], axis=1)

    features["r_t"] = returns
    return features.reset_index().rename(columns={"index": "Date"})


__all__ = ["build_features"]
