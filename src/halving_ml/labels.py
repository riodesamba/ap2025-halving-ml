"""Label construction for volatility prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_realized_vol(returns: pd.Series) -> pd.Series:
    return np.sqrt(365) * returns.rolling(config.ROLLING_VOL_WINDOW).std()


def add_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    vol = compute_realized_vol(df.set_index("Date")["r_t"])
    # Align by position to avoid index mismatch after resetting indices
    df["volatility"] = vol.to_numpy()
    return df


def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df["volatility"] >= threshold).astype(int)


__all__ = ["add_labels", "apply_threshold", "compute_realized_vol"]
