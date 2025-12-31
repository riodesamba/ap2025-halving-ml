import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from halving_ml import split  # noqa: E402
from halving_ml import labels  # noqa: E402


def test_threshold_computed_from_train_only():
    dates = pd.date_range("2017-11-01", periods=300, freq="D")
    close = pd.Series(np.arange(len(dates), dtype=float) + 100, index=dates)
    df = pd.DataFrame({"Date": dates, "Close": close.values})

    # fabricate returns and volatility
    df["r_t"] = 0.01
    df["volatility"] = 0.03

    # Introduce class separation in training and high volatility only in the test horizon
    train_mask = df["Date"] < pd.Timestamp("2018-01-01")
    low_train_idx = train_mask[train_mask].index[:20]
    df.loc[low_train_idx, "volatility"] = 0.01

    # Introduce class separation in training and high volatility only in the test horizon
    train_mask = df["Date"] < pd.Timestamp("2018-01-01")
    low_train_idx = train_mask[train_mask].index[:20]
    df.loc[low_train_idx, "volatility"] = 0.01

    # Make the first test window (Jan-Jun 2018) contain high-vol targets
    df.loc[df["Date"] >= pd.Timestamp("2018-03-01"), "volatility"] = 10.0

    folds = list(split.walk_forward_splits(df))
    assert folds, "No folds generated"
    first = folds[0]
    train_df = df.loc[first["train_idx"]]
    test_df = df.loc[first["test_idx"]]

    threshold = train_df["volatility"].quantile(0.75)
    y_train = labels.apply_threshold(train_df, threshold)
    y_test = labels.apply_threshold(test_df, threshold)

    assert y_train.min() == 0
    assert y_train.max() == 1
    assert y_test.max() == 1
    assert threshold < test_df.loc[test_df["volatility"] == 10.0, "volatility"].min()
