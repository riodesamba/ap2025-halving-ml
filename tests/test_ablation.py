import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from halving_ml import train  # noqa: E402


def test_delta_uses_only_overlapping_valid_folds():
    metrics_with = pd.DataFrame(
        {
            "model": ["m1"] * 4,
            "fold": [0, 1, 2, 3],
            "pr_auc": [0.10, np.nan, 0.30, 0.40],
        }
    )
    metrics_without = pd.DataFrame(
        {
            "model": ["m1"] * 4,
            "fold": [0, 1, 2, 3],
            "pr_auc": [0.05, 0.20, np.nan, 0.45],
        }
    )

    delta = train._compute_delta_stats(metrics_with, metrics_without)
    row = delta.loc[delta["model"] == "m1"].iloc[0]

    assert row["n_folds_total_delta"] == 4
    assert row["n_folds_valid_delta"] == 2  # folds 0 and 3 only
    assert np.isclose(row["delta_pr_auc_mean"], 0.0)
    assert np.isclose(row["delta_pr_auc"], row["delta_pr_auc_mean"])
    assert np.isclose(row["delta_pr_auc_std"], np.sqrt(0.005))
