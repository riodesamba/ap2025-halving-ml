import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from halving_ml import train  # noqa: E402


def test_all_zero_labels_are_handled_without_warnings():
    y_true = pd.Series([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        metrics = train._evaluate_metrics(y_true, scores, threshold=0.5)

    assert not caught
    assert np.isnan(metrics["pr_auc"])
    assert metrics["f1"] == 0.0
    assert np.isnan(metrics["roc_auc"])
    assert metrics["n_samples"] == len(y_true)
    assert metrics["n_positives"] == 0
    assert metrics["n_predicted_positives"] == 0
    assert metrics["degenerate_fold"] is True
    assert metrics["n_pos_test"] == 0
    assert metrics["n_neg_test"] == len(y_true)
    assert metrics["pos_rate_test"] == 0.0


def test_select_threshold_handles_all_zero_labels_without_warnings():
    y_true = pd.Series([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        threshold = train._select_threshold(y_true, scores)

    assert not caught
    assert threshold == 0.5


def test_degenerate_fold_sets_auc_to_nan_and_marks_flag():
    y_true = pd.Series([1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    metrics = train._evaluate_metrics(y_true, scores, threshold=0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert metrics["degenerate_fold"] is True
    assert metrics["n_pos_test"] == 4
    assert metrics["n_neg_test"] == 0
    assert metrics["pos_rate_test"] == 1.0
