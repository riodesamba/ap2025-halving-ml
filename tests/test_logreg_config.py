import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from halving_ml import config, train


def test_logreg_params_force_elasticnet_penalty():
    params = {"C": 1.0, "l1_ratio": 0.5}
    kwargs = train._logreg_params(params)

    assert kwargs["penalty"] == "elasticnet"
    assert kwargs["solver"] == "saga"
    assert kwargs["C"] == 1.0
    assert kwargs["l1_ratio"] == 0.5
    assert kwargs["max_iter"] == 1000

    with pytest.raises(ValueError):
        train._logreg_params({"penalty": "l2", "C": 1.0, "l1_ratio": 0.5})


def test_grid_search_uses_elasticnet_logreg():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, len(config.BASE_FEATURES))), columns=config.BASE_FEATURES)
    y = pd.Series(rng.integers(0, 2, size=20))

    best_params, threshold = train._grid_search("logreg", X, y)

    assert 0 <= threshold <= 1
    assert "penalty" not in best_params or best_params["penalty"] == "elasticnet"
