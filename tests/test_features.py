import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from halving_ml import features  # noqa: E402
from halving_ml import config  # noqa: E402


def _dummy_prices(n: int = 80, start_price: float = 100.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = start_price * (1 + 0.001) ** np.arange(n)
    return pd.DataFrame({"Date": dates, "Close": prices})


def test_feature_alignment_ignores_same_day_price():
    base_df = _dummy_prices()
    features_base = features.build_features(base_df, include_halving=False)

    modified = base_df.copy()
    modified.loc[modified.index[-1], "Close"] *= 10
    features_modified = features.build_features(modified, include_halving=False)

    cols = [c for c in features_base.columns if c not in {"Date", "r_t"}]
    last_idx = features_base.index[-1]
    pd.testing.assert_series_equal(
        features_base.loc[last_idx, cols], features_modified.loc[last_idx, cols]
    )


def test_lagged_return_matches_manual_computation():
    df = _dummy_prices(n=70, start_price=100)
    feats = features.build_features(df, include_halving=False)
    feats = feats.dropna()
    first_idx = feats.index[0]
    expected = np.log(df.loc[first_idx - 1, "Close"] / df.loc[first_idx - 2, "Close"])
    assert np.isclose(feats.iloc[0]["r_lag"], expected)


def test_config_feature_names_align_with_built_features():
    df = _dummy_prices()

    feats_with = features.build_features(df, include_halving=True).dropna()
    feats_without = features.build_features(df, include_halving=False).dropna()

    expected_with = set(config.get_feature_names(include_halving=True))
    expected_without = set(config.get_feature_names(include_halving=False))

    assert expected_with.issubset(set(feats_with.columns))
    assert expected_without.issubset(set(feats_without.columns))
    assert expected_with.issuperset(expected_without)
    assert not expected_without.intersection(config.HALVING_FEATURE_NAMES)
