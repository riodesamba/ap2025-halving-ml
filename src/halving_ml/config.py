"""Default configuration for the halving-ML pipeline."""
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "report.md"
RAW_DATA_PATH = DATA_DIR / "raw.csv"

SYMBOL = "BTC-USD"
DATA_START = datetime(2011, 1, 1)

HALVING_DATES = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 20),
]

FEATURE_WINDOWS = [5, 10, 30]
ROLLING_VOL_WINDOW = 30
LABEL_QUANTILE = 0.75

HALVING_FEATURES = ["days_since_last_halving", "pre_halving_30d", "post_halving_30d"]
BASE_FEATURES = [
    "r_lag",
    "price_lag",
    "ret_mean_5",
    "ret_std_5",
    "ret_skew_5",
    "ret_kurt_5",
    "ret_mean_10",
    "ret_std_10",
    "ret_skew_10",
    "ret_kurt_10",
    "ret_mean_30",
    "ret_std_30",
    "ret_skew_30",
    "ret_kurt_30",
    "ret_sum_5",
    "sma10_sma50_ratio",
    "drawdown",
]
HALVING_FEATURE_NAMES = list(HALVING_FEATURES)

SPLIT_START = datetime(2017, 12, 31)
SPLIT_FREQ_MONTHS = 6

MODEL_GRIDS = {
    "logreg": {
        "C": [0.5, 1.0],
        "l1_ratio": [0.5],
        "max_iter": [1000],
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "n_jobs": [1],
        "random_state": [42],
    },
    "random_forest": {
        "n_estimators": [100],
        "max_depth": [5],
        "min_samples_leaf": [2],
        "n_jobs": [1],
        "random_state": [42],
    },
    "xgboost": {
        "n_estimators": [120, 200],
        "learning_rate": [0.05],
        "max_depth": [3],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "random_state": [42],
        "tree_method": ["hist"],
        "n_jobs": [1],
    },
}

MODEL_SEARCH_BUDGET = {"logreg": 2, "random_forest": 1, "xgboost": 2}

PLOT_STYLE = {
    "style": "seaborn-v0_8",
    "figsize": (10, 5),
}

__all__ = [
    "DATA_DIR",
    "OUTPUT_DIR",
    "TABLE_DIR",
    "FIGURE_DIR",
    "REPORT_PATH",
    "RAW_DATA_PATH",
    "SYMBOL",
    "DATA_START",
    "HALVING_DATES",
    "FEATURE_WINDOWS",
    "ROLLING_VOL_WINDOW",
    "LABEL_QUANTILE",
    "HALVING_FEATURES",
    "BASE_FEATURES",
    "SPLIT_START",
    "SPLIT_FREQ_MONTHS",
    "MODEL_GRIDS",
    "MODEL_SEARCH_BUDGET",
    "PLOT_STYLE",
    "HALVING_FEATURE_NAMES",
    "get_feature_names",
]


def get_feature_names(include_halving: bool = True):
    """Return the feature names used for modeling."""
    names = list(BASE_FEATURES)
    if include_halving:
        names.extend(HALVING_FEATURES)
    return names
