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

SPLIT_START = datetime(2017, 12, 31)
SPLIT_FREQ_MONTHS = 6

MODEL_GRIDS = {
    "logreg": {
        "C": [0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "max_iter": [2000],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 3],
        "random_state": [42],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "random_state": [42],
    },
}

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
    "SPLIT_START",
    "SPLIT_FREQ_MONTHS",
    "MODEL_GRIDS",
    "PLOT_STYLE",
]
