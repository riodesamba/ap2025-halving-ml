# Halving-ML

## What
Predict **high-volatility days** in Bitcoin using ML (LogReg, RF, XGBoost) with **walk-forward** validation.

## Run

```bash
make all
# or
python -m halving_ml.train --config src/halving_ml/config.py
```

`make all` will install dependencies from `requirements.txt` before running the pipeline. You can also install them manually with `make install`.

If your environment blocks downloads, place your own BTC-USD price history as `data/raw.csv` or `data/raw.xlsx` with `Date` and `Close` columns (Date should be parseable as dates, Close as numeric). The pipeline will load that instead of fetching from yfinance.

If no local data is available and downloads fail, the pipeline falls back to a synthetic price series saved to `data/raw.csv` so the run can complete. If you want to load Excel data (`.xlsx`/`.xls`), install `openpyxl` manually since it is an optional dependency.

## Outputs

Running the pipeline writes summaries to `outputs/` and figures to `outputs/figures/`. Key report-ready plots:

- `volatility_with_halvings.png`: realized 30-day annualized volatility with Bitcoin halving dates overlaid.
- `label_threshold_example.png`: example training-fold volatility distribution with the 75th percentile threshold computed on train only.
- `walk_forward_splits.png`: visual depiction of the expanding-window train and fixed test walk-forward folds.
- `pos_rate_by_fold.png`: positive class rate per test fold, flagging degenerate folds (all one class).
- `pr_auc_by_fold.png`: PR-AUC per fold for ML models with halving features, highlighting degenerate folds.
- `metrics_summary_bar.png`: mean PR-AUC with error bars across valid folds for all baselines and ML models.
- `ablation_delta.png`: PR-AUC deltas (with vs. without halving features) for every model, including baselines with zero/undefined deltas.
