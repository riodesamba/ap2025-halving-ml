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
