## Statistical Analysis of Bitcoin Prices and Halving Effects  
*Category:* Data science · Time series · Event study

## Problem statement
Every ~4 years Bitcoin halves its block reward. People commonly claim volatility spikes around those dates, but is it really the case. I’m turning this into a prediction problem: can we predict high-volatility days using features available at time t, including simple halving signals? The goal isn’t to prove causality, it’s to build and compare ML models with proper temporal validation, and to see whether halving-related features add any predictive value beyond standard time-series signals.

## Planned approach and technologies
**Data.** Daily BTC close in USD.
Compute log-returns and 30-day realized volatility on a rolling basis.
**What I’ll do.**
Label (binary). “High vol” = top-quartile of 30-day realized vol. The threshold is computed within each training fold (no look-ahead).

Features (all lagged). Rolling stats of returns (mean/std/skew/kurt over 5/10/30 days), momentum (SMA10/SMA50, last-5d return), drawdown, days-since-halving and pre/post-halving flags.
Models. Logistic Regression (Elastic Net), Random Forest, XGBoost. (LSTM only if time.)

Validation. Strict walk-forward (expanding window). Hyper-parameter tuning happens inside training folds only. Final metrics on a held-out recent period.

Baselines. Class-prior (always majority), “last-regime” (today = yesterday), and a simple GARCH(1,1) signal mapped to the binary label.
Tech. Python 3.11 with pandas, numpy, scikit-learn, xgboost, matplotlib (statsmodels for GARCH). Light tests with pytest. A small CLI/Makefile so one command rebuilds figures (PNG), tables (CSV), and a short report (Markdown).

## Expected challenges
- **Imbalanced classes.** High-vol days are rarer. I’ll report ROC-AUC, precision–recall AUC, and F1, and adjust decision thresholds/class weights if needed.
- **Strange return behavior.** Crypto has outliers. I’ll work with returns/log-returns, use robust summaries, and scale features within each training window.
- **Few halving events** Labels are daily, so there’s enough data. I’ll keep models simple and regularized to avoid overfitting.

## Success criteria
- **Reproducible:** one command rebuilds data, labels, features, training, and plots from scratch.
- **Beats baselines:** ROC-AUC and PR-AUC clearly above the majority/last-regime baselines across multiple walk-forward splits.
- **Consistent & readable:** similar results across splits; plots and tables that make the comparison obvious.
- **Insight:** a short ablation (with vs. without halving features) shows whether halving actually helps prediction.


## Stretch goals
- **Volatility regression:** predict next-30-day log-vol and compare XGBoost/LSTM to **GARCH(1,1)**.
- **Direction task:** classify up/down returns with the same time-aware validation.
- **Tiny UI:** a simple Streamlit page to pick window sizes and view metrics/plots.