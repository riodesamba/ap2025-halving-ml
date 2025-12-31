"""Training and evaluation pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import baselines, config, data, features, labels, plots, split


def ensure_output_dirs():
    for path in [config.DATA_DIR, config.OUTPUT_DIR, config.TABLE_DIR, config.FIGURE_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def _count_non_nan(values: np.ndarray) -> int:
    return int(np.sum(~np.isnan(values)))


def _safe_nanmean(values: np.ndarray) -> float:
    n_valid = _count_non_nan(values)
    if n_valid == 0:
        return float("nan")
    return float(np.nanmean(values))


def _safe_nanstd(values: np.ndarray) -> float:
    n_valid = _count_non_nan(values)
    if n_valid == 0:
        return float("nan")
    ddof = 1 if n_valid > 1 else 0
    return float(np.nanstd(values, ddof=ddof))


def _log_split_counts(context: str, n_samples: int, n_pos: int, n_pred_pos: int, note: str | None = None) -> None:
    prefix = f"[{context}] " if context else ""
    message = f"{prefix}n_samples={n_samples}, n_positives={n_pos}, n_predicted_positives={n_pred_pos}"
    if note:
        message = f"{message} ({note})"
    print(message)


def _evaluate_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    *,
    log_counts: bool = False,
    log_context: str | None = None,
) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    valid_mask = ~np.isnan(scores)

    n_samples_raw = len(y_true)
    n_pos_raw = int(y_true.sum())
    n_neg_raw = n_samples_raw - n_pos_raw
    pos_rate_raw = float(n_pos_raw / n_samples_raw) if n_samples_raw else float("nan")
    degenerate_raw = y_true.nunique() < 2 or n_pos_raw == 0 or n_neg_raw == 0
    if not valid_mask.any():
        metrics = {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "threshold": threshold,
            "n_samples": n_samples_raw,
            "n_positives": n_pos_raw,
            "n_negatives": n_neg_raw,
            "n_predicted_positives": 0,
            "n_pos_test": n_pos_raw,
            "n_neg_test": n_neg_raw,
            "pos_rate_test": pos_rate_raw,
            "degenerate_fold": degenerate_raw,
        }
        if log_counts:
            _log_split_counts(log_context or "metrics", n_samples_raw, int(y_true.sum()), 0, "no valid scores")
        return metrics

    if not valid_mask.all():
        y_true = y_true.iloc[valid_mask]
        scores = scores[valid_mask]

    n_samples = len(y_true)
    n_positives = int(y_true.sum())
    n_negatives = n_samples - n_positives
    pos_rate = float(n_positives / n_samples) if n_samples else float("nan")
    degenerate = y_true.nunique() < 2 or n_positives == 0 or n_negatives == 0

    roc_auc = float("nan") if degenerate else roc_auc_score(y_true, scores)
    preds = (scores >= threshold).astype(int)
    n_predicted_positives = int(preds.sum())
    if degenerate:
        pr_auc = float("nan")
        f1 = f1_score(y_true, preds, zero_division=0)
    else:
        pr_auc = average_precision_score(y_true, scores)
        f1 = f1_score(y_true, preds, zero_division=0)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "threshold": threshold,
        "n_samples": n_samples,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "n_predicted_positives": n_predicted_positives,
        "n_pos_test": n_positives,
        "n_neg_test": n_negatives,
        "pos_rate_test": pos_rate,
        "degenerate_fold": degenerate,
    }

    if log_counts:
        note = None
        if degenerate:
            note = "degenerate test fold; AUC metrics undefined"
        _log_split_counts(log_context or "metrics", n_samples, n_positives, n_predicted_positives, note=note)

    return metrics


def _select_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    if y_true.nunique() < 2 or y_true.sum() == 0:
        return 0.5
    precision, recall, thresh = precision_recall_curve(y_true, scores)
    if len(thresh) == 0:
        return 0.5
    f1_vals = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.nanargmax(f1_vals))
    return float(thresh[best_idx])


def _prepare_features(include_halving: bool) -> pd.DataFrame:
    raw = data.load_data()
    feats = features.build_features(raw, include_halving=include_halving)
    df = labels.add_labels(feats)
    df = df.dropna().reset_index(drop=True)
    return df


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"Date", "volatility", "y", "r_t"}
    return [c for c in df.columns if c not in exclude]


def _grid_search(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
    params_grid = config.MODEL_GRIDS[model_name]
    param_options = list(ParameterGrid(params_grid))
    if not param_options:
        raise RuntimeError(f"No hyperparameter options found for model '{model_name}'.")
    inner_train, val_idx = split.inner_train_val_split(X_train.index)
    X_inner_train, X_val = X_train.loc[inner_train], X_train.loc[val_idx]
    y_inner_train, y_val = y_train.loc[inner_train], y_train.loc[val_idx]

    best_score = -np.inf
    best_params = None
    best_threshold = 0.5

    for params in param_options:
        if model_name == "logreg":
            estimator = LogisticRegression(penalty="elasticnet", solver="saga", **params)
        elif model_name == "random_forest":
            estimator = RandomForestClassifier(**params)
        elif model_name == "xgboost":
            estimator = XGBClassifier(**params)
        else:
            raise ValueError(model_name)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X_inner_train, y_inner_train)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        threshold = _select_threshold(y_val, val_scores)
        metrics = _evaluate_metrics(y_val, val_scores, threshold)
        if np.isnan(metrics["pr_auc"]):
            continue
        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_params = params
            best_threshold = threshold

    if best_params is None:
        best_params = param_options[0]
        best_threshold = 0.5

    return best_params, best_threshold


def _fit_and_eval_model(
    model_name: str,
    params: Dict,
    threshold: float,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    log_counts: bool = False,
    log_context: str | None = None,
):
    if model_name == "logreg":
        estimator = LogisticRegression(penalty="elasticnet", solver="saga", **params)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(**params)
    elif model_name == "xgboost":
        estimator = XGBClassifier(**params)
    else:
        raise ValueError(model_name)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    test_scores = pipe.predict_proba(X_test)[:, 1]
    metrics = _evaluate_metrics(y_test, test_scores, threshold, log_counts=log_counts, log_context=log_context)
    return metrics, test_scores


def run_single_pipeline(include_halving: bool) -> pd.DataFrame:
    df = _prepare_features(include_halving=include_halving)
    feature_cols = _get_feature_columns(df)

    records = []
    for wf in split.walk_forward_splits(df):
        train_idx, test_idx = wf["train_idx"], wf["test_idx"]
        fold = wf["fold"]

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        threshold = train_df["volatility"].quantile(0.75)
        y_train = labels.apply_threshold(train_df, threshold)
        y_test = labels.apply_threshold(test_df, threshold)

        X_train, X_test = train_df[feature_cols], test_df[feature_cols]
cd /files/ap2025-halving-ml-backup

git init
git branch -M main
git remote add origin https://github.com/riodesamba/ap2025-halving-ml.git

git add .
git commit -m "Initial project implementation"

git push -f origin main

        # Baselines
        majority_scores = baselines.majority_class_baseline(y_train, len(test_idx))
        majority_metrics = _evaluate_metrics(
            y_test,
            majority_scores.values,
            0.5,
            log_counts=True,
            log_context=f"fold={fold}, model=majority",
        )
        majority_metrics.update({"model": "majority", "fold": fold, "include_halving": include_halving})
        records.append(majority_metrics)

        last_regime_scores = baselines.last_regime_baseline(pd.concat([y_train, y_test]), test_idx)
        last_regime_metrics = _evaluate_metrics(
            y_test,
            last_regime_scores.values,
            0.5,
            log_counts=True,
            log_context=f"fold={fold}, model=last_regime",
        )
        last_regime_metrics.update({"model": "last_regime", "fold": fold, "include_halving": include_halving})
        records.append(last_regime_metrics)

        garch_scores = baselines.garch_baseline(df.set_index("Date")["r_t"], test_idx)
        garch_metrics = _evaluate_metrics(
            y_test,
            garch_scores.values,
            threshold,
            log_counts=True,
            log_context=f"fold={fold}, model=garch",
        )
        garch_metrics.update({"model": "garch", "fold": fold, "include_halving": include_halving})
        records.append(garch_metrics)

        for model_name in ["logreg", "random_forest", "xgboost"]:
            params, best_threshold = _grid_search(model_name, X_train, y_train)
            metrics, scores = _fit_and_eval_model(
                model_name,
                params,
                best_threshold,
                X_train,
                y_train,
                X_test,
                y_test,
                log_counts=True,
                log_context=f"fold={fold}, model={model_name}",
            )
            metrics.update({
                "model": model_name,
                "fold": fold,
                "include_halving": include_halving,
                "param_choice": params,
            })
            records.append(metrics)

    metrics_df = pd.DataFrame(records)
    return metrics_df


def aggregate_results(metrics_df_with: pd.DataFrame, metrics_df_without: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([metrics_df_with, metrics_df_without], ignore_index=True)
    if combined.empty:
        raise RuntimeError(
            "No evaluation records were produced. Ensure price data is available or "
            "place a CSV/Excel file at data/raw.csv before running the pipeline."
        )
    summary = (
        combined.groupby(["model", "include_halving"])
        .agg(
            n_folds_total=("roc_auc", "size"),
            n_folds_valid_roc_auc=("roc_auc", lambda s: _count_non_nan(s.to_numpy())),
            roc_auc_mean=("roc_auc", lambda s: _safe_nanmean(s.to_numpy())),
            roc_auc_std=("roc_auc", lambda s: _safe_nanstd(s.to_numpy())),
            n_folds_valid_pr_auc=("pr_auc", lambda s: _count_non_nan(s.to_numpy())),
            pr_auc_mean=("pr_auc", lambda s: _safe_nanmean(s.to_numpy())),
            pr_auc_std=("pr_auc", lambda s: _safe_nanstd(s.to_numpy())),
            f1_mean=("f1", "mean"),
            degenerate_folds_count=("degenerate_fold", "sum"),
        )
        .reset_index()
    )

    ablation = summary.pivot(index="model", columns="include_halving", values="pr_auc_mean")
    ablation["delta_pr_auc"] = ablation.get(True, 0) - ablation.get(False, 0)
    ablation = ablation.reset_index().rename(columns={True: "pr_auc_with_halving", False: "pr_auc_without_halving"})

    summary = summary.merge(ablation[["model", "pr_auc_with_halving", "pr_auc_without_halving", "delta_pr_auc"]], on="model", how="left")
    return combined, summary


def main(args=None):
    parser = argparse.ArgumentParser(description="Halving ML pipeline")
    parser.add_argument("--config", default="src/halving_ml/config.py", help="Path to config (unused placeholder)")
    parsed = parser.parse_args(args=args)
    _ = parsed

    ensure_output_dirs()

    metrics_with = run_single_pipeline(include_halving=True)
    metrics_without = run_single_pipeline(include_halving=False)

    combined_metrics, summary = aggregate_results(metrics_with, metrics_without)

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    combined_metrics.to_csv(config.TABLE_DIR / "metrics_by_fold.csv", index=False)
    summary.to_csv(config.TABLE_DIR / "summary.csv", index=False)

    plots.plot_volatility_with_halvings()
    plots.plot_pr_auc_by_fold(combined_metrics)
    plots.plot_ablation_delta(summary)
    plots.write_report(summary, combined_metrics)


if __name__ == "__main__":
    main()
