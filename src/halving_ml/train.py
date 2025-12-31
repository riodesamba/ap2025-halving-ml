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


def _feature_columns(include_halving: bool) -> List[str]:
    cols = list(config.BASE_FEATURES)
    if include_halving:
        cols.extend(config.HALVING_FEATURES)
    return cols


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


def _prepare_base_frame() -> pd.DataFrame:
    raw = data.load_data()
    feats = features.build_features(raw, include_halving=True)
    df = labels.add_labels(feats)
    df = df.dropna().reset_index(drop=True)
    return df


def _summarize_delta(group: pd.DataFrame) -> pd.Series:
    pr_auc_with = group["pr_auc_with"].to_numpy()
    pr_auc_without = group["pr_auc_without"].to_numpy()
    valid = ~np.isnan(pr_auc_with) & ~np.isnan(pr_auc_without)
    deltas = pr_auc_with[valid] - pr_auc_without[valid]

    return pd.Series(
        {
            "delta_pr_auc_mean": _safe_nanmean(deltas),
            "delta_pr_auc_std": _safe_nanstd(deltas),
            "n_folds_valid_delta": int(valid.sum()),
            "n_folds_total_delta": len(group),
        }
    )


def _compute_delta_stats(metrics_df_with: pd.DataFrame, metrics_df_without: pd.DataFrame) -> pd.DataFrame:
    merged = (
        metrics_df_with[["model", "fold", "pr_auc"]]
        .rename(columns={"pr_auc": "pr_auc_with"})
        .merge(
            metrics_df_without[["model", "fold", "pr_auc"]].rename(columns={"pr_auc": "pr_auc_without"}),
            on=["model", "fold"],
            how="outer",
        )
    )

    grouped = merged.groupby("model")[["pr_auc_with", "pr_auc_without"]]
    delta_stats = grouped.apply(_summarize_delta).reset_index()
    delta_stats["delta_pr_auc"] = delta_stats["delta_pr_auc_mean"]
    return delta_stats


def _build_fold_definitions(df: pd.DataFrame) -> List[Dict]:
    folds = []
    for split_def in split.walk_forward_splits(df):
        train_idx, test_idx = split_def["train_idx"], split_def["test_idx"]
        threshold = float(df.loc[train_idx, "volatility"].quantile(config.LABEL_QUANTILE))
        folds.append(
            {
                **split_def,
                "threshold": threshold,
                "train_size": len(train_idx),
                "eval_size": len(test_idx),
            }
        )
    return folds


def _log_split_counts(context: str, n_samples: int, n_pos: int, n_pred_pos: int, note: str | None = None) -> None:
    prefix = f"[{context}] " if context else ""
    message = f"{prefix}n_samples={n_samples}, n_positives={n_pos}, n_predicted_positives={n_pred_pos}"
    if note:
        message = f"{message} ({note})"
    print(message)


def _persist_partial_metrics(records: List[Dict], include_halving: bool) -> None:
    if not records:
        return
    suffix = "with_halving" if include_halving else "without_halving"
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(config.TABLE_DIR / f"metrics_by_fold_{suffix}_partial.csv", index=False)


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


def _iter_param_configs(model_name: str) -> List[Dict]:
    grid = list(ParameterGrid(config.MODEL_GRIDS[model_name]))
    budget = config.MODEL_SEARCH_BUDGET.get(model_name, len(grid))
    if budget >= len(grid):
        return grid
    rng = np.random.default_rng(seed=42)
    chosen = rng.choice(len(grid), size=budget, replace=False)
    return [grid[i] for i in chosen]


def _grid_search(model_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], float]:
    param_grid = _iter_param_configs(model_name)
    best_params = None
    best_f1 = -np.inf
    best_threshold = 0.5

    for params in param_grid:
        if model_name == "logreg":
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])
        elif model_name == "random_forest":
            model = RandomForestClassifier(**params)
        elif model_name == "xgboost":
            model = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        threshold = _select_threshold(y, scores)
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(y, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_threshold = threshold

    return best_params, best_threshold


def _fit_and_eval_model(
    model_name: str,
    params: Dict[str, float],
    threshold: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    log_counts: bool = False,
    log_context: str | None = None,
) -> Tuple[Dict[str, float], pd.Series]:
    if model_name == "logreg":
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_name == "xgboost":
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate_metrics(y_test, scores, threshold, log_counts=log_counts, log_context=log_context)
    return metrics, pd.Series(scores, index=y_test.index)


def run_single_pipeline(include_halving: bool, base_df: pd.DataFrame, folds: List[Dict]) -> pd.DataFrame:
    feature_cols = _feature_columns(include_halving)
    records: List[Dict[str, float]] = []

    for fold_def in folds:
        fold = fold_def["fold"]
        train_idx, test_idx = fold_def["train_idx"], fold_def["test_idx"]
        threshold = fold_def["threshold"]

        train_df = base_df.loc[train_idx]
        test_df = base_df.loc[test_idx]

        y_train = labels.apply_threshold(train_df, threshold)
        y_test = labels.apply_threshold(test_df, threshold)

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        majority_scores = baselines.majority_class_baseline(y_train, len(test_idx))
        majority_metrics = _evaluate_metrics(
            y_test,
            majority_scores.values,
            0.5,
            log_counts=True,
            log_context=f"fold={fold}, model=majority",
        )
        majority_metrics.update(
            {
                "model": "majority",
                "fold": fold,
                "include_halving": include_halving,
                "train_size": fold_def["train_size"],
                "eval_size": fold_def["eval_size"],
                "label_threshold": threshold,
            }
        )
        records.append(majority_metrics)

        last_regime_scores = baselines.last_regime_baseline(pd.concat([y_train, y_test]), test_idx)
        last_regime_metrics = _evaluate_metrics(
            y_test,
            last_regime_scores.values,
            0.5,
            log_counts=True,
            log_context=f"fold={fold}, model=last_regime",
        )
        last_regime_metrics.update(
            {
                "model": "last_regime",
                "fold": fold,
                "include_halving": include_halving,
                "train_size": fold_def["train_size"],
                "eval_size": fold_def["eval_size"],
                "label_threshold": threshold,
            }
        )
        records.append(last_regime_metrics)

        garch_scores = baselines.garch_baseline(base_df.set_index("Date")["r_t"], test_idx)
        garch_metrics = _evaluate_metrics(
            y_test,
            garch_scores.values,
            threshold,
            log_counts=True,
            log_context=f"fold={fold}, model=garch",
        )
        garch_metrics.update(
            {
                "model": "garch",
                "fold": fold,
                "include_halving": include_halving,
                "train_size": fold_def["train_size"],
                "eval_size": fold_def["eval_size"],
                "label_threshold": threshold,
            }
        )
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
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold,
                    "include_halving": include_halving,
                    "param_choice": params,
                    "train_size": fold_def["train_size"],
                    "eval_size": fold_def["eval_size"],
                    "label_threshold": threshold,
                }
            )
            records.append(metrics)

        _persist_partial_metrics(records, include_halving)

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

    delta_stats = _compute_delta_stats(metrics_df_with, metrics_df_without)
    ablation = summary.pivot(index="model", columns="include_halving", values="pr_auc_mean")
    ablation = ablation.reset_index().rename(columns={True: "pr_auc_with_halving", False: "pr_auc_without_halving"})
    ablation = ablation.merge(delta_stats, on="model", how="left")

    summary = summary.merge(
        ablation[
            [
                "model",
                "pr_auc_with_halving",
                "pr_auc_without_halving",
                "delta_pr_auc",
                "delta_pr_auc_mean",
                "delta_pr_auc_std",
                "n_folds_valid_delta",
                "n_folds_total_delta",
            ]
        ],
        on="model",
        how="left",
    )
    return combined, summary


def main(args=None):
    parser = argparse.ArgumentParser(description="Halving ML pipeline")
    parser.add_argument("--config", default="src/halving_ml/config.py", help="Path to config (unused placeholder)")
    parsed = parser.parse_args(args=args)
    _ = parsed

    ensure_output_dirs()

    base_df = _prepare_base_frame()
    folds = _build_fold_definitions(base_df)
    if not folds:
        raise RuntimeError("No walk-forward folds were generated; check the available data range.")

    fold_metadata = pd.DataFrame(
        [
            {
                "fold": f["fold"],
                "start": f["start"],
                "end": f["end"],
                "train_size": f["train_size"],
                "eval_size": f["eval_size"],
                "label_threshold": f["threshold"],
            }
            for f in folds
        ]
    )
    fold_metadata.to_csv(config.TABLE_DIR / "fold_metadata.csv", index=False)

    metrics_with = run_single_pipeline(include_halving=True, base_df=base_df, folds=folds)
    metrics_without = run_single_pipeline(include_halving=False, base_df=base_df, folds=folds)

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
