"""Plotting and reporting utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


plt.style.use(config.PLOT_STYLE["style"])


_MODEL_ORDER: List[str] = ["majority", "last_regime", "garch", "logreg", "random_forest", "xgboost"]


def _fold_level_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse metrics to one row per fold (pos rate / degeneracy are the same across models)."""
    cols = [
        "fold",
        "pos_rate_test",
        "n_pos_test",
        "n_neg_test",
        "degenerate_fold",
        "label_threshold",
        "train_size",
        "eval_size",
    ]
    return (
        metrics_df.sort_values(["fold", "model"])
        .drop_duplicates(subset=["fold"])[cols]
        .reset_index(drop=True)
        .sort_values("fold")
    )


def plot_volatility_with_halvings(base_df: pd.DataFrame):
    df = base_df.dropna(subset=["volatility"])
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    ax.plot(df["Date"], df["volatility"], label="Volatility (30d annualized)", color="steelblue")
    for date in config.HALVING_DATES:
        ax.axvline(
            date,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Halving" if date == config.HALVING_DATES[0] else None,
        )
    ax.set_title("BTC Realized Volatility (30d) with Halving Dates")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "volatility_with_halvings.png", dpi=150)
    plt.close(fig)


def plot_label_threshold_example(base_df: pd.DataFrame, folds: Iterable[Dict], metrics_df: pd.DataFrame):
    fold_df = _fold_level_frame(metrics_df)
    candidate = fold_df[(fold_df["pos_rate_test"] > 0) & (fold_df["pos_rate_test"] < 1)]
    if candidate.empty:
        return
    fold_id = int(candidate.iloc[0]["fold"])
    fold_lookup = {f["fold"]: f for f in folds}
    if fold_id not in fold_lookup:
        return
    fold = fold_lookup[fold_id]
    train_idx = fold["train_idx"]
    threshold = fold["threshold"]
    train_vol = base_df.loc[train_idx, "volatility"]

    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    ax.hist(train_vol, bins=30, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(threshold, color="darkorange", linestyle="--", linewidth=2, label="75th percentile (train)")
    ax.set_title(f"Fold {fold_id}: Train Volatility Distribution")
    ax.set_xlabel("Realized volatility (30d annualized)")
    ax.set_ylabel("Density")
    ax.annotate(
        "threshold computed on train only",
        xy=(threshold, ax.get_ylim()[1] * 0.9),
        xytext=(threshold, ax.get_ylim()[1] * 0.7),
        arrowprops={"arrowstyle": "->", "color": "darkorange"},
        color="darkorange",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "label_threshold_example.png", dpi=150)
    plt.close(fig)


def plot_walk_forward_splits(folds: Iterable[Dict], base_df: pd.DataFrame):
    if not folds:
        return
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    train_color, test_color = "#4C72B0", "#DD8452"
    train_start = pd.to_datetime(base_df["Date"]).min()

    for i, fold in enumerate(folds, 1):
        y = i - 0.4
        test_start = fold["start"]
        test_end = fold["end"]
        train_end = test_start - pd.DateOffset(days=1)

        ax.broken_barh(
            [(mdates.date2num(train_start), mdates.date2num(train_end) - mdates.date2num(train_start))],
            (y, 0.25),
            facecolors=train_color,
        )
        ax.broken_barh(
            [(mdates.date2num(test_start), mdates.date2num(test_end) - mdates.date2num(test_start))],
            (y + 0.3, 0.25),
            facecolors=test_color,
        )
        ax.text(test_end, y + 0.55, f"Fold {fold['fold']}", fontsize=9, va="center", ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")
    ax.set_yticks([])
    ax.set_title("Walk-Forward Expanding Window Splits")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_tick_params(rotation=30)
    legend_handles = [
        plt.Line2D([0], [0], color=train_color, lw=6, label="Train (expanding)"),
        plt.Line2D([0], [0], color=test_color, lw=6, label="Test window"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "walk_forward_splits.png", dpi=150)
    plt.close(fig)


def plot_pos_rate_by_fold(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    fold_df = _fold_level_frame(metrics_df)
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    colors = np.where(fold_df["degenerate_fold"], "#C44E52", "#4C72B0")
    bars = ax.bar(fold_df["fold"], fold_df["pos_rate_test"], color=colors)

    deg_patch = plt.Rectangle((0, 0), 1, 1, color="#C44E52", label="degenerate fold")
    reg_patch = plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="pos rate (test)")
    ax.legend(handles=[reg_patch, deg_patch])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Positive rate (test)")
    ax.set_title("Test Positive Rate by Fold")
    ax.set_ylim(0, 1)

    for bar, pos_rate in zip(bars, fold_df["pos_rate_test"]):
        if pd.isna(pos_rate):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{pos_rate:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "pos_rate_by_fold.png", dpi=150)
    plt.close(fig)


def plot_pr_auc_by_fold(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    subset = metrics_df[(metrics_df["include_halving"]) & (metrics_df["model"].isin(["logreg", "random_forest", "xgboost"]))]
    if subset.empty:
        return

    fold_df = _fold_level_frame(metrics_df)
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])

    for model, group in subset.groupby("model"):
        group = group.sort_values("fold")
        valid = group["pr_auc"].notna()
        ax.plot(group.loc[valid, "fold"], group.loc[valid, "pr_auc"], marker="o", label=model)

    degenerate_folds = fold_df[fold_df["degenerate_fold"]]["fold"].tolist()
    if degenerate_folds:
        for fold in degenerate_folds:
            ax.axvspan(fold - 0.45, fold + 0.45, color="grey", alpha=0.15)
        ax.plot([], [], color="grey", alpha=0.3, lw=10, label="degenerate fold")

    ax.set_xlabel("Fold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC by Fold (halving features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "pr_auc_by_fold.png", dpi=150)
    plt.close(fig)


def plot_metrics_summary_bar(summary_df: pd.DataFrame):
    if summary_df.empty:
        return
    subset = summary_df[summary_df["include_halving"]].copy()
    subset = subset.set_index("model").reindex(_MODEL_ORDER)
    means = subset["pr_auc_mean"].fillna(0)
    errs = subset["pr_auc_std"].fillna(0)
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    bars = ax.bar(_MODEL_ORDER, means, yerr=errs, color="#4C72B0", alpha=0.8, capsize=5)
    ax.set_ylabel("PR-AUC (mean across valid folds)")
    ax.set_title("Model Performance (halving features)")
    ax.set_ylim(0, max(0.01, (means.max() + errs.max()) * 1.1))

    for bar, model in zip(bars, _MODEL_ORDER):
        n_valid_raw = subset.loc[model, "n_folds_valid_pr_auc"] if model in subset.index else 0
        n_valid = 0 if pd.isna(n_valid_raw) else int(n_valid_raw)
        if n_valid == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01, "no valid\nfolds", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "metrics_summary_bar.png", dpi=150)
    plt.close(fig)


def plot_ablation_delta(summary_df: pd.DataFrame):
    if summary_df.empty:
        return
    subset = summary_df.drop_duplicates("model").set_index("model").reindex(_MODEL_ORDER)
    deltas = subset["delta_pr_auc_mean"].fillna(0)
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    bars = ax.bar(_MODEL_ORDER, deltas, color="steelblue", alpha=0.85)
    for model, bar in zip(_MODEL_ORDER, bars):
        n_valid_raw = subset.loc[model, "n_folds_valid_delta"] if model in subset.index else 0
        n_valid = 0 if pd.isna(n_valid_raw) else int(n_valid_raw)
        if n_valid == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01, "no valid\n delta folds", ha="center", va="bottom", fontsize=8)

    total_valid = subset["n_folds_valid_delta"].dropna()
    if len(total_valid.unique()) == 1 and not total_valid.empty:
        title_suffix = f"n_folds_valid_delta={int(total_valid.iloc[0])}"
    elif not total_valid.empty:
        title_suffix = f"n_folds_valid_delta={int(total_valid.min())}-{int(total_valid.max())}"
    else:
        title_suffix = "n_folds_valid_delta=0"
    ax.set_title(f"Impact of Halving Features ({title_suffix})")
    ax.set_ylabel("PR-AUC delta (with - without)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "ablation_delta.png", dpi=150)
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, metrics_df: pd.DataFrame):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        summary_md = summary_df.to_markdown(index=False)
        metrics_md = metrics_df.head().to_markdown(index=False)
    except ImportError:
        summary_md = summary_df.to_string(index=False)
        metrics_md = metrics_df.head().to_string(index=False)

    degenerate_note = (
        "Note: Some test folds contained only one class (no positives or no negatives). "
        "AUC metrics are undefined for those folds and are excluded from mean calculations; "
        "AUC averages are computed over valid folds only; if a model/run has 0 valid folds, mean AUC is reported as NaN. "
        "Delta PR-AUC values are computed only on folds that are valid for both halving and non-halving runs. "
        "Refer to prevalence columns for context."
    )
    lines = [
        "# Halving-ML Report",
        "",
        "## Summary",
        "",
        summary_md,
        "",
        "## Metrics by Fold",
        "",
        metrics_md,
        "",
        degenerate_note,
    ]
    (config.REPORT_PATH).write_text("\n".join(lines))


def plot_results(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, base_df: pd.DataFrame, folds: Iterable[Dict]):
    """Generate all plots for a pipeline run."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_volatility_with_halvings(base_df)
    plot_label_threshold_example(base_df, folds, metrics_df)
    plot_walk_forward_splits(folds, base_df)
    plot_pos_rate_by_fold(metrics_df)
    plot_pr_auc_by_fold(metrics_df)
    plot_metrics_summary_bar(summary_df)
    plot_ablation_delta(summary_df)


def save_report(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, *, silent: bool = False):
    """Persist report to disk."""
    write_report(summary_df, metrics_df)
    if not silent:
        print(f"Report saved to {config.REPORT_PATH}")


__all__ = [
    "plot_volatility_with_halvings",
    "plot_label_threshold_example",
    "plot_walk_forward_splits",
    "plot_pos_rate_by_fold",
    "plot_pr_auc_by_fold",
    "plot_metrics_summary_bar",
    "plot_ablation_delta",
    "plot_results",
    "write_report",
    "save_report",
]
