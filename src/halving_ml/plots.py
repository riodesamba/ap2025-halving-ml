"""Plotting and reporting utilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import config, data, features, labels


plt.style.use(config.PLOT_STYLE["style"])


def plot_volatility_with_halvings():
    df = labels.add_labels(features.build_features(data.load_data(), include_halving=True))
    df = df.dropna()
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    ax.plot(df["Date"], df["volatility"], label="Volatility (30d annualized)")
    for date in config.HALVING_DATES:
        ax.axvline(date, color="red", linestyle="--", alpha=0.6, label="Halving" if date == config.HALVING_DATES[0] else None)
    ax.set_title("BTC Volatility with Halving Dates")
    ax.legend()
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "vol_with_halvings.png")
    plt.close(fig)


def plot_pr_auc_by_fold(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    degenerate_present = False
    subset = metrics_df[metrics_df["model"].isin(["logreg", "random_forest", "xgboost"])]
    for model, group in subset.groupby("model"):
        group = group.copy()
        group = group.sort_values("fold")
        valid = group["pr_auc"].notna()
        ax.plot(group.loc[valid, "fold"], group.loc[valid, "pr_auc"], marker="o", label=model)
        if (~valid).any():
            degenerate_present = True
    ax.set_xlabel("Fold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC by Fold")
    handles, labels = ax.get_legend_handles_labels()
    if degenerate_present:
        handles.append(plt.Line2D([], [], color="grey", marker="x", linestyle="None"))
        labels.append("degenerate fold")
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "pr_auc_by_fold.png")
    plt.close(fig)


def plot_ablation_delta(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=config.PLOT_STYLE["figsize"])
    subset = summary_df.drop_duplicates("model")
    ax.bar(subset["model"], subset["delta_pr_auc"], color="steelblue")
    ax.set_ylabel("PR-AUC delta (with - without)")
    ax.set_title("Impact of Halving Features")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "ablation_delta.png")
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


__all__ = ["plot_volatility_with_halvings", "plot_pr_auc_by_fold", "plot_ablation_delta", "write_report"]
