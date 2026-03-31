"""Plot helpers for report-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


sns.set_theme(style="whitegrid", context="talk")


LABEL_ORDER = ["contradiction", "neutral", "entailment"]
LABEL_PALETTE = {
    "contradiction": "#c44e52",
    "neutral": "#dd8452",
    "entailment": "#4c72b0",
}


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_training_curves(history: list[dict[str, float]], output_path: str | Path, title: str) -> None:
    frame = pd.DataFrame(history)
    output = _prepare_output(output_path)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(frame["epoch"], frame["train_loss"], label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(frame["epoch"], frame["val_macro_f1"], color="tab:orange", label="Val Macro-F1", linewidth=2)
    ax2.set_ylabel("Macro-F1")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram(score_frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    label_order = [label for label in LABEL_ORDER if label in set(score_frame["label"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(
        score_frame,
        x="raw_score",
        hue="label",
        hue_order=label_order,
        palette=LABEL_PALETTE,
        bins=24,
        element="step",
        fill=True,
        alpha=0.18,
        linewidth=1.8,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    if ax.legend_ is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), title="Label", frameon=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep(search_frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    if search_frame.empty:
        raise ValueError("search_frame must contain threshold sweep results.")

    output = _prepare_output(output_path)
    best_row = search_frame.loc[search_frame["macro_f1"].idxmax()]
    score_min = float(search_frame[["tau_low", "tau_high"]].min().min())
    score_max = float(search_frame[["tau_low", "tau_high"]].max().max())

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(
        search_frame["tau_low"],
        search_frame["tau_high"],
        c=search_frame["macro_f1"],
        cmap="viridis",
        s=24,
        edgecolors="none",
        alpha=0.9,
        rasterized=len(search_frame) > 5_000,
    )
    ax.plot(
        [score_min, score_max],
        [score_min, score_max],
        linestyle="--",
        color="black",
        linewidth=1.1,
        alpha=0.55,
    )
    ax.scatter(
        best_row["tau_low"],
        best_row["tau_high"],
        marker="*",
        s=260,
        color="#d62728",
        edgecolors="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.annotate(
        (
            "Best pair\n"
            f"tau_low={best_row['tau_low']:.4f}\n"
            f"tau_high={best_row['tau_high']:.4f}\n"
            f"macro-F1={best_row['macro_f1']:.3f}"
        ),
        xy=(best_row["tau_low"], best_row["tau_high"]),
        xytext=(0.04, 0.96),
        textcoords="axes fraction",
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#555555", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.2},
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Validation Macro-F1")
    ax.set_xlabel("tau_low")
    ax.set_ylabel("tau_high")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(frame: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(frame, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_reliability_diagram(bin_centers: np.ndarray, bin_accuracy: np.ndarray, bin_confidence: np.ndarray, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(bin_confidence, bin_accuracy, marker="o", linewidth=2)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_panel(frame: pd.DataFrame, output_path: str | Path, title: str, max_rows: int = 4) -> None:
    output = _prepare_output(output_path)
    subset = frame.head(max_rows).reset_index(drop=True)
    fig, axes = plt.subplots(len(subset), 2, figsize=(10, 4 * max(len(subset), 1)))
    if len(subset) == 1:
        axes = np.array([axes])
    for axis_row, (_, row) in zip(axes, subset.iterrows(), strict=True):
        image = Image.open(row["file_path"]).convert("RGB")
        axis_row[0].imshow(image)
        axis_row[0].axis("off")
        axis_row[1].axis("off")
        axis_row[1].text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"Label: {row['label']}",
                    f"Source: {row['source_caption']}",
                    f"Edited: {row['edited_caption']}",
                ]
            ),
            va="top",
            fontsize=10,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
