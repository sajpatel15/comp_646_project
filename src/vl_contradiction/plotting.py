"""Plot helpers for report-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


sns.set_theme(style="whitegrid", context="talk")


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
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(score_frame, x="raw_score", hue="label", bins=30, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep(search_frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    pivot = search_frame.pivot_table(index="tau_low", columns="tau_high", values="macro_f1")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, cmap="magma", ax=ax)
    ax.set_xlabel("tau_high")
    ax.set_ylabel("tau_low")
    ax.set_title(title)
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
