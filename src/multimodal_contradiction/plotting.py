"""Plotting and PDF export helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .io_utils import ensure_dir
from .schemas import LABELS


def save_pdf_figure(fig: plt.Figure, destination: Path) -> Path:
    ensure_dir(destination.parent)
    fig.savefig(destination, bbox_inches="tight", format="pdf")
    return destination


def plot_training_history(history: list[dict[str, float]], title: str) -> plt.Figure:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], marker="o", label="Macro F1")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], marker="o", label="Accuracy")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(confusion: np.ndarray | list[list[int]], title: str) -> plt.Figure:
    matrix = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep_frame: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep_frame["tau_low"], sweep_frame["macro_f1"], label="Macro F1")
    ax.set_xlabel("Lower Threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_group_metrics(frame: pd.DataFrame, *, group_column: str, metric: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=frame, x=group_column, y=metric, ax=ax, color="#4C72B0")
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_calibration_curve(probabilities: np.ndarray, labels: np.ndarray, title: str, n_bins: int = 10) -> plt.Figure:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    xs = []
    ys = []
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (confidences >= left) & (confidences < right if idx < n_bins - 1 else confidences <= right)
        if np.any(mask):
            xs.append(confidences[mask].mean())
            ys.append(correctness[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    ax.plot(xs, ys, marker="o", label="Observed")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_qualitative_panel(
    frame: pd.DataFrame,
    *,
    image_root_resolver,
    title: str,
    max_rows: int = 4,
) -> plt.Figure:
    sample_frame = frame.head(max_rows).reset_index(drop=True)
    fig, axes = plt.subplots(len(sample_frame), 2, figsize=(11, 3 * max(len(sample_frame), 1)))
    if len(sample_frame) == 1:
        axes = np.array([axes])

    for row_idx, row in sample_frame.iterrows():
        image = Image.open(image_root_resolver(row)).convert("RGB")
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(f"Image {row['image_id']}")

        text_lines = [
            f"Label: {row.get('label', 'n/a')}",
            f"Caption: {row.get('edited_caption', '')}",
        ]
        if "pred_label" in row:
            text_lines.append(f"Prediction: {row['pred_label']}")
        if "edit_family" in row:
            text_lines.append(f"Edit family: {row['edit_family']}")

        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].text(0.0, 1.0, "\n".join(text_lines), va="top", wrap=True)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
