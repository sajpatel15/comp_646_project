from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _finalize_figure(output_path: str | Path | None = None) -> None:
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")


def plot_confusion_matrix(confusion: list[list[int]], labels: list[str], output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    _finalize_figure(output_path)


def plot_score_histograms(frame: pd.DataFrame, score_col: str = "raw_score", output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=frame, x=score_col, hue="label", bins=30, stat="density", common_norm=False, element="step")
    plt.xlabel("Similarity score")
    plt.ylabel("Density")
    plt.title("Score Distributions by Label")
    _finalize_figure(output_path)


def plot_threshold_sweep(sweep_frame: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=sweep_frame, x="threshold_id", y="macro_f1")
    plt.xlabel("Threshold setting")
    plt.ylabel("Validation macro-F1")
    plt.title("Threshold Search")
    _finalize_figure(output_path)


def plot_per_edit_family(metrics_frame: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=metrics_frame, x="edit_family", y="macro_f1", color="#1f77b4")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Edit family")
    plt.ylabel("Macro-F1")
    plt.title("Performance by Edit Family")
    _finalize_figure(output_path)


def plot_training_history(history_frame: pd.DataFrame, output_path: str | Path | None = None) -> None:
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=history_frame, x="epoch", y="train_loss", label="train_loss")
    if "val_loss" in history_frame:
        sns.lineplot(data=history_frame, x="epoch", y="val_loss", label="val_loss")
    if "val_macro_f1" in history_frame:
        sns.lineplot(data=history_frame, x="epoch", y="val_macro_f1", label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training History")
    _finalize_figure(output_path)


def plot_reliability_diagram(
    probabilities,
    labels,
    output_path: str | Path | None = None,
    n_bins: int = 10,
) -> None:
    import numpy as np

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accuracies = []
    avg_confidences = []

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        accuracies.append(np.mean(predictions[mask] == labels[mask]))
        avg_confidences.append(np.mean(confidences[mask]))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(avg_confidences, accuracies, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    _finalize_figure(output_path)
