"""Metrics and calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .labels import CLASS_ORDER, LABEL_TO_INDEX


@dataclass(slots=True)
class CalibrationResult:
    temperature: float
    ece: float
    bin_centers: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    """Compute the main classification metrics used in the report."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_contradiction": float(precision[0]),
        "precision_entailment": float(precision[1]),
        "recall_contradiction": float(recall[0]),
        "recall_entailment": float(recall[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER)))).tolist(),
    }


def bootstrap_macro_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, samples: int = 500, seed: int = 42) -> tuple[float, float]:
    """Estimate a simple bootstrap confidence interval for macro-F1."""

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(f1_score(y_true[indices], y_pred[indices], average="macro"))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, steps: int = 50) -> float:
    """Fit a scalar temperature on validation logits."""

    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=steps)
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp_min(1e-4), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-4).cpu().item())


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    """Compute expected calibration error for multiclass predictions."""

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracy = []
    bin_confidence = []
    weights = []
    for start, end in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= start) & (confidences < end if end < 1.0 else confidences <= end)
        if mask.sum() == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append((start + end) / 2.0)
            weights.append(0.0)
            continue
        bin_accuracy.append(float(correctness[mask].mean()))
        bin_confidence.append(float(confidences[mask].mean()))
        weights.append(float(mask.mean()))
    bin_accuracy_array = np.array(bin_accuracy)
    bin_confidence_array = np.array(bin_confidence)
    ece = np.sum(np.abs(bin_accuracy_array - bin_confidence_array) * np.array(weights))
    centers = (bins[:-1] + bins[1:]) / 2.0
    return CalibrationResult(
        temperature=1.0,
        ece=float(ece),
        bin_centers=centers,
        bin_accuracy=bin_accuracy_array,
        bin_confidence=bin_confidence_array,
    )


def per_edit_family_metrics(frame: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    """Compute accuracy and macro-F1 for each edit family in a labeled frame."""

    rows: list[dict[str, float | int | str]] = []
    for edit_family, group in frame.groupby("edit_family"):
        y_true = group["label"].map(LABEL_TO_INDEX).to_numpy()
        y_pred = group[pred_col].map(LABEL_TO_INDEX).to_numpy()
        metrics = compute_classification_metrics(y_true, y_pred)
        rows.append(
            {
                "edit_family": edit_family,
                "count": int(len(group)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    columns = ["edit_family", "count", "accuracy", "macro_f1"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("edit_family").reset_index(drop=True)
