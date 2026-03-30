"""Metrics, threshold fitting, and calibration helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .schemas import LABELS, ThresholdConfig


def predict_from_thresholds(scores: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    predictions = np.full_like(scores, fill_value=2, dtype=int)
    predictions[scores < tau_low] = 0
    predictions[(scores >= tau_low) & (scores < tau_high)] = 1
    return predictions


def fit_score_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    grid_size: int = 200,
) -> ThresholdConfig:
    score_min, score_max = float(scores.min()), float(scores.max())
    grid = np.linspace(score_min, score_max, grid_size)
    best = (-1.0, ThresholdConfig(tau_low=score_min, tau_high=score_max))

    for tau_low in grid[:-1]:
        for tau_high in grid[1:]:
            if tau_low >= tau_high:
                continue
            predicted = predict_from_thresholds(scores, tau_low, tau_high)
            macro_f1 = f1_score(labels, predicted, average="macro")
            if macro_f1 > best[0]:
                best = (macro_f1, ThresholdConfig(tau_low=float(tau_low), tau_high=float(tau_high)))
    return best[1]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(LABELS)),
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": [],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS))).tolist(),
    }
    for label_idx, label_name in enumerate(LABELS):
        metrics["per_class"].append(
            {
                "label": label_name,
                "precision": float(precision[label_idx]),
                "recall": float(recall[label_idx]),
                "f1": float(f1[label_idx]),
                "support": int(support[label_idx]),
            }
        )
    return metrics


def metrics_frame_by_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    y_true_column: str = "label_id",
    y_pred_column: str = "pred_label_id",
) -> pd.DataFrame:
    rows = []
    for group_value, group_frame in frame.groupby(group_column):
        metrics = classification_metrics(
            group_frame[y_true_column].to_numpy(),
            group_frame[y_pred_column].to_numpy(),
        )
        rows.append(
            {
                group_column: group_value,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "support": int(len(group_frame)),
            }
        )
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        estimates.append(metric_fn(y_true[indices], y_pred[indices]))
    lower = float(np.quantile(estimates, alpha / 2))
    upper = float(np.quantile(estimates, 1 - alpha / 2))
    return {
        "mean": float(np.mean(estimates)),
        "lower": lower,
        "upper": upper,
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        mask = (confidences >= left) & (confidences < right if idx < n_bins - 1 else confidences <= right)
        if not np.any(mask):
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(correctness[mask].mean())
        ece += (mask.mean()) * abs(avg_confidence - avg_accuracy)
    return float(ece)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def label_to_id(label: str) -> int:
    return LABELS.index(label)


def threshold_config_to_dict(config: ThresholdConfig) -> dict[str, float | str]:
    return asdict(config)
