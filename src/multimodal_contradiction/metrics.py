from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABEL_TO_INDEX = {"contradiction": 0, "neutral": 1, "entailment": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def compute_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
) -> dict[str, float | list[list[int]]]:
    labels = list(LABEL_TO_INDEX.keys())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_per_class": {label: float(value) for label, value in zip(labels, precision)},
        "recall_per_class": {label: float(value) for label, value in zip(labels, recall)},
        "f1_per_class": {label: float(value) for label, value in zip(labels, f1)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_macro_f1(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        score = f1_score(
            y_true[indices],
            y_pred[indices],
            labels=list(LABEL_TO_INDEX.keys()),
            average="macro",
            zero_division=0,
        )
        scores.append(score)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_ci_lower": float(lower),
        "macro_f1_ci_upper": float(upper),
    }


def per_edit_family_metrics(frame: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    rows = []
    for edit_family, group in frame.groupby("edit_family"):
        metrics = compute_classification_metrics(group["label"].tolist(), group[pred_col].tolist())
        rows.append(
            {
                "edit_family": edit_family,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "count": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("edit_family").reset_index(drop=True)
