"""Shared label definitions for the active binary benchmark."""

from __future__ import annotations

import numpy as np


CLASS_ORDER = ["contradiction", "entailment"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
VALID_LABELS = frozenset(CLASS_ORDER)
NUM_CLASSES = len(CLASS_ORDER)


def encode_labels(labels: list[str] | tuple[str, ...] | np.ndarray) -> np.ndarray:
    """Convert string labels into integer class indices."""

    return np.array([LABEL_TO_INDEX[str(label)] for label in labels], dtype=np.int64)


def decode_predictions(predictions: list[int] | tuple[int, ...] | np.ndarray) -> list[str]:
    """Convert integer class indices into string labels."""

    return [INDEX_TO_LABEL[int(index)] for index in predictions]
