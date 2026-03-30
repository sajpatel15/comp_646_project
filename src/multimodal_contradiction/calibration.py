from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        accuracy = np.mean(predictions[mask] == labels[mask])
        confidence = np.mean(confidences[mask])
        error += np.abs(accuracy - confidence) * np.mean(mask)
    return float(error)
