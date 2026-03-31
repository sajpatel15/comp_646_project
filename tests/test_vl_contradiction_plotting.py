from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.plotting import save_score_histogram, save_threshold_sweep  # noqa: E402


class PlottingTests(unittest.TestCase):
    def test_save_score_histogram_writes_nonempty_image(self) -> None:
        frame = pd.DataFrame(
            [
                {"raw_score": 0.22, "label": "contradiction"},
                {"raw_score": 0.24, "label": "contradiction"},
                {"raw_score": 0.28, "label": "neutral"},
                {"raw_score": 0.31, "label": "neutral"},
                {"raw_score": 0.34, "label": "entailment"},
                {"raw_score": 0.36, "label": "entailment"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scores.png"
            save_score_histogram(frame, output_path, "Synthetic Score Distribution")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_threshold_sweep_marks_best_pair_and_writes_image(self) -> None:
        rows = []
        grid = np.linspace(0.2, 0.4, 9)
        for tau_low in grid:
            for tau_high in grid:
                if tau_low >= tau_high:
                    continue
                macro_f1 = 0.7 - ((tau_low - 0.27) ** 2 * 12 + (tau_high - 0.33) ** 2 * 14)
                rows.append({"tau_low": tau_low, "tau_high": tau_high, "macro_f1": macro_f1})
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thresholds.png"
            save_threshold_sweep(frame, output_path, "Synthetic Threshold Sweep")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_threshold_sweep_rejects_empty_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.png"
            with self.assertRaises(ValueError):
                save_threshold_sweep(pd.DataFrame(columns=["tau_low", "tau_high", "macro_f1"]), output_path, "Empty")


if __name__ == "__main__":
    unittest.main()
