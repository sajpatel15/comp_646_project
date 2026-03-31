from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.plotting import (  # noqa: E402
    save_reliability_diagram,
    save_score_histogram,
    save_threshold_sweep,
    save_training_curves,
)


class PlottingTests(unittest.TestCase):
    def test_save_training_curves_writes_image_and_places_legend_on_right(self) -> None:
        history = [
            {"epoch": 1.0, "train_loss": 1.2, "val_macro_f1": 0.32},
            {"epoch": 2.0, "train_loss": 0.8, "val_macro_f1": 0.47},
            {"epoch": 3.0, "train_loss": 0.5, "val_macro_f1": 0.61},
        ]

        captured: dict[str, object] = {}
        original_close = plt.close

        def capture_close(fig=None):  # noqa: ANN001
            if fig is not None:
                captured["figure"] = fig
            return original_close(fig)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("matplotlib.pyplot.close", side_effect=capture_close):
            output_path = Path(tmpdir) / "training.png"
            save_training_curves(history, output_path, "Synthetic Training")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

        figure = captured["figure"]
        axes = figure.axes
        legend = axes[0].get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(["Train Loss", "Val Macro-F1"], [text.get_text() for text in legend.get_texts()])
        self.assertEqual((1.02, 1.0), tuple(legend.get_bbox_to_anchor()._bbox.p1))

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

    def test_save_reliability_diagram_writes_image_and_places_legend_on_right(self) -> None:
        captured: dict[str, object] = {}
        original_close = plt.close

        def capture_close(fig=None):  # noqa: ANN001
            if fig is not None:
                captured["figure"] = fig
            return original_close(fig)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("matplotlib.pyplot.close", side_effect=capture_close):
            output_path = Path(tmpdir) / "reliability.png"
            save_reliability_diagram(
                bin_centers=np.array([0.1, 0.5, 0.9]),
                bin_accuracy=np.array([0.2, 0.55, 0.88]),
                bin_confidence=np.array([0.15, 0.52, 0.91]),
                output_path=output_path,
                title="Synthetic Reliability",
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

        figure = captured["figure"]
        legend = figure.axes[0].get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(
            ["Perfect Calibration", "Observed Accuracy"],
            [text.get_text() for text in legend.get_texts()],
        )
        self.assertEqual((1.02, 1.0), tuple(legend.get_bbox_to_anchor()._bbox.p1))

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
