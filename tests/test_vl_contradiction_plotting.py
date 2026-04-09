from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.plotting import (  # noqa: E402
    save_grouped_comparison_chart,
    save_qualitative_panel,
    save_per_family_accuracy_heatmap,
    save_reliability_diagram,
    save_score_histogram,
    save_threshold_sweep,
    save_training_curves,
)
from vl_contradiction.reporting import (  # noqa: E402
    build_model_comparison_per_family,
    build_model_comparison_summary,
    save_comparison_tables,
    save_prediction_export,
    select_matched_qualitative_samples,
    standardize_prediction_frame,
)


class PlottingTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :] = np.array(color, dtype=np.uint8)
        Image.fromarray(array).save(path)

    def test_standardize_prediction_frame_and_save_prediction_export(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "sample-0",
                    "label": "contradiction",
                    "pred_label": "contradiction",
                    "edit_family": "family-a",
                    "file_path": "a.jpg",
                    "source_caption": "source a",
                    "edited_caption": "edited a",
                    "confidence": 0.91,
                    "raw_score": 1.2,
                    "rationale": "match",
                    "extra_field": "keep-me",
                },
                {
                    "sample_id": "sample-1",
                    "label": "entailment",
                    "pred_label": "contradiction",
                    "edit_family": "family-b",
                    "file_path": "b.jpg",
                    "source_caption": "source b",
                    "edited_caption": "edited b",
                },
            ]
        )

        standardized = standardize_prediction_frame(
            frame,
            model="clip",
            stage="final",
            eval_scope="comparison_subset",
        )
        self.assertEqual(
            [
                "sample_id",
                "model",
                "stage",
                "eval_scope",
                "label",
                "pred_label",
                "correct",
                "edit_family",
                "file_path",
                "source_caption",
                "edited_caption",
                "confidence",
                "raw_score",
                "rationale",
                "extra_field",
            ],
            standardized.columns.tolist(),
        )
        self.assertEqual(["clip", "clip"], standardized["model"].tolist())
        self.assertEqual(["final", "final"], standardized["stage"].tolist())
        self.assertEqual(["comparison_subset", "comparison_subset"], standardized["eval_scope"].tolist())
        self.assertEqual([True, False], standardized["correct"].tolist())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.csv"
            saved = save_prediction_export(
                frame,
                output_path,
                model="clip",
                stage="final",
                eval_scope="comparison_subset",
            )
            self.assertTrue(output_path.exists())
            reloaded = pd.read_csv(output_path)
            self.assertEqual(["sample-0", "sample-1"], reloaded["sample_id"].tolist())
            self.assertEqual(saved.columns.tolist(), standardized.columns.tolist())

    def test_select_matched_qualitative_samples_returns_shared_sample_ids(self) -> None:
        clip = pd.DataFrame(
            [
                {"sample_id": "s0", "label": "contradiction", "pred_label": "contradiction"},
                {"sample_id": "s1", "label": "entailment", "pred_label": "entailment"},
                {"sample_id": "s2", "label": "entailment", "pred_label": "contradiction"},
            ]
        )
        probe = pd.DataFrame(
            [
                {"sample_id": "s0", "label": "contradiction", "pred_label": "contradiction"},
                {"sample_id": "s1", "label": "entailment", "pred_label": "entailment"},
                {"sample_id": "s2", "label": "entailment", "pred_label": "entailment"},
            ]
        )

        selection = select_matched_qualitative_samples(
            {"clip": clip, "probe": probe},
            correct_count=2,
            failure_count=2,
            seed=7,
        )

        self.assertEqual(2, len(selection.correct_sample_ids))
        self.assertEqual(2, len(selection.failure_sample_ids))
        self.assertIn("s2", selection.failure_sample_ids)
        self.assertTrue(set(selection.correct_sample_ids).issubset({"s0", "s1"}))
        self.assertTrue(set(selection.failure_sample_ids).issubset({"s0", "s1", "s2"}))
        self.assertEqual(["correct", "correct", "failure", "failure"], selection.manifest["pool"].tolist())

    def test_save_training_curves_writes_image_and_places_legend_below_plot(self) -> None:
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
        legend_axes = [axis for axis in axes if axis.get_legend() is not None]
        self.assertEqual(1, len(legend_axes))
        legend_axis = legend_axes[0]
        legend = legend_axis.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(["Train Loss", "Val Macro-F1"], [text.get_text() for text in legend.get_texts()])
        self.assertFalse(legend_axis.axison)
        self.assertLess(legend_axis.get_position().y1, axes[0].get_position().y0)

    def test_save_score_histogram_writes_nonempty_image(self) -> None:
        frame = pd.DataFrame(
            [
                {"raw_score": 0.22, "label": "contradiction"},
                {"raw_score": 0.24, "label": "contradiction"},
                {"raw_score": 0.34, "label": "entailment"},
                {"raw_score": 0.36, "label": "entailment"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scores.png"
            save_score_histogram(frame, output_path, "Synthetic Score Distribution")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_qualitative_panel_pads_thumbnails_and_renders_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_a = tmpdir_path / "a.png"
            image_b = tmpdir_path / "b.png"
            self._write_image(image_a, 180, 260, (255, 0, 0))
            self._write_image(image_b, 420, 160, (0, 128, 255))

            frame = pd.DataFrame(
                [
                    {
                        "sample_id": "sample-0",
                        "model": "clip",
                        "stage": "final",
                        "eval_scope": "comparison_subset",
                        "label": "contradiction",
                        "pred_label": "contradiction",
                        "correct": True,
                        "edit_family": "family-a",
                        "confidence": 0.91,
                        "raw_score": 1.22,
                        "rationale": "The model matched the contradiction target.",
                        "source_caption": "Source caption 0 with enough detail to wrap cleanly.",
                        "edited_caption": "Edited caption 0 with enough detail to wrap cleanly.",
                        "file_path": str(image_a),
                    },
                    {
                        "sample_id": "sample-1",
                        "model": "clip",
                        "stage": "final",
                        "eval_scope": "comparison_subset",
                        "label": "entailment",
                        "pred_label": "contradiction",
                        "correct": False,
                        "edit_family": "family-b",
                        "confidence": 0.27,
                        "raw_score": -0.44,
                        "rationale": "The model missed the entailment cue.",
                        "source_caption": "Source caption 1 with enough detail to wrap cleanly.",
                        "edited_caption": "Edited caption 1 with enough detail to wrap cleanly.",
                        "file_path": str(image_b),
                    },
                ]
            )

            pad_sizes: list[tuple[int, int]] = []
            captured: dict[str, object] = {}
            original_close = plt.close
            original_contain = ImageOps.contain

            def capture_contain(image, size, *args, **kwargs):  # noqa: ANN001
                pad_sizes.append(size)
                return original_contain(image, size, *args, **kwargs)

            def capture_close(fig=None):  # noqa: ANN001
                if fig is not None:
                    captured["figure"] = fig
                return original_close(fig)

            with mock.patch("vl_contradiction.plotting.ImageOps.contain", side_effect=capture_contain), mock.patch(
                "matplotlib.pyplot.close",
                side_effect=capture_close,
            ):
                output_path = tmpdir_path / "panel.png"
                save_qualitative_panel(frame, output_path, "Synthetic Panel", max_rows=2)
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

            self.assertEqual([(320, 320), (320, 320)], pad_sizes)
            figure = captured["figure"]
            text_content = "\n".join(text.get_text() for axis in figure.axes for text in axis.texts)
            self.assertIn("True:", text_content)
            self.assertIn("Pred:", text_content)
            self.assertIn("Correct:", text_content)
            self.assertIn("Source:", text_content)
            self.assertIn("Edited:", text_content)

    def test_save_grouped_comparison_chart_writes_nonempty_image(self) -> None:
        frame = pd.DataFrame(
            [
                {"model": "clip", "accuracy": 0.65, "macro_f1": 0.61},
                {"model": "probe", "accuracy": 0.72, "macro_f1": 0.69},
                {"model": "cross", "accuracy": 0.74, "macro_f1": 0.70},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.png"
            save_grouped_comparison_chart(frame, output_path, "Synthetic Comparison")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_per_family_accuracy_heatmap_writes_nonempty_image(self) -> None:
        frame = pd.DataFrame(
            [
                {"model": "clip", "edit_family": "family-a", "accuracy": 0.5},
                {"model": "probe", "edit_family": "family-a", "accuracy": 0.7},
                {"model": "clip", "edit_family": "family-b", "accuracy": 0.8},
                {"model": "probe", "edit_family": "family-b", "accuracy": 0.6},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap.png"
            save_per_family_accuracy_heatmap(frame, output_path, "Synthetic Heatmap")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_comparison_tables_writes_csv_tables(self) -> None:
        clip = pd.DataFrame(
            [
                {
                    "sample_id": "s0",
                    "label": "contradiction",
                    "pred_label": "contradiction",
                    "edit_family": "family-a",
                },
                {
                    "sample_id": "s1",
                    "label": "entailment",
                    "pred_label": "contradiction",
                    "edit_family": "family-b",
                },
            ]
        )
        probe = pd.DataFrame(
            [
                {
                    "sample_id": "s0",
                    "label": "contradiction",
                    "pred_label": "contradiction",
                    "edit_family": "family-a",
                },
                {
                    "sample_id": "s1",
                    "label": "entailment",
                    "pred_label": "entailment",
                    "edit_family": "family-b",
                },
            ]
        )
        summary = build_model_comparison_summary({"clip": clip, "probe": probe})
        family = build_model_comparison_per_family({"clip": clip, "probe": probe})
        self.assertEqual(["model", "sample_count", "stage", "eval_scope", "accuracy", "macro_f1", "precision_contradiction", "recall_contradiction", "precision_entailment", "recall_entailment"], summary.columns.tolist())
        self.assertEqual(["model", "edit_family", "count", "stage", "eval_scope", "accuracy", "macro_f1", "precision_contradiction", "recall_contradiction", "precision_entailment", "recall_entailment"], family.columns.tolist())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.csv"
            family_path = tmpdir_path / "family.csv"
            saved_summary, saved_family = save_comparison_tables(
                {"clip": clip, "probe": probe},
                summary_path,
                family_path,
            )
            self.assertTrue(summary_path.exists())
            self.assertTrue(family_path.exists())
            self.assertEqual(summary.shape, saved_summary.shape)
            self.assertEqual(family.shape, saved_family.shape)

    def test_save_reliability_diagram_writes_image_and_places_legend_below_plot(self) -> None:
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
        legend_axes = [axis for axis in figure.axes if axis.get_legend() is not None]
        self.assertEqual(1, len(legend_axes))
        legend_axis = legend_axes[0]
        legend = legend_axis.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(
            ["Perfect Calibration", "Observed Accuracy"],
            [text.get_text() for text in legend.get_texts()],
        )
        self.assertFalse(legend_axis.axison)
        self.assertLess(legend_axis.get_position().y1, figure.axes[0].get_position().y0)

    def test_save_threshold_sweep_marks_best_pair_and_writes_image(self) -> None:
        frame = pd.DataFrame(
            [
                {"tau": 0.20, "macro_f1": 0.51, "accuracy": 0.60},
                {"tau": 0.25, "macro_f1": 0.64, "accuracy": 0.68},
                {"tau": 0.30, "macro_f1": 0.72, "accuracy": 0.74},
                {"tau": 0.35, "macro_f1": 0.69, "accuracy": 0.71},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thresholds.png"
            save_threshold_sweep(frame, output_path, "Synthetic Threshold Sweep")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_threshold_sweep_rejects_empty_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.png"
            with self.assertRaises(ValueError):
                save_threshold_sweep(pd.DataFrame(columns=["tau", "macro_f1"]), output_path, "Empty")


if __name__ == "__main__":
    unittest.main()
