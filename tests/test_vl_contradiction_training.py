from __future__ import annotations

import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.models import LinearProbe  # noqa: E402
from vl_contradiction.training import (  # noqa: E402
    FeatureDataset,
    TrainingTrialConfig,
    evaluate_model,
    get_stage_trials,
    run_training_sweep,
)


class TrainingConfigTests(unittest.TestCase):
    def test_stage_trial_resolution_uses_nested_stage_and_model_keys(self) -> None:
        training_config = SimpleNamespace(
            sweeps={
                "prototype": {
                    "linear_probe": {
                        "trials": [
                            {"name": "lp_p1", "epochs": 5, "batch_size": 16, "learning_rate": 0.0005, "weight_decay": 0.01},
                            {"name": "lp_p2", "epochs": 8, "batch_size": 16, "learning_rate": 0.0003, "weight_decay": 0.01},
                            {"name": "lp_p3", "epochs": 10, "batch_size": 32, "learning_rate": 0.0003, "weight_decay": 0.005},
                        ]
                    }
                },
                "midscale": {
                    "cross_attention": {
                        "trials": [
                            {"name": "ca_m1", "epochs": 8, "batch_size": 8, "learning_rate": 0.0005, "weight_decay": 0.01},
                            {"name": "ca_m2", "epochs": 12, "batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.01},
                            {"name": "ca_m3", "epochs": 12, "batch_size": 4, "learning_rate": 0.0001, "weight_decay": 0.01},
                            {"name": "ca_m4", "epochs": 16, "batch_size": 8, "learning_rate": 0.0001, "weight_decay": 0.005},
                            {"name": "ca_m5", "epochs": 20, "batch_size": 4, "learning_rate": 0.0003, "weight_decay": 0.001},
                        ]
                    }
                },
            }
        )

        prototype_trials = get_stage_trials(training_config, "prototype", "linear_probe")
        self.assertEqual(["lp_p1", "lp_p2", "lp_p3"], [trial.name for trial in prototype_trials])

        midscale_trials = get_stage_trials(training_config, "midscale", "cross_attention")
        self.assertEqual(["ca_m1", "ca_m2", "ca_m3", "ca_m4", "ca_m5"], [trial.name for trial in midscale_trials])

    def test_unknown_stage_or_model_raises_clear_error(self) -> None:
        training_config = SimpleNamespace(
            sweeps={
                "prototype": {
                    "linear_probe": {
                        "trials": [
                            {"name": "lp_p1", "epochs": 5, "batch_size": 16, "learning_rate": 0.0005, "weight_decay": 0.01}
                        ]
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "Unknown training stage"):
            get_stage_trials(training_config, "missing_stage", "linear_probe")

        with self.assertRaisesRegex(ValueError, "Unknown training model"):
            get_stage_trials(training_config, "prototype", "missing_model")


class TrainingSweepTests(unittest.TestCase):
    def test_evaluate_model_returns_float32_logits_for_numpy_safe_reporting(self) -> None:
        class BF16ToyModel(torch.nn.Module):
            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.stack((features[:, 0], -features[:, 0]), dim=1).to(torch.bfloat16)

        features = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        loader = torch.utils.data.DataLoader(FeatureDataset(features, labels), batch_size=2, shuffle=False)

        metrics, logits, returned_labels = evaluate_model(BF16ToyModel(), loader, torch.device("cpu"), amp=False)

        self.assertEqual(torch.float32, logits.dtype)
        self.assertEqual("cpu", logits.device.type)
        self.assertEqual((2, 2), tuple(logits.shape))
        self.assertEqual((2,), tuple(returned_labels.shape))
        self.assertGreaterEqual(float(metrics["accuracy"]), 1.0)
        self.assertEqual((2, 2), tuple(torch.softmax(logits, dim=1).numpy().shape))

    def test_run_training_sweep_selects_best_trial_and_writes_checkpoints(self) -> None:
        torch.manual_seed(0)

        train_features = torch.tensor(
            [
                [-2.0, -2.0],
                [-2.2, -1.8],
                [-1.8, -2.1],
                [2.0, -2.0],
                [1.8, -2.2],
                [2.2, -1.9],
            ],
            dtype=torch.float32,
        )
        train_labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        val_features = torch.tensor(
            [
                [-2.1, -2.1],
                [2.1, -2.1],
            ],
            dtype=torch.float32,
        )
        val_labels = torch.tensor([0, 1], dtype=torch.long)

        test_features = torch.tensor(
            [
                [-1.9, -2.0],
                [2.0, -1.8],
            ],
            dtype=torch.float32,
        )
        test_labels = torch.tensor([0, 1], dtype=torch.long)

        trials = [
            TrainingTrialConfig(name="weak", epochs=1, batch_size=4, learning_rate=1e-6, weight_decay=0.01),
            TrainingTrialConfig(name="strong", epochs=60, batch_size=3, learning_rate=0.1, weight_decay=0.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = run_training_sweep(
                model_name="linear_probe",
                model_factory=lambda: LinearProbe(input_dim=2),
                train_dataset=FeatureDataset(train_features, train_labels),
                val_dataset=FeatureDataset(val_features, val_labels),
                test_dataset=FeatureDataset(test_features, test_labels),
                trials=trials,
                device=torch.device("cpu"),
                num_workers=0,
                selection_metric="macro_f1",
                log_root=root / "logs",
                checkpoint_root=root / "checkpoints",
                canonical_checkpoint_path=root / "checkpoints" / "linear_probe_best.pt",
            )

            self.assertEqual(2, len(result.trial_rows))
            self.assertEqual("strong", result.best_trial.name)
            self.assertGreaterEqual(float(result.best_trial_row["val_macro_f1"]), 0.95)
            self.assertTrue((root / "checkpoints" / "linear_probe__weak.pt").exists())
            self.assertTrue((root / "checkpoints" / "linear_probe__strong.pt").exists())
            self.assertTrue((root / "checkpoints" / "linear_probe_best.pt").exists())
            self.assertEqual(root / "checkpoints" / "linear_probe_best.pt", result.best_checkpoint)
            self.assertGreaterEqual(float(result.best_test_metrics["macro_f1"]), 0.95)
            self.assertEqual((2, 2), tuple(result.best_test_logits.shape))
            self.assertEqual(torch.float32, result.best_test_logits.dtype)


if __name__ == "__main__":
    unittest.main()
