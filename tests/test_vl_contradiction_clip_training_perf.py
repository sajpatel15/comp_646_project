from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.clip_baselines import (  # noqa: E402
    ClipBundle,
    compute_similarity_scores,
    extract_joint_features,
    extract_token_features,
)
from vl_contradiction.models import LinearProbe  # noqa: E402
from vl_contradiction.training import FeatureDataset, create_loader, train_model  # noqa: E402
from vl_contradiction.training import _resolve_amp_settings  # noqa: E402


class _FakeInputs(dict):
    def to(self, device: torch.device, non_blocking: bool = False):  # noqa: ARG002
        return self


class _FakeProcessor:
    def __call__(self, *, text, images, padding, return_tensors, truncation=None, max_length=None):  # noqa: ARG002
        values = []
        for caption in text:
            digits = "".join(ch for ch in caption if ch.isdigit())
            values.append(float(digits or 0))
        batch = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        return _FakeInputs(input_ids=batch, pixel_values=batch.clone())


class _FakeCLIPModel:
    def __init__(self) -> None:
        self.name_or_path = "fake-clip"
        self.forward_calls = 0
        self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=8))

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids, pixel_values, output_hidden_states, return_dict):  # noqa: ARG002
        self.forward_calls += 1
        sample_ids = input_ids.squeeze(1)
        image_embeds = torch.stack([sample_ids + 1.0, sample_ids + 2.0], dim=1)
        text_embeds = torch.stack([sample_ids + 2.0, sample_ids + 3.0], dim=1)
        image_hidden = torch.stack([sample_ids, sample_ids + 0.5], dim=1).unsqueeze(1).repeat(1, 2, 1)
        text_hidden = torch.stack([sample_ids + 1.0, sample_ids + 1.5], dim=1).unsqueeze(1).repeat(1, 2, 1)
        return SimpleNamespace(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            vision_model_output=SimpleNamespace(last_hidden_state=image_hidden),
            text_model_output=SimpleNamespace(last_hidden_state=text_hidden),
        )


class ClipExtractionPerfTests(unittest.TestCase):
    def test_shared_split_extraction_reuses_forward_passes_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_paths = []
            for index in range(3):
                image_path = root / f"image_{index}.png"
                Image.new("RGB", (4, 4), color=(index * 20, index * 20, index * 20)).save(image_path)
                image_paths.append(image_path)

            records = pd.DataFrame(
                {
                    "sample_id": ["s2", "s0", "s1"],
                    "file_path": [str(path) for path in image_paths],
                    "edited_caption": ["caption 2", "caption 0", "caption 1"],
                    "label": ["contradiction", "entailment", "contradiction"],
                }
            )
            bundle = ClipBundle(
                model=_FakeCLIPModel(),
                processor=_FakeProcessor(),
                device=torch.device("cpu"),
                precision="fp32",
                autocast_dtype=None,
                num_workers=0,
                persistent_workers=False,
                prefetch_factor=None,
            )

            scores = compute_similarity_scores(records, bundle, batch_size=2)
            joint_features, joint_labels = extract_joint_features(records, bundle, batch_size=2)
            image_tokens, text_tokens, token_labels = extract_token_features(records, bundle, batch_size=2)

            self.assertEqual(["s2", "s0", "s1"], scores["sample_id"].tolist())
            self.assertEqual(["contradiction", "entailment", "contradiction"], scores["label"].tolist())
            self.assertEqual((3,), tuple(scores["raw_score"].shape))
            self.assertEqual((3, 3), tuple(scores.shape))
            self.assertEqual((3, 5), tuple(joint_features.shape))
            self.assertEqual((3,), tuple(joint_labels.shape))
            self.assertEqual((3, 2, 2), tuple(image_tokens.shape))
            self.assertEqual((3, 2, 2), tuple(text_tokens.shape))
            self.assertTrue(torch.equal(joint_labels, token_labels))
            self.assertEqual(2, bundle.model.forward_calls)


class TrainingPerfTests(unittest.TestCase):
    def test_explicit_amp_precision_uses_requested_dtype_without_probe(self) -> None:
        with patch(
            "vl_contradiction.training._cuda_bf16_supported",
            side_effect=AssertionError("bf16 probe should not run"),
        ):
            amp_enabled, amp_dtype = _resolve_amp_settings(
                torch.device("cuda"),
                amp=True,
                amp_precision="fp16",
            )

        self.assertTrue(amp_enabled)
        self.assertEqual(torch.float16, amp_dtype)

    def test_auto_amp_precision_falls_back_when_probe_errors(self) -> None:
        with patch(
            "vl_contradiction.performance.torch.cuda.is_bf16_supported",
            side_effect=RuntimeError("bf16 probe unavailable"),
        ):
            amp_enabled, amp_dtype = _resolve_amp_settings(
                torch.device("cuda"),
                amp=True,
                amp_precision=None,
            )

        self.assertTrue(amp_enabled)
        self.assertEqual(torch.float16, amp_dtype)

    def test_explicit_bf16_precision_falls_back_when_unsupported(self) -> None:
        with patch(
            "vl_contradiction.training._cuda_bf16_supported",
            return_value=False,
        ):
            amp_enabled, amp_dtype = _resolve_amp_settings(
                torch.device("cuda"),
                amp=True,
                amp_precision="bf16",
            )

        self.assertTrue(amp_enabled)
        self.assertEqual(torch.float16, amp_dtype)

    def test_early_stopping_and_cpu_amp_fallback(self) -> None:
        torch.manual_seed(0)
        features = torch.tensor(
            [
                [-1.0, -1.0],
                [-0.9, -1.1],
                [1.0, 1.0],
                [1.1, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        dataset = FeatureDataset(features, labels)
        train_loader = create_loader(dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = create_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        model = LinearProbe(input_dim=2)

        metric_sequence = [
            {"macro_f1": 0.25, "accuracy": 0.25},
            {"macro_f1": 0.25, "accuracy": 0.25},
        ]

        def fake_evaluate(*args, **kwargs):  # noqa: ANN001, ANN003
            index = fake_evaluate.calls
            fake_evaluate.calls += 1
            metrics = metric_sequence[min(index, len(metric_sequence) - 1)]
            logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            labels_tensor = torch.tensor([0, 1], dtype=torch.long)
            return metrics, logits, labels_tensor

        fake_evaluate.calls = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pt"
            with patch("vl_contradiction.training.evaluate_model", side_effect=fake_evaluate):
                result = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=torch.device("cpu"),
                    epochs=5,
                    learning_rate=0.01,
                    weight_decay=0.0,
                    checkpoint_path=checkpoint_path,
                    amp=True,
                    amp_precision=None,
                    early_stopping_patience=1,
                    early_stopping_min_delta=0.0,
                )

            self.assertEqual(2, len(result.history))
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(2, fake_evaluate.calls)
            self.assertAlmostEqual(0.25, float(result.best_val_metrics["macro_f1"]))


if __name__ == "__main__":
    unittest.main()
