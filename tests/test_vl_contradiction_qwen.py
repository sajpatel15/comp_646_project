from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.qwen import (  # noqa: E402
    QwenBundle,
    QwenRuntimePolicy,
    load_qwen_bundle,
    parse_qwen_output,
    run_qwen_inference,
)


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"][1]["text"]

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        batch_size = len(text)
        input_ids = torch.full((batch_size, 4), 7, dtype=torch.long)
        attention_mask = torch.ones((batch_size, 4), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        return [
            '{"label": "contradiction", "rationale": "stable"}'
            for _ in range(len(generated_tokens))
        ]


class FakeModel(torch.nn.Module):
    def __init__(self, oom_threshold: int | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.oom_threshold = oom_threshold
        self.generate_calls: list[int] = []

    def generate(self, **kwargs):
        batch_size = int(kwargs["input_ids"].shape[0])
        self.generate_calls.append(batch_size)
        if self.oom_threshold is not None and batch_size > self.oom_threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate.")
        seq_len = int(kwargs["input_ids"].shape[1]) + 2
        return torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)


class FakeModelClass:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        cls.calls.append(kwargs)
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] == torch.float16:
            raise RuntimeError("CUDA out of memory. Tried to allocate.")
        return FakeModel()


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (2, 2), color=color).save(path)


class QwenTests(unittest.TestCase):
    def test_parse_qwen_output_handles_json_and_keyword_fallback(self) -> None:
        self.assertEqual(
            {"label": "neutral", "rationale": "because"},
            parse_qwen_output('prefix {"label": "neutral", "rationale": "because"} suffix'),
        )
        self.assertEqual(
            {"label": "entailment", "rationale": "entailment appears in text"},
            parse_qwen_output("entailment appears in text"),
        )

    def test_load_qwen_bundle_auto_precision_falls_back_to_4bit(self) -> None:
        FakeModelClass.calls = []
        fake_processor = SimpleNamespace()

        with mock.patch("vl_contradiction.qwen.AutoProcessor.from_pretrained", return_value=fake_processor), mock.patch(
            "vl_contradiction.qwen._resolve_qwen_model_cls", return_value=FakeModelClass
        ), mock.patch("vl_contradiction.qwen.torch.cuda.is_available", return_value=True), mock.patch(
            "vl_contradiction.qwen.torch.cuda.is_bf16_supported", return_value=False, create=True
        ), mock.patch("vl_contradiction.qwen.BitsAndBytesConfig") as fake_bnb:
            fake_bnb.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
            bundle = load_qwen_bundle(
                "qwen/test-model",
                use_4bit=True,
                precision="auto",
            )

        self.assertIs(bundle.processor, fake_processor)
        self.assertEqual("4bit", bundle.policy.precision)
        self.assertGreaterEqual(len(FakeModelClass.calls), 2)
        self.assertIn("torch_dtype", FakeModelClass.calls[0])
        self.assertIn("quantization_config", FakeModelClass.calls[-1])

    def test_load_qwen_bundle_fp16_falls_back_to_4bit_on_oom(self) -> None:
        FakeModelClass.calls = []
        fake_processor = SimpleNamespace()

        with mock.patch("vl_contradiction.qwen.AutoProcessor.from_pretrained", return_value=fake_processor), mock.patch(
            "vl_contradiction.qwen._resolve_qwen_model_cls", return_value=FakeModelClass
        ), mock.patch("vl_contradiction.qwen.torch.cuda.is_available", return_value=True), mock.patch(
            "vl_contradiction.qwen.torch.cuda.is_bf16_supported", return_value=False, create=True
        ), mock.patch("vl_contradiction.qwen.BitsAndBytesConfig") as fake_bnb:
            fake_bnb.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
            bundle = load_qwen_bundle(
                "qwen/test-model",
                use_4bit=True,
                precision="fp16",
            )

        self.assertEqual("4bit", bundle.policy.precision)
        self.assertEqual(torch.float16, FakeModelClass.calls[0]["torch_dtype"])
        self.assertIn("quantization_config", FakeModelClass.calls[-1])

    def test_run_qwen_inference_batches_with_oom_backoff_and_scratch_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_dir = root / "qwen"
            scratch_root = root / "scratch"
            image_paths = []
            for index in range(5):
                image_path = root / f"image_{index}.png"
                _write_image(image_path, (index * 20, 0, 0))
                image_paths.append(image_path)

            records = pd.DataFrame(
                {
                    "sample_id": [f"sample/{i}" for i in range(5)],
                    "label": ["contradiction"] * 5,
                    "edited_caption": [f"caption {i}" for i in range(5)],
                    "file_path": image_paths,
                }
            )
            model = FakeModel(oom_threshold=2)
            bundle = QwenBundle(
                model=model,
                processor=FakeProcessor(),
                device=torch.device("cpu"),
                policy=QwenRuntimePolicy(
                    profile_name="t4",
                    precision="fp16",
                    batch_size=4,
                    cache_mode="scratch_then_sync",
                    cache_flush_every=2,
                    scratch_root=scratch_root,
                ),
            )

            outputs = run_qwen_inference(records, bundle, final_dir, max_new_tokens=8)

            self.assertEqual(records["sample_id"].tolist(), outputs["sample_id"].tolist())
            self.assertEqual([4, 2, 2, 1], model.generate_calls)
            self.assertEqual(5, len(list(final_dir.glob("*.json"))))
            self.assertEqual(5, len(list((scratch_root / final_dir.name).glob("*.json"))))

    def test_run_qwen_inference_promotes_scratch_hits_into_canonical_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_dir = root / "qwen" / "prototype"
            scratch_root = root / "scratch" / "qwen" / "prototype"
            scratch_root.mkdir(parents=True, exist_ok=True)

            image_path = root / "promote.png"
            _write_image(image_path, (10, 10, 10))
            payload = {
                "sample_id": "sample/0",
                "label": "neutral",
                "pred_label": "neutral",
                "rationale": "cached",
                "raw_output": '{"label": "neutral", "rationale": "cached"}',
                "runtime_ms": 1.0,
            }
            (scratch_root / "sample_0.json").write_text(pd.Series(payload).to_json(), encoding="utf-8")
            records = pd.DataFrame(
                {
                    "sample_id": ["sample/0"],
                    "label": ["neutral"],
                    "edited_caption": ["caption 0"],
                    "file_path": [image_path],
                }
            )
            model = FakeModel()
            bundle = QwenBundle(
                model=model,
                processor=FakeProcessor(),
                device=torch.device("cpu"),
                policy=QwenRuntimePolicy(
                    profile_name="t4",
                    precision="fp16",
                    batch_size=2,
                    cache_mode="scratch_then_sync",
                    cache_flush_every=1,
                    scratch_root=scratch_root,
                ),
            )

            outputs = run_qwen_inference(records, bundle, final_dir, max_new_tokens=8)

            self.assertEqual(["sample/0"], outputs["sample_id"].tolist())
            self.assertEqual([], model.generate_calls)
            self.assertTrue((final_dir / "sample_0.json").exists())

    def test_run_qwen_inference_compatibility_mode_forces_single_row_direct_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_dir = root / "qwen"
            scratch_root = root / "scratch"
            image_paths = []
            for index in range(3):
                image_path = root / f"compat_{index}.png"
                _write_image(image_path, (0, index * 20, 0))
                image_paths.append(image_path)

            records = pd.DataFrame(
                {
                    "sample_id": [f"compat/{i}" for i in range(3)],
                    "label": ["neutral"] * 3,
                    "edited_caption": [f"caption {i}" for i in range(3)],
                    "file_path": image_paths,
                }
            )
            model = FakeModel()
            bundle = QwenBundle(
                model=model,
                processor=FakeProcessor(),
                device=torch.device("cpu"),
                policy=QwenRuntimePolicy(
                    profile_name="compatibility",
                    precision="4bit",
                    batch_size=4,
                    compatibility_mode=True,
                    cache_mode="scratch_then_sync",
                    cache_flush_every=1,
                    scratch_root=scratch_root,
                ),
            )

            outputs = run_qwen_inference(records, bundle, final_dir, max_new_tokens=8)

            self.assertEqual(records["sample_id"].tolist(), outputs["sample_id"].tolist())
            self.assertEqual([1, 1, 1], model.generate_calls)
            self.assertEqual(3, len(list(final_dir.glob("*.json"))))
            self.assertFalse(scratch_root.exists() and list(scratch_root.rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
