from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.config import load_config  # noqa: E402
from vl_contradiction.performance import resolve_performance_profile  # noqa: E402
from vl_contradiction.runtime import detect_runtime  # noqa: E402


class PerformanceResolutionTests(unittest.TestCase):
    def test_auto_profile_selects_t4_defaults(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

        with (
            mock.patch("vl_contradiction.performance.torch.cuda.is_available", return_value=True),
            mock.patch("vl_contradiction.performance.torch.cuda.get_device_name", return_value="NVIDIA T4"),
            mock.patch(
                "vl_contradiction.performance.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            mock.patch("vl_contradiction.performance.torch.cuda.is_bf16_supported", return_value=False),
        ):
            profile = resolve_performance_profile(
                config.performance,
                device=torch.device("cuda"),
                is_colab=True,
                cache_root=PROJECT_ROOT / "artifacts",
            )

        self.assertEqual("t4", profile.name)
        self.assertEqual("fp16", profile.qwen_precision)
        self.assertEqual("fp16", profile.training_amp_precision)
        self.assertEqual(4, profile.qwen_batch_size)
        self.assertEqual("scratch_then_sync", profile.qwen_cache_mode)
        self.assertEqual(Path("/content/comp646_scratch"), profile.scratch_root)

    def test_t4_training_amp_precision_does_not_drift_to_bf16_when_probe_is_true(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

        with (
            mock.patch("vl_contradiction.performance.torch.cuda.is_available", return_value=True),
            mock.patch("vl_contradiction.performance.torch.cuda.get_device_name", return_value="Tesla T4"),
            mock.patch(
                "vl_contradiction.performance.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            mock.patch("vl_contradiction.performance.torch.cuda.is_bf16_supported", return_value=True),
        ):
            profile = resolve_performance_profile(
                config.performance,
                device=torch.device("cuda"),
                is_colab=True,
                cache_root=PROJECT_ROOT / "artifacts",
            )

        self.assertTrue(profile.amp_training)
        self.assertEqual("fp16", profile.training_amp_precision)

    def test_explicit_bf16_training_profile_falls_back_to_fp16_when_cuda_lacks_support(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        config.performance.active_profile = "h100"

        with (
            mock.patch("vl_contradiction.performance.torch.cuda.is_available", return_value=True),
            mock.patch("vl_contradiction.performance.torch.cuda.get_device_name", return_value="Tesla T4"),
            mock.patch(
                "vl_contradiction.performance.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            mock.patch("vl_contradiction.performance.torch.cuda.is_bf16_supported", return_value=False),
        ):
            profile = resolve_performance_profile(
                config.performance,
                device=torch.device("cuda"),
                is_colab=True,
                cache_root=PROJECT_ROOT / "artifacts",
            )

        self.assertEqual("h100", profile.name)
        self.assertEqual("fp16", profile.training_amp_precision)

    def test_compatibility_mode_forces_conservative_qwen_settings(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        config.performance.compatibility_mode = True

        with (
            mock.patch("vl_contradiction.performance.torch.cuda.is_available", return_value=True),
            mock.patch("vl_contradiction.performance.torch.cuda.get_device_name", return_value="NVIDIA T4"),
            mock.patch(
                "vl_contradiction.performance.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),
            ),
            mock.patch("vl_contradiction.performance.torch.cuda.is_bf16_supported", return_value=False),
        ):
            profile = resolve_performance_profile(
                config.performance,
                device=torch.device("cuda"),
                is_colab=True,
                cache_root=PROJECT_ROOT / "artifacts",
            )

        self.assertEqual("4bit", profile.qwen_precision)
        self.assertEqual(1, profile.qwen_batch_size)
        self.assertEqual("direct", profile.qwen_cache_mode)

    def test_detect_runtime_respects_training_device_override(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        config.training.device = "cpu"

        runtime = detect_runtime(PROJECT_ROOT, config)

        self.assertEqual("cpu", runtime.device.type)
        self.assertEqual("default", runtime.performance.name)
        self.assertIsNone(runtime.performance.training_amp_precision)
        self.assertEqual(runtime.cache_root / ".scratch" / "qwen", runtime.qwen_scratch_root)


if __name__ == "__main__":
    unittest.main()
