"""Resolved runtime performance policies for GPU-backed experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import GPUProfileConfig, PerformanceConfig


@dataclass(slots=True)
class ResolvedPerformanceProfile:
    name: str
    compatibility_mode: bool
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    clip_precision: str
    qwen_precision: str
    training_amp_precision: str | None
    qwen_batch_size: int
    clip_num_workers: int
    persistent_workers: bool
    prefetch_factor: int | None
    amp_training: bool
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    qwen_cache_mode: str
    qwen_cache_flush_every: int
    scratch_root: Path


def _cuda_bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except (AssertionError, RuntimeError):
        return False


def _gpu_name(device: torch.device) -> str | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return str(torch.cuda.get_device_name(device))
    except (AssertionError, RuntimeError):
        return None


def _gpu_total_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        properties = torch.cuda.get_device_properties(device)
    except (AssertionError, RuntimeError):
        return None
    return float(properties.total_memory) / (1024.0**3)


def _select_profile_name(active_profile: str, gpu_profiles: dict[str, GPUProfileConfig], gpu_name: str | None) -> str:
    normalized = active_profile.strip().lower()
    if normalized != "auto":
        if normalized not in gpu_profiles:
            available = ", ".join(sorted(gpu_profiles))
            raise ValueError(f"Unknown performance profile '{active_profile}'. Available profiles: {available}")
        return normalized

    if not gpu_name:
        return "default"

    lowered_name = gpu_name.lower()
    candidates = [name for name in gpu_profiles if name != "default" and name in lowered_name]
    if not candidates:
        return "default"
    return sorted(candidates, key=len, reverse=True)[0]


def _resolve_precision(mode: str, *, device: torch.device, prefer_bf16: bool) -> str:
    normalized = mode.strip().lower()
    if normalized != "auto":
        return normalized
    if device.type != "cuda":
        return "fp32"
    if prefer_bf16 and _cuda_bf16_supported():
        return "bf16"
    return "fp16"


def _resolve_training_amp_precision(mode: str, *, device: torch.device, amp_enabled: bool) -> str | None:
    if not amp_enabled or device.type != "cuda":
        return None

    normalized = mode.strip().lower()
    if normalized not in {"auto", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported training_amp_precision '{mode}'. Expected one of: auto, fp16, bf16"
        )
    if normalized == "auto":
        return "bf16" if _cuda_bf16_supported() else "fp16"
    if normalized == "bf16" and not _cuda_bf16_supported():
        return "fp16"
    return normalized


def _resolve_qwen_batch_size(
    raw_value: str | int,
    *,
    device: torch.device,
    precision: str,
    gpu_name: str | None,
    total_memory_gb: float | None,
    compatibility_mode: bool,
) -> int:
    if compatibility_mode:
        return 1
    if isinstance(raw_value, int):
        return max(raw_value, 1)

    if device.type != "cuda":
        return 1

    lowered_name = (gpu_name or "").lower()
    memory_gb = total_memory_gb or 0.0
    if "h100" in lowered_name:
        return 8
    if "t4" in lowered_name:
        return 2 if precision == "fp16" else 4
    if precision == "bf16":
        return 6 if memory_gb >= 40.0 else 4
    if precision == "fp16":
        return 4 if memory_gb >= 24.0 else 2
    if precision == "4bit":
        return 8 if memory_gb >= 24.0 else 4
    return 1


def resolve_performance_profile(
    config: PerformanceConfig,
    *,
    device: torch.device,
    is_colab: bool,
    cache_root: Path,
) -> ResolvedPerformanceProfile:
    gpu_name = _gpu_name(device)
    total_memory_gb = _gpu_total_memory_gb(device)
    profile_name = _select_profile_name(config.active_profile, config.gpu_profiles, gpu_name)
    selected = config.gpu_profiles[profile_name]

    clip_precision = _resolve_precision(
        selected.clip_precision,
        device=device,
        prefer_bf16=profile_name == "h100",
    )
    qwen_precision = _resolve_precision(
        selected.qwen_precision,
        device=device,
        prefer_bf16=profile_name == "h100",
    )
    amp_training = bool(selected.amp_training and device.type == "cuda")
    training_amp_precision = _resolve_training_amp_precision(
        selected.training_amp_precision,
        device=device,
        amp_enabled=amp_training,
    )
    if config.compatibility_mode:
        qwen_precision = "4bit" if device.type == "cuda" else "fp32"

    qwen_batch_size = _resolve_qwen_batch_size(
        selected.qwen_batch_size,
        device=device,
        precision=qwen_precision,
        gpu_name=gpu_name,
        total_memory_gb=total_memory_gb,
        compatibility_mode=config.compatibility_mode,
    )

    scratch_root = Path(config.colab_scratch_root).expanduser() if is_colab else cache_root / ".scratch"

    return ResolvedPerformanceProfile(
        name=profile_name,
        compatibility_mode=config.compatibility_mode,
        gpu_name=gpu_name,
        gpu_total_memory_gb=total_memory_gb,
        clip_precision=clip_precision,
        qwen_precision=qwen_precision,
        training_amp_precision=training_amp_precision,
        qwen_batch_size=qwen_batch_size,
        clip_num_workers=max(int(selected.clip_num_workers), 0),
        persistent_workers=bool(selected.persistent_workers),
        prefetch_factor=selected.prefetch_factor,
        amp_training=amp_training,
        early_stopping_patience=selected.early_stopping_patience,
        early_stopping_min_delta=float(selected.early_stopping_min_delta),
        qwen_cache_mode="direct" if config.compatibility_mode else selected.qwen_cache_mode,
        qwen_cache_flush_every=max(int(selected.qwen_cache_flush_every), 1),
        scratch_root=scratch_root,
    )
