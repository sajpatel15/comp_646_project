"""Configuration loading for the multimodal contradiction project."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light test environments
    yaml = None


DEFAULT_PERFORMANCE_PAYLOAD: dict[str, Any] = {
    "active_profile": "auto",
    "compatibility_mode": False,
    "colab_scratch_root": "artifacts/.scratch",
    "gpu_profiles": {
        "default": {
            "clip_precision": "auto",
            "qwen_precision": "auto",
            "training_amp_precision": "auto",
            "qwen_batch_size": "auto",
            "clip_num_workers": 2,
            "persistent_workers": False,
            "prefetch_factor": None,
            "amp_training": True,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.001,
            "qwen_cache_mode": "direct",
            "qwen_cache_flush_every": 8,
        },
        "t4": {
            "clip_precision": "fp16",
            "qwen_precision": "fp16",
            "training_amp_precision": "fp16",
            "qwen_batch_size": "auto",
            "clip_num_workers": 4,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "amp_training": True,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
            "qwen_cache_mode": "direct",
            "qwen_cache_flush_every": 8,
        },
        "h100": {
            "clip_precision": "bf16",
            "qwen_precision": "bf16",
            "training_amp_precision": "bf16",
            "qwen_batch_size": "auto",
            "clip_num_workers": 8,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "amp_training": True,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0005,
            "qwen_cache_mode": "direct",
            "qwen_cache_flush_every": 16,
        },
    },
}


@dataclass(slots=True)
class RuntimeConfig:
    seed: int
    split_seed: int
    local_root: str


@dataclass(slots=True)
class PathsConfig:
    dataset_root: str
    benchmark_root: str
    checkpoint_root: str
    log_root: str
    metrics_root: str
    figure_root: str
    qwen_root: str


@dataclass(slots=True)
class DataConfig:
    prototype_families: int
    midscale_families: int
    final_families: int
    split_ratio: list[float]
    audit_samples_per_family: int
    qwen_subset_size: int
    image_splits: list[str]


@dataclass(slots=True)
class ModelConfig:
    clip_name: str
    qwen_name: str
    use_qwen_4bit: bool
    hidden_dim: int
    num_attention_heads: int
    dropout: float
    max_qwen_tokens: int


@dataclass(slots=True)
class TrainingConfig:
    num_workers: int
    device: str
    log_every_epoch: bool
    selection_metric: str
    clip_batch_size: int
    joint_feature_batch_size: int
    token_feature_batch_size: int
    sweeps: dict[str, dict[str, dict[str, list[dict[str, Any]]]]]


@dataclass(slots=True)
class EvaluationConfig:
    bootstrap_samples: int
    threshold_grid_size: int
    save_figures_as_pdf: bool


@dataclass(slots=True)
class AuditConfig:
    overall_label_valid_threshold: float
    overall_grammar_ok_threshold: float
    per_family_label_valid_threshold: float
    launch_review_ui: bool
    require_all_rows_reviewed: bool
    require_qwen_for_readiness: bool


@dataclass(slots=True)
class GPUProfileConfig:
    clip_precision: str
    qwen_precision: str
    training_amp_precision: str
    qwen_batch_size: str | int
    clip_num_workers: int
    persistent_workers: bool
    prefetch_factor: int | None
    amp_training: bool
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    qwen_cache_mode: str
    qwen_cache_flush_every: int


@dataclass(slots=True)
class PerformanceConfig:
    active_profile: str
    compatibility_mode: bool
    colab_scratch_root: str
    gpu_profiles: dict[str, GPUProfileConfig]


@dataclass(slots=True)
class ProjectConfig:
    runtime: RuntimeConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    audit: AuditConfig
    performance: PerformanceConfig
    source_path: Path

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        return data


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to load project configuration files.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_performance_config(raw: dict[str, Any]) -> PerformanceConfig:
    merged = _merge_dicts(DEFAULT_PERFORMANCE_PAYLOAD, raw)
    gpu_profiles = {
        name: GPUProfileConfig(**profile_payload)
        for name, profile_payload in merged["gpu_profiles"].items()
    }
    return PerformanceConfig(
        active_profile=str(merged["active_profile"]),
        compatibility_mode=bool(merged["compatibility_mode"]),
        colab_scratch_root=str(merged["colab_scratch_root"]),
        gpu_profiles=gpu_profiles,
    )


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load the project configuration from a YAML file."""

    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    return ProjectConfig(
        runtime=RuntimeConfig(**raw["runtime"]),
        paths=PathsConfig(**raw["paths"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        audit=AuditConfig(**raw["audit"]),
        performance=_load_performance_config(raw.get("performance", {})),
        source_path=path,
    )
