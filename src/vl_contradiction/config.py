"""Configuration loading for the multimodal contradiction project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    seed: int
    split_seed: int
    auto_mount_drive: bool
    local_root: str
    colab_drive_root: str
    colab_dataset_root: str


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
    require_all_rows_reviewed: bool
    require_qwen_for_readiness: bool


@dataclass(slots=True)
class ProjectConfig:
    runtime: RuntimeConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    audit: AuditConfig
    source_path: Path

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
        source_path=path,
    )
