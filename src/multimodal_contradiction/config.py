"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int


@dataclass(slots=True)
class ProjectPaths:
    project_root: Path
    config_path: Path
    cache_root: Path
    dataset_root: Path
    benchmark_root: Path
    artifacts_root: Path
    tensorboard_root: Path
    figure_root: Path
    qwen_cache_root: Path
    checkpoint_root: Path
    output_root: Path
    drive_root: Path | None = None


@dataclass(slots=True)
class ProjectConfig:
    project_name: str
    env: str
    seed: int
    split_seed: int
    cache_in_drive: bool
    prototype_family_count: int
    mid_family_count: int
    final_family_count: int
    split_ratios: dict[str, float]
    clip_model_name: str
    qwen_model_name: str
    qwen_quantized_4bit: bool
    qwen_subset_size: int
    linear_probe: TrainingConfig
    cross_attention: TrainingConfig
    paths: ProjectPaths = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload["paths"].items():
            if value is not None:
                payload["paths"][key] = str(value)
        return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_training_config(payload: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        epochs=int(payload["epochs"]),
    )


def load_project_config(
    *,
    env: str,
    project_root: Path,
    cache_root: Path,
    drive_root: Path | None = None,
    config_path: Path | None = None,
) -> ProjectConfig:
    config_file = config_path or project_root / "config" / "defaults.yaml"
    payload = _read_yaml(config_file)

    paths = ProjectPaths(
        project_root=project_root,
        config_path=config_file,
        cache_root=cache_root,
        dataset_root=cache_root / "coco",
        benchmark_root=cache_root / "benchmark",
        artifacts_root=cache_root / "artifacts",
        tensorboard_root=cache_root / "tensorboard",
        figure_root=cache_root / "figures",
        qwen_cache_root=cache_root / "qwen",
        checkpoint_root=cache_root / "checkpoints",
        output_root=project_root / "output",
        drive_root=drive_root,
    )

    return ProjectConfig(
        project_name=str(payload["project_name"]),
        env=env,
        seed=int(payload["seed"]),
        split_seed=int(payload["split_seed"]),
        cache_in_drive=bool(payload["cache_in_drive"]),
        prototype_family_count=int(payload["prototype_family_count"]),
        mid_family_count=int(payload["mid_family_count"]),
        final_family_count=int(payload["final_family_count"]),
        split_ratios={key: float(value) for key, value in payload["split_ratios"].items()},
        clip_model_name=str(payload["clip_model_name"]),
        qwen_model_name=str(payload["qwen_model_name"]),
        qwen_quantized_4bit=bool(payload["qwen_quantized_4bit"]),
        qwen_subset_size=int(payload["qwen_subset_size"]),
        linear_probe=build_training_config(payload["linear_probe"]),
        cross_attention=build_training_config(payload["cross_attention"]),
        paths=paths,
    )
