"""Runtime setup shared by the notebook and helper modules."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ProjectConfig, load_project_config
from .io_utils import ensure_dir


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def detect_environment() -> str:
    return "colab" if is_colab() else "local"


def discover_project_root(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    candidates = [cursor, *cursor.parents]
    for candidate in candidates:
        if (candidate / "project_grading_criteria.txt").exists():
            return candidate
    return cursor


def resolve_cache_root(project_root: Path) -> Path:
    return project_root / "output" / "cache"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in (
        config.paths.cache_root,
        config.paths.dataset_root,
        config.paths.benchmark_root,
        config.paths.artifacts_root,
        config.paths.tensorboard_root,
        config.paths.figure_root,
        config.paths.qwen_cache_root,
        config.paths.checkpoint_root,
        config.paths.output_root,
    ):
        ensure_dir(path)


def prepare_runtime(
    *,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> ProjectConfig:
    root = discover_project_root(project_root)
    env = detect_environment()
    cache_root = resolve_cache_root(root)
    config = load_project_config(
        env=env,
        project_root=root,
        cache_root=cache_root,
        config_path=config_path,
    )

    ensure_project_dirs(config)
    configure_plot_style()
    seed_everything(config.seed)
    return config


def summarize_runtime(config: ProjectConfig) -> dict[str, Any]:
    return {
        "project_name": config.project_name,
        "env": config.env,
        "cache_root": str(config.paths.cache_root),
        "dataset_root": str(config.paths.dataset_root),
        "benchmark_root": str(config.paths.benchmark_root),
        "artifacts_root": str(config.paths.artifacts_root),
        "clip_model_name": config.clip_model_name,
        "qwen_model_name": config.qwen_model_name,
    }
