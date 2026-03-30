"""Runtime helpers shared by the notebook and helper modules."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .config import ProjectConfig


@dataclass(slots=True)
class RuntimeInfo:
    project_root: Path
    is_colab: bool
    device: torch.device
    cache_root: Path
    dataset_root: Path
    benchmark_root: Path
    checkpoint_root: Path
    log_root: Path
    metrics_root: Path
    figure_root: Path
    qwen_root: Path


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _resolve_root(project_root: Path, config: ProjectConfig, is_colab: bool) -> Path:
    if is_colab:
        return Path(config.runtime.colab_drive_root).expanduser()
    return project_root / config.runtime.local_root


def _resolve_subpath(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def mount_google_drive_if_needed(config: ProjectConfig) -> None:
    """Mount Google Drive when running inside Colab."""

    if not config.runtime.auto_mount_drive or not _in_colab():
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return
    drive.mount("/content/drive", force_remount=False)


def detect_runtime(project_root: str | Path, config: ProjectConfig) -> RuntimeInfo:
    """Resolve runtime information and artifact directories."""

    root = Path(project_root).expanduser().resolve()
    is_colab = _in_colab()
    mount_google_drive_if_needed(config)
    cache_root = _resolve_root(root, config, is_colab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RuntimeInfo(
        project_root=root,
        is_colab=is_colab,
        device=device,
        cache_root=cache_root,
        dataset_root=(
            Path(config.runtime.colab_dataset_root).expanduser()
            if is_colab
            else _resolve_subpath(cache_root, config.paths.dataset_root)
        ),
        benchmark_root=_resolve_subpath(cache_root, config.paths.benchmark_root),
        checkpoint_root=_resolve_subpath(cache_root, config.paths.checkpoint_root),
        log_root=_resolve_subpath(cache_root, config.paths.log_root),
        metrics_root=_resolve_subpath(cache_root, config.paths.metrics_root),
        figure_root=_resolve_subpath(cache_root, config.paths.figure_root),
        qwen_root=_resolve_subpath(cache_root, config.paths.qwen_root),
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_runtime_summary(runtime: RuntimeInfo) -> None:
    """Print a compact runtime summary for the notebook."""

    print("Runtime summary")
    print(f"  project_root: {runtime.project_root}")
    print(f"  cache_root:   {runtime.cache_root}")
    print(f"  dataset_root: {runtime.dataset_root}")
    print(f"  benchmark:    {runtime.benchmark_root}")
    print(f"  checkpoints:  {runtime.checkpoint_root}")
    print(f"  metrics:      {runtime.metrics_root}")
    print(f"  figures:      {runtime.figure_root}")
    print(f"  colab:        {runtime.is_colab}")
    print(f"  device:       {runtime.device}")
