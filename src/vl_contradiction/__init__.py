"""Core helpers for the multimodal contradiction project."""

from .config import ProjectConfig, load_config
from .runtime import RuntimeInfo, detect_runtime, ensure_directories, set_global_seed

__all__ = [
    "ProjectConfig",
    "RuntimeInfo",
    "detect_runtime",
    "ensure_directories",
    "load_config",
    "set_global_seed",
]
