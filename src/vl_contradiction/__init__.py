"""Core helpers for the multimodal contradiction project."""

from __future__ import annotations


__all__ = [
    "AuditConfig",
    "PerformanceConfig",
    "ProjectConfig",
    "RuntimeInfo",
    "auto_fill_audit_sheet",
    "detect_runtime",
    "ensure_directories",
    "launch_audit_reviewer",
    "load_config",
    "scope_runtime",
    "set_global_seed",
]


def __getattr__(name: str):
    if name == "auto_fill_audit_sheet":
        from .audit_automation import auto_fill_audit_sheet

        return auto_fill_audit_sheet
    if name == "launch_audit_reviewer":
        from .audit_ui import launch_audit_reviewer

        return launch_audit_reviewer
    if name in {"AuditConfig", "PerformanceConfig", "ProjectConfig", "load_config"}:
        from .config import AuditConfig, PerformanceConfig, ProjectConfig, load_config

        return {
            "AuditConfig": AuditConfig,
            "PerformanceConfig": PerformanceConfig,
            "ProjectConfig": ProjectConfig,
            "load_config": load_config,
        }[name]
    if name in {"RuntimeInfo", "detect_runtime", "ensure_directories", "scope_runtime", "set_global_seed"}:
        from .runtime import RuntimeInfo, detect_runtime, ensure_directories, scope_runtime, set_global_seed

        return {
            "RuntimeInfo": RuntimeInfo,
            "detect_runtime": detect_runtime,
            "ensure_directories": ensure_directories,
            "scope_runtime": scope_runtime,
            "set_global_seed": set_global_seed,
        }[name]
    raise AttributeError(f"module 'vl_contradiction' has no attribute {name!r}")
