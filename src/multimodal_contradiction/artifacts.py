from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    frame.to_csv(target, index=index)
    return target


def stage_metrics_path(metrics_root: str | Path, stage: str) -> Path:
    return ensure_directory(metrics_root) / f"{stage}_metrics.json"
