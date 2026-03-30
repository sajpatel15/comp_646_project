"""COCO download, validation, and normalized table builders."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import pandas as pd
from tqdm.auto import tqdm

from .io_utils import ensure_dir

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


@dataclass(slots=True)
class DownloadReport:
    name: str
    url: str
    archive_path: Path
    extracted_to: Path


def _progress_hook(progress_bar: tqdm):
    last_block = {"value": 0}

    def update(blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size and progress_bar.total is None:
            progress_bar.total = total_size
        delta = (blocks - last_block["value"]) * block_size
        progress_bar.update(max(delta, 0))
        last_block["value"] = blocks

    return update


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    with tqdm(unit="B", unit_scale=True, desc=f"Downloading {destination.name}") as progress:
        urlretrieve(url, destination, reporthook=_progress_hook(progress))
    return destination


def extract_zip(archive_path: Path, destination: Path) -> Path:
    ensure_dir(destination)
    with zipfile.ZipFile(archive_path, "r") as handle:
        handle.extractall(destination)
    return destination


def expected_coco_paths(dataset_root: Path) -> dict[str, list[Path]]:
    annotations_root = dataset_root / "annotations"
    return {
        "train2017": [dataset_root / "train2017"],
        "val2017": [dataset_root / "val2017"],
        "annotations": [
            annotations_root / "captions_train2017.json",
            annotations_root / "captions_val2017.json",
            annotations_root / "instances_train2017.json",
            annotations_root / "instances_val2017.json",
        ],
    }


def find_missing_coco_assets(dataset_root: Path) -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    for asset_name, paths in expected_coco_paths(dataset_root).items():
        for path in paths:
            if not path.exists():
                missing.append((asset_name, path))
    return missing


def ensure_coco_dataset(
    dataset_root: Path,
    *,
    auto_download: bool = True,
    image_splits: Iterable[str] = ("train2017", "val2017"),
) -> list[DownloadReport]:
    ensure_dir(dataset_root)
    missing = find_missing_coco_assets(dataset_root)
    if not missing:
        return []
    if not auto_download:
        missing_str = ", ".join(str(path) for _, path in missing)
        raise FileNotFoundError(f"Missing COCO assets: {missing_str}")

    reports: list[DownloadReport] = []
    for split in image_splits:
        image_dir = dataset_root / split
        if image_dir.exists():
            continue
        archive_path = dataset_root / f"{split}.zip"
        download_file(COCO_URLS[split], archive_path)
        extract_zip(archive_path, dataset_root)
        reports.append(
            DownloadReport(
                name=split,
                url=COCO_URLS[split],
                archive_path=archive_path,
                extracted_to=image_dir,
            )
        )

    annotations_root = dataset_root / "annotations"
    if not annotations_root.exists():
        archive_path = dataset_root / "annotations_trainval2017.zip"
        download_file(COCO_URLS["annotations"], archive_path)
        extract_zip(archive_path, dataset_root)
        reports.append(
            DownloadReport(
                name="annotations",
                url=COCO_URLS["annotations"],
                archive_path=archive_path,
                extracted_to=annotations_root,
            )
        )

    final_missing = find_missing_coco_assets(dataset_root)
    if final_missing:
        missing_str = ", ".join(str(path) for _, path in final_missing)
        raise FileNotFoundError(f"COCO download incomplete. Missing: {missing_str}")
    return reports


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_coco_caption_table(dataset_root: Path, split: str) -> pd.DataFrame:
    annotations_root = dataset_root / "annotations"
    captions = _read_json(annotations_root / f"captions_{split}.json")
    instances = _read_json(annotations_root / f"instances_{split}.json")

    image_lookup = {item["id"]: item for item in captions["images"]}
    category_lookup = {item["id"]: item["name"] for item in instances["categories"]}
    object_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for item in instances["annotations"]:
        image_id = int(item["image_id"])
        category_name = category_lookup[int(item["category_id"])]
        object_counts[image_id][category_name] += 1

    rows = []
    for item in captions["annotations"]:
        image_id = int(item["image_id"])
        image_meta = image_lookup[image_id]
        counts = dict(sorted(object_counts.get(image_id, Counter()).items()))
        rows.append(
            {
                "image_id": image_id,
                "caption_id": int(item["id"]),
                "source_split": split,
                "file_name": image_meta["file_name"],
                "caption": str(item["caption"]).strip(),
                "objects": sorted(counts.keys()),
                "object_counts": counts,
            }
        )

    return pd.DataFrame(rows)


def build_coco_caption_index(dataset_root: Path, splits: Iterable[str] = ("train2017", "val2017")) -> pd.DataFrame:
    frames = [load_coco_caption_table(dataset_root, split) for split in splits]
    return pd.concat(frames, axis=0, ignore_index=True)


def image_path_for_row(dataset_root: Path, source_split: str, file_name: str) -> Path:
    return dataset_root / source_split / file_name
