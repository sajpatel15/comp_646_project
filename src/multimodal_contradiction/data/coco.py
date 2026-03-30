from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ..artifacts import ensure_directory

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def validate_coco_cache(data_root: str | Path) -> dict[str, bool]:
    root = Path(data_root)
    expected = {
        "train_images": root / "train2017",
        "val_images": root / "val2017",
        "captions_train": root / "annotations" / "captions_train2017.json",
        "captions_val": root / "annotations" / "captions_val2017.json",
        "instances_train": root / "annotations" / "instances_train2017.json",
        "instances_val": root / "annotations" / "instances_val2017.json",
    }
    return {name: path.exists() for name, path in expected.items()}


def _download_with_progress(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        total = int(response.info().get("Content-Length", "0"))
        with destination.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination)


def download_coco_assets(data_root: str | Path, force: bool = False) -> dict[str, bool]:
    root = ensure_directory(data_root)
    cache_status = validate_coco_cache(root)
    if all(cache_status.values()) and not force:
        return cache_status

    download_root = ensure_directory(root / "_downloads")
    for asset_name, url in COCO_URLS.items():
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = download_root / archive_name
        should_fetch = force or not archive_path.exists()
        if should_fetch:
            print(f"[COCO] Downloading {asset_name}...")
            _download_with_progress(url, archive_path)
        print(f"[COCO] Extracting {archive_name}...")
        _extract_zip(archive_path, root)

    return validate_coco_cache(root)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_caption(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned.lower()


def build_coco_source_table(data_root: str | Path, splits: tuple[str, ...] = ("train", "val")) -> pd.DataFrame:
    root = Path(data_root)
    rows: list[dict] = []

    for split in splits:
        caption_payload = _load_json(root / "annotations" / f"captions_{split}2017.json")
        instance_payload = _load_json(root / "annotations" / f"instances_{split}2017.json")

        image_lookup = {image["id"]: image["file_name"] for image in caption_payload["images"]}
        categories = {
            category["id"]: category["name"].lower()
            for category in instance_payload["categories"]
        }

        objects_by_image: dict[int, list[str]] = defaultdict(list)
        counts_by_image: dict[int, Counter[str]] = defaultdict(Counter)
        for annotation in instance_payload["annotations"]:
            object_name = categories[annotation["category_id"]]
            objects_by_image[annotation["image_id"]].append(object_name)
            counts_by_image[annotation["image_id"]][object_name] += 1

        for annotation in caption_payload["annotations"]:
            image_id = annotation["image_id"]
            file_name = image_lookup[image_id]
            rows.append(
                {
                    "family_id": f"{split}_{image_id}_{annotation['id']}",
                    "split_source": split,
                    "image_id": image_id,
                    "caption_id": annotation["id"],
                    "image_path": str(root / f"{split}2017" / file_name),
                    "caption": _normalize_caption(annotation["caption"]),
                    "present_objects": sorted(set(objects_by_image[image_id])),
                    "object_counts": dict(counts_by_image[image_id]),
                }
            )

    return pd.DataFrame(rows)
