"""COCO download, validation, and normalization helpers."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import pandas as pd
from tqdm.auto import tqdm


COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


@dataclass(slots=True)
class CocoPaths:
    dataset_root: Path
    train_images_dir: Path
    val_images_dir: Path
    annotations_dir: Path
    instances_train: Path
    instances_val: Path
    captions_train: Path
    captions_val: Path


def build_coco_paths(dataset_root: str | Path) -> CocoPaths:
    """Build the canonical path layout used by the notebook."""

    root = Path(dataset_root).expanduser().resolve()
    annotations = root / "annotations"
    return CocoPaths(
        dataset_root=root,
        train_images_dir=root / "train2017",
        val_images_dir=root / "val2017",
        annotations_dir=annotations,
        instances_train=annotations / "instances_train2017.json",
        instances_val=annotations / "instances_val2017.json",
        captions_train=annotations / "captions_train2017.json",
        captions_val=annotations / "captions_val2017.json",
    )


def coco_assets_present(paths: CocoPaths) -> bool:
    """Return True when the expected COCO assets already exist."""

    required = [
        paths.train_images_dir,
        paths.val_images_dir,
        paths.instances_train,
        paths.instances_val,
        paths.captions_train,
        paths.captions_val,
    ]
    return all(path.exists() for path in required)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:  # noqa: S310 - controlled URLs
        total = int(response.headers.get("Content-Length", 0))
        desc = f"Downloading {destination.name}"
        with destination.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def ensure_coco_dataset(dataset_root: str | Path, download: bool = True) -> CocoPaths:
    """Validate COCO assets and optionally download missing files."""

    paths = build_coco_paths(dataset_root)
    if coco_assets_present(paths):
        print(f"COCO dataset already available at {paths.dataset_root}")
        return paths
    if not download:
        raise FileNotFoundError(f"Missing COCO assets under {paths.dataset_root}")

    print(f"Preparing COCO dataset under {paths.dataset_root}")
    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    download_dir = paths.dataset_root / "_downloads"
    for key, url in COCO_URLS.items():
        zip_path = download_dir / f"{key}.zip"
        if not zip_path.exists():
            _download_file(url, zip_path)
        print(f"Extracting {zip_path.name}")
        _extract_zip(zip_path, paths.dataset_root)
    return paths


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_object_lookup(instances_json: dict) -> tuple[dict[int, list[str]], dict[int, dict[str, int]], list[str]]:
    categories = {item["id"]: item["name"] for item in instances_json["categories"]}
    image_objects: dict[int, list[str]] = defaultdict(list)
    object_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for annotation in instances_json["annotations"]:
        image_id = annotation["image_id"]
        category_name = categories[annotation["category_id"]]
        image_objects[image_id].append(category_name)
        object_counts[image_id][category_name] += 1
    unique_categories = sorted(set(categories.values()))
    object_counts_dict = {image_id: dict(counter) for image_id, counter in object_counts.items()}
    return dict(image_objects), object_counts_dict, unique_categories


def _records_for_split(coco_paths: CocoPaths, split: str) -> pd.DataFrame:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported COCO split: {split}")

    caption_path = coco_paths.captions_train if split == "train" else coco_paths.captions_val
    instance_path = coco_paths.instances_train if split == "train" else coco_paths.instances_val
    image_dir = coco_paths.train_images_dir if split == "train" else coco_paths.val_images_dir

    captions = _load_json(caption_path)
    instances = _load_json(instance_path)
    image_lookup = {image["id"]: image for image in captions["images"]}
    image_objects, object_counts, category_vocab = _build_object_lookup(instances)

    rows = []
    for annotation in captions["annotations"]:
        image = image_lookup[annotation["image_id"]]
        rows.append(
            {
                "family_id": f"{split}-{annotation['id']}",
                "caption_id": annotation["id"],
                "image_id": annotation["image_id"],
                "caption": annotation["caption"].strip(),
                "split_source": split,
                "file_name": image["file_name"],
                "file_path": str(image_dir / image["file_name"]),
                "objects": sorted(set(image_objects.get(annotation["image_id"], []))),
                "object_counts": object_counts.get(annotation["image_id"], {}),
                "height": image.get("height"),
                "width": image.get("width"),
                "category_vocab": category_vocab,
            }
        )
    return pd.DataFrame(rows)


def load_coco_caption_context(coco_paths: CocoPaths, splits: Iterable[str] = ("train", "val")) -> pd.DataFrame:
    """Load caption rows with image context for the requested splits."""

    frames = []
    for split in splits:
        print(f"Loading COCO metadata for split={split}")
        frames.append(_records_for_split(coco_paths, split))
    frame = pd.concat(frames, ignore_index=True)
    frame["caption_length"] = frame["caption"].str.split().str.len()
    return frame
