from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LABEL_ORDER = ["contradiction", "neutral", "entailment"]

SYNONYM_MAP = {
    "bike": "bicycle",
    "bicycle": "bike",
    "couch": "sofa",
    "sofa": "couch",
    "kid": "child",
    "child": "kid",
    "man": "person",
    "woman": "person",
    "boy": "child",
    "girl": "child",
}

HYPERNYM_MAP = {
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "cow": "animal",
    "bus": "vehicle",
    "truck": "vehicle",
    "car": "vehicle",
    "bicycle": "vehicle",
    "bike": "vehicle",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
}

ATTRIBUTE_FLIPS = {
    "black": "white",
    "white": "black",
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "green",
    "small": "large",
    "large": "small",
}

ACTION_FLIPS = {
    "standing": "sitting",
    "sitting": "standing",
    "walking": "running",
    "running": "walking",
    "holding": "dropping",
    "riding": "pushing",
}

COUNT_WORDS = {
    "one": "two",
    "two": "three",
    "three": "one",
    "1": "2",
    "2": "3",
    "3": "1",
}

COCO_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


@dataclass(slots=True)
class GeneratedRecord:
    family_id: str
    image_id: int
    image_path: str
    source_caption: str
    edited_caption: str
    label: str
    edit_family: str
    edit_rule: str


def _replace_whole_word(text: str, source: str, target: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
    if not pattern.search(text):
        return None
    replaced = pattern.sub(target, text, count=1)
    return _cleanup_caption(replaced)


def _cleanup_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text


def _entailment_edit(caption: str) -> tuple[str, str] | None:
    for source, target in SYNONYM_MAP.items():
        edited = _replace_whole_word(caption, source, target)
        if edited and edited != caption:
            return edited, f"synonym:{source}->{target}"
    return None


def _neutral_edit(caption: str) -> tuple[str, str, str] | None:
    for source, target in HYPERNYM_MAP.items():
        edited = _replace_whole_word(caption, source, target)
        if edited and edited != caption:
            return edited, "neutral_hypernym", f"hypernym:{source}->{target}"

    tokens = caption.split()
    for index, token in enumerate(tokens):
        if token in ATTRIBUTE_FLIPS:
            edited = _cleanup_caption(" ".join(tokens[:index] + tokens[index + 1 :]))
            if edited and edited != caption:
                return edited, "neutral_attribute_drop", f"drop_attribute:{token}"
    return None


def _contradiction_object(caption: str, present_objects: Iterable[str], rng: np.random.Generator) -> tuple[str, str, str] | None:
    present_set = set(present_objects)
    candidate_sources = [name for name in present_set if re.search(rf"\b{re.escape(name)}\b", caption)]
    if not candidate_sources:
        return None
    source = sorted(candidate_sources)[0]
    absent_objects = sorted(COCO_OBJECTS - present_set)
    if not absent_objects:
        return None
    target = rng.choice(absent_objects).item()
    edited = _replace_whole_word(caption, source, target)
    if edited and edited != caption:
        return edited, "contradiction_object", f"object_swap:{source}->{target}"
    return None


def _contradiction_attribute(caption: str) -> tuple[str, str, str] | None:
    for source, target in ATTRIBUTE_FLIPS.items():
        edited = _replace_whole_word(caption, source, target)
        if edited and edited != caption:
            return edited, "contradiction_attribute", f"attribute_flip:{source}->{target}"
    return None


def _contradiction_count(caption: str) -> tuple[str, str, str] | None:
    for source, target in COUNT_WORDS.items():
        edited = _replace_whole_word(caption, source, target)
        if edited and edited != caption:
            return edited, "contradiction_count", f"count_flip:{source}->{target}"
    return None


def _contradiction_action(caption: str) -> tuple[str, str, str] | None:
    for source, target in ACTION_FLIPS.items():
        edited = _replace_whole_word(caption, source, target)
        if edited and edited != caption:
            return edited, "contradiction_action", f"action_flip:{source}->{target}"
    return None


def _generate_family_records(row: pd.Series, rng: np.random.Generator) -> list[GeneratedRecord] | None:
    source_caption = row["caption"]
    entailment = _entailment_edit(source_caption)
    neutral = _neutral_edit(source_caption)
    contradiction = (
        _contradiction_object(source_caption, row["present_objects"], rng)
        or _contradiction_attribute(source_caption)
        or _contradiction_count(source_caption)
        or _contradiction_action(source_caption)
    )

    if not entailment or not neutral or not contradiction:
        return None

    return [
        GeneratedRecord(
            family_id=row["family_id"],
            image_id=row["image_id"],
            image_path=row["image_path"],
            source_caption=source_caption,
            edited_caption=entailment[0],
            label="entailment",
            edit_family="entailment_synonym",
            edit_rule=entailment[1],
        ),
        GeneratedRecord(
            family_id=row["family_id"],
            image_id=row["image_id"],
            image_path=row["image_path"],
            source_caption=source_caption,
            edited_caption=neutral[0],
            label="neutral",
            edit_family=neutral[1],
            edit_rule=neutral[2],
        ),
        GeneratedRecord(
            family_id=row["family_id"],
            image_id=row["image_id"],
            image_path=row["image_path"],
            source_caption=source_caption,
            edited_caption=contradiction[0],
            label="contradiction",
            edit_family=contradiction[1],
            edit_rule=contradiction[2],
        ),
    ]


def build_benchmark_dataset(
    source_table: pd.DataFrame,
    max_families: int | None = None,
    seed: int = 42,
    split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    family_rows = source_table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    generated: list[dict] = []
    family_count = 0
    for _, row in tqdm(family_rows.iterrows(), total=len(family_rows), desc="Generating benchmark"):
        if max_families is not None and family_count >= max_families:
            break
        family_records = _generate_family_records(row, rng)
        if not family_records:
            continue
        family_count += 1
        for record in family_records:
            generated.append(record.__dict__)
    benchmark_frame = pd.DataFrame(generated)
    if benchmark_frame.empty:
        return benchmark_frame

    family_ids = benchmark_frame["family_id"].drop_duplicates().tolist()
    rng.shuffle(family_ids)
    train_cut = int(len(family_ids) * split_ratio[0])
    val_cut = train_cut + int(len(family_ids) * split_ratio[1])
    split_lookup = {}
    for family_id in family_ids[:train_cut]:
        split_lookup[family_id] = "train"
    for family_id in family_ids[train_cut:val_cut]:
        split_lookup[family_id] = "val"
    for family_id in family_ids[val_cut:]:
        split_lookup[family_id] = "test"

    benchmark_frame["split"] = benchmark_frame["family_id"].map(split_lookup)
    benchmark_frame["sample_id"] = benchmark_frame.apply(
        lambda row: f"{row['family_id']}_{row['label']}", axis=1
    )
    return benchmark_frame


def summarize_benchmark(benchmark_frame: pd.DataFrame) -> pd.DataFrame:
    if benchmark_frame.empty:
        return pd.DataFrame(columns=["label", "edit_family", "count"])
    summary = (
        benchmark_frame.groupby(["label", "edit_family"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "edit_family"])
    )
    return summary
