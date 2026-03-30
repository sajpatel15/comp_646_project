"""Benchmark generation helpers for entailment, neutral, and contradiction labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SYNONYM_MAP = {
    "person": "individual",
    "people": "individuals",
    "bike": "bicycle",
    "bikes": "bicycles",
    "tv": "television",
    "couch": "sofa",
    "cell phone": "mobile phone",
}

HYPERNYM_MAP = {
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "apple": "fruit",
    "banana": "fruit",
    "orange": "fruit",
}

ATTRIBUTE_FLIPS = {
    "red": "blue",
    "blue": "red",
    "black": "white",
    "white": "black",
    "small": "large",
    "large": "small",
    "young": "old",
    "old": "young",
}

ACTION_FLIPS = {
    "standing": "sleeping",
    "sitting": "running",
    "running": "sleeping",
    "walking": "sitting",
    "holding": "dropping",
    "riding": "pushing",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

NUMBER_LOOKUP = {value: key for key, value in NUMBER_WORDS.items()}
ATTRIBUTE_WORDS = set(ATTRIBUTE_FLIPS)


@dataclass(slots=True)
class BenchmarkBuildResult:
    records: pd.DataFrame
    family_manifest: pd.DataFrame


def _replace_first(text: str, source: str, target: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
    if not pattern.search(text):
        return None
    replaced = pattern.sub(target, text, count=1)
    return _normalize_whitespace(replaced)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.")


def _remove_attribute_word(text: str) -> tuple[str | None, str | None]:
    tokens = text.split()
    for index, token in enumerate(tokens):
        normalized = re.sub(r"[^a-zA-Z]", "", token.lower())
        if normalized in ATTRIBUTE_WORDS:
            updated = tokens[:index] + tokens[index + 1 :]
            return _normalize_whitespace(" ".join(updated)), normalized
    return None, None


def _entailment_variant(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in SYNONYM_MAP.items():
        if source in lower:
            updated = _replace_first(caption, source, target)
            if updated and updated.lower() != caption.lower():
                return updated, f"synonym:{source}->{target}"
    return None, None


def _neutral_variant(caption: str) -> tuple[str | None, str | None, str | None]:
    lower = caption.lower()
    for source, target in HYPERNYM_MAP.items():
        if re.search(rf"\b{re.escape(source)}\b", lower):
            updated = _replace_first(caption, source, target)
            if updated and updated.lower() != caption.lower():
                return updated, "neutral_hypernym", f"hypernym:{source}->{target}"
    updated, removed = _remove_attribute_word(caption)
    if updated and removed:
        return updated, "neutral_attribute_drop", f"attribute_drop:{removed}"
    return None, None, None


def _count_contradiction(caption: str, object_counts: dict[str, int]) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lower):
            replacement_value = 1 if value != 1 else 2
            replacement_word = NUMBER_LOOKUP[replacement_value]
            updated = _replace_first(caption, word, replacement_word)
            if updated:
                return updated, f"count:{word}->{replacement_word}"
    for object_name, count in object_counts.items():
        if count <= 1:
            continue
        article_pattern = re.compile(rf"\b(a|an)\s+{re.escape(object_name)}\b", flags=re.IGNORECASE)
        if article_pattern.search(caption):
            updated = article_pattern.sub(f"two {object_name}", caption, count=1)
            return _normalize_whitespace(updated), f"count:article->{object_name}"
    return None, None


def _attribute_contradiction(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in ATTRIBUTE_FLIPS.items():
        if re.search(rf"\b{re.escape(source)}\b", lower):
            updated = _replace_first(caption, source, target)
            if updated:
                return updated, f"attribute:{source}->{target}"
    return None, None


def _object_contradiction(caption: str, present_objects: Iterable[str], absent_objects: Iterable[str]) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for present in present_objects:
        if not re.search(rf"\b{re.escape(present)}\b", lower):
            continue
        for absent in absent_objects:
            if absent == present:
                continue
            updated = _replace_first(caption, present, absent)
            if updated:
                return updated, f"object:{present}->{absent}"
    return None, None


def _action_contradiction(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in ACTION_FLIPS.items():
        if re.search(rf"\b{re.escape(source)}\b", lower):
            updated = _replace_first(caption, source, target)
            if updated:
                return updated, f"action:{source}->{target}"
    return None, None


def _contradiction_variant(row: pd.Series, all_categories: list[str]) -> tuple[str | None, str | None, str | None]:
    caption = row["caption"]
    object_counts = row["object_counts"]
    present_objects = row["objects"]
    absent_objects = [category for category in all_categories if category not in set(present_objects)]

    for family, fn in (
        ("contradiction_count", lambda: _count_contradiction(caption, object_counts)),
        ("contradiction_attribute", lambda: _attribute_contradiction(caption)),
        ("contradiction_object", lambda: _object_contradiction(caption, present_objects, absent_objects)),
        ("contradiction_action", lambda: _action_contradiction(caption)),
    ):
        updated, rule = fn()
        if updated:
            return updated, family, rule
    return None, None, None


def _make_record(row: pd.Series, edited_caption: str, label: str, edit_family: str, edit_rule: str) -> dict:
    sample_id = f"{row['family_id']}::{label}"
    return {
        "sample_id": sample_id,
        "family_id": row["family_id"],
        "image_id": row["image_id"],
        "source_caption": row["caption"],
        "edited_caption": edited_caption,
        "label": label,
        "edit_family": edit_family,
        "edit_rule": edit_rule,
        "audit_status": "pending",
        "file_path": row["file_path"],
        "objects": row["objects"],
        "object_counts": row["object_counts"],
    }


def _family_records(row: pd.Series, all_categories: list[str]) -> list[dict] | None:
    entailment_text, entailment_rule = _entailment_variant(row["caption"])
    neutral_text, neutral_family, neutral_rule = _neutral_variant(row["caption"])
    contradiction_text, contradiction_family, contradiction_rule = _contradiction_variant(row, all_categories)
    if not all([entailment_text, neutral_text, contradiction_text]):
        return None
    return [
        _make_record(row, entailment_text, "entailment", "entailment_synonym", entailment_rule or "synonym"),
        _make_record(row, neutral_text, "neutral", neutral_family or "neutral", neutral_rule or "neutral"),
        _make_record(row, contradiction_text, "contradiction", contradiction_family or "contradiction", contradiction_rule or "contradiction"),
    ]


def _assign_splits(family_ids: list[str], split_ratio: list[float], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = family_ids.copy()
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * split_ratio[0])
    val_end = train_end + int(len(shuffled) * split_ratio[1])
    split_lookup: dict[str, str] = {}
    for family_id in shuffled[:train_end]:
        split_lookup[family_id] = "train"
    for family_id in shuffled[train_end:val_end]:
        split_lookup[family_id] = "val"
    for family_id in shuffled[val_end:]:
        split_lookup[family_id] = "test"
    return split_lookup


def build_benchmark(
    coco_frame: pd.DataFrame,
    family_limit: int,
    split_ratio: list[float],
    seed: int,
) -> BenchmarkBuildResult:
    """Construct benchmark records from COCO caption rows."""

    print(f"Building benchmark from {family_limit} source caption families")
    family_rows = coco_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    categories = sorted({category for vocab in family_rows["category_vocab"] for category in vocab})

    records: list[dict] = []
    kept_families = 0
    for _, row in family_rows.iterrows():
        family_records = _family_records(row, categories)
        if family_records is None:
            continue
        records.extend(family_records)
        kept_families += 1
        if kept_families >= family_limit:
            break

    benchmark = pd.DataFrame(records)
    split_lookup = _assign_splits(sorted(benchmark["family_id"].unique().tolist()), split_ratio, seed)
    benchmark["split"] = benchmark["family_id"].map(split_lookup)
    family_manifest = benchmark[["family_id", "image_id", "split"]].drop_duplicates().sort_values("family_id")
    print(f"Built {len(benchmark)} benchmark rows across {family_manifest.shape[0]} families")
    return BenchmarkBuildResult(records=benchmark, family_manifest=family_manifest)


def sample_qwen_subset(records: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    """Draw a fixed stratified subset for Qwen evaluation."""

    grouped = records.groupby(["label", "edit_family"], group_keys=False)
    target_per_group = max(subset_size // max(grouped.ngroups, 1), 1)
    sampled = grouped.apply(lambda frame: frame.sample(min(len(frame), target_per_group), random_state=seed))
    if len(sampled) > subset_size:
        sampled = sampled.sample(subset_size, random_state=seed)
    return sampled.sort_values("sample_id").reset_index(drop=True)
