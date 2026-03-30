"""Benchmark construction and audit helpers."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io_utils import stable_hash
from .schemas import BenchmarkRecord

ENTAILMENT_RULES = [
    (r"\bbicycle\b", "bike", "bicycle_to_bike"),
    (r"\bbikes\b", "bicycles", "bikes_to_bicycles"),
    (r"\bcouch\b", "sofa", "couch_to_sofa"),
    (r"\bsofa\b", "couch", "sofa_to_couch"),
    (r"\bairplane\b", "plane", "airplane_to_plane"),
    (r"\bcell phone\b", "phone", "cell_phone_to_phone"),
    (r"\btelevision\b", "tv", "television_to_tv"),
    (r"\bmotorcycle\b", "motorbike", "motorcycle_to_motorbike"),
]

HYPERNYM_RULES = [
    (r"\bdog\b", "animal", "dog_to_animal"),
    (r"\bcat\b", "animal", "cat_to_animal"),
    (r"\bhorse\b", "animal", "horse_to_animal"),
    (r"\bcow\b", "animal", "cow_to_animal"),
    (r"\bcar\b", "vehicle", "car_to_vehicle"),
    (r"\bbus\b", "vehicle", "bus_to_vehicle"),
    (r"\btruck\b", "vehicle", "truck_to_vehicle"),
    (r"\bbicycle\b", "vehicle", "bicycle_to_vehicle"),
    (r"\bapple\b", "fruit", "apple_to_fruit"),
    (r"\bbanana\b", "fruit", "banana_to_fruit"),
    (r"\borange\b", "fruit", "orange_to_fruit"),
    (r"\bcouch\b", "furniture", "couch_to_furniture"),
    (r"\bchair\b", "furniture", "chair_to_furniture"),
]

ATTRIBUTE_WORDS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "white",
    "black",
    "brown",
    "gray",
    "grey",
    "small",
    "large",
    "little",
    "big",
    "young",
    "old",
    "wooden",
]

ATTRIBUTE_FLIPS = {
    "red": "blue",
    "blue": "red",
    "green": "purple",
    "black": "white",
    "white": "black",
    "small": "large",
    "large": "small",
    "young": "old",
    "old": "young",
}

OBJECT_SWAP_RULES = {
    "dog": "cat",
    "cat": "dog",
    "car": "bus",
    "bus": "car",
    "truck": "bicycle",
    "bicycle": "motorcycle",
    "motorcycle": "bicycle",
    "horse": "cow",
    "cow": "horse",
    "apple": "banana",
    "banana": "apple",
    "chair": "bench",
    "bench": "chair",
}

ACTION_FLIPS = {
    "standing": "sitting",
    "sitting": "standing",
    "running": "sleeping",
    "sleeping": "running",
    "holding": "throwing",
    "throwing": "holding",
    "eating": "drinking",
    "drinking": "eating",
    "walking": "lying",
    "lying": "walking",
    "riding": "pushing",
    "pushing": "riding",
}

COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

NUMBER_WORDS = {value: key for key, value in COUNT_WORDS.items() if key not in {"a", "an"}}


def _clean_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\ba ([aeiou])", r"an \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban ([^aeiouAEIOU\s])", r"a \1", text)
    return text


def _apply_substitution(text: str, pattern: str, replacement: str) -> str | None:
    edited, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    if count == 0 or edited == text:
        return None
    return _clean_caption(edited)


def _pluralize(noun: str) -> str:
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _apply_entailment_edit(caption: str) -> tuple[str, str] | None:
    for pattern, replacement, rule_name in ENTAILMENT_RULES:
        edited = _apply_substitution(caption, pattern, replacement)
        if edited:
            return edited, rule_name
    return None


def _apply_neutral_hypernym(caption: str) -> tuple[str, str] | None:
    for pattern, replacement, rule_name in HYPERNYM_RULES:
        edited = _apply_substitution(caption, pattern, replacement)
        if edited:
            return edited, rule_name
    return None


def _apply_neutral_attribute_removal(caption: str) -> tuple[str, str] | None:
    for word in ATTRIBUTE_WORDS:
        pattern = rf"\b{re.escape(word)}\b\s*"
        edited, count = re.subn(pattern, "", caption, count=1, flags=re.IGNORECASE)
        if count:
            return _clean_caption(edited), f"drop_{word}"
    return None


def _apply_attribute_flip(caption: str) -> tuple[str, str] | None:
    for source, target in ATTRIBUTE_FLIPS.items():
        edited = _apply_substitution(caption, rf"\b{re.escape(source)}\b", target)
        if edited:
            return edited, f"{source}_to_{target}"
    return None


def _apply_action_flip(caption: str) -> tuple[str, str] | None:
    for source, target in ACTION_FLIPS.items():
        edited = _apply_substitution(caption, rf"\b{re.escape(source)}\b", target)
        if edited:
            return edited, f"{source}_to_{target}"
    return None


def _apply_object_swap(caption: str, objects: Iterable[str]) -> tuple[str, str] | None:
    present_objects = {item.lower() for item in objects}
    for source, target in OBJECT_SWAP_RULES.items():
        if target in present_objects:
            continue
        edited = _apply_substitution(caption, rf"\b{re.escape(source)}\b", target)
        if edited:
            return edited, f"{source}_to_{target}"
    return None


def _apply_count_change(caption: str, object_counts: dict[str, int]) -> tuple[str, str] | None:
    lowered = caption.lower()
    for noun, count in object_counts.items():
        singular = noun.lower()
        plural = _pluralize(singular)
        for source_word, source_count in COUNT_WORDS.items():
            pattern = rf"\b{re.escape(source_word)}\s+{re.escape(plural if source_count > 1 else singular)}\b"
            if not re.search(pattern, lowered):
                continue
            target_count = 2 if count == 1 else 1
            replacement_noun = plural if target_count > 1 else singular
            replacement_word = NUMBER_WORDS.get(target_count, str(target_count))
            replacement = f"{replacement_word} {replacement_noun}"
            edited = re.sub(pattern, replacement, lowered, count=1)
            return _clean_caption(edited), f"{source_word}_{singular}_to_{replacement_word}"

    for match in re.finditer(r"\b(one|two|three|four|five)\s+([a-z]+)\b", lowered):
        source_word, noun = match.groups()
        target_count = 1 if COUNT_WORDS[source_word] > 1 else 2
        replacement_noun = noun if target_count == 1 else _pluralize(noun)
        replacement_word = NUMBER_WORDS.get(target_count, str(target_count))
        edited = lowered[: match.start()] + f"{replacement_word} {replacement_noun}" + lowered[match.end() :]
        return _clean_caption(edited), f"{source_word}_{noun}_to_{replacement_word}"
    return None


def _select_neutral_edit(caption: str) -> tuple[str, str, str] | None:
    hypernym = _apply_neutral_hypernym(caption)
    if hypernym:
        return hypernym[0], "neutral_hypernym", hypernym[1]
    attribute_drop = _apply_neutral_attribute_removal(caption)
    if attribute_drop:
        return attribute_drop[0], "neutral_attribute_removal", attribute_drop[1]
    return None


def _select_contradiction_edit(caption: str, row: pd.Series) -> tuple[str, str, str] | None:
    for family_name, builder in (
        ("contradiction_object_swap", lambda: _apply_object_swap(caption, row["objects"])),
        ("contradiction_count_change", lambda: _apply_count_change(caption, row["object_counts"])),
        ("contradiction_attribute_flip", lambda: _apply_attribute_flip(caption)),
        ("contradiction_action_swap", lambda: _apply_action_flip(caption)),
    ):
        result = builder()
        if result:
            return result[0], family_name, result[1]
    return None


def build_family_records(row: pd.Series) -> list[BenchmarkRecord] | None:
    source_caption = _clean_caption(str(row["caption"]))
    family_id = f"{row['source_split']}:{row['image_id']}:{row['caption_id']}"

    entailment = _apply_entailment_edit(source_caption)
    neutral = _select_neutral_edit(source_caption)
    contradiction = _select_contradiction_edit(source_caption, row)
    if not entailment or not neutral or not contradiction:
        return None

    records: list[BenchmarkRecord] = []
    for edited_caption, label, family_name, rule_name in (
        (entailment[0], "entailment", "entailment_substitution", entailment[1]),
        (neutral[0], "neutral", neutral[1], neutral[2]),
        (contradiction[0], "contradiction", contradiction[1], contradiction[2]),
    ):
        sample_key = f"{family_id}:{label}:{edited_caption}"
        records.append(
            BenchmarkRecord(
                sample_id=stable_hash(sample_key, length=16),
                family_id=family_id,
                image_id=int(row["image_id"]),
                source_split=str(row["source_split"]),
                file_name=str(row["file_name"]),
                source_caption=source_caption,
                edited_caption=edited_caption,
                label=label,
                edit_family=family_name,
                edit_rule=rule_name,
                split="unassigned",
            )
        )
    return records


def _assign_splits(family_ids: list[str], split_ratios: dict[str, float], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(family_ids, dtype=object)
    rng.shuffle(shuffled)

    train_cutoff = int(len(shuffled) * split_ratios["train"])
    val_cutoff = train_cutoff + int(len(shuffled) * split_ratios["val"])
    assignments: dict[str, str] = {}
    for family_id in shuffled[:train_cutoff]:
        assignments[str(family_id)] = "train"
    for family_id in shuffled[train_cutoff:val_cutoff]:
        assignments[str(family_id)] = "val"
    for family_id in shuffled[val_cutoff:]:
        assignments[str(family_id)] = "test"
    return assignments


def build_benchmark_frame(
    coco_table: pd.DataFrame,
    *,
    family_limit: int,
    split_ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    candidate_rows = coco_table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    families: list[list[BenchmarkRecord]] = []

    for _, row in candidate_rows.iterrows():
        family_records = build_family_records(row)
        if family_records is None:
            continue
        families.append(family_records)
        if len(families) >= family_limit:
            break

    if not families:
        raise ValueError("No valid caption families were generated from the provided COCO table.")

    family_ids = [records[0].family_id for records in families]
    split_map = _assign_splits(family_ids, split_ratios, seed)

    payload = []
    for records in families:
        for record in records:
            record.split = split_map[record.family_id]
            payload.append(asdict(record))
    return pd.DataFrame(payload)


def build_audit_sheet(benchmark_frame: pd.DataFrame, *, per_edit_family: int, seed: int) -> pd.DataFrame:
    samples = []
    rng = np.random.default_rng(seed)
    for edit_family, family_frame in benchmark_frame.groupby("edit_family"):
        take = min(per_edit_family, len(family_frame))
        indices = rng.choice(family_frame.index.to_numpy(), size=take, replace=False)
        sampled = family_frame.loc[indices].copy()
        sampled["label_valid"] = ""
        sampled["grammar_ok"] = ""
        sampled["review_notes"] = ""
        samples.append(sampled)
    return pd.concat(samples, axis=0, ignore_index=True).sort_values(["edit_family", "sample_id"])


def summarize_audit_sheet(audit_frame: pd.DataFrame) -> pd.DataFrame:
    review_frame = audit_frame.copy()
    review_frame["label_valid"] = review_frame["label_valid"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    review_frame["grammar_ok"] = review_frame["grammar_ok"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    summary = (
        review_frame.groupby("edit_family")
        .agg(
            samples=("sample_id", "count"),
            label_valid_rate=("label_valid", "mean"),
            grammar_ok_rate=("grammar_ok", "mean"),
        )
        .reset_index()
    )
    summary["label_valid_rate"] = summary["label_valid_rate"].round(4)
    summary["grammar_ok_rate"] = summary["grammar_ok_rate"].round(4)
    return summary


def select_fixed_subset(frame: pd.DataFrame, *, size: int, seed: int) -> pd.DataFrame:
    take = min(size, len(frame))
    if take == 0:
        raise ValueError("Cannot select a subset from an empty frame.")
    sampled = (
        frame.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=max(1, take // frame["label"].nunique()), random_state=seed))
        .reset_index(drop=True)
    )
    if len(sampled) < take:
        missing = take - len(sampled)
        remainder = frame[~frame["sample_id"].isin(sampled["sample_id"])]
        if not remainder.empty:
            sampled = pd.concat(
                [sampled, remainder.sample(n=min(missing, len(remainder)), random_state=seed)],
                axis=0,
                ignore_index=True,
            )
    return sampled.head(take).reset_index(drop=True)
