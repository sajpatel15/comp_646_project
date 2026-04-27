"""Benchmark generation helpers for binary entailment and contradiction labels."""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .labels import CLASS_ORDER


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
    "cow": "animal",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
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

CONTRADICTION_OBJECT_MAP = {
    "dog": "cat",
    "cat": "dog",
    "horse": "cow",
    "cow": "horse",
    "bicycle": "motorcycle",
    "motorcycle": "bicycle",
    "bus": "train",
    "train": "bus",
    "couch": "bench",
    "bench": "couch",
    "tv": "laptop",
    "laptop": "tv",
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
IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "sheep": "sheep",
}
IRREGULAR_SINGULARS = {plural: singular for singular, plural in IRREGULAR_PLURALS.items()}

ATTRIBUTE_WORDS = set(ATTRIBUTE_FLIPS)
CONTRADICTION_FAMILIES = (
    "contradiction_action",
    "contradiction_attribute",
    "contradiction_count",
    "contradiction_object",
)
PROTECTED_PHRASES = (
    "old fashioned",
    "black and white",
    "white and black",
    "hot dog",
    "cell phone",
    "traffic light",
    "party bus",
    "double decker bus",
    "double decker",
)
ARTICLE_EXCEPTIONS = ("honest", "hour", "heir")
CONSONANT_SOUND_EXCEPTIONS = ("uni", "use", "user", "euro", "one")
SKIPPABLE_NUMBER_MODIFIERS = ATTRIBUTE_WORDS | {
    "other",
    "elder",
    "elderly",
    "single",
    "double",
    "mini",
    "colorful",
    "colored",
    "brown",
    "green",
    "orange",
    "yellow",
    "gray",
    "grey",
}


@dataclass(slots=True)
class BenchmarkBuildResult:
    records: pd.DataFrame
    family_manifest: pd.DataFrame
    coverage_summary: pd.DataFrame


@dataclass(slots=True)
class _CandidatePack:
    row: dict[str, Any]
    entailment: tuple[str, str] | None
    contradictions: dict[str, tuple[str, str]]


def _normalize_whitespace(text: str) -> str:
    compact = re.sub(r"\s+", " ", text)
    compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
    return compact.strip(" ,.")


def _word_pattern(phrase: str) -> str:
    parts = [re.escape(part) for part in phrase.split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def _token_key(token: str) -> str:
    return re.sub(r"(^[^A-Za-z]+|[^A-Za-z]+$)", "", token).lower()


def _replace_token_core(token: str, replacement: str) -> str:
    match = re.match(r"(^[^A-Za-z]*)(.*?)([^A-Za-z]*$)", token)
    if not match:
        return replacement
    prefix, core, suffix = match.groups()
    if core.isupper():
        replacement = replacement.upper()
    elif core[:1].isupper():
        replacement = replacement.capitalize()
    return f"{prefix}{replacement}{suffix}"


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    for phrase in PROTECTED_PHRASES:
        pattern = re.compile(_word_pattern(phrase), flags=re.IGNORECASE)
        spans.extend(match.span() for match in pattern.finditer(lower))
    return spans


def _span_is_protected(span: tuple[int, int], protected_spans: list[tuple[int, int]]) -> bool:
    return any(max(span[0], start) < min(span[1], end) for start, end in protected_spans)


def _starts_with_vowel_sound(token: str) -> bool:
    lowered = token.lower()
    if lowered.startswith(ARTICLE_EXCEPTIONS):
        return True
    if lowered.startswith(CONSONANT_SOUND_EXCEPTIONS):
        return False
    return lowered[:1] in {"a", "e", "i", "o", "u"}


def _singularize_word(word: str) -> str:
    lowered = word.lower()
    if lowered in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[lowered]
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("es") and lowered[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 1 and not lowered.endswith(("ss", "us")):
        return lowered[:-1]
    return lowered


def _pluralize_word(word: str) -> str:
    lowered = word.lower()
    if lowered in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[lowered]
    if lowered in IRREGULAR_SINGULARS:
        return lowered
    if _singularize_word(lowered) != lowered:
        return lowered
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return lowered + "es"
    if lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in {"a", "e", "i", "o", "u"}:
        return lowered[:-1] + "ies"
    return lowered + "s"


def _pluralize_phrase(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    parts[-1] = _pluralize_word(parts[-1])
    return " ".join(parts)


def _next_count_noun_index(tokens: list[str], start_index: int) -> int | None:
    skips = 0
    for index in range(start_index, len(tokens)):
        core = _token_key(tokens[index])
        if not core:
            continue
        if core in SKIPPABLE_NUMBER_MODIFIERS and skips < 3:
            skips += 1
            continue
        return index
    return None


def _apply_article_agreement(text: str) -> str:
    tokens = text.split()
    for index in range(len(tokens) - 1):
        article = _token_key(tokens[index])
        if article not in {"a", "an"}:
            continue
        next_token = _token_key(tokens[index + 1])
        if not next_token:
            continue
        expected = "an" if _starts_with_vowel_sound(next_token) else "a"
        tokens[index] = _replace_token_core(tokens[index], expected)
    return " ".join(tokens)


def _apply_number_agreement(text: str) -> str:
    tokens = text.split()
    for index, token in enumerate(tokens):
        number_word = _token_key(token)
        if number_word not in NUMBER_WORDS:
            continue
        noun_index = _next_count_noun_index(tokens, index + 1)
        if noun_index is None:
            continue
        noun_core = _token_key(tokens[noun_index])
        if not noun_core:
            continue
        singular = NUMBER_WORDS[number_word] == 1
        replacement = _singularize_word(noun_core) if singular else _pluralize_word(noun_core)
        tokens[noun_index] = _replace_token_core(tokens[noun_index], replacement)

        if noun_index + 1 >= len(tokens):
            continue
        verb_core = _token_key(tokens[noun_index + 1])
        if singular and verb_core in {"are", "were"}:
            verb = "is" if verb_core == "are" else "was"
            tokens[noun_index + 1] = _replace_token_core(tokens[noun_index + 1], verb)
        if not singular and verb_core in {"is", "was"}:
            verb = "are" if verb_core == "is" else "were"
            tokens[noun_index + 1] = _replace_token_core(tokens[noun_index + 1], verb)
    return " ".join(tokens)


def _normalize_caption(text: str) -> str:
    normalized = _normalize_whitespace(text)
    normalized = _apply_number_agreement(normalized)
    normalized = _apply_article_agreement(normalized)
    return _normalize_whitespace(normalized)


def _replace_first_safe(text: str, source: str, target: str) -> str | None:
    pattern = re.compile(_word_pattern(source), flags=re.IGNORECASE)
    protected_spans = _protected_spans(text)
    for match in pattern.finditer(text):
        if _span_is_protected(match.span(), protected_spans):
            continue
        replacement = _match_case(match.group(0), target)
        updated = text[: match.start()] + replacement + text[match.end() :]
        return _normalize_caption(updated)
    return None


def _remove_first_safe(text: str, source: str) -> str | None:
    pattern = re.compile(_word_pattern(source), flags=re.IGNORECASE)
    protected_spans = _protected_spans(text)
    for match in pattern.finditer(text):
        if _span_is_protected(match.span(), protected_spans):
            continue
        updated = text[: match.start()] + text[match.end() :]
        return _normalize_caption(updated)
    return None


def _entailment_candidate(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in SYNONYM_MAP.items():
        if re.search(_word_pattern(source), lower):
            updated = _replace_first_safe(caption, source, target)
            if updated and updated.lower() != caption.lower():
                return updated, f"synonym:{source}->{target}"
    return None, None


def _count_contradiction(caption: str, object_counts: dict[str, int]) -> tuple[str | None, str | None]:
    tokens = caption.split()
    for index, token in enumerate(tokens):
        number_word = _token_key(token)
        if number_word not in NUMBER_WORDS:
            continue
        replacement_value = 1 if NUMBER_WORDS[number_word] != 1 else 2
        tokens[index] = _replace_token_core(token, NUMBER_LOOKUP[replacement_value])
        updated = _normalize_caption(" ".join(tokens))
        return updated, f"count:{number_word}->{NUMBER_LOOKUP[replacement_value]}"

    protected_spans = _protected_spans(caption)
    for object_name, count in sorted(object_counts.items(), key=lambda item: (-len(item[0]), item[0])):
        if count <= 1:
            continue
        pattern = re.compile(rf"\b(a|an)\s+{re.escape(object_name)}\b", flags=re.IGNORECASE)
        for match in pattern.finditer(caption):
            if _span_is_protected(match.span(), protected_spans):
                continue
            replacement = f"two {_pluralize_phrase(object_name)}"
            if match.group(0)[:1].isupper():
                replacement = replacement.capitalize()
            updated = caption[: match.start()] + replacement + caption[match.end() :]
            return _normalize_caption(updated), f"count:article->{object_name}"
    return None, None


def _attribute_contradiction(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in ATTRIBUTE_FLIPS.items():
        if re.search(_word_pattern(source), lower):
            updated = _replace_first_safe(caption, source, target)
            if updated and updated.lower() != caption.lower():
                return updated, f"attribute:{source}->{target}"
    return None, None


def _object_contradiction(caption: str, present_objects: list[str]) -> tuple[str | None, str | None]:
    lower = caption.lower()
    present_set = set(present_objects)
    for source, target in CONTRADICTION_OBJECT_MAP.items():
        if source not in present_set or target in present_set:
            continue
        if not re.search(_word_pattern(source), lower):
            continue
        updated = _replace_first_safe(caption, source, target)
        if updated and updated.lower() != caption.lower():
            return updated, f"object:{source}->{target}"
    return None, None


def _action_contradiction(caption: str) -> tuple[str | None, str | None]:
    lower = caption.lower()
    for source, target in ACTION_FLIPS.items():
        if re.search(_word_pattern(source), lower):
            updated = _replace_first_safe(caption, source, target)
            if updated and updated.lower() != caption.lower():
                return updated, f"action:{source}->{target}"
    return None, None


def _contradiction_candidates(row: pd.Series) -> dict[str, tuple[str, str]]:
    caption = row["caption"]
    candidates: dict[str, tuple[str, str]] = {}

    count_candidate = _count_contradiction(caption, row["object_counts"])
    if all(count_candidate):
        candidates["contradiction_count"] = (count_candidate[0], count_candidate[1])

    attribute_candidate = _attribute_contradiction(caption)
    if all(attribute_candidate):
        candidates["contradiction_attribute"] = (attribute_candidate[0], attribute_candidate[1])

    object_candidate = _object_contradiction(caption, row["objects"])
    if all(object_candidate):
        candidates["contradiction_object"] = (object_candidate[0], object_candidate[1])

    action_candidate = _action_contradiction(caption)
    if all(action_candidate):
        candidates["contradiction_action"] = (action_candidate[0], action_candidate[1])
    return candidates


def _make_record(row: dict[str, Any], edited_caption: str, label: str, edit_family: str, edit_rule: str) -> dict[str, Any]:
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


def _candidate_pack(row: pd.Series) -> _CandidatePack | None:
    entailment = _entailment_candidate(row["caption"])
    contradictions = _contradiction_candidates(row)
    if not all(entailment) or not contradictions:
        return None
    return _CandidatePack(
        row={
            "family_id": row["family_id"],
            "image_id": row["image_id"],
            "caption": row["caption"],
            "file_path": row["file_path"],
            "objects": row["objects"],
            "object_counts": row["object_counts"],
        },
        entailment=(entailment[0], entailment[1]),
        contradictions=contradictions,
    )


def _target_counts(total: int, families: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(total, len(families))
    targets = {family: base for family in families}
    for family in families[:remainder]:
        targets[family] += 1
    return targets


def _choose_contradiction(
    pack: _CandidatePack,
    contradiction_targets: dict[str, int],
    contradiction_counts: Counter[str],
    require_deficit: bool,
) -> str | None:
    best_choice: str | None = None
    best_score: tuple[int, int] | None = None
    for contradiction_family in pack.contradictions:
        contradiction_gap = contradiction_targets[contradiction_family] - contradiction_counts[contradiction_family]
        if require_deficit and contradiction_gap <= 0:
            continue
        score = (
            contradiction_gap,
            -contradiction_counts[contradiction_family],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_choice = contradiction_family
    return best_choice


def _select_balanced_records(candidate_packs: list[_CandidatePack], family_limit: int) -> list[dict[str, Any]]:
    contradiction_targets = _target_counts(family_limit, CONTRADICTION_FAMILIES)
    contradiction_counts: Counter[str] = Counter()
    selected_records: list[dict[str, Any]] = []
    used_family_ids: set[str] = set()
    remaining: list[_CandidatePack] = []

    for pack in candidate_packs:
        if len(selected_records) >= family_limit * 2:
            break
        choice = _choose_contradiction(pack, contradiction_targets, contradiction_counts, require_deficit=True)
        if choice is None:
            remaining.append(pack)
            continue
        used_family_ids.add(pack.row["family_id"])
        contradiction_counts[choice] += 1
        selected_records.extend(
            [
                _make_record(pack.row, pack.entailment[0], "entailment", "entailment_synonym", pack.entailment[1]),
                _make_record(pack.row, pack.contradictions[choice][0], "contradiction", choice, pack.contradictions[choice][1]),
            ]
        )

    if len(selected_records) >= family_limit * 2:
        return selected_records

    for pack in remaining:
        if len(selected_records) >= family_limit * 2:
            break
        if pack.row["family_id"] in used_family_ids:
            continue
        choice = _choose_contradiction(pack, contradiction_targets, contradiction_counts, require_deficit=False)
        if choice is None:
            continue
        used_family_ids.add(pack.row["family_id"])
        contradiction_counts[choice] += 1
        selected_records.extend(
            [
                _make_record(pack.row, pack.entailment[0], "entailment", "entailment_synonym", pack.entailment[1]),
                _make_record(pack.row, pack.contradictions[choice][0], "contradiction", choice, pack.contradictions[choice][1]),
            ]
        )
    return selected_records


def summarize_family_coverage(candidate_packs: list[_CandidatePack], records: pd.DataFrame, family_limit: int) -> pd.DataFrame:
    """Summarize candidate availability and selected coverage for each edit family."""

    contradiction_targets = _target_counts(family_limit, CONTRADICTION_FAMILIES)
    candidate_counts: Counter[str] = Counter({"entailment_synonym": sum(1 for pack in candidate_packs if pack.entailment)})
    for family in CONTRADICTION_FAMILIES:
        candidate_counts[family] = sum(1 for pack in candidate_packs if family in pack.contradictions)

    selected_counts = Counter(records["edit_family"]) if not records.empty else Counter()
    rows = [
        {
            "label": "entailment",
            "edit_family": "entailment_synonym",
            "candidate_count": int(candidate_counts["entailment_synonym"]),
            "selected_count": int(selected_counts["entailment_synonym"]),
            "target_count": int(family_limit),
            "meets_target": int(selected_counts["entailment_synonym"]) >= int(family_limit),
        }
    ]
    for family in CONTRADICTION_FAMILIES:
        rows.append(
            {
                "label": "contradiction",
                "edit_family": family,
                "candidate_count": int(candidate_counts[family]),
                "selected_count": int(selected_counts[family]),
                "target_count": int(contradiction_targets[family]),
                "meets_target": int(selected_counts[family]) >= int(contradiction_targets[family]),
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "edit_family"]).reset_index(drop=True)


def _assign_splits(family_manifest: pd.DataFrame, split_ratio: list[float], seed: int) -> dict[str, str]:
    split_names = ["train", "val", "test"]
    if family_manifest.empty:
        return {}

    unique_families = family_manifest[["family_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    image_groups = (
        unique_families.groupby("image_id", sort=False)["family_id"]
        .agg(lambda values: sorted(values.tolist()))
        .reset_index()
    )
    image_groups["family_count"] = image_groups["family_id"].str.len()

    rng = np.random.default_rng(seed)
    image_groups["_tie_breaker"] = rng.permutation(len(image_groups))
    image_groups = image_groups.sort_values(
        by=["family_count", "_tie_breaker"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    total_families = int(unique_families["family_id"].nunique())
    target_counts = {
        split_name: total_families * float(split_ratio[index])
        for index, split_name in enumerate(split_names)
    }
    assigned_counts = {split_name: 0 for split_name in split_names}
    split_lookup: dict[str, str] = {}

    for _, row in image_groups.iterrows():
        deficits = {
            split_name: target_counts[split_name] - assigned_counts[split_name]
            for split_name in split_names
        }
        chosen_split = max(split_names, key=lambda split_name: (deficits[split_name], -split_names.index(split_name)))
        for family_id in row["family_id"]:
            split_lookup[str(family_id)] = chosen_split
        assigned_counts[chosen_split] += int(row["family_count"])

    return split_lookup


def build_benchmark(
    coco_frame: pd.DataFrame,
    family_limit: int,
    split_ratio: list[float],
    seed: int,
) -> BenchmarkBuildResult:
    """Construct a balanced benchmark from COCO caption rows."""

    print(f"Building benchmark from {family_limit} source caption families")
    family_rows = coco_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    candidate_packs = [pack for _, row in family_rows.iterrows() if (pack := _candidate_pack(row)) is not None]

    selected_records = _select_balanced_records(candidate_packs, family_limit)
    benchmark = pd.DataFrame(selected_records)
    coverage_summary = summarize_family_coverage(candidate_packs, benchmark, family_limit)

    if benchmark.empty:
        family_manifest = pd.DataFrame(columns=["family_id", "image_id", "split"])
        print("Benchmark generation returned zero rows after safe edit filtering.")
        print(coverage_summary)
        return BenchmarkBuildResult(records=benchmark, family_manifest=family_manifest, coverage_summary=coverage_summary)

    family_manifest = benchmark[["family_id", "image_id"]].drop_duplicates().sort_values("family_id")
    split_lookup = _assign_splits(family_manifest, split_ratio, seed)
    benchmark["split"] = benchmark["family_id"].map(split_lookup)
    family_manifest = family_manifest.assign(split=family_manifest["family_id"].map(split_lookup))
    print(f"Built {len(benchmark)} benchmark rows across {family_manifest.shape[0]} families")
    print("Family coverage summary")
    print(coverage_summary)
    return BenchmarkBuildResult(records=benchmark, family_manifest=family_manifest, coverage_summary=coverage_summary)


def sample_comparison_subset(records: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    """Draw a fixed stratified subset for apples-to-apples model comparison."""

    filtered = records.loc[records["label"].isin(CLASS_ORDER)].copy()
    if filtered.empty or subset_size <= 0:
        return filtered.sort_values("sample_id").reset_index(drop=True)
    if subset_size >= len(filtered):
        return filtered.sort_values("sample_id").reset_index(drop=True)

    grouped_frames = list(filtered.groupby(["label", "edit_family"], sort=True))
    target_per_group = max(subset_size // max(len(grouped_frames), 1), 1)
    selected_indices: list[int] = []
    leftover_indices_by_group: list[deque[int]] = []

    for group_offset, (_, frame) in enumerate(grouped_frames):
        shuffled = frame.sample(frac=1.0, random_state=seed + group_offset)
        base_count = min(len(shuffled), target_per_group)
        selected_indices.extend(shuffled.index[:base_count].tolist())
        leftover_indices_by_group.append(deque(shuffled.index[base_count:].tolist()))

    remaining = subset_size - len(selected_indices)
    while remaining > 0 and any(leftover_indices_by_group):
        for leftover_indices in leftover_indices_by_group:
            if remaining <= 0:
                break
            if not leftover_indices:
                continue
            selected_indices.append(leftover_indices.popleft())
            remaining -= 1

    sampled = filtered.loc[selected_indices]
    return sampled.sort_values("sample_id").reset_index(drop=True)


def sample_qwen_subset(records: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    """Backward-compatible alias for the shared comparison subset sampler."""

    return sample_comparison_subset(records, subset_size=subset_size, seed=seed)
