from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def create_audit_sample(
    benchmark_frame: pd.DataFrame,
    per_edit_family: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for _, group in benchmark_frame.groupby("edit_family"):
        sampled_frames.append(
            group.sample(n=min(per_edit_family, len(group)), random_state=seed)
        )
    audit_frame = pd.concat(sampled_frames, ignore_index=True)
    audit_frame["label_valid"] = ""
    audit_frame["grammar_ok"] = ""
    audit_frame["notes"] = ""
    return audit_frame.sort_values(["label", "edit_family", "family_id"]).reset_index(drop=True)


def summarize_audit(audit_frame: pd.DataFrame) -> dict:
    valid = audit_frame["label_valid"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    grammar = audit_frame["grammar_ok"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    per_family = []
    for family, family_frame in audit_frame.groupby("edit_family"):
        family_valid = family_frame["label_valid"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        family_grammar = family_frame["grammar_ok"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        per_family.append(
            {
                "edit_family": family,
                "label_valid_rate": float(family_valid.mean()) if len(family_valid) else 0.0,
                "grammar_rate": float(family_grammar.mean()) if len(family_grammar) else 0.0,
            }
        )
    return {
        "overall_label_valid_rate": float(valid.mean()) if len(valid) else 0.0,
        "overall_grammar_rate": float(grammar.mean()) if len(grammar) else 0.0,
        "per_family": per_family,
    }


def audit_passes(
    summary: dict,
    label_threshold: float = 0.90,
    family_threshold: float = 0.80,
    grammar_threshold: float = 0.90,
) -> bool:
    if summary["overall_label_valid_rate"] < label_threshold:
        return False
    if summary["overall_grammar_rate"] < grammar_threshold:
        return False
    return all(
        family["label_valid_rate"] >= family_threshold for family in summary["per_family"]
    )
