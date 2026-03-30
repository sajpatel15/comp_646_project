"""Audit helpers for benchmark validation."""

from __future__ import annotations

import pandas as pd


def build_audit_sheet(records: pd.DataFrame, per_family: int, seed: int) -> pd.DataFrame:
    """Sample records for manual audit and add reviewer columns."""

    sampled = (
        records.groupby("edit_family", group_keys=False)
        .apply(lambda frame: frame.sample(min(len(frame), per_family), random_state=seed))
        .reset_index(drop=True)
    )
    sampled["reviewed_label"] = ""
    sampled["label_valid"] = ""
    sampled["grammar_ok"] = ""
    sampled["notes"] = ""
    return sampled[
        [
            "sample_id",
            "family_id",
            "image_id",
            "label",
            "edit_family",
            "edit_rule",
            "source_caption",
            "edited_caption",
            "reviewed_label",
            "label_valid",
            "grammar_ok",
            "notes",
        ]
    ]


def summarize_audit(audit_sheet: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed audit rows when reviewer fields have been filled."""

    frame = audit_sheet.copy()
    frame["label_valid"] = frame["label_valid"].astype(str).str.lower().map({"true": True, "false": False})
    frame["grammar_ok"] = frame["grammar_ok"].astype(str).str.lower().map({"true": True, "false": False})
    summary = (
        frame.groupby("edit_family")
        .agg(
            samples=("sample_id", "count"),
            label_valid_rate=("label_valid", "mean"),
            grammar_ok_rate=("grammar_ok", "mean"),
        )
        .reset_index()
    )
    return summary
