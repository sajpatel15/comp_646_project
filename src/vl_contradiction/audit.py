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


def audit_readiness(
    audit_sheet: pd.DataFrame,
    *,
    overall_label_valid_threshold: float,
    overall_grammar_ok_threshold: float,
    per_family_label_valid_threshold: float,
    require_all_rows_reviewed: bool = True,
) -> dict[str, object]:
    """Evaluate whether the audit sheet is complete and meets readiness thresholds."""

    frame = audit_sheet.copy()
    frame["reviewed_label"] = frame["reviewed_label"].astype(str).str.strip()
    frame["label_valid"] = frame["label_valid"].astype(str).str.lower().map({"true": True, "false": False})
    frame["grammar_ok"] = frame["grammar_ok"].astype(str).str.lower().map({"true": True, "false": False})

    unresolved_mask = (
        frame["reviewed_label"].eq("")
        | frame["label_valid"].isna()
        | frame["grammar_ok"].isna()
    )
    resolved = frame.loc[~unresolved_mask].copy()
    summary = summarize_audit(resolved) if not resolved.empty else pd.DataFrame(
        columns=["edit_family", "samples", "label_valid_rate", "grammar_ok_rate"]
    )

    overall_label_valid_rate = float(resolved["label_valid"].mean()) if not resolved.empty else 0.0
    overall_grammar_ok_rate = float(resolved["grammar_ok"].mean()) if not resolved.empty else 0.0

    failing_families = summary.loc[
        summary["label_valid_rate"] < per_family_label_valid_threshold,
        "edit_family",
    ].tolist()

    reasons: list[str] = []
    unresolved_rows = int(unresolved_mask.sum())
    if require_all_rows_reviewed and unresolved_rows > 0:
        reasons.append(f"{unresolved_rows} audit rows are still unresolved")
    if overall_label_valid_rate < overall_label_valid_threshold:
        reasons.append(
            f"overall label_valid_rate {overall_label_valid_rate:.3f} is below {overall_label_valid_threshold:.3f}"
        )
    if overall_grammar_ok_rate < overall_grammar_ok_threshold:
        reasons.append(
            f"overall grammar_ok_rate {overall_grammar_ok_rate:.3f} is below {overall_grammar_ok_threshold:.3f}"
        )
    if failing_families:
        joined = ", ".join(sorted(failing_families))
        reasons.append(f"per-family label validity failed for: {joined}")

    return {
        "passed": not reasons,
        "rows_total": int(len(frame)),
        "rows_reviewed": int(len(resolved)),
        "unresolved_rows": unresolved_rows,
        "overall_label_valid_rate": overall_label_valid_rate,
        "overall_grammar_ok_rate": overall_grammar_ok_rate,
        "failing_families": failing_families,
        "reasons": reasons,
    }
