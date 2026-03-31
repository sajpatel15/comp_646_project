"""Heuristic prefill helpers for reducing manual audit workload."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


VOWELS = set("aeiou")
ARTICLE_EXCEPTIONS = ("hour", "honest", "heir")
FLAG_PREFIX = "AUTO FLAG:"
FILL_PREFIX = "AUTO:"
FRUIT_HYPERNYM_RULES = {
    "hypernym:apple->fruit",
    "hypernym:banana->fruit",
    "hypernym:orange->fruit",
}
INVARIANT_PLURALS = {"sheep"}


def _starts_with_vowel_sound(token: str) -> bool:
    lowered = token.lower()
    return lowered.startswith(ARTICLE_EXCEPTIONS) or lowered[:1] in VOWELS


def _append_note(existing: str, prefix: str, reasons: list[str]) -> str:
    if not reasons:
        return existing
    message = f"{prefix} " + "; ".join(reasons)
    if not existing:
        return message
    if message in existing:
        return existing
    return f"{existing} | {message}"


def _next_token(text: str, article: str) -> str | None:
    match = re.search(rf"\b{article}\s+([A-Za-z][A-Za-z-]*)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _grammar_issues(row: pd.Series) -> list[str]:
    edited = str(row.get("edited_caption", "")).strip()
    lower = edited.lower()
    issues: list[str] = []

    token_after_a = _next_token(edited, "a")
    if token_after_a and _starts_with_vowel_sound(token_after_a):
        issues.append(f'expected "an" before "{token_after_a}"')

    token_after_an = _next_token(edited, "an")
    if token_after_an and not _starts_with_vowel_sound(token_after_an):
        issues.append(f'expected "a" before "{token_after_an}"')

    if re.search(r"\bone\s+(people|men|women|children|hands)\b", lower):
        issues.append("count agreement error after singular number")

    if re.search(r"\b(two|three|four|five|six)\s+(person|man|woman|child|motorcycle|bicycle|horse|dog|cat|car|bus|bird|kite|pizza|traffic light)\b", lower):
        issues.append("count agreement error after plural number")

    if re.search(r"\b(a|an)\s+and\b", lower):
        issues.append("dangling conjunction after attribute drop")

    if "fashioned" in lower and "old fashioned" not in lower:
        issues.append('dangling "fashioned" phrase')

    if re.search(r"\bhaired\b", lower) and not re.search(r"\b[a-z-]+\s+haired\b", lower):
        issues.append('dangling "haired" phrase')

    if re.search(r"\b(and white|and black)\b", lower) and "black and white" not in lower and "white and black" not in lower:
        issues.append("broken color coordination")

    edit_rule = str(row.get("edit_rule", ""))
    count_article_match = re.fullmatch(r"count:article->(.+)", edit_rule)
    if count_article_match:
        noun = count_article_match.group(1).strip().lower()
        if noun not in INVARIANT_PLURALS and " " not in noun and not noun.endswith("s"):
            issues.append(f'expected plural noun after "two {noun}"')

    if "fruit blanket" in lower or "fruit shirt" in lower or "fruit pants" in lower or "fruit vase" in lower:
        issues.append("unlikely fruit noun substitution")

    return list(dict.fromkeys(issues))


def _needs_manual_review(row: pd.Series, issues: list[str]) -> list[str]:
    reasons: list[str] = []
    edit_rule = str(row.get("edit_rule", ""))
    edit_family = str(row.get("edit_family", ""))
    lower = str(row.get("edited_caption", "")).lower()

    if edit_rule in FRUIT_HYPERNYM_RULES:
        reasons.append("fruit hypernym substitution can change sense, not just specificity")

    if edit_family == "neutral_attribute_drop" and any(
        phrase in lower for phrase in ("fashioned", "haired", "and white", "and black")
    ):
        reasons.append("attribute drop may have broken a fixed phrase")

    if edit_family == "contradiction_object" and re.search(r"\bairplane\s+(party|leathers|seat|stop)\b", lower):
        reasons.append("object replacement landed in a suspicious compound phrase")

    if any("unlikely fruit noun substitution" in issue for issue in issues):
        reasons.append("edited caption looks semantically suspicious")

    return list(dict.fromkeys(reasons))


def auto_fill_audit_sheet(
    audit_csv_path: str | Path,
    *,
    overwrite_existing: bool = False,
) -> dict[str, int]:
    """Prefill only clearly safe audit rows and leave risky rows unresolved."""

    audit_path = Path(audit_csv_path)
    frame = pd.read_csv(audit_path, keep_default_na=False)

    for column in ("reviewed_label", "label_valid", "grammar_ok", "notes"):
        if column not in frame.columns:
            frame[column] = ""

    auto_filled = 0
    flagged = 0
    already_complete = 0

    for index, row in frame.iterrows():
        label_done = bool(str(row.get("label_valid", "")).strip())
        grammar_done = bool(str(row.get("grammar_ok", "")).strip())
        if not overwrite_existing and label_done and grammar_done:
            already_complete += 1
            continue

        issues = _grammar_issues(row)
        review_reasons = _needs_manual_review(row, issues)

        existing_notes = str(row.get("notes", "")).strip()
        frame.at[index, "reviewed_label"] = str(row.get("reviewed_label", "")).strip() or str(row["label"])

        if review_reasons or issues:
            frame.at[index, "notes"] = _append_note(existing_notes, FLAG_PREFIX, review_reasons + issues)
            flagged += 1
            continue

        frame.at[index, "label_valid"] = str(row.get("label_valid", "")).strip() or "true"
        frame.at[index, "grammar_ok"] = str(row.get("grammar_ok", "")).strip() or "true"
        frame.at[index, "notes"] = _append_note(existing_notes, FILL_PREFIX, ["no obvious issues detected"])
        auto_filled += 1

    frame.to_csv(audit_path, index=False)
    return {
        "rows": int(len(frame)),
        "auto_filled": auto_filled,
        "flagged_for_review": flagged,
        "already_complete": already_complete,
    }
