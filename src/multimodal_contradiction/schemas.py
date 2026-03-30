"""Shared data contracts used across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LABELS = ("contradiction", "neutral", "entailment")
EDIT_FAMILIES = (
    "entailment_substitution",
    "neutral_hypernym",
    "neutral_attribute_removal",
    "contradiction_object_swap",
    "contradiction_attribute_flip",
    "contradiction_count_change",
    "contradiction_action_swap",
)


@dataclass(slots=True)
class BenchmarkRecord:
    sample_id: str
    family_id: str
    image_id: int
    source_split: str
    file_name: str
    source_caption: str
    edited_caption: str
    label: str
    edit_family: str
    edit_rule: str
    split: str
    audit_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PredictionRecord:
    model_name: str
    sample_id: str
    pred_label: str
    confidence: float | None
    raw_score: float | None
    raw_output_ref: str | None
    runtime_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThresholdConfig:
    tau_low: float
    tau_high: float
    objective: str = "macro_f1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditDecision:
    sample_id: str
    label_valid: bool
    grammar_ok: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
