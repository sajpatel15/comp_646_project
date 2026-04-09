"""Interactive notebook UI for one-at-a-time audit review."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display
from PIL import Image

from .labels import CLASS_ORDER


TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


def _normalize_flag(value: object) -> str:
    lowered = str(value).strip().lower()
    if lowered in TRUE_VALUES:
        return "true"
    if lowered in FALSE_VALUES:
        return "false"
    return ""


def _is_completed(row: pd.Series) -> bool:
    return bool(_normalize_flag(row.get("label_valid", "")) and _normalize_flag(row.get("grammar_ok", "")))


@dataclass
class AuditReviewSession:
    """Stateful widget bundle for reviewing one audit example at a time."""

    audit_path: Path
    review_frame: pd.DataFrame
    audit_columns: list[str]
    index: int = 0

    def __post_init__(self) -> None:
        self.progress_html = widgets.HTML()
        self.status_html = widgets.HTML()
        self.details_html = widgets.HTML(layout=widgets.Layout(width="100%"))
        self.image_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))
        self.reviewed_label = widgets.Dropdown(
            options=[("Keep assigned label", "")] + [(label.title(), label) for label in CLASS_ORDER],
            description="Reviewed",
            layout=widgets.Layout(width="320px"),
        )
        self.label_valid = widgets.ToggleButtons(
            options=[("Unset", ""), ("Valid", "true"), ("Invalid", "false")],
            description="Label",
            layout=widgets.Layout(width="320px"),
        )
        self.grammar_ok = widgets.ToggleButtons(
            options=[("Unset", ""), ("OK", "true"), ("Bad", "false")],
            description="Grammar",
            layout=widgets.Layout(width="320px"),
        )
        self.notes = widgets.Textarea(
            description="Notes",
            placeholder="Why is this invalid or ungrammatical?",
            layout=widgets.Layout(width="100%", height="110px"),
        )
        self.index_input = widgets.BoundedIntText(
            value=self.index + 1,
            min=1,
            max=max(len(self.review_frame), 1),
            description="Row",
            layout=widgets.Layout(width="160px"),
        )
        self.prev_button = widgets.Button(description="Previous")
        self.save_button = widgets.Button(description="Save", button_style="info")
        self.next_button = widgets.Button(description="Save & Next", button_style="success")
        self.next_unreviewed_button = widgets.Button(description="Next Unreviewed")
        self.jump_button = widgets.Button(description="Jump")

        self.prev_button.on_click(self._on_previous)
        self.save_button.on_click(self._on_save)
        self.next_button.on_click(self._on_next)
        self.next_unreviewed_button.on_click(self._on_next_unreviewed)
        self.jump_button.on_click(self._on_jump)

        toolbar = widgets.HBox(
            [
                self.prev_button,
                self.save_button,
                self.next_button,
                self.next_unreviewed_button,
                self.index_input,
                self.jump_button,
            ]
        )
        controls = widgets.VBox(
            [
                self.progress_html,
                self.status_html,
                self.details_html,
                self.reviewed_label,
                self.label_valid,
                self.grammar_ok,
                self.notes,
                toolbar,
            ]
        )
        self.root = widgets.VBox([self.image_output, controls], layout=widgets.Layout(gap="12px"))
        self._render_row()

    def display(self) -> "AuditReviewSession":
        display(self.root)
        return self

    def _persist(self) -> None:
        self.review_frame[self.audit_columns].to_csv(self.audit_path, index=False)

    def _save_current(self) -> None:
        self.review_frame.at[self.index, "reviewed_label"] = self.reviewed_label.value
        self.review_frame.at[self.index, "label_valid"] = self.label_valid.value
        self.review_frame.at[self.index, "grammar_ok"] = self.grammar_ok.value
        self.review_frame.at[self.index, "notes"] = self.notes.value.strip()
        self._persist()

    def _completed_count(self) -> int:
        return int(self.review_frame.apply(_is_completed, axis=1).sum())

    def _render_row(self) -> None:
        row = self.review_frame.iloc[self.index]
        self.index_input.value = self.index + 1
        reviewed_label = str(row.get("reviewed_label", "") or "")
        self.reviewed_label.value = reviewed_label if reviewed_label in CLASS_ORDER or reviewed_label == "" else ""
        self.label_valid.value = _normalize_flag(row.get("label_valid", ""))
        self.grammar_ok.value = _normalize_flag(row.get("grammar_ok", ""))
        self.notes.value = str(row.get("notes", "") or "")

        completed = self._completed_count()
        self.progress_html.value = (
            f"<b>Row {self.index + 1}/{len(self.review_frame)}</b>"
            f" | Completed {completed}/{len(self.review_frame)}"
        )
        self.status_html.value = (
            f"<b>sample_id:</b> {escape(str(row['sample_id']))} | "
            f"<b>assigned:</b> {escape(str(row['label']))} | "
            f"<b>family:</b> {escape(str(row['edit_family']))}"
        )
        self.details_html.value = (
            f"<div><b>edit_rule:</b> {escape(str(row['edit_rule']))}</div>"
            f"<div style='margin-top:8px'><b>Source caption</b><br>{escape(str(row['source_caption']))}</div>"
            f"<div style='margin-top:8px'><b>Edited caption</b><br>{escape(str(row['edited_caption']))}</div>"
        )

        with self.image_output:
            clear_output(wait=True)
            image_path = Path(str(row["file_path"]))
            if not image_path.exists():
                print(f"Missing image: {image_path}")
            else:
                display(Image.open(image_path).convert("RGB"))

    def _move_to(self, new_index: int) -> None:
        self.index = max(0, min(new_index, len(self.review_frame) - 1))
        self._render_row()

    def _find_next_unreviewed(self) -> int:
        total = len(self.review_frame)
        for offset in range(1, total + 1):
            candidate = (self.index + offset) % total
            if not _is_completed(self.review_frame.iloc[candidate]):
                return candidate
        return self.index

    def _on_previous(self, _: widgets.Button) -> None:
        self._save_current()
        self._move_to(self.index - 1)

    def _on_save(self, _: widgets.Button) -> None:
        self._save_current()
        self.progress_html.value += " | Saved"

    def _on_next(self, _: widgets.Button) -> None:
        self._save_current()
        self._move_to(self.index + 1)

    def _on_next_unreviewed(self, _: widgets.Button) -> None:
        self._save_current()
        self._move_to(self._find_next_unreviewed())

    def _on_jump(self, _: widgets.Button) -> None:
        self._save_current()
        self._move_to(self.index_input.value - 1)


def launch_audit_reviewer(
    audit_csv_path: str | Path,
    benchmark_csv_path: str | Path,
    *,
    start_at_first_unreviewed: bool = True,
) -> AuditReviewSession:
    """Launch a simple widget UI for reviewing benchmark audit rows."""

    audit_path = Path(audit_csv_path)
    benchmark_path = Path(benchmark_csv_path)
    audit_frame = pd.read_csv(audit_path, keep_default_na=False)
    for column in ("reviewed_label", "label_valid", "grammar_ok", "notes"):
        if column not in audit_frame.columns:
            audit_frame[column] = ""
    audit_columns = list(audit_frame.columns)

    benchmark_frame = pd.read_csv(benchmark_path, usecols=["sample_id", "file_path"], keep_default_na=False)
    review_frame = audit_frame.merge(benchmark_frame, on="sample_id", how="left", validate="one_to_one")

    if review_frame["file_path"].eq("").any():
        missing = review_frame.loc[review_frame["file_path"].eq(""), "sample_id"].tolist()[:5]
        raise ValueError(f"Missing file_path for audit rows: {missing}")

    start_index = 0
    if start_at_first_unreviewed:
        unresolved = review_frame.index[~review_frame.apply(_is_completed, axis=1)]
        if len(unresolved):
            start_index = int(unresolved[0])

    return AuditReviewSession(
        audit_path=audit_path,
        review_frame=review_frame,
        audit_columns=audit_columns,
        index=start_index,
    ).display()
