from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.audit import audit_readiness, summarize_audit  # noqa: E402
from vl_contradiction.audit_automation import auto_fill_audit_sheet  # noqa: E402
from vl_contradiction.metrics import per_edit_family_metrics  # noqa: E402


class AuditWorkflowTests(unittest.TestCase):
    def test_audit_readiness_fails_when_rows_are_unresolved(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "edit_family": "entailment_synonym",
                    "reviewed_label": "",
                    "label_valid": "",
                    "grammar_ok": "",
                }
            ]
        )
        readiness = audit_readiness(
            frame,
            overall_label_valid_threshold=0.9,
            overall_grammar_ok_threshold=0.9,
            per_family_label_valid_threshold=0.8,
            require_all_rows_reviewed=True,
        )
        self.assertFalse(readiness["passed"])
        self.assertEqual(1, readiness["unresolved_rows"])

    def test_audit_readiness_passes_when_thresholds_are_met(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "edit_family": "entailment_synonym",
                    "reviewed_label": "entailment",
                    "label_valid": "true",
                    "grammar_ok": "true",
                },
                {
                    "sample_id": "b",
                    "edit_family": "contradiction_object",
                    "reviewed_label": "contradiction",
                    "label_valid": "true",
                    "grammar_ok": "true",
                },
            ]
        )
        readiness = audit_readiness(
            frame,
            overall_label_valid_threshold=0.9,
            overall_grammar_ok_threshold=0.9,
            per_family_label_valid_threshold=0.8,
            require_all_rows_reviewed=True,
        )
        self.assertTrue(readiness["passed"])
        self.assertEqual([], readiness["reasons"])

    def test_audit_readiness_can_pass_with_unresolved_rows_when_manual_review_is_optional(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "edit_family": "entailment_synonym",
                    "reviewed_label": "entailment",
                    "label_valid": "true",
                    "grammar_ok": "true",
                },
                {
                    "sample_id": "b",
                    "edit_family": "contradiction_object",
                    "reviewed_label": "",
                    "label_valid": "",
                    "grammar_ok": "",
                },
            ]
        )
        readiness = audit_readiness(
            frame,
            overall_label_valid_threshold=0.9,
            overall_grammar_ok_threshold=0.9,
            per_family_label_valid_threshold=0.8,
            require_all_rows_reviewed=False,
        )
        self.assertTrue(readiness["passed"])
        self.assertEqual(1, readiness["unresolved_rows"])
        self.assertEqual([], readiness["reasons"])

    def test_auto_fill_only_completes_clearly_safe_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "safe",
                    "family_id": "fam-safe",
                    "image_id": 1,
                    "label": "entailment",
                    "edit_family": "entailment_synonym",
                    "edit_rule": "synonym:couch->sofa",
                    "source_caption": "A dog on a couch.",
                    "edited_caption": "A dog on a sofa.",
                    "reviewed_label": "",
                    "label_valid": "",
                    "grammar_ok": "",
                    "notes": "",
                },
                {
                    "sample_id": "flagged",
                    "family_id": "fam-flagged",
                    "image_id": 2,
                    "label": "entailment",
                    "edit_family": "entailment_synonym",
                    "edit_rule": "synonym:person->individual",
                    "source_caption": "A person waits.",
                    "edited_caption": "A individual waits.",
                    "reviewed_label": "",
                    "label_valid": "",
                    "grammar_ok": "",
                    "notes": "",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "audit.csv"
            frame.to_csv(csv_path, index=False)
            summary = auto_fill_audit_sheet(csv_path)
            updated = pd.read_csv(csv_path, keep_default_na=False)

        self.assertEqual(1, summary["auto_filled"])
        self.assertEqual(1, summary["flagged_for_review"])

        safe_row = updated.loc[updated["sample_id"] == "safe"].iloc[0]
        self.assertEqual("true", safe_row["label_valid"])
        self.assertEqual("true", safe_row["grammar_ok"])

        flagged_row = updated.loc[updated["sample_id"] == "flagged"].iloc[0]
        self.assertEqual("", flagged_row["label_valid"])
        self.assertEqual("", flagged_row["grammar_ok"])
        self.assertIn("AUTO FLAG:", flagged_row["notes"])

    def test_summarize_audit_and_per_family_metrics_return_expected_frames(self) -> None:
        audit_frame = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "edit_family": "entailment_synonym",
                    "label_valid": "true",
                    "grammar_ok": "true",
                },
                {
                    "sample_id": "b",
                    "edit_family": "entailment_synonym",
                    "label_valid": "false",
                    "grammar_ok": "true",
                },
            ]
        )
        summary = summarize_audit(audit_frame)
        self.assertEqual(["edit_family", "samples", "label_valid_rate", "grammar_ok_rate"], summary.columns.tolist())
        self.assertEqual(0.5, float(summary.loc[0, "label_valid_rate"]))

        metrics_frame = pd.DataFrame(
            [
                {"edit_family": "entailment_synonym", "label": "entailment", "pred_label": "entailment"},
                {"edit_family": "entailment_synonym", "label": "entailment", "pred_label": "contradiction"},
                {"edit_family": "contradiction_object", "label": "contradiction", "pred_label": "contradiction"},
            ]
        )
        family_metrics = per_edit_family_metrics(metrics_frame)
        self.assertEqual(["edit_family", "count", "accuracy", "macro_f1"], family_metrics.columns.tolist())
        self.assertEqual(2, int(family_metrics.loc[family_metrics["edit_family"] == "entailment_synonym", "count"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
