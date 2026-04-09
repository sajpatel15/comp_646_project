from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.benchmark import (  # noqa: E402
    _attribute_contradiction,
    _count_contradiction,
    _normalize_caption,
    _object_contradiction,
    build_benchmark,
)


class BenchmarkRuleTests(unittest.TestCase):
    def test_protected_phrases_skip_broken_edits(self) -> None:
        attribute_candidate = _attribute_contradiction("A black and white photo of a dog.")
        self.assertEqual((None, None), attribute_candidate)

        object_candidate = _object_contradiction(
            "A group of people standing outside of a double decker bus.",
            ["bus", "person"],
        )
        self.assertEqual((None, None), object_candidate)

    def test_article_normalization_handles_individual_animal_and_television(self) -> None:
        self.assertEqual("an individual", _normalize_caption("a individual"))
        self.assertEqual("an animal", _normalize_caption("a animal"))
        self.assertEqual("a television", _normalize_caption("an television"))

    def test_count_rewrites_do_not_emit_known_malformed_phrases(self) -> None:
        updated, rule = _count_contradiction("Two people are riding a motorcycle on the beach.", {})
        self.assertEqual("count:two->one", rule)
        self.assertEqual("One person is riding a motorcycle on the beach", updated)
        self.assertNotIn("one people", updated.lower())

        updated, rule = _count_contradiction("A person is riding a horse.", {"person": 2})
        self.assertEqual("count:article->person", rule)
        self.assertEqual("Two people are riding a horse", updated)
        self.assertNotIn("two person", updated.lower())

        updated, rule = _count_contradiction("A person is riding a horse.", {"horse": 2})
        self.assertEqual("count:article->horse", rule)
        self.assertEqual("A person is riding two horses", updated)
        self.assertNotIn("two horse ", updated.lower())

        updated, _ = _count_contradiction("There are two men talking.", {})
        self.assertIn("one man", updated.lower())
        self.assertNotIn("one men", updated.lower())

    def test_object_contradictions_are_curated_and_skip_present_targets(self) -> None:
        updated, rule = _object_contradiction("A dog on a couch.", ["dog", "couch"])
        self.assertEqual("object:dog->cat", rule)
        self.assertEqual("A cat on a couch", updated)

        updated, rule = _object_contradiction("A dog and a cat.", ["dog", "cat"])
        self.assertEqual((None, None), (updated, rule))

    def test_build_benchmark_balances_edit_families(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "family_id": f"fam-{index}",
                    "image_id": index,
                    "caption": "A person is standing near a red dog.",
                    "file_path": f"/tmp/fake-{index}.jpg",
                    "objects": ["dog", "person"],
                    "object_counts": {"person": 2, "dog": 1},
                }
            )
        frame = pd.DataFrame(rows)

        result = build_benchmark(frame, family_limit=8, split_ratio=[0.5, 0.25, 0.25], seed=7)
        benchmark = result.records
        coverage = result.coverage_summary.set_index("edit_family")

        self.assertEqual(16, len(benchmark))
        self.assertEqual(8, benchmark.loc[benchmark["edit_family"] == "entailment_synonym"].shape[0])
        self.assertEqual(2, benchmark.loc[benchmark["edit_family"] == "contradiction_action"].shape[0])
        self.assertEqual(2, benchmark.loc[benchmark["edit_family"] == "contradiction_attribute"].shape[0])
        self.assertEqual(2, benchmark.loc[benchmark["edit_family"] == "contradiction_count"].shape[0])
        self.assertEqual(2, benchmark.loc[benchmark["edit_family"] == "contradiction_object"].shape[0])
        self.assertEqual({"contradiction", "entailment"}, set(benchmark["label"]))
        self.assertTrue(bool(coverage["meets_target"].all()))


if __name__ == "__main__":
    unittest.main()
