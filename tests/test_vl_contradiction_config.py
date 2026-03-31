from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vl_contradiction.config import load_config  # noqa: E402


class ConfigLoadingTests(unittest.TestCase):
    def test_default_config_loads_stage_aware_training_sweeps(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

        self.assertEqual("macro_f1", config.training.selection_metric)
        self.assertEqual(16, config.training.clip_batch_size)
        self.assertEqual(16, config.training.joint_feature_batch_size)
        self.assertEqual(8, config.training.token_feature_batch_size)
        self.assertEqual("lp_p1", config.training.sweeps["prototype"]["linear_probe"]["trials"][0]["name"])
        self.assertEqual("ca_m5", config.training.sweeps["midscale"]["cross_attention"]["trials"][-1]["name"])
        self.assertEqual(7, len(config.training.sweeps["final"]["linear_probe"]["trials"]))
        self.assertEqual(7, len(config.training.sweeps["final"]["cross_attention"]["trials"]))


if __name__ == "__main__":
    unittest.main()
