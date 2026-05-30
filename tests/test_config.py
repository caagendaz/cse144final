"""Checks for the recommended experiment configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from image_ensemble.config import default_colab_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_recommended_config_has_diverse_variants(self) -> None:
        config = default_colab_config("/tmp/kaggle")
        families = [variant.family for variant in config.variants]
        self.assertIn("convnext", families)
        self.assertIn("swin", families)
        self.assertGreaterEqual(families.count("eva02"), 2)

    def test_eva_seed_variants_are_independent(self) -> None:
        config = default_colab_config("/tmp/kaggle")
        evas = [variant for variant in config.variants if variant.family == "eva02"]
        self.assertNotEqual(evas[0].seed_offset, evas[1].seed_offset)

    def test_recommended_config_uses_memory_conscious_defaults(self) -> None:
        config = default_colab_config("/tmp/kaggle")
        self.assertLessEqual(config.prediction_batch_size, 8)
        self.assertFalse(config.pin_memory)
        self.assertFalse(config.persistent_workers)
        self.assertEqual(config.prefetch_factor, 1)
        for variant in config.variants:
            self.assertTrue(variant.gradient_checkpointing)
            self.assertFalse(variant.use_ema)
            self.assertGreater(variant.accum_steps, 1)


if __name__ == "__main__":
    unittest.main()
