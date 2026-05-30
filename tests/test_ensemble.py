"""Fast tests for the validation-driven blending logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from image_ensemble.ensemble import (  # noqa: E402
    blend_predictions,
    classification_accuracy,
    fold_weights_from_losses,
    multiclass_log_loss,
    normalize_probabilities,
    optimize_blend,
)


class EnsembleTests(unittest.TestCase):
    def test_normalize_probabilities_clips_and_normalizes_rows(self) -> None:
        probabilities = normalize_probabilities(np.array([[2.0, 2.0], [-1.0, 4.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(probabilities > 0))

    def test_probability_blend_uses_requested_weights(self) -> None:
        first = np.array([[0.9, 0.1], [0.2, 0.8]])
        second = np.array([[0.1, 0.9], [0.6, 0.4]])
        blended = blend_predictions([first, second], [0.75, 0.25])
        np.testing.assert_allclose(blended, np.array([[0.7, 0.3], [0.3, 0.7]]))

    def test_logit_blend_is_normalized(self) -> None:
        first = np.array([[0.9, 0.1], [0.2, 0.8]])
        second = np.array([[0.1, 0.9], [0.6, 0.4]])
        blended = blend_predictions([first, second], [0.5, 0.5], mode="logits")
        np.testing.assert_allclose(blended.sum(axis=1), np.ones(2))

    def test_lower_loss_fold_gets_more_weight(self) -> None:
        weights = fold_weights_from_losses([0.6, 1.2], temperature=0.08)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_optimizer_prefers_the_more_accurate_model(self) -> None:
        labels = np.array([0, 1, 0, 1])
        strong = np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
                [0.90, 0.10],
                [0.10, 0.90],
            ]
        )
        weak = np.array(
            [
                [0.10, 0.90],
                [0.90, 0.10],
                [0.20, 0.80],
                [0.80, 0.20],
            ]
        )
        result = optimize_blend(
            {"strong": strong, "weak": weak},
            labels,
            iterations=100,
            seed=7,
        )
        self.assertGreater(result.weights[0], result.weights[1])
        self.assertEqual(result.accuracy, 1.0)
        self.assertLessEqual(
            result.log_loss,
            multiclass_log_loss(strong, labels) + 1e-9,
        )
        self.assertEqual(classification_accuracy(strong, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
