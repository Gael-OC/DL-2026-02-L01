"""La mediana usa exactamente las probabilidades del ajuste Softmax equiparado."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import main
from src.evaluation import compute_all_metrics
from src.methods import build_model


class SoftmaxMatchedTests(unittest.TestCase):
    def test_same_training_cycle_produces_argmax_and_median(self) -> None:
        torch.set_num_threads(1)
        X = np.zeros((6, 15), dtype=np.float32)
        y = np.array([0, 1, 2, 0, 1, 2])
        with patch("main.build_model", wraps=build_model) as model_factory:
            result = main.run_training_cycle(
                X, y, X, y, num_classes=3, hidden_dim=64, dropout=0.15,
                learning_rate=1e-3, weight_decay=1e-4, batch_size=6,
                epochs=1, seed=42, device=torch.device("cpu"),
                method="softmax_matched",
            )
        self.assertEqual(model_factory.call_count, 1)
        self.assertEqual(result["model"].fc1.out_features, 64)
        self.assertEqual(result["model"].output.out_features, 3)
        np.testing.assert_array_equal(
            result["median_pred"],
            (np.cumsum(result["y_proba"], axis=1) < 0.5).sum(axis=1),
        )
        self.assertTrue(np.isfinite(result["y_proba"]).all())

    def test_median_does_not_apply_softmax_twice(self) -> None:
        X = np.zeros((3, 15), dtype=np.float32)
        y = np.array([0, 1, 2])
        probabilities = np.tile(np.array([[0.6, 0.2, 0.2]], dtype=np.float32), (3, 1))
        metrics = compute_all_metrics(y, np.zeros_like(y), num_classes=3)
        with patch("main.evaluate_model", return_value=(
            metrics, y, np.zeros_like(y), probabilities,
        )):
            result = main.run_training_cycle(
                X, y, X, y, num_classes=3, hidden_dim=32, dropout=0.15,
                learning_rate=1e-3, weight_decay=1e-4, batch_size=3,
                epochs=1, seed=42, device=torch.device("cpu"),
                method="softmax_matched",
            )
        np.testing.assert_array_equal(result["median_pred"], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
