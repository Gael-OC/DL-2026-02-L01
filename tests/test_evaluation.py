"""Pruebas pequeñas para las metricas del laboratorio."""

import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import compute_all_metrics, compute_confusion_matrix  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_keep_the_full_class_scale(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = compute_all_metrics(y_true, y_pred, num_classes=3)
        matrix = compute_confusion_matrix(y_true, y_pred, num_classes=3)

        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], 22 / 45)
        np.testing.assert_array_equal(
            matrix,
            [[1, 1, 0], [0, 2, 0], [0, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
