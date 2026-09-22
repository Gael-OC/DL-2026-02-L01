"""Pruebas de codificacion, perdida y pesos CORAL."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from losses import coral_loss, effective_number_weights, labels_to_levels  # noqa: E402


class LossesTest(unittest.TestCase):
    def test_levels_use_zero_based_classes(self) -> None:
        for num_classes in (2, 3, 7):
            with self.subTest(num_classes=num_classes):
                labels = torch.arange(num_classes)
                expected = torch.tensor(
                    [
                        [float(label > threshold) for threshold in range(num_classes - 1)]
                        for label in range(num_classes)
                    ]
                )
                torch.testing.assert_close(
                    labels_to_levels(labels, num_classes), expected
                )

    def test_coral_loss_uses_raw_logits_and_has_gradients(self) -> None:
        logits = torch.tensor(
            [[2.1, 1.8, -2.1, -1.8],
             [1.9, -1.0, -1.5, -1.3],
             [1.9, 1.8, 1.7, 1.6]],
            requires_grad=True,
        )
        loss = coral_loss(logits, torch.tensor([2, 1, 4]), num_classes=5)

        self.assertAlmostEqual(loss.item(), 0.1730, places=4)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_class_weights_apply_per_sample(self) -> None:
        logits = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        labels = torch.tensor([0, 2])
        unweighted = coral_loss(logits, labels, num_classes=3)
        unit_weights = coral_loss(logits, labels, 3, torch.ones(3))
        weighted = coral_loss(logits, labels, 3, torch.tensor([1.0, 1.0, 3.0]))
        # Dos muestras: (ln 2 + 3 * ln(1 + exp(-2))) / 2.
        expected = (
            torch.log(torch.tensor(2.0))
            + 3 * F.softplus(torch.tensor(-2.0))
        ) / 2

        torch.testing.assert_close(unit_weights, unweighted)
        torch.testing.assert_close(weighted, expected)

    def test_effective_weights_handle_missing_class(self) -> None:
        weights = effective_number_weights(np.array([0, 0, 1]), 3, beta=0.5)

        torch.testing.assert_close(weights, torch.tensor([1.2, 1.8, 0.0]))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(weights.mean().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
