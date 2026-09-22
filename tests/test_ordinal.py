"""Pruebas de prediccion y probabilidades CORAL."""

from pathlib import Path
import sys
import unittest

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ordinal import (  # noqa: E402
    logits_to_ordinal_predictions,
    logits_to_ordinal_probabilities,
)


class OrdinalTest(unittest.TestCase):
    def test_decode_uses_zero_based_classes(self) -> None:
        logits = torch.tensor([[-2.0, -3.0], [2.0, -2.0], [2.0, 1.0]])
        predictions = logits_to_ordinal_predictions(logits)

        torch.testing.assert_close(predictions, torch.tensor([0, 1, 2]))

    def test_probabilities_are_valid_for_two_and_seven_classes(self) -> None:
        for num_classes in (2, 7):
            with self.subTest(num_classes=num_classes):
                logits = torch.arange(num_classes - 1, dtype=torch.float32)
                probabilities = logits_to_ordinal_probabilities(logits[None, :])

                self.assertEqual(probabilities.shape, (1, num_classes))
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue((probabilities >= 0).all())
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))

        known_logits = torch.logit(torch.tensor([[0.8, 0.3]]))
        torch.testing.assert_close(
            logits_to_ordinal_probabilities(known_logits),
            torch.tensor([[0.2, 0.5, 0.3]]),
        )


if __name__ == "__main__":
    unittest.main()
