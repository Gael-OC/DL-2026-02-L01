"""Pruebas del modelo CORAL real."""

from pathlib import Path
import sys
import unittest

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import CoralLayer, MLPCoral  # noqa: E402


class ModelsTest(unittest.TestCase):
    def test_coral_head_has_trainable_ordered_logits(self) -> None:
        for num_classes in (2, 3, 7):
            with self.subTest(num_classes=num_classes):
                layer = CoralLayer(4, num_classes)
                with torch.no_grad():
                    layer.score.weight.zero_()
                    layer.biases[0] = 2.0
                inputs = torch.ones(3, 4)
                logits = layer(inputs)

                self.assertEqual(logits.shape, (3, num_classes - 1))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.equal(logits, layer(inputs)))
                self.assertGreater(logits[0, 0].item(), 1.0)  # logits, no sigmoid
                self.assertTrue((logits[:, :-1] >= logits[:, 1:]).all())

                logits.sum().backward()
                self.assertIsNotNone(layer.score.weight.grad)
                self.assertIsNotNone(layer.biases.grad)

    def test_mlp_returns_trainable_threshold_logits(self) -> None:
        model = MLPCoral(num_features=15, num_classes=3, hidden_dim=64)
        self.assertEqual(model.fc1.out_features, 64)
        self.assertEqual(model.bn1d.num_features, 64)
        logits = model(torch.ones(4, 15))

        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits[:, 0] >= logits[:, 1]).all())
        logits.sum().backward()
        self.assertIsNotNone(model.CLayer.score.weight.grad)


if __name__ == "__main__":
    unittest.main()
