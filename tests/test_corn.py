"""Contrato minimo de CORN, incluida una mascara condicional vacia."""

import unittest

import torch
import torch.nn.functional as F

import main  # noqa: F401 - agrega src al path como los otros tests de integracion
from src.losses import corn_loss
from src.methods import build_model, decode, predict_proba


class CornTests(unittest.TestCase):
    def test_conditional_loss_skips_empty_upper_threshold(self) -> None:
        logits = torch.tensor([[0.2, 1.0, -2.0], [-0.4, 0.7, 3.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        actual = corn_loss(logits, labels)
        expected = (
            F.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([0.0, 1.0]))
            + F.binary_cross_entropy_with_logits(logits[1:2, 1], torch.tensor([0.0]))
        ) / 2
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[:, 2], torch.zeros(2))

    def test_product_probabilities_and_model_contract(self) -> None:
        model = build_model("corn", {"hidden_dim": 64, "dropout": 0.15}, 15, 3)
        self.assertEqual(model.fc1.out_features, 64)
        self.assertEqual(model(torch.ones(4, 15)).shape, (4, 2))
        logits = torch.logit(torch.tensor([[0.8, 0.5]]))
        torch.testing.assert_close(
            predict_proba("corn", logits), torch.tensor([[0.2, 0.4, 0.4]])
        )
        torch.testing.assert_close(decode("corn", logits), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
