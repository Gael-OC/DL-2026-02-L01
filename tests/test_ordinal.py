from pathlib import Path
import sys
import unittest
import torch
from ordinal import logits_to_ordinal_predictions

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class OrdinalTest(unittest.TestCase):
    """
    **resultado esperado:**\n
    logits_sigmoid:\n
    tensor([
        [0.8909, 0.8581, 0.1091, 0.1419],\n
        [0.8699, 0.2689, 0.1824, 0.2142],\n
        [0.8699, 0.8581, 0.8455, 0.8320]\n
    ])\n
    y_hat: tensor([3, 2, 5])
    """
    def test_ltop(self):
        y_hat_esperado = torch.tensor([3, 2, 5])
        logits = torch.tensor([[2.1, 1.8, -2.1, -1.8],[1.9, -1., -1.5, -1.3],[1.9, 1.8, 1.7, 1.6]])
        y_hat = logits_to_ordinal_predictions(logits)

        print("\n|| test_ordinal.py: test_ltop ||")
        print("y_hat esperado:", y_hat_esperado)
        print("y_hat:", y_hat)

        self.assertEqual(y_hat_esperado.tolist(), y_hat.tolist())

if __name__ == "__main__":
    unittest.main()