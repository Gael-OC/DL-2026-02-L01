from pathlib import Path
import sys
import unittest
import torch
import numpy as np
from losses import labels_to_levels, coral_loss, effective_number_weights

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class LossesTest(unittest.TestCase):
    """
    **Tests de las 3 funciones de lossses.py**\n
    + labels_to_levels
    + coral_loss
    + effective_number_weights
    """

    def test_labels_to_levels(self):
        t2 = torch.tensor([0,1])
        t3 = torch.tensor([0,1,2])
        t5 = torch.tensor([0,1,2,3,4])
        t7 = torch.tensor([0,1,2,3,4,5,6])

        l2 = labels_to_levels(t2, 2)
        l3 = labels_to_levels(t3, 3)
        l5 = labels_to_levels(t5, 5)
        l7 = labels_to_levels(t7, 7)

        print("\n|| test_labels_to_levels (prueba de forma) ||")
        print("K=2:\n", f"labels: {t2}\n levels:\n",l2)
        print("K=3:\n", f"labels: {t3}\n levels:\n",l3)
        print("K=5:\n", f"labels: {t5}\n levels:\n",l5)
        print("K=7:\n", f"labels: {t7}\n levels:\n",l7)

        self.assertEqual(l2.size(), torch.Size([2,1]))
        self.assertEqual(l3.size(), torch.Size([3,2]))
        self.assertEqual(l5.size(), torch.Size([5,4]))
        self.assertEqual(l7.size(), torch.Size([7,6]))

    def test_coral_loss(self):
        """
        levels esperados: \n\t torch.tensor([[1, 1, 0, 0],[1, 0, 0, 0],[1, 1, 1, 1]])\n
        resultado esperado: 0.1730
        """
        labels = np.array([2,1,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0])
        weights = effective_number_weights(labels, 5)
        
        logits = torch.tensor(
            [
                [2.1, 1.8, -2.1, -1.8],[1.9, -1., -1.5, -1.3],[1.9, 1.8, 1.7, 1.6],
                [1.9, 1.8, 1.7, 1.6],[1.9, 1.8, 1.7, 1.6],[1.9, 1.8, 1.7, 1.6],
                [1.9, 1.8, 1.7, 1.6],[1.9, 1.8, 1.7, 1.6],[1.9, 1.8, 1.7, 1.6],
                [-0.9, -1., -1.5, -1.3],[-0.9, -1., -1.5, -1.3],[-0.9, -1., -1.5, -1.3],
                [-0.9, -1., -1.5, -1.3],[-0.9, -1., -1.5, -1.3],[-0.9, -1., -1.5, -1.3],
                [-0.9, -1., -1.5, -1.3],[-0.9, -1., -1.5, -1.3]
             ]
        )
        resultado = coral_loss(logits, labels, 5, weights)

        print("\n|| test_coral_loss ||")
        print("logits:\n", logits)
        print("resultado:\n", resultado)

        self.assertAlmostEqual(resultado.item(), 0.2611, places=4)

    def test_effective_number_weights(self):
        """
        **salida esperada:**\n
        (descomentar en el origen de la funcion para ver todas las salidas)\n
        \tdict_labels: {0: 8, 1: 1, 2: 1, 3: 0, 4: 7}\n
        \tN: dict_values([8, 1, 1, 0, 7])\n
        \twc: [0.19752078 1.52595284 1.52595284 1.52595284 0.22462069]\n
        \twc_mean: 1.0\n
        \ttensor([0.1975, 1.5260, 1.5260, 1.5260, 0.2246], dtype=torch.float64)
        """

        labels = np.array([2,1,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0])
        weights = effective_number_weights(labels, 5)

        print("\n|| test_effective_number_weights ||")
        print("weights:")
        print(weights, "\n wc_mean:", weights.mean().item())

        self.assertAlmostEqual(weights.mean().item(), 1., places=4)

    def test_enw_zero_weights(self):
        """
        Test para probar que effective_number_weights (enw) funciona para clases ausentes.
        """
        labels = np.array([])
        weights = effective_number_weights(labels, 5)

        print("\n|| test_enw_zero_weights ||")
        print("weights:")
        print(weights, "\n wc_mean:", weights.mean().item())
        
        self.assertListEqual(weights.tolist(), [1., 1., 1., 1., 1.])

if __name__ == "__main__":
    unittest.main()