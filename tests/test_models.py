from pathlib import Path
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

from models import MLPCoral
from ordinal import logits_to_ordinal_predictions
from losses import coral_loss, effective_number_weights
from math import sin

torch.manual_seed(42)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class ModelsTest(unittest.TestCase):
    batch_size = 32 # N
    features_in = 15 # entradas
    classes_out = 3 # clases de salida (k)

    def test_walkthrough(self):
        """
        Prueba todo el proceso relacionado con la generación de logits, calculo de pesos,\n
        predicciones y cálculo de la pérdida.
        """
        mlp = MLPCoral(
                    num_features=self.features_in,
                    num_classes=self.classes_out)
        
        #input_ = torch.randint(low=0, high=2, size=(self.batch_size,self.features_in)).float()
        input_ = torch.tensor([[0. if sin(i*j/5) > 0.5 or i%5 == 0 else 1. for j in range(self.features_in)] for i in range(self.batch_size)])

        labels = np.array(torch.tensor(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2, 0,
             2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2]))
        
        weights = effective_number_weights(labels, self.classes_out)
        logits = mlp.forward(input_)
        y_hat = logits_to_ordinal_predictions(logits)

        loss = coral_loss(logits, labels, self.classes_out, weights)

        print("\n|| test_models.py: test_walkthrough ||")
        print("")
        print("input:", input_)
        print("\nlabels:", labels)
        print("\nweights:", weights)
        print("\nlogits:", logits)
        print("\ny_hat:\n", y_hat.tolist())
        print("\ncoral_loss:", loss)

        self.assertTrue(1)

    def test_forward_layer(self):
        biases = torch.tensor([k for k in range(2)]).float()
        biases = nn.Softplus().forward(biases)
        biases = torch.cumsum(biases, dim=0)
        fc1 = nn.Linear(4, 2)
        hidden = fc1(torch.tensor([[1,2,4,8],[1,1,2,2],[1,1,1,1]]).float())
        logits = hidden + biases

        example = torch.tensor([
            [5.5521, 3.1073],
            [2.6155, 1.7314],
            [2.2734, 1.6812]
        ])

        print("\n|| test_models.py: test_forward_layer ||")
        print("example:\n", example, f'\n\n', example.tolist(), f'\n',)
        print("logits:\n", logits, f'\n\n', logits.tolist(), f'\n',)
        print("Se usaron 4 decimales para evaluar la similitud en promedio (valor esperado vs real)\n")

        self.assertAlmostEqual(
            example.mean().item(),
            logits.mean().item(),
            places=4
        )

    def test_forward_forma(self):
        input_ = torch.randint(low=0, high=2, size=(self.batch_size,self.features_in)).float()

        mlp = MLPCoral(
            num_features=self.features_in,
            num_classes=self.classes_out
        )

        logits = mlp.forward(input_)

        print("\n|| test_models.py: test_forward_forma ||")
        print("input:", input_)
        print("logits:", logits)
        
        self.assertEqual(logits.size(), torch.Size([self.batch_size, self.classes_out-1]))

    def test_grad(self):
        #Ambos deben ser float64 y requires_grad debe estar activo para inputs
        fc1 = nn.Linear(4, 2, bias=False, dtype=torch.float64)
        input_ = torch.tensor([[1.,2.,4.,8.],[1.,1.,2.,2.],[1.,1.,1.,1.]], dtype=torch.float64, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(fc1, input_, eps=1e-6, atol=1e-4))

if __name__ == "__main__":
    unittest.main()