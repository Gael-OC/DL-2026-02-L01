from pathlib import Path
import sys
import unittest
import torch
import torch.nn as nn

from models import MLPCoral

torch.manual_seed(42)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class ModelsTest(unittest.TestCase):
    """
    **CoralLayer.forward (ejemplo con input features 4 (input_size) y out features 2 (logits k-1))**\n
    #logits output:
    tensor( [[0.9940, 0.9699],[0.8980, 0.8907],[0.8621, 0.8857]] )
    """

    def test_forward(self):
        biases = torch.tensor([k for k in range(2)]).float()
        biases = nn.Softplus().forward(biases)
        biases = torch.cumsum(biases, dim=0)
        fc1 = nn.Linear(4, 2, bias=False)
        hidden = fc1(torch.tensor([[1,2,4,8],[1,1,2,2],[1,1,1,1]]).float())
        hidden = hidden + biases
        logits = torch.sigmoid(hidden)

        example = torch.tensor(
            [[0.9940, 0.9699],
            [0.8980, 0.8907],
            [0.8621, 0.8857]]
        )

        print("\n|| test_models.py: test_forward ||")
        print("example:", example, f'\n', example.tolist(), f'\n',)
        print("logits:", logits, f'\n', logits.tolist(), f'\n',)
        print("Se usaron 4 decimales para evaluar la similitud en promedio (valor esperado vs real)")

        self.assertAlmostEqual(
            example.mean().item(),
            logits.mean().item(),
            places=4
        )

    def test_forward_forma(self):
        """
        **ejemplo con forma 20x4 --> 4x16 --> 16x8 --> 3 clases de salida**
        """
        batch_size = 20 # N
        features_in = 4 # entradas
        classes_out = 3 # clases de salida (k)

        input_ = torch.randint(low=0, high=2, size=(batch_size,features_in)).float()

        mlp = MLPCoral(
            num_features=features_in,
            num_classes=classes_out,
            linear1=16,
            linear2=8,
        )

        logits = mlp.forward(input_)

        print("\n|| test_models.py: test_forward_forma ||")
        print("input:", input_)
        print("logits:", logits)
        
        self.assertEqual(logits.size(), torch.Size([batch_size, classes_out-1]))

    def test_grad(self):
        #Ambos deben ser float64 y requires_grad debe estar activo para inputs
        fc1 = nn.Linear(4, 2, bias=False, dtype=torch.float64)
        input_ = torch.tensor([[1.,2.,4.,8.],[1.,1.,2.,2.],[1.,1.,1.,1.]], dtype=torch.float64, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(fc1, input_, eps=1e-6, atol=1e-4))

if __name__ == "__main__":
    unittest.main()