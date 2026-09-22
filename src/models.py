"""Modelos del laboratorio."""

import torch
import torch.nn as nn

class ShallowMultiClassNet(nn.Module):
    """
    Red neuronal poco profunda para clasificacion multiclase.

    Tiene:
    - una capa de entrada,
    - una capa oculta,
    - dropout,
    - y una capa de salida con tantas neuronas como clases tenga el experimento.

    El forward devuelve logits. No se agrega Softmax aqui porque
    CrossEntropyLoss lo aplica internamente.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        dropout: float = 0.15,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class CoralLayer(nn.Module):
    """
    TODO(alumno):
    Capa de salida CORAL.

    Debe producir K-1 logits acumulativos a partir de un vector de
    caracteristicas de tamano input_size.

    Pistas:
    - un peso lineal compartido hacia un unico puntaje latente,
    - K-1 sesgos ordenados,
    - las diferencias entre sesgos pueden construirse con softplus y cumsum
      para forzar b0 >= b1 >= ... >= b_{K-2}.

    Formas esperadas:
    - x: (batch_size, input_size) (input size es 16 segun la arq recomendada)
    - salida: (batch_size, num_classes - 1)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_size, self.num_classes-1)
        self.softplus = nn.Softplus()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        |---------------------------------------------------------------\n
        **inputs -> sumar bias_k -> sigmoide**\n
        |---------------------------------------------------------------\n
        ej: con 2 clases seria solo b_0 (k-1 biases) ya que de esa forma seran los logits/levels\n
        cada bias se asigna a cada clase\n
        s = w * h | w: pesos, h: inputs, s: hidden\n
        logits o zk = s + bk = pesos_clase*inputs + bias_clase \n
        |---------------------------------------------------------------\n
        nn.Softplus() = ln(1+e^x), es como un ReLU con curva suavizada\n
        torch.cumsum() = suma de elementos anteriores de la lista para cada elemento
        """

        if inputs.dtype != torch.float32: raise TypeError("Asegurarse de hacer inputs.float() antes.")

        biases = torch.tensor([k for k in range(self.num_classes-1)]).float()
        biases = self.softplus.forward(biases)
        biases = torch.cumsum(biases, dim=0)

        hidden = self.fc1(inputs)
        logits = hidden + biases

        return logits


class MLPCoral(nn.Module):
    """
    TODO(alumno):
    MLP ordinal poco profunda con cabeza CORAL.

    Arquitectura sugerida:
    15 features -> Linear(32) -> ReLU -> BatchNorm1d -> Dropout
       -> Linear(16) -> ReLU
       -> CoralLayer(16, K)

    El forward debe devolver logits de forma (batch_size, K-1).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1d = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.CLayer = CoralLayer(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.bn1d(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        logits = self.CLayer(hidden)
        return logits
