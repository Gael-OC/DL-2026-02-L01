"""Modelos del laboratorio."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Puntaje compartido y K-1 umbrales ordenados para CORAL.

    Devuelve logits de forma (batch_size, num_classes - 1).
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.score = nn.Linear(input_size, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gaps = F.softplus(self.biases[1:]).cumsum(dim=0)
        ordered_biases = torch.cat((self.biases[:1], self.biases[:1] - gaps))
        return self.score(inputs) + ordered_biases


class MLPCoral(nn.Module):
    """
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
        hidden = self.relu(hidden)
        logits = self.CLayer(hidden)
        return logits
