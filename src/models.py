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


class _OrdinalFeatures(nn.Module):
    """Bloque oculto compartido por las arquitecturas equiparadas."""
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.15,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1d = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.bn1d(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.relu(hidden)
        return hidden


class MLPCoral(_OrdinalFeatures):
    """Bloque oculto de dos capas y cabeza CORAL de K-1 logits."""

    def __init__(
        self, num_features: int, num_classes: int,
        dropout: float = 0.15, hidden_dim: int = 32,
    ) -> None:
        super().__init__(num_features, num_classes, dropout, hidden_dim)
        self.CLayer = CoralLayer(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.CLayer(super().forward(inputs))


class MLPCorn(_OrdinalFeatures):
    """Bloque oculto equiparado con K-1 logits condicionales independientes."""

    def __init__(
        self, num_features: int, num_classes: int,
        dropout: float = 0.15, hidden_dim: int = 32,
    ) -> None:
        super().__init__(num_features, num_classes, dropout, hidden_dim)
        self.output = nn.Linear(16, num_classes - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(super().forward(inputs))


class MLPMatchedSoftmax(_OrdinalFeatures):
    """Bloque oculto de CORAL con salida Softmax de K logits."""

    def __init__(
        self, num_features: int, num_classes: int,
        dropout: float = 0.15, hidden_dim: int = 32,
    ) -> None:
        super().__init__(num_features, num_classes, dropout, hidden_dim)
        self.output = nn.Linear(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(super().forward(inputs))
