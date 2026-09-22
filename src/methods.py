"""Contrato comun de los cuatro metodos obligatorios."""

import torch
import torch.nn.functional as F

from losses import coral_loss
from models import MLPCoral, ShallowMultiClassNet
from ordinal import logits_to_ordinal_predictions, logits_to_ordinal_probabilities

METHODS = ("softmax_fixed", "softmax_hp", "coral", "coral_weighted")
METHOD_NAMES = {
    "softmax_fixed": "Softmax fijo",
    "softmax_hp": "Softmax HP",
    "coral": "CORAL sin pesos",
    "coral_weighted": "CORAL con pesos",
}


def build_model(method: str, config: dict, num_features: int, num_classes: int):
    if method in ("softmax_fixed", "softmax_hp"):
        return ShallowMultiClassNet(
            input_dim=num_features,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            output_dim=num_classes,
        )
    if method in ("coral", "coral_weighted"):
        return MLPCoral(num_features, num_classes, dropout=config["dropout"])
    raise ValueError(f"Metodo desconocido: {method}")


def loss(method: str, logits: torch.Tensor, labels: torch.Tensor, context: dict):
    if method in ("softmax_fixed", "softmax_hp"):
        return F.cross_entropy(logits, labels)
    if method in ("coral", "coral_weighted"):
        return coral_loss(
            logits, labels, context["num_classes"],
            context.get("class_weights") if method == "coral_weighted" else None,
        )
    raise ValueError(f"Metodo desconocido: {method}")


def decode(method: str, logits: torch.Tensor) -> torch.Tensor:
    if method in ("softmax_fixed", "softmax_hp"):
        return logits.argmax(dim=1)
    if method in ("coral", "coral_weighted"):
        return logits_to_ordinal_predictions(logits)
    raise ValueError(f"Metodo desconocido: {method}")


def predict_proba(method: str, logits: torch.Tensor) -> torch.Tensor:
    if method in ("softmax_fixed", "softmax_hp"):
        return torch.softmax(logits, dim=1)
    if method in ("coral", "coral_weighted"):
        return logits_to_ordinal_probabilities(logits)
    raise ValueError(f"Metodo desconocido: {method}")
