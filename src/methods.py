"""Contrato comun de los metodos del laboratorio."""

import torch
import torch.nn.functional as F

from losses import coral_loss, corn_loss
from models import MLPCoral, MLPCorn, MLPMatchedSoftmax, ShallowMultiClassNet
from ordinal import (
    corn_predictions, corn_probabilities,
    logits_to_ordinal_predictions, logits_to_ordinal_probabilities,
)

METHODS = ("softmax_fixed", "softmax_hp", "coral", "coral_weighted", "corn", "softmax_matched")
METHOD_NAMES = {
    "softmax_fixed": "Softmax fijo",
    "softmax_hp": "Softmax HP",
    "coral": "CORAL sin pesos",
    "coral_weighted": "CORAL con pesos",
    "corn": "CORN v2 (pares elegibles)",
    "softmax_matched": "Softmax equiparado (argmax)",
    "softmax_matched_median": "Softmax equiparado (mediana)",
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
        return MLPCoral(
            num_features, num_classes, dropout=config["dropout"],
            hidden_dim=config["hidden_dim"],
        )
    if method == "corn":
        return MLPCorn(
            num_features, num_classes, dropout=config["dropout"],
            hidden_dim=config["hidden_dim"],
        )
    if method == "softmax_matched":
        return MLPMatchedSoftmax(
            num_features, num_classes, dropout=config["dropout"],
            hidden_dim=config["hidden_dim"],
        )
    raise ValueError(f"Metodo desconocido: {method}")


def loss(method: str, logits: torch.Tensor, labels: torch.Tensor, context: dict):
    if method in ("softmax_fixed", "softmax_hp", "softmax_matched"):
        return F.cross_entropy(logits, labels)
    if method in ("coral", "coral_weighted"):
        return coral_loss(
            logits, labels, context["num_classes"],
            context.get("class_weights") if method == "coral_weighted" else None,
        )
    if method == "corn":
        return corn_loss(logits, labels)
    raise ValueError(f"Metodo desconocido: {method}")


def decode(method: str, logits: torch.Tensor) -> torch.Tensor:
    if method in ("softmax_fixed", "softmax_hp", "softmax_matched"):
        return logits.argmax(dim=1)
    if method == "softmax_matched_median":
        return (torch.softmax(logits, dim=1).cumsum(dim=1) < 0.5).sum(dim=1)
    if method in ("coral", "coral_weighted"):
        return logits_to_ordinal_predictions(logits)
    if method == "corn":
        return corn_predictions(logits)
    raise ValueError(f"Metodo desconocido: {method}")


def predict_proba(method: str, logits: torch.Tensor) -> torch.Tensor:
    if method in ("softmax_fixed", "softmax_hp", "softmax_matched", "softmax_matched_median"):
        return torch.softmax(logits, dim=1)
    if method in ("coral", "coral_weighted"):
        return logits_to_ordinal_probabilities(logits)
    if method == "corn":
        return corn_probabilities(logits)
    raise ValueError(f"Metodo desconocido: {method}")
