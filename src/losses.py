"""Perdida CORAL y pesos de clase del laboratorio."""

import numpy as np
import torch
import torch.nn.functional as F


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convierte etiquetas 0..K-1 en K-1 umbrales acumulativos."""

    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > thresholds).to(torch.float32)


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE con logits, promediada por muestra y ponderada por su clase."""

    levels = labels_to_levels(labels, num_classes).to(logits.dtype)
    sample_losses = F.binary_cross_entropy_with_logits(
        logits, levels, reduction="none"
    ).mean(dim=1)
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        sample_losses = sample_losses * weights[labels]
    return sample_losses.mean()


def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE condicional por umbral; ignora umbrales sin muestras elegibles."""

    total = logits.sum() * 0.0
    active = 0
    for threshold in range(logits.shape[1]):
        eligible = labels >= threshold
        if eligible.any():
            total = total + F.binary_cross_entropy_with_logits(
                logits[eligible, threshold],
                (labels[eligible] > threshold).to(logits.dtype),
            )
            active += 1
    return total / active


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """Pesos efectivos para clases 0..K-1; una clase ausente recibe peso cero."""

    if not 0 <= beta < 1:
        raise ValueError("beta debe estar entre 0 y 1 (sin incluir 1).")

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if counts.size != num_classes or not counts.any():
        raise ValueError("labels debe contener clases entre 0 y num_classes - 1.")

    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = (1 - beta) / (1 - beta ** counts[present])
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
