"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch
import torch.nn.functional as nnFunctional


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1)
    """
    umbrales = np.arange(num_classes)[1:]
    levels = (np.array(labels)[:, np.newaxis] > umbrales).astype(int)
    return torch.tensor(levels)


def coral_loss(
    logits: torch.Tensor, # y predicho
    labels: torch.Tensor, # y
    num_classes: int, # K
    class_weights: torch.Tensor | None = None, # W_k o w_c
) -> torch.Tensor:
    """
    TODO(alumno):
    BCE con logits sobre los K-1 umbrales ordinales.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    """
    levels = labels_to_levels(labels, num_classes) #Y_nk

    if not logits.shape == levels.shape:
        raise ValueError("Los logits (%s) no tienen la misma forma que los niveles (%s). "
                         % (logits.shape, levels.shape))
    
    logits_sigmoid = nnFunctional.logsigmoid(logits)

    loss = (logits_sigmoid*levels + (logits_sigmoid - logits)*(1-levels))

    if class_weights is not None:
        loss *= class_weights
    
    loss = torch.mean((-torch.sum(loss, dim=1)))
    
    return loss


def effective_number_weights(
    labels: np.ndarray, #y
    num_classes: int, #k
    beta: float = 0.99,
) -> torch.Tensor:
    """
    TODO(alumno):
    Pesos por numero efectivo de muestras:

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normalizar los pesos para que su media sea 1.

    Formas:
    - labels: (N,)
    - salida: (num_classes,)
    """

    _, N = np.unique(labels, return_counts=True) #cantidades de cada clase

    w_c = [(1 - beta) / (1 - beta ** n_c) for n_c  in N]

    return torch.tensor(w_c)
