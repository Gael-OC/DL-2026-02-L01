"""Utilidades ordinales que los alumnos deben implementar."""

import torch
import numpy as np
import torch.nn.functional as nnFunctional

@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte logits CORAL en una clase entera.

    Pista:
    aplicar sigmoide, contar cuantos umbrales superan threshold
    y devolver ese conteo como y_hat.

    Formas:
    - logits: (batch_size, K-1)
    - salida: (batch_size,)
    """
    logits_sigmoid = nnFunctional.sigmoid(logits)
    
    """
    #descomentar para debug
    print("logits_sigmoid:",logits_sigmoid)
    """

    y_hat = [
        sum(logit > threshold for logit in fila)+1 #suma 1 si se cumple
        for fila in logits_sigmoid
        ]

    return torch.tensor(y_hat)
