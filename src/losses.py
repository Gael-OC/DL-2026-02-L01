"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch
import torch.nn.functional as nnFunctional
from ordinal import logits_to_ordinal_predictions

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
    umbrales = np.arange(num_classes-1)
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

    if class_weights is not None: class_weights = class_weights[1:].float()

    """
    #descomentar para debug
    print("logits:\n",logits.float())
    print("")
    print("levels:\n",levels.float())
    print("")
    print("y_hat:\n",logits_to_ordinal_predictions(logits))
    print("")
    print("class_weights:\n",class_weights)
    print("")
    """

    loss = nnFunctional.binary_cross_entropy_with_logits(logits.float(), levels.float(), class_weights)

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

    dict_labels = {clase: 0 for clase in range(num_classes)}
    for label in labels:
        if dict_labels[label]:
            dict_labels[label] = dict_labels[label]+1
            continue
        dict_labels[label] = 1

    N = dict_labels.values() #cantidades de cada clase

    w_c = np.array([(1 - beta) / (1 - beta ** n_c) if n_c != 0 else 1 for n_c in N])
    w_c = w_c / np.mean(w_c)

    """
    #descomentar para debug
    print("dict_labels:",dict_labels, '\n')
    print("N:", N, '\n')
    print("wc:", w_c)
    print("wc_mean:",np.mean(w_c))
    """

    return torch.tensor(w_c)
