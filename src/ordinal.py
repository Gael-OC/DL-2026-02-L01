"""Decodificacion y probabilidades ordinales de CORAL."""

import torch

@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Cuenta umbrales superados y devuelve clases 0..K-1."""

    cumulative = torch.sigmoid(logits).cummin(dim=1).values
    return (cumulative > threshold).sum(dim=1)


@torch.no_grad()
def logits_to_ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convierte probabilidades acumulativas en probabilidades (B, K)."""

    cumulative = torch.sigmoid(logits).cummin(dim=1).values
    return torch.cat(
        (
            1 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        dim=1,
    )
