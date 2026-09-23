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
    return _cumulative_to_probabilities(cumulative)


def _cumulative_to_probabilities(cumulative: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            1 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        dim=1,
    )


@torch.no_grad()
def corn_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convierte condicionales CORN en una distribucion de K clases."""

    return _cumulative_to_probabilities(torch.sigmoid(logits).cumprod(dim=1))


@torch.no_grad()
def corn_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Cuenta probabilidades acumuladas superiores a 0.5."""

    return (torch.sigmoid(logits).cumprod(dim=1) > 0.5).sum(dim=1)
