"""Primitivas estadisticas compartidas por las familias del evaluador."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quantile_wasserstein(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    n_quantiles: int,
) -> float:
    """Aproxima Wasserstein-1 mediante una malla comun de cuantiles."""
    probabilities = np.linspace(0.0, 1.0, n_quantiles)
    return float(
        np.mean(
            np.abs(
                np.quantile(reference, probabilities)
                - np.quantile(candidate, probabilities)
            )
        )
    )


def quantile_name(probability: float) -> str:
    """Nombre estable para una probabilidad expresada como cuantil porcentual."""
    percentage = probability * 100
    if percentage.is_integer():
        return f"q{int(percentage):02d}"
    return f"q{percentage:g}".replace(".", "_")
