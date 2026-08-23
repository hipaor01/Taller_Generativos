"""Baseline de moving block bootstrap multivariante reutilizable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlockBootstrapConfig:
    """Configuración del baseline de bloques conjuntos."""

    block_length: int = 12
    horizon_steps: int = 120
    random_state: int = 42

    def validate(self) -> None:
        if self.block_length <= 0:
            raise ValueError("block_length debe ser positivo")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps debe ser positivo")


class MultivariateBlockBootstrap:
    """Remuestrea bloques conjuntos sin cruzar discontinuidades temporales."""

    def __init__(
        self,
        config: Optional[BlockBootstrapConfig] = None,
        *,
        block_length: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        if config is not None and block_length is not None:
            raise ValueError("Usa config o block_length, no ambos")
        self.config = config or BlockBootstrapConfig(
            block_length=(12 if block_length is None else block_length),
            random_state=random_state,
        )
        self.config.validate()
        self.blocks: Optional[NDArray[np.float64]] = None

    def fit(
        self,
        returns: NDArray[np.float64],
        segment_ids: NDArray[np.int64],
    ) -> "MultivariateBlockBootstrap":
        """Construye todos los bloques admisibles de cada segmento contiguo."""
        values = np.asarray(returns, dtype=np.float64)
        segments = np.asarray(segment_ids)
        if values.ndim != 2:
            raise ValueError("returns debe tener forma [tiempo, activos]")
        if len(values) != len(segments):
            raise ValueError("returns y segment_ids no están alineados")
        if len(values) == 0:
            raise ValueError("returns no puede estar vacío")
        if not np.isfinite(values).all():
            raise ValueError("returns contiene valores no finitos")

        blocks = []
        for segment_id in np.unique(segments):
            segment = values[segments == segment_id]
            if len(segment) < self.config.block_length:
                continue
            blocks.extend(
                segment[start : start + self.config.block_length]
                for start in range(len(segment) - self.config.block_length + 1)
            )
        if not blocks:
            raise ValueError("No hay segmentos suficientemente largos")
        self.blocks = np.stack(blocks)
        return self

    def sample(
        self,
        n: Optional[int] = None,
        cond: object = None,
        *,
        seed: Optional[int] = None,
        n_scenarios: Optional[int] = None,
        horizon_steps: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Genera ``n`` trayectorias con la interfaz común ``sample(n, cond)``."""
        del cond  # El baseline es deliberadamente incondicional.
        if self.blocks is None:
            raise RuntimeError("Hay que ejecutar fit antes de sample")
        if n is not None and n_scenarios is not None:
            raise ValueError("Usa n o n_scenarios, no ambos")
        n = n if n is not None else n_scenarios
        if n is None:
            raise ValueError("Falta n")
        if n <= 0:
            raise ValueError("n debe ser positivo")

        horizon = self.config.horizon_steps if horizon_steps is None else horizon_steps
        if horizon <= 0:
            raise ValueError("horizon_steps debe ser positivo")

        rng = np.random.default_rng(
            self.config.random_state if seed is None else seed
        )
        blocks_per_path = int(
            np.ceil(horizon / self.config.block_length)
        )
        choices = rng.integers(
            0,
            len(self.blocks),
            size=(n, blocks_per_path),
        )
        sampled = self.blocks[choices]
        paths = sampled.reshape(n, -1, self.blocks.shape[-1])
        return paths[:, :horizon].copy()

    @property
    def block_length(self) -> int:
        """Alias compatible con el notebook original."""
        return self.config.block_length
