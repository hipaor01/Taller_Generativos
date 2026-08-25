"""Baselines de moving block bootstrap multivariante reutilizables."""

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


@dataclass(frozen=True)
class ConditionalBlockBootstrapConfig(BlockBootstrapConfig):
    """Configuración del bootstrap condicionado por vecinos próximos."""

    n_neighbors: int = 128
    distance_batch_size: int = 256

    def validate(self) -> None:
        super().validate()
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors debe ser positivo")
        if self.distance_batch_size <= 0:
            raise ValueError("distance_batch_size debe ser positivo")


FROZEN_CONDITIONAL_BLOCK_LENGTH = 12
FROZEN_CONDITIONAL_NEIGHBORS = 128


def frozen_conditional_bootstrap_config(
    *,
    random_state: int,
    horizon_steps: int = 120,
) -> ConditionalBlockBootstrapConfig:
    """Crea la configuración seleccionada exclusivamente con validación."""

    return ConditionalBlockBootstrapConfig(
        block_length=FROZEN_CONDITIONAL_BLOCK_LENGTH,
        horizon_steps=horizon_steps,
        n_neighbors=FROZEN_CONDITIONAL_NEIGHBORS,
        random_state=random_state,
    )


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


class ConditionalMultivariateBlockBootstrap(MultivariateBlockBootstrap):
    """Bootstrap de bloques condicionado mediante vecinos de mercado.

    Cada trayectoria de entrenamiento aporta un único bloque que comienza en
    su primer paso objetivo. Así, su vector de condición describe exactamente
    el mercado observado justo antes del bloque y las ventanas solapadas no
    duplican un mismo inicio. Al generar, cada bloque se elige uniformemente
    entre los ``n_neighbors`` estados de entrenamiento más próximos a la
    condición solicitada.

    Las condiciones deben estar en la misma escala para ``fit`` y ``sample``.
    En el proyecto se usan directamente las 14 variables normalizadas con los
    estadísticos de train que también reciben CVAE, Flow y GAN.
    """

    def __init__(
        self,
        config: Optional[ConditionalBlockBootstrapConfig] = None,
        *,
        block_length: Optional[int] = None,
        n_neighbors: int = 128,
        random_state: int = 42,
        distance_batch_size: int = 256,
    ) -> None:
        if config is not None and block_length is not None:
            raise ValueError("Usa config o block_length, no ambos")
        selected_config = config or ConditionalBlockBootstrapConfig(
            block_length=(12 if block_length is None else block_length),
            n_neighbors=n_neighbors,
            random_state=random_state,
            distance_batch_size=distance_batch_size,
        )
        selected_config.validate()
        super().__init__(config=selected_config)
        self.config: ConditionalBlockBootstrapConfig = selected_config
        self.training_conditions: Optional[NDArray[np.float64]] = None
        self._training_condition_squared_norms: Optional[NDArray[np.float64]] = None

    def fit(
        self,
        paths: NDArray[np.float64],
        conditions: NDArray[np.float64],
    ) -> "ConditionalMultivariateBlockBootstrap":
        """Alinea el primer bloque de cada trayectoria con su condición."""

        values = np.asarray(paths, dtype=np.float64)
        condition_values = np.asarray(conditions, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("paths debe tener forma [muestras, tiempo, activos]")
        if values.shape[1] < self.config.block_length:
            raise ValueError("Las trayectorias son más cortas que block_length")
        if condition_values.ndim != 2:
            raise ValueError("conditions debe tener forma [muestras, variables]")
        if len(values) != len(condition_values):
            raise ValueError("paths y conditions no están alineados")
        if len(values) == 0:
            raise ValueError("paths no puede estar vacío")
        if values.shape[2] == 0 or condition_values.shape[1] == 0:
            raise ValueError("paths y conditions deben contener variables")
        if not np.isfinite(values).all():
            raise ValueError("paths contiene valores no finitos")
        if not np.isfinite(condition_values).all():
            raise ValueError("conditions contiene valores no finitos")

        self.blocks = values[:, : self.config.block_length, :].copy()
        self.training_conditions = condition_values.copy()
        self._training_condition_squared_norms = np.sum(
            np.square(self.training_conditions), axis=1
        )
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
        """Genera trayectorias remuestreando bloques cercanos a ``cond``."""

        if (
            self.blocks is None
            or self.training_conditions is None
            or self._training_condition_squared_norms is None
        ):
            raise RuntimeError("Hay que ejecutar fit antes de sample")
        if n is not None and n_scenarios is not None:
            raise ValueError("Usa n o n_scenarios, no ambos")
        sample_count = n if n is not None else n_scenarios
        if sample_count is None:
            raise ValueError("Falta n")
        if sample_count <= 0:
            raise ValueError("n debe ser positivo")
        if cond is None:
            raise ValueError("cond es obligatorio para el bootstrap condicionado")

        query = np.asarray(cond, dtype=np.float64)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != self.training_conditions.shape[1]:
            raise ValueError(
                "cond debe tener forma [batch, "
                f"{self.training_conditions.shape[1]}]"
            )
        if query.shape[0] == 1:
            query = np.repeat(query, sample_count, axis=0)
        elif query.shape[0] != sample_count:
            raise ValueError("cond debe contener una fila o exactamente n filas")
        if not np.isfinite(query).all():
            raise ValueError("cond contiene valores no finitos")

        horizon = self.config.horizon_steps if horizon_steps is None else horizon_steps
        if horizon <= 0:
            raise ValueError("horizon_steps debe ser positivo")
        blocks_per_path = int(np.ceil(horizon / self.config.block_length))
        neighbor_count = min(self.config.n_neighbors, len(self.blocks))
        rng = np.random.default_rng(
            self.config.random_state if seed is None else seed
        )
        sampled_batches = []
        for start in range(0, sample_count, self.config.distance_batch_size):
            stop = min(start + self.config.distance_batch_size, sample_count)
            query_batch = query[start:stop]
            distances = (
                np.sum(np.square(query_batch), axis=1, keepdims=True)
                + self._training_condition_squared_norms[None, :]
                - 2.0 * query_batch @ self.training_conditions.T
            )
            nearest = np.argpartition(
                distances,
                kth=neighbor_count - 1,
                axis=1,
            )[:, :neighbor_count]
            neighbor_choices = rng.integers(
                0,
                neighbor_count,
                size=(stop - start, blocks_per_path),
            )
            block_indices = np.take_along_axis(
                nearest,
                neighbor_choices,
                axis=1,
            )
            sampled_batches.append(self.blocks[block_indices])

        sampled = np.concatenate(sampled_batches, axis=0)
        paths = sampled.reshape(sample_count, -1, self.blocks.shape[-1])
        return paths[:, :horizon].copy()

    @property
    def n_neighbors(self) -> int:
        """Número máximo de vecinos usados por condición."""

        return min(
            self.config.n_neighbors,
            0 if self.blocks is None else len(self.blocks),
        )
