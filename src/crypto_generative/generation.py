"""Generación masiva y acotada en memoria para un único estado de mercado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from .portfolio import (
    BuyAndHoldPortfolio,
    PortfolioScenarioAccumulator,
    ScenarioCategory,
    ScenarioStressSummary,
)


class ConditionalScenarioSampler(Protocol):
    """Interfaz mínima compartida por CVAE, flow y GAN."""

    config: Any

    def sample(
        self,
        n: int,
        cond: NDArray[np.float32],
        *,
        seed: int | None = None,
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class MassiveGenerationConfig:
    scenario_count: int = 100_000
    batch_size: int = 1_000
    seed: int = 20_260_823

    def validate(self) -> None:
        if self.scenario_count <= 0:
            raise ValueError("scenario_count debe ser positivo")
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        if self.seed < 0:
            raise ValueError("seed no puede ser negativa")


@dataclass(frozen=True)
class MassiveGenerationResult:
    model_name: str
    output_path: Path
    scenario_count: int
    batch_size: int
    seed: int
    summary: ScenarioStressSummary


class MassiveConditionalScenarioGenerator:
    """Genera, desnormaliza, persiste y valora escenarios por lotes."""

    def __init__(
        self,
        return_mean: NDArray[np.float64],
        return_scale: NDArray[np.float64],
        *,
        portfolio: BuyAndHoldPortfolio | None = None,
    ) -> None:
        self.return_mean = np.asarray(return_mean, dtype=np.float32)
        self.return_scale = np.asarray(return_scale, dtype=np.float32)
        if self.return_mean.shape != self.return_scale.shape or self.return_mean.ndim != 1:
            raise ValueError("return_mean y return_scale deben ser vectores compatibles")
        if not np.isfinite(self.return_mean).all() or not np.isfinite(
            self.return_scale
        ).all():
            raise ValueError("Los parámetros de retorno contienen valores no finitos")
        if np.any(self.return_scale <= 0):
            raise ValueError("return_scale debe ser positivo")
        self.portfolio = portfolio or BuyAndHoldPortfolio()

    def generate(
        self,
        model_name: str,
        sampler: ConditionalScenarioSampler,
        normalized_condition: NDArray[np.float32],
        output_path: str | Path,
        *,
        config: MassiveGenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> MassiveGenerationResult:
        generation = config or MassiveGenerationConfig()
        generation.validate()
        condition = np.asarray(normalized_condition, dtype=np.float32)
        if condition.ndim == 1:
            condition = condition[None, :]
        condition_dim = int(sampler.config.condition_dim)
        if condition.shape != (1, condition_dim):
            raise ValueError(
                f"normalized_condition debe tener forma (1, {condition_dim})"
            )
        if not np.isfinite(condition).all():
            raise ValueError("normalized_condition contiene valores no finitos")

        horizon_steps = int(sampler.config.trajectory_length)
        n_assets = int(sampler.config.n_assets)
        if n_assets != len(self.return_mean):
            raise ValueError("El número de activos del modelo no coincide con el normalizador")

        destination = Path(output_path)
        if destination.suffix != ".npy":
            raise ValueError("output_path debe terminar en .npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()

        accumulator = PortfolioScenarioAccumulator(
            self.portfolio,
            name=model_name,
            category=ScenarioCategory.GENERATIVE,
            scenario_count=generation.scenario_count,
            horizon_steps=horizon_steps,
            metadata=dict(metadata or {}),
        )
        stored = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(generation.scenario_count, horizon_steps, n_assets),
        )
        try:
            for batch_index, start in enumerate(
                range(0, generation.scenario_count, generation.batch_size)
            ):
                stop = min(start + generation.batch_size, generation.scenario_count)
                batch_count = stop - start
                generated_normalized = np.asarray(
                    sampler.sample(
                        batch_count,
                        condition,
                        seed=generation.seed + batch_index,
                    ),
                    dtype=np.float32,
                )
                expected_shape = (batch_count, horizon_steps, n_assets)
                if generated_normalized.shape != expected_shape:
                    raise ValueError(
                        f"{model_name} produjo {generated_normalized.shape}; "
                        f"se esperaba {expected_shape}"
                    )
                generated_returns = (
                    generated_normalized * self.return_scale[None, None, :]
                    + self.return_mean[None, None, :]
                ).astype(np.float32)
                if not np.isfinite(generated_returns).all():
                    raise ValueError(f"{model_name} produjo retornos no finitos")
                stored[start:stop] = generated_returns
                accumulator.add(generated_returns)
                if progress is not None:
                    progress(stop, generation.scenario_count)
        except BaseException:
            del stored
            temporary.unlink(missing_ok=True)
            raise
        else:
            stored.flush()
            del stored
            temporary.replace(destination)

        return MassiveGenerationResult(
            model_name=model_name,
            output_path=destination,
            scenario_count=generation.scenario_count,
            batch_size=generation.batch_size,
            seed=generation.seed,
            summary=accumulator.finalize(),
        )
