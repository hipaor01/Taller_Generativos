"""Aplicación común de cartera buy-and-hold y escenarios de estrés."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


class ScenarioCategory(str, Enum):
    """Familias comparables de escenarios de la aplicación."""

    HISTORICAL = "historical"
    PREFIXED = "prefixed"
    GENERATIVE = "generative"
    BASELINE = "baseline"


@dataclass(frozen=True)
class PortfolioConfig:
    """Cartera inicial sin rebalanceo, apalancamiento ni costes."""

    assets: Tuple[str, ...] = ("BTC", "ETH")
    weights: Tuple[float, ...] = (0.60, 0.40)
    initial_value: float = 100_000.0
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)
    summary_quantiles: Tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99)

    def validate(self) -> None:
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("assets debe contener identificadores únicos")
        if len(self.assets) != len(self.weights):
            raise ValueError("weights no coincide con assets")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights no admite pesos negativos")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("weights debe sumar 1")
        if not np.isfinite(self.initial_value) or self.initial_value <= 0:
            raise ValueError("initial_value debe ser positivo y finito")
        if not self.confidence_levels or any(
            not 0 < level < 1 for level in self.confidence_levels
        ):
            raise ValueError("confidence_levels debe estar entre 0 y 1")
        if not self.summary_quantiles or any(
            not 0 < value < 1 for value in self.summary_quantiles
        ):
            raise ValueError("summary_quantiles debe estar entre 0 y 1")

    @property
    def weight_by_asset(self) -> Mapping[str, float]:
        return dict(zip(self.assets, self.weights))


@dataclass(frozen=True)
class DistributionSummary:
    """Resumen serializable de una distribución escalar."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    quantiles: Mapping[str, float]


@dataclass(frozen=True)
class PortfolioRiskLevel:
    confidence_level: float
    value_at_risk_fraction: float
    expected_shortfall_fraction: float
    value_at_risk_amount: float
    expected_shortfall_amount: float


@dataclass(frozen=True)
class PortfolioPaths:
    """Revalorización completa de un lote de trayectorias."""

    values: NDArray[np.float64]
    loss_amount_paths: NDArray[np.float64]
    drawdown_paths: NDArray[np.float64]
    final_values: NDArray[np.float64]
    final_loss_amounts: NDArray[np.float64]
    maximum_loss_amounts: NDArray[np.float64]
    maximum_drawdowns: NDArray[np.float64]


@dataclass(frozen=True)
class StressScenarioSet:
    """Lote etiquetado de retornos logarítmicos conjuntos."""

    name: str
    category: ScenarioCategory
    log_returns: NDArray[np.float64]
    labels: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioStressSummary:
    """Resultado de cartera para un conjunto de escenarios."""

    name: str
    category: ScenarioCategory
    scenario_count: int
    horizon_steps: int
    final_value: DistributionSummary
    final_loss_fraction: DistributionSummary
    final_loss_amount: DistributionSummary
    maximum_loss_fraction: DistributionSummary
    maximum_loss_amount: DistributionSummary
    maximum_drawdown: DistributionSummary
    risk: Mapping[str, PortfolioRiskLevel]
    worst_final_loss_index: int
    worst_drawdown_index: int
    worst_final_loss_label: Optional[str]
    worst_drawdown_label: Optional[str]
    worst_final_loss_value_path: Tuple[float, ...]
    worst_drawdown_value_path: Tuple[float, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["category"] = self.category.value
        return payload


@dataclass(frozen=True)
class PortfolioStressReport:
    """Informe común para escenarios históricos, prefijados y generativos."""

    config: PortfolioConfig
    scenarios: Mapping[str, ScenarioStressSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "scenarios": {
                name: summary.to_dict() for name, summary in self.scenarios.items()
            },
        }

    def summary_records(self) -> Sequence[Dict[str, Any]]:
        records = []
        for summary in self.scenarios.values():
            records.append(
                {
                    "scenario_set": summary.name,
                    "category": summary.category.value,
                    "scenario_count": summary.scenario_count,
                    "horizon_steps": summary.horizon_steps,
                    "mean_final_value": summary.final_value.mean,
                    "mean_final_loss_fraction": summary.final_loss_fraction.mean,
                    "q95_final_loss_fraction": summary.final_loss_fraction.quantiles[
                        "q95"
                    ],
                    "mean_maximum_loss_fraction": summary.maximum_loss_fraction.mean,
                    "q95_maximum_loss_fraction": summary.maximum_loss_fraction.quantiles[
                        "q95"
                    ],
                    "mean_maximum_drawdown": summary.maximum_drawdown.mean,
                    "q95_maximum_drawdown": summary.maximum_drawdown.quantiles["q95"],
                    "worst_final_loss_fraction": summary.final_loss_fraction.maximum,
                    "worst_maximum_drawdown": summary.maximum_drawdown.maximum,
                }
            )
        return records

    def risk_records(self) -> Sequence[Dict[str, Any]]:
        records = []
        for summary in self.scenarios.values():
            for level in summary.risk.values():
                records.append(
                    {
                        "scenario_set": summary.name,
                        "category": summary.category.value,
                        "scenario_count": summary.scenario_count,
                        **asdict(level),
                    }
                )
        return records


class BuyAndHoldPortfolio:
    """Revaloriza y resume una cartera de pesos iniciales fijos."""

    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        self.config = config or PortfolioConfig()
        self.config.validate()

    def revalue(self, log_returns: NDArray[np.float64]) -> PortfolioPaths:
        """Convierte retornos 3D/4D en valores de cartera, pérdidas y drawdowns."""
        paths = _as_path_batch(log_returns, len(self.config.assets))
        cumulative = np.cumsum(paths, axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            asset_growth = np.exp(cumulative)
        if not np.isfinite(asset_growth).all():
            raise ValueError("La reconstrucción de riqueza produjo valores no finitos")

        allocations = self.config.initial_value * np.asarray(self.config.weights)
        values_without_initial = np.sum(asset_growth * allocations, axis=-1)
        initial = np.full(
            (len(paths), 1), self.config.initial_value, dtype=np.float64
        )
        values = np.concatenate((initial, values_without_initial), axis=1)
        loss_amount_paths = self.config.initial_value - values
        running_peak = np.maximum.accumulate(values, axis=1)
        drawdowns = 1.0 - values / running_peak
        return PortfolioPaths(
            values=values,
            loss_amount_paths=loss_amount_paths,
            drawdown_paths=drawdowns,
            final_values=values[:, -1],
            final_loss_amounts=loss_amount_paths[:, -1],
            maximum_loss_amounts=loss_amount_paths.max(axis=1),
            maximum_drawdowns=drawdowns.max(axis=1),
        )

    def summarize(self, scenarios: StressScenarioSet) -> ScenarioStressSummary:
        paths = _as_path_batch(scenarios.log_returns, len(self.config.assets))
        accumulator = PortfolioScenarioAccumulator(
            portfolio=self,
            name=scenarios.name,
            category=scenarios.category,
            scenario_count=len(paths),
            horizon_steps=paths.shape[1],
            labels=scenarios.labels,
            metadata=dict(scenarios.metadata),
        )
        accumulator.add(paths)
        return accumulator.finalize()

    def select_stress_paths(
        self,
        log_returns: NDArray[np.float64],
        count: int,
        *,
        criterion: str = "maximum_drawdown",
    ) -> NDArray[np.int64]:
        """Selecciona las peores trayectorias históricas sin alterar su contenido."""
        paths = _as_path_batch(log_returns, len(self.config.assets))
        if not 1 <= count <= len(paths):
            raise ValueError("count debe estar entre 1 y el número de trayectorias")
        valued = self.revalue(paths)
        if criterion == "maximum_drawdown":
            values = valued.maximum_drawdowns
        elif criterion == "final_loss":
            values = valued.final_loss_amounts
        elif criterion == "maximum_loss":
            values = valued.maximum_loss_amounts
        else:
            raise ValueError(
                "criterion debe ser maximum_drawdown, final_loss o maximum_loss"
            )
        return np.argsort(values)[-count:][::-1].astype(np.int64)


class PortfolioScenarioAccumulator:
    """Resume lotes de escenarios con métricas exactas y memoria acotada."""

    def __init__(
        self,
        portfolio: BuyAndHoldPortfolio,
        *,
        name: str,
        category: ScenarioCategory,
        scenario_count: int,
        horizon_steps: int,
        labels: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if scenario_count <= 0 or horizon_steps <= 0:
            raise ValueError("scenario_count y horizon_steps deben ser positivos")
        self.portfolio = portfolio
        self.name = name
        self.category = category
        self.scenario_count = scenario_count
        self.horizon_steps = horizon_steps
        self.labels = tuple(str(label) for label in labels)
        if self.labels and len(self.labels) != scenario_count:
            raise ValueError("labels no está alineado con scenario_count")
        self.metadata = dict(metadata or {})
        self._cursor = 0
        self._final_values = np.empty(scenario_count, dtype=np.float64)
        self._final_loss_amounts = np.empty(scenario_count, dtype=np.float64)
        self._maximum_loss_amounts = np.empty(scenario_count, dtype=np.float64)
        self._maximum_drawdowns = np.empty(scenario_count, dtype=np.float64)
        self._worst_final_loss_index = -1
        self._worst_drawdown_index = -1
        self._worst_final_loss_value_path: Tuple[float, ...] = ()
        self._worst_drawdown_value_path: Tuple[float, ...] = ()

    def add(self, log_returns: NDArray[np.float64]) -> None:
        paths = _as_path_batch(log_returns, len(self.portfolio.config.assets))
        if paths.shape[1] != self.horizon_steps:
            raise ValueError("El horizonte del lote no coincide con el acumulador")
        stop = self._cursor + len(paths)
        if stop > self.scenario_count:
            raise ValueError("El lote excede scenario_count")

        valued = self.portfolio.revalue(paths)
        target = slice(self._cursor, stop)
        self._final_values[target] = valued.final_values
        self._final_loss_amounts[target] = valued.final_loss_amounts
        self._maximum_loss_amounts[target] = valued.maximum_loss_amounts
        self._maximum_drawdowns[target] = valued.maximum_drawdowns

        local_final = int(np.argmax(valued.final_loss_amounts))
        global_final = self._cursor + local_final
        if (
            self._worst_final_loss_index < 0
            or valued.final_loss_amounts[local_final]
            > self._final_loss_amounts[self._worst_final_loss_index]
        ):
            self._worst_final_loss_index = global_final
            self._worst_final_loss_value_path = tuple(
                float(value) for value in valued.values[local_final]
            )

        local_drawdown = int(np.argmax(valued.maximum_drawdowns))
        global_drawdown = self._cursor + local_drawdown
        if (
            self._worst_drawdown_index < 0
            or valued.maximum_drawdowns[local_drawdown]
            > self._maximum_drawdowns[self._worst_drawdown_index]
        ):
            self._worst_drawdown_index = global_drawdown
            self._worst_drawdown_value_path = tuple(
                float(value) for value in valued.values[local_drawdown]
            )
        self._cursor = stop

    def finalize(self) -> ScenarioStressSummary:
        if self._cursor != self.scenario_count:
            raise ValueError(
                f"Faltan escenarios: recibidos {self._cursor} de {self.scenario_count}"
            )
        initial_value = self.portfolio.config.initial_value
        final_loss_fraction = self._final_loss_amounts / initial_value
        maximum_loss_fraction = self._maximum_loss_amounts / initial_value
        risk = {}
        for confidence_level in self.portfolio.config.confidence_levels:
            value_at_risk = float(np.quantile(final_loss_fraction, confidence_level))
            tail = final_loss_fraction[final_loss_fraction >= value_at_risk]
            expected_shortfall = float(tail.mean()) if len(tail) else value_at_risk
            key = f"{confidence_level:.4f}".rstrip("0").rstrip(".")
            risk[key] = PortfolioRiskLevel(
                confidence_level=confidence_level,
                value_at_risk_fraction=value_at_risk,
                expected_shortfall_fraction=expected_shortfall,
                value_at_risk_amount=value_at_risk * initial_value,
                expected_shortfall_amount=expected_shortfall * initial_value,
            )

        return ScenarioStressSummary(
            name=self.name,
            category=self.category,
            scenario_count=self.scenario_count,
            horizon_steps=self.horizon_steps,
            final_value=_summarize(
                self._final_values, self.portfolio.config.summary_quantiles
            ),
            final_loss_fraction=_summarize(
                final_loss_fraction, self.portfolio.config.summary_quantiles
            ),
            final_loss_amount=_summarize(
                self._final_loss_amounts, self.portfolio.config.summary_quantiles
            ),
            maximum_loss_fraction=_summarize(
                maximum_loss_fraction, self.portfolio.config.summary_quantiles
            ),
            maximum_loss_amount=_summarize(
                self._maximum_loss_amounts, self.portfolio.config.summary_quantiles
            ),
            maximum_drawdown=_summarize(
                self._maximum_drawdowns, self.portfolio.config.summary_quantiles
            ),
            risk=risk,
            worst_final_loss_index=self._worst_final_loss_index,
            worst_drawdown_index=self._worst_drawdown_index,
            worst_final_loss_label=(
                self.labels[self._worst_final_loss_index] if self.labels else None
            ),
            worst_drawdown_label=(
                self.labels[self._worst_drawdown_index] if self.labels else None
            ),
            worst_final_loss_value_path=self._worst_final_loss_value_path,
            worst_drawdown_value_path=self._worst_drawdown_value_path,
            metadata=self.metadata,
        )


class PortfolioStressApplication:
    """Ejecuta la misma valoración sobre cualquier conjunto de escenarios."""

    def __init__(self, portfolio: Optional[BuyAndHoldPortfolio] = None) -> None:
        self.portfolio = portfolio or BuyAndHoldPortfolio()

    def run(
        self, scenario_sets: Sequence[StressScenarioSet]
    ) -> PortfolioStressReport:
        if not scenario_sets:
            raise ValueError("Se necesita al menos un conjunto de escenarios")
        names = [scenario.name for scenario in scenario_sets]
        if len(names) != len(set(names)):
            raise ValueError("Los nombres de escenarios deben ser únicos")
        summaries = {
            scenario.name: self.portfolio.summarize(scenario)
            for scenario in scenario_sets
        }
        return PortfolioStressReport(self.portfolio.config, summaries)


def build_joint_shock_path(
    terminal_simple_returns: Sequence[float],
    *,
    horizon_steps: int = 120,
    shock_steps: int = 20,
    recovery_fraction: float = 0.0,
) -> NDArray[np.float64]:
    """Construye un shock conjunto reproducible con recuperación opcional."""
    terminal_returns = np.asarray(terminal_simple_returns, dtype=np.float64)
    if terminal_returns.ndim != 1 or len(terminal_returns) == 0:
        raise ValueError("terminal_simple_returns debe ser un vector no vacío")
    if np.any(terminal_returns <= -1) or not np.isfinite(terminal_returns).all():
        raise ValueError("Los retornos terminales deben ser finitos y mayores que -1")
    if horizon_steps <= 0 or not 1 <= shock_steps <= horizon_steps:
        raise ValueError("shock_steps debe estar entre 1 y horizon_steps")
    if not 0 <= recovery_fraction <= 1:
        raise ValueError("recovery_fraction debe estar entre 0 y 1")

    shock_log_wealth = np.log1p(terminal_returns)
    log_wealth = np.empty((horizon_steps, len(terminal_returns)), dtype=np.float64)
    log_wealth[:shock_steps] = np.linspace(
        shock_log_wealth / shock_steps,
        shock_log_wealth,
        shock_steps,
    )
    if shock_steps < horizon_steps:
        recovered_log_wealth = shock_log_wealth * (1.0 - recovery_fraction)
        log_wealth[shock_steps:] = np.linspace(
            shock_log_wealth,
            recovered_log_wealth,
            horizon_steps - shock_steps + 1,
        )[1:]
    return np.diff(
        np.concatenate((np.zeros((1, len(terminal_returns))), log_wealth), axis=0),
        axis=0,
    )


def default_prefixed_scenarios() -> Sequence[StressScenarioSet]:
    """Escenarios transparentes fijados antes de comparar generadores."""
    definitions = (
        ("prefixed_fast_crash", 20, 0.0),
        ("prefixed_gradual_selloff", 120, 0.0),
        ("prefixed_crash_half_recovery", 20, 0.50),
    )
    return [
        StressScenarioSet(
            name=name,
            category=ScenarioCategory.PREFIXED,
            log_returns=build_joint_shock_path(
                (-0.30, -0.40),
                horizon_steps=120,
                shock_steps=shock_steps,
                recovery_fraction=recovery_fraction,
            )[None, :, :],
            labels=(name,),
            metadata={
                "btc_terminal_shock": -0.30,
                "eth_terminal_shock": -0.40,
                "shock_steps": shock_steps,
                "recovery_fraction": recovery_fraction,
            },
        )
        for name, shock_steps, recovery_fraction in definitions
    ]


def _as_path_batch(
    log_returns: NDArray[np.float64], n_assets: int
) -> NDArray[np.float64]:
    paths = np.asarray(log_returns, dtype=np.float64)
    if paths.ndim == 4:
        paths = paths.reshape(-1, paths.shape[-2], paths.shape[-1])
    if paths.ndim != 3:
        raise ValueError(
            "log_returns debe tener forma [escenarios, tiempo, activos] "
            "o [condiciones, draws, tiempo, activos]"
        )
    if not paths.shape[0] or not paths.shape[1]:
        raise ValueError("log_returns no puede contener dimensiones vacías")
    if paths.shape[2] != n_assets:
        raise ValueError("El número de activos no coincide con la cartera")
    if not np.isfinite(paths).all():
        raise ValueError("log_returns contiene valores no finitos")
    return paths


def _summarize(
    values: NDArray[np.float64], quantiles: Sequence[float]
) -> DistributionSummary:
    array = np.asarray(values, dtype=np.float64).ravel()
    return DistributionSummary(
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        minimum=float(array.min()),
        maximum=float(array.max()),
        quantiles={
            _quantile_name(probability): float(np.quantile(array, probability))
            for probability in quantiles
        },
    )


def _quantile_name(probability: float) -> str:
    return f"q{int(round(probability * 100)):02d}"
