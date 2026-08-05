"""Metricas de forma y evolucion de trayectorias financieras."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ._statistics import quantile_name, quantile_wasserstein


@dataclass(frozen=True)
class TrajectoryMetricsConfig:
    """Configuracion de métricas de trayectoria."""

    periods_per_year: int = 4 * 365
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99)

    def validate(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year debe ser positivo")
        if not self.quantiles or any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles debe contener probabilidades entre 0 y 1")


@dataclass(frozen=True)
class ScalarDistributionSummary:
    mean: float
    standard_deviation: float
    quantiles: Mapping[str, float]


@dataclass(frozen=True)
class PathMetricEvaluation:
    metric: str
    unit: str
    reference: ScalarDistributionSummary
    candidate: ScalarDistributionSummary
    wasserstein_1: float
    normalized_wasserstein_1: Optional[float]


@dataclass(frozen=True)
class TrajectoryAssetEvaluation:
    asset: str
    metrics: Mapping[str, PathMetricEvaluation]


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Resultado de trayectoria con una fila tabular por activo y métrica."""

    assets: Tuple[str, ...]
    config: TrajectoryMetricsConfig
    by_asset: Mapping[str, TrajectoryAssetEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": list(self.assets),
            "config": asdict(self.config),
            "by_asset": {
                asset: {
                    metric: asdict(evaluation)
                    for metric, evaluation in asset_evaluation.metrics.items()
                }
                for asset, asset_evaluation in self.by_asset.items()
            },
        }

    def to_records(self) -> Sequence[Dict[str, Any]]:
        records = []
        for asset in self.assets:
            for metric, evaluation in self.by_asset[asset].metrics.items():
                record: Dict[str, Any] = {
                    "asset": asset,
                    "metric": metric,
                    "unit": evaluation.unit,
                    "wasserstein_1": evaluation.wasserstein_1,
                    "normalized_wasserstein_1": evaluation.normalized_wasserstein_1,
                    "reference_mean": evaluation.reference.mean,
                    "reference_standard_deviation": (
                        evaluation.reference.standard_deviation
                    ),
                    "candidate_mean": evaluation.candidate.mean,
                    "candidate_standard_deviation": (
                        evaluation.candidate.standard_deviation
                    ),
                }
                for prefix, summary in (
                    ("reference", evaluation.reference),
                    ("candidate", evaluation.candidate),
                ):
                    record.update(
                        {
                            f"{prefix}_{name}": value
                            for name, value in summary.quantiles.items()
                        }
                    )
                records.append(record)
        return records


METRIC_UNITS = {
    "final_cumulative_return": "simple_return",
    "realized_volatility": "annualized_volatility",
    "maximum_drawdown": "fraction",
    "maximum_drawdown_duration_steps": "steps",
    "intrahorizon_maximum_return": "simple_return",
    "intrahorizon_minimum_return": "simple_return",
    "time_to_minimum_value_steps": "steps",
}


def evaluate_trajectory_paths(
    reference_paths: NDArray[np.float64],
    candidate_paths: NDArray[np.float64],
    assets: Sequence[str],
    config: TrajectoryMetricsConfig,
    wasserstein_quantiles: int,
) -> TrajectoryEvaluation:
    """Compara distribuciones de métricas calculadas trayectoria a trayectoria."""
    config.validate()
    evaluations: Dict[str, TrajectoryAssetEvaluation] = {}

    for asset_index, asset in enumerate(assets):
        reference_metrics = compute_asset_path_metrics(
            reference_paths[:, :, asset_index],
            config.periods_per_year,
        )
        candidate_metrics = compute_asset_path_metrics(
            candidate_paths[:, :, asset_index],
            config.periods_per_year,
        )
        metric_evaluations: Dict[str, PathMetricEvaluation] = {}
        for metric, unit in METRIC_UNITS.items():
            reference_values = reference_metrics[metric]
            candidate_values = candidate_metrics[metric]
            reference_summary = _summarize_distribution(
                reference_values,
                config.quantiles,
            )
            candidate_summary = _summarize_distribution(
                candidate_values,
                config.quantiles,
            )
            wasserstein = quantile_wasserstein(
                reference_values,
                candidate_values,
                wasserstein_quantiles,
            )
            reference_scale = reference_summary.standard_deviation
            metric_evaluations[metric] = PathMetricEvaluation(
                metric=metric,
                unit=unit,
                reference=reference_summary,
                candidate=candidate_summary,
                wasserstein_1=wasserstein,
                normalized_wasserstein_1=(
                    wasserstein / reference_scale if reference_scale > 0 else None
                ),
            )
        evaluations[str(asset)] = TrajectoryAssetEvaluation(
            asset=str(asset),
            metrics=metric_evaluations,
        )

    normalized_assets = tuple(str(asset) for asset in assets)
    return TrajectoryEvaluation(normalized_assets, config, evaluations)


def compute_asset_path_metrics(
    log_returns: NDArray[np.float64],
    periods_per_year: int,
) -> Mapping[str, NDArray[np.float64]]:
    """Calcula métricas por path partiendo de riqueza inicial unitaria."""
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        wealth_without_initial = np.exp(cumulative_log_returns)
    if not np.isfinite(wealth_without_initial).all():
        raise ValueError("La reconstruccion de riqueza produjo valores no finitos")
    initial_wealth = np.ones((len(log_returns), 1), dtype=np.float64)
    wealth = np.concatenate((initial_wealth, wealth_without_initial), axis=1)
    running_peak = np.maximum.accumulate(wealth, axis=1)
    drawdown = 1.0 - wealth / running_peak

    current_duration = np.zeros(len(log_returns), dtype=np.int64)
    maximum_duration = np.zeros(len(log_returns), dtype=np.int64)
    for step in range(1, wealth.shape[1]):
        underwater = drawdown[:, step] > 1e-12
        current_duration = np.where(underwater, current_duration + 1, 0)
        maximum_duration = np.maximum(maximum_duration, current_duration)

    simple_returns = wealth_without_initial - 1.0
    realized_volatility = (
        log_returns.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
        if log_returns.shape[1] > 1
        else np.zeros(len(log_returns), dtype=np.float64)
    )
    return {
        "final_cumulative_return": simple_returns[:, -1],
        "realized_volatility": realized_volatility,
        "maximum_drawdown": drawdown.max(axis=1),
        "maximum_drawdown_duration_steps": maximum_duration.astype(np.float64),
        "intrahorizon_maximum_return": simple_returns.max(axis=1),
        "intrahorizon_minimum_return": simple_returns.min(axis=1),
        "time_to_minimum_value_steps": (
            np.argmin(wealth_without_initial, axis=1) + 1
        ).astype(np.float64),
    }


def _summarize_distribution(
    values: NDArray[np.float64],
    quantiles: Sequence[float],
) -> ScalarDistributionSummary:
    return ScalarDistributionSummary(
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        quantiles={
            quantile_name(probability): float(np.quantile(values, probability))
            for probability in quantiles
        },
    )
