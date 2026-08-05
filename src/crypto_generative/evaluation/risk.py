"""Metricas de VaR, Expected Shortfall y calibracion de perdidas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RiskMetricsConfig:
    """Configuracion común de riesgo para activos y cartera sin rebalanceo."""

    confidence_levels: Tuple[float, ...] = (0.95, 0.99)
    portfolio_weights: Optional[Tuple[float, ...]] = (0.60, 0.40)
    portfolio_name: str = "portfolio_60_40"
    es_stability_repetitions: int = 100
    es_stability_sample_size: int = 1_000
    random_state: int = 42

    def validate(self, n_assets: int) -> None:
        if not self.confidence_levels or any(
            not 0 < level < 1 for level in self.confidence_levels
        ):
            raise ValueError("confidence_levels debe contener valores entre 0 y 1")
        if len(set(self.confidence_levels)) != len(self.confidence_levels):
            raise ValueError("confidence_levels no puede contener duplicados")
        if self.portfolio_weights is not None:
            if len(self.portfolio_weights) != n_assets:
                raise ValueError("portfolio_weights no coincide con el numero de activos")
            if any(weight < 0 for weight in self.portfolio_weights):
                raise ValueError("portfolio_weights no admite pesos negativos")
            if not np.isclose(sum(self.portfolio_weights), 1.0):
                raise ValueError("portfolio_weights debe sumar 1")
        if self.es_stability_repetitions < 2:
            raise ValueError("es_stability_repetitions debe ser al menos 2")
        if self.es_stability_sample_size < 2:
            raise ValueError("es_stability_sample_size debe ser al menos 2")
        if not self.portfolio_name:
            raise ValueError("portfolio_name no puede estar vacio")


@dataclass(frozen=True)
class LossDistributionSummary:
    mean: float
    standard_deviation: float
    q01: float
    q05: float
    q50: float
    q95: float
    q99: float
    worst_loss: float
    profit_probability: float


@dataclass(frozen=True)
class LossPercentileSummary:
    mean: float
    q05: float
    q50: float
    q95: float
    uniform_ks_distance: float


@dataclass(frozen=True)
class RiskLevelEvaluation:
    confidence_level: float
    reference_unconditional_var: float
    candidate_unconditional_var: float
    unconditional_var_absolute_error: float
    reference_unconditional_es: float
    candidate_unconditional_es: float
    unconditional_es_absolute_error: float
    mean_forecast_var: float
    mean_forecast_es: float
    exception_count: int
    exception_rate: float
    expected_exception_rate: float
    coverage_absolute_error: float
    candidate_es_stability_standard_deviation: float
    candidate_es_stability_relative_standard_deviation: Optional[float]


@dataclass(frozen=True)
class RiskTargetEvaluation:
    target: str
    portfolio_weights: Optional[Mapping[str, float]]
    reference_losses: LossDistributionSummary
    candidate_losses: LossDistributionSummary
    reference_loss_candidate_percentiles: LossPercentileSummary
    levels: Mapping[str, RiskLevelEvaluation]


@dataclass(frozen=True)
class RiskEvaluation:
    """Resultado de riesgo para activos y cartera, agregado o condicional."""

    assets: Tuple[str, ...]
    forecast_mode: str
    config: RiskMetricsConfig
    by_target: Mapping[str, RiskTargetEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": list(self.assets),
            "forecast_mode": self.forecast_mode,
            "config": asdict(self.config),
            "by_target": {
                target: asdict(evaluation)
                for target, evaluation in self.by_target.items()
            },
        }

    def to_records(self) -> Sequence[Dict[str, Any]]:
        """Una fila por objetivo y nivel de confianza."""
        records = []
        for target, target_evaluation in self.by_target.items():
            for level_name, level in target_evaluation.levels.items():
                records.append(
                    {
                        "target": target,
                        "forecast_mode": self.forecast_mode,
                        "confidence_level": level.confidence_level,
                        "portfolio_weights": target_evaluation.portfolio_weights,
                        "reference_mean_loss": (
                            target_evaluation.reference_losses.mean
                        ),
                        "candidate_mean_loss": (
                            target_evaluation.candidate_losses.mean
                        ),
                        "reference_worst_loss": (
                            target_evaluation.reference_losses.worst_loss
                        ),
                        "candidate_worst_loss": (
                            target_evaluation.candidate_losses.worst_loss
                        ),
                        "loss_percentile_uniform_ks_distance": (
                            target_evaluation.reference_loss_candidate_percentiles.uniform_ks_distance
                        ),
                        **asdict(level),
                        "level_name": level_name,
                    }
                )
        return records


def evaluate_risk_paths(
    reference_paths: NDArray[np.float64],
    candidate_paths: NDArray[np.float64],
    assets: Sequence[str],
    config: RiskMetricsConfig,
) -> RiskEvaluation:
    """Evalúa riesgo final; el candidato puede ser agregado o condicional."""
    config.validate(len(assets))
    conditional = candidate_paths.ndim == 4
    forecast_mode = "conditional" if conditional else "unconditional"
    reference_losses = _loss_targets(
        reference_paths,
        assets,
        config.portfolio_weights,
        config.portfolio_name,
    )
    candidate_losses = _loss_targets(
        candidate_paths,
        assets,
        config.portfolio_weights,
        config.portfolio_name,
    )
    normalized_assets = tuple(str(asset) for asset in assets)
    evaluations: Dict[str, RiskTargetEvaluation] = {}

    for target_index, target in enumerate(reference_losses):
        reference_target_losses = reference_losses[target]
        candidate_target_losses = candidate_losses[target]
        flattened_candidate_losses = candidate_target_losses.reshape(-1)
        percentiles = _reference_loss_percentiles(
            reference_target_losses,
            candidate_target_losses,
        )
        levels: Dict[str, RiskLevelEvaluation] = {}

        for level_index, confidence_level in enumerate(config.confidence_levels):
            reference_var = float(
                np.quantile(reference_target_losses, confidence_level)
            )
            candidate_unconditional_var = float(
                np.quantile(flattened_candidate_losses, confidence_level)
            )
            reference_es = _expected_shortfall(reference_target_losses, reference_var)
            candidate_unconditional_es = _expected_shortfall(
                flattened_candidate_losses,
                candidate_unconditional_var,
            )

            if conditional:
                forecast_var = np.quantile(
                    candidate_target_losses,
                    confidence_level,
                    axis=1,
                )
                forecast_es = np.asarray(
                    [
                        _expected_shortfall(row, threshold)
                        for row, threshold in zip(candidate_target_losses, forecast_var)
                    ],
                    dtype=np.float64,
                )
                exceptions = reference_target_losses > forecast_var
            else:
                forecast_var = np.asarray([candidate_unconditional_var])
                forecast_es = np.asarray([candidate_unconditional_es])
                exceptions = reference_target_losses > candidate_unconditional_var

            stability_standard_deviation = _es_stability(
                flattened_candidate_losses,
                confidence_level,
                config,
                seed_offset=target_index * 1_000 + level_index,
            )
            expected_exception_rate = 1.0 - confidence_level
            exception_rate = float(exceptions.mean())
            level_name = f"{confidence_level:.4f}".rstrip("0").rstrip(".")
            levels[level_name] = RiskLevelEvaluation(
                confidence_level=confidence_level,
                reference_unconditional_var=reference_var,
                candidate_unconditional_var=candidate_unconditional_var,
                unconditional_var_absolute_error=abs(
                    reference_var - candidate_unconditional_var
                ),
                reference_unconditional_es=reference_es,
                candidate_unconditional_es=candidate_unconditional_es,
                unconditional_es_absolute_error=abs(
                    reference_es - candidate_unconditional_es
                ),
                mean_forecast_var=float(forecast_var.mean()),
                mean_forecast_es=float(forecast_es.mean()),
                exception_count=int(exceptions.sum()),
                exception_rate=exception_rate,
                expected_exception_rate=expected_exception_rate,
                coverage_absolute_error=abs(
                    exception_rate - expected_exception_rate
                ),
                candidate_es_stability_standard_deviation=(
                    stability_standard_deviation
                ),
                candidate_es_stability_relative_standard_deviation=(
                    stability_standard_deviation / abs(candidate_unconditional_es)
                    if candidate_unconditional_es != 0
                    else None
                ),
            )

        portfolio_weights = None
        if target == config.portfolio_name and config.portfolio_weights is not None:
            portfolio_weights = {
                normalized_assets[index]: float(weight)
                for index, weight in enumerate(config.portfolio_weights)
            }
        evaluations[target] = RiskTargetEvaluation(
            target=target,
            portfolio_weights=portfolio_weights,
            reference_losses=_summarize_losses(reference_target_losses),
            candidate_losses=_summarize_losses(flattened_candidate_losses),
            reference_loss_candidate_percentiles=_summarize_percentiles(percentiles),
            levels=levels,
        )

    return RiskEvaluation(normalized_assets, forecast_mode, config, evaluations)


def _loss_targets(
    paths: NDArray[np.float64],
    assets: Sequence[str],
    portfolio_weights: Optional[Sequence[float]],
    portfolio_name: str,
) -> Mapping[str, NDArray[np.float64]]:
    time_axis = 2 if paths.ndim == 4 else 1
    cumulative_log_return = paths.sum(axis=time_axis)
    with np.errstate(over="ignore", invalid="ignore"):
        terminal_wealth = np.exp(cumulative_log_return)
    if not np.isfinite(terminal_wealth).all():
        raise ValueError("La reconstruccion de riqueza produjo valores no finitos")
    losses: Dict[str, NDArray[np.float64]] = {
        str(asset): 1.0 - terminal_wealth[..., index]
        for index, asset in enumerate(assets)
    }
    if portfolio_weights is not None:
        portfolio_wealth = np.sum(
            terminal_wealth * np.asarray(portfolio_weights),
            axis=-1,
        )
        losses[portfolio_name] = 1.0 - portfolio_wealth
    return losses


def _expected_shortfall(losses: NDArray[np.float64], value_at_risk: float) -> float:
    tail = losses[losses >= value_at_risk]
    return float(tail.mean()) if len(tail) else float(value_at_risk)


def _reference_loss_percentiles(
    reference_losses: NDArray[np.float64],
    candidate_losses: NDArray[np.float64],
) -> NDArray[np.float64]:
    if candidate_losses.ndim == 1:
        sorted_candidate = np.sort(candidate_losses)
        return np.searchsorted(
            sorted_candidate,
            reference_losses,
            side="right",
        ) / len(sorted_candidate)
    return np.mean(candidate_losses <= reference_losses[:, None], axis=1)


def _es_stability(
    candidate_losses: NDArray[np.float64],
    confidence_level: float,
    config: RiskMetricsConfig,
    seed_offset: int,
) -> float:
    rng = np.random.default_rng(config.random_state + seed_offset)
    sample_size = min(config.es_stability_sample_size, len(candidate_losses))
    estimates = []
    for _ in range(config.es_stability_repetitions):
        sample = rng.choice(candidate_losses, size=sample_size, replace=True)
        value_at_risk = float(np.quantile(sample, confidence_level))
        estimates.append(_expected_shortfall(sample, value_at_risk))
    return float(np.std(estimates, ddof=1))


def _summarize_losses(losses: NDArray[np.float64]) -> LossDistributionSummary:
    return LossDistributionSummary(
        mean=float(losses.mean()),
        standard_deviation=float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
        q01=float(np.quantile(losses, 0.01)),
        q05=float(np.quantile(losses, 0.05)),
        q50=float(np.quantile(losses, 0.50)),
        q95=float(np.quantile(losses, 0.95)),
        q99=float(np.quantile(losses, 0.99)),
        worst_loss=float(losses.max()),
        profit_probability=float(np.mean(losses < 0)),
    )


def _summarize_percentiles(
    percentiles: NDArray[np.float64],
) -> LossPercentileSummary:
    sorted_percentiles = np.sort(percentiles)
    sample_size = len(sorted_percentiles)
    upper_gap = np.arange(1, sample_size + 1) / sample_size - sorted_percentiles
    lower_gap = sorted_percentiles - np.arange(sample_size) / sample_size
    return LossPercentileSummary(
        mean=float(percentiles.mean()),
        q05=float(np.quantile(percentiles, 0.05)),
        q50=float(np.quantile(percentiles, 0.50)),
        q95=float(np.quantile(percentiles, 0.95)),
        uniform_ks_distance=float(max(upper_gap.max(), lower_gap.max())),
    )
