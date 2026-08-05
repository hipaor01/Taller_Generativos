"""Metricas de dependencia cruzada para las trayectorias BTC-ETH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ._statistics import quantile_wasserstein


@dataclass(frozen=True)
class CrossAssetDependenceConfig:
    """Configuracion de dependencia BTC-ETH expresada en pasos temporales."""

    rolling_window: int = 20
    stress_quantile: float = 0.90
    joint_drop_quantile: float = 0.05
    lower_tail_quantile: float = 0.05

    def validate(self, horizon_steps: int) -> None:
        if self.rolling_window < 2 or self.rolling_window > horizon_steps:
            raise ValueError("rolling_window debe estar entre 2 y el horizonte")
        for name, value in (
            ("stress_quantile", self.stress_quantile),
            ("joint_drop_quantile", self.joint_drop_quantile),
            ("lower_tail_quantile", self.lower_tail_quantile),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} debe estar entre 0 y 1")


@dataclass(frozen=True)
class RollingCorrelationSummary:
    mean: Optional[float]
    standard_deviation: Optional[float]
    q05: Optional[float]
    q50: Optional[float]
    q95: Optional[float]
    valid_fraction: float


@dataclass(frozen=True)
class CrossAssetSummary:
    contemporaneous_correlation: Optional[float]
    rolling_correlation: RollingCorrelationSummary
    calm_correlation: Optional[float]
    stress_correlation: Optional[float]
    stress_frequency: float
    joint_drop_probability: float
    lower_tail_dependence: float


@dataclass(frozen=True)
class CrossAssetDependenceEvaluation:
    """Comparacion de dependencia para un unico par de activos."""

    asset_pair: Tuple[str, str]
    config: CrossAssetDependenceConfig
    reference: CrossAssetSummary
    candidate: CrossAssetSummary
    stress_threshold: float
    joint_drop_thresholds: Mapping[str, float]
    contemporaneous_correlation_absolute_error: Optional[float]
    rolling_correlation_wasserstein_1: Optional[float]
    calm_correlation_absolute_error: Optional[float]
    stress_correlation_absolute_error: Optional[float]
    stress_frequency_absolute_error: float
    joint_drop_probability_absolute_error: float
    lower_tail_dependence_absolute_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_pair": list(self.asset_pair),
            "config": asdict(self.config),
            "reference": asdict(self.reference),
            "candidate": asdict(self.candidate),
            "stress_threshold": self.stress_threshold,
            "joint_drop_thresholds": dict(self.joint_drop_thresholds),
            "errors": {
                "contemporaneous_correlation_absolute_error": (
                    self.contemporaneous_correlation_absolute_error
                ),
                "rolling_correlation_wasserstein_1": (
                    self.rolling_correlation_wasserstein_1
                ),
                "calm_correlation_absolute_error": (
                    self.calm_correlation_absolute_error
                ),
                "stress_correlation_absolute_error": (
                    self.stress_correlation_absolute_error
                ),
                "stress_frequency_absolute_error": (
                    self.stress_frequency_absolute_error
                ),
                "joint_drop_probability_absolute_error": (
                    self.joint_drop_probability_absolute_error
                ),
                "lower_tail_dependence_absolute_error": (
                    self.lower_tail_dependence_absolute_error
                ),
            },
        }

    def to_records(self) -> Sequence[Dict[str, Any]]:
        """Una fila lista para pandas.DataFrame."""
        record: Dict[str, Any] = {
            "asset_pair": "-".join(self.asset_pair),
            "stress_threshold": self.stress_threshold,
            **{
                f"joint_drop_threshold_{asset}": threshold
                for asset, threshold in self.joint_drop_thresholds.items()
            },
            "contemporaneous_correlation_absolute_error": (
                self.contemporaneous_correlation_absolute_error
            ),
            "rolling_correlation_wasserstein_1": (
                self.rolling_correlation_wasserstein_1
            ),
            "calm_correlation_absolute_error": self.calm_correlation_absolute_error,
            "stress_correlation_absolute_error": (
                self.stress_correlation_absolute_error
            ),
            "stress_frequency_absolute_error": self.stress_frequency_absolute_error,
            "joint_drop_probability_absolute_error": (
                self.joint_drop_probability_absolute_error
            ),
            "lower_tail_dependence_absolute_error": (
                self.lower_tail_dependence_absolute_error
            ),
        }
        for prefix, summary in (
            ("reference", self.reference),
            ("candidate", self.candidate),
        ):
            record.update(
                {
                    f"{prefix}_contemporaneous_correlation": (
                        summary.contemporaneous_correlation
                    ),
                    f"{prefix}_rolling_correlation_mean": (
                        summary.rolling_correlation.mean
                    ),
                    f"{prefix}_rolling_correlation_standard_deviation": (
                        summary.rolling_correlation.standard_deviation
                    ),
                    f"{prefix}_rolling_correlation_q05": (
                        summary.rolling_correlation.q05
                    ),
                    f"{prefix}_rolling_correlation_q50": (
                        summary.rolling_correlation.q50
                    ),
                    f"{prefix}_rolling_correlation_q95": (
                        summary.rolling_correlation.q95
                    ),
                    f"{prefix}_rolling_correlation_valid_fraction": (
                        summary.rolling_correlation.valid_fraction
                    ),
                    f"{prefix}_calm_correlation": summary.calm_correlation,
                    f"{prefix}_stress_correlation": summary.stress_correlation,
                    f"{prefix}_stress_frequency": summary.stress_frequency,
                    f"{prefix}_joint_drop_probability": (
                        summary.joint_drop_probability
                    ),
                    f"{prefix}_lower_tail_dependence": (
                        summary.lower_tail_dependence
                    ),
                }
            )
        return [record]


def evaluate_cross_asset_paths(
    reference_paths: NDArray[np.float64],
    candidate_paths: NDArray[np.float64],
    assets: Sequence[str],
    config: CrossAssetDependenceConfig,
    wasserstein_quantiles: int,
) -> CrossAssetDependenceEvaluation:
    """Compara dependencia BTC-ETH sin cruzar limites de trayectorias."""
    if len(assets) != 2:
        raise ValueError("La dependencia cruzada actual requiere exactamente 2 activos")
    config.validate(reference_paths.shape[1])

    reference_rolling_correlation, reference_valid = _rolling_correlation(
        reference_paths,
        config.rolling_window,
    )
    candidate_rolling_correlation, candidate_valid = _rolling_correlation(
        candidate_paths,
        config.rolling_window,
    )
    reference_rolling_volatility = _rolling_asset_volatility(
        reference_paths,
        config.rolling_window,
    )
    candidate_rolling_volatility = _rolling_asset_volatility(
        candidate_paths,
        config.rolling_window,
    )
    volatility_scale = np.median(reference_rolling_volatility, axis=(0, 1))
    volatility_scale = np.where(volatility_scale > 0, volatility_scale, 1.0)
    reference_stress_score = np.mean(
        reference_rolling_volatility / volatility_scale,
        axis=2,
    )
    candidate_stress_score = np.mean(
        candidate_rolling_volatility / volatility_scale,
        axis=2,
    )
    stress_threshold = float(
        np.quantile(reference_stress_score, config.stress_quantile)
    )
    joint_drop_values = np.quantile(
        reference_paths,
        config.joint_drop_quantile,
        axis=(0, 1),
    )

    reference_summary = _summarize_cross_asset(
        reference_paths,
        reference_rolling_correlation,
        reference_valid,
        reference_stress_score,
        stress_threshold,
        joint_drop_values,
        config.lower_tail_quantile,
    )
    candidate_summary = _summarize_cross_asset(
        candidate_paths,
        candidate_rolling_correlation,
        candidate_valid,
        candidate_stress_score,
        stress_threshold,
        joint_drop_values,
        config.lower_tail_quantile,
    )

    reference_valid_correlations = reference_rolling_correlation[reference_valid]
    candidate_valid_correlations = candidate_rolling_correlation[candidate_valid]
    rolling_wasserstein = (
        quantile_wasserstein(
            reference_valid_correlations,
            candidate_valid_correlations,
            wasserstein_quantiles,
        )
        if len(reference_valid_correlations) and len(candidate_valid_correlations)
        else None
    )
    normalized_assets = tuple(str(asset) for asset in assets)
    return CrossAssetDependenceEvaluation(
        asset_pair=(normalized_assets[0], normalized_assets[1]),
        config=config,
        reference=reference_summary,
        candidate=candidate_summary,
        stress_threshold=stress_threshold,
        joint_drop_thresholds={
            normalized_assets[index]: float(joint_drop_values[index])
            for index in range(2)
        },
        contemporaneous_correlation_absolute_error=_optional_absolute_error(
            reference_summary.contemporaneous_correlation,
            candidate_summary.contemporaneous_correlation,
        ),
        rolling_correlation_wasserstein_1=rolling_wasserstein,
        calm_correlation_absolute_error=_optional_absolute_error(
            reference_summary.calm_correlation,
            candidate_summary.calm_correlation,
        ),
        stress_correlation_absolute_error=_optional_absolute_error(
            reference_summary.stress_correlation,
            candidate_summary.stress_correlation,
        ),
        stress_frequency_absolute_error=abs(
            reference_summary.stress_frequency - candidate_summary.stress_frequency
        ),
        joint_drop_probability_absolute_error=abs(
            reference_summary.joint_drop_probability
            - candidate_summary.joint_drop_probability
        ),
        lower_tail_dependence_absolute_error=abs(
            reference_summary.lower_tail_dependence
            - candidate_summary.lower_tail_dependence
        ),
    )


def _summarize_cross_asset(
    paths: NDArray[np.float64],
    rolling_correlation: NDArray[np.float64],
    valid_correlation: NDArray[np.bool_],
    stress_score: NDArray[np.float64],
    stress_threshold: float,
    joint_drop_thresholds: NDArray[np.float64],
    lower_tail_quantile: float,
) -> CrossAssetSummary:
    stress = stress_score >= stress_threshold
    calm = ~stress
    flattened = paths.reshape(-1, 2)
    joint_drop = np.logical_and(
        flattened[:, 0] <= joint_drop_thresholds[0],
        flattened[:, 1] <= joint_drop_thresholds[1],
    )
    return CrossAssetSummary(
        contemporaneous_correlation=_safe_correlation(
            flattened[:, 0],
            flattened[:, 1],
        ),
        rolling_correlation=_summarize_rolling_correlation(
            rolling_correlation,
            valid_correlation,
        ),
        calm_correlation=_masked_mean(rolling_correlation, valid_correlation & calm),
        stress_correlation=_masked_mean(
            rolling_correlation,
            valid_correlation & stress,
        ),
        stress_frequency=float(stress.mean()),
        joint_drop_probability=float(joint_drop.mean()),
        lower_tail_dependence=_lower_tail_dependence(
            flattened,
            lower_tail_quantile,
        ),
    )


def _rolling_correlation(
    paths: NDArray[np.float64],
    window: int,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    first = np.lib.stride_tricks.sliding_window_view(
        paths[:, :, 0],
        window_shape=window,
        axis=1,
    )
    second = np.lib.stride_tricks.sliding_window_view(
        paths[:, :, 1],
        window_shape=window,
        axis=1,
    )
    first_centered = first - first.mean(axis=-1, keepdims=True)
    second_centered = second - second.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(
        np.sum(first_centered**2, axis=-1)
        * np.sum(second_centered**2, axis=-1)
    )
    valid = denominator > 0
    correlation = np.full(denominator.shape, np.nan, dtype=np.float64)
    correlation[valid] = (
        np.sum(first_centered * second_centered, axis=-1)[valid]
        / denominator[valid]
    )
    return correlation, valid


def _rolling_asset_volatility(
    paths: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    windows = np.lib.stride_tricks.sliding_window_view(
        paths,
        window_shape=window,
        axis=1,
    )
    return np.asarray(windows.std(axis=-1, ddof=1), dtype=np.float64)


def _summarize_rolling_correlation(
    correlations: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> RollingCorrelationSummary:
    values = correlations[valid]
    if not len(values):
        return RollingCorrelationSummary(None, None, None, None, None, 0.0)
    return RollingCorrelationSummary(
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        q05=float(np.quantile(values, 0.05)),
        q50=float(np.quantile(values, 0.50)),
        q95=float(np.quantile(values, 0.95)),
        valid_fraction=float(valid.mean()),
    )


def _lower_tail_dependence(
    flattened_paths: NDArray[np.float64],
    quantile: float,
) -> float:
    thresholds = np.quantile(flattened_paths, quantile, axis=0)
    joint_tail = np.logical_and(
        flattened_paths[:, 0] <= thresholds[0],
        flattened_paths[:, 1] <= thresholds[1],
    )
    return float(np.clip(joint_tail.mean() / quantile, 0.0, 1.0))


def _safe_correlation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> Optional[float]:
    if first.std(ddof=0) == 0 or second.std(ddof=0) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _masked_mean(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> Optional[float]:
    return float(values[mask].mean()) if bool(mask.any()) else None


def _optional_absolute_error(
    reference: Optional[float],
    candidate: Optional[float],
) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return abs(reference - candidate)
