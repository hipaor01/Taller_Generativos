"""Metricas comunes de dependencia temporal para trayectorias multivariantes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TemporalDependenceConfig:
    """Configuracion expresada en pasos de la frecuencia original."""

    max_lag: int = 20
    volatility_window: int = 20
    high_volatility_quantile: float = 0.90
    extreme_quantile: float = 0.99
    extreme_clustering_window: int = 4

    def validate(self, horizon_steps: int) -> None:
        if self.max_lag < 1 or self.max_lag >= horizon_steps:
            raise ValueError("max_lag debe estar entre 1 y horizonte - 1")
        if self.volatility_window < 2 or self.volatility_window > horizon_steps:
            raise ValueError("volatility_window debe estar entre 2 y el horizonte")
        if not 0 < self.high_volatility_quantile < 1:
            raise ValueError("high_volatility_quantile debe estar entre 0 y 1")
        if not 0 < self.extreme_quantile < 1:
            raise ValueError("extreme_quantile debe estar entre 0 y 1")
        if not 1 <= self.extreme_clustering_window < horizon_steps:
            raise ValueError(
                "extreme_clustering_window debe estar entre 1 y horizonte - 1"
            )


@dataclass(frozen=True)
class TemporalSeriesSummary:
    """Dependencia temporal de un activo dentro de un lote de trayectorias."""

    return_acf: Tuple[float, ...]
    absolute_return_acf: Tuple[float, ...]
    squared_return_acf: Tuple[float, ...]
    volatility_persistence: float
    high_volatility_frequency: float
    mean_high_volatility_run_length: float
    p95_high_volatility_run_length: float
    extreme_clustering_ratio: Optional[float]


@dataclass(frozen=True)
class TemporalAssetEvaluation:
    """Comparacion temporal entre referencia y candidato para un activo."""

    asset: str
    reference: TemporalSeriesSummary
    candidate: TemporalSeriesSummary
    high_volatility_threshold: float
    extreme_threshold: float
    return_acf_rmse: float
    absolute_return_acf_rmse: float
    squared_return_acf_rmse: float
    volatility_persistence_absolute_error: float
    high_volatility_frequency_absolute_error: float
    mean_high_volatility_run_length_absolute_error: float
    p95_high_volatility_run_length_absolute_error: float
    extreme_clustering_ratio_absolute_error: Optional[float]


@dataclass(frozen=True)
class TemporalDependenceEvaluation:
    """Resultado temporal completo en un formato serializable y tabular."""

    assets: Tuple[str, ...]
    config: TemporalDependenceConfig
    by_asset: Mapping[str, TemporalAssetEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": list(self.assets),
            "config": asdict(self.config),
            "by_asset": {
                asset: asdict(evaluation)
                for asset, evaluation in self.by_asset.items()
            },
        }

    def to_records(self) -> Sequence[Dict[str, Any]]:
        """Una fila por activo con métricas comparativas y resúmenes escalares."""
        records = []
        comparison_fields = (
            "return_acf_rmse",
            "absolute_return_acf_rmse",
            "squared_return_acf_rmse",
            "volatility_persistence_absolute_error",
            "high_volatility_frequency_absolute_error",
            "mean_high_volatility_run_length_absolute_error",
            "p95_high_volatility_run_length_absolute_error",
            "extreme_clustering_ratio_absolute_error",
        )
        summary_fields = (
            "volatility_persistence",
            "high_volatility_frequency",
            "mean_high_volatility_run_length",
            "p95_high_volatility_run_length",
            "extreme_clustering_ratio",
        )
        for asset in self.assets:
            evaluation = self.by_asset[asset]
            record: Dict[str, Any] = {
                "asset": asset,
                "high_volatility_threshold": evaluation.high_volatility_threshold,
                "extreme_threshold": evaluation.extreme_threshold,
            }
            record.update(
                {
                    field: getattr(evaluation, field)
                    for field in comparison_fields
                }
            )
            for prefix, summary in (
                ("reference", evaluation.reference),
                ("candidate", evaluation.candidate),
            ):
                record.update(
                    {
                        f"{prefix}_{field}": getattr(summary, field)
                        for field in summary_fields
                    }
                )
            records.append(record)
        return records


def evaluate_temporal_paths(
    reference_paths: NDArray[np.float64],
    candidate_paths: NDArray[np.float64],
    assets: Sequence[str],
    config: TemporalDependenceConfig,
) -> TemporalDependenceEvaluation:
    """Calcula métricas temporales sin crear pares entre trayectorias."""
    config.validate(reference_paths.shape[1])
    evaluations: Dict[str, TemporalAssetEvaluation] = {}

    for asset_index, asset in enumerate(assets):
        reference = reference_paths[:, :, asset_index]
        candidate = candidate_paths[:, :, asset_index]
        reference_rolling_volatility = _rolling_volatility(
            reference,
            config.volatility_window,
        )
        high_volatility_threshold = float(
            np.quantile(
                reference_rolling_volatility,
                config.high_volatility_quantile,
            )
        )
        extreme_threshold = float(
            np.quantile(np.abs(reference), config.extreme_quantile)
        )

        reference_summary = _summarize_temporal_series(
            reference,
            reference_rolling_volatility,
            high_volatility_threshold,
            extreme_threshold,
            config,
        )
        candidate_summary = _summarize_temporal_series(
            candidate,
            _rolling_volatility(candidate, config.volatility_window),
            high_volatility_threshold,
            extreme_threshold,
            config,
        )

        evaluations[asset] = TemporalAssetEvaluation(
            asset=str(asset),
            reference=reference_summary,
            candidate=candidate_summary,
            high_volatility_threshold=high_volatility_threshold,
            extreme_threshold=extreme_threshold,
            return_acf_rmse=_rmse(
                reference_summary.return_acf,
                candidate_summary.return_acf,
            ),
            absolute_return_acf_rmse=_rmse(
                reference_summary.absolute_return_acf,
                candidate_summary.absolute_return_acf,
            ),
            squared_return_acf_rmse=_rmse(
                reference_summary.squared_return_acf,
                candidate_summary.squared_return_acf,
            ),
            volatility_persistence_absolute_error=abs(
                reference_summary.volatility_persistence
                - candidate_summary.volatility_persistence
            ),
            high_volatility_frequency_absolute_error=abs(
                reference_summary.high_volatility_frequency
                - candidate_summary.high_volatility_frequency
            ),
            mean_high_volatility_run_length_absolute_error=abs(
                reference_summary.mean_high_volatility_run_length
                - candidate_summary.mean_high_volatility_run_length
            ),
            p95_high_volatility_run_length_absolute_error=abs(
                reference_summary.p95_high_volatility_run_length
                - candidate_summary.p95_high_volatility_run_length
            ),
            extreme_clustering_ratio_absolute_error=_optional_absolute_error(
                reference_summary.extreme_clustering_ratio,
                candidate_summary.extreme_clustering_ratio,
            ),
        )

    normalized_assets = tuple(str(asset) for asset in assets)
    return TemporalDependenceEvaluation(normalized_assets, config, evaluations)


def _summarize_temporal_series(
    paths: NDArray[np.float64],
    rolling_volatility: NDArray[np.float64],
    high_volatility_threshold: float,
    extreme_threshold: float,
    config: TemporalDependenceConfig,
) -> TemporalSeriesSummary:
    return_acf = _pooled_acf(paths, config.max_lag)
    absolute_return_acf = _pooled_acf(np.abs(paths), config.max_lag)
    squared_return_acf = _pooled_acf(paths**2, config.max_lag)
    high_volatility = rolling_volatility >= high_volatility_threshold
    high_volatility_run_lengths = _true_run_lengths(high_volatility)
    extreme_events = np.abs(paths) >= extreme_threshold

    return TemporalSeriesSummary(
        return_acf=tuple(float(value) for value in return_acf),
        absolute_return_acf=tuple(float(value) for value in absolute_return_acf),
        squared_return_acf=tuple(float(value) for value in squared_return_acf),
        volatility_persistence=float(np.clip(absolute_return_acf, 0, None).sum()),
        high_volatility_frequency=float(high_volatility.mean()),
        mean_high_volatility_run_length=(
            float(high_volatility_run_lengths.mean())
            if len(high_volatility_run_lengths)
            else 0.0
        ),
        p95_high_volatility_run_length=(
            float(np.quantile(high_volatility_run_lengths, 0.95))
            if len(high_volatility_run_lengths)
            else 0.0
        ),
        extreme_clustering_ratio=_extreme_clustering_ratio(
            extreme_events,
            config.extreme_clustering_window,
        ),
    )


def _pooled_acf(paths: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    centered = paths - paths.mean()
    correlations = []
    for lag in range(1, max_lag + 1):
        left = centered[:, :-lag]
        right = centered[:, lag:]
        denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
        correlation = np.sum(left * right) / denominator if denominator > 0 else 0.0
        correlations.append(correlation)
    return np.asarray(correlations, dtype=np.float64)


def _rolling_volatility(
    paths: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    windows = np.lib.stride_tricks.sliding_window_view(
        paths,
        window_shape=window,
        axis=1,
    )
    return np.asarray(windows.std(axis=-1, ddof=1), dtype=np.float64)


def _true_run_lengths(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    lengths = []
    for path_mask in mask:
        padded = np.concatenate(([False], path_mask, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        lengths.extend((stops - starts).tolist())
    return np.asarray(lengths, dtype=np.int64)


def _extreme_clustering_ratio(
    extreme_events: NDArray[np.bool_],
    window: int,
) -> Optional[float]:
    past_windows = np.lib.stride_tricks.sliding_window_view(
        extreme_events[:, :-1],
        window_shape=window,
        axis=1,
    )
    prior_extreme = past_windows.any(axis=-1)
    current_extreme = extreme_events[:, window:]
    baseline_probability = float(current_extreme.mean())
    if baseline_probability == 0 or not bool(prior_extreme.any()):
        return None
    conditional_probability = float(current_extreme[prior_extreme].mean())
    return conditional_probability / baseline_probability


def _rmse(reference: Sequence[float], candidate: Sequence[float]) -> float:
    difference = np.asarray(reference) - np.asarray(candidate)
    return float(np.sqrt(np.mean(difference**2)))


def _optional_absolute_error(
    reference: Optional[float],
    candidate: Optional[float],
) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return abs(reference - candidate)
