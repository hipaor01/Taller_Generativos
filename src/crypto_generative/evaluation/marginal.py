"""Interfaz del evaluador comun y metricas de distribucion marginal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ._statistics import quantile_name, quantile_wasserstein
from .cross_asset import (
    CrossAssetDependenceConfig,
    CrossAssetDependenceEvaluation,
    evaluate_cross_asset_paths,
)
from .diversity import (
    DiversityMemorizationConfig,
    DiversityMemorizationEvaluation,
    evaluate_diversity_paths,
)
from .temporal import (
    TemporalDependenceConfig,
    TemporalDependenceEvaluation,
    evaluate_temporal_paths,
)
from .risk import RiskEvaluation, RiskMetricsConfig, evaluate_risk_paths
from .trajectory import (
    TrajectoryEvaluation,
    TrajectoryMetricsConfig,
    evaluate_trajectory_paths,
)


@dataclass(frozen=True)
class DistributionSummary:
    """Resumen univariante en unidades originales de retorno logaritmico."""

    mean: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    quantiles: Mapping[str, float]
    extreme_frequency: float
    mean_absolute_extreme: Optional[float]


@dataclass(frozen=True)
class MarginalAssetEvaluation:
    """Comparacion marginal entre referencia y candidato para un activo."""

    asset: str
    reference: DistributionSummary
    candidate: DistributionSummary
    extreme_threshold: float
    wasserstein_1: float
    normalized_wasserstein_1: Optional[float]


@dataclass(frozen=True)
class MarginalEvaluation:
    """Resultado marginal completo, independiente de pandas u otras librerias."""

    assets: Tuple[str, ...]
    by_asset: Mapping[str, MarginalAssetEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": list(self.assets),
            "by_asset": {
                asset: asdict(evaluation)
                for asset, evaluation in self.by_asset.items()
            },
        }

    def to_records(self) -> Sequence[Dict[str, Any]]:
        """Formato tabular que puede pasarse directamente a pandas.DataFrame."""
        records = []
        for asset in self.assets:
            evaluation = self.by_asset[asset]
            record: Dict[str, Any] = {
                "asset": asset,
                "wasserstein_1": evaluation.wasserstein_1,
                "normalized_wasserstein_1": evaluation.normalized_wasserstein_1,
                "extreme_threshold": evaluation.extreme_threshold,
            }
            for prefix, summary in (
                ("reference", evaluation.reference),
                ("candidate", evaluation.candidate),
            ):
                record.update(
                    {
                        f"{prefix}_mean": summary.mean,
                        f"{prefix}_standard_deviation": summary.standard_deviation,
                        f"{prefix}_skewness": summary.skewness,
                        f"{prefix}_excess_kurtosis": summary.excess_kurtosis,
                        f"{prefix}_extreme_frequency": summary.extreme_frequency,
                        f"{prefix}_mean_absolute_extreme": summary.mean_absolute_extreme,
                    }
                )
                record.update(
                    {
                        f"{prefix}_{name}": value
                        for name, value in summary.quantiles.items()
                    }
                )
            records.append(record)
        return records


class TrajectoryEvaluator:
    """Interfaz comun para comparar retornos reales y generados.

    Los lotes deben tener forma ``[trayectorias, tiempo, activos]`` y contener
    retornos logaritmicos en unidades originales, nunca normalizados.
    """

    DEFAULT_QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)

    def __init__(
        self,
        assets: Sequence[str] = ("BTC", "ETH"),
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        extreme_quantile: float = 0.99,
        wasserstein_quantiles: int = 1_001,
    ) -> None:
        normalized_assets = tuple(str(asset) for asset in assets)
        if not normalized_assets or len(set(normalized_assets)) != len(normalized_assets):
            raise ValueError("assets debe contener nombres unicos")
        if not quantiles or any(not 0 < value < 1 for value in quantiles):
            raise ValueError("quantiles debe contener probabilidades entre 0 y 1")
        if not 0 < extreme_quantile < 1:
            raise ValueError("extreme_quantile debe estar entre 0 y 1")
        if wasserstein_quantiles < 2:
            raise ValueError("wasserstein_quantiles debe ser al menos 2")

        self.assets = normalized_assets
        self.quantiles = tuple(float(value) for value in quantiles)
        self.extreme_quantile = float(extreme_quantile)
        self.wasserstein_quantiles = int(wasserstein_quantiles)

    def evaluate_marginals(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
    ) -> MarginalEvaluation:
        """Compara marginales agregando trayectorias y pasos por activo."""
        reference, candidate = self._validate_pair(reference_paths, candidate_paths)
        evaluations: Dict[str, MarginalAssetEvaluation] = {}

        for asset_index, asset in enumerate(self.assets):
            reference_values = reference[:, :, asset_index].ravel()
            candidate_values = candidate[:, :, asset_index].ravel()
            extreme_threshold = float(
                np.quantile(np.abs(reference_values), self.extreme_quantile)
            )
            reference_summary = self._summarize(reference_values, extreme_threshold)
            candidate_summary = self._summarize(candidate_values, extreme_threshold)
            wasserstein = quantile_wasserstein(
                reference_values,
                candidate_values,
                self.wasserstein_quantiles,
            )
            reference_scale = reference_summary.standard_deviation
            normalized_wasserstein = (
                wasserstein / reference_scale if reference_scale > 0 else None
            )
            evaluations[asset] = MarginalAssetEvaluation(
                asset=asset,
                reference=reference_summary,
                candidate=candidate_summary,
                extreme_threshold=extreme_threshold,
                wasserstein_1=wasserstein,
                normalized_wasserstein_1=normalized_wasserstein,
            )

        return MarginalEvaluation(self.assets, evaluations)

    def evaluate_temporal_dependence(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
        config: Optional[TemporalDependenceConfig] = None,
    ) -> TemporalDependenceEvaluation:
        """Compara dependencia temporal respetando los limites de cada path."""
        reference, candidate = self._validate_pair(reference_paths, candidate_paths)
        temporal_config = config or TemporalDependenceConfig()
        return evaluate_temporal_paths(
            reference,
            candidate,
            self.assets,
            temporal_config,
        )

    def evaluate_cross_asset_dependence(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
        config: Optional[CrossAssetDependenceConfig] = None,
    ) -> CrossAssetDependenceEvaluation:
        """Compara la dependencia conjunta del par de activos."""
        reference, candidate = self._validate_pair(reference_paths, candidate_paths)
        cross_asset_config = config or CrossAssetDependenceConfig()
        return evaluate_cross_asset_paths(
            reference,
            candidate,
            self.assets,
            cross_asset_config,
            self.wasserstein_quantiles,
        )

    def evaluate_trajectories(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
        config: Optional[TrajectoryMetricsConfig] = None,
    ) -> TrajectoryEvaluation:
        """Compara la forma y evolución financiera de las trayectorias."""
        reference, candidate = self._validate_pair(reference_paths, candidate_paths)
        trajectory_config = config or TrajectoryMetricsConfig()
        return evaluate_trajectory_paths(
            reference,
            candidate,
            self.assets,
            trajectory_config,
            self.wasserstein_quantiles,
        )

    def evaluate_risk(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
        config: Optional[RiskMetricsConfig] = None,
    ) -> RiskEvaluation:
        """Evalúa VaR y ES agregados o condicionales según el candidato."""
        reference = self._validate_paths(reference_paths, "reference_paths")
        candidate = self._validate_risk_candidate(candidate_paths, reference)
        risk_config = config or RiskMetricsConfig()
        return evaluate_risk_paths(reference, candidate, self.assets, risk_config)

    def evaluate_diversity_and_memorization(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
        training_paths: Optional[NDArray[np.float64]] = None,
        config: Optional[DiversityMemorizationConfig] = None,
    ) -> DiversityMemorizationEvaluation:
        """Compara diversidad, cobertura y cercanía a trayectorias de train."""
        reference = self._validate_paths(reference_paths, "reference_paths")
        candidate = self._validate_diversity_candidate(candidate_paths, reference)
        training = None
        if training_paths is not None:
            training = self._validate_paths(training_paths, "training_paths")
            if training.shape[1:] != reference.shape[1:]:
                raise ValueError(
                    "training_paths debe compartir horizonte y activos con referencia"
                )
        diversity_config = config or DiversityMemorizationConfig()
        return evaluate_diversity_paths(
            reference,
            candidate,
            training,
            self.assets,
            diversity_config,
        )

    def _validate_pair(
        self,
        reference_paths: NDArray[np.float64],
        candidate_paths: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        reference = self._validate_paths(reference_paths, "reference_paths")
        candidate = self._validate_paths(candidate_paths, "candidate_paths")
        if reference.shape[1:] != candidate.shape[1:]:
            raise ValueError(
                "reference_paths y candidate_paths deben compartir horizonte y activos"
            )
        return reference, candidate

    def _validate_paths(
        self,
        paths: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        values = np.asarray(paths, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError(f"{name} debe tener forma [trayectorias, tiempo, activos]")
        if not values.shape[0] or not values.shape[1]:
            raise ValueError(f"{name} no puede contener dimensiones vacias")
        if values.shape[2] != len(self.assets):
            raise ValueError(
                f"{name} contiene {values.shape[2]} activos; se esperaban {len(self.assets)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contiene valores no finitos")
        return values

    def _validate_risk_candidate(
        self,
        paths: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.asarray(paths, dtype=np.float64)
        if values.ndim == 3:
            candidate = self._validate_paths(values, "candidate_paths")
            if candidate.shape[1:] != reference.shape[1:]:
                raise ValueError(
                    "reference_paths y candidate_paths deben compartir horizonte y activos"
                )
            return candidate
        if values.ndim != 4:
            raise ValueError(
                "candidate_paths de riesgo debe ser [paths, tiempo, activos] o "
                "[condiciones, draws, tiempo, activos]"
            )
        if not values.shape[0] or not values.shape[1]:
            raise ValueError("candidate_paths no puede contener dimensiones vacias")
        if values.shape[0] != reference.shape[0]:
            raise ValueError("candidate_paths no coincide con las condiciones reales")
        if values.shape[2:] != reference.shape[1:]:
            raise ValueError(
                "reference_paths y candidate_paths deben compartir horizonte y activos"
            )
        if not np.isfinite(values).all():
            raise ValueError("candidate_paths contiene valores no finitos")
        return values

    def _validate_diversity_candidate(
        self,
        paths: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.asarray(paths, dtype=np.float64)
        if values.ndim == 4:
            if not values.shape[0] or not values.shape[1]:
                raise ValueError("candidate_paths no puede contener dimensiones vacias")
            values = values.reshape(-1, values.shape[-2], values.shape[-1])
        candidate = self._validate_paths(values, "candidate_paths")
        if candidate.shape[1:] != reference.shape[1:]:
            raise ValueError(
                "reference_paths y candidate_paths deben compartir horizonte y activos"
            )
        return candidate

    def _summarize(
        self,
        values: NDArray[np.float64],
        extreme_threshold: float,
    ) -> DistributionSummary:
        mean = float(values.mean())
        population_scale = float(values.std(ddof=0))
        sample_scale = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if population_scale > 0:
            standardized = (values - mean) / population_scale
            skewness = float(np.mean(standardized**3))
            excess_kurtosis = float(np.mean(standardized**4) - 3.0)
        else:
            skewness = 0.0
            excess_kurtosis = 0.0

        extreme_mask = np.abs(values) >= extreme_threshold
        mean_absolute_extreme = (
            float(np.abs(values[extreme_mask]).mean())
            if bool(extreme_mask.any())
            else None
        )
        return DistributionSummary(
            mean=mean,
            standard_deviation=sample_scale,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            quantiles={
                quantile_name(probability): float(np.quantile(values, probability))
                for probability in self.quantiles
            },
            extreme_frequency=float(extreme_mask.mean()),
            mean_absolute_extreme=mean_absolute_extreme,
        )
