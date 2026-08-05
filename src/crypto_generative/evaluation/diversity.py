"""Metricas de diversidad, cobertura y memorizacion de trayectorias."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .trajectory import compute_asset_path_metrics


@dataclass(frozen=True)
class DiversityMemorizationConfig:
    """Configuracion reproducible y acotada para búsquedas de vecinos."""

    max_paths_per_set: int = 2_000
    projection_dimensions: int = 24
    neighbor_candidates: int = 8
    near_memorization_quantile: float = 0.01
    coverage_radius_quantile: float = 0.95
    rounding_decimals: int = 12
    discriminator_repetitions: int = 5
    discriminator_test_fraction: float = 0.30
    discriminator_iterations: int = 500
    discriminator_learning_rate: float = 0.10
    discriminator_l2: float = 1e-3
    random_state: int = 42

    def validate(self) -> None:
        if self.max_paths_per_set < 2:
            raise ValueError("max_paths_per_set debe ser al menos 2")
        if self.projection_dimensions < 2:
            raise ValueError("projection_dimensions debe ser al menos 2")
        if self.neighbor_candidates < 1:
            raise ValueError("neighbor_candidates debe ser positivo")
        if not 0 < self.near_memorization_quantile < 1:
            raise ValueError("near_memorization_quantile debe estar entre 0 y 1")
        if not 0 < self.coverage_radius_quantile < 1:
            raise ValueError("coverage_radius_quantile debe estar entre 0 y 1")
        if not 0 <= self.rounding_decimals <= 15:
            raise ValueError("rounding_decimals debe estar entre 0 y 15")
        if self.discriminator_repetitions < 2:
            raise ValueError("discriminator_repetitions debe ser al menos 2")
        if not 0 < self.discriminator_test_fraction < 1:
            raise ValueError("discriminator_test_fraction debe estar entre 0 y 1")
        if self.discriminator_iterations < 1:
            raise ValueError("discriminator_iterations debe ser positivo")
        if self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator_learning_rate debe ser positivo")
        if self.discriminator_l2 < 0:
            raise ValueError("discriminator_l2 no puede ser negativo")


@dataclass(frozen=True)
class DistanceSummary:
    mean: float
    standard_deviation: float
    q01: float
    q05: float
    q50: float
    q95: float


@dataclass(frozen=True)
class DiversityMemorizationEvaluation:
    """Resultado global; las distancias son RMSE en retornos estandarizados."""

    assets: Tuple[str, ...]
    config: DiversityMemorizationConfig
    total_reference_paths: int
    total_candidate_paths: int
    total_training_paths: Optional[int]
    evaluated_reference_paths: int
    evaluated_candidate_paths: int
    evaluated_training_paths: Optional[int]
    return_scale: Mapping[str, float]
    candidate_unique_fraction: float
    candidate_redundant_fraction: float
    candidate_nearest_neighbor_distance: DistanceSummary
    near_duplicate_threshold: float
    candidate_near_duplicate_fraction: float
    exact_training_match_fraction: Optional[float]
    candidate_to_training_distance: Optional[DistanceSummary]
    near_memorization_threshold: Optional[float]
    near_memorization_fraction: Optional[float]
    reference_coverage_radius: float
    reference_coverage_fraction: float
    reference_to_candidate_distance: DistanceSummary
    reference_regime_proportions: Mapping[str, float]
    candidate_regime_proportions: Mapping[str, float]
    regime_total_variation_distance: float
    discriminator_accuracy_mean: float
    discriminator_accuracy_standard_deviation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_records(self) -> Sequence[Dict[str, Any]]:
        record: Dict[str, Any] = {
            "total_reference_paths": self.total_reference_paths,
            "total_candidate_paths": self.total_candidate_paths,
            "total_training_paths": self.total_training_paths,
            "evaluated_reference_paths": self.evaluated_reference_paths,
            "evaluated_candidate_paths": self.evaluated_candidate_paths,
            "evaluated_training_paths": self.evaluated_training_paths,
            "candidate_unique_fraction": self.candidate_unique_fraction,
            "candidate_redundant_fraction": self.candidate_redundant_fraction,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "candidate_near_duplicate_fraction": (
                self.candidate_near_duplicate_fraction
            ),
            "exact_training_match_fraction": self.exact_training_match_fraction,
            "near_memorization_threshold": self.near_memorization_threshold,
            "near_memorization_fraction": self.near_memorization_fraction,
            "reference_coverage_radius": self.reference_coverage_radius,
            "reference_coverage_fraction": self.reference_coverage_fraction,
            "regime_total_variation_distance": (
                self.regime_total_variation_distance
            ),
            "discriminator_accuracy_mean": self.discriminator_accuracy_mean,
            "discriminator_accuracy_standard_deviation": (
                self.discriminator_accuracy_standard_deviation
            ),
        }
        for prefix, summary in (
            ("candidate_neighbor", self.candidate_nearest_neighbor_distance),
            ("candidate_to_training", self.candidate_to_training_distance),
            ("reference_coverage", self.reference_to_candidate_distance),
        ):
            if summary is None:
                continue
            record.update(
                {
                    f"{prefix}_{field}": getattr(summary, field)
                    for field in (
                        "mean",
                        "standard_deviation",
                        "q01",
                        "q05",
                        "q50",
                        "q95",
                    )
                }
            )
        record.update(
            {
                f"reference_regime_{regime}": proportion
                for regime, proportion in self.reference_regime_proportions.items()
            }
        )
        record.update(
            {
                f"candidate_regime_{regime}": proportion
                for regime, proportion in self.candidate_regime_proportions.items()
            }
        )
        return [record]


def evaluate_diversity_paths(
    reference_paths: NDArray[np.float64],
    candidate_paths: NDArray[np.float64],
    training_paths: Optional[NDArray[np.float64]],
    assets: Sequence[str],
    config: DiversityMemorizationConfig,
) -> DiversityMemorizationEvaluation:
    """Evalúa diversidad del candidato, cobertura real y copia de train."""
    config.validate()
    if len(assets) != 2:
        raise ValueError("La evaluación de diversidad actual requiere exactamente 2 activos")
    rng = np.random.default_rng(config.random_state)
    reference = _sample_paths(reference_paths, config.max_paths_per_set, rng)
    candidate = _sample_paths(candidate_paths, config.max_paths_per_set, rng)
    training = (
        _sample_paths(training_paths, config.max_paths_per_set, rng)
        if training_paths is not None
        else None
    )
    if min(len(reference), len(candidate)) < 2:
        raise ValueError("Se necesitan al menos dos trayectorias reales y candidatas")
    if training is not None and len(training) < 2:
        raise ValueError("Se necesitan al menos dos trayectorias de entrenamiento")

    scale_source = training if training is not None else reference
    return_scale_values = scale_source.std(axis=(0, 1), ddof=0)
    return_scale_values = np.where(return_scale_values > 0, return_scale_values, 1.0)
    reference_flat = (reference / return_scale_values).reshape(len(reference), -1)
    candidate_flat = (candidate / return_scale_values).reshape(len(candidate), -1)
    training_flat = (
        (training / return_scale_values).reshape(len(training), -1)
        if training is not None
        else None
    )

    projection = rng.normal(
        size=(reference_flat.shape[1], config.projection_dimensions)
    ) / np.sqrt(config.projection_dimensions)
    reference_self_distances = _approximate_nearest_distances(
        reference_flat,
        reference_flat,
        projection,
        config.neighbor_candidates,
        exclude_self=True,
    )
    candidate_self_distances = _approximate_nearest_distances(
        candidate_flat,
        candidate_flat,
        projection,
        config.neighbor_candidates,
        exclude_self=True,
    )
    reference_to_candidate_distances = _approximate_nearest_distances(
        reference_flat,
        candidate_flat,
        projection,
        config.neighbor_candidates,
    )

    if training_flat is not None:
        training_self_distances = _approximate_nearest_distances(
            training_flat,
            training_flat,
            projection,
            config.neighbor_candidates,
            exclude_self=True,
        )
        candidate_to_training_distances = _approximate_nearest_distances(
            candidate_flat,
            training_flat,
            projection,
            config.neighbor_candidates,
        )
        near_memorization_threshold = float(
            np.quantile(
                training_self_distances,
                config.near_memorization_quantile,
            )
        )
        exact_training_match_fraction = _exact_match_fraction(
            candidate,
            training_paths,
        )
        near_memorization_fraction = float(
            np.mean(candidate_to_training_distances <= near_memorization_threshold)
        )
        training_distance_summary = _summarize_distances(
            candidate_to_training_distances
        )
        duplicate_threshold_source = training_self_distances
    else:
        near_memorization_threshold = None
        exact_training_match_fraction = None
        near_memorization_fraction = None
        training_distance_summary = None
        duplicate_threshold_source = reference_self_distances

    near_duplicate_threshold = float(
        np.quantile(
            duplicate_threshold_source,
            config.near_memorization_quantile,
        )
    )
    coverage_radius = float(
        np.quantile(reference_self_distances, config.coverage_radius_quantile)
    )
    unique_fraction, redundant_fraction = _duplicate_fractions(
        candidate,
        config.rounding_decimals,
    )
    reference_regimes, candidate_regimes = _regime_proportions(
        reference,
        candidate,
    )
    regime_total_variation = 0.5 * sum(
        abs(reference_regimes[regime] - candidate_regimes[regime])
        for regime in reference_regimes
    )
    discriminator_mean, discriminator_std = _discriminator_accuracy(
        reference,
        candidate,
        config,
    )
    normalized_assets = tuple(str(asset) for asset in assets)

    return DiversityMemorizationEvaluation(
        assets=normalized_assets,
        config=config,
        total_reference_paths=len(reference_paths),
        total_candidate_paths=len(candidate_paths),
        total_training_paths=(
            len(training_paths) if training_paths is not None else None
        ),
        evaluated_reference_paths=len(reference),
        evaluated_candidate_paths=len(candidate),
        evaluated_training_paths=len(training) if training is not None else None,
        return_scale={
            normalized_assets[index]: float(return_scale_values[index])
            for index in range(len(normalized_assets))
        },
        candidate_unique_fraction=unique_fraction,
        candidate_redundant_fraction=redundant_fraction,
        candidate_nearest_neighbor_distance=_summarize_distances(
            candidate_self_distances
        ),
        near_duplicate_threshold=near_duplicate_threshold,
        candidate_near_duplicate_fraction=float(
            np.mean(candidate_self_distances <= near_duplicate_threshold)
        ),
        exact_training_match_fraction=exact_training_match_fraction,
        candidate_to_training_distance=training_distance_summary,
        near_memorization_threshold=near_memorization_threshold,
        near_memorization_fraction=near_memorization_fraction,
        reference_coverage_radius=coverage_radius,
        reference_coverage_fraction=float(
            np.mean(reference_to_candidate_distances <= coverage_radius)
        ),
        reference_to_candidate_distance=_summarize_distances(
            reference_to_candidate_distances
        ),
        reference_regime_proportions=reference_regimes,
        candidate_regime_proportions=candidate_regimes,
        regime_total_variation_distance=float(regime_total_variation),
        discriminator_accuracy_mean=discriminator_mean,
        discriminator_accuracy_standard_deviation=discriminator_std,
    )


def _sample_paths(
    paths: NDArray[np.float64],
    maximum: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if len(paths) <= maximum:
        return paths.copy()
    indices = np.sort(rng.choice(len(paths), size=maximum, replace=False))
    return paths[indices].copy()


def _approximate_nearest_distances(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    projection: NDArray[np.float64],
    neighbor_candidates: int,
    exclude_self: bool = False,
    batch_size: int = 256,
) -> NDArray[np.float64]:
    if exclude_self and len(query) != len(reference):
        raise ValueError("exclude_self requiere matrices alineadas")
    available_neighbors = len(reference) - int(exclude_self)
    if available_neighbors < 1:
        raise ValueError("No hay vecinos suficientes")
    top_k = min(neighbor_candidates, available_neighbors)
    projected_query = query @ projection
    projected_reference = reference @ projection
    reference_norm = np.sum(projected_reference**2, axis=1)
    distances = np.empty(len(query), dtype=np.float64)

    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        query_batch = projected_query[start:stop]
        projected_squared_distance = (
            np.sum(query_batch**2, axis=1)[:, None]
            + reference_norm[None, :]
            - 2 * query_batch @ projected_reference.T
        )
        np.maximum(projected_squared_distance, 0, out=projected_squared_distance)
        if exclude_self:
            local_rows = np.arange(stop - start)
            projected_squared_distance[local_rows, np.arange(start, stop)] = np.inf
        finalists = np.argpartition(
            projected_squared_distance,
            kth=top_k - 1,
            axis=1,
        )[:, :top_k]
        exact_difference = (
            query[start:stop, None, :] - reference[finalists]
        )
        exact_rmse = np.sqrt(np.mean(exact_difference**2, axis=2))
        distances[start:stop] = exact_rmse.min(axis=1)
    return distances


def _duplicate_fractions(
    paths: NDArray[np.float64],
    decimals: int,
) -> Tuple[float, float]:
    rounded = np.round(paths.reshape(len(paths), -1), decimals=decimals)
    unique_count = len(np.unique(rounded, axis=0))
    unique_fraction = unique_count / len(paths)
    return float(unique_fraction), float(1.0 - unique_fraction)


def _exact_match_fraction(
    candidate: NDArray[np.float64],
    training: NDArray[np.float64],
) -> float:
    flattened_training = np.ascontiguousarray(training.reshape(len(training), -1))
    training_rows = {row.tobytes() for row in flattened_training}
    flattened_candidate = np.ascontiguousarray(candidate.reshape(len(candidate), -1))
    return float(
        np.mean([row.tobytes() in training_rows for row in flattened_candidate])
    )


def _regime_proportions(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    ddof = 1 if reference.shape[1] > 1 else 0
    reference_asset_volatility = reference.std(axis=1, ddof=ddof)
    candidate_asset_volatility = candidate.std(axis=1, ddof=ddof)
    scale = np.median(reference_asset_volatility, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    reference_score = np.mean(reference_asset_volatility / scale, axis=1)
    candidate_score = np.mean(candidate_asset_volatility / scale, axis=1)
    thresholds = np.quantile(reference_score, (1 / 3, 2 / 3))
    names = ("low", "medium", "high")

    def proportions(scores: NDArray[np.float64]) -> Mapping[str, float]:
        labels = np.digitize(scores, thresholds, right=True)
        return {
            name: float(np.mean(labels == index))
            for index, name in enumerate(names)
        }

    return proportions(reference_score), proportions(candidate_score)


def _path_features(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    features = []
    for asset_index in range(paths.shape[2]):
        asset_paths = paths[:, :, asset_index]
        metrics = compute_asset_path_metrics(asset_paths, periods_per_year=1)
        features.extend(
            (
                asset_paths.mean(axis=1),
                asset_paths.std(axis=1, ddof=0),
                np.mean(np.abs(asset_paths), axis=1),
                np.quantile(asset_paths, 0.05, axis=1),
                np.quantile(asset_paths, 0.95, axis=1),
                metrics["final_cumulative_return"],
                metrics["maximum_drawdown"],
                metrics["maximum_drawdown_duration_steps"] / paths.shape[1],
                metrics["intrahorizon_maximum_return"],
                metrics["intrahorizon_minimum_return"],
            )
        )
    first = paths[:, :, 0]
    second = paths[:, :, 1]
    first_centered = first - first.mean(axis=1, keepdims=True)
    second_centered = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(first_centered**2, axis=1)
        * np.sum(second_centered**2, axis=1)
    )
    correlation = np.divide(
        np.sum(first_centered * second_centered, axis=1),
        denominator,
        out=np.zeros(len(paths), dtype=np.float64),
        where=denominator > 0,
    )
    features.append(correlation)
    return np.column_stack(features)


def _discriminator_accuracy(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    config: DiversityMemorizationConfig,
) -> Tuple[float, float]:
    reference_features = _path_features(reference)
    candidate_features = _path_features(candidate)
    balanced_size = min(len(reference_features), len(candidate_features))
    accuracies = []

    for repetition in range(config.discriminator_repetitions):
        rng = np.random.default_rng(config.random_state + 10_000 + repetition)
        reference_indices = rng.choice(
            len(reference_features),
            size=balanced_size,
            replace=False,
        )
        candidate_indices = rng.choice(
            len(candidate_features),
            size=balanced_size,
            replace=False,
        )
        split = int(round(balanced_size * (1 - config.discriminator_test_fraction)))
        split = min(max(split, 1), balanced_size - 1)
        reference_selected = reference_features[reference_indices]
        candidate_selected = candidate_features[candidate_indices]
        train_features = np.vstack(
            (reference_selected[:split], candidate_selected[:split])
        )
        train_labels = np.concatenate((np.zeros(split), np.ones(split)))
        test_features = np.vstack(
            (reference_selected[split:], candidate_selected[split:])
        )
        test_labels = np.concatenate(
            (
                np.zeros(balanced_size - split),
                np.ones(balanced_size - split),
            )
        )
        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        train_standardized = (train_features - mean) / scale
        test_standardized = (test_features - mean) / scale
        train_design = np.column_stack((np.ones(len(train_standardized)), train_standardized))
        test_design = np.column_stack((np.ones(len(test_standardized)), test_standardized))
        weights = np.zeros(train_design.shape[1], dtype=np.float64)

        for _ in range(config.discriminator_iterations):
            logits = np.clip(train_design @ weights, -30, 30)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = train_design.T @ (probabilities - train_labels)
            gradient /= len(train_design)
            gradient[1:] += config.discriminator_l2 * weights[1:]
            weights -= config.discriminator_learning_rate * gradient

        predictions = (test_design @ weights >= 0).astype(np.float64)
        reference_accuracy = np.mean(predictions[test_labels == 0] == 0)
        candidate_accuracy = np.mean(predictions[test_labels == 1] == 1)
        balanced_accuracy = 0.5 * (reference_accuracy + candidate_accuracy)
        # Un resultado bajo azar indica un clasificador inestable, no una
        # capacidad útil de distinguir que deba premiarse invirtiendo etiquetas.
        accuracies.append(max(0.5, balanced_accuracy))

    return float(np.mean(accuracies)), float(np.std(accuracies, ddof=1))


def _summarize_distances(distances: NDArray[np.float64]) -> DistanceSummary:
    return DistanceSummary(
        mean=float(distances.mean()),
        standard_deviation=(
            float(distances.std(ddof=1)) if len(distances) > 1 else 0.0
        ),
        q01=float(np.quantile(distances, 0.01)),
        q05=float(np.quantile(distances, 0.05)),
        q50=float(np.quantile(distances, 0.50)),
        q95=float(np.quantile(distances, 0.95)),
    )
