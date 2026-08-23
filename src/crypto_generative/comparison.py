"""Extracción transparente de métricas para la comparación final."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ComparisonMetric:
    name: str
    label: str
    lower_is_better: bool = True


COMPARISON_METRICS = (
    ComparisonMetric("marginal_normalized_w1_mean", "Fidelidad marginal"),
    ComparisonMetric("absolute_return_acf_rmse_mean", "Persistencia de volatilidad"),
    ComparisonMetric("stress_correlation_absolute_error", "Correlación en estrés"),
    ComparisonMetric("lower_tail_dependence_absolute_error", "Dependencia de cola"),
    ComparisonMetric(
        "final_return_normalized_w1_mean", "Distribución del retorno final"
    ),
    ComparisonMetric(
        "realized_volatility_normalized_w1_mean", "Volatilidad realizada"
    ),
    ComparisonMetric("maximum_drawdown_normalized_w1_mean", "Drawdown"),
    ComparisonMetric("portfolio_coverage_error_95", "Cobertura VaR 95 %"),
    ComparisonMetric("portfolio_coverage_error_99", "Cobertura VaR 99 %"),
    ComparisonMetric(
        "reference_coverage_fraction", "Cobertura de la referencia", False
    ),
    ComparisonMetric("regime_total_variation_distance", "Cobertura de regímenes"),
    ComparisonMetric(
        "discriminator_distance_to_random", "Indistinguibilidad real/sintético"
    ),
)


class FinalComparisonBuilder:
    """Convierte metadatos homogéneos del evaluador en tablas comparables."""

    def __init__(self, model_metadata: Mapping[str, Mapping[str, Any]]) -> None:
        if not model_metadata:
            raise ValueError("Se necesita al menos un modelo")
        self.model_metadata = dict(model_metadata)

    def quality_records(self) -> list[dict[str, Any]]:
        return [
            self._quality_record(model, metadata)
            for model, metadata in self.model_metadata.items()
        ]

    def ranking_records(
        self, quality_records: Sequence[Mapping[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        quality = list(quality_records or self.quality_records())
        rankings = []
        for metric in COMPARISON_METRICS:
            ordered = sorted(
                quality,
                key=lambda row: float(row[metric.name]),
                reverse=not metric.lower_is_better,
            )
            rankings.append(
                {
                    "dimension": metric.label,
                    "metric": metric.name,
                    "lower_is_better": metric.lower_is_better,
                    "winner": ordered[0]["model"],
                    "winner_value": ordered[0][metric.name],
                    "ranking": " > ".join(str(row["model"]) for row in ordered),
                }
            )
        return rankings

    @staticmethod
    def _quality_record(model: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        evaluation = metadata["evaluation"]
        assets = tuple(evaluation["marginal"]["assets"])
        marginal = evaluation["marginal"]["by_asset"]
        temporal = evaluation["temporal"]["by_asset"]
        trajectory = evaluation["trajectory"]["by_asset"]
        cross_errors = evaluation["cross_asset"]["errors"]
        portfolio_risk = evaluation["risk"]["by_target"]["portfolio_60_40"]
        diversity = evaluation["diversity_and_memorization"]

        def asset_mean(values: Mapping[str, Any], field: str) -> float:
            result = [float(values[asset][field]) for asset in assets]
            return float(np.mean(result))

        def trajectory_mean(metric: str) -> float:
            return float(
                np.mean(
                    [
                        trajectory[asset][metric]["normalized_wasserstein_1"]
                        for asset in assets
                    ]
                )
            )

        discriminator_accuracy = float(diversity["discriminator_accuracy_mean"])
        return {
            "model": model,
            "candidate_paths": int(diversity["total_candidate_paths"]),
            "marginal_normalized_w1_mean": asset_mean(
                marginal, "normalized_wasserstein_1"
            ),
            "return_acf_rmse_mean": asset_mean(temporal, "return_acf_rmse"),
            "absolute_return_acf_rmse_mean": asset_mean(
                temporal, "absolute_return_acf_rmse"
            ),
            "squared_return_acf_rmse_mean": asset_mean(
                temporal, "squared_return_acf_rmse"
            ),
            "contemporaneous_correlation_absolute_error": float(
                cross_errors["contemporaneous_correlation_absolute_error"]
            ),
            "stress_correlation_absolute_error": float(
                cross_errors["stress_correlation_absolute_error"]
            ),
            "lower_tail_dependence_absolute_error": float(
                cross_errors["lower_tail_dependence_absolute_error"]
            ),
            "final_return_normalized_w1_mean": trajectory_mean(
                "final_cumulative_return"
            ),
            "realized_volatility_normalized_w1_mean": trajectory_mean(
                "realized_volatility"
            ),
            "maximum_drawdown_normalized_w1_mean": trajectory_mean(
                "maximum_drawdown"
            ),
            "portfolio_coverage_error_95": float(
                portfolio_risk["levels"]["0.95"]["coverage_absolute_error"]
            ),
            "portfolio_coverage_error_99": float(
                portfolio_risk["levels"]["0.99"]["coverage_absolute_error"]
            ),
            "portfolio_var_absolute_error_95": float(
                portfolio_risk["levels"]["0.95"][
                    "unconditional_var_absolute_error"
                ]
            ),
            "portfolio_var_absolute_error_99": float(
                portfolio_risk["levels"]["0.99"][
                    "unconditional_var_absolute_error"
                ]
            ),
            "candidate_unique_fraction": float(
                diversity["candidate_unique_fraction"]
            ),
            "near_memorization_fraction": float(
                diversity["near_memorization_fraction"]
            ),
            "reference_coverage_fraction": float(
                diversity["reference_coverage_fraction"]
            ),
            "regime_total_variation_distance": float(
                diversity["regime_total_variation_distance"]
            ),
            "discriminator_accuracy_mean": discriminator_accuracy,
            "discriminator_distance_to_random": abs(
                discriminator_accuracy - 0.5
            ),
        }


class DownstreamComparisonBuilder:
    """Resume mezclas downstream sin seleccionar retrospectivamente con test."""

    MODEL_ORDER = (
        "block_bootstrap",
        "cvae",
        "normalizing_flow",
        "conditional_gan",
    )

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]
        for model in self.MODEL_ORDER:
            model_rows = [row for row in self.rows if row["model"] == model]
            ratios = {float(row["additional_synthetic_ratio"]) for row in model_rows}
            if ratios != {0.0, 0.25, 0.5, 1.0}:
                raise ValueError(
                    f"{model} debe contener ratios 0, 0.25, 0.5 y 1.0"
                )

    def selection_records(self) -> list[dict[str, Any]]:
        records = []
        for model in self.MODEL_ORDER:
            model_rows = [row for row in self.rows if row["model"] == model]
            selected = min(
                model_rows,
                key=lambda row: (
                    float(row["validation_mae"]),
                    float(row["additional_synthetic_ratio"]),
                ),
            )
            descriptive_best = min(
                model_rows, key=lambda row: float(row["test_mae"])
            )
            real_only = next(
                row
                for row in model_rows
                if float(row["additional_synthetic_ratio"]) == 0
            )
            records.append(
                {
                    "model": model,
                    "validation_selected_dataset": selected["dataset"],
                    "validation_selected_ratio": selected[
                        "additional_synthetic_ratio"
                    ],
                    "validation_selected_mae": selected["validation_mae"],
                    "validation_selected_test_mae": selected["test_mae"],
                    "validation_selected_test_r2": selected["test_r2"],
                    "test_descriptive_best_dataset": descriptive_best["dataset"],
                    "test_descriptive_best_mae": descriptive_best["test_mae"],
                    "test_descriptive_improvement_vs_real_only": (
                        float(real_only["test_mae"])
                        - float(descriptive_best["test_mae"])
                    )
                    / float(real_only["test_mae"]),
                }
            )
        return records
