#!/usr/bin/env python3
"""Construye la comparación final de bootstrap, CVAE, flow y GAN."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _bootstrap import PROJECT_ROOT

from crypto_generative.comparison import (
    DownstreamComparisonBuilder,
    FinalComparisonBuilder,
)
from crypto_generative.data.artifacts import (
    relative_or_absolute,
    sha256_file,
    write_checksums,
    write_csv_atomic,
    write_json_atomic,
)
from crypto_generative.data.stress import ProjectScenarioLoader
from crypto_generative.evaluation import (
    CrossAssetDependenceConfig,
    DiversityMemorizationConfig,
    RiskMetricsConfig,
    TemporalDependenceConfig,
    TrajectoryEvaluator,
    TrajectoryMetricsConfig,
)
from crypto_generative.models import (
    ConditionalMultivariateBlockBootstrap,
    frozen_conditional_bootstrap_config,
)


NORMALIZED = (
    PROJECT_ROOT
    / "data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz"
)
SPLIT = PROJECT_ROOT / "data/splits/binance/btc_eth_6h_c240_t120_purged_split.npz"
SPLIT_INDEX = (
    PROJECT_ROOT
    / "data/splits/binance/btc_eth_6h_c240_t120_purged_split_index.csv"
)
PANEL = PROJECT_ROOT / "data/processed/binance/btc_eth_6h_panel.csv"
MODEL_METADATA = {
    "cvae": PROJECT_ROOT / "outputs/cvae_best/metadata.json",
    "normalizing_flow": PROJECT_ROOT / "outputs/normalizing_flow_best/metadata.json",
    "conditional_gan": PROJECT_ROOT / "outputs/conditional_gan_best/metadata.json",
}
TEST_RISK = PROJECT_ROOT / "outputs/portfolio_stress_notebook/risk_comparison.csv"
LATEST_RISK = PROJECT_ROOT / "outputs/latest_market_scenarios/risk_comparison.csv"
LATEST_METADATA = PROJECT_ROOT / "outputs/latest_market_scenarios/metadata.json"
DOWNSTREAM = PROJECT_ROOT / "outputs/downstream_common/comparison.csv"
BASELINE_OUTPUT = PROJECT_ROOT / "outputs/block_bootstrap_best/metadata.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/final_comparison"
SCENARIOS_PER_CONDITION = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-output", type=Path, default=BASELINE_OUTPUT)
    return parser


def evaluate_bootstrap(path: Path) -> Mapping[str, Any]:
    loader = ProjectScenarioLoader(NORMALIZED, SPLIT, SPLIT_INDEX, PANEL)
    train = loader.load_split("train")
    test = loader.load_split("test")
    train_conditions = loader.load_normalized_conditions(train.sample_ids)
    test_conditions = loader.load_normalized_conditions(test.sample_ids)
    bootstrap_config = frozen_conditional_bootstrap_config(random_state=5_042)
    bootstrap = ConditionalMultivariateBlockBootstrap(bootstrap_config).fit(
        train.log_returns, train_conditions
    )
    repeated_conditions = np.repeat(
        test_conditions, SCENARIOS_PER_CONDITION, axis=0
    )
    candidate = bootstrap.sample(
        n_scenarios=len(repeated_conditions),
        cond=repeated_conditions,
        horizon_steps=120,
    )
    conditional_candidate = candidate.reshape(
        len(test.log_returns), SCENARIOS_PER_CONDITION, 120, 2
    )

    evaluator = TrajectoryEvaluator(assets=("BTC", "ETH"))
    evaluation = {
        "marginal": evaluator.evaluate_marginals(
            test.log_returns, candidate
        ).to_dict(),
        "temporal": evaluator.evaluate_temporal_dependence(
            test.log_returns,
            candidate,
            config=TemporalDependenceConfig(
                max_lag=20,
                volatility_window=20,
                high_volatility_quantile=0.90,
                extreme_quantile=0.99,
                extreme_clustering_window=4,
            ),
        ).to_dict(),
        "cross_asset": evaluator.evaluate_cross_asset_dependence(
            test.log_returns,
            candidate,
            config=CrossAssetDependenceConfig(
                rolling_window=20,
                stress_quantile=0.90,
                joint_drop_quantile=0.05,
                lower_tail_quantile=0.05,
            ),
        ).to_dict(),
        "trajectory": evaluator.evaluate_trajectories(
            test.log_returns,
            candidate,
            config=TrajectoryMetricsConfig(periods_per_year=4 * 365),
        ).to_dict(),
        "risk": evaluator.evaluate_risk(
            test.log_returns,
            conditional_candidate,
            config=RiskMetricsConfig(
                confidence_levels=(0.95, 0.99),
                portfolio_weights=(0.60, 0.40),
                portfolio_name="portfolio_60_40",
                es_stability_repetitions=100,
                es_stability_sample_size=1_000,
                random_state=42,
            ),
        ).to_dict(),
        "diversity_and_memorization": evaluator.evaluate_diversity_and_memorization(
            test.log_returns,
            candidate,
            training_paths=train.log_returns,
            config=DiversityMemorizationConfig(
                max_paths_per_set=2_000,
                projection_dimensions=24,
                neighbor_candidates=8,
                near_memorization_quantile=0.01,
                coverage_radius_quantile=0.95,
                discriminator_repetitions=5,
                random_state=42,
            ),
        ).to_dict(),
    }
    metadata = {
        "model": "conditional_moving_block_bootstrap_multivariate",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "selected_config": {
            "block_length_steps": bootstrap_config.block_length,
            "block_length_days": bootstrap_config.block_length * 6 / 24,
            "n_neighbors": bootstrap_config.n_neighbors,
            "horizon_steps": 120,
            "random_state": 5_042,
            "scenarios_per_condition": SCENARIOS_PER_CONDITION,
            "scenario_count": len(candidate),
        },
        "selection": (
            "Longitud y número de vecinos seleccionados exclusivamente con validación"
        ),
        "train_samples": len(train.log_returns),
        "test_samples": len(test.log_returns),
        "evaluation": evaluation,
    }
    write_json_atomic(metadata, path)
    return metadata


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_risk_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "scenario_count": int,
        "confidence_level": float,
        "value_at_risk_fraction": float,
        "expected_shortfall_fraction": float,
        "value_at_risk_amount": float,
        "expected_shortfall_amount": float,
    }
    return [
        {
            key: numeric[key](value) if key in numeric else value
            for key, value in row.items()
        }
        for row in rows
    ]


def read_downstream_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {"real_train", "synthetic_train", "total_train", "best_epoch"}
    text_fields = {"model", "dataset"}
    converted = [
        {
            key: (
                value
                if key in text_fields
                else int(value)
                if key in integer_fields
                else float(value)
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    models = {row["model"] for row in converted}
    if len(converted) != 16 or any(
        sum(row["model"] == model for row in converted) != 4 for model in models
    ):
        raise ValueError("La comparación downstream debe contener 4 modelos x 4 ratios")
    return converted


def build_test_risk_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included = {
        "historical_test_distribution",
        "block_bootstrap",
        "cvae",
        "normalizing_flow",
        "conditional_gan",
    }
    filtered = [row for row in rows if row["scenario_set"] in included]
    reference = {
        row["confidence_level"]: row
        for row in filtered
        if row["scenario_set"] == "historical_test_distribution"
    }
    result = []
    for row in filtered:
        observed = reference[row["confidence_level"]]
        result.append(
            {
                **row,
                "scope": "aggregated_test_distribution",
                "var_absolute_error_vs_historical": abs(
                    row["value_at_risk_fraction"]
                    - observed["value_at_risk_fraction"]
                ),
                "es_absolute_error_vs_historical": abs(
                    row["expected_shortfall_fraction"]
                    - observed["expected_shortfall_fraction"]
                ),
            }
        )
    return result


def build_latest_comparison(
    latest_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    test_lookup = {
        (row["scenario_set"], row["confidence_level"]): row
        for row in test_rows
        if row["scenario_set"]
        in {"cvae", "normalizing_flow", "conditional_gan"}
    }
    return [
        {
            **row,
            "scope": "latest_available_condition",
            "var_change_vs_aggregated_test": (
                row["value_at_risk_fraction"]
                - test_lookup[(row["scenario_set"], row["confidence_level"])][
                    "value_at_risk_fraction"
                ]
            ),
            "es_change_vs_aggregated_test": (
                row["expected_shortfall_fraction"]
                - test_lookup[(row["scenario_set"], row["confidence_level"])][
                    "expected_shortfall_fraction"
                ]
            ),
        }
        for row in latest_rows
    ]


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_bootstrap(args.baseline_output)
    metadata_by_model = {
        "block_bootstrap": baseline,
        **{name: read_json(path) for name, path in MODEL_METADATA.items()},
    }
    builder = FinalComparisonBuilder(metadata_by_model)
    quality = builder.quality_records()
    rankings = builder.ranking_records(quality)
    test_risk = build_test_risk_comparison(read_risk_rows(TEST_RISK))
    latest_risk = build_latest_comparison(read_risk_rows(LATEST_RISK), test_risk)
    latest_metadata = read_json(LATEST_METADATA)
    downstream = read_downstream_rows(DOWNSTREAM)
    downstream_selection = DownstreamComparisonBuilder(
        downstream
    ).selection_records()

    quality_path = args.output_dir / "model_quality.csv"
    rankings_path = args.output_dir / "rankings_by_dimension.csv"
    test_path = args.output_dir / "test_portfolio_risk.csv"
    latest_path = args.output_dir / "latest_condition_risk.csv"
    downstream_path = args.output_dir / "downstream_comparison.csv"
    downstream_selection_path = (
        args.output_dir / "downstream_selected_by_validation.csv"
    )
    report_path = args.output_dir / "final_comparison.json"
    write_csv_atomic(quality, tuple(quality[0]), quality_path)
    write_csv_atomic(rankings, tuple(rankings[0]), rankings_path)
    write_csv_atomic(test_risk, tuple(test_risk[0]), test_path)
    write_csv_atomic(latest_risk, tuple(latest_risk[0]), latest_path)
    write_csv_atomic(downstream, tuple(downstream[0]), downstream_path)
    write_csv_atomic(
        downstream_selection,
        tuple(downstream_selection[0]),
        downstream_selection_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "methodology": {
            "models": list(metadata_by_model),
            "test_scope": (
                "Evaluación fuera de muestra agregada; los escenarios generativos "
                "conservan 20 draws por condición y las ventanas se solapan."
            ),
            "latest_scope": (
                "100.000 draws por generador bajo una sola condición de mercado; "
                "no es un backtest ni añade historias económicas independientes."
            ),
            "downstream_scope": (
                "Misma MLP para 0, +25, +50 y +100 % de sintéticos; validación "
                "y test permanecen reales. La proporción defendible se selecciona "
                "por validación, no por el mejor resultado descriptivo de test."
            ),
            "ranking_policy": (
                "Se informa un ganador por métrica. No se calcula un score global "
                "porque requeriría ponderaciones arbitrarias entre objetivos."
            ),
        },
        "latest_condition": latest_metadata["condition"],
        "quality": quality,
        "rankings_by_dimension": rankings,
        "test_portfolio_risk": test_risk,
        "latest_condition_risk": latest_risk,
        "downstream_comparison": downstream,
        "downstream_selected_by_validation": downstream_selection,
        "sources": {
            "baseline_metadata": relative_or_absolute(
                args.baseline_output, PROJECT_ROOT
            ),
            **{
                f"{name}_metadata": relative_or_absolute(path, PROJECT_ROOT)
                for name, path in MODEL_METADATA.items()
            },
            "test_risk": relative_or_absolute(TEST_RISK, PROJECT_ROOT),
            "latest_risk": relative_or_absolute(LATEST_RISK, PROJECT_ROOT),
            "downstream": relative_or_absolute(DOWNSTREAM, PROJECT_ROOT),
            "downstream_sha256": sha256_file(DOWNSTREAM),
        },
    }
    write_json_atomic(report, report_path)
    checksums_path = args.output_dir / "SHA256SUMS"
    write_checksums(
        {
            quality_path.name: quality_path,
            rankings_path.name: rankings_path,
            test_path.name: test_path,
            latest_path.name: latest_path,
            downstream_path.name: downstream_path,
            downstream_selection_path.name: downstream_selection_path,
            report_path.name: report_path,
        },
        checksums_path,
    )
    print(f"Comparación final -> {args.output_dir}")
    print(f"Bootstrap estructurado -> {args.baseline_output}")
    print(f"Dimensiones comparadas: {len(rankings)}; score global: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
