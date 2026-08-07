#!/usr/bin/env python3
"""Evaluate Conditional RealNVP with the project's shared evaluator.

Descriptive families consume de-normalized 3D paths:
    [paths, time, assets]

Conditional risk consumes multiple draws per validation condition:
    [conditions, draws, time, assets]

The test split is never used by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _bootstrap import PROJECT_ROOT as ROOT  # noqa: F401
from crypto_generative.evaluation import (
    CrossAssetDependenceConfig,
    DiversityMemorizationConfig,
    RiskMetricsConfig,
    TemporalDependenceConfig,
    TrajectoryEvaluator,
    TrajectoryMetricsConfig,
)
from crypto_generative.models import ConditionalFlowGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-path",
        type=Path,
        default=ROOT / "data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=ROOT / "data/splits/binance/btc_eth_6h_c240_t120_purged_split.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Flow checkpoint. When omitted, search normalizing_flow_60epochs and "
            "normalizing_flow in that order."
        ),
    )
    parser.add_argument(
        "--validation-paths",
        type=Path,
        default=None,
        help=(
            "Existing one-draw-per-condition NPZ for descriptive metrics. By default, "
            "use validation_synthetic_paths.npz next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/normalizing_flow_common_evaluator",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--risk-conditions",
        type=int,
        default=100,
        help="Number of validation conditions for conditional risk (0 means all).",
    )
    parser.add_argument(
        "--risk-draws",
        type=int,
        default=100,
        help="Synthetic draws generated per selected validation condition.",
    )
    parser.add_argument(
        "--generation-condition-batch-size",
        type=int,
        default=8,
        help="Conditions generated together; total paths per batch = this * risk_draws.",
    )
    parser.add_argument(
        "--save-risk-paths",
        action="store_true",
        help="Persist the potentially large 4D conditional risk tensor.",
    )
    parser.add_argument(
        "--skip-conditional-risk",
        action="store_true",
        help="Run only 3D descriptive and diversity families.",
    )
    return parser.parse_args()


def resolve_checkpoint(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    candidates = (
        ROOT / "results/normalizing_flow_60epochs/conditional_realnvp_best.pt",
        ROOT / "results/normalizing_flow/conditional_realnvp_best.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No flow checkpoint found. Pass --checkpoint explicitly. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_common_data(normalized_path: Path, split_path: Path) -> dict[str, np.ndarray]:
    with np.load(normalized_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(split_path, allow_pickle=False) as split:
        arrays["train_ids"] = split["train_sample_ids"]
        arrays["validation_ids"] = split["validation_sample_ids"]
    return arrays


def load_descriptive_candidate(
    path: Path,
    reference_returns: np.ndarray,
    expected_validation_ids: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Descriptive synthetic path file not found: {path}. "
            "Run train_conditional_flow.py or pass --validation-paths."
        )
    with np.load(path, allow_pickle=False) as data:
        candidate = np.asarray(data["generated_returns"], dtype=np.float64)
        stored_reference = np.asarray(data["real_returns"], dtype=np.float64)
        sample_ids = np.asarray(data["sample_ids"])

    if candidate.ndim != 3:
        raise ValueError(f"generated_returns must be 3D, got {candidate.shape}")
    if candidate.shape[1:] != reference_returns.shape[1:]:
        raise ValueError("Candidate and validation reference have incompatible shapes")
    if not np.array_equal(sample_ids, expected_validation_ids[: len(sample_ids)]):
        raise ValueError("Saved sample_ids do not match the frozen validation split")
    if not np.allclose(stored_reference, reference_returns[: len(sample_ids)], atol=1e-6):
        raise ValueError("Saved real_returns do not match inverse-normalized validation data")
    return candidate


def generate_conditional_draws(
    flow: ConditionalFlowGenerator,
    conditions: np.ndarray,
    draws: int,
    return_mean: np.ndarray,
    return_scale: np.ndarray,
    condition_batch_size: int,
    seed: int,
) -> np.ndarray:
    if draws < 2:
        raise ValueError("risk_draws must be at least 2")
    if condition_batch_size < 1:
        raise ValueError("generation_condition_batch_size must be positive")

    n_conditions = len(conditions)
    output = np.empty(
        (
            n_conditions,
            draws,
            flow.config.trajectory_length,
            flow.config.n_assets,
        ),
        dtype=np.float32,
    )

    for start in range(0, n_conditions, condition_batch_size):
        stop = min(start + condition_batch_size, n_conditions)
        condition_batch = conditions[start:stop]
        repeated = np.repeat(condition_batch, draws, axis=0)
        generated_normalized = flow.sample(
            n=len(repeated),
            cond=repeated,
            seed=seed + start,
        )
        generated = generated_normalized * return_scale + return_mean
        output[start:stop] = generated.reshape(
            stop - start,
            draws,
            flow.config.trajectory_length,
            flow.config.n_assets,
        )
        print(
            f"Conditional risk generation: {stop:4d}/{n_conditions} conditions "
            f"({draws} draws each)"
        )
    return output


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_safe(record.get(key)) for key in fieldnames})


def project_relative(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint)
    validation_paths = (
        args.validation_paths.resolve()
        if args.validation_paths is not None
        else checkpoint.parent / "validation_synthetic_paths.npz"
    )

    arrays = load_common_data(args.normalized_path, args.split_path)
    train_ids = arrays["train_ids"]
    validation_ids = arrays["validation_ids"]
    target_normalized = arrays["target_returns"].astype(np.float32)
    conditions_normalized = arrays["condition_features"].astype(np.float32)
    return_mean = arrays["return_mean"].astype(np.float32)
    return_scale = arrays["return_scale"].astype(np.float32)
    assets = tuple(str(asset) for asset in arrays["assets"])

    training_returns = (
        target_normalized[train_ids] * return_scale + return_mean
    ).astype(np.float64)
    reference_returns = (
        target_normalized[validation_ids] * return_scale + return_mean
    ).astype(np.float64)
    candidate_returns = load_descriptive_candidate(
        validation_paths,
        reference_returns,
        validation_ids,
    )

    print("\nCOMMON EVALUATOR — CONDITIONAL REALNVP")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint}")
    print(f"Reference validation paths: {reference_returns.shape}")
    print(f"Descriptive candidate paths: {candidate_returns.shape}")
    print("Inputs are inverse-normalized six-hour log returns.")

    evaluator = TrajectoryEvaluator(assets=assets)
    reports: dict[str, Any] = {}

    marginal = evaluator.evaluate_marginals(reference_returns, candidate_returns)
    reports["marginal"] = marginal

    temporal = evaluator.evaluate_temporal_dependence(
        reference_returns,
        candidate_returns,
        config=TemporalDependenceConfig(
            max_lag=20,
            volatility_window=20,
            high_volatility_quantile=0.90,
            extreme_quantile=0.99,
            extreme_clustering_window=4,
        ),
    )
    reports["temporal"] = temporal

    cross_asset = evaluator.evaluate_cross_asset_dependence(
        reference_returns,
        candidate_returns,
        config=CrossAssetDependenceConfig(
            rolling_window=20,
            stress_quantile=0.90,
            joint_drop_quantile=0.05,
            lower_tail_quantile=0.05,
        ),
    )
    reports["cross_asset"] = cross_asset

    trajectory = evaluator.evaluate_trajectories(
        reference_returns,
        candidate_returns,
        config=TrajectoryMetricsConfig(periods_per_year=4 * 365),
    )
    reports["trajectory"] = trajectory

    diversity = evaluator.evaluate_diversity_and_memorization(
        reference_returns,
        candidate_returns,
        training_paths=training_returns,
        config=DiversityMemorizationConfig(
            max_paths_per_set=2_000,
            projection_dimensions=24,
            neighbor_candidates=8,
            near_memorization_quantile=0.01,
            coverage_radius_quantile=0.95,
            discriminator_repetitions=5,
            random_state=args.seed,
        ),
    )
    reports["diversity"] = diversity

    risk_condition_ids = np.array([], dtype=np.int64)
    conditional_draws = None
    if not args.skip_conditional_risk:
        n_risk_conditions = (
            len(validation_ids)
            if args.risk_conditions == 0
            else min(args.risk_conditions, len(validation_ids))
        )
        if n_risk_conditions < 2:
            raise ValueError("At least two risk conditions are required")
        rng = np.random.default_rng(args.seed)
        positions = np.sort(
            rng.choice(len(validation_ids), size=n_risk_conditions, replace=False)
        )
        risk_condition_ids = validation_ids[positions]
        reference_risk = reference_returns[positions]
        flow = ConditionalFlowGenerator.load(checkpoint, device=args.device)
        conditional_draws = generate_conditional_draws(
            flow=flow,
            conditions=conditions_normalized[risk_condition_ids],
            draws=args.risk_draws,
            return_mean=return_mean,
            return_scale=return_scale,
            condition_batch_size=args.generation_condition_batch_size,
            seed=args.seed + 10_000,
        )
        risk = evaluator.evaluate_risk(
            reference_paths=reference_risk,
            candidate_paths=conditional_draws,
            config=RiskMetricsConfig(
                confidence_levels=(0.95, 0.99),
                portfolio_weights=(0.60, 0.40),
                portfolio_name="portfolio_60_40",
                es_stability_repetitions=100,
                es_stability_sample_size=1_000,
                random_state=args.seed,
            ),
        )
        reports["risk"] = risk
        if args.save_risk_paths:
            np.savez_compressed(
                args.output_dir / "conditional_risk_paths.npz",
                validation_sample_ids=risk_condition_ids,
                reference_returns=reference_risk.astype(np.float32),
                generated_returns=conditional_draws,
                assets=np.asarray(assets),
                risk_draws=np.asarray(args.risk_draws),
            )

    for name, report in reports.items():
        save_json(args.output_dir / f"{name}_report.json", report.to_dict())
        save_records(args.output_dir / f"{name}_table.csv", report.to_records())

    metadata = {
        "scope": "validation_only",
        "checkpoint": project_relative(checkpoint),
        "validation_paths": project_relative(validation_paths),
        "assets": list(assets),
        "reference_shape": list(reference_returns.shape),
        "candidate_shape": list(candidate_returns.shape),
        "training_shape": list(training_returns.shape),
        "conditional_risk_shape": (
            list(conditional_draws.shape) if conditional_draws is not None else None
        ),
        "risk_validation_sample_ids": risk_condition_ids.tolist(),
        "test_split_used": False,
    }
    save_json(args.output_dir / "evaluation_metadata.json", metadata)

    print("\nMARGINAL")
    for row in marginal.to_records():
        print(
            f"{row['asset']}: normalized W1={row['normalized_wasserstein_1']:.4f}, "
            f"real std={row['reference_standard_deviation']:.6f}, "
            f"synthetic std={row['candidate_standard_deviation']:.6f}"
        )

    cross_row = cross_asset.to_records()[0]
    print("\nCROSS-ASSET")
    print(
        "Contemporaneous correlation: "
        f"real={cross_row['reference_contemporaneous_correlation']:.4f}, "
        f"synthetic={cross_row['candidate_contemporaneous_correlation']:.4f}"
    )
    print(
        "Joint-drop probability: "
        f"real={cross_row['reference_joint_drop_probability']:.4f}, "
        f"synthetic={cross_row['candidate_joint_drop_probability']:.4f}"
    )

    print("\nTRAJECTORY")
    trajectory_rows = trajectory.to_records()
    for row in trajectory_rows:
        if row["metric"] in ("final_cumulative_return", "maximum_drawdown"):
            print(
                f"{row['asset']} {row['metric']}: "
                f"real mean={row['reference_mean']:.4f}, "
                f"synthetic mean={row['candidate_mean']:.4f}, "
                f"normalized W1={row['normalized_wasserstein_1']:.4f}"
            )

    diversity_row = diversity.to_records()[0]
    print("\nDIVERSITY / MEMORIZATION")
    print(f"Unique fraction: {diversity_row['candidate_unique_fraction']:.4f}")
    print(
        "Near-memorization fraction: "
        f"{diversity_row.get('near_memorization_fraction')}"
    )
    print(
        "Real-vs-synthetic discriminator accuracy: "
        f"{diversity_row['discriminator_accuracy_mean']:.4f}"
    )

    if "risk" in reports:
        print("\nCONDITIONAL RISK")
        for row in reports["risk"].to_records():
            if row["target"] != "portfolio_60_40":
                continue
            print(
                f"Portfolio {row['confidence_level']:.0%}: "
                f"exception rate={row['exception_rate']:.4f} "
                f"(expected={row['expected_exception_rate']:.4f}), "
                f"coverage error={row['coverage_absolute_error']:.4f}, "
                f"mean forecast VaR={row['mean_forecast_var']:.4f}, "
                f"mean forecast ES={row['mean_forecast_es']:.4f}"
            )

    print(f"\nSaved shared-evaluator reports in: {args.output_dir}")


if __name__ == "__main__":
    main()
