#!/usr/bin/env python3
"""Genera escenarios masivos condicionados al último estado disponible."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gc
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _bootstrap import PROJECT_ROOT

from crypto_generative.data.artifacts import (
    relative_or_absolute,
    sha256_file,
    write_checksums,
    write_csv_atomic,
    write_json_atomic,
)
from crypto_generative.data.condition import ConditionFeatureBuilder
from crypto_generative.generation import (
    MassiveConditionalScenarioGenerator,
    MassiveGenerationConfig,
)
from crypto_generative.portfolio import (
    BuyAndHoldPortfolio,
    PortfolioStressReport,
)


DEFAULT_PANEL = PROJECT_ROOT / "data/processed/binance/btc_eth_6h_panel.csv"
DEFAULT_NORMALIZED = (
    PROJECT_ROOT
    / "data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/latest_market_scenarios"
MODEL_ARTIFACTS = {
    "cvae": PROJECT_ROOT / "outputs/cvae_best/decoder.keras",
    "normalizing_flow": (
        PROJECT_ROOT / "outputs/normalizing_flow_best/conditional_realnvp_best.pt"
    ),
    "conditional_gan": (
        PROJECT_ROOT / "outputs/conditional_gan_best/conditional_gan_best.pt"
    ),
}
MODEL_SEED_OFFSETS = {
    "cvae": 0,
    "normalizing_flow": 1_000_000,
    "conditional_gan": 2_000_000,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--normalized-data", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_ARTIFACTS),
        default=list(MODEL_ARTIFACTS),
    )
    parser.add_argument("--n-scenarios", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_823)
    parser.add_argument("--condition-steps", type=int, default=240)
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo PyTorch: auto, cpu, cuda o mps.",
    )
    return parser


def load_sampler(model_name: str, *, device: str | None) -> tuple[Any, Path]:
    artifact = MODEL_ARTIFACTS[model_name]
    if model_name == "cvae":
        from crypto_generative.models import ConditionalCVAEDecoder

        metadata = artifact.with_name("metadata.json")
        return (
            ConditionalCVAEDecoder.load(artifact, metadata_path=metadata),
            artifact,
        )
    if model_name == "normalizing_flow":
        from crypto_generative.models import ConditionalFlowGenerator

        return ConditionalFlowGenerator.load(artifact, device=device), artifact
    if model_name == "conditional_gan":
        from crypto_generative.models import ConditionalGANGenerator

        return ConditionalGANGenerator.load(artifact, device=device), artifact
    raise ValueError(f"Modelo no soportado: {model_name}")


def progress_printer(model_name: str) -> Callable[[int, int], None]:
    last_bucket = -1

    def report(done: int, total: int) -> None:
        nonlocal last_bucket
        bucket = int(10 * done / total)
        if bucket != last_bucket or done == total:
            print(f"[{model_name}] {done:,}/{total:,} escenarios")
            last_bucket = bucket

    return report


def write_condition_artifact(
    destination: Path,
    *,
    raw: np.ndarray,
    normalized: np.ndarray,
    feature_names: np.ndarray,
    condition_returns: np.ndarray,
    initial_prices: np.ndarray,
    return_mean: np.ndarray,
    return_scale: np.ndarray,
    condition_start_utc: str,
    condition_end_utc: str,
    forecast_start_utc: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            condition_features_raw=raw,
            condition_features_normalized=normalized,
            feature_names=feature_names,
            condition_returns=condition_returns,
            initial_prices=initial_prices,
            assets=np.asarray(["BTC", "ETH"]),
            return_mean=return_mean,
            return_scale=return_scale,
            condition_start_utc=np.asarray(condition_start_utc),
            condition_end_utc=np.asarray(condition_end_utc),
            forecast_start_utc=np.asarray(forecast_start_utc),
        )
    temporary.replace(destination)


def main() -> int:
    args = build_parser().parse_args()
    generation_template = MassiveGenerationConfig(
        scenario_count=args.n_scenarios,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    generation_template.validate()
    for required in (args.panel, args.normalized_data):
        if not required.exists():
            raise FileNotFoundError(required)

    builder = ConditionFeatureBuilder()
    snapshot = builder.build_latest(
        args.panel,
        condition_steps=args.condition_steps,
    )
    with np.load(args.normalized_data, allow_pickle=False) as normalized_data:
        feature_names = normalized_data["feature_names"].copy()
        condition_mean = normalized_data["condition_feature_mean"].astype(np.float64)
        condition_scale = normalized_data["condition_feature_scale"].astype(np.float64)
        return_mean = normalized_data["return_mean"].astype(np.float64)
        return_scale = normalized_data["return_scale"].astype(np.float64)
        assets = tuple(str(asset) for asset in normalized_data["assets"])
    if tuple(str(name) for name in feature_names) != tuple(snapshot.feature_names):
        raise ValueError("Las variables del estado actual no coinciden con el entrenamiento")
    if assets != snapshot.assets:
        raise ValueError("El orden de activos no coincide con el entrenamiento")
    normalized_condition = (
        (snapshot.features - condition_mean) / condition_scale
    ).astype(np.float32)[None, :]
    if not np.isfinite(normalized_condition).all():
        raise ValueError("La normalización del estado actual produjo valores no finitos")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = args.output_dir / "latest_condition.npz"
    write_condition_artifact(
        condition_path,
        raw=snapshot.features,
        normalized=normalized_condition,
        feature_names=feature_names,
        condition_returns=snapshot.condition_returns,
        initial_prices=snapshot.initial_prices,
        return_mean=return_mean,
        return_scale=return_scale,
        condition_start_utc=snapshot.condition_start_utc,
        condition_end_utc=snapshot.condition_end_utc,
        forecast_start_utc=snapshot.forecast_start_utc,
    )

    portfolio = BuyAndHoldPortfolio()
    generator = MassiveConditionalScenarioGenerator(
        return_mean,
        return_scale,
        portfolio=portfolio,
    )
    device = None if args.device == "auto" else args.device
    summaries = {}
    model_records = {}
    scenario_paths = {}
    for model_name in args.models:
        sampler, checkpoint = load_sampler(model_name, device=device)
        model_seed = args.seed + MODEL_SEED_OFFSETS[model_name]
        scenario_path = args.output_dir / f"{model_name}_scenarios.npy"
        config = MassiveGenerationConfig(
            scenario_count=args.n_scenarios,
            batch_size=args.batch_size,
            seed=model_seed,
        )
        result = generator.generate(
            model_name,
            sampler,
            normalized_condition,
            scenario_path,
            config=config,
            metadata={
                "condition_end_utc": snapshot.condition_end_utc,
                "forecast_start_utc": snapshot.forecast_start_utc,
                "checkpoint": relative_or_absolute(checkpoint, PROJECT_ROOT),
            },
            progress=progress_printer(model_name),
        )
        summaries[model_name] = result.summary
        scenario_paths[scenario_path.name] = scenario_path
        model_records[model_name] = {
            "checkpoint": relative_or_absolute(checkpoint, PROJECT_ROOT),
            "checkpoint_sha256": sha256_file(checkpoint),
            "scenario_path": relative_or_absolute(scenario_path, PROJECT_ROOT),
            "scenario_sha256": sha256_file(scenario_path),
            "shape": [args.n_scenarios, sampler.config.trajectory_length, sampler.config.n_assets],
            "dtype": "float32",
            "seed": model_seed,
        }
        del sampler
        gc.collect()

    report = PortfolioStressReport(portfolio.config, summaries)
    report_path = args.output_dir / "portfolio_stress_report.json"
    summary_path = args.output_dir / "scenario_summary.csv"
    risk_path = args.output_dir / "risk_comparison.csv"
    metadata_path = args.output_dir / "metadata.json"
    write_json_atomic(report.to_dict(), report_path)
    summary_records = report.summary_records()
    risk_records = report.risk_records()
    write_csv_atomic(summary_records, tuple(summary_records[0]), summary_path)
    write_csv_atomic(risk_records, tuple(risk_records[0]), risk_path)

    forecast_start = datetime.fromisoformat(
        snapshot.forecast_start_utc.replace("Z", "+00:00")
    )
    horizon_steps = next(iter(summaries.values())).horizon_steps
    forecast_end = forecast_start + timedelta(hours=6 * horizon_steps)
    maximum_abs_z = float(np.abs(normalized_condition).max())
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "massive_generation_conditioned_on_latest_available_market_state",
        "condition": {
            "panel": relative_or_absolute(args.panel, PROJECT_ROOT),
            "panel_sha256": sha256_file(args.panel),
            "condition_start_utc": snapshot.condition_start_utc,
            "condition_end_utc": snapshot.condition_end_utc,
            "forecast_start_utc": snapshot.forecast_start_utc,
            "forecast_end_exclusive_utc": forecast_end.isoformat().replace("+00:00", "Z"),
            "condition_steps": snapshot.condition_steps,
            "initial_prices": dict(zip(snapshot.assets, snapshot.initial_prices.tolist())),
            "maximum_absolute_normalized_feature": maximum_abs_z,
            "outside_five_training_standard_deviations": maximum_abs_z > 5.0,
            "features": {
                str(name): {
                    "raw": float(raw_value),
                    "normalized": float(normalized_value),
                }
                for name, raw_value, normalized_value in zip(
                    feature_names, snapshot.features, normalized_condition[0]
                )
            },
        },
        "generation": {
            "scenario_count_per_model": args.n_scenarios,
            "batch_size": args.batch_size,
            "base_seed": args.seed,
            "models": model_records,
            "note": (
                "Los escenarios reducen error Monte Carlo condicionado al modelo; "
                "no son observaciones históricas independientes."
            ),
        },
    }
    write_json_atomic(metadata, metadata_path)
    checksums_path = args.output_dir / "SHA256SUMS"
    write_checksums(
        {
            condition_path.name: condition_path,
            report_path.name: report_path,
            summary_path.name: summary_path,
            risk_path.name: risk_path,
            metadata_path.name: metadata_path,
            **scenario_paths,
        },
        checksums_path,
    )

    print(f"Estado condicionado hasta: {snapshot.condition_end_utc}")
    print(f"Inicio del horizonte simulado: {snapshot.forecast_start_utc}")
    print(f"Máximo |z| de la condición: {maximum_abs_z:.2f}")
    print(f"Artefactos: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
