#!/usr/bin/env python3
"""Ejecuta la aplicación común de cartera y escenarios de estrés.

La ejecución base siempre compara test histórico, crisis históricas seleccionadas,
shocks prefijados y el block bootstrap. Los artefactos compatibles de CVAE, flow y
GAN se incorporan automáticamente si existen o mediante ``--generative NAME=PATH``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto_generative_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/crypto_generative_cache")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import PROJECT_ROOT as ROOT  # noqa: F401
from crypto_generative.data import ProjectScenarioLoader
from crypto_generative.data.artifacts import write_csv_atomic, write_json_atomic
from crypto_generative.models import BlockBootstrapConfig, MultivariateBlockBootstrap
from crypto_generative.portfolio import (
    BuyAndHoldPortfolio,
    PortfolioConfig,
    PortfolioStressApplication,
    ScenarioCategory,
    StressScenarioSet,
    default_prefixed_scenarios,
)


DEFAULT_GENERATIVE_ARTIFACTS = {
    "cvae": ROOT / "outputs/cvae_best/test_scenarios.npz",
    "normalizing_flow": ROOT / "outputs/normalizing_flow_best/test_scenarios.npz",
    "conditional_gan": ROOT / "outputs/conditional_gan_best/test_scenarios.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-path",
        type=Path,
        default=ROOT
        / "data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=ROOT
        / "data/splits/binance/btc_eth_6h_c240_t120_purged_split.npz",
    )
    parser.add_argument(
        "--split-index-path",
        type=Path,
        default=ROOT
        / "data/splits/binance/btc_eth_6h_c240_t120_purged_split_index.csv",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=ROOT / "data/processed/binance/btc_eth_6h_panel.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/portfolio_stress_test",
    )
    parser.add_argument("--initial-value", type=float, default=100_000.0)
    parser.add_argument("--btc-weight", type=float, default=0.60)
    parser.add_argument("--eth-weight", type=float, default=0.40)
    parser.add_argument("--bootstrap-scenarios", type=int, default=5_000)
    parser.add_argument("--bootstrap-block-length", type=int, default=12)
    parser.add_argument("--historical-stress-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generative",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Artefacto NPZ adicional; se puede repetir.",
    )
    parser.add_argument(
        "--no-auto-generative",
        action="store_true",
        help="No buscar automáticamente los artefactos estándar de los notebooks.",
    )
    return parser.parse_args()


def build_scenario_sets(
    args: argparse.Namespace,
    loader: ProjectScenarioLoader,
    portfolio: BuyAndHoldPortfolio,
) -> Sequence[StressScenarioSet]:
    test = loader.load_split("test")
    training = loader.load_bootstrap_training_series()
    if test.assets != portfolio.config.assets or training.assets != test.assets:
        raise ValueError("El orden de activos de datos y cartera no coincide")

    scenario_sets: list[StressScenarioSet] = [
        StressScenarioSet(
            name="historical_test_distribution",
            category=ScenarioCategory.HISTORICAL,
            log_returns=test.log_returns,
            labels=test.labels,
            metadata={
                "split": "test",
                "selection": "all_sliding_windows",
                "overlapping_windows": True,
            },
        )
    ]
    worst_indices = portfolio.select_stress_paths(
        test.log_returns,
        args.historical_stress_count,
        criterion="maximum_drawdown",
    )
    scenario_sets.append(
        StressScenarioSet(
            name="historical_worst_drawdowns",
            category=ScenarioCategory.HISTORICAL,
            log_returns=test.log_returns[worst_indices],
            labels=tuple(test.labels[index] for index in worst_indices),
            metadata={
                "split": "test",
                "selection": "largest_portfolio_maximum_drawdown",
                "selected_sample_ids": [
                    int(test.sample_ids[index]) for index in worst_indices
                ],
                "tail_selected": True,
                "warning": "No interpretar VaR/ES de este subconjunto como cobertura.",
            },
        )
    )
    scenario_sets.extend(default_prefixed_scenarios())

    bootstrap = MultivariateBlockBootstrap(
        BlockBootstrapConfig(
            block_length=args.bootstrap_block_length,
            horizon_steps=test.log_returns.shape[1],
            random_state=args.seed + 5_000,
        )
    ).fit(training.log_returns, training.segment_ids)
    scenario_sets.append(
        StressScenarioSet(
            name="block_bootstrap",
            category=ScenarioCategory.BASELINE,
            log_returns=bootstrap.sample(args.bootstrap_scenarios),
            metadata={
                "block_length_steps": args.bootstrap_block_length,
                "seed": args.seed + 5_000,
                "training_unique_returns": len(training.log_returns),
                "candidate_blocks": len(bootstrap.blocks),
            },
        )
    )

    artifact_specs = {}
    if not args.no_auto_generative:
        artifact_specs.update(
            {
                name: path
                for name, path in DEFAULT_GENERATIVE_ARTIFACTS.items()
                if path.exists()
            }
        )
    for specification in args.generative:
        if "=" not in specification:
            raise ValueError("--generative debe usar NAME=PATH")
        name, raw_path = specification.split("=", 1)
        if not name or not raw_path:
            raise ValueError("--generative debe usar NAME=PATH")
        artifact_specs[name] = Path(raw_path)

    for name, path in artifact_specs.items():
        scenario_sets.append(
            loader.load_generated_scenarios(
                name,
                path,
                expected_reference=test,
            )
        )
    return scenario_sets


def save_plots(
    scenario_sets: Sequence[StressScenarioSet],
    portfolio: BuyAndHoldPortfolio,
    output_dir: Path,
) -> None:
    distribution_sets = [
        scenario
        for scenario in scenario_sets
        if scenario.name != "historical_worst_drawdowns"
        and scenario.category is not ScenarioCategory.PREFIXED
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario in distribution_sets:
        valued = portfolio.revalue(scenario.log_returns)
        losses = valued.final_loss_amounts / portfolio.config.initial_value
        ax.hist(losses, bins=60, density=True, alpha=0.35, label=scenario.name)
    ax.set(
        title="Distribución de pérdida final de la cartera 60/40",
        xlabel="Pérdida / valor inicial",
        ylabel="Densidad",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_final_loss_distributions.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for scenario in scenario_sets:
        valued = portfolio.revalue(scenario.log_returns)
        worst_index = int(np.argmax(valued.maximum_drawdowns))
        ax.plot(
            valued.values[worst_index] / portfolio.config.initial_value,
            label=scenario.name,
        )
    ax.set(
        title="Peor trayectoria de drawdown por conjunto",
        xlabel="Paso de 6 horas",
        ylabel="Valor de cartera / valor inicial",
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "02_worst_portfolio_paths.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = ProjectScenarioLoader(
        args.normalized_path,
        args.split_path,
        args.split_index_path,
        args.panel_path,
    )
    portfolio = BuyAndHoldPortfolio(
        PortfolioConfig(
            assets=("BTC", "ETH"),
            weights=(args.btc_weight, args.eth_weight),
            initial_value=args.initial_value,
        )
    )
    scenario_sets = build_scenario_sets(args, loader, portfolio)
    report = PortfolioStressApplication(portfolio).run(scenario_sets)

    write_json_atomic(report.to_dict(), args.output_dir / "stress_report.json")
    write_csv_atomic(
        report.summary_records(),
        tuple(report.summary_records()[0].keys()),
        args.output_dir / "scenario_summary.csv",
    )
    write_csv_atomic(
        report.risk_records(),
        tuple(report.risk_records()[0].keys()),
        args.output_dir / "risk_comparison.csv",
    )
    save_plots(scenario_sets, portfolio, args.output_dir)

    print("\nAPLICACIÓN COMÚN DE STRESS TESTING — CARTERA BTC/ETH 60/40")
    print("=" * 72)
    print(f"Valor inicial: {portfolio.config.initial_value:,.2f} USD")
    for row in report.summary_records():
        summary = report.scenarios[row["scenario_set"]]
        risk_95 = summary.risk["0.95"]
        print(
            f"{summary.name:34s} n={summary.scenario_count:6d} "
            f"VaR95={risk_95.value_at_risk_fraction:7.2%} "
            f"ES95={risk_95.expected_shortfall_fraction:7.2%} "
            f"MDD95={summary.maximum_drawdown.quantiles['q95']:7.2%}"
        )
    missing_defaults = [
        name
        for name, path in DEFAULT_GENERATIVE_ARTIFACTS.items()
        if not path.exists()
    ]
    if missing_defaults and not args.no_auto_generative:
        print(
            "\nArtefactos generativos no encontrados (se añadirán al rerun): "
            + ", ".join(missing_defaults)
        )
    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
