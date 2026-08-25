#!/usr/bin/env python3
"""Compara la misma tarea downstream con sintéticos de los cuatro generadores."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from train_downstream_drawdown_models import (
    TrainConfig,
    maximum_drawdown_magnitude,
    plot_history,
    predict,
    regression_metrics,
    resolve_device,
    train_one_model,
)

from crypto_generative.data.stress import ProjectScenarioLoader
from crypto_generative.models import (
    ConditionalCVAEDecoder,
    ConditionalFlowGenerator,
    ConditionalGANGenerator,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/downstream_common"
MODEL_ORDER = ("block_bootstrap", "cvae", "normalizing_flow", "conditional_gan")
RATIOS = (0.0, 0.25, 0.50, 1.0)


class ScenarioSampler(Protocol):
    def sample(
        self,
        n: int,
        cond: np.ndarray,
        *,
        seed: int | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    sampler: ScenarioSampler
    normalized_output: bool
    generation_seed: int


@dataclass(frozen=True)
class FrozenDownstreamData:
    train_conditions: np.ndarray
    train_returns: np.ndarray
    validation_conditions: np.ndarray
    validation_returns: np.ndarray
    test_conditions: np.ndarray
    test_returns: np.ndarray
    return_mean: np.ndarray
    return_scale: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _positions(stored_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    position_by_id = {
        int(sample_id): index for index, sample_id in enumerate(stored_ids)
    }
    try:
        return np.asarray(
            [position_by_id[int(sample_id)] for sample_id in requested_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"sample_id desconocido: {error.args[0]}") from error


def load_frozen_data() -> FrozenDownstreamData:
    with np.load(NORMALIZED, allow_pickle=False) as arrays:
        stored_ids = arrays["sample_ids"].astype(np.int64)
        condition = arrays["condition_features"].astype(np.float32)
        normalized_returns = arrays["target_returns"].astype(np.float32)
        return_mean = arrays["return_mean"].astype(np.float32)
        return_scale = arrays["return_scale"].astype(np.float32)
    with np.load(SPLIT, allow_pickle=False) as split:
        train_ids = split["train_sample_ids"].astype(np.int64)
        validation_ids = split["validation_sample_ids"].astype(np.int64)
        test_ids = split["test_sample_ids"].astype(np.int64)

    def select(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = _positions(stored_ids, ids)
        original = (
            normalized_returns[positions] * return_scale[None, None, :]
            + return_mean[None, None, :]
        ).astype(np.float32)
        return condition[positions], original

    train_condition, train_returns = select(train_ids)
    validation_condition, validation_returns = select(validation_ids)
    test_condition, test_returns = select(test_ids)
    return FrozenDownstreamData(
        train_conditions=train_condition,
        train_returns=train_returns,
        validation_conditions=validation_condition,
        validation_returns=validation_returns,
        test_conditions=test_condition,
        test_returns=test_returns,
        return_mean=return_mean,
        return_scale=return_scale,
    )


def load_sampler(name: str, *, device: str, seed: int) -> SamplerSpec:
    generation_seed = seed + 10_000
    if name == "cvae":
        sampler = ConditionalCVAEDecoder.load(
            PROJECT_ROOT / "outputs/cvae_best/decoder.keras",
            metadata_path=PROJECT_ROOT / "outputs/cvae_best/metadata.json",
        )
        return SamplerSpec(name, sampler, True, generation_seed)
    if name == "normalizing_flow":
        sampler = ConditionalFlowGenerator.load(
            PROJECT_ROOT
            / "outputs/normalizing_flow_best/conditional_realnvp_best.pt",
            device=device,
        )
        return SamplerSpec(name, sampler, True, generation_seed)
    if name == "conditional_gan":
        sampler = ConditionalGANGenerator.load(
            PROJECT_ROOT / "outputs/conditional_gan_best/conditional_gan_best.pt",
            device=device,
        )
        return SamplerSpec(name, sampler, True, generation_seed)
    if name == "block_bootstrap":
        loader = ProjectScenarioLoader(NORMALIZED, SPLIT, SPLIT_INDEX, PANEL)
        train = loader.load_split("train")
        train_conditions = loader.load_normalized_conditions(train.sample_ids)
        sampler = ConditionalMultivariateBlockBootstrap(
            frozen_conditional_bootstrap_config(random_state=generation_seed)
        ).fit(train.log_returns, train_conditions)
        return SamplerSpec(name, sampler, False, generation_seed)
    raise ValueError(f"Modelo no soportado: {name}")


def generate_training_paths(
    spec: SamplerSpec,
    conditions: np.ndarray,
    *,
    return_mean: np.ndarray,
    return_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("generation-batch-size debe ser positivo")
    chunks = []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        generated = np.asarray(
            spec.sampler.sample(
                stop - start,
                conditions[start:stop],
                seed=spec.generation_seed + batch_index,
            ),
            dtype=np.float32,
        )
        if generated.shape != (stop - start, 120, 2):
            raise ValueError(f"{spec.name} produjo {generated.shape}")
        if spec.normalized_output:
            generated = (
                generated * return_scale[None, None, :]
                + return_mean[None, None, :]
            ).astype(np.float32)
        if not np.isfinite(generated).all():
            raise ValueError(f"{spec.name} produjo valores no finitos")
        chunks.append(generated)
        print(f"{spec.name}: generadas {stop:4d}/{len(conditions)} trayectorias")
    return np.concatenate(chunks, axis=0)


class CommonDownstreamExperiment:
    """Entrena una única MLP cambiando solo la fuente y proporción sintética."""

    def __init__(
        self,
        data: FrozenDownstreamData,
        *,
        config: TrainConfig,
        device: str,
        output_dir: Path,
    ) -> None:
        self.data = data
        self.config = config
        self.device = resolve_device(device)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_y = self._drawdown(data.train_returns)
        self.validation_y = self._drawdown(data.validation_returns)
        self.test_y = self._drawdown(data.test_returns)
        self.target_mean = float(self.train_y.mean())
        self.target_std = float(self.train_y.std(ddof=0))
        if self.target_std <= 0:
            raise ValueError("El target real de train tiene desviación nula")

    @staticmethod
    def _drawdown(paths: np.ndarray) -> np.ndarray:
        return maximum_drawdown_magnitude(
            paths,
            btc_weight=0.60,
            eth_weight=0.40,
        )

    def run(
        self,
        model_name: str,
        synthetic_conditions: np.ndarray,
        synthetic_returns: np.ndarray,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, float]]]]:
        synthetic_y = self._drawdown(synthetic_returns)
        rows = []
        histories = {}
        for ratio_index, ratio in enumerate(RATIOS):
            synthetic_count = int(round(len(self.train_y) * ratio))
            train_x = np.concatenate(
                [self.data.train_conditions, synthetic_conditions[:synthetic_count]],
                axis=0,
            ).astype(np.float32)
            train_y = np.concatenate(
                [self.train_y, synthetic_y[:synthetic_count]], axis=0
            ).astype(np.float32)
            order = np.random.default_rng(self.config.seed + ratio_index).permutation(
                len(train_x)
            )
            train_x = train_x[order]
            train_y = train_y[order]

            label = "real_only" if ratio == 0 else f"real_plus_{int(100 * ratio)}pct"
            print(
                f"\n{model_name}/{label}: real={len(self.train_y)}, "
                f"sintético={synthetic_count}, total={len(train_x)}"
            )
            model, history, best_epoch, best_validation = train_one_model(
                train_x,
                train_y,
                self.data.validation_conditions,
                self.validation_y,
                target_mean=self.target_mean,
                target_std=self.target_std,
                config=self.config,
                device=self.device,
            )
            validation_prediction = predict(
                model,
                self.data.validation_conditions,
                target_mean=self.target_mean,
                target_std=self.target_std,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            test_prediction = predict(
                model,
                self.data.test_conditions,
                target_mean=self.target_mean,
                target_std=self.target_std,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            validation_metrics = regression_metrics(
                self.validation_y, validation_prediction
            )
            test_metrics = regression_metrics(self.test_y, test_prediction)
            row = {
                "model": model_name,
                "dataset": label,
                "additional_synthetic_ratio": ratio,
                "synthetic_share_of_total": (
                    synthetic_count / len(train_x) if synthetic_count else 0.0
                ),
                "real_train": len(self.train_y),
                "synthetic_train": synthetic_count,
                "total_train": len(train_x),
                "best_epoch": best_epoch,
                "best_validation_mse_scaled": best_validation,
                "validation_mae": validation_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_correlation": test_metrics["pearson_correlation"],
                "test_tail_mae_worst_10pct": test_metrics[
                    "tail_mae_worst_10pct"
                ],
                "test_tail_mae_worst_5pct": test_metrics[
                    "tail_mae_worst_5pct"
                ],
            }
            rows.append(row)
            histories[label] = history
            print(
                f"test MAE={row['test_mae']:.4%} | "
                f"R2={row['test_r2']:.4f} | "
                f"tail MAE={row['test_tail_mae_worst_10pct']:.4%}"
            )
        return rows, histories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    ratios = list(RATIOS)
    for metric, ylabel, filename in (
        ("test_mae", "MAE de drawdown", "comparison_test_mae.png"),
        (
            "test_tail_mae_worst_10pct",
            "MAE en el peor 10 %",
            "comparison_test_tail_mae.png",
        ),
        ("test_r2", "R²", "comparison_test_r2.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name in MODEL_ORDER:
            model_rows = [row for row in rows if row["model"] == model_name]
            if not model_rows:
                continue
            ax.plot(
                ratios,
                [float(row[metric]) for row in model_rows],
                marker="o",
                label=model_name,
            )
        ax.set_xticks(ratios, ["0 %", "+25 %", "+50 %", "+100 %"])
        ax.set_xlabel("Sintéticos adicionales respecto al train real")
        ax.set_ylabel(ylabel)
        ax.set_title("Comparación downstream común")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.generation_batch_size <= 0:
        raise ValueError("generation-batch-size debe ser positivo")
    config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    data = load_frozen_data()
    rng = np.random.default_rng(args.seed)
    synthetic_order = rng.permutation(len(data.train_conditions))
    synthetic_conditions = data.train_conditions[synthetic_order]
    experiment = CommonDownstreamExperiment(
        data,
        config=config,
        device=args.device,
        output_dir=args.output_dir,
    )

    all_rows = []
    metadata: dict[str, Any] = {
        "task": "predict_30d_maximum_drawdown_magnitude_60_40_btc_eth",
        "configuration": asdict(config),
        "ratios": list(RATIOS),
        "ratio_definition": "synthetic samples added / real training samples",
        "validation_and_test": "real_only",
        "models": {},
    }
    for model_name in MODEL_ORDER:
        if model_name not in args.models:
            continue
        print(f"\n{'=' * 80}\nMODELO: {model_name}\n{'=' * 80}")
        spec = load_sampler(model_name, device=args.device, seed=args.seed)
        synthetic_returns = generate_training_paths(
            spec,
            synthetic_conditions,
            return_mean=data.return_mean,
            return_scale=data.return_scale,
            batch_size=args.generation_batch_size,
        )
        rows, histories = experiment.run(
            model_name, synthetic_conditions, synthetic_returns
        )
        all_rows.extend(rows)
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        write_csv(model_dir / "comparison.csv", rows)
        for row in rows:
            label = str(row["dataset"])
            write_csv(model_dir / f"history_{label}.csv", histories[label])
            plot_history(
                histories[label],
                title=f"Downstream {model_name} - {label}",
                best_epoch=int(row["best_epoch"]),
                destination=model_dir / f"history_{label}.png",
            )
        metadata["models"][model_name] = {
            "normalized_generator_output": spec.normalized_output,
            "generation_seed": spec.generation_seed,
            "comparison": rows,
        }

    if not all_rows:
        raise ValueError("No se seleccionó ningún modelo")
    write_csv(args.output_dir / "comparison.csv", all_rows)
    plot_comparison(all_rows, args.output_dir)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResultados guardados en {args.output_dir}")


if __name__ == "__main__":
    main()
