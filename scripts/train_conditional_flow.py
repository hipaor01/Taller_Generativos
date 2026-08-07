#!/usr/bin/env python3
"""Train and diagnose the conditional RealNVP model on the frozen BTC/ETH data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT as ROOT  # noqa: F401  (adds src/ to sys.path)
from crypto_generative.models import ConditionalFlowConfig, ConditionalFlowGenerator


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
        "--output-dir",
        type=Path,
        default=ROOT / "results/normalizing_flow_60epochs",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--scale-limit", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto when omitted")
    parser.add_argument(
        "--generate-count",
        type=int,
        default=1832,
        help="Number of validation-conditioned synthetic paths to save.",
    )
    return parser.parse_args()


def load_data(normalized_path: Path, split_path: Path) -> dict[str, np.ndarray]:
    with np.load(normalized_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(split_path, allow_pickle=False) as split:
        train_ids = split["train_sample_ids"]
        validation_ids = split["validation_sample_ids"]
        test_ids = split["test_sample_ids"]
    arrays.update(
        {
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "test_ids": test_ids,
        }
    )
    return arrays


def save_history(output_dir: Path, history) -> None:
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_nll_per_dimension", "validation_nll_per_dimension"])
        for epoch, (train_value, validation_value) in enumerate(
            zip(history.train_nll, history.validation_nll), start=1
        ):
            writer.writerow([epoch, train_value, validation_value])

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history.train_nll) + 1)
    ax.plot(epochs, history.train_nll, label="Train")
    ax.plot(epochs, history.validation_nll, label="Validation")
    if history.best_epoch >= 0:
        ax.axvline(history.best_epoch + 1, linestyle="--", label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood per dimension")
    ax.set_title("Conditional RealNVP training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def path_statistics(paths: np.ndarray) -> dict[str, np.ndarray]:
    cumulative = paths.sum(axis=1)
    volatility = paths.std(axis=1, ddof=1)
    path_correlations = np.array(
        [np.corrcoef(path[:, 0], path[:, 1])[0, 1] for path in paths], dtype=float
    )
    return {
        "cumulative_log_return_mean": cumulative.mean(axis=0),
        "cumulative_log_return_std": cumulative.std(axis=0, ddof=1),
        "six_hour_return_std_mean": volatility.mean(axis=0),
        "path_correlation_mean": np.array(path_correlations.mean()),
        "path_correlation_std": np.array(path_correlations.std(ddof=1)),
    }


def jsonable_metrics(metrics: dict[str, np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        result[key] = float(array) if array.ndim == 0 else array.tolist()
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_data(args.normalized_path, args.split_path)

    target = arrays["target_returns"].astype(np.float32)
    condition = arrays["condition_features"].astype(np.float32)
    train_ids = arrays["train_ids"]
    validation_ids = arrays["validation_ids"]

    config = ConditionalFlowConfig(
        n_coupling_layers=args.layers,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.hidden_layers,
        scale_limit=args.scale_limit,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    flow = ConditionalFlowGenerator(config=config, device=args.device)
    print(f"Training on {flow.device}; train={len(train_ids)}, validation={len(validation_ids)}")
    history = flow.fit(
        target[train_ids],
        condition[train_ids],
        X_validation=target[validation_ids],
        cond_validation=condition[validation_ids],
    )

    validation_log_prob = flow.log_prob(
        target[validation_ids], condition[validation_ids], per_dimension=True
    )
    checkpoint_path = flow.save(
        args.output_dir / "conditional_realnvp_best.pt",
        extra={
            "normalized_path": str(args.normalized_path.relative_to(ROOT)),
            "split_path": str(args.split_path.relative_to(ROOT)),
            "validation_log_prob_per_dimension_mean": float(validation_log_prob.mean()),
        },
    )
    flow.export_metadata(args.output_dir / "conditional_realnvp_metadata.json")
    save_history(args.output_dir, history)

    n_generate = min(args.generate_count, len(validation_ids))
    selected_ids = validation_ids[:n_generate]
    generated_normalized = flow.sample(
        n_generate, condition[selected_ids], seed=args.seed + 1
    )
    return_mean = arrays["return_mean"].astype(np.float32)
    return_scale = arrays["return_scale"].astype(np.float32)
    generated_returns = generated_normalized * return_scale + return_mean
    real_returns = target[selected_ids] * return_scale + return_mean

    np.savez_compressed(
        args.output_dir / "validation_synthetic_paths.npz",
        sample_ids=selected_ids,
        condition_features=condition[selected_ids],
        generated_normalized=generated_normalized,
        generated_returns=generated_returns,
        real_returns=real_returns,
        assets=arrays["assets"],
        return_mean=return_mean,
        return_scale=return_scale,
    )

    diagnostics = {
        "model": "conditional_realnvp",
        "config": asdict(config),
        "best_epoch_1_based": history.best_epoch + 1,
        "validation_nll_per_dimension": float(-validation_log_prob.mean()),
        "real_validation": jsonable_metrics(path_statistics(real_returns)),
        "synthetic_validation": jsonable_metrics(path_statistics(generated_returns)),
        "note": (
            "All financial diagnostics use inverse-normalized six-hour log returns. "
            "The test split has not been used."
        ),
    }
    (args.output_dir / "validation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best validation NLL/dim: {-validation_log_prob.mean():.6f}")
    print(f"Saved {n_generate} validation-conditioned synthetic paths")


if __name__ == "__main__":
    main()
