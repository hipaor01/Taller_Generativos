#!/usr/bin/env python3
"""Select Conditional RealNVP hyperparameters using only the validation block."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT as ROOT  # adds src/ to sys.path
from crypto_generative.models import ConditionalFlowConfig, ConditionalFlowGenerator


CANDIDATES = [
    {"n_coupling_layers": 4, "hidden_dim": 128, "learning_rate": 5e-4},
    {"n_coupling_layers": 6, "hidden_dim": 128, "learning_rate": 5e-4},
    {"n_coupling_layers": 8, "hidden_dim": 128, "learning_rate": 1e-3},
    {"n_coupling_layers": 10, "hidden_dim": 128, "learning_rate": 1e-3},
    {"n_coupling_layers": 8, "hidden_dim": 256, "learning_rate": 1e-3},
]


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
        default=ROOT / "results/normalizing_flow/search",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-trials", type=int, default=len(CANDIDATES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_arrays(normalized_path: Path, split_path: Path):
    with np.load(normalized_path, allow_pickle=False) as data:
        target = data["target_returns"].astype(np.float32)
        condition = data["condition_features"].astype(np.float32)
    with np.load(split_path, allow_pickle=False) as split:
        train_ids = split["train_sample_ids"]
        validation_ids = split["validation_sample_ids"]
    return target, condition, train_ids, validation_ids


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target, condition, train_ids, validation_ids = load_arrays(
        args.normalized_path, args.split_path
    )

    rows: list[dict[str, object]] = []
    best_nll = float("inf")
    best_trial = -1

    for trial_index, candidate in enumerate(CANDIDATES[: args.max_trials], start=1):
        config = ConditionalFlowConfig(
            n_coupling_layers=candidate["n_coupling_layers"],
            hidden_dim=candidate["hidden_dim"],
            learning_rate=candidate["learning_rate"],
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        print(f"\nTrial {trial_index}: {candidate}")
        flow = ConditionalFlowGenerator(config=config, device=args.device)
        history = flow.fit(
            target[train_ids],
            condition[train_ids],
            X_validation=target[validation_ids],
            cond_validation=condition[validation_ids],
            verbose=False,
        )
        val_nll = float(
            -flow.log_prob(
                target[validation_ids],
                condition[validation_ids],
                per_dimension=True,
            ).mean()
        )
        parameter_count = sum(
            parameter.numel()
            for parameter in flow.model.parameters()
            if parameter.requires_grad
        )
        row = {
            "trial": trial_index,
            "n_coupling_layers": config.n_coupling_layers,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.learning_rate,
            "parameter_count": parameter_count,
            "best_epoch_1_based": history.best_epoch + 1,
            "validation_nll_per_dimension": val_nll,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

        if val_nll < best_nll:
            best_nll = val_nll
            best_trial = trial_index
            flow.save(
                args.output_dir / "best_search_checkpoint.pt",
                extra={"trial": trial_index, "validation_nll_per_dimension": val_nll},
            )
            (args.output_dir / "best_search_config.json").write_text(
                json.dumps(asdict(config), indent=2), encoding="utf-8"
            )

    csv_path = args.output_dir / "hyperparameter_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selection_metric": "validation_nll_per_dimension",
        "best_trial": best_trial,
        "best_validation_nll_per_dimension": best_nll,
        "test_split_used": False,
        "n_trials": len(rows),
    }
    (args.output_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved search results to {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
