#!/usr/bin/env python3
"""Generate train-conditioned synthetic BTC/ETH paths and augmentation datasets.

The script loads the best Conditional RealNVP checkpoint, conditions it only on
market states from the temporal training block, and creates nested training
sets containing 0%, 25%, 50% and 100% additional synthetic samples.

Validation and test remain real-only. The test block is never used to train,
select, condition, or generate the synthetic training observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

# Make the repository's src/ package importable when the script is launched
# directly from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_generative.models import ConditionalFlowGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-path",
        type=Path,
        default=PROJECT_ROOT
        / "data/normalized/binance/btc_eth_6h_c240_t120_train_normalized.npz",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT
        / "data/splits/binance/btc_eth_6h_c240_t120_purged_split.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "results/normalizing_flow_60epochs/conditional_realnvp_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/augmented/normalizing_flow",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 1.00],
        help="Additional synthetic samples as a fraction of the real train size.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=256,
        help="Chunk size used while sampling to limit memory use.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_ratios(ratios: Iterable[float]) -> list[float]:
    cleaned = sorted(set(float(value) for value in ratios))
    if not cleaned:
        raise ValueError("At least one augmentation ratio is required")
    if any(value <= 0 for value in cleaned):
        raise ValueError("Every augmentation ratio must be greater than zero")
    if any(value > 1 for value in cleaned):
        raise ValueError(
            "This script supports up to one synthetic path per training condition "
            "(ratio <= 1.0)."
        )
    return cleaned


def sample_in_batches(
    flow: ConditionalFlowGenerator,
    conditions: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("generation-batch-size must be positive")

    generated: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        batch = flow.sample(
            stop - start,
            conditions[start:stop],
            seed=seed + batch_index,
        )
        generated.append(batch.astype(np.float32, copy=False))
        print(f"Generated {stop:5d}/{len(conditions)} synthetic train paths")
    return np.concatenate(generated, axis=0)


def save_real_split(
    destination: Path,
    *,
    split_name: str,
    ids: np.ndarray,
    condition: np.ndarray,
    target: np.ndarray,
    return_mean: np.ndarray,
    return_scale: np.ndarray,
    assets: np.ndarray,
    feature_names: np.ndarray,
) -> None:
    original_scale = target[ids] * return_scale + return_mean
    np.savez_compressed(
        destination,
        split_name=np.asarray(split_name),
        condition_features=condition[ids].astype(np.float32),
        target_returns=target[ids].astype(np.float32),
        target_returns_original_scale=original_scale.astype(np.float32),
        source_is_synthetic=np.zeros(len(ids), dtype=np.int8),
        condition_source_sample_ids=ids.astype(np.int64),
        assets=assets,
        feature_names=feature_names,
        return_mean=return_mean.astype(np.float32),
        return_scale=return_scale.astype(np.float32),
    )


def save_augmented_train_set(
    destination: Path,
    *,
    real_ids: np.ndarray,
    condition: np.ndarray,
    target: np.ndarray,
    synthetic_conditions: np.ndarray,
    synthetic_targets: np.ndarray,
    synthetic_condition_ids: np.ndarray,
    synthetic_count: int,
    return_mean: np.ndarray,
    return_scale: np.ndarray,
    assets: np.ndarray,
    feature_names: np.ndarray,
    seed: int,
) -> None:
    real_condition = condition[real_ids].astype(np.float32)
    real_target = target[real_ids].astype(np.float32)

    combined_condition = np.concatenate(
        [real_condition, synthetic_conditions[:synthetic_count]], axis=0
    )
    combined_target = np.concatenate(
        [real_target, synthetic_targets[:synthetic_count]], axis=0
    )
    source_is_synthetic = np.concatenate(
        [
            np.zeros(len(real_ids), dtype=np.int8),
            np.ones(synthetic_count, dtype=np.int8),
        ]
    )
    source_ids = np.concatenate(
        [real_ids.astype(np.int64), synthetic_condition_ids[:synthetic_count]]
    )

    # Shuffle real and synthetic observations together while preserving a fully
    # deterministic and reproducible dataset order.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combined_condition))
    combined_condition = combined_condition[order]
    combined_target = combined_target[order]
    source_is_synthetic = source_is_synthetic[order]
    source_ids = source_ids[order]

    original_scale = combined_target * return_scale + return_mean
    np.savez_compressed(
        destination,
        split_name=np.asarray("train"),
        condition_features=combined_condition,
        target_returns=combined_target,
        target_returns_original_scale=original_scale.astype(np.float32),
        source_is_synthetic=source_is_synthetic,
        condition_source_sample_ids=source_ids,
        assets=assets,
        feature_names=feature_names,
        return_mean=return_mean.astype(np.float32),
        return_scale=return_scale.astype(np.float32),
    )


def project_relative(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args.ratios)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_npz(args.normalized_path)
    split = load_npz(args.split_path)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    condition = arrays["condition_features"].astype(np.float32)
    target = arrays["target_returns"].astype(np.float32)
    return_mean = arrays["return_mean"].astype(np.float32)
    return_scale = arrays["return_scale"].astype(np.float32)
    assets = arrays["assets"]
    feature_names = arrays["feature_names"]

    train_ids = split["train_sample_ids"].astype(np.int64)
    validation_ids = split["validation_sample_ids"].astype(np.int64)
    test_ids = split["test_sample_ids"].astype(np.int64)

    print(f"Loading checkpoint: {args.checkpoint}")
    flow = ConditionalFlowGenerator.load(args.checkpoint, device=args.device)
    print(f"Flow loaded on {flow.device}")
    print(
        f"Real splits | train={len(train_ids)} "
        f"validation={len(validation_ids)} test={len(test_ids)}"
    )

    # Shuffle the training condition IDs once. Smaller augmentation datasets are
    # nested subsets of the largest one, enabling fair ratio comparisons.
    rng = np.random.default_rng(args.seed)
    synthetic_condition_ids = rng.permutation(train_ids)
    synthetic_conditions = condition[synthetic_condition_ids]

    synthetic_targets = sample_in_batches(
        flow,
        synthetic_conditions,
        batch_size=args.generation_batch_size,
        seed=args.seed + 10_000,
    )
    synthetic_original_scale = synthetic_targets * return_scale + return_mean

    master_path = args.output_dir / "flow_train_synthetic_master.npz"
    np.savez_compressed(
        master_path,
        condition_features=synthetic_conditions,
        generated_normalized=synthetic_targets,
        generated_returns=synthetic_original_scale.astype(np.float32),
        condition_source_sample_ids=synthetic_condition_ids,
        source_is_synthetic=np.ones(len(synthetic_targets), dtype=np.int8),
        assets=assets,
        feature_names=feature_names,
        return_mean=return_mean,
        return_scale=return_scale,
    )

    # Export immutable real-only validation/test sets for downstream evaluation.
    save_real_split(
        args.output_dir / "validation_real_only.npz",
        split_name="validation",
        ids=validation_ids,
        condition=condition,
        target=target,
        return_mean=return_mean,
        return_scale=return_scale,
        assets=assets,
        feature_names=feature_names,
    )
    save_real_split(
        args.output_dir / "test_real_only.npz",
        split_name="test",
        ids=test_ids,
        condition=condition,
        target=target,
        return_mean=return_mean,
        return_scale=return_scale,
        assets=assets,
        feature_names=feature_names,
    )

    # Baseline training dataset: real observations only.
    save_augmented_train_set(
        args.output_dir / "train_real_only.npz",
        real_ids=train_ids,
        condition=condition,
        target=target,
        synthetic_conditions=synthetic_conditions,
        synthetic_targets=synthetic_targets,
        synthetic_condition_ids=synthetic_condition_ids,
        synthetic_count=0,
        return_mean=return_mean,
        return_scale=return_scale,
        assets=assets,
        feature_names=feature_names,
        seed=args.seed,
    )

    manifest_rows: list[dict[str, object]] = [
        {
            "dataset": "train_real_only.npz",
            "real_samples": len(train_ids),
            "synthetic_samples": 0,
            "total_samples": len(train_ids),
            "additional_synthetic_ratio": 0.0,
            "synthetic_share_of_total": 0.0,
        }
    ]

    for ratio_index, ratio in enumerate(ratios, start=1):
        synthetic_count = int(round(len(train_ids) * ratio))
        label = f"{int(round(ratio * 100)):03d}pct"
        filename = f"train_real_plus_{label}_flow.npz"
        save_augmented_train_set(
            args.output_dir / filename,
            real_ids=train_ids,
            condition=condition,
            target=target,
            synthetic_conditions=synthetic_conditions,
            synthetic_targets=synthetic_targets,
            synthetic_condition_ids=synthetic_condition_ids,
            synthetic_count=synthetic_count,
            return_mean=return_mean,
            return_scale=return_scale,
            assets=assets,
            feature_names=feature_names,
            seed=args.seed + ratio_index,
        )
        manifest_rows.append(
            {
                "dataset": filename,
                "real_samples": len(train_ids),
                "synthetic_samples": synthetic_count,
                "total_samples": len(train_ids) + synthetic_count,
                "additional_synthetic_ratio": ratio,
                "synthetic_share_of_total": synthetic_count
                / (len(train_ids) + synthetic_count),
            }
        )

    with (args.output_dir / "dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "model": "conditional_realnvp",
        "checkpoint": project_relative(args.checkpoint),
        "normalized_path": project_relative(args.normalized_path),
        "split_path": project_relative(args.split_path),
        "seed": args.seed,
        "training_condition_count": len(train_ids),
        "validation_real_count": len(validation_ids),
        "test_real_count": len(test_ids),
        "ratios": ratios,
        "datasets": manifest_rows,
        "leakage_control": {
            "synthetic_condition_source": "temporal training block only",
            "validation": "real-only; used for downstream model selection",
            "test": "real-only; reserved for one final out-of-sample evaluation",
            "note": (
                "The flow checkpoint was selected with validation NLL, but no "
                "validation or test condition vector is used to construct the "
                "augmented training rows."
            ),
        },
    }
    (args.output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nFLOW TRAINING AUGMENTATION DATASETS")
    print("=" * 62)
    for row in manifest_rows:
        print(
            f"{row['dataset']:<38} "
            f"real={row['real_samples']:4d} "
            f"synthetic={row['synthetic_samples']:4d} "
            f"total={row['total_samples']:4d}"
        )
    print(f"\nSaved in: {args.output_dir}")
    print("Validation and test were exported unchanged as real-only datasets.")


if __name__ == "__main__":
    main()
