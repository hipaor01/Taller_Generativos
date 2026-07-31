#!/usr/bin/env python3
"""Ajusta la normalizacion en train y la aplica sin reajuste al resto."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from crypto_generative.data.artifacts import (
    relative_or_absolute,
    sha256_file,
    write_checksums,
    write_json_atomic,
)
from crypto_generative.data.normalization import TrainingOnlyNormalizer


WINDOW_STEM = "btc_eth_6h_c240_t120_s1"
CONDITION_STEM = "btc_eth_6h_c240_condition_summary"
SPLIT_STEM = "btc_eth_6h_c240_t120_purged_split"
OUTPUT_STEM = "btc_eth_6h_c240_t120_train_normalized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=PROJECT_ROOT / "data" / "windows" / "binance" / f"{WINDOW_STEM}.npz",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "features" / "binance" / f"{CONDITION_STEM}.npz"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "binance" / f"{SPLIT_STEM}.npz",
    )
    parser.add_argument(
        "--returns",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "features"
            / "binance"
            / "btc_eth_6h_log_returns.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized" / "binance",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    normalizer = TrainingOnlyNormalizer()
    dataset = normalizer.build(args.windows, args.conditions, args.split, args.returns)

    arrays_path = args.output_dir / f"{OUTPUT_STEM}.npz"
    manifest_path = args.output_dir / f"{OUTPUT_STEM}_manifest.json"
    checksums_path = args.output_dir / f"{OUTPUT_STEM}_SHA256SUMS"
    normalizer.write_npz(dataset, arrays_path)

    inputs = {
        "windows": args.windows,
        "conditions": args.conditions,
        "split": args.split,
        "returns": args.returns,
    }
    manifest = {
        "dataset": "BTC-ETH normalizado exclusivamente con entrenamiento",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": {
            "transform": "z = (x - mean_train) / std_train",
            "standard_deviation": "Poblacional, ddof=0",
            "condition_fit": "Solo condition_features[train_sample_ids]",
            "return_fit": (
                "Retornos temporales unicos cubiertos por ventanas train; "
                "se evita ponderarlos por solapamiento"
            ),
            "validation_test": "Transformados con parametros train, sin reajuste ni clipping",
        },
        "inputs": {
            name: {
                "path": relative_or_absolute(path, PROJECT_ROOT),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "output": {
            "path": relative_or_absolute(arrays_path, PROJECT_ROOT),
            "sha256": sha256_file(arrays_path),
            "condition_features_shape": list(dataset.condition_features.shape),
            "condition_returns_shape": list(dataset.condition_returns.shape),
            "target_returns_shape": list(dataset.target_returns.shape),
            "dtype": str(dataset.condition_features.dtype),
        },
        "parameters": {
            "feature_names": dataset.feature_names.tolist(),
            "condition_feature_mean": dataset.condition_parameters.mean.tolist(),
            "condition_feature_scale": dataset.condition_parameters.scale.tolist(),
            "assets": dataset.assets.tolist(),
            "return_mean": dataset.return_parameters.mean.tolist(),
            "return_scale": dataset.return_parameters.scale.tolist(),
        },
        "audit": dataset.audit.to_dict(),
    }
    write_json_atomic(manifest, manifest_path)
    write_checksums(
        {arrays_path.name: arrays_path, manifest_path.name: manifest_path},
        checksums_path,
    )

    audit = dataset.audit
    print(f"Dataset normalizado -> {arrays_path}")
    print(
        f"Ajuste: {audit.train_samples} condiciones y "
        f"{audit.unique_training_return_rows} retornos temporales unicos"
    )
    print(
        f"Error maximo media/std train: condiciones "
        f"{audit.train_condition_max_abs_mean:.3g}/"
        f"{audit.train_condition_max_abs_std_error:.3g}; retornos "
        f"{audit.training_returns_max_abs_mean:.3g}/"
        f"{audit.training_returns_max_abs_std_error:.3g}"
    )
    print(f"Valores no finitos -> {audit.normalized_non_finite_values}")
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
