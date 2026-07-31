#!/usr/bin/env python3
"""Congela el split temporal BTC-ETH con purgas de 90 dias."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT

from crypto_generative.data.artifacts import (
    relative_or_absolute,
    sha256_file,
    write_checksums,
    write_json_atomic,
)
from crypto_generative.data.splits import PurgedTemporalSplitBuilder, TemporalSplitConfig


WINDOW_STEM = "btc_eth_6h_c240_t120_s1"
CONDITION_STEM = "btc_eth_6h_c240_condition_summary"
SPLIT_STEM = "btc_eth_6h_c240_t120_purged_split"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "windows" / "binance" / f"{WINDOW_STEM}_index.csv"
        ),
    )
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "binance",
    )
    parser.add_argument("--train-end", default="2023-07-01T00:00:00Z")
    parser.add_argument("--validation-start", default="2023-10-01T00:00:00Z")
    parser.add_argument("--validation-end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--test-start", default="2025-04-01T00:00:00Z")
    parser.add_argument("--minimum-purge-days", type=int, default=90)
    return parser


def validate_artifact_alignment(windows_path: Path, conditions_path: Path) -> int:
    with np.load(windows_path, allow_pickle=False) as windows:
        window_starts = windows["start_indices"]
        window_samples = windows["target_returns"].shape[0]
    with np.load(conditions_path, allow_pickle=False) as conditions:
        condition_ids = conditions["sample_ids"]
        condition_samples = conditions["condition_features"].shape[0]
    if window_samples != condition_samples:
        raise ValueError("Ventanas y condiciones no tienen el mismo numero de muestras")
    if len(window_starts) != window_samples:
        raise ValueError("start_indices no coincide con las ventanas")
    if not np.array_equal(condition_ids, np.arange(window_samples)):
        raise ValueError("Los sample_ids de condicion no estan alineados")
    return int(window_samples)


def main() -> int:
    args = build_parser().parse_args()
    expected_samples = validate_artifact_alignment(args.windows, args.conditions)
    config = TemporalSplitConfig(
        train_end_exclusive_utc=args.train_end,
        validation_start_utc=args.validation_start,
        validation_end_exclusive_utc=args.validation_end,
        test_start_utc=args.test_start,
        minimum_purge_days=args.minimum_purge_days,
    )
    builder = PurgedTemporalSplitBuilder(config)
    split = builder.build(args.index)
    if split.audit.total_samples != expected_samples:
        raise ValueError("El indice no esta alineado con las matrices")

    arrays_path = args.output_dir / f"{SPLIT_STEM}.npz"
    index_path = args.output_dir / f"{SPLIT_STEM}_index.csv"
    manifest_path = args.output_dir / f"{SPLIT_STEM}_manifest.json"
    checksums_path = args.output_dir / f"{SPLIT_STEM}_SHA256SUMS"
    builder.write_npz(split, arrays_path)
    builder.write_index(split, index_path)

    inputs = {
        "window_index": args.index,
        "windows": args.windows,
        "conditions": args.conditions,
    }
    outputs = {"split_arrays": arrays_path, "split_index": index_path}
    manifest = {
        "dataset": "Split temporal purgado BTC-ETH",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assignment_key": "target_start_utc",
        "config": {
            **config.__dict__,
            "purge_interpretation": (
                "Se eliminan 90 dias de inicios de objetivo; esto separa intervalos completos "
                "de condicion (60d) y objetivo (30d) sin timestamps compartidos"
            ),
        },
        "inputs": {
            name: {
                "path": relative_or_absolute(path, PROJECT_ROOT),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "outputs": {
            name: {
                "path": relative_or_absolute(path, PROJECT_ROOT),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "audit": split.audit.to_dict(),
        "rules": {
            "temporal_only": True,
            "random_shuffle": False,
            "test_dates_frozen": True,
            "normalization": "Pendiente; ajustar solo con train_sample_ids",
        },
    }
    write_json_atomic(manifest, manifest_path)
    write_checksums(
        {
            arrays_path.name: arrays_path,
            index_path.name: index_path,
            manifest_path.name: manifest_path,
        },
        checksums_path,
    )

    audit = split.audit
    print(f"Split -> {arrays_path}")
    print(
        f"Train={audit.train_samples}; validation={audit.validation_samples}; "
        f"test={audit.test_samples}; purgadas={audit.purged_samples}"
    )
    print(
        f"Separacion bruta train-validation={audit.train_validation_raw_gap_hours:g}h; "
        f"validation-test={audit.validation_test_raw_gap_hours:g}h"
    )
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
