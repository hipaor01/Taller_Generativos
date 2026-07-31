#!/usr/bin/env python3
"""Construye ventanas de condicion y objetivo para BTC-ETH."""

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
from crypto_generative.data.binance import INTERVAL_MILLISECONDS
from crypto_generative.data.windows import TemporalWindowBuilder


def steps_for_days(days: int, interval: str) -> int:
    day_ms = 24 * 60 * 60 * 1000
    interval_ms = INTERVAL_MILLISECONDS[interval]
    if day_ms % interval_ms:
        raise ValueError(f"El intervalo {interval} no divide dias completos")
    return days * day_ms // interval_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=PROJECT_ROOT / "data" / "windows" / "binance",
    )
    parser.add_argument("--interval", default="6h", choices=INTERVAL_MILLISECONDS)
    parser.add_argument("--condition-days", type=int, default=60)
    parser.add_argument("--target-days", type=int, default=30)
    parser.add_argument("--stride-steps", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    condition_steps = steps_for_days(args.condition_days, args.interval)
    target_steps = steps_for_days(args.target_days, args.interval)
    builder = TemporalWindowBuilder(
        condition_steps=condition_steps,
        target_steps=target_steps,
        stride_steps=args.stride_steps,
        interval=args.interval,
    )
    dataset = builder.build(args.returns)

    stem = f"btc_eth_{args.interval}_c{condition_steps}_t{target_steps}_s{args.stride_steps}"
    arrays_path = args.output_dir / f"{stem}.npz"
    index_path = args.output_dir / f"{stem}_index.csv"
    manifest_path = args.output_dir / f"{stem}_manifest.json"
    checksums_path = args.output_dir / f"{stem}_SHA256SUMS"
    builder.write_npz(dataset, arrays_path)
    builder.write_index(dataset, index_path)

    manifest = {
        "dataset": "Ventanas temporales conjuntas BTC-ETH",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "path": relative_or_absolute(args.returns, PROJECT_ROOT),
            "sha256": sha256_file(args.returns),
        },
        "outputs": {
            "arrays": {
                "path": relative_or_absolute(arrays_path, PROJECT_ROOT),
                "sha256": sha256_file(arrays_path),
                "condition_shape": list(dataset.condition_returns.shape),
                "target_shape": list(dataset.target_returns.shape),
                "dtype": str(dataset.condition_returns.dtype),
                "asset_order": list(builder.assets),
            },
            "index": {
                "path": relative_or_absolute(index_path, PROJECT_ROOT),
                "sha256": sha256_file(index_path),
            },
        },
        "window_definition": {
            "condition_days": args.condition_days,
            "target_days": args.target_days,
            "condition_steps": condition_steps,
            "target_steps": target_steps,
            "stride_steps": args.stride_steps,
            "overlap_note": "Las ventanas consecutivas se solapan y no son independientes",
            "quality_rule": "Todos los retornos de condicion y objetivo deben ser validos",
        },
        "audit": dataset.audit.to_dict(),
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

    print(f"Matrices -> {arrays_path}")
    print(f"Indice -> {index_path}")
    print(
        f"{dataset.audit.candidate_windows} candidatas: "
        f"{dataset.audit.valid_windows} validas, "
        f"{dataset.audit.rejected_windows} rechazadas"
    )
    print(f"Condicion: {dataset.condition_returns.shape}; objetivo: {dataset.target_returns.shape}")
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

