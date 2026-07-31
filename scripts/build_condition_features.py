#!/usr/bin/env python3
"""Resume cada ventana de condicion BTC-ETH en variables de estado de mercado."""

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
from crypto_generative.data.condition import ConditionFeatureBuilder


WINDOW_STEM = "btc_eth_6h_c240_t120_s1"


def steps_for_days(days: int, interval: str) -> int:
    day_ms = 24 * 60 * 60 * 1000
    interval_ms = INTERVAL_MILLISECONDS[interval]
    if day_ms % interval_ms:
        raise ValueError(f"El intervalo {interval} no divide dias completos")
    return days * day_ms // interval_ms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=PROJECT_ROOT / "data" / "windows" / "binance" / f"{WINDOW_STEM}.npz",
    )
    parser.add_argument(
        "--window-index",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "windows" / "binance" / f"{WINDOW_STEM}_index.csv"
        ),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "processed" / "binance" / "btc_eth_6h_panel.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "binance",
    )
    parser.add_argument("--interval", default="6h", choices=INTERVAL_MILLISECONDS)
    parser.add_argument("--volume-recent-days", type=int, default=7)
    parser.add_argument("--correlation-days", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    builder = ConditionFeatureBuilder(
        interval=args.interval,
        volume_recent_steps=steps_for_days(args.volume_recent_days, args.interval),
        correlation_steps=steps_for_days(args.correlation_days, args.interval),
    )
    dataset = builder.build(args.windows, args.window_index, args.panel)

    output_stem = (
        f"btc_eth_{args.interval}_c{dataset.audit.condition_steps}_condition_summary"
    )
    arrays_path = args.output_dir / f"{output_stem}.npz"
    table_path = args.output_dir / f"{output_stem}.csv"
    manifest_path = args.output_dir / f"{output_stem}_manifest.json"
    checksums_path = args.output_dir / f"{output_stem}_SHA256SUMS"
    builder.write_npz(dataset, arrays_path)
    builder.write_table(dataset, table_path)

    inputs = {
        "windows": args.windows,
        "window_index": args.window_index,
        "panel": args.panel,
    }
    outputs = {"arrays": arrays_path, "table": table_path}
    manifest = {
        "dataset": "Vector resumido de condicion BTC-ETH",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        "shape": list(dataset.features.shape),
        "dtype": str(dataset.features.dtype),
        "feature_names": list(dataset.feature_names),
        "definitions": {
            "cumulative_log_return": "Suma de retornos logaritmicos durante la condicion",
            "realized_volatility_ann": (
                "Desviacion estandar muestral anualizada con "
                f"{dataset.audit.annualization_steps} pasos"
            ),
            "current_drawdown": "exp(log_wealth_final - max_log_wealth) - 1",
            "log_volume_z": (
                "Media log1p(volumen) reciente centrada y escalada dentro de la condicion"
            ),
            "log_volume_change": (
                "Media log1p(volumen) reciente menos el bloque inmediatamente anterior"
            ),
            "mean_log_range": "Media reciente de log(high/low)",
            "btc_eth_correlation": "Correlacion reciente de Pearson de los retornos",
            "joint_trend_regime_score": (
                "Tendencia log conjunta dividida por volatilidad media y raiz del numero de pasos"
            ),
            "normalization_note": (
                "No se aplica normalizacion global; debe ajustarse solo con el split de entrenamiento"
            ),
        },
        "audit": dataset.audit.to_dict(),
    }
    write_json_atomic(manifest, manifest_path)
    write_checksums(
        {
            arrays_path.name: arrays_path,
            table_path.name: table_path,
            manifest_path.name: manifest_path,
        },
        checksums_path,
    )

    print(f"Vectores -> {arrays_path}")
    print(f"Tabla -> {table_path}")
    print(f"Forma: {dataset.features.shape}; no finitos: {dataset.audit.non_finite_values}")
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
