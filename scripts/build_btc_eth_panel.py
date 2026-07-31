#!/usr/bin/env python3
"""Valida y construye el panel regular BTC-ETH de seis horas."""

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
from crypto_generative.data.panel import AssetInput, BtcEthPanelBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--btc",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "binance" / "btcusdt_6h.csv",
    )
    parser.add_argument(
        "--eth",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "binance" / "ethusdt_6h.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "binance",
    )
    parser.add_argument("--interval", default="6h")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assets = (
        AssetInput("btc", "BTCUSDT", "BTC", "USDT", args.btc),
        AssetInput("eth", "ETHUSDT", "ETH", "USDT", args.eth),
    )
    builder = BtcEthPanelBuilder(args.interval)
    rows, audit = builder.build(assets)

    panel_path = args.output_dir / f"btc_eth_{args.interval}_panel.csv"
    manifest_path = args.output_dir / f"manifest_{args.interval}.json"
    checksums_path = args.output_dir / "SHA256SUMS"
    builder.write(rows, panel_path)

    manifest = {
        "dataset": "Panel limpio Binance spot BTC-ETH",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": {
            "calendar": "Rejilla UTC regular con frecuencia de 6 horas",
            "alignment": "Cobertura comun BTC-ETH",
            "missing_values": "Sin imputacion; campos numericos vacios e indicadores *_missing=1",
            "truncated_candles": (
                "Se conservan valores y cierre real en *_source_close_time_utc; "
                "se marcan con *_duration_valid=0 y no se imputan"
            ),
            "downstream_rule": "Excluir ventanas que contengan filas con is_complete=0",
            "price_currency_note": "USDT es la divisa cotizada; no es USD fiat",
        },
        "inputs": [
            {
                "symbol": asset.symbol,
                "path": relative_or_absolute(asset.path, PROJECT_ROOT),
                "sha256": sha256_file(asset.path),
            }
            for asset in assets
        ],
        "output": {
            "path": relative_or_absolute(panel_path, PROJECT_ROOT),
            "sha256": sha256_file(panel_path),
        },
        "audit": audit.to_dict(),
    }
    write_json_atomic(manifest, manifest_path)
    write_checksums(
        {panel_path.name: panel_path, manifest_path.name: manifest_path},
        checksums_path,
    )

    print(f"Panel -> {panel_path}")
    print(
        f"{audit.calendar_rows} timestamps: {audit.complete_rows} completos, "
        f"{audit.incomplete_rows} no aptos para ventanas"
    )
    print(
        f"Huecos comunes: {audit.common_missing_rows}; "
        f"huecos de un solo activo: {audit.one_sided_missing_rows}"
    )
    print(
        f"Velas truncadas comunes: {audit.common_invalid_duration_rows}; "
        f"truncadas de un solo activo: {audit.one_sided_invalid_duration_rows}"
    )
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
