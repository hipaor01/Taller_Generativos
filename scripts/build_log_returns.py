#!/usr/bin/env python3
"""Calcula retornos logaritmicos de seis horas para el panel BTC-ETH."""

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
from crypto_generative.data.returns import LogReturnBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "binance"
            / "btc_eth_6h_panel.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "binance",
    )
    parser.add_argument("--interval", default="6h")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    builder = LogReturnBuilder(args.interval)
    rows, audit = builder.build(args.panel)

    returns_path = args.output_dir / f"btc_eth_{args.interval}_log_returns.csv"
    manifest_path = args.output_dir / f"manifest_{args.interval}.json"
    checksums_path = args.output_dir / "SHA256SUMS"
    builder.write(rows, returns_path)

    manifest = {
        "dataset": "Retornos logaritmicos conjuntos BTC-ETH",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "definition": "r_t = log(close_t / close_(t-1))",
        "validity_rule": (
            "El retorno solo es valido si las velas actual y anterior tienen is_complete=1"
        ),
        "input": {
            "path": relative_or_absolute(args.panel, PROJECT_ROOT),
            "sha256": sha256_file(args.panel),
        },
        "output": {
            "path": relative_or_absolute(returns_path, PROJECT_ROOT),
            "sha256": sha256_file(returns_path),
        },
        "audit": audit.to_dict(),
    }
    write_json_atomic(manifest, manifest_path)
    write_checksums(
        {returns_path.name: returns_path, manifest_path.name: manifest_path},
        checksums_path,
    )

    print(f"Retornos -> {returns_path}")
    print(
        f"{audit.total_rows} timestamps: {audit.valid_return_rows} retornos validos, "
        f"{audit.invalid_return_rows} no validos"
    )
    print(f"Retornos no finitos: {audit.non_finite_returns}")
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
