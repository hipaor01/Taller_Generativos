#!/usr/bin/env python3
"""Descarga el panel bruto BTCUSDT-ETHUSDT de Binance en velas de 6 horas."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from crypto_generative.data.binance import (  # noqa: E402
    INTERVAL_MILLISECONDS,
    BinanceKlineClient,
    KlineDownload,
    write_manifest,
)


MARKETS = {
    "BTCUSDT": ("BTC", "USDT"),
    "ETHUSDT": ("ETH", "USDT"),
}


def parse_utc(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def floor_to_interval(value: datetime, interval: str) -> datetime:
    interval_ms = INTERVAL_MILLISECONDS[interval]
    floored_ms = milliseconds(value) // interval_ms * interval_ms
    return datetime.fromtimestamp(floored_ms / 1000, tz=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-01", help="Inicio UTC inclusivo (ISO 8601)")
    parser.add_argument(
        "--end",
        help="Fin UTC exclusivo (ISO 8601). Por defecto, inicio de la vela actual.",
    )
    parser.add_argument("--interval", default="6h", choices=INTERVAL_MILLISECONDS)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(MARKETS),
        choices=MARKETS,
        help="Mercados spot de Binance global",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "binance",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end) if args.end else floor_to_interval(datetime.now(timezone.utc), args.interval)
    if end <= start:
        raise SystemExit("--end debe ser posterior a --start")

    client = BinanceKlineClient()
    downloaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    downloads = []
    for symbol in args.symbols:
        base_asset, quote_asset = MARKETS[symbol]
        print(f"Descargando {symbol} {args.interval}...", flush=True)
        rows = client.fetch_klines(symbol, args.interval, milliseconds(start), milliseconds(end))
        if not rows:
            raise SystemExit(f"Binance no devolvio velas para {symbol}")

        output_path = args.output_dir / f"{symbol.lower()}_{args.interval}.csv"
        audit = client.write_dataset(
            rows,
            output_path,
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            interval=args.interval,
        )
        downloads.append(
            KlineDownload(
                source=f"{client.base_url}/api/v3/klines",
                exchange="Binance",
                market_type="spot",
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                interval=args.interval,
                requested_start_utc=start.isoformat().replace("+00:00", "Z"),
                requested_end_exclusive_utc=end.isoformat().replace("+00:00", "Z"),
                downloaded_at_utc=downloaded_at,
                csv_path=str(output_path.relative_to(PROJECT_ROOT)),
                audit=audit,
            )
        )
        print(
            f"  {audit.observed_rows} velas; {audit.missing_candles} huecos; "
            f"{audit.duplicate_open_times} duplicados -> {output_path}",
            flush=True,
        )

    manifest_path = args.output_dir / f"manifest_{args.interval}.json"
    write_manifest(downloads, manifest_path)
    print(f"Metadatos -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
