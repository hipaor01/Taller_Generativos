"""Descarga reproducible de velas spot desde la API publica de Binance."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .artifacts import write_json_atomic


INTERVAL_MILLISECONDS: Mapping[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

OUTPUT_COLUMNS = (
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "exchange",
    "market_type",
    "symbol",
    "base_asset",
    "quote_asset",
    "interval",
)


def _iso_utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class DatasetAudit:
    """Controles basicos sobre un fichero de velas descargado."""

    observed_rows: int
    expected_rows_between_first_and_last: int
    first_open_time_utc: Optional[str]
    last_open_time_utc: Optional[str]
    duplicate_open_times: int
    missing_candles: int
    non_positive_ohlc_rows: int
    negative_volume_rows: int


@dataclass(frozen=True)
class KlineDownload:
    """Resultado y procedencia de una descarga."""

    source: str
    exchange: str
    market_type: str
    symbol: str
    base_asset: str
    quote_asset: str
    interval: str
    requested_start_utc: str
    requested_end_exclusive_utc: str
    downloaded_at_utc: str
    csv_path: str
    audit: DatasetAudit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BinanceKlineClient:
    """Cliente paginado para el endpoint publico ``/api/v3/klines``."""

    def __init__(
        self,
        base_url: str = "https://data-api.binance.vision",
        timeout_seconds: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_exclusive_ms: int,
    ) -> List[List[Any]]:
        """Obtiene todas las velas cuyo tiempo de apertura cae en ``[start, end)``."""
        interval_ms = self._interval_ms(interval)
        if end_exclusive_ms <= start_ms:
            raise ValueError("La fecha final debe ser posterior a la inicial")

        cursor = start_ms
        rows: List[List[Any]] = []
        while cursor < end_exclusive_ms:
            batch = self._request_json(
                "/api/v3/klines",
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_exclusive_ms - 1,
                    "limit": 1000,
                },
            )
            if not isinstance(batch, list):
                raise RuntimeError(f"Respuesta inesperada de Binance: {batch!r}")
            if not batch:
                break

            valid_batch = [row for row in batch if int(row[0]) < end_exclusive_ms]
            rows.extend(valid_batch)
            next_cursor = int(batch[-1][0]) + interval_ms
            if next_cursor <= cursor:
                raise RuntimeError("El paginado de Binance no avanzo")
            cursor = next_cursor

            if len(batch) < 1000:
                break

        return rows

    def write_dataset(
        self,
        rows: Sequence[Sequence[Any]],
        output_path: Path,
        *,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        interval: str,
    ) -> DatasetAudit:
        """Escribe un CSV atomico con procedencia explicita y devuelve su auditoria."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for raw_row in rows:
                row = dict(zip(KLINE_COLUMNS, raw_row))
                writer.writerow(
                    {
                        "open_time_utc": _iso_utc(int(row["open_time"])),
                        "close_time_utc": _iso_utc(int(row["close_time"])),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "quote_asset_volume": row["quote_asset_volume"],
                        "number_of_trades": row["number_of_trades"],
                        "taker_buy_base_volume": row["taker_buy_base_volume"],
                        "taker_buy_quote_volume": row["taker_buy_quote_volume"],
                        "exchange": "Binance",
                        "market_type": "spot",
                        "symbol": symbol.upper(),
                        "base_asset": base_asset.upper(),
                        "quote_asset": quote_asset.upper(),
                        "interval": interval,
                    }
                )
        temporary_path.replace(output_path)
        return self.audit(rows, interval)

    def audit(self, rows: Sequence[Sequence[Any]], interval: str) -> DatasetAudit:
        interval_ms = self._interval_ms(interval)
        if not rows:
            return DatasetAudit(0, 0, None, None, 0, 0, 0, 0)

        open_times = [int(row[0]) for row in rows]
        unique_times = set(open_times)
        first, last = min(open_times), max(open_times)
        expected = ((last - first) // interval_ms) + 1
        non_positive_ohlc = sum(
            1
            for row in rows
            if any((not math.isfinite(float(row[index]))) or float(row[index]) <= 0 for index in range(1, 5))
        )
        negative_volume = sum(
            1
            for row in rows
            if (not math.isfinite(float(row[5]))) or float(row[5]) < 0
        )
        return DatasetAudit(
            observed_rows=len(rows),
            expected_rows_between_first_and_last=expected,
            first_open_time_utc=_iso_utc(first),
            last_open_time_utc=_iso_utc(last),
            duplicate_open_times=len(open_times) - len(unique_times),
            missing_candles=expected - len(unique_times),
            non_positive_ohlc_rows=non_positive_ohlc,
            negative_volume_rows=negative_volume,
        )

    def _request_json(self, path: str, params: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = Request(url, headers={"User-Agent": "crypto-generative-data/0.1"})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                retryable = error.code in {418, 429} or 500 <= error.code < 600
                if not retryable or attempt == self.max_attempts:
                    body = error.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Binance respondio HTTP {error.code}: {body}") from error
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2 ** (attempt - 1), 30)
                time.sleep(delay)
            except URLError as error:
                if attempt == self.max_attempts:
                    raise RuntimeError(f"No se pudo conectar con Binance: {error.reason}") from error
                time.sleep(min(2 ** (attempt - 1), 30))
        raise AssertionError("Bucle de reintentos agotado sin devolver ni lanzar excepcion")

    @staticmethod
    def _interval_ms(interval: str) -> int:
        try:
            return INTERVAL_MILLISECONDS[interval]
        except KeyError as error:
            supported = ", ".join(INTERVAL_MILLISECONDS)
            raise ValueError(f"Intervalo no soportado: {interval}. Opciones: {supported}") from error


def write_manifest(downloads: Sequence[KlineDownload], path: Path) -> None:
    """Guarda metadatos comunes de descarga en JSON."""
    payload = {
        "dataset": "Binance spot BTC-ETH",
        "price_currency_note": (
            "Los mercados estan cotizados en USDT. USDT se usa como proxy operativo de USD, "
            "pero no es USD fiat."
        ),
        "downloads": [download.to_dict() for download in downloads],
    }
    write_json_atomic(payload, path)
