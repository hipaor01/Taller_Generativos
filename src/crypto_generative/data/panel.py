"""Construccion y auditoria del panel temporal conjunto BTC-ETH."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .binance import INTERVAL_MILLISECONDS
from .artifacts import write_csv_atomic


PRICE_COLUMNS = ("open", "high", "low", "close")
VOLUME_COLUMNS = (
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
VALUE_COLUMNS = PRICE_COLUMNS + VOLUME_COLUMNS + ("number_of_trades",)
REQUIRED_COLUMNS = (
    "open_time_utc",
    "close_time_utc",
    *VALUE_COLUMNS,
    "exchange",
    "market_type",
    "symbol",
    "base_asset",
    "quote_asset",
    "interval",
)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp sin zona horaria: {value}")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AssetInput:
    prefix: str
    symbol: str
    base_asset: str
    quote_asset: str
    path: Path


@dataclass(frozen=True)
class PanelAudit:
    interval: str
    first_open_time_utc: str
    last_open_time_utc: str
    calendar_rows: int
    complete_rows: int
    incomplete_rows: int
    missing_rows_by_asset: Mapping[str, int]
    common_missing_rows: int
    one_sided_missing_rows: int
    gap_open_times_utc: Sequence[str]
    invalid_duration_rows_by_asset: Mapping[str, int]
    common_invalid_duration_rows: int
    one_sided_invalid_duration_rows: int
    invalid_duration_open_times_utc: Sequence[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BtcEthPanelBuilder:
    """Valida, alinea y regulariza dos series OHLCV sin imputarlas."""

    def __init__(self, interval: str = "6h") -> None:
        if interval not in INTERVAL_MILLISECONDS:
            raise ValueError(f"Intervalo no soportado: {interval}")
        self.interval = interval
        self.interval_delta = timedelta(milliseconds=INTERVAL_MILLISECONDS[interval])

    def build(
        self, assets: Sequence[AssetInput]
    ) -> Tuple[List[Dict[str, str]], PanelAudit]:
        if len(assets) != 2:
            raise ValueError("El panel BTC-ETH requiere exactamente dos activos")
        if len({asset.prefix for asset in assets}) != len(assets):
            raise ValueError("Los prefijos de los activos deben ser unicos")

        series = {asset.prefix: self._read_asset(asset) for asset in assets}
        first = max(min(rows) for rows in series.values())
        last = min(max(rows) for rows in series.values())
        if first > last:
            raise ValueError("Los activos no tienen cobertura temporal comun")

        rows: List[Dict[str, str]] = []
        missing_by_asset = {asset.prefix: 0 for asset in assets}
        invalid_duration_by_asset = {asset.prefix: 0 for asset in assets}
        gaps: List[str] = []
        invalid_duration_times: List[str] = []
        common_missing = 0
        one_sided_missing = 0
        common_invalid_duration = 0
        one_sided_invalid_duration = 0
        timestamp = first
        while timestamp <= last:
            missing = {
                asset.prefix: timestamp not in series[asset.prefix] for asset in assets
            }
            missing_count = sum(missing.values())
            if missing_count:
                gaps.append(_format_utc(timestamp))
                for prefix, is_missing in missing.items():
                    missing_by_asset[prefix] += int(is_missing)
                common_missing += int(missing_count == len(assets))
                one_sided_missing += int(missing_count == 1)

            invalid_duration = {
                asset.prefix: (
                    not missing[asset.prefix]
                    and series[asset.prefix][timestamp]["_duration_valid"] == "0"
                )
                for asset in assets
            }
            invalid_duration_count = sum(invalid_duration.values())
            if invalid_duration_count:
                invalid_duration_times.append(_format_utc(timestamp))
                for prefix, is_invalid in invalid_duration.items():
                    invalid_duration_by_asset[prefix] += int(is_invalid)
                common_invalid_duration += int(invalid_duration_count == len(assets))
                one_sided_invalid_duration += int(invalid_duration_count == 1)

            output: Dict[str, str] = {
                "open_time_utc": _format_utc(timestamp),
                "expected_close_time_utc": _format_utc(
                    timestamp + self.interval_delta - timedelta(milliseconds=1)
                ),
            }
            for asset in assets:
                source = series[asset.prefix].get(timestamp)
                output[f"{asset.prefix}_source_close_time_utc"] = (
                    "" if source is None else source["close_time_utc"]
                )
                for column in VALUE_COLUMNS:
                    output[f"{asset.prefix}_{column}"] = "" if source is None else source[column]
                output[f"{asset.prefix}_missing"] = "1" if source is None else "0"
                output[f"{asset.prefix}_duration_valid"] = (
                    "0" if source is None else source["_duration_valid"]
                )
            output["is_complete"] = (
                "1" if missing_count == 0 and invalid_duration_count == 0 else "0"
            )
            rows.append(output)
            timestamp += self.interval_delta

        complete_rows = sum(row["is_complete"] == "1" for row in rows)
        audit = PanelAudit(
            interval=self.interval,
            first_open_time_utc=_format_utc(first),
            last_open_time_utc=_format_utc(last),
            calendar_rows=len(rows),
            complete_rows=complete_rows,
            incomplete_rows=len(rows) - complete_rows,
            missing_rows_by_asset=missing_by_asset,
            common_missing_rows=common_missing,
            one_sided_missing_rows=one_sided_missing,
            gap_open_times_utc=gaps,
            invalid_duration_rows_by_asset=invalid_duration_by_asset,
            common_invalid_duration_rows=common_invalid_duration,
            one_sided_invalid_duration_rows=one_sided_invalid_duration,
            invalid_duration_open_times_utc=invalid_duration_times,
        )
        return rows, audit

    def write(self, rows: Sequence[Mapping[str, str]], path: Path) -> None:
        if not rows:
            raise ValueError("No hay filas que escribir")
        write_csv_atomic(rows, list(rows[0]), path)

    def _read_asset(self, asset: AssetInput) -> Dict[datetime, Dict[str, str]]:
        rows: Dict[datetime, Dict[str, str]] = {}
        with asset.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"{asset.path} no contiene columnas requeridas: {sorted(missing_columns)}"
                )
            for line_number, row in enumerate(reader, start=2):
                duration_valid = self._validate_row(row, asset, line_number)
                timestamp = _parse_utc(row["open_time_utc"])
                if timestamp in rows:
                    raise ValueError(
                        f"Timestamp duplicado en {asset.path}:{line_number}: {_format_utc(timestamp)}"
                    )
                row["_duration_valid"] = "1" if duration_valid else "0"
                rows[timestamp] = row
        if not rows:
            raise ValueError(f"El fichero esta vacio: {asset.path}")
        return rows

    def _validate_row(
        self, row: Mapping[str, str], asset: AssetInput, line_number: int
    ) -> bool:
        location = f"{asset.path}:{line_number}"
        expected_metadata = {
            "exchange": "Binance",
            "market_type": "spot",
            "symbol": asset.symbol,
            "base_asset": asset.base_asset,
            "quote_asset": asset.quote_asset,
            "interval": self.interval,
        }
        for column, expected in expected_metadata.items():
            if row[column] != expected:
                raise ValueError(
                    f"Metadato invalido en {location}: {column}={row[column]!r}, esperado {expected!r}"
                )

        try:
            values = {column: Decimal(row[column]) for column in PRICE_COLUMNS + VOLUME_COLUMNS}
            trades = int(row["number_of_trades"])
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Valor numerico invalido en {location}") from error
        if any(values[column] <= 0 for column in PRICE_COLUMNS):
            raise ValueError(f"Precio no positivo en {location}")
        if any(values[column] < 0 for column in VOLUME_COLUMNS) or trades < 0:
            raise ValueError(f"Volumen o numero de operaciones negativo en {location}")
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError(f"Maximo OHLC inconsistente en {location}")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError(f"Minimo OHLC inconsistente en {location}")

        open_time = _parse_utc(row["open_time_utc"])
        close_time = _parse_utc(row["close_time_utc"])
        expected_close = open_time + self.interval_delta - timedelta(milliseconds=1)
        interval_ms = INTERVAL_MILLISECONDS[self.interval]
        if int(open_time.timestamp() * 1000) % interval_ms:
            raise ValueError(f"Timestamp fuera de la rejilla {self.interval} en {location}")
        return close_time == expected_close
