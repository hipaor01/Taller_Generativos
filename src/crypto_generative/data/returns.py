"""Calculo reproducible de retornos logaritmicos conjuntos BTC-ETH."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .binance import INTERVAL_MILLISECONDS
from .artifacts import write_csv_atomic


REQUIRED_PANEL_COLUMNS = (
    "open_time_utc",
    "btc_close",
    "eth_close",
    "btc_missing",
    "btc_duration_valid",
    "eth_missing",
    "eth_duration_valid",
    "is_complete",
)

RETURN_COLUMNS = (
    "open_time_utc",
    "btc_log_return",
    "eth_log_return",
    "returns_valid",
    "invalid_reason",
)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp sin zona horaria: {value}")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReturnAudit:
    interval: str
    total_rows: int
    valid_return_rows: int
    invalid_return_rows: int
    first_valid_return_utc: Optional[str]
    last_valid_return_utc: Optional[str]
    invalid_reason_counts: Mapping[str, int]
    non_finite_returns: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LogReturnBuilder:
    """Deriva retornos sin saltar discontinuidades del panel regular."""

    def __init__(self, interval: str = "6h") -> None:
        if interval not in INTERVAL_MILLISECONDS:
            raise ValueError(f"Intervalo no soportado: {interval}")
        self.interval = interval
        self.interval_delta = timedelta(milliseconds=INTERVAL_MILLISECONDS[interval])

    def build(self, panel_path: Path) -> Tuple[List[Dict[str, str]], ReturnAudit]:
        panel_rows = self._read_panel(panel_path)
        output_rows: List[Dict[str, str]] = []
        reason_counts: Counter[str] = Counter()
        previous: Optional[Mapping[str, str]] = None
        valid_timestamps: List[str] = []
        non_finite_returns = 0

        for current in panel_rows:
            reasons: List[str] = []
            if previous is None:
                reasons.append("first_observation")
            else:
                if current["is_complete"] != "1":
                    reasons.append("current_candle_incomplete")
                if previous["is_complete"] != "1":
                    reasons.append("previous_candle_incomplete")

            if reasons:
                reason = "|".join(reasons)
                reason_counts[reason] += 1
                output = {
                    "open_time_utc": current["open_time_utc"],
                    "btc_log_return": "",
                    "eth_log_return": "",
                    "returns_valid": "0",
                    "invalid_reason": reason,
                }
            else:
                assert previous is not None
                btc_return = self._log_return(
                    current["btc_close"], previous["btc_close"], "BTC", current["open_time_utc"]
                )
                eth_return = self._log_return(
                    current["eth_close"], previous["eth_close"], "ETH", current["open_time_utc"]
                )
                non_finite_returns += int(not math.isfinite(btc_return))
                non_finite_returns += int(not math.isfinite(eth_return))
                output = {
                    "open_time_utc": current["open_time_utc"],
                    "btc_log_return": format(btc_return, ".17g"),
                    "eth_log_return": format(eth_return, ".17g"),
                    "returns_valid": "1",
                    "invalid_reason": "",
                }
                valid_timestamps.append(current["open_time_utc"])

            output_rows.append(output)
            previous = current

        audit = ReturnAudit(
            interval=self.interval,
            total_rows=len(output_rows),
            valid_return_rows=len(valid_timestamps),
            invalid_return_rows=len(output_rows) - len(valid_timestamps),
            first_valid_return_utc=valid_timestamps[0] if valid_timestamps else None,
            last_valid_return_utc=valid_timestamps[-1] if valid_timestamps else None,
            invalid_reason_counts=dict(sorted(reason_counts.items())),
            non_finite_returns=non_finite_returns,
        )
        return output_rows, audit

    def write(self, rows: Sequence[Mapping[str, str]], path: Path) -> None:
        if not rows:
            raise ValueError("No hay retornos que escribir")
        write_csv_atomic(rows, RETURN_COLUMNS, path)

    def _read_panel(self, panel_path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        previous_time: Optional[datetime] = None
        with panel_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = set(REQUIRED_PANEL_COLUMNS) - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"{panel_path} no contiene columnas requeridas: {sorted(missing_columns)}"
                )
            for line_number, row in enumerate(reader, start=2):
                timestamp = _parse_utc(row["open_time_utc"])
                if previous_time is not None and timestamp - previous_time != self.interval_delta:
                    raise ValueError(
                        f"Calendario no regular en {panel_path}:{line_number}: "
                        f"{row['open_time_utc']}"
                    )
                self._validate_row(row, panel_path, line_number)
                rows.append(row)
                previous_time = timestamp
        if not rows:
            raise ValueError(f"El panel esta vacio: {panel_path}")
        return rows

    @staticmethod
    def _validate_row(row: Mapping[str, str], path: Path, line_number: int) -> None:
        location = f"{path}:{line_number}"
        flag_columns = (
            "btc_missing",
            "btc_duration_valid",
            "eth_missing",
            "eth_duration_valid",
            "is_complete",
        )
        if any(row[column] not in {"0", "1"} for column in flag_columns):
            raise ValueError(f"Indicador binario invalido en {location}")

        expected_complete = all(
            (
                row["btc_missing"] == "0",
                row["btc_duration_valid"] == "1",
                row["eth_missing"] == "0",
                row["eth_duration_valid"] == "1",
            )
        )
        if (row["is_complete"] == "1") != expected_complete:
            raise ValueError(f"Indicador is_complete inconsistente en {location}")

        if expected_complete:
            for asset in ("btc", "eth"):
                try:
                    close = Decimal(row[f"{asset}_close"])
                except InvalidOperation as error:
                    raise ValueError(f"Cierre {asset.upper()} invalido en {location}") from error
                if not close.is_finite() or close <= 0:
                    raise ValueError(f"Cierre {asset.upper()} no positivo o no finito en {location}")

    @staticmethod
    def _log_return(current: str, previous: str, asset: str, timestamp: str) -> float:
        try:
            current_close = Decimal(current)
            previous_close = Decimal(previous)
        except InvalidOperation as error:
            raise ValueError(f"Cierre {asset} invalido en {timestamp}") from error
        if (
            not current_close.is_finite()
            or not previous_close.is_finite()
            or current_close <= 0
            or previous_close <= 0
        ):
            raise ValueError(f"Cierre {asset} no positivo o no finito en {timestamp}")
        return math.log(float(current_close / previous_close))
