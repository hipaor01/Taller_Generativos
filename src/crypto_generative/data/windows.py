"""Ventanas temporales conjuntas para condicion y objetivo generativo."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .artifacts import write_csv_atomic
from .binance import INTERVAL_MILLISECONDS


REQUIRED_RETURN_COLUMNS = (
    "open_time_utc",
    "btc_log_return",
    "eth_log_return",
    "returns_valid",
    "invalid_reason",
)

INDEX_COLUMNS = (
    "sample_id",
    "condition_start_utc",
    "condition_end_utc",
    "target_start_utc",
    "target_end_utc",
)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp sin zona horaria: {value}")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class WindowAudit:
    interval: str
    assets: Sequence[str]
    condition_steps: int
    target_steps: int
    stride_steps: int
    total_return_rows: int
    candidate_windows: int
    valid_windows: int
    rejected_windows: int
    rejected_condition_only: int
    rejected_target_only: int
    rejected_condition_and_target: int
    first_target_start_utc: Optional[str]
    last_target_start_utc: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowDataset:
    condition_returns: NDArray[np.float64]
    target_returns: NDArray[np.float64]
    start_indices: NDArray[np.int64]
    index_rows: Sequence[Mapping[str, Any]]
    audit: WindowAudit


class TemporalWindowBuilder:
    """Construye ejemplos deslizantes y rechaza toda discontinuidad."""

    assets = ("BTC", "ETH")

    def __init__(
        self,
        condition_steps: int = 240,
        target_steps: int = 120,
        stride_steps: int = 1,
        interval: str = "6h",
    ) -> None:
        if interval not in INTERVAL_MILLISECONDS:
            raise ValueError(f"Intervalo no soportado: {interval}")
        if condition_steps <= 0 or target_steps <= 0 or stride_steps <= 0:
            raise ValueError("Las longitudes y el desplazamiento deben ser positivos")
        self.condition_steps = condition_steps
        self.target_steps = target_steps
        self.stride_steps = stride_steps
        self.interval = interval
        self.interval_delta = timedelta(milliseconds=INTERVAL_MILLISECONDS[interval])

    def build(self, returns_path: Path) -> WindowDataset:
        timestamps, values, valid = self._read_returns(returns_path)
        total_steps = self.condition_steps + self.target_steps
        if len(timestamps) < total_steps:
            raise ValueError(
                f"Se necesitan al menos {total_steps} retornos y solo hay {len(timestamps)}"
            )

        candidate_starts = np.arange(
            0,
            len(timestamps) - total_steps + 1,
            self.stride_steps,
            dtype=np.int64,
        )
        accepted_starts: List[int] = []
        rejected_condition_only = 0
        rejected_target_only = 0
        rejected_both = 0
        for raw_start in candidate_starts:
            start = int(raw_start)
            target_start = start + self.condition_steps
            end = target_start + self.target_steps
            condition_valid = bool(valid[start:target_start].all())
            target_valid = bool(valid[target_start:end].all())
            if condition_valid and target_valid:
                accepted_starts.append(start)
            elif not condition_valid and target_valid:
                rejected_condition_only += 1
            elif condition_valid and not target_valid:
                rejected_target_only += 1
            else:
                rejected_both += 1

        starts = np.asarray(accepted_starts, dtype=np.int64)
        condition = np.empty((len(starts), self.condition_steps, 2), dtype=np.float64)
        target = np.empty((len(starts), self.target_steps, 2), dtype=np.float64)
        index_rows: List[Mapping[str, Any]] = []
        for sample_id, raw_start in enumerate(starts):
            start = int(raw_start)
            target_start = start + self.condition_steps
            end = target_start + self.target_steps
            condition[sample_id] = values[start:target_start]
            target[sample_id] = values[target_start:end]
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "condition_start_utc": timestamps[start],
                    "condition_end_utc": timestamps[target_start - 1],
                    "target_start_utc": timestamps[target_start],
                    "target_end_utc": timestamps[end - 1],
                }
            )

        if not len(starts):
            raise ValueError("Ninguna ventana cumple los controles de calidad")
        audit = WindowAudit(
            interval=self.interval,
            assets=self.assets,
            condition_steps=self.condition_steps,
            target_steps=self.target_steps,
            stride_steps=self.stride_steps,
            total_return_rows=len(timestamps),
            candidate_windows=len(candidate_starts),
            valid_windows=len(starts),
            rejected_windows=len(candidate_starts) - len(starts),
            rejected_condition_only=rejected_condition_only,
            rejected_target_only=rejected_target_only,
            rejected_condition_and_target=rejected_both,
            first_target_start_utc=str(index_rows[0]["target_start_utc"]),
            last_target_start_utc=str(index_rows[-1]["target_start_utc"]),
        )
        return WindowDataset(condition, target, starts, index_rows, audit)

    def write_npz(self, dataset: WindowDataset, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                condition_returns=dataset.condition_returns,
                target_returns=dataset.target_returns,
                start_indices=dataset.start_indices,
                assets=np.asarray(self.assets),
            )
        temporary_path.replace(path)

    @staticmethod
    def write_index(dataset: WindowDataset, path: Path) -> None:
        write_csv_atomic(dataset.index_rows, INDEX_COLUMNS, path)

    def _read_returns(
        self, returns_path: Path
    ) -> Tuple[List[str], NDArray[np.float64], NDArray[np.bool_]]:
        timestamps: List[str] = []
        values: List[Tuple[float, float]] = []
        validity: List[bool] = []
        previous_time: Optional[datetime] = None
        with returns_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = set(REQUIRED_RETURN_COLUMNS) - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"{returns_path} no contiene columnas requeridas: {sorted(missing_columns)}"
                )
            for line_number, row in enumerate(reader, start=2):
                timestamp = _parse_utc(row["open_time_utc"])
                if previous_time is not None and timestamp - previous_time != self.interval_delta:
                    raise ValueError(
                        f"Calendario no regular en {returns_path}:{line_number}: "
                        f"{row['open_time_utc']}"
                    )
                is_valid = row["returns_valid"] == "1"
                if row["returns_valid"] not in {"0", "1"}:
                    raise ValueError(f"returns_valid invalido en {returns_path}:{line_number}")
                if is_valid:
                    try:
                        pair = (float(row["btc_log_return"]), float(row["eth_log_return"]))
                    except ValueError as error:
                        raise ValueError(
                            f"Retorno invalido en {returns_path}:{line_number}"
                        ) from error
                    if not all(np.isfinite(pair)):
                        raise ValueError(f"Retorno no finito en {returns_path}:{line_number}")
                    if row["invalid_reason"]:
                        raise ValueError(
                            f"Retorno valido con invalid_reason en {returns_path}:{line_number}"
                        )
                else:
                    if row["btc_log_return"] or row["eth_log_return"]:
                        raise ValueError(
                            f"Retorno no valido con valores en {returns_path}:{line_number}"
                        )
                    if not row["invalid_reason"]:
                        raise ValueError(
                            f"Retorno no valido sin motivo en {returns_path}:{line_number}"
                        )
                    pair = (np.nan, np.nan)
                timestamps.append(row["open_time_utc"])
                values.append(pair)
                validity.append(is_valid)
                previous_time = timestamp

        if not timestamps:
            raise ValueError(f"El fichero de retornos esta vacio: {returns_path}")
        return (
            timestamps,
            np.asarray(values, dtype=np.float64),
            np.asarray(validity, dtype=np.bool_),
        )

