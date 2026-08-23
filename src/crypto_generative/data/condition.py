"""Resumen del estado de mercado previo a cada trayectoria objetivo."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .artifacts import write_csv_atomic
from .binance import INTERVAL_MILLISECONDS


PANEL_FEATURE_COLUMNS = (
    "open_time_utc",
    "btc_high",
    "btc_low",
    "btc_volume",
    "eth_high",
    "eth_low",
    "eth_volume",
    "is_complete",
)

PANEL_CLOSE_COLUMNS = ("btc_close", "eth_close")

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
class ConditionAudit:
    samples: int
    features: int
    condition_steps: int
    condition_days: float
    volume_recent_steps: int
    volume_recent_days: float
    correlation_steps: int
    correlation_days: float
    annualization_steps: int
    non_finite_values: int
    first_target_start_utc: str
    last_target_start_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionDataset:
    features: NDArray[np.float64]
    feature_names: Sequence[str]
    sample_ids: NDArray[np.int64]
    table_rows: Sequence[Mapping[str, Any]]
    audit: ConditionAudit


@dataclass(frozen=True)
class MarketConditionSnapshot:
    """Estado de mercado calculado al final del último bloque completo."""

    features: NDArray[np.float64]
    feature_names: Sequence[str]
    condition_returns: NDArray[np.float64]
    initial_prices: NDArray[np.float64]
    assets: Tuple[str, str]
    condition_start_utc: str
    condition_end_utc: str
    forecast_start_utc: str
    condition_steps: int


class ConditionFeatureBuilder:
    """Convierte cada historia de retornos y mercado en un vector compacto."""

    assets = ("btc", "eth")

    def __init__(
        self,
        interval: str = "6h",
        volume_recent_steps: int = 28,
        correlation_steps: int = 120,
        annualization_days: int = 365,
    ) -> None:
        if interval not in INTERVAL_MILLISECONDS:
            raise ValueError(f"Intervalo no soportado: {interval}")
        interval_ms = INTERVAL_MILLISECONDS[interval]
        day_ms = 24 * 60 * 60 * 1000
        if day_ms % interval_ms:
            raise ValueError(f"El intervalo {interval} no divide dias completos")
        if volume_recent_steps <= 0 or correlation_steps <= 1:
            raise ValueError("Los horizontes de resumen deben ser positivos")
        self.interval = interval
        self.interval_delta = timedelta(milliseconds=interval_ms)
        self.steps_per_day = day_ms // interval_ms
        self.volume_recent_steps = volume_recent_steps
        self.correlation_steps = correlation_steps
        self.annualization_steps = annualization_days * self.steps_per_day

    def build(
        self,
        windows_path: Path,
        window_index_path: Path,
        panel_path: Path,
    ) -> ConditionDataset:
        condition_returns, target_steps, starts = self._read_windows(windows_path)
        index_rows = self._read_index(window_index_path)
        panel = self._read_panel(panel_path)
        samples, condition_steps, assets = condition_returns.shape
        if assets != len(self.assets):
            raise ValueError(f"Se esperaban dos activos y se encontraron {assets}")
        if samples != len(starts) or samples != len(index_rows):
            raise ValueError("NPZ e indice no contienen el mismo numero de muestras")
        if self.correlation_steps > condition_steps:
            raise ValueError("El horizonte de correlacion supera la condicion")
        if 2 * self.volume_recent_steps > condition_steps:
            raise ValueError("Se necesitan dos bloques recientes de volumen dentro de la condicion")

        feature_names = self._feature_names(condition_steps)
        feature_matrix = np.empty((samples, len(feature_names)), dtype=np.float64)
        table_rows: List[Mapping[str, Any]] = []
        for sample_id, raw_start in enumerate(starts):
            start = int(raw_start)
            condition_end = start + condition_steps
            target_end = condition_end + target_steps
            if target_end > len(panel["timestamps"]):
                raise ValueError(f"La muestra {sample_id} excede la cobertura del panel")
            panel_valid = panel["complete"][start:condition_end]
            if not bool(panel_valid.all()):
                raise ValueError(f"La condicion de la muestra {sample_id} contiene velas incompletas")

            expected_boundaries = (
                panel["timestamps"][start],
                panel["timestamps"][condition_end - 1],
                panel["timestamps"][condition_end],
                panel["timestamps"][target_end - 1],
            )
            observed_boundaries = tuple(index_rows[sample_id][column] for column in INDEX_COLUMNS[1:])
            if expected_boundaries != observed_boundaries:
                raise ValueError(f"Fronteras temporales inconsistentes en la muestra {sample_id}")

            vector = self._summarize(
                condition_returns[sample_id],
                panel["high"][start:condition_end],
                panel["low"][start:condition_end],
                panel["volume"][start:condition_end],
            )
            if not np.isfinite(vector).all():
                raise ValueError(f"Vector de condicion no finito en la muestra {sample_id}")
            feature_matrix[sample_id] = vector
            row: Dict[str, Any] = dict(index_rows[sample_id])
            row.update(
                {name: format(float(value), ".17g") for name, value in zip(feature_names, vector)}
            )
            table_rows.append(row)

        condition_days = condition_steps / self.steps_per_day
        audit = ConditionAudit(
            samples=samples,
            features=len(feature_names),
            condition_steps=condition_steps,
            condition_days=condition_days,
            volume_recent_steps=self.volume_recent_steps,
            volume_recent_days=self.volume_recent_steps / self.steps_per_day,
            correlation_steps=self.correlation_steps,
            correlation_days=self.correlation_steps / self.steps_per_day,
            annualization_steps=self.annualization_steps,
            non_finite_values=int((~np.isfinite(feature_matrix)).sum()),
            first_target_start_utc=str(index_rows[0]["target_start_utc"]),
            last_target_start_utc=str(index_rows[-1]["target_start_utc"]),
        )
        return ConditionDataset(
            features=feature_matrix,
            feature_names=feature_names,
            sample_ids=np.arange(samples, dtype=np.int64),
            table_rows=table_rows,
            audit=audit,
        )

    def build_latest(
        self,
        panel_path: Path,
        *,
        condition_steps: int = 240,
    ) -> MarketConditionSnapshot:
        """Resume las últimas ``condition_steps`` velas de un bloque completo.

        Se necesitan ``condition_steps + 1`` cierres consecutivos para calcular
        los retornos. Las velas incompletas al final del panel se omiten, pero
        nunca se atraviesa un hueco histórico.
        """

        if condition_steps <= 1:
            raise ValueError("condition_steps debe ser mayor que uno")
        if self.correlation_steps > condition_steps:
            raise ValueError("El horizonte de correlacion supera la condicion")
        if 2 * self.volume_recent_steps > condition_steps:
            raise ValueError(
                "Se necesitan dos bloques recientes de volumen dentro de la condicion"
            )

        panel = self._read_panel(panel_path, require_close=True)
        complete = panel["complete"]
        required_rows = condition_steps + 1
        last_index: Optional[int] = None
        for candidate in range(len(complete) - 1, required_rows - 2, -1):
            start = candidate - condition_steps
            if bool(complete[start : candidate + 1].all()):
                last_index = candidate
                break
        if last_index is None:
            raise ValueError(
                f"No existe un bloque de {required_rows} velas completas consecutivas"
            )

        first_return_index = last_index - condition_steps + 1
        close = panel["close"][first_return_index - 1 : last_index + 1]
        condition_returns = np.diff(np.log(close), axis=0)
        feature_slice = slice(first_return_index, last_index + 1)
        features = self._summarize(
            condition_returns,
            panel["high"][feature_slice],
            panel["low"][feature_slice],
            panel["volume"][feature_slice],
        )
        if not np.isfinite(features).all():
            raise ValueError("El último vector de condición contiene valores no finitos")

        forecast_start = _parse_utc(panel["timestamps"][last_index]) + self.interval_delta
        return MarketConditionSnapshot(
            features=features,
            feature_names=self._feature_names(condition_steps),
            condition_returns=condition_returns,
            initial_prices=close[-1].copy(),
            assets=("BTC", "ETH"),
            condition_start_utc=panel["timestamps"][first_return_index],
            condition_end_utc=panel["timestamps"][last_index],
            forecast_start_utc=forecast_start.isoformat().replace("+00:00", "Z"),
            condition_steps=condition_steps,
        )

    def write_npz(self, dataset: ConditionDataset, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                condition_features=dataset.features,
                feature_names=np.asarray(dataset.feature_names),
                sample_ids=dataset.sample_ids,
            )
        temporary_path.replace(path)

    @staticmethod
    def write_table(dataset: ConditionDataset, path: Path) -> None:
        fieldnames = (*INDEX_COLUMNS, *dataset.feature_names)
        write_csv_atomic(dataset.table_rows, fieldnames, path)

    def _summarize(
        self,
        returns: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        volume: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cumulative = returns.sum(axis=0)
        step_volatility = returns.std(axis=0, ddof=1)
        realized_volatility = step_volatility * math.sqrt(self.annualization_steps)
        cumulative_paths = np.vstack((np.zeros((1, 2)), np.cumsum(returns, axis=0)))
        current_drawdown = np.exp(cumulative_paths[-1] - cumulative_paths.max(axis=0)) - 1

        log_volume = np.log1p(volume)
        recent = slice(-self.volume_recent_steps, None)
        previous = slice(-2 * self.volume_recent_steps, -self.volume_recent_steps)
        recent_volume_mean = log_volume[recent].mean(axis=0)
        previous_volume_mean = log_volume[previous].mean(axis=0)
        full_volume_std = log_volume.std(axis=0, ddof=1)
        volume_z = np.divide(
            recent_volume_mean - log_volume.mean(axis=0),
            full_volume_std,
            out=np.zeros(2, dtype=np.float64),
            where=full_volume_std > 0,
        )
        volume_change = recent_volume_mean - previous_volume_mean
        recent_range = np.log(high[recent] / low[recent]).mean(axis=0)

        recent_returns = returns[-self.correlation_steps :]
        return_std = recent_returns.std(axis=0, ddof=1)
        if np.any(return_std == 0):
            raise ValueError("La correlacion no esta definida para retornos constantes")
        correlation = float(np.corrcoef(recent_returns[:, 0], recent_returns[:, 1])[0, 1])
        regime_denominator = float(step_volatility.mean() * math.sqrt(len(returns)))
        if regime_denominator == 0:
            raise ValueError("El indicador de regimen no esta definido con volatilidad cero")
        regime_score = float(cumulative.mean() / regime_denominator)

        values: List[float] = []
        for asset_index in range(2):
            values.extend(
                (
                    float(cumulative[asset_index]),
                    float(realized_volatility[asset_index]),
                    float(current_drawdown[asset_index]),
                    float(volume_z[asset_index]),
                    float(volume_change[asset_index]),
                    float(recent_range[asset_index]),
                )
            )
        values.extend((correlation, regime_score))
        return np.asarray(values, dtype=np.float64)

    def _feature_names(self, condition_steps: int) -> Tuple[str, ...]:
        condition_days = condition_steps / self.steps_per_day
        recent_days = self.volume_recent_steps / self.steps_per_day
        correlation_days = self.correlation_steps / self.steps_per_day
        condition_label = f"{condition_days:g}d"
        recent_label = f"{recent_days:g}d"
        correlation_label = f"{correlation_days:g}d"
        names: List[str] = []
        for asset in self.assets:
            names.extend(
                (
                    f"{asset}_cumulative_log_return_{condition_label}",
                    f"{asset}_realized_volatility_ann_{condition_label}",
                    f"{asset}_current_drawdown_{condition_label}",
                    f"{asset}_log_volume_z_{recent_label}",
                    f"{asset}_log_volume_change_{recent_label}",
                    f"{asset}_mean_log_range_{recent_label}",
                )
            )
        names.extend(
            (
                f"btc_eth_correlation_{correlation_label}",
                f"joint_trend_regime_score_{condition_label}",
            )
        )
        return tuple(names)

    @staticmethod
    def _read_windows(
        windows_path: Path,
    ) -> Tuple[NDArray[np.float64], int, NDArray[np.int64]]:
        with np.load(windows_path, allow_pickle=False) as data:
            required = {"condition_returns", "target_returns", "start_indices", "assets"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{windows_path} no contiene arrays requeridos: {sorted(missing)}")
            condition = np.asarray(data["condition_returns"], dtype=np.float64)
            target = data["target_returns"]
            starts = np.asarray(data["start_indices"], dtype=np.int64)
            assets = data["assets"].tolist()
        if condition.ndim != 3 or target.ndim != 3:
            raise ValueError("Las matrices de ventanas deben tener tres dimensiones")
        if condition.shape[0] != target.shape[0] or condition.shape[2] != 2 or target.shape[2] != 2:
            raise ValueError("Formas incompatibles en las ventanas")
        if assets != ["BTC", "ETH"]:
            raise ValueError(f"Orden de activos inesperado: {assets}")
        if not np.isfinite(condition).all():
            raise ValueError("La matriz de condicion contiene valores no finitos")
        return condition, int(target.shape[1]), starts

    @staticmethod
    def _read_index(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(INDEX_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} no contiene columnas requeridas: {sorted(missing)}")
            rows = list(reader)
        for expected_id, row in enumerate(rows):
            if row["sample_id"] != str(expected_id):
                raise ValueError(f"sample_id no secuencial en {path}: {row['sample_id']}")
        return rows

    def _read_panel(self, path: Path, *, require_close: bool = False) -> Dict[str, Any]:
        timestamps: List[str] = []
        complete: List[bool] = []
        high: List[Tuple[float, float]] = []
        low: List[Tuple[float, float]] = []
        volume: List[Tuple[float, float]] = []
        close: List[Tuple[float, float]] = []
        previous_time: Optional[datetime] = None
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(PANEL_FEATURE_COLUMNS) - set(reader.fieldnames or ())
            if require_close:
                missing.update(set(PANEL_CLOSE_COLUMNS) - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} no contiene columnas requeridas: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                timestamp = _parse_utc(row["open_time_utc"])
                if previous_time is not None and timestamp - previous_time != self.interval_delta:
                    raise ValueError(f"Calendario no regular en {path}:{line_number}")
                is_complete = row["is_complete"] == "1"
                if row["is_complete"] not in {"0", "1"}:
                    raise ValueError(f"is_complete invalido en {path}:{line_number}")
                if is_complete:
                    try:
                        high_pair = (float(row["btc_high"]), float(row["eth_high"]))
                        low_pair = (float(row["btc_low"]), float(row["eth_low"]))
                        volume_pair = (float(row["btc_volume"]), float(row["eth_volume"]))
                        close_pair = (
                            (float(row["btc_close"]), float(row["eth_close"]))
                            if all(column in row for column in PANEL_CLOSE_COLUMNS)
                            else (np.nan, np.nan)
                        )
                    except ValueError as error:
                        raise ValueError(f"OHLCV invalido en {path}:{line_number}") from error
                    required_values = (*high_pair, *low_pair, *volume_pair)
                    if require_close:
                        required_values = (*required_values, *close_pair)
                    if not all(np.isfinite(required_values)):
                        raise ValueError(f"OHLCV no finito en {path}:{line_number}")
                    positive_values = (*high_pair, *low_pair)
                    if require_close:
                        positive_values = (*positive_values, *close_pair)
                    if any(value <= 0 for value in positive_values):
                        raise ValueError(f"Precio no positivo en {path}:{line_number}")
                    if any(value < 0 for value in volume_pair):
                        raise ValueError(f"Volumen negativo en {path}:{line_number}")
                else:
                    high_pair = low_pair = volume_pair = (np.nan, np.nan)
                    close_pair = (np.nan, np.nan)
                timestamps.append(row["open_time_utc"])
                complete.append(is_complete)
                high.append(high_pair)
                low.append(low_pair)
                volume.append(volume_pair)
                close.append(close_pair)
                previous_time = timestamp
        return {
            "timestamps": timestamps,
            "complete": np.asarray(complete, dtype=np.bool_),
            "high": np.asarray(high, dtype=np.float64),
            "low": np.asarray(low, dtype=np.float64),
            "volume": np.asarray(volume, dtype=np.float64),
            "close": np.asarray(close, dtype=np.float64),
        }
