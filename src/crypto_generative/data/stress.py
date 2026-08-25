"""Carga común de datos congelados y artefactos de escenarios."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from crypto_generative.portfolio import ScenarioCategory, StressScenarioSet

from .returns import LogReturnBuilder


@dataclass(frozen=True)
class FrozenPathBatch:
    """Trayectorias desnormalizadas y etiquetas temporales de un split."""

    split: str
    sample_ids: NDArray[np.int64]
    log_returns: NDArray[np.float64]
    labels: Tuple[str, ...]
    assets: Tuple[str, ...]


@dataclass(frozen=True)
class BootstrapTrainingSeries:
    """Serie conjunta única, separada en segmentos contiguos."""

    log_returns: NDArray[np.float64]
    segment_ids: NDArray[np.int64]
    assets: Tuple[str, ...]


class ProjectScenarioLoader:
    """Adapta los artefactos del proyecto al contrato de stress testing."""

    def __init__(
        self,
        normalized_path: Path,
        split_path: Path,
        split_index_path: Path,
        panel_path: Path,
    ) -> None:
        self.normalized_path = Path(normalized_path)
        self.split_path = Path(split_path)
        self.split_index_path = Path(split_index_path)
        self.panel_path = Path(panel_path)

    def load_split(self, split: str) -> FrozenPathBatch:
        key_by_split = {
            "train": "train_sample_ids",
            "validation": "validation_sample_ids",
            "test": "test_sample_ids",
        }
        if split not in key_by_split:
            raise ValueError("split debe ser train, validation o test")
        self._require(self.normalized_path, self.split_path, self.split_index_path)

        with np.load(self.normalized_path, allow_pickle=False) as data:
            stored_ids = data["sample_ids"].astype(np.int64)
            targets = data["target_returns"].astype(np.float64)
            return_mean = data["return_mean"].astype(np.float64)
            return_scale = data["return_scale"].astype(np.float64)
            assets = tuple(str(asset) for asset in data["assets"])
        with np.load(self.split_path, allow_pickle=False) as split_data:
            sample_ids = split_data[key_by_split[split]].astype(np.int64)

        positions = self._positions_by_id(stored_ids, sample_ids)
        log_returns = targets[positions] * return_scale + return_mean
        labels_by_id = self._load_target_labels()
        labels = tuple(labels_by_id[int(sample_id)] for sample_id in sample_ids)
        return FrozenPathBatch(split, sample_ids, log_returns, labels, assets)

    def load_normalized_conditions(
        self,
        sample_ids: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Carga las condiciones normalizadas alineadas con ``sample_ids``."""

        self._require(self.normalized_path)
        requested_ids = np.asarray(sample_ids, dtype=np.int64)
        if requested_ids.ndim != 1:
            raise ValueError("sample_ids debe ser un vector")
        with np.load(self.normalized_path, allow_pickle=False) as data:
            stored_ids = data["sample_ids"].astype(np.int64)
            conditions = data["condition_features"].astype(np.float64)
        positions = self._positions_by_id(stored_ids, requested_ids)
        selected = conditions[positions]
        if selected.ndim != 2 or not np.isfinite(selected).all():
            raise ValueError("Las condiciones cargadas no son una matriz finita")
        return selected

    def load_bootstrap_training_series(self) -> BootstrapTrainingSeries:
        """Reconstruye retornos únicos de train sin multiplicar ventanas solapadas."""
        self._require(self.panel_path, self.split_index_path)
        intervals = self._load_train_intervals()
        rows, _ = LogReturnBuilder(interval="6h").build(self.panel_path)

        selected_times = []
        selected_values = []
        interval_index = 0
        for row in rows:
            if row["returns_valid"] != "1":
                continue
            timestamp = _parse_utc(row["open_time_utc"])
            while (
                interval_index < len(intervals)
                and timestamp > intervals[interval_index][1]
            ):
                interval_index += 1
            if interval_index == len(intervals):
                break
            start, end = intervals[interval_index]
            if start <= timestamp <= end:
                selected_times.append(timestamp)
                selected_values.append(
                    (float(row["btc_log_return"]), float(row["eth_log_return"]))
                )

        if not selected_values:
            raise ValueError("No se encontraron retornos de entrenamiento para bootstrap")
        segment_ids = np.zeros(len(selected_times), dtype=np.int64)
        expected_delta = timedelta(hours=6)
        for index in range(1, len(selected_times)):
            segment_ids[index] = segment_ids[index - 1] + int(
                selected_times[index] - selected_times[index - 1] != expected_delta
            )
        return BootstrapTrainingSeries(
            log_returns=np.asarray(selected_values, dtype=np.float64),
            segment_ids=segment_ids,
            assets=("BTC", "ETH"),
        )

    def load_generated_scenarios(
        self,
        name: str,
        path: Path,
        *,
        expected_reference: FrozenPathBatch | None = None,
    ) -> StressScenarioSet:
        """Carga el contrato NPZ compartido por CVAE, flow y GAN."""
        path = Path(path)
        self._require(path)
        with np.load(path, allow_pickle=False) as data:
            if "generated_returns" in data:
                generated = np.asarray(data["generated_returns"], dtype=np.float64)
            elif "generated_conditional_returns" in data:
                conditional = np.asarray(
                    data["generated_conditional_returns"], dtype=np.float64
                )
                generated = conditional.reshape(
                    -1, conditional.shape[-2], conditional.shape[-1]
                )
            else:
                raise ValueError(
                    f"{path} no contiene generated_returns ni "
                    "generated_conditional_returns"
                )
            assets = tuple(str(asset) for asset in data["assets"])
            stored_reference = (
                np.asarray(data["real_returns"], dtype=np.float64)
                if "real_returns" in data
                else None
            )

        if expected_reference is not None:
            if assets != expected_reference.assets:
                raise ValueError(f"Orden de activos incompatible en {path}")
            if stored_reference is not None and not np.allclose(
                stored_reference, expected_reference.log_returns, atol=1e-10
            ):
                raise ValueError(f"La referencia guardada no coincide con test: {path}")
        return StressScenarioSet(
            name=name,
            category=ScenarioCategory.GENERATIVE,
            log_returns=generated,
            metadata={"source_path": str(path), "assets": list(assets)},
        )

    def _load_target_labels(self) -> dict[int, str]:
        labels = {}
        with self.split_index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                labels[int(row["sample_id"])] = (
                    f"{row['target_start_utc']} -> {row['target_end_utc']}"
                )
        return labels

    def _load_train_intervals(self) -> Sequence[Tuple[datetime, datetime]]:
        intervals = []
        with self.split_index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "train":
                    intervals.append(
                        (
                            _parse_utc(row["condition_start_utc"]),
                            _parse_utc(row["target_end_utc"]),
                        )
                    )
        if not intervals:
            raise ValueError("El índice no contiene intervalos de train")

        merged = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    @staticmethod
    def _require(*paths: Path) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Faltan artefactos: " + ", ".join(missing))

    @staticmethod
    def _positions_by_id(
        stored_ids: NDArray[np.int64],
        requested_ids: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        position_by_id = {
            int(sample_id): index for index, sample_id in enumerate(stored_ids)
        }
        try:
            return np.asarray(
                [position_by_id[int(sample_id)] for sample_id in requested_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"sample_id no encontrado en dataset: {error.args[0]}"
            ) from error


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp sin zona horaria: {value}")
    return parsed.astimezone(timezone.utc)
