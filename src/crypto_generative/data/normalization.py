"""Normalizacion ajustada exclusivamente con el bloque de entrenamiento."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ZScoreParameters:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    zero_variance_columns: Sequence[int]


class ZScoreScaler:
    """Transformacion z-score sin estado mutable ni ajuste implicito."""

    @staticmethod
    def fit(values: NDArray[np.float64]) -> ZScoreParameters:
        if values.ndim != 2 or not len(values):
            raise ValueError("El ajuste requiere una matriz bidimensional no vacia")
        if not np.isfinite(values).all():
            raise ValueError("Los datos de ajuste contienen valores no finitos")
        mean = values.mean(axis=0)
        raw_scale = values.std(axis=0, ddof=0)
        zero_variance = np.flatnonzero(raw_scale == 0).tolist()
        scale = np.where(raw_scale == 0, 1.0, raw_scale)
        return ZScoreParameters(mean, scale, zero_variance)

    @staticmethod
    def transform(
        values: NDArray[np.float64], parameters: ZScoreParameters
    ) -> NDArray[np.float64]:
        if values.shape[-1] != len(parameters.mean):
            raise ValueError("El numero de variables no coincide con el normalizador")
        transformed = (values - parameters.mean) / parameters.scale
        if not np.isfinite(transformed).all():
            raise ValueError("La normalizacion ha producido valores no finitos")
        return np.asarray(transformed, dtype=np.float64)


@dataclass(frozen=True)
class NormalizationAudit:
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    unique_training_return_rows: int
    condition_features: int
    return_assets: int
    condition_zero_variance_features: Sequence[int]
    return_zero_variance_assets: Sequence[int]
    train_condition_max_abs_mean: float
    train_condition_max_abs_std_error: float
    training_returns_max_abs_mean: float
    training_returns_max_abs_std_error: float
    normalized_non_finite_values: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedDataset:
    condition_features: NDArray[np.float64]
    condition_returns: NDArray[np.float64]
    target_returns: NDArray[np.float64]
    feature_names: NDArray[np.str_]
    assets: NDArray[np.str_]
    sample_ids: NDArray[np.int64]
    condition_parameters: ZScoreParameters
    return_parameters: ZScoreParameters
    audit: NormalizationAudit


class TrainingOnlyNormalizer:
    """Ajusta en train y transforma todos los bloques con parametros congelados."""

    def build(
        self,
        windows_path: Path,
        conditions_path: Path,
        split_path: Path,
        returns_path: Path,
    ) -> NormalizedDataset:
        windows = self._read_windows(windows_path)
        conditions = self._read_conditions(conditions_path)
        split = self._read_split(split_path, len(windows["start_indices"]))
        raw_returns, raw_valid = self._read_returns(returns_path)

        samples = len(windows["start_indices"])
        if conditions["features"].shape[0] != samples:
            raise ValueError("Ventanas y condiciones no tienen el mismo numero de muestras")
        if not np.array_equal(conditions["sample_ids"], np.arange(samples)):
            raise ValueError("Los sample_ids de condicion no estan alineados")

        train_ids = split["train"]
        condition_parameters = ZScoreScaler.fit(conditions["features"][train_ids])

        condition_steps = windows["condition"].shape[1]
        target_steps = windows["target"].shape[1]
        total_steps = condition_steps + target_steps
        training_return_mask = np.zeros(len(raw_returns), dtype=np.bool_)
        for sample_id in train_ids:
            start = int(windows["start_indices"][sample_id])
            end = start + total_steps
            if end > len(raw_returns):
                raise ValueError(f"La muestra {sample_id} excede el fichero de retornos")
            training_return_mask[start:end] = True
        if not bool(raw_valid[training_return_mask].all()):
            raise ValueError("La cobertura de retornos de entrenamiento contiene filas no validas")
        unique_training_returns = raw_returns[training_return_mask]
        return_parameters = ZScoreScaler.fit(unique_training_returns)

        normalized_conditions = ZScoreScaler.transform(
            conditions["features"], condition_parameters
        )
        normalized_condition_returns = ZScoreScaler.transform(
            windows["condition"], return_parameters
        )
        normalized_target_returns = ZScoreScaler.transform(
            windows["target"], return_parameters
        )

        train_condition = normalized_conditions[train_ids]
        normalized_unique_returns = ZScoreScaler.transform(
            unique_training_returns, return_parameters
        )
        all_normalized = (
            normalized_conditions,
            normalized_condition_returns,
            normalized_target_returns,
        )
        audit = NormalizationAudit(
            total_samples=samples,
            train_samples=len(train_ids),
            validation_samples=len(split["validation"]),
            test_samples=len(split["test"]),
            unique_training_return_rows=len(unique_training_returns),
            condition_features=normalized_conditions.shape[1],
            return_assets=normalized_target_returns.shape[2],
            condition_zero_variance_features=condition_parameters.zero_variance_columns,
            return_zero_variance_assets=return_parameters.zero_variance_columns,
            train_condition_max_abs_mean=float(np.abs(train_condition.mean(axis=0)).max()),
            train_condition_max_abs_std_error=float(
                np.abs(train_condition.std(axis=0, ddof=0) - 1).max()
            ),
            training_returns_max_abs_mean=float(
                np.abs(normalized_unique_returns.mean(axis=0)).max()
            ),
            training_returns_max_abs_std_error=float(
                np.abs(normalized_unique_returns.std(axis=0, ddof=0) - 1).max()
            ),
            normalized_non_finite_values=sum(
                int((~np.isfinite(values)).sum()) for values in all_normalized
            ),
        )
        return NormalizedDataset(
            condition_features=normalized_conditions,
            condition_returns=normalized_condition_returns,
            target_returns=normalized_target_returns,
            feature_names=conditions["feature_names"],
            assets=windows["assets"],
            sample_ids=conditions["sample_ids"],
            condition_parameters=condition_parameters,
            return_parameters=return_parameters,
            audit=audit,
        )

    @staticmethod
    def write_npz(dataset: NormalizedDataset, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                condition_features=dataset.condition_features,
                condition_returns=dataset.condition_returns,
                target_returns=dataset.target_returns,
                feature_names=dataset.feature_names,
                assets=dataset.assets,
                sample_ids=dataset.sample_ids,
                condition_feature_mean=dataset.condition_parameters.mean,
                condition_feature_scale=dataset.condition_parameters.scale,
                return_mean=dataset.return_parameters.mean,
                return_scale=dataset.return_parameters.scale,
            )
        temporary_path.replace(path)

    @staticmethod
    def _read_windows(path: Path) -> Dict[str, NDArray[Any]]:
        with np.load(path, allow_pickle=False) as data:
            required = {"condition_returns", "target_returns", "start_indices", "assets"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} no contiene arrays requeridos: {sorted(missing)}")
            result = {
                "condition": np.asarray(data["condition_returns"], dtype=np.float64),
                "target": np.asarray(data["target_returns"], dtype=np.float64),
                "start_indices": np.asarray(data["start_indices"], dtype=np.int64),
                "assets": data["assets"].copy(),
            }
        condition = result["condition"]
        target = result["target"]
        if condition.ndim != 3 or target.ndim != 3 or condition.shape[0] != target.shape[0]:
            raise ValueError("Formas incompatibles en las ventanas")
        if result["assets"].tolist() != ["BTC", "ETH"]:
            raise ValueError("Orden de activos inesperado")
        if not np.isfinite(condition).all() or not np.isfinite(target).all():
            raise ValueError("Las ventanas contienen valores no finitos")
        return result

    @staticmethod
    def _read_conditions(path: Path) -> Dict[str, NDArray[Any]]:
        with np.load(path, allow_pickle=False) as data:
            required = {"condition_features", "feature_names", "sample_ids"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} no contiene arrays requeridos: {sorted(missing)}")
            result = {
                "features": np.asarray(data["condition_features"], dtype=np.float64),
                "feature_names": data["feature_names"].copy(),
                "sample_ids": np.asarray(data["sample_ids"], dtype=np.int64),
            }
        if result["features"].ndim != 2 or not np.isfinite(result["features"]).all():
            raise ValueError("La matriz de condiciones no es valida")
        return result

    @staticmethod
    def _read_split(path: Path, samples: int) -> Dict[str, NDArray[np.int64]]:
        names = {
            "train": "train_sample_ids",
            "validation": "validation_sample_ids",
            "test": "test_sample_ids",
            "purge_train_validation": "purge_train_validation_sample_ids",
            "purge_validation_test": "purge_validation_test_sample_ids",
        }
        with np.load(path, allow_pickle=False) as data:
            missing = set(names.values()) - set(data.files)
            if missing:
                raise ValueError(f"{path} no contiene arrays requeridos: {sorted(missing)}")
            result = {
                name: np.asarray(data[array_name], dtype=np.int64)
                for name, array_name in names.items()
            }
        all_ids = np.concatenate(list(result.values()))
        if not np.array_equal(np.sort(all_ids), np.arange(samples)):
            raise ValueError("El split no es una particion completa y disjunta")
        return result

    @staticmethod
    def _read_returns(path: Path) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        values = []
        valid = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"btc_log_return", "eth_log_return", "returns_valid"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} no contiene columnas requeridas: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                is_valid = row["returns_valid"] == "1"
                if row["returns_valid"] not in {"0", "1"}:
                    raise ValueError(f"returns_valid invalido en {path}:{line_number}")
                if is_valid:
                    try:
                        pair = (float(row["btc_log_return"]), float(row["eth_log_return"]))
                    except ValueError as error:
                        raise ValueError(f"Retorno invalido en {path}:{line_number}") from error
                    if not all(np.isfinite(pair)):
                        raise ValueError(f"Retorno no finito en {path}:{line_number}")
                else:
                    pair = (np.nan, np.nan)
                values.append(pair)
                valid.append(is_valid)
        return np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=np.bool_)
