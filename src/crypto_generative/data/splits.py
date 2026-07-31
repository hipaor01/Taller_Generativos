"""Division temporal purgada para ventanas financieras solapadas."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .artifacts import write_csv_atomic


REQUIRED_INDEX_COLUMNS = (
    "sample_id",
    "condition_start_utc",
    "condition_end_utc",
    "target_start_utc",
    "target_end_utc",
)

SPLIT_INDEX_COLUMNS = (*REQUIRED_INDEX_COLUMNS, "split")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp sin zona horaria: {value}")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_end_exclusive_utc: str
    validation_start_utc: str
    validation_end_exclusive_utc: str
    test_start_utc: str
    minimum_purge_days: int = 90

    def validate(self) -> None:
        train_end = parse_utc(self.train_end_exclusive_utc)
        validation_start = parse_utc(self.validation_start_utc)
        validation_end = parse_utc(self.validation_end_exclusive_utc)
        test_start = parse_utc(self.test_start_utc)
        minimum = timedelta(days=self.minimum_purge_days)
        if validation_start - train_end < minimum:
            raise ValueError("La purga entrenamiento-validacion es inferior al minimo")
        if test_start - validation_end < minimum:
            raise ValueError("La purga validacion-prueba es inferior al minimo")
        if not train_end < validation_start < validation_end < test_start:
            raise ValueError("Los cortes temporales no estan ordenados")


@dataclass(frozen=True)
class SplitAudit:
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    purge_train_validation_samples: int
    purge_validation_test_samples: int
    retained_samples: int
    purged_samples: int
    retained_ratios: Mapping[str, float]
    train_first_target_start_utc: str
    train_last_target_start_utc: str
    validation_first_target_start_utc: str
    validation_last_target_start_utc: str
    test_first_target_start_utc: str
    test_last_target_start_utc: str
    train_validation_raw_gap_hours: float
    validation_test_raw_gap_hours: float
    raw_intervals_overlap: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalSplit:
    train_ids: NDArray[np.int64]
    validation_ids: NDArray[np.int64]
    test_ids: NDArray[np.int64]
    purge_train_validation_ids: NDArray[np.int64]
    purge_validation_test_ids: NDArray[np.int64]
    index_rows: Sequence[Mapping[str, Any]]
    audit: SplitAudit


class PurgedTemporalSplitBuilder:
    """Asigna por inicio del objetivo y comprueba que no haya datos compartidos."""

    def __init__(self, config: TemporalSplitConfig) -> None:
        config.validate()
        self.config = config
        self.train_end = parse_utc(config.train_end_exclusive_utc)
        self.validation_start = parse_utc(config.validation_start_utc)
        self.validation_end = parse_utc(config.validation_end_exclusive_utc)
        self.test_start = parse_utc(config.test_start_utc)

    def build(self, index_path: Path) -> TemporalSplit:
        rows = self._read_index(index_path)
        assignments: Dict[str, List[int]] = {
            "train": [],
            "purge_train_validation": [],
            "validation": [],
            "purge_validation_test": [],
            "test": [],
        }
        output_rows: List[Mapping[str, Any]] = []
        for row in rows:
            sample_id = int(row["sample_id"])
            target_start = parse_utc(row["target_start_utc"])
            split = self._assignment(target_start)
            assignments[split].append(sample_id)
            output = dict(row)
            output["split"] = split
            output_rows.append(output)

        for split in ("train", "validation", "test"):
            if not assignments[split]:
                raise ValueError(f"El split {split} ha quedado vacio")

        train_last = rows[assignments["train"][-1]]
        validation_first = rows[assignments["validation"][0]]
        validation_last = rows[assignments["validation"][-1]]
        test_first = rows[assignments["test"][0]]
        train_validation_gap = (
            parse_utc(validation_first["condition_start_utc"])
            - parse_utc(train_last["target_end_utc"])
        )
        validation_test_gap = (
            parse_utc(test_first["condition_start_utc"])
            - parse_utc(validation_last["target_end_utc"])
        )
        overlaps = train_validation_gap <= timedelta(0) or validation_test_gap <= timedelta(0)
        if overlaps:
            raise ValueError("Los intervalos brutos de los splits se solapan pese a la purga")

        retained = sum(len(assignments[name]) for name in ("train", "validation", "test"))
        purged = len(rows) - retained
        audit = SplitAudit(
            total_samples=len(rows),
            train_samples=len(assignments["train"]),
            validation_samples=len(assignments["validation"]),
            test_samples=len(assignments["test"]),
            purge_train_validation_samples=len(assignments["purge_train_validation"]),
            purge_validation_test_samples=len(assignments["purge_validation_test"]),
            retained_samples=retained,
            purged_samples=purged,
            retained_ratios={
                name: len(assignments[name]) / retained
                for name in ("train", "validation", "test")
            },
            train_first_target_start_utc=rows[assignments["train"][0]]["target_start_utc"],
            train_last_target_start_utc=train_last["target_start_utc"],
            validation_first_target_start_utc=validation_first["target_start_utc"],
            validation_last_target_start_utc=validation_last["target_start_utc"],
            test_first_target_start_utc=test_first["target_start_utc"],
            test_last_target_start_utc=rows[assignments["test"][-1]]["target_start_utc"],
            train_validation_raw_gap_hours=train_validation_gap.total_seconds() / 3600,
            validation_test_raw_gap_hours=validation_test_gap.total_seconds() / 3600,
            raw_intervals_overlap=overlaps,
        )
        return TemporalSplit(
            train_ids=np.asarray(assignments["train"], dtype=np.int64),
            validation_ids=np.asarray(assignments["validation"], dtype=np.int64),
            test_ids=np.asarray(assignments["test"], dtype=np.int64),
            purge_train_validation_ids=np.asarray(
                assignments["purge_train_validation"], dtype=np.int64
            ),
            purge_validation_test_ids=np.asarray(
                assignments["purge_validation_test"], dtype=np.int64
            ),
            index_rows=output_rows,
            audit=audit,
        )

    def write_npz(self, split: TemporalSplit, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                train_sample_ids=split.train_ids,
                validation_sample_ids=split.validation_ids,
                test_sample_ids=split.test_ids,
                purge_train_validation_sample_ids=split.purge_train_validation_ids,
                purge_validation_test_sample_ids=split.purge_validation_test_ids,
            )
        temporary_path.replace(path)

    @staticmethod
    def write_index(split: TemporalSplit, path: Path) -> None:
        write_csv_atomic(split.index_rows, SPLIT_INDEX_COLUMNS, path)

    def _assignment(self, target_start: datetime) -> str:
        if target_start < self.train_end:
            return "train"
        if target_start < self.validation_start:
            return "purge_train_validation"
        if target_start < self.validation_end:
            return "validation"
        if target_start < self.test_start:
            return "purge_validation_test"
        return "test"

    @staticmethod
    def _read_index(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(REQUIRED_INDEX_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} no contiene columnas requeridas: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"El indice esta vacio: {path}")
        previous_target: Optional[datetime] = None
        for expected_id, row in enumerate(rows):
            if row["sample_id"] != str(expected_id):
                raise ValueError(f"sample_id no secuencial en {path}: {row['sample_id']}")
            target_start = parse_utc(row["target_start_utc"])
            if previous_target is not None and target_start <= previous_target:
                raise ValueError(f"El indice no esta ordenado temporalmente: {path}")
            previous_target = target_start
        return rows
