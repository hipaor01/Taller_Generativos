"""Utilidades comunes para artefactos de datos reproducibles."""

from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 sin cargar el fichero completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    """Escribe JSON mediante reemplazo atomico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_checksums(files: Mapping[str, Path], path: Path) -> None:
    """Escribe un fichero compatible con ``shasum -a 256 -c``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    lines = [f"{sha256_file(file_path)}  {name}" for name, file_path in files.items()]
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def write_csv_atomic(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], path: Path
) -> None:
    """Escribe una tabla CSV mediante reemplazo atomico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def relative_or_absolute(path: Path, root: Path) -> str:
    """Representa rutas del proyecto de forma portable y conserva las externas."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)
