#!/usr/bin/env python3
"""Comprueba el intérprete y las dependencias directas declaradas del proyecto."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys
import tomllib

from _bootstrap import PROJECT_ROOT


def exact_requirements(pyproject_path: Path) -> dict[str, str]:
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = list(project["dependencies"])
    for values in project.get("optional-dependencies", {}).values():
        requirements.extend(values)
    exact = {}
    for requirement in requirements:
        if "==" not in requirement:
            raise ValueError(f"Dependencia directa no fijada exactamente: {requirement}")
        name, expected = requirement.split("==", maxsplit=1)
        name = name.strip()
        expected = expected.strip()
        previous = exact.get(name)
        if previous is not None and previous != expected:
            raise ValueError(
                f"Pines incompatibles para {name}: {previous} y {expected}"
            )
        exact[name] = expected
    return exact


def main() -> int:
    expected_python = (PROJECT_ROOT / ".python-version").read_text(
        encoding="utf-8"
    ).strip()
    actual_python = platform.python_version()
    failures = []
    if actual_python != expected_python:
        failures.append(
            f"Python esperado {expected_python}; encontrado {actual_python}"
        )

    expected_packages = exact_requirements(PROJECT_ROOT / "pyproject.toml")
    for package, expected in sorted(expected_packages.items()):
        try:
            actual = version(package)
        except PackageNotFoundError:
            failures.append(f"Falta {package}=={expected}")
            continue
        if actual != expected:
            failures.append(
                f"{package}: versión esperada {expected}; encontrada {actual}"
            )

    if failures:
        print("Entorno NO reproducible:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Python: {actual_python}")
    print(f"Plataforma: {platform.platform()} ({platform.machine()})")
    print(f"Dependencias directas verificadas: {len(expected_packages)}")
    print("Entorno: compatible con pyproject.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
