#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV_BIN="${UV_BIN:-uv}"
EXPECTED_UV_VERSION="0.11.24"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "ERROR: no se encuentra uv. Instala uv ${EXPECTED_UV_VERSION}: https://docs.astral.sh/uv/"
  exit 1
fi

ACTUAL_UV_VERSION="$("$UV_BIN" --version | awk '{print $2}')"
if [ "$ACTUAL_UV_VERSION" != "$EXPECTED_UV_VERSION" ]; then
  echo "ERROR: se requiere uv ${EXPECTED_UV_VERSION}; encontrado ${ACTUAL_UV_VERSION}."
  exit 1
fi

"$UV_BIN" sync --frozen --all-extras
"$UV_BIN" run --frozen python scripts/check_environment.py

printf '\nEntorno preparado. Actívalo con:\n  source .venv/bin/activate\n'
