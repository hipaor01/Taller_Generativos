#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "quick" ]]; then
  echo "Uso: $0 [full|quick]"
  exit 2
fi

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: falta .venv. Ejecuta primero: bash scripts/setup_environment.sh"
  exit 1
fi

PYTHON="$ROOT/.venv/bin/python"

printf '\n[1/6] Tests\n'
"$PYTHON" -m pytest -q

if [[ "$MODE" == "quick" ]]; then
  FLOW_DIR="results/normalizing_flow_smoke"
  COMMON_DIR="results/normalizing_flow_common_evaluator_smoke"
  EVAL_DIR="results/normalizing_flow_evaluation_smoke"
  AUG_DIR="data/augmented/normalizing_flow_smoke"
  DOWNSTREAM_DIR="results/downstream_drawdown_flow_smoke"
  FLOW_ARGS=(--epochs 2 --patience 1000 --batch-size 256 --layers 2 --hidden-dim 64 --generate-count 64)
  COMMON_ARGS=(--risk-conditions 20 --risk-draws 50 --generation-condition-batch-size 4)
  DOWNSTREAM_ARGS=(--epochs 3 --patience 3 --batch-size 128)
else
  FLOW_DIR="results/normalizing_flow_60epochs"
  COMMON_DIR="results/normalizing_flow_common_evaluator_final"
  EVAL_DIR="results/normalizing_flow_evaluation"
  AUG_DIR="data/augmented/normalizing_flow"
  DOWNSTREAM_DIR="results/downstream_drawdown_flow"
  FLOW_ARGS=(--epochs 60 --patience 1000 --batch-size 256 --layers 8 --hidden-dim 256 --generate-count 1832)
  COMMON_ARGS=(--risk-conditions 250 --risk-draws 1000 --generation-condition-batch-size 2)
  DOWNSTREAM_ARGS=(--epochs 100 --patience 15 --batch-size 128)
fi

rm -rf "$FLOW_DIR" "$COMMON_DIR" "$EVAL_DIR" "$AUG_DIR" "$DOWNSTREAM_DIR"

printf '\n[2/6] Entrenamiento Conditional RealNVP (%s)\n' "$MODE"
"$PYTHON" scripts/train_conditional_flow.py \
  "${FLOW_ARGS[@]}" \
  --device cpu \
  --output-dir "$FLOW_DIR"

printf '\n[3/6] Evaluación financiera propia\n'
"$PYTHON" scripts/evaluate_conditional_flow.py \
  --input "$FLOW_DIR/validation_synthetic_paths.npz" \
  --output-dir "$EVAL_DIR"

printf '\n[4/6] Evaluador común\n'
"$PYTHON" scripts/evaluate_flow_with_common_evaluator.py \
  --checkpoint "$FLOW_DIR/conditional_realnvp_best.pt" \
  --validation-paths "$FLOW_DIR/validation_synthetic_paths.npz" \
  --device cpu \
  "${COMMON_ARGS[@]}" \
  --output-dir "$COMMON_DIR"

printf '\n[5/6] Datasets aumentados\n'
"$PYTHON" scripts/generate_flow_training_datasets.py \
  --checkpoint "$FLOW_DIR/conditional_realnvp_best.pt" \
  --output-dir "$AUG_DIR" \
  --device cpu \
  --generation-batch-size 256

printf '\n[6/6] Modelo downstream de drawdown\n'
"$PYTHON" scripts/train_downstream_drawdown_models.py \
  --data-dir "$AUG_DIR" \
  --output-dir "$DOWNSTREAM_DIR" \
  --device cpu \
  "${DOWNSTREAM_ARGS[@]}"

printf '\nPipeline %s completado.\n' "$MODE"
printf 'Resultados principales:\n  %s\n  %s\n  %s\n' "$FLOW_DIR" "$COMMON_DIR" "$DOWNSTREAM_DIR"
