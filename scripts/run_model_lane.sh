#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/full}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [MODEL ...]" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

NON_OOD_CELLS=primary_A0,robustness_A0,morphology_rural_A0,morphology_semiurban_A0,morphology_urban_A0,morphology_commercial_A0,true_state,no_tie_ranking,no_simulator,no_memory,schema_only,reliability_seed0
OOD_CELLS=topology_ood_A0,generation_ood_A0,load_ood_A0

models=("$@")
for model in "${models[@]}"; do
  "$PYTHON_BIN" -u -m neurips_grid.worker \
    --model "$model" \
    --cells "$NON_OOD_CELLS" \
    --output-root "$OUTPUT_ROOT"
done

if [[ -f "$PROJECT_ROOT/artifacts/panels/topology_ood/panel_manifest.json" \
   && -f "$PROJECT_ROOT/artifacts/panels/generation_ood/panel_manifest.json" \
   && -f "$PROJECT_ROOT/artifacts/panels/load_ood/panel_manifest.json" ]]; then
  for model in "${models[@]}"; do
    "$PYTHON_BIN" -u -m neurips_grid.worker \
      --model "$model" \
      --cells "$OOD_CELLS" \
      --output-root "$OUTPUT_ROOT"
  done
else
  echo "OOD manifests are not all present; non-OOD phase completed and OOD phase was deferred."
fi
