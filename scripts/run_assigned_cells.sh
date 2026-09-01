#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/full}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL COMMA_SEPARATED_CELLS" >&2
  exit 2
fi

model=$1
cells=$2

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" -u -m neurips_grid.worker \
  --model "$model" \
  --cells "$cells" \
  --output-root "$OUTPUT_ROOT"
