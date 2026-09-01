#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 MODEL CELL SHARD_INDEX NUM_SHARDS CANONICAL OUTPUT_ROOT" >&2
  exit 2
fi

model=$1
cell=$2
shard_index=$3
num_shards=$4
canonical=$5
output_root=$6

cd "$PROJECT_ROOT"
mkdir -p "$output_root"

"$PYTHON_BIN" -u -m neurips_grid.sharded_worker \
  --model "$model" \
  --cell "$cell" \
  --shard-index "$shard_index" \
  --num-shards "$num_shards" \
  --canonical "$canonical" \
  --output-root "$output_root"
