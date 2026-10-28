#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_DIR:?Set RUN_DIR to a NEW reproduction output directory}"
PYTHON="${PYTHON:-python}"
if [[ -e "$RUN_DIR/training.jsonl" ]]; then
  echo "RUN_DIR already contains training records. Use an explicit resume command." >&2
  exit 1
fi
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON" -u "$HERE/delivery_runner.py" train --output-dir "$RUN_DIR" --epochs 100 --learning-rate 0.003 --attention-layers 1 --ffn-layers 1 --entangling-layers 3 --batch-size 1 --eval-batch-size 1 --eval-every 0 --save-every 50 --log-every 1 --seed 2026 --require-gpu
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8"))["metadata"]; sys.exit(0 if m["progress"]["epoch"]>=100 else "Stage one interrupted; resume it explicitly.")' "$RUN_DIR/latest.json"
"$PYTHON" -u "$HERE/delivery_runner.py" train --output-dir "$RUN_DIR" --resume "$RUN_DIR/latest" --epochs 300 --learning-rate 0.001 --allow-learning-rate-change --attention-layers 1 --ffn-layers 1 --entangling-layers 3 --batch-size 1 --eval-batch-size 1 --eval-every 0 --save-every 50 --log-every 1 --seed 2026 --require-gpu
