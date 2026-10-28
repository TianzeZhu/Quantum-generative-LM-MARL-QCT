#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${EVAL_DIR:?Set EVAL_DIR to a new output directory}"
PYTHON="${PYTHON:-python}"
if [[ -e "$EVAL_DIR/training.jsonl" ]]; then echo "EVAL_DIR already contains evaluation records." >&2; exit 1; fi
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON" -u "$HERE/delivery_runner.py" evaluate --split test --batch-size 1 --decode-examples 49 --max-decode-steps 6 --output-dir "$EVAL_DIR" --require-gpu "$@"
