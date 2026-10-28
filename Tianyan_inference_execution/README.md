# Tianyan quantum-Transformer inference execution

This directory contains the code used to construct, inspect, transpile, submit,
and time quantum circuits for the self-attention and feed-forward-network (FFN)
parts of the quantum Transformer described in the accompanying supplementary
material.

The main paper-scale benchmark is fixed to:

- Tianyan backend: `tianyan-287`
- token register size: 4 qubits
- sequence length: 3 tokens
- attention heads: 4
- shots per submitted circuit: 10
- repetitions per principal circuit: 3

## Main entry points

### `tianyan_layer_benchmark.py`

This is the primary timing runner. It constructs the paper-oriented FFN and
self-attention critical paths, transpiles them to physical QCIS for
Tianyan-287, and writes a timestamped manifest and timing data.

The complete self-attention circuit is too deep for a single Tianyan-287 job,
so it is divided into independently submitted timing segments. The complete
FFN is attempted first; finer FFN and attention recovery segments are also
available. Because an unknown quantum state cannot be transferred between
independent cloud jobs, summing segment durations estimates the hardware cost
of a critical path but is not a functionally chained end-to-end inference.

The runner preserves server-returned timing fields. If Tianyan does not return
an unambiguous QPU/execution-duration field, it calculates a physical
gate-schedule estimate from the transpiled QCIS and the live Tianyan-287
calibration. The estimate uses calibrated X/2 and CZ pulse lengths, excludes
network delay and queue time, and reports boundary readout separately.

Useful commands:

```powershell
# Compile, map, and estimate without submitting jobs.
python tianyan_layer_benchmark.py

# Submit the main benchmark. A 06:00-10:00 Asia/Hong_Kong guard is enabled.
python tianyan_layer_benchmark.py --submit

# Submit only the finer recovery segments.
python tianyan_layer_benchmark.py --submit --recovery-only

# Submit SU(16) repetition-scaling jobs for repetitions 1 through 5.
python tianyan_layer_benchmark.py --submit --su16-scaling

# Submit only repetitions 2 through 5.
python tianyan_layer_benchmark.py --submit --su16-scaling `
  --su16-min-repeat 2 --su16-max-repeat 5

# Resume collection for an existing run without resubmitting.
python tianyan_layer_benchmark.py --collect-manifest PATH_TO_MANIFEST
```

Outputs are written by default to `tianyan_benchmark_outputs/<timestamp>/`.
They can include the manifest, mapped QCIS, raw API responses, per-task timing
records, and aggregate timing reports.

### `tianyan_forward_stages.py`

This runner exercises one forward-stage circuit at a time. Supported stages
are `su16`, `ffn`, `gq`, `mixing`, `attention`, and `attention-ffn`. It is useful
for circuit-size inspection and hardware smoke tests; it is not the primary
segmented timing benchmark.

```powershell
# Inspect the three-token FFN configuration without submitting.
python tianyan_forward_stages.py --stage ffn --token-count 3 `
  --tianyan --backend tianyan-287 --shots 10

# Submit that stage.
python tianyan_forward_stages.py --stage ffn --token-count 3 `
  --tianyan --backend tianyan-287 --shots 10 --submit
```

Its reports are written to `tianyan_forward_outputs/` unless `--output-dir` is
specified.

## Supporting modules

- `attention.py`: Q/K/V controlled-unitary and diagonal attention blocks.
- `circuits.py`: SU(2^n) ansatz, unitary blocks, bridges, and diagonal dilation.
- `swaptest.py`: ordered-token SwapTest layout and circuit construction.
- `entangling.py`: pairwise token-mixing layout and entangling layers.
- `ffn_new.py`: quantum FFN construction.
- `nonlinear.py` and `diag_op.py`: nonlinear/postselected diagonal operations.
- `embedding.py`: amplitude-embedding parameterization and helpers.
- `my_model.py`: complete model register layout and forward composition.
- `backend_config.py`: backend selection, Tianyan compatibility fixes, physical
  qubit exclusions, and sparse-readout handling.
- `SU16.py`: legacy/reference SU(16) wrapper retained for comparison.

`CIRCUIT_AUDIT.md` records the detailed comparison between the implementation
and the supplementary-paper circuit design.

## Installation

Python 3.12 was used for the experiments. Create an isolated environment and
install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Tianyan connection key must be supplied locally. Copy `.env.example` to
`.env` and replace the placeholder, or set the variable in the environment:

```powershell
$env:TIANYAN_LOGIN_KEY = "your-connection-key"
```

Never commit or publish the real connection key. The `.gitignore` file excludes
`.env`.

## Verification

Run the regression tests from this directory:

```powershell
python -m pytest -q
```

The included tests check the fixed benchmark scale, segment ordering, circuit
composition, backend compatibility helpers, register layout, and model
structure. `smoke_test_small_parallel.py` provides a small executable model
smoke test.

