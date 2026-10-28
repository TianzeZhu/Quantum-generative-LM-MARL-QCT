# Quantum Transformer circuit audit

Scope: the supplementary draft's self-attention and FFN designs, evaluated at
four qubits per token and sequence length three, against the maintained code in
this directory. The audit distinguishes the model implementation from the new
hardware timing runner.

## Findings

| Component | Draft design | Existing implementation | Assessment |
|---|---|---|---|
| Attention registers | `3m-2` encoded token-register groups; for `n=4`, `m=3`, the complete layout is 92 qubits | `attention.py` allocates Q/K/V copies and the diagonal output accumulators with the same ordered-token organization | Matches the register-count construction |
| Q/K/V heads | Each head is `V† -> D -> U`; controlled use of the nonunitary diagonal requires an additional selector ancilla | `controlled_su16_array` implements two unitary blocks, the diagonal dilation, and separate work/selector ancillas | Substantially matches; the code's final diagonal `RZ` is not itself selector-controlled, whereas the draft describes a controlled phase operation. The timing runner uses the controlled phase from the draft |
| SwapTest/mixing | All ordered pairs `i != j`; pairs for one parallel round operate on disjoint copies; `m-1` rounds are sequential | `attention.py`, `swaptest.py`, and `entangling.py` use ordered pairs and disjoint value copies, with one StronglyEntangling block per pair | Matches the draft's dataflow and critical-path ordering |
| Softmax/nonlinear block | Two sequential parameterized circuits and one ancilla; the draft extends the final multi-controlled X with the ancilla control and postselects ancilla zero | `nonlinear.py` controls the whole last SU block through `extra_control_wires` and postselects ancilla one | Does not exactly match the draft |
| FFN | `G_W1 -> Sigmoid -> G_W2 -> Sigmoid`, with each `G_Wi = U D V†` | `ffn_new.py` provides one U-D-V block; `my_model.py` invokes it once per token and omits both sigmoid stages and the second U-D-V block | Existing FFN path is incomplete relative to the draft |
| SU(16) ansatz | The 255-parameter four-qubit decomposition shown in the draft figures | The legacy `SU16` helper has that decomposition, but the actively called `su16_block` delegates to a newer CZ brick-wall ansatz | Parameter count matches; active gate structure does not |
| Wider unitary bridge | Draft figures/time-depth analysis use multi-controlled CNOT bridge layers between SU(16) blocks | `circuits.py::unitary_part` uses long-range parameterized CRX/CRY/CRZ bridges | Does not match for circuits wider than one four-qubit SU(16) block |
| Existing Tianyan attention stage | Complete attention design | `tianyan_forward_stages.py` deliberately sets the trainable diagonal angles to zero and disables nonlinear layers for its full-register smoke test | Routing smoke test only, not the draft layer |
| Existing Tianyan FFN stage | Complete two-linear/two-sigmoid FFN | The stage submits one five-wire U-D-V block | Block smoke test only, not the draft layer |

## Timing runner and segmentation

`tianyan_layer_benchmark.py` models the draft's layer critical paths, including
the draft SU(16) decomposition and nonlinear final-gate-only control, with these
independent hardware segments:

- attention: four sequential head blocks, two sequential SwapTest/mixing
  rounds, and two sequential softmax halves;
- FFN: `G_W1`, two sigmoid halves, `G_W2`, and two more sigmoid halves.

Before those segments, the runner submits the complete FFN as a single
paper-faithful circuit. Four fresh ancillas implement deferred postselection for
the two nonunitary linear blocks and two sigmoid blocks without illegal ancilla
reuse. It also submits the first FFN half and the complete attention softmax as
single-task comparison circuits. Each composite is paired with exactly the same
gate components in the segmented run, enabling repeat-by-repeat whole-task
versus segmented-sum comparisons when Tianyan returns QPU timing metadata.

Register copies that the draft identifies as parallel are represented by one
critical-path block rather than serialized copies. This avoids counting
parallel work multiple times and keeps each QCIS payload below the scope that
previous monolithic submissions exceeded.

The segmentation is for timing only. A cloud job cannot export an unknown
quantum state and restore it into a later job, so the independently submitted
segments are not a functionally composable end-to-end forward pass. Coherent
terminal ancillas replace unsupported mid-circuit postselection; one terminal
measurement is added only to make each job executable.

## Execution-time methodology

The runner submits each segment as three identical tasks with 10 shots on
`tianyan-287`. It saves the raw API response and every field whose name suggests
timing, but accepts only an unambiguously named QPU/execution-duration field as
measured hardware time. Queue, wait, client wall-clock, upload, polling, and
download durations are never relabeled as QPU time.

The currently installed CQLib API and current Tianyan SDK result schema do not
document a QPU-duration field. Therefore the guaranteed timing output is a
calibrated physical gate-schedule duration computed from mapped QCIS and the
live Tianyan-287 X/2, CZ, identity, and readout calibration. The primary
gate-only total excludes the terminal boundary readout. If the live result adds
an explicit execution-duration field, the runner also reports and aggregates
that measured value by repeat.

The live machine configuration exposes topology and calibration but no maximum
circuit depth/time field. Historical result files in this directory show that
the earlier monolithic attention jobs were rejected as exceeding the machine
scope, which is the empirical reason for the finer segmentation.

## Pre-submission dry run

The updated mapped dry run completed for all 14 critical-path segments and all
three comparison composites. The calibrated critical-path estimates were:

| Layer | Segment count | Gate-only scheduled duration |
|---|---:|---:|
| Self-attention | 8 | 452,774 ns (452.774 us) |
| FFN | 6 | 302,326 ns (302.326 us) |

The same mapping/calibration pass produced these pre-execution additivity
checks:

| Comparison | Whole task | Segment sum | Whole / sum |
|---|---:|---:|---:|
| Complete FFN | 308,646 ns | 302,326 ns | 1.0209 |
| First FFN half | 150,311 ns | 150,480 ns | 0.9989 |
| Complete attention softmax | 77,303 ns | 78,528 ns | 0.9844 |

These values are pre-execution calibration-based estimates, not measurements
of run-to-run fluctuation. The three hardware repetitions are retained so that
any explicit server-side QPU duration returned at execution time can be used to
quantify fluctuation.
