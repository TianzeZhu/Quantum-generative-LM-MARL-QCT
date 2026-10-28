#!/usr/bin/env python3
"""Benchmark paper-scale FFN and self-attention critical paths on Tianyan-287.

The paper describes extensive register parallelism.  At n=4 token qubits,
m=3 tokens, and p=4 attention heads, this program submits the critical-path
segments rather than one monolithic circuit.  Segment boundaries are timing
boundaries: a separate cloud job cannot preserve an unknown quantum state.
Consequently these jobs measure the hardware cost of the layer's gate schedule,
not a functionally chained end-to-end inference.

Each circuit is submitted three times with 10 shots, the minimum accepted by
Tianyan-287. Server-returned timing
metadata is preserved verbatim.  Independently, the program calculates a pure
gate-schedule duration from the transpiled physical circuit and Tianyan-287's
calibrated pulse lengths.  That estimate excludes network, queueing, polling,
and the terminal readout inserted only to make each segment a valid job.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time as clock_time
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable

import numpy as np
import pennylane as qml

from attention import SU8 as paper_su8
from attention import su16_block
from backend_config import install_cqlib_openqasm_compatibility
from cqlib import QuantumLanguage, TianYanPlatform
from cqlib.mapping import transpile_qcis
from cqlib_adapter.utils.api_client import ApiClient
from cqlib_adapter.utils.converter import pennylane_to_cqlib
from swaptest import swaptest_circuit


BACKEND = "tianyan-287"
SHOTS = 10
REPEATS = 3
TOKEN_QUBITS = 4
TOKEN_COUNT = 3
ATTENTION_HEADS = 4
TIMEZONE_LABEL = "Asia/Hong_Kong"
WINDOW_START = clock_time(6, 0)
WINDOW_END = clock_time(10, 0)


@dataclass(frozen=True)
class SegmentSpec:
    layer: str
    name: str
    order: int
    wire_count: int
    description: str
    build_tape: Callable[[], qml.tape.QuantumScript]
    role: str = "critical_path_segment"
    component_names: tuple[str, ...] = ()
    task_repetitions: int = REPEATS


def _angles(count: int, salt: int) -> np.ndarray:
    idx = np.arange(count, dtype=float)
    return 0.021 * np.sin(0.31 * idx + 0.17 * salt)


def _coherent_nonunitary_diag(params, wires, ancilla) -> None:
    """Fig. 5 dilation, leaving its ancilla for terminal measurement."""

    qml.Hadamard(wires=ancilla)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, True)(
        qml.math.exp(params * (-1j)), wires
    )
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, False)(
        qml.math.exp(params * 1j), wires
    )
    qml.Hadamard(wires=ancilla)
    qml.RZ(-np.pi / 2, wires=ancilla)


def _paper_su16(params, wires, extra_final_control=None) -> None:
    """Fig. 12 SU(16), optionally extending only its final controlled-X gate.

    The maintained nonlinear helper controls the complete unitary when an extra
    control is requested.  The draft instead adds that control only to the last
    multi-controlled X, so the timing runner spells out that final distinction.
    """

    if len(params) < 255 or len(wires) != 4:
        raise ValueError("The paper SU(16) requires 255 parameters and four wires.")
    w0, w1, w2, w3 = wires
    paper_su8(params[:63], [w0, w1, w2])
    qml.ctrl(qml.Toffoli([w1, w2, w3]), w0, True)
    paper_su8(params[63:126], [w0, w1, w3])
    qml.ctrl(qml.Toffoli([w1, w3, w2]), w0, True)
    paper_su8(params[126:189], [w0, w2, w3])
    qml.ctrl(qml.Toffoli([w2, w3, w1]), w0, True)
    paper_su8(params[189:252], [w1, w2, w3])
    controls = [w1]
    control_values = [True]
    if extra_final_control is not None:
        controls.insert(0, extra_final_control)
        control_values.insert(0, True)
    qml.ctrl(
        qml.Toffoli([w2, w3, w0]),
        control=controls,
        control_values=control_values,
    )
    qml.Rot(params[252], params[253], params[254], wires=w3)


def _coherent_head_selector(head_index, control_wires, ancilla) -> None:
    selector = np.zeros(2 ** len(control_wires))
    selector[head_index] = np.pi / 4
    _coherent_nonunitary_diag(selector, control_wires, ancilla)


def _one_controlled_head(
    head_index: int,
    data_wires: list[int],
    work_ancilla: int,
    selector_ancilla: int,
    salt: int,
) -> None:
    """One controlled U-D-V head block from the current model implementation."""

    control_wires = data_wires[:2]
    target_wires = data_wires[2:]
    control_values = [bool((head_index >> bit) & 1) for bit in (1, 0)]
    su_count = su16_block([], target_wires, count_only=True)
    diag_count = 2 ** len(target_wires)
    params = _angles(2 * su_count + diag_count, salt)
    diag_end = su_count + diag_count

    qml.ctrl(
        su16_block,
        control=control_wires,
        control_values=control_values,
    )(params[:su_count], target_wires, 1)

    qml.Hadamard(wires=work_ancilla)
    qml.ctrl(
        qml.DiagonalQubitUnitary,
        control=[work_ancilla, *control_wires],
        control_values=[True, *control_values],
    )(qml.math.exp(params[su_count:diag_end] * (-1j)), target_wires)
    qml.ctrl(
        qml.DiagonalQubitUnitary,
        control=[work_ancilla, *control_wires],
        control_values=[False, *control_values],
    )(qml.math.exp(params[su_count:diag_end] * 1j), target_wires)
    qml.ctrl(
        qml.Hadamard,
        control=control_wires,
        control_values=control_values,
    )(wires=work_ancilla)
    qml.ctrl(
        qml.RZ,
        control=control_wires,
        control_values=control_values,
    )(-np.pi / 2, wires=work_ancilla)
    _coherent_head_selector(head_index, control_wires, selector_ancilla)

    qml.ctrl(
        su16_block,
        control=control_wires,
        control_values=control_values,
    )(params[diag_end:], target_wires, 1)


def _attention_head_tape(head_index: int) -> qml.tape.QuantumScript:
    # Q, K, V and non-cloning copies run in parallel in the paper.  One
    # representative register therefore supplies the critical-path duration.
    register_count = 1
    wires_per_register = TOKEN_QUBITS + 2
    operations = []
    with qml.queuing.AnnotatedQueue() as queue:
        for register_index in range(register_count):
            base = register_index * wires_per_register
            _one_controlled_head(
                head_index=head_index,
                data_wires=list(range(base, base + TOKEN_QUBITS)),
                work_ancilla=base + TOKEN_QUBITS,
                selector_ancilla=base + TOKEN_QUBITS + 1,
                salt=100 * head_index + register_index,
            )
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _attention_head_part_tape(head_index: int, part: str) -> qml.tape.QuantumScript:
    """One timing-only subsegment of a controlled attention-head block."""

    data_wires = list(range(TOKEN_QUBITS))
    control_wires = data_wires[:2]
    target_wires = data_wires[2:]
    work_ancilla = TOKEN_QUBITS
    selector_ancilla = TOKEN_QUBITS + 1
    control_values = [bool((head_index >> bit) & 1) for bit in (1, 0)]
    su_count = su16_block([], target_wires, count_only=True)
    diag_count = 2 ** len(target_wires)
    params = _angles(2 * su_count + diag_count, 100 * head_index)
    diag_end = su_count + diag_count

    with qml.queuing.AnnotatedQueue() as queue:
        if part == "v":
            qml.ctrl(
                su16_block,
                control=control_wires,
                control_values=control_values,
            )(params[:su_count], target_wires, 1)
        elif part == "diag_minus":
            qml.Hadamard(wires=work_ancilla)
            qml.ctrl(
                qml.DiagonalQubitUnitary,
                control=[work_ancilla, *control_wires],
                control_values=[True, *control_values],
            )(qml.math.exp(params[su_count:diag_end] * (-1j)), target_wires)
        elif part == "diag_plus_finish":
            qml.ctrl(
                qml.DiagonalQubitUnitary,
                control=[work_ancilla, *control_wires],
                control_values=[False, *control_values],
            )(qml.math.exp(params[su_count:diag_end] * 1j), target_wires)
            qml.ctrl(
                qml.Hadamard,
                control=control_wires,
                control_values=control_values,
            )(wires=work_ancilla)
            qml.ctrl(
                qml.RZ,
                control=control_wires,
                control_values=control_values,
            )(-np.pi / 2, wires=work_ancilla)
        elif part == "selector":
            _coherent_head_selector(head_index, control_wires, selector_ancilla)
        elif part == "u":
            qml.ctrl(
                su16_block,
                control=control_wires,
                control_values=control_values,
            )(params[diag_end:], target_wires, 1)
        else:
            raise ValueError(f"Unknown attention-head part: {part}")
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _mixing_round_tape(round_index: int) -> qml.tape.QuantumScript:
    # The three output blocks in one round run in parallel.  One representative
    # block supplies that round's critical-path duration.
    block_width = 4 * TOKEN_QUBITS + 1
    with qml.queuing.AnnotatedQueue() as queue:
        for output_index in range(1):
            base = output_index * block_width
            query = list(range(base, base + TOKEN_QUBITS))
            key = list(range(base + TOKEN_QUBITS, base + 2 * TOKEN_QUBITS))
            accumulator = list(
                range(base + 2 * TOKEN_QUBITS, base + 3 * TOKEN_QUBITS)
            )
            other_value = list(
                range(base + 3 * TOKEN_QUBITS, base + 4 * TOKEN_QUBITS)
            )
            ancilla = base + 4 * TOKEN_QUBITS
            swaptest_circuit(query, key, ancilla)
            params = _angles(3 * (2 * TOKEN_QUBITS + 1), 300 + round_index * 10 + output_index)
            qml.StronglyEntanglingLayers(
                params.reshape(1, 2 * TOKEN_QUBITS + 1, 3),
                wires=[ancilla, *accumulator, *other_value],
            )
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _nonlinear_half_tape(part: int, salt: int) -> qml.tape.QuantumScript:
    # The current code's softmax and sigmoid templates are structurally the
    # same.  Token blocks run in parallel, so one representative block is used.
    block_width = TOKEN_QUBITS + 1
    su_count = 4**TOKEN_QUBITS - 1
    with qml.queuing.AnnotatedQueue() as queue:
        for token_index in range(1):
            base = token_index * block_width
            data = list(range(base, base + TOKEN_QUBITS))
            ancilla = base + TOKEN_QUBITS
            params = _angles(su_count, salt + token_index)
            if part == 1:
                qml.Hadamard(wires=ancilla)
                _paper_su16(params, data, extra_final_control=ancilla)
            elif part == 2:
                _paper_su16(
                    params,
                    data[:-1] + [ancilla],
                    extra_final_control=data[-1],
                )
            else:
                raise ValueError("part must be 1 or 2")
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _paper_nonlinear_block(data: list[int], ancilla: int, salt: int) -> None:
    """Complete draft softmax/sigmoid block with one deferred ancilla."""

    su_count = 4**TOKEN_QUBITS - 1
    qml.Hadamard(wires=ancilla)
    _paper_su16(_angles(su_count, salt + 10), data, extra_final_control=ancilla)
    _paper_su16(
        _angles(su_count, salt + 20),
        data[:-1] + [ancilla],
        extra_final_control=data[-1],
    )


def _ffn_linear_block(linear_index: int, data: list[int], ancilla: int) -> None:
    su_count = 4**TOKEN_QUBITS - 1
    diag_count = 2**TOKEN_QUBITS
    params = _angles(2 * su_count + diag_count, 600 + 10 * linear_index)
    diag_end = su_count + diag_count
    _paper_su16(params[:su_count], data)
    _coherent_nonunitary_diag(params[su_count:diag_end], data, ancilla)
    _paper_su16(params[diag_end:], data)


def _ffn_linear_tape(linear_index: int) -> qml.tape.QuantumScript:
    with qml.queuing.AnnotatedQueue() as queue:
        _ffn_linear_block(
            linear_index,
            list(range(TOKEN_QUBITS)),
            TOKEN_QUBITS,
        )
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _ffn_linear_part_tape(linear_index: int, part: str) -> qml.tape.QuantumScript:
    data = list(range(TOKEN_QUBITS))
    ancilla = TOKEN_QUBITS
    su_count = 4**TOKEN_QUBITS - 1
    diag_count = 2**TOKEN_QUBITS
    params = _angles(2 * su_count + diag_count, 600 + 10 * linear_index)
    diag_end = su_count + diag_count
    with qml.queuing.AnnotatedQueue() as queue:
        if part == "v":
            _paper_su16(params[:su_count], data)
        elif part == "diag":
            _coherent_nonunitary_diag(params[su_count:diag_end], data, ancilla)
        elif part == "u":
            _paper_su16(params[diag_end:], data)
        else:
            raise ValueError(f"Unknown FFN linear part: {part}")
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _ffn_composite_tape(first_half_only: bool = False) -> qml.tape.QuantumScript:
    """One coherent job for the complete FFN, or its first linear/activation half.

    Fresh ancillas defer the paper's intermediate postselections to the terminal
    measurement without reusing an entangled work qubit.
    """

    data = list(range(TOKEN_QUBITS))
    with qml.queuing.AnnotatedQueue() as queue:
        _ffn_linear_block(1, data, 4)
        _paper_nonlinear_block(data, 5, 800)
        if not first_half_only:
            _ffn_linear_block(2, data, 6)
            _paper_nonlinear_block(data, 7, 900)
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _attention_softmax_composite_tape() -> qml.tape.QuantumScript:
    data = list(range(TOKEN_QUBITS))
    with qml.queuing.AnnotatedQueue() as queue:
        _paper_nonlinear_block(data, TOKEN_QUBITS, 400)
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def _su16_scaling_tape(repeat_count: int) -> qml.tape.QuantumScript:
    params = _angles(4**TOKEN_QUBITS - 1, 1200)
    with qml.queuing.AnnotatedQueue() as queue:
        for _ in range(repeat_count):
            _paper_su16(params, list(range(TOKEN_QUBITS)))
    operations = qml.tape.QuantumScript.from_queue(queue).operations
    return qml.tape.QuantumScript(operations, shots=SHOTS)


def build_segments() -> list[SegmentSpec]:
    segments = []
    order = 0
    for head_index in range(ATTENTION_HEADS):
        segments.append(
            SegmentSpec(
                layer="attention",
                name=f"attention_head_{head_index}",
                order=order,
                wire_count=TOKEN_QUBITS + 2,
                description=(
                    f"controlled U-D-V attention head {head_index}; representative "
                    "critical path (Q/K/V registers and token copies run in parallel)"
                ),
                build_tape=lambda head_index=head_index: _attention_head_tape(head_index),
            )
        )
        order += 1
    for round_index in range(TOKEN_COUNT - 1):
        segments.append(
            SegmentSpec(
                layer="attention",
                name=f"attention_mixing_round_{round_index}",
                order=order,
                wire_count=4 * TOKEN_QUBITS + 1,
                description=(
                    f"SwapTest and StronglyEntangling round {round_index}; "
                    "representative critical path (three output blocks run in parallel)"
                ),
                build_tape=lambda round_index=round_index: _mixing_round_tape(round_index),
            )
        )
        order += 1
    for part in (1, 2):
        segments.append(
            SegmentSpec(
                layer="attention",
                name=f"attention_softmax_part_{part}",
                order=order,
                wire_count=TOKEN_QUBITS + 1,
                description=(
                    f"softmax template part {part}; representative critical path "
                    "(three tokens run in parallel)"
                ),
                build_tape=lambda part=part: _nonlinear_half_tape(part, 400 + part * 10),
            )
        )
        order += 1

    order = 0
    for linear_index in (1, 2):
        segments.append(
            SegmentSpec(
                layer="ffn",
                name=f"ffn_gw{linear_index}",
                order=order,
                wire_count=TOKEN_QUBITS + 1,
                description=(
                    f"G_W{linear_index}=U-D-V; representative critical path "
                    "(three tokens run in parallel)"
                ),
                build_tape=lambda linear_index=linear_index: _ffn_linear_tape(linear_index),
            )
        )
        order += 1
        for part in (1, 2):
            segments.append(
                SegmentSpec(
                    layer="ffn",
                    name=f"ffn_sigmoid_{linear_index}_part_{part}",
                    order=order,
                    wire_count=TOKEN_QUBITS + 1,
                    description=(
                        f"sigmoid after G_W{linear_index}, template part {part}, "
                        "representative critical path (three tokens run in parallel)"
                    ),
                    build_tape=lambda linear_index=linear_index, part=part: _nonlinear_half_tape(
                        part, 700 + 100 * linear_index + 10 * part
                    ),
                )
            )
            order += 1
    return segments


def build_comparison_circuits() -> list[SegmentSpec]:
    """Composite jobs used to quantify whole-task versus segmented timing."""

    return [
        SegmentSpec(
            layer="ffn",
            name="ffn_whole_layer",
            order=-3,
            wire_count=8,
            description=(
                "complete draft FFN in one job: G_W1, sigmoid, G_W2, sigmoid; "
                "four fresh ancillas defer the four postselections"
            ),
            build_tape=_ffn_composite_tape,
            role="comparison_composite",
            component_names=(
                "ffn_gw1",
                "ffn_sigmoid_1_part_1",
                "ffn_sigmoid_1_part_2",
                "ffn_gw2",
                "ffn_sigmoid_2_part_1",
                "ffn_sigmoid_2_part_2",
            ),
        ),
        SegmentSpec(
            layer="ffn",
            name="ffn_first_half",
            order=-2,
            wire_count=6,
            description="G_W1 and its complete sigmoid in one comparison job",
            build_tape=lambda: _ffn_composite_tape(first_half_only=True),
            role="comparison_composite",
            component_names=(
                "ffn_gw1",
                "ffn_sigmoid_1_part_1",
                "ffn_sigmoid_1_part_2",
            ),
        ),
        SegmentSpec(
            layer="attention",
            name="attention_softmax_whole",
            order=-1,
            wire_count=TOKEN_QUBITS + 1,
            description="both draft softmax parameterized circuits in one comparison job",
            build_tape=_attention_softmax_composite_tape,
            role="comparison_composite",
            component_names=(
                "attention_softmax_part_1",
                "attention_softmax_part_2",
            ),
        ),
    ]


def build_circuits() -> list[SegmentSpec]:
    # Submission order is intentional: try the complete FFN before every other
    # circuit, as it may fit Tianyan-287 without segmentation.
    return build_comparison_circuits() + build_segments()


def build_recovery_segments() -> list[SegmentSpec]:
    """Finer timing segments for blocks rejected by Tianyan's runtime limit."""

    segments = []
    head_parts = ("v", "diag_minus", "diag_plus_finish", "selector", "u")
    for head_index in range(ATTENTION_HEADS):
        parent = f"attention_head_{head_index}"
        for part in head_parts:
            segments.append(
                SegmentSpec(
                    layer="attention",
                    name=f"{parent}_{part}",
                    order=len(segments),
                    wire_count=TOKEN_QUBITS + 2,
                    description=f"recovery subsegment {part} of {parent}",
                    build_tape=lambda head_index=head_index, part=part: _attention_head_part_tape(
                        head_index, part
                    ),
                    role="recovery_segment",
                    component_names=(parent,),
                )
            )
    for linear_index in (1, 2):
        parent = f"ffn_gw{linear_index}"
        for part in ("v", "diag", "u"):
            segments.append(
                SegmentSpec(
                    layer="ffn",
                    name=f"{parent}_{part}",
                    order=len(segments),
                    wire_count=TOKEN_QUBITS + 1,
                    description=f"recovery subsegment {part} of {parent}",
                    build_tape=lambda linear_index=linear_index, part=part: _ffn_linear_part_tape(
                        linear_index, part
                    ),
                    role="recovery_segment",
                    component_names=(parent,),
                )
            )
    return segments


def build_su16_scaling_circuits(
    min_repeat: int = 1, max_repeat: int = 5
) -> list[SegmentSpec]:
    if min_repeat not in range(1, 6):
        raise ValueError("min_repeat must be between 1 and 5.")
    if max_repeat not in range(1, 6):
        raise ValueError("max_repeat must be between 1 and 5.")
    if min_repeat > max_repeat:
        raise ValueError("min_repeat must not exceed max_repeat.")
    return [
        SegmentSpec(
            layer="comparison",
            name=f"su16_repeat_{repeat_count}",
            order=repeat_count - 1,
            wire_count=TOKEN_QUBITS,
            description=f"paper SU(16) repeated {repeat_count} time(s) sequentially",
            build_tape=lambda repeat_count=repeat_count: _su16_scaling_tape(repeat_count),
            role="scaling_comparison",
            task_repetitions=1,
        )
        for repeat_count in range(min_repeat, max_repeat + 1)
    ]


def _parameter_entry(entry: Any) -> tuple[list[Any], list[str], str]:
    if not isinstance(entry, dict):
        return [], [], ""
    return (
        list(entry.get("param_list", [])),
        [str(item) for item in entry.get("qubit_used", [])],
        str(entry.get("unit", "")),
    )


def _calibration_maps(config: dict) -> dict:
    x_values, x_qubits, x_unit = _parameter_entry(
        config["qubit"]["singleQubit"]["X/2 length"]
    )
    x_half_ns = dict(zip(x_qubits, map(float, x_values)))
    if x_unit.lower() != "ns":
        raise ValueError(f"Unexpected X/2 length unit: {x_unit!r}")

    cz_values, cz_couplers, cz_unit = _parameter_entry(
        config["twoQubitGate"]["czGate"]["length"]
    )
    if cz_unit.lower() != "ns":
        raise ValueError(f"Unexpected CZ length unit: {cz_unit!r}")
    coupler_pairs = config["overview"]["coupler_map"]
    cz_ns = {}
    for coupler, value in zip(cz_couplers, cz_values):
        pair = frozenset(coupler_pairs[coupler])
        cz_ns[pair] = float(value)

    readout_ns = {}
    readout_groups = dict(config["readout"].get("readout_wv_setting1_length", []))
    for qubit_index in range(1, len(config["overview"]["qubits"]) + 1):
        group = f"R{math.ceil(qubit_index / 7):02d}"
        slot = (qubit_index - 1) % 7
        values = readout_groups.get(group, [])
        if slot < len(values):
            readout_ns[f"Q{qubit_index}"] = float(values[slot])
    return {"x_half_ns": x_half_ns, "cz_ns": cz_ns, "readout_ns": readout_ns}


def _instruction_duration_ns(instruction_data, calibration: dict) -> tuple[float, bool]:
    name = str(instruction_data.instruction.name).upper()
    qubits = [str(q) for q in instruction_data.qubits]
    params = list(instruction_data.instruction.params or [])
    x_half = calibration["x_half_ns"]
    if name in {"RZ", "Z", "S", "SD", "T", "TD", "B"}:
        return 0.0, True
    if name in {"X2P", "X2M", "Y2P", "Y2M", "XY2P", "XY2M", "H"}:
        return max(x_half[q] for q in qubits), True
    if name in {"X", "Y", "RX", "RY", "RXY"}:
        return 2.0 * max(x_half[q] for q in qubits), True
    if name == "CZ":
        value = calibration["cz_ns"].get(frozenset(qubits))
        return (float(value), True) if value is not None else (0.0, False)
    if name == "SWAP":
        # Tianyan's mapper inserts logical SWAPs on a calibrated CZ edge.
        # The server decomposition is three CNOTs; each CNOT is H-CZ-H.
        cz_value = calibration["cz_ns"].get(frozenset(qubits))
        if cz_value is None:
            return 0.0, False
        return 3.0 * float(cz_value) + 6.0 * max(x_half[q] for q in qubits), True
    if name == "I":
        return (0.5 * float(params[0]), True) if params else (0.0, False)
    if name == "M":
        values = [calibration["readout_ns"].get(q) for q in qubits]
        return (max(values), True) if values and all(v is not None for v in values) else (0.0, False)
    return 0.0, False


def scheduled_duration(compiled_circuit, machine_config: dict) -> dict:
    calibration = _calibration_maps(machine_config)
    ready_gate = defaultdict(float)
    ready_with_readout = defaultdict(float)
    unknown = Counter()
    operation_counts = Counter()
    gate_instruction_count = 0
    readout_instruction_count = 0

    for item in compiled_circuit.circuit_data:
        name = str(item.instruction.name).upper()
        operation_counts[name] += 1
        qubits = [str(q) for q in item.qubits]
        duration, known = _instruction_duration_ns(item, calibration)
        if not known:
            unknown[name] += 1
        is_readout = name == "M"
        ready = ready_with_readout if is_readout else ready_gate
        if not is_readout:
            # Keep the all-in duration synchronized with the gate schedule.
            start = max((ready_with_readout[q] for q in qubits), default=0.0)
            finish = start + duration
            for qubit in qubits:
                ready_gate[qubit] = finish
                ready_with_readout[qubit] = finish
            gate_instruction_count += 1
        else:
            start = max((ready_with_readout[q] for q in qubits), default=0.0)
            finish = start + duration
            for qubit in qubits:
                ready_with_readout[qubit] = finish
            readout_instruction_count += 1

    return {
        "gate_only_ns": max(ready_gate.values(), default=0.0),
        "including_boundary_readout_ns": max(ready_with_readout.values(), default=0.0),
        "gate_instruction_count": gate_instruction_count,
        "readout_instruction_count": readout_instruction_count,
        "operation_counts": dict(sorted(operation_counts.items())),
        "unknown_duration_operations": dict(sorted(unknown.items())),
        "calibration_time": machine_config.get("calibrationTime"),
    }


def compile_segments(
    platform: TianYanPlatform,
    machine_config: dict,
    specs: Iterable[SegmentSpec] | None = None,
) -> list[dict]:
    install_cqlib_openqasm_compatibility()
    compiled = []
    for spec in specs or build_circuits():
        tape = spec.build_tape()
        logical = pennylane_to_cqlib(tape=tape, decompose=True)
        transpiled = transpile_qcis(
            logical.qcis,
            platform,
            seed=1200 if spec.role == "scaling_comparison" else None,
        )
        physical = transpiled[0]
        physical.measure_all()
        timing = scheduled_duration(physical, machine_config)
        if timing["unknown_duration_operations"]:
            raise RuntimeError(
                f"Cannot claim pure scheduled duration for {spec.name}; unknown "
                f"native operations: {timing['unknown_duration_operations']}"
            )
        compiled.append(
            {
                "spec": spec,
                "logical_qubits": logical.num_qubits,
                "logical_depth": logical.depth(),
                "logical_instruction_count": len(logical.circuit_data),
                "physical_qubits": physical.num_qubits,
                "physical_depth": physical.depth(),
                "physical_instruction_count": len(physical.circuit_data),
                "qcis": physical.qcis,
                "scheduled_timing": timing,
            }
        )
    return compiled


_TIMING_KEY = re.compile(
    r"(time|duration|elapsed|queue|wait|start|finish|end|execute|execution|runtime|qpu)",
    re.IGNORECASE,
)


def find_api_timing_fields(value: Any, path: str = "") -> dict[str, Any]:
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _TIMING_KEY.search(str(key)) and not isinstance(child, (dict, list)):
                found[child_path] = child
            found.update(find_api_timing_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(find_api_timing_fields(child, f"{path}[{index}]"))
    return found


def _explicit_qpu_duration(timing_fields: dict[str, Any]) -> dict | None:
    accepted = {
        "qputime",
        "qpuduration",
        "quantumexecutiontime",
        "quantumexecutionduration",
        "executionduration",
        "executetime",
    }
    for path, value in timing_fields.items():
        key = re.sub(r"[^a-z]", "", path.rsplit(".", 1)[-1].lower())
        if key not in accepted or not isinstance(value, (int, float)):
            continue
        unit = "ms"
        lower_path = path.lower()
        if lower_path.endswith(("_ns", "ns")):
            unit = "ns"
        elif lower_path.endswith(("_us", "us")):
            unit = "us"
        elif lower_path.endswith(("_s", "seconds")):
            unit = "s"
        return {"path": path, "value": float(value), "unit": unit}
    return None


def _duration_to_ns(duration: dict) -> float:
    factors = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
    return float(duration["value"]) * factors[duration["unit"]]


def _task_id(raw_result: dict) -> str | None:
    for key in ("experimentTaskId", "query_id", "queryId", "taskId"):
        if key in raw_result and raw_result[key] is not None:
            return str(raw_result[key])
    return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_payload(compiled: list[dict], machine_row: Any) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "timezone": TIMEZONE_LABEL,
        "backend": BACKEND,
        "shots": SHOTS,
        "repeats": REPEATS,
        "token_qubits": TOKEN_QUBITS,
        "token_count": TOKEN_COUNT,
        "attention_heads": ATTENTION_HEADS,
        "machine_row": machine_row,
        "method": {
            "kind": "segmented_critical_path_timing",
            "functional_end_to_end": False,
            "reason": (
                "Independent cloud jobs cannot carry an unknown quantum state across "
                "segment boundaries. Durations are summed along the paper's critical path."
            ),
            "primary_duration": (
                "explicit API QPU duration when unambiguously named; otherwise the "
                "calibration-based physical gate schedule excluding boundary readout"
            ),
            "submission_order": (
                "complete FFN first, then comparison composites, then individual "
                "critical-path segments"
            ),
        },
        "segments": [
            {
                "layer": item["spec"].layer,
                "name": item["spec"].name,
                "order": item["spec"].order,
                "role": item["spec"].role,
                "component_names": list(item["spec"].component_names),
                "task_repetitions": item["spec"].task_repetitions,
                "description": item["spec"].description,
                "declared_wire_count": item["spec"].wire_count,
                "logical_qubits": item["logical_qubits"],
                "logical_depth": item["logical_depth"],
                "logical_instruction_count": item["logical_instruction_count"],
                "physical_qubits": item["physical_qubits"],
                "physical_depth": item["physical_depth"],
                "physical_instruction_count": item["physical_instruction_count"],
                "scheduled_timing": item["scheduled_timing"],
                "query_ids": [],
                "submission_error": None,
            }
            for item in compiled
        ],
    }


def _machine_row(platform: TianYanPlatform) -> Any:
    rows = platform.query_quantum_computer_list()
    return next((row for row in rows if BACKEND in str(row)), None)


def _inside_window(now: datetime | None = None) -> bool:
    now = now or datetime.now().astimezone()
    return WINDOW_START <= now.time().replace(tzinfo=None) < WINDOW_END


def _wait_for_machine(platform: TianYanPlatform, deadline: clock_time) -> Any:
    row = _machine_row(platform)
    status = str(row).lower() if row is not None else ""
    if "running" in status or "free" in status:
        return row
    while datetime.now().time() < deadline:
        row = _machine_row(platform)
        status = str(row).lower() if row is not None else ""
        if "running" in status or "free" in status:
            return row
        print(f"{datetime.now().isoformat(timespec='seconds')}: {BACKEND} not ready: {row}")
        time.sleep(30)
    raise TimeoutError(f"{BACKEND} did not become available before {deadline}.")


def submit(compiled: list[dict], platform: TianYanPlatform, manifest: dict, run_dir: Path) -> None:
    for item, segment in zip(compiled, manifest["segments"]):
        try:
            query_ids = platform.submit_experiment(
                circuit=[item["qcis"]] * item["spec"].task_repetitions,
                language=QuantumLanguage.QCIS,
                name=item["spec"].name,
                lab_name=f"qtransformer_n4_m3_{run_dir.name}",
                num_shots=SHOTS,
                machine_name=BACKEND,
                is_verify=True,
            )
            segment["query_ids"] = [str(query_id) for query_id in query_ids]
            print(f"submitted {item['spec'].name}: {len(query_ids)} task(s)")
        except Exception as exc:  # keep later segments from being lost
            segment["submission_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"submission failed for {item['spec'].name}: {exc}")
        _write_json(run_dir / "manifest.json", manifest)


def collect(manifest: dict, run_dir: Path, login_key: str, deadline: clock_time) -> None:
    expected = {
        query_id
        for segment in manifest["segments"]
        for query_id in segment["query_ids"]
    }
    if not expected:
        return
    client = ApiClient(token=login_key)
    raw_by_id = {}
    while datetime.now().time() < deadline and set(raw_by_id) != expected:
        pending = sorted(expected - set(raw_by_id))
        for start in range(0, len(pending), 50):
            batch = pending[start : start + 50]
            try:
                raw_results = client.query_job(batch) or []
            except Exception as exc:
                print(f"batch result query failed; isolating task statuses: {exc}")
                raw_results = []
                for query_id in batch:
                    try:
                        raw_results.extend(client.query_job([query_id]) or [])
                    except Exception as single_exc:
                        message = str(single_exc)
                        if "tasks have failed" in message.lower() or "run failure" in message.lower():
                            raw_by_id[query_id] = {
                                "experimentTaskId": query_id,
                                "terminalFailure": True,
                                "collectionError": message,
                            }
                        else:
                            print(f"result query failed for {query_id}: {single_exc}")
            for raw in raw_results:
                query_id = _task_id(raw)
                if query_id is not None:
                    raw_by_id[query_id] = raw
        _write_json(run_dir / "raw_results.json", raw_by_id)
        print(f"completed {len(raw_by_id)}/{len(expected)} task(s)")
        if set(raw_by_id) != expected:
            time.sleep(15)

    results = []
    for segment in manifest["segments"]:
        for repeat_index, query_id in enumerate(segment["query_ids"], start=1):
            raw = raw_by_id.get(query_id)
            timing_fields = find_api_timing_fields(raw) if raw is not None else {}
            results.append(
                {
                    "layer": segment["layer"],
                    "segment": segment["name"],
                    "role": segment["role"],
                    "component_names": segment["component_names"],
                    "repeat": repeat_index,
                    "query_id": query_id,
                    "complete": raw is not None and not raw.get("terminalFailure", False),
                    "server_failed": bool(raw and raw.get("terminalFailure", False)),
                    "collection_error": raw.get("collectionError") if raw else None,
                    "api_timing_fields": timing_fields,
                    "explicit_qpu_duration": _explicit_qpu_duration(timing_fields),
                    "scheduled_gate_only_ns": segment["scheduled_timing"]["gate_only_ns"],
                    "scheduled_including_boundary_readout_ns": segment["scheduled_timing"][
                        "including_boundary_readout_ns"
                    ],
                }
            )
    _write_json(run_dir / "results.json", results)
    _write_json(run_dir / "summary.json", summarize(manifest, results))


def summarize(manifest: dict, results: list[dict]) -> dict:
    layers = {}
    for layer in ("attention", "ffn"):
        layer_segments = [
            s
            for s in manifest["segments"]
            if s["layer"] == layer and s["role"] == "critical_path_segment"
        ]
        scheduled_ns = sum(s["scheduled_timing"]["gate_only_ns"] for s in layer_segments)
        layer_results = [
            row
            for row in results
            if row["layer"] == layer and row["role"] == "critical_path_segment"
        ]
        explicit = [
            row["explicit_qpu_duration"]
            for row in layer_results
            if row["explicit_qpu_duration"] is not None
        ]
        measured_repeat_ns = []
        expected_segment_names = {segment["name"] for segment in layer_segments}
        for repeat in range(1, REPEATS + 1):
            repeat_rows = [row for row in layer_results if row["repeat"] == repeat]
            measured_names = {
                row["segment"]
                for row in repeat_rows
                if row["explicit_qpu_duration"] is not None
            }
            if measured_names == expected_segment_names:
                measured_repeat_ns.append(
                    sum(_duration_to_ns(row["explicit_qpu_duration"]) for row in repeat_rows)
                )
        layers[layer] = {
            "segment_count": len(layer_segments),
            "repeat_count_per_segment": REPEATS,
            "complete_task_count": sum(
                1 for row in results if row["layer"] == layer and row["complete"]
            ),
            "critical_path_scheduled_gate_only_ns": scheduled_ns,
            "critical_path_scheduled_gate_only_us": scheduled_ns / 1000.0,
            "explicit_api_qpu_duration_records": explicit,
            "critical_path_api_qpu_duration_by_repeat_ns": measured_repeat_ns,
            "critical_path_api_qpu_duration_mean_ns": (
                float(np.mean(measured_repeat_ns)) if measured_repeat_ns else None
            ),
            "critical_path_api_qpu_duration_std_ns": (
                float(np.std(measured_repeat_ns)) if measured_repeat_ns else None
            ),
        }
    comparisons = {}
    for composite in (
        segment
        for segment in manifest["segments"]
        if segment["role"] == "comparison_composite"
    ):
        components = [
            segment
            for segment in manifest["segments"]
            if segment["name"] in composite["component_names"]
        ]
        whole_scheduled_ns = composite["scheduled_timing"]["gate_only_ns"]
        segmented_scheduled_ns = sum(
            component["scheduled_timing"]["gate_only_ns"] for component in components
        )
        measured = []
        for repeat in range(1, REPEATS + 1):
            whole_row = next(
                (
                    row
                    for row in results
                    if row["segment"] == composite["name"] and row["repeat"] == repeat
                ),
                None,
            )
            component_rows = [
                row
                for row in results
                if row["segment"] in composite["component_names"]
                and row["repeat"] == repeat
            ]
            if (
                whole_row is None
                or whole_row["explicit_qpu_duration"] is None
                or len(component_rows) != len(composite["component_names"])
                or any(row["explicit_qpu_duration"] is None for row in component_rows)
            ):
                continue
            whole_ns = _duration_to_ns(whole_row["explicit_qpu_duration"])
            segmented_ns = sum(
                _duration_to_ns(row["explicit_qpu_duration"]) for row in component_rows
            )
            measured.append(
                {
                    "repeat": repeat,
                    "whole_task_ns": whole_ns,
                    "segmented_sum_ns": segmented_ns,
                    "whole_minus_segmented_ns": whole_ns - segmented_ns,
                    "whole_over_segmented_ratio": (
                        whole_ns / segmented_ns if segmented_ns else None
                    ),
                }
            )
        comparisons[composite["name"]] = {
            "component_names": composite["component_names"],
            "whole_task_scheduled_gate_only_ns": whole_scheduled_ns,
            "segmented_sum_scheduled_gate_only_ns": segmented_scheduled_ns,
            "scheduled_whole_minus_segmented_ns": (
                whole_scheduled_ns - segmented_scheduled_ns
            ),
            "scheduled_whole_over_segmented_ratio": (
                whole_scheduled_ns / segmented_scheduled_ns
                if segmented_scheduled_ns
                else None
            ),
            "api_qpu_duration_comparisons": measured,
        }
    return {
        "backend": BACKEND,
        "shots": SHOTS,
        "token_qubits": TOKEN_QUBITS,
        "token_count": TOKEN_COUNT,
        "attention_heads": ATTENTION_HEADS,
        "functional_end_to_end": False,
        "layers": layers,
        "whole_vs_segmented_comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "tianyan_benchmark_outputs",
    )
    parser.add_argument(
        "--allow-outside-window",
        action="store_true",
        help="Override the 06:00-10:00 local submission guard.",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Submit and save query IDs without waiting for results.",
    )
    parser.add_argument(
        "--collect-manifest",
        type=Path,
        help="Resume result collection for an existing manifest without resubmitting.",
    )
    parser.add_argument(
        "--recovery-only",
        action="store_true",
        help="Submit only finer subsegments of runtime-rejected head/GW blocks.",
    )
    parser.add_argument(
        "--su16-scaling",
        action="store_true",
        help="Submit one task each for one through five sequential paper SU(16) blocks.",
    )
    parser.add_argument(
        "--su16-min-repeat",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Lowest SU(16) repetition count to submit (default: 1).",
    )
    parser.add_argument(
        "--su16-max-repeat",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Highest SU(16) repetition count to submit (default: 5).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    login_key = os.environ.get("TIANYAN_LOGIN_KEY", "").strip()
    if not login_key:
        raise RuntimeError("TIANYAN_LOGIN_KEY is not configured.")
    if args.collect_manifest is not None:
        manifest_path = args.collect_manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collect(manifest, manifest_path.parent, login_key, clock_time(9, 55))
        return 0
    if args.submit and not args.allow_outside_window and not _inside_window():
        raise RuntimeError(
            f"Submissions are guarded to 06:00-10:00 {TIMEZONE_LABEL}; current "
            f"local time is {datetime.now().isoformat(timespec='seconds')}."
        )

    platform = TianYanPlatform(login_key=login_key, machine_name=BACKEND)
    machine_row = _machine_row(platform)
    machine_config = platform.download_config(machine=BACKEND)
    if len(machine_config["overview"]["qubits"]) != 105:
        raise RuntimeError("Live Tianyan-287 configuration did not report 105 qubits.")

    if args.su16_scaling:
        specs = build_su16_scaling_circuits(
            args.su16_min_repeat, args.su16_max_repeat
        )
    elif args.recovery_only:
        specs = build_recovery_segments()
    else:
        specs = None
    compiled = compile_segments(platform, machine_config, specs=specs)
    manifest = _manifest_payload(compiled, machine_row)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "manifest.json", manifest)
    for item in compiled:
        (run_dir / "qcis").mkdir(exist_ok=True)
        (run_dir / "qcis" / f"{item['spec'].name}.qcis").write_text(
            item["qcis"], encoding="utf-8"
        )

    print(f"compiled {len(compiled)} segments; manifest: {run_dir / 'manifest.json'}")
    if not args.submit:
        _write_json(run_dir / "summary.json", summarize(manifest, []))
        return 0

    if args.allow_outside_window and not _inside_window():
        # Diagnostic submissions outside the service window are explicitly
        # requested by the caller; let Tianyan return its authoritative error.
        machine_row = _machine_row(platform)
    else:
        machine_row = _wait_for_machine(platform, clock_time(9, 30))
    manifest["machine_row_at_submission"] = machine_row
    manifest["submission_started_at"] = datetime.now().astimezone().isoformat()
    submit(compiled, platform, manifest, run_dir)
    manifest["submission_finished_at"] = datetime.now().astimezone().isoformat()
    _write_json(run_dir / "manifest.json", manifest)
    if not args.no_collect:
        collect(manifest, run_dir, login_key, clock_time(9, 55))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
