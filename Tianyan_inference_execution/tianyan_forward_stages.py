#!/usr/bin/env python3
"""Submit staged forward-only checks of the parallel model to TianYan.

Stages are intentionally submitted one at a time:

1. ``su16``: the four-qubit 255-parameter unitary block used by the model;
2. ``ffn``: one four-qubit FFN block;
3. ``gq``: one four-qubit G_Q with four controlled attention heads;
4. ``mixing``: SwapTest + pairwise StronglyEntangling token mixing;
5. ``attention``: G_Q/G_K/G_V + mixing;
6. ``attention-ffn``: complete attention followed by the shared FFN.

The large stages use ``n=4`` and a selectable token count ``m``.  The requested
default is ``m=3``, providing a 90/92-wire profile that fits Tianyan-287's 105
computational qubits, while
``--token-count 2`` provides a 34/36-wire profile for Tianyan-176's 66
computational qubits.  Because
CQLib exposes deferred rather than native mid-circuit measurements, isolated
FFN/G_Q blocks use fresh ancillas and terminal classical postselection.  The
full-chain hardware smoke stages use theta=0 diagonal blocks and disable the
nonlinear postselection layer so that no deferred-measurement ancillas are added
to the large register. No training or differentiation is performed. A TianYan
device is created only when both ``--tianyan`` and ``--submit`` are present.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pennylane as qml

from attention import controlled_su16_array, su16_block
from backend_config import (
    MAX_DENSE_READOUT_CALIBRATION_QUBITS,
    add_backend_arguments,
    backend_config_from_args,
    create_device,
)
from ffn_new import ffn_new
from my_model import (
    _collect_shared_attention_layout,
    _primary_v_token_wires,
    _qkv_register_layout,
    my_model,
    my_model_with_attention_fnn,
    my_model_with_attention_fnn_parameter_breakdown,
)


TOKEN_QUBITS = 4
TOKEN_COUNT = 3
ATTENTION_HEADS = 4
ATTENTION_REPEAT_TIME = 1
FFN_REPEAT_TIME = 1
ENTANGLING_LAYERS = 1
NONLINEAR_LAYERS1 = 0
NONLINEAR_LAYERS2 = 0
BACKEND_COMPUTATIONAL_QUBITS = {
    "tianyan176": 66,
    "tianyan-287": 105,
}
BACKEND_SAFE_TOKEN_COUNTS = {
    "tianyan176": 2,
    "tianyan-287": 3,
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    wire_count: int
    output_wires: tuple
    parameter_count: int
    prepare: object
    apply: object
    postselect_wires: tuple = ()
    postselect_values: tuple = ()


def _expected_stage_layout(stage_name, token_count=None):
    """Return an independent wire/parameter invariant for a smoke stage."""

    m = TOKEN_COUNT if token_count is None else token_count
    pair_count = m * (m - 1)
    model_wires = (
        2 * pair_count * TOKEN_QUBITS
        + m * m * TOKEN_QUBITS
        + pair_count
    )
    mixing_params = pair_count * (2 * TOKEN_QUBITS + 1) * 3
    layouts = {
        "su16": (TOKEN_QUBITS, 255),
        "ffn": (TOKEN_QUBITS + 1, 526),
        "gq": (TOKEN_QUBITS + 2 * ATTENTION_HEADS, 136),
        "mixing": (model_wires, mixing_params),
        "attention": (model_wires + 2, 3 * 136 + mixing_params),
        "attention-ffn": (
            model_wires + 2,
            3 * 136 + mixing_params + 526,
        ),
    }
    return layouts[stage_name]


def _source_logical_wire_count(stage_name):
    if stage_name == "su16":
        return TOKEN_QUBITS
    if stage_name == "ffn":
        return TOKEN_QUBITS + 1
    if stage_name == "gq":
        return TOKEN_QUBITS + 2
    return _expected_stage_layout(stage_name)[0]


def _fixed_parameters(count):
    """Small deterministic angles keep postselection success reasonably high."""

    indices = np.arange(count, dtype=float)
    return 0.015 * np.sin(0.37 * indices + 0.11)


def _coherent_nonunitary_diag(params, wires, ancilla):
    """Unitary dilation without a mid-circuit measurement.

    The ancilla is measured and postselected classically at the end of the
    hardware circuit.
    """

    qml.Hadamard(wires=ancilla)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, True)(
        qml.math.exp(params * (-1j)), wires
    )
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, False)(
        qml.math.exp(params * 1j), wires
    )
    qml.Hadamard(wires=ancilla)
    qml.RZ(-np.pi / 2, wires=ancilla)


def _coherent_head_selector(position, control_wires, ancilla):
    selector = np.zeros(2 ** len(control_wires))
    selector[position] = np.pi / 4
    _coherent_nonunitary_diag(selector, control_wires, ancilla)


def _apply_gq_terminal_postselection(params, data_wires, ancillas):
    """Four-head G_Q with fresh ancillas and terminal postselection."""

    control_wires = data_wires[:2]
    target_wires = data_wires[2:]
    su_param_count = su16_block([], target_wires, count_only=True)
    diag_param_count = 2 ** len(target_wires)
    params_per_head = 2 * su_param_count + diag_param_count

    if len(ancillas) != 2 * ATTENTION_HEADS:
        raise ValueError("G_Q terminal postselection needs two fresh ancillas per head.")

    for head_index in range(ATTENTION_HEADS):
        control_values = [bool((head_index >> bit) & 1) for bit in (1, 0)]
        start = head_index * params_per_head
        su1_end = start + su_param_count
        diag_end = su1_end + diag_param_count
        su2_end = diag_end + su_param_count
        work_ancilla = ancillas[2 * head_index]
        selector_ancilla = ancillas[2 * head_index + 1]

        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(params[start:su1_end], target_wires, ATTENTION_REPEAT_TIME)

        qml.Hadamard(wires=work_ancilla)
        qml.ctrl(
            qml.DiagonalQubitUnitary,
            control=[work_ancilla, *control_wires],
            control_values=[True, *control_values],
        )(qml.math.exp(params[su1_end:diag_end] * (-1j)), target_wires)
        qml.ctrl(
            qml.DiagonalQubitUnitary,
            control=[work_ancilla, *control_wires],
            control_values=[False, *control_values],
        )(qml.math.exp(params[su1_end:diag_end] * 1j), target_wires)
        qml.ctrl(
            qml.Hadamard,
            control=control_wires,
            control_values=control_values,
        )(wires=work_ancilla)
        qml.RZ(-np.pi / 2, wires=work_ancilla)

        _coherent_head_selector(head_index, control_wires, selector_ancilla)

        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(params[diag_end:su2_end], target_wires, ATTENTION_REPEAT_TIME)


def _apply_controlled_attention_identity_diag(params, data_wires):
    """Apply the four controlled U/V parts for the theta=0 hardware smoke path."""

    control_wires = data_wires[:2]
    target_wires = data_wires[2:]
    su_param_count = su16_block([], target_wires, count_only=True)
    diag_param_count = 2 ** len(target_wires)
    params_per_head = 2 * su_param_count + diag_param_count

    for head_index in range(ATTENTION_HEADS):
        control_values = [bool((head_index >> bit) & 1) for bit in (1, 0)]
        start = head_index * params_per_head
        su1_end = start + su_param_count
        diag_end = su1_end + diag_param_count
        su2_end = diag_end + su_param_count
        if not np.allclose(params[su1_end:diag_end], 0.0):
            raise ValueError("Hardware-safe full attention requires theta=0 diagonals.")
        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(params[start:su1_end], target_wires, ATTENTION_REPEAT_TIME)
        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(params[diag_end:su2_end], target_wires, ATTENTION_REPEAT_TIME)


def _apply_ffn_identity_diag(params, wires):
    """Apply U and V for the theta=0 FFN diagonal used in the full-chain smoke path."""

    su_param_count = su16_block(
        [], wires, repeat_time=FFN_REPEAT_TIME, count_only=True
    )
    diag_end = su_param_count + 2 ** len(wires)
    if not np.allclose(params[su_param_count:diag_end], 0.0):
        raise ValueError("Hardware-safe full FFN requires a theta=0 diagonal.")
    su16_block(params[:su_param_count], wires, FFN_REPEAT_TIME)
    su16_block(params[diag_end : diag_end + su_param_count], wires, FFN_REPEAT_TIME)


def _prepare_single_register(wires, token_index=0):
    """Prepare a shallow normalized test state without a dataset dependency."""

    for local_index, wire in enumerate(wires):
        angle = 0.08 * (token_index + 1) * (local_index + 1)
        qml.RY(angle, wires=wire)
        qml.RZ(-0.5 * angle, wires=wire)


def _prepare_all_qkv_registers(wires):
    layout = _collect_shared_attention_layout(
        wires,
        TOKEN_QUBITS,
        TOKEN_COUNT,
        ATTENTION_HEADS,
    )
    for register_name in ("q_groups", "k_groups", "v_groups"):
        for token_index, token_copies in enumerate(layout[register_name]):
            for copy_wires in token_copies:
                _prepare_single_register(copy_wires, token_index)


def _large_layout():
    _, _, _, model_wires, _, total_wires = _qkv_register_layout(
        TOKEN_QUBITS,
        TOKEN_COUNT,
        ATTENTION_HEADS,
    )
    wires = list(range(total_wires))
    outputs = tuple(
        wire
        for token_wires in _primary_v_token_wires(
            wires[:model_wires], TOKEN_QUBITS, TOKEN_COUNT
        )
        for wire in token_wires
    )
    return wires, model_wires, total_wires, outputs


def _build_su16_stage():
    """Build the exact four-qubit unitary block used inside FFN/attention."""

    wires = list(range(TOKEN_QUBITS))
    parameter_count = su16_block(
        [],
        wires,
        repeat_time=1,
        count_only=True,
    )

    def prepare():
        _prepare_single_register(wires)

    def apply(params):
        su16_block(params, wires, repeat_time=1)

    return StageSpec(
        name="su16",
        description=(
            "one 4-qubit, 255-parameter SU(16) brick-wall unitary block"
        ),
        wire_count=TOKEN_QUBITS,
        output_wires=tuple(wires),
        parameter_count=parameter_count,
        prepare=prepare,
        apply=apply,
    )


def _build_ffn_stage():
    data_wires = list(range(TOKEN_QUBITS))
    ancilla = TOKEN_QUBITS
    parameter_count = ffn_new(
        [],
        total_qubits=TOKEN_QUBITS,
        num_blocks=ATTENTION_HEADS,
        repeat_time=FFN_REPEAT_TIME,
        wires=data_wires,
        ancilla=[ancilla],
        count_only=True,
    )

    def prepare():
        _prepare_single_register(data_wires)

    def apply(params):
        su_param_count = su16_block(
            [], data_wires, repeat_time=FFN_REPEAT_TIME, count_only=True
        )
        diag_end = su_param_count + 2 ** len(data_wires)
        su16_block(params[:su_param_count], data_wires, FFN_REPEAT_TIME)
        _coherent_nonunitary_diag(
            params[su_param_count:diag_end], data_wires, ancilla
        )
        su16_block(
            params[diag_end : diag_end + su_param_count],
            data_wires,
            FFN_REPEAT_TIME,
        )

    return StageSpec(
        name="ffn",
        description="one 4-qubit U-D-V FFN block",
        wire_count=TOKEN_QUBITS + 1,
        output_wires=tuple(data_wires),
        parameter_count=parameter_count,
        prepare=prepare,
        apply=apply,
        postselect_wires=(ancilla,),
        postselect_values=(0,),
    )


def _build_gq_stage():
    data_wires = list(range(TOKEN_QUBITS))
    logical_ancillas = [TOKEN_QUBITS, TOKEN_QUBITS + 1]
    terminal_ancillas = list(range(TOKEN_QUBITS, TOKEN_QUBITS + 2 * ATTENTION_HEADS))
    parameter_count = controlled_su16_array(
        [],
        total_qubits=TOKEN_QUBITS,
        num_blocks=ATTENTION_HEADS,
        repeat_time=ATTENTION_REPEAT_TIME,
        wires=data_wires,
        ancilla=logical_ancillas,
        count_only=True,
    )

    def prepare():
        # The first two wires select four heads.  Superposition exercises all
        # four control values in one forward circuit.
        qml.Hadamard(wires=data_wires[0])
        qml.Hadamard(wires=data_wires[1])
        _prepare_single_register(data_wires[2:])

    def apply(params):
        _apply_gq_terminal_postselection(params, data_wires, terminal_ancillas)

    return StageSpec(
        name="gq",
        description="one G_Q register with four controlled attention heads",
        wire_count=TOKEN_QUBITS + 2 * ATTENTION_HEADS,
        output_wires=tuple(data_wires),
        parameter_count=parameter_count,
        prepare=prepare,
        apply=apply,
        postselect_wires=tuple(terminal_ancillas),
        postselect_values=(0,) * len(terminal_ancillas),
    )


def _build_mixing_stage():
    wires, model_wire_count, _, output_wires = _large_layout()
    model_wires = wires[:model_wire_count]
    parameter_count = my_model(
        model_wires,
        TOKEN_QUBITS,
        TOKEN_COUNT,
        params=None,
        entangling_layers=ENTANGLING_LAYERS,
        nonlinear_layers1=0,
        nonlinear_layers2=0,
        count_only=True,
    )

    def prepare():
        _prepare_all_qkv_registers(wires)

    def apply(params):
        my_model(
            model_wires,
            TOKEN_QUBITS,
            TOKEN_COUNT,
            params=params,
            entangling_layers=ENTANGLING_LAYERS,
            nonlinear_layers1=0,
            nonlinear_layers2=0,
        )

    return StageSpec(
        name="mixing",
        description=(
            f"{TOKEN_COUNT}-token SwapTest + pairwise "
            "StronglyEntangling mixing only"
        ),
        wire_count=model_wire_count,
        output_wires=output_wires,
        parameter_count=parameter_count,
        prepare=prepare,
        apply=apply,
    )


def _attention_counts(wires):
    breakdown = my_model_with_attention_fnn_parameter_breakdown(
        wires=wires,
        n=TOKEN_QUBITS,
        m=TOKEN_COUNT,
        entangling_layers=ENTANGLING_LAYERS,
        nonlinear_layers1=NONLINEAR_LAYERS1,
        nonlinear_layers2=NONLINEAR_LAYERS2,
        attention_num_blocks=ATTENTION_HEADS,
        attention_repeat_time=ATTENTION_REPEAT_TIME,
        ffn_repeat_time=FFN_REPEAT_TIME,
    )
    attention_count = breakdown["attention"]["total"] + breakdown["model"]["total"]
    return breakdown, attention_count


def _apply_attention_groups_identity_diag(params, token_groups):
    for token_copies in token_groups:
        for copy_wires in token_copies:
            _apply_controlled_attention_identity_diag(params, copy_wires)


def _build_attention_stage(include_ffn):
    wires, model_wire_count, total_wires, output_wires = _large_layout()
    breakdown, attention_count = _attention_counts(wires)
    layout = _collect_shared_attention_layout(
        wires,
        TOKEN_QUBITS,
        TOKEN_COUNT,
        ATTENTION_HEADS,
    )
    q_count = breakdown["attention"]["shared_per_q"]
    k_count = breakdown["attention"]["shared_per_k"]
    v_count = breakdown["attention"]["shared_per_v"]
    mixing_count = breakdown["model"]["entangling_total"]

    if include_ffn:
        parameter_count = my_model_with_attention_fnn(
            wires=wires,
            n=TOKEN_QUBITS,
            m=TOKEN_COUNT,
            params=None,
            entangling_layers=ENTANGLING_LAYERS,
            nonlinear_layers1=NONLINEAR_LAYERS1,
            nonlinear_layers2=NONLINEAR_LAYERS2,
            attention_num_blocks=ATTENTION_HEADS,
            attention_repeat_time=ATTENTION_REPEAT_TIME,
            ffn_repeat_time=FFN_REPEAT_TIME,
            count_only=True,
        )
    else:
        parameter_count = attention_count

    def prepare():
        _prepare_all_qkv_registers(wires)

    def apply_attention_core(params):
        index = 0
        q_params = params[index : index + q_count]
        index += q_count
        k_params = params[index : index + k_count]
        index += k_count
        v_params = params[index : index + v_count]
        index += v_count
        mixing_params = params[index : index + mixing_count]

        _apply_attention_groups_identity_diag(q_params, layout["q_groups"])
        _apply_attention_groups_identity_diag(k_params, layout["k_groups"])
        _apply_attention_groups_identity_diag(v_params, layout["v_groups"])
        my_model(
            wires[:model_wire_count],
            TOKEN_QUBITS,
            TOKEN_COUNT,
            params=mixing_params,
            entangling_layers=ENTANGLING_LAYERS,
            nonlinear_layers1=0,
            nonlinear_layers2=0,
        )

    if include_ffn:

        def apply(params):
            apply_attention_core(params[:attention_count])
            ffn_params = params[attention_count:parameter_count]
            for token_wires in _primary_v_token_wires(
                wires[:model_wire_count], TOKEN_QUBITS, TOKEN_COUNT
            ):
                _apply_ffn_identity_diag(ffn_params, token_wires)

        name = "attention-ffn"
        description = (
            "hardware-safe G_Q/G_K/G_V + mixing + FFN, theta=0 diagonals"
        )
    else:

        def apply(params):
            apply_attention_core(params)

        name = "attention"
        description = "hardware-safe G_Q/G_K/G_V + mixing, theta=0 diagonals"

    return StageSpec(
        name=name,
        description=description,
        wire_count=total_wires,
        output_wires=output_wires,
        parameter_count=parameter_count,
        prepare=prepare,
        apply=apply,
    ), breakdown


def build_stage(stage_name):
    if stage_name == "su16":
        return _build_su16_stage(), None
    if stage_name == "ffn":
        return _build_ffn_stage(), None
    if stage_name == "gq":
        return _build_gq_stage(), None
    if stage_name == "mixing":
        return _build_mixing_stage(), None
    if stage_name == "attention":
        return _build_attention_stage(include_ffn=False)
    if stage_name == "attention-ffn":
        return _build_attention_stage(include_ffn=True)
    raise ValueError(f"Unknown stage: {stage_name}")


def _zero_four_head_diagonals(params, section_start):
    su_count = 15
    diag_count = 4
    params_per_head = 2 * su_count + diag_count
    for head_index in range(ATTENTION_HEADS):
        diag_start = section_start + head_index * params_per_head + su_count
        params[diag_start : diag_start + diag_count] = 0.0


def _parameters_for_stage(spec):
    params = _fixed_parameters(spec.parameter_count)
    if spec.name in {"attention", "attention-ffn"}:
        per_qkv_stage = 136
        for section_start in (0, per_qkv_stage, 2 * per_qkv_stage):
            _zero_four_head_diagonals(params, section_start)
    if spec.name == "attention-ffn":
        attention_count = _expected_stage_layout("attention")[1]
        ffn_diag_start = attention_count + 255
        params[ffn_diag_start : ffn_diag_start + 16] = 0.0
    return params


def _record_top_level_structure(spec, params):
    with qml.tape.QuantumTape() as tape:
        spec.prepare()
        spec.apply(params)
        qml.probs(wires=[*spec.postselect_wires, *spec.output_wires])
    operations = Counter(operation.name for operation in tape.operations)
    return tape, dict(sorted(operations.items()))


def _top_probabilities(probabilities, output_width, count=8):
    indices = np.argsort(probabilities)[::-1][:count]
    return [
        {
            "state": format(int(index), f"0{output_width}b"),
            "probability": float(probabilities[index]),
        }
        for index in indices
    ]


def _postselect_output_probabilities(raw_probabilities, spec):
    postselect_count = len(spec.postselect_wires)
    output_dimension = 2 ** len(spec.output_wires)
    table = np.asarray(raw_probabilities, dtype=float).reshape(
        (2**postselect_count, output_dimension)
    )
    selected_index = 0
    for value in spec.postselect_values:
        selected_index = (selected_index << 1) | int(value)
    accepted = table[selected_index]
    success_probability = float(accepted.sum())
    if success_probability <= 0.0:
        raise RuntimeError(
            "No shots satisfied terminal ancilla postselection; increase --shots."
        )
    return accepted / success_probability, success_probability


def _write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report['stage']}_forward_result.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one forward-only parallel-model stage on TianYan."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "su16",
            "ffn",
            "gq",
            "mixing",
            "attention",
            "attention-ffn",
        ),
    )
    parser.add_argument(
        "--token-count",
        type=int,
        choices=(2, 3, 4),
        default=3,
        help=(
            "Token count m for the large stages. Use 2 for Tianyan-176 "
            "(34/36 wires) or 3 for Tianyan-287 (90/92 wires)."
        ),
    )
    add_backend_arguments(parser)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually create the CQLib device and submit this one forward job.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tianyan_forward_outputs",
    )
    return parser


def main(argv=None):
    global TOKEN_COUNT

    parser = build_parser()
    args = parser.parse_args(argv)
    TOKEN_COUNT = args.token_count
    runtime_config = backend_config_from_args(args)
    if args.submit and not runtime_config.use_tianyan:
        parser.error("--submit requires --tianyan.")
    if not runtime_config.use_tianyan and args.stage in {
        "mixing",
        "attention",
        "attention-ffn",
    }:
        parser.error(
            "The full-register mixing/attention stages are intentionally "
            "disabled on default.qubit. Use --tianyan (and --submit when ready)."
        )

    spec, breakdown = build_stage(args.stage)
    expected_wires, expected_params = _expected_stage_layout(args.stage)
    if (spec.wire_count, spec.parameter_count) != (
        expected_wires,
        expected_params,
    ):
        raise RuntimeError(
            f"Unexpected {args.stage} layout: got "
            f"{spec.wire_count} wires/{spec.parameter_count} parameters, "
            f"expected {expected_wires}/{expected_params}."
        )
    backend_capacity = BACKEND_COMPUTATIONAL_QUBITS.get(runtime_config.backend)
    if (
        runtime_config.use_tianyan
        and backend_capacity is not None
        and spec.wire_count > backend_capacity
    ):
        safe_token_count = BACKEND_SAFE_TOKEN_COUNTS[runtime_config.backend]
        parser.error(
            f"{runtime_config.backend} provides {backend_capacity} computational "
            f"qubits, but stage {args.stage!r} with m={TOKEN_COUNT} requires "
            f"{spec.wire_count}. Re-run with --token-count {safe_token_count}."
        )
    params = _parameters_for_stage(spec)
    tape, operation_counts = _record_top_level_structure(spec, params)
    report = {
        "stage": spec.name,
        "description": spec.description,
        "submitted": False,
        "submission_requested": bool(
            runtime_config.use_tianyan and args.submit
        ),
        "runtime": runtime_config.label,
        "backend": (
            runtime_config.backend
            if runtime_config.use_tianyan
            else "default.qubit"
        ),
        "shots": runtime_config.shots if runtime_config.use_tianyan else None,
        "readout_probability_mode": (
            "sparse_uncalibrated_shot_frequencies"
            if runtime_config.is_hardware
            and spec.wire_count > MAX_DENSE_READOUT_CALIBRATION_QUBITS
            else "cqlib_default"
        ),
        "excluded_physical_qubits": list(
            runtime_config.excluded_physical_qubits
        ),
        "token_qubits": TOKEN_QUBITS,
        "token_count": TOKEN_COUNT,
        "attention_heads": ATTENTION_HEADS,
        "source_logical_wire_count": _source_logical_wire_count(spec.name),
        "wire_count": spec.wire_count,
        "output_wires": list(spec.output_wires),
        "terminal_postselect_wires": list(spec.postselect_wires),
        "terminal_postselect_values": list(spec.postselect_values),
        "parameter_count": spec.parameter_count,
        "top_level_operation_count": len(tape.operations),
        "top_level_operations": operation_counts,
    }
    if breakdown is not None:
        report["parameter_breakdown"] = breakdown

    report_path = _write_report(report, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if runtime_config.use_tianyan and not args.submit:
        print("\nDry run only: add --submit to create the TianYan device and run this stage.")
        print(f"Report written to: {report_path.resolve()}")
        return

    try:
        device = create_device(spec.wire_count, runtime_config)
    except Exception as exc:
        report.update(
            {
                "execution_status": "device_creation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_report(report, args.output_dir)
        raise

    @qml.qnode(
        device,
        diff_method=None,
    )
    def forward(stage_params):
        spec.prepare()
        spec.apply(stage_params)
        return qml.probs(wires=[*spec.postselect_wires, *spec.output_wires])

    report["execution_status"] = "submitted_waiting_for_result"
    _write_report(report, args.output_dir)
    try:
        raw_probabilities = np.asarray(forward(params), dtype=float)
        probabilities, postselection_success = _postselect_output_probabilities(
            raw_probabilities, spec
        )
    except Exception as exc:
        report.update(
            {
                "execution_status": "forward_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_report(report, args.output_dir)
        raise
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("The backend returned NaN or infinite probabilities.")
    probability_sum = float(probabilities.sum())
    if not np.isclose(probability_sum, 1.0, atol=1e-6):
        raise RuntimeError(
            f"Output probabilities sum to {probability_sum}, expected 1."
        )

    report.update(
        {
            "submitted": runtime_config.use_tianyan,
            "execution_status": "complete",
            "postselection_success_probability": postselection_success,
            "probability_sum": probability_sum,
            "top_probabilities": _top_probabilities(
                probabilities,
                output_width=len(spec.output_wires),
            ),
        }
    )
    report_path = _write_report(report, args.output_dir)
    print("\nForward result")
    print(json.dumps(report["top_probabilities"], indent=2, ensure_ascii=False))
    print(f"Probability sum: {probability_sum:.8f}")
    print(f"Report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
