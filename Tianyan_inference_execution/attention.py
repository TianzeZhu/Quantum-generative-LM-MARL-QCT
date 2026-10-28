import math
import pennylane as qml
from pennylane import numpy as np
from diag_op import diag_op
from circuits import unitary_part
#import torch


def SU4(params, wires):
    """Implement an arbitrary SU(4) gate on two qubits using the decomposition from Theorem 5 in
    https://arxiv.org/pdf/quant-ph/0308006.pdf"""
    i, j = wires
    qml.Rot(*params[:3], wires=i)
    qml.Rot(*params[3:6], wires=j)
    qml.CNOT(wires=[j, i])
    qml.RZ(params[6], wires=i)
    qml.RY(params[7], wires=j)
    qml.CNOT(wires=[i, j])
    qml.RY(params[8], wires=j)
    qml.CNOT(wires=[j, i])
    qml.Rot(*params[9:12], wires=i)
    qml.Rot(*params[12:15], wires=j)


def SU8(params, wires):
    if len(params) < 63:
        raise ValueError('The number of parameters should be 63 to construct a SU(8).')
    if len(wires) < 3:
        raise ValueError('The SU(8) circuit should be operated on 3 qubits.')
    SU4(params[0:15], [wires[1], wires[2]])
    qml.Rot(params[15], params[16], params[17], wires[0])
    qml.Toffoli([wires[1], wires[2], wires[0]])
    SU4(params[18:33], [wires[2], wires[0]])
    qml.Rot(params[33], params[34], params[35], wires[1])
    qml.Toffoli([wires[2], wires[0], wires[1]])
    SU4(params[36:51], [wires[0], wires[1]])
    qml.Rot(params[51], params[52], params[53], wires[2])
    qml.Toffoli(wires)
    qml.Rot(params[54], params[55], params[56], wires[0])
    qml.Rot(params[57], params[58], params[59], wires[1])
    qml.Rot(params[60], params[61], params[62], wires[2])
    qml.Toffoli([wires[1], wires[2], wires[0]])


def SU16(params, wires):
    if len(params) < 255:
        raise ValueError('The number of parameters should be 255 to construct a SU(16).')
    if len(wires) < 4:
        raise ValueError('The SU(16) circuit should be operated on 4 qubits.')
    SU8(params[:63], wires[:3])
    qml.ctrl(qml.Toffoli(wires[1:]), wires[0], True)
    SU8(params[63:126], [wires[0], wires[1], wires[3]])
    qml.ctrl(qml.Toffoli([wires[1], wires[3], wires[2]]), wires[0], True)
    SU8(params[126:189], [wires[0], wires[2], wires[3]])
    qml.ctrl(qml.Toffoli([wires[2], wires[3], wires[1]]), wires[0], True)
    SU8(params[189:252], wires[1:])
    qml.ctrl(qml.Toffoli([wires[2], wires[3], wires[0]]), wires[1], True)
    qml.Rot(params[252], params[253], params[254], wires[3])


def su16_block(params, wires, repeat_time: int = 1, count_only=False):
    """Compatibility wrapper backed by ``circuits.unitary_part``.

    Registers below four qubits keep their universal SU(2/4/8) fallback because
    ``unitary_part`` is defined in terms of four-qubit windows.
    """
    if len(wires) == 0:
        raise ValueError("wires must be non-empty.")
    if len(wires) < 4:
        if len(wires) == 3:
            required_params = 63
            gate = SU8
        elif len(wires) == 2:
            required_params = 15
            gate = SU4
        else:  # len(wires) == 1
            required_params = 3
            gate = None

        if count_only:
            return required_params
        if len(params) < required_params:
            raise ValueError(
                f"The number of parameters is not enough. Need at least "
                f"{required_params}, given {len(params)}."
            )
        if gate is None:
            qml.Rot(params[0], params[1], params[2], wires=wires[0])
        else:
            gate(params, wires)
        return
    if repeat_time <= 0:
        raise ValueError("repeat_time must be positive.")

    least_parameter_number = unitary_part(
        [],
        range(len(wires)),
        n_layers=repeat_time,
        block_pattern="stride2",
        bridge="long_range",
        count_only=True,
    )
    if count_only:
        return least_parameter_number
    if len(params) < least_parameter_number:
        raise ValueError(f'The number of parameters is not enough for the circuit. Need at least {least_parameter_number} parameters, given {len(params)}.')
    unitary_part(
        params[:least_parameter_number],
        wires,
        repeat_time,
        "stride2",
        "long_range",
    )


def _controlled_su16_array_legacy(params, total_qubits, num_blocks, repeat_time=1, wires=None, ancilla=None, count_only=False):
    """
    Legacy controlled SU16 array.

    Each module has the following structure:
    Controlled[SU16_1] -> diag_op (with postselection measurement) -> Controlled[SU16_2]

    params: All parameters arranged in module order.
    total_qubits: Total number of control and target qubits, excluding ancilla.
    num_blocks: Number of logical modules; must be a power of 2.
    repeat_time: Repeat count passed to su16_block.
    wires: Must have length total_qubits, with control qubits first.
    ancilla: Two ancilla qubits [work_ancilla, diag_ancilla], distinct from the main wires.
    count_only: When True, return only the parameter count.
    Returns: (control_wires, target_wires).
    """

    if ancilla is None:
        raise ValueError("controlled_su16_array requires two ancilla wires: [work_ancilla, diag_ancilla].")

    control_bits = math.log2(num_blocks)
    if not control_bits.is_integer():
        raise ValueError(f"num_blocks must be a power of 2, got {num_blocks}.")
    control_bits = int(control_bits)

    target_qubits = total_qubits - control_bits
    if target_qubits <= 0:
        raise ValueError(
            f"total_qubits={total_qubits} must be greater than control_bits={control_bits}."
        )

    if wires is None:
        wires = list(range(total_qubits))
    else:
        if len(wires) != total_qubits:
            raise ValueError(
                f"wires length must equal total_qubits={total_qubits}, got {len(wires)}."
            )

    if not isinstance(ancilla, (list, tuple)):
        raise ValueError("ancilla must be a sequence of two wires: [work_ancilla, diag_ancilla].")
    if len(ancilla) < 2:
        raise ValueError("controlled_su16_array requires two ancilla wires: [work_ancilla, diag_ancilla].")
    ancilla_wire = ancilla[0]
    diag_ancilla_wire = ancilla[1]

    # Ensure ancilla wires do not overlap with the main circuit wires.
    if ancilla_wire in wires:
        raise ValueError("work ancilla wire index must not be among the main circuit wires (control or target).")
    if diag_ancilla_wire in wires:
        raise ValueError("diag ancilla wire index must not be among the main circuit wires (control or target).")
    if ancilla_wire == diag_ancilla_wire:
        raise ValueError("work ancilla and diag ancilla must be different wires.")

    control_wires = wires[:control_bits]
    target_wires = wires[control_bits:]
    # print(f"control wires: {control_wires}, target wires: {target_wires}, ancilla: {ancilla}")

    # Compute the parameter count per module: su1 + diag + su2.
    # su_param_count is the number of parameters needed by one su16_block side.
    su_param_count = su16_block([], target_wires, repeat_time=repeat_time, count_only=True)
    diag_param_count = 2 ** len(target_wires)
    params_per_block = 2 * su_param_count + diag_param_count

    if count_only:
        return params_per_block * num_blocks

    total_params_needed = num_blocks * params_per_block
    if len(params) < total_params_needed:
        raise ValueError(
            f"Parameter count is insufficient. Need {total_params_needed}, got {len(params)}."
        )
    # print(f"params per module: {params_per_block}, total params: {total_params_needed}")

    # Iterate through each module: controlled SU1 -> diag_op -> controlled SU2.
    for block_idx in range(num_blocks):
        control_values = [bool((block_idx >> i) & 1) for i in reversed(range(control_bits))]

        start_idx = block_idx * params_per_block
        # Split parameters for the current block.
        su1_start = start_idx
        su1_end = su1_start + su_param_count
        diag_start = su1_end
        diag_end = diag_start + diag_param_count
        su2_start = diag_end
        su2_end = su2_start + su_param_count

        su1_params = params[su1_start:su1_end]
        diag_params = params[diag_start:diag_end]
        su2_params = params[su2_start:su2_end]

        # if torch.isnan(torch.tensor(diag_params)).any():
        #     print(f"warning: diag_params for block {block_idx} contains NaN")
        #     print(f"diag_params: {diag_params}")

        # 1) Controlled SU1, using only the first half of the SU stack.
        # Use qml.ctrl to add controls to su16_block.
        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values
        )(su1_params, target_wires, repeat_time)


        qml.Hadamard(wires=ancilla_wire)
        # print("round:", block_idx, ancilla_wire)
        qml.ctrl(qml.DiagonalQubitUnitary,
                control=[ancilla_wire, *control_wires],          
                control_values=[True, *control_values]      
                )(qml.math.exp(diag_params * (-1j)), target_wires)
        #print("look here:",diag_params * (-1j))
        qml.ctrl(qml.DiagonalQubitUnitary,
                control=[ancilla_wire, *control_wires],
                control_values=[False, *control_values]     
                )(qml.math.exp(diag_params * 1j), target_wires)
        qml.Hadamard(wires=ancilla_wire)
        qml.RZ(-np.pi / 2, wires=ancilla_wire)
        qml.measure(wires=ancilla_wire, reset=False, postselect=0)
        # qml.Projector([1, 0], wires=ancilla_wire)
        diag_op(pos=block_idx,wires=control_wires,ancilla=diag_ancilla_wire)

        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values
        )(su2_params, target_wires, repeat_time)



    return control_wires, target_wires


def controlled_su16_array(params, total_qubits, num_blocks, repeat_time=1, wires=None, ancilla=None, count_only=False):
    # Support the single-qubit total_qubits case.

    if ancilla is None:
        raise ValueError("controlled_su16_array requires two ancilla wires: [work_ancilla, diag_ancilla].")
    if total_qubits <= 0:
        raise ValueError(f"total_qubits must be positive, got {total_qubits}.")
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

    if wires is None:
        wires = list(range(total_qubits))
    elif len(wires) != total_qubits:
        raise ValueError(f"wires length must equal total_qubits={total_qubits}, got {len(wires)}.")

    if not isinstance(ancilla, (list, tuple)):
        raise ValueError("ancilla must be a sequence of two wires: [work_ancilla, diag_ancilla].")
    if len(ancilla) < 2:
        raise ValueError("controlled_su16_array requires two ancilla wires: [work_ancilla, diag_ancilla].")
    ancilla_wire = ancilla[0]
    diag_ancilla_wire = ancilla[1]

    if ancilla_wire in wires:
        raise ValueError("work ancilla wire index must not be among the main circuit wires (control or target).")
    if diag_ancilla_wire in wires:
        raise ValueError("diag ancilla wire index must not be among the main circuit wires (control or target).")
    if ancilla_wire == diag_ancilla_wire:
        raise ValueError("work ancilla and diag ancilla must be different wires.")

    if total_qubits == 1:
        diag_param_count = 2 ** total_qubits
        if count_only:
            return diag_param_count
        if len(params) < diag_param_count:
            raise ValueError(f"Parameter count is insufficient. Need {diag_param_count}, got {len(params)}.")

        from ffn_new import diag_op_non

        diag_op_non(np.asarray(params[:diag_param_count]), wires, ancilla_wire)
        return [], list(wires)

    control_bits = max(1, int(math.ceil(math.log2(num_blocks))))
    target_qubits = total_qubits - control_bits
    if target_qubits <= 0:
        raise ValueError(
            f"total_qubits={total_qubits} must exceed the implied control bit count {control_bits} for num_blocks={num_blocks}."
        )

    control_wires = wires[:control_bits]
    target_wires = wires[control_bits:]
    su_param_count = su16_block([], target_wires, repeat_time=repeat_time, count_only=True)
    diag_param_count = 2 ** len(target_wires)
    params_per_block = 2 * su_param_count + diag_param_count

    if count_only:
        return params_per_block * num_blocks

    total_params_needed = num_blocks * params_per_block
    if len(params) < total_params_needed:
        raise ValueError(f"Parameter count is insufficient. Need {total_params_needed}, got {len(params)}.")

    for block_idx in range(num_blocks):
        control_values = [bool((block_idx >> i) & 1) for i in reversed(range(control_bits))]

        start_idx = block_idx * params_per_block
        su1_end = start_idx + su_param_count
        diag_end = su1_end + diag_param_count
        su2_end = diag_end + su_param_count

        su1_params = params[start_idx:su1_end]
        diag_params = params[su1_end:diag_end]
        su2_params = params[diag_end:su2_end]

        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(su1_params, target_wires, repeat_time)

        qml.Hadamard(wires=ancilla_wire)
        qml.ctrl(
            qml.DiagonalQubitUnitary,
            control=[ancilla_wire, *control_wires],
            control_values=[True, *control_values],
        )(qml.math.exp(diag_params * (-1j)), target_wires)
        qml.ctrl(
            qml.DiagonalQubitUnitary,
            control=[ancilla_wire, *control_wires],
            control_values=[False, *control_values],
        )(qml.math.exp(diag_params * 1j), target_wires)
        qml.ctrl(qml.Hadamard, control=control_wires, control_values=control_values)(wires=ancilla_wire)
        qml.RZ(-np.pi / 2, wires=ancilla_wire)
        qml.measure(wires=ancilla_wire, reset=False, postselect=0)
        diag_op(pos=block_idx, wires=control_wires, ancilla=diag_ancilla_wire)
        # qml.Projector([0], wires=ancilla_wire)
        qml.ctrl(
            su16_block,
            control=control_wires,
            control_values=control_values,
        )(su2_params, target_wires, repeat_time)

    return control_wires, target_wires
