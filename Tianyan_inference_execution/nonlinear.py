import pennylane as qml
from pennylane import numpy as np
from circuits import unitary_part


def _apply_optional_control(op, params, wires, extra_control_wires=None):
    """Apply an operation directly or under all-ones controls."""
    if extra_control_wires:
        qml.ctrl(
            op,
            control=extra_control_wires,
            control_values=[True] * len(extra_control_wires),
        )(params, wires)
        return
    op(params, wires)


def _apply_optional_control_rot(params, wire, extra_control_wires=None):
    """Single-qubit version of _apply_optional_control."""
    if extra_control_wires:
        qml.ctrl(
            qml.Rot,
            control=extra_control_wires,
            control_values=[True] * len(extra_control_wires),
        )(params[0], params[1], params[2], wires=wire)
        return
    qml.Rot(params[0], params[1], params[2], wires=wire)

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

def SU8(params, wires):  # New version.
    if len(params)<63:
        raise ValueError('The number of parameters should be 63 to construct a SU(8).')
    if len(wires)<3:
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
    if len(params)<255:
        raise ValueError('The number of parameters should be 255 to construct a SU(16).')
    if len(wires)<4:
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


def su16_block(params, wires, repeat_time: int = 1, count_only=False, extra_control_wires=None):
    """Compatibility wrapper backed by ``circuits.unitary_part``.

    Registers below four qubits keep their SU(2/4/8) fallback.  For wider
    registers, ``extra_control_wires`` controls the complete unitary ansatz.
    """
    if len(wires) == 0:
        raise ValueError("wires must be non-empty.")
    if len(wires) == 1:
        if count_only:
            return 3
        if len(params) < 3:
            raise ValueError('The number of parameters should be 3 to construct a SU(2).')
        _apply_optional_control_rot(params[:3], wires[0], extra_control_wires)
        return
    if len(wires) == 2:
        if count_only:
            return 15
        if len(params) < 15:
            raise ValueError('The number of parameters should be 15 to construct a SU(4).')
        _apply_optional_control(SU4, params[:15], wires, extra_control_wires)
        return
    if len(wires) == 3:
        if count_only:
            return 63
        if len(params) < 63:
            raise ValueError('The number of parameters should be 63 to construct a SU(8).')
        _apply_optional_control(SU8, params[:63], wires, extra_control_wires)
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
    unitary_args = (
        params[:least_parameter_number],
        wires,
        repeat_time,
        "stride2",
        "long_range",
    )
    if extra_control_wires:
        qml.ctrl(
            unitary_part,
            control=extra_control_wires,
            control_values=[True] * len(extra_control_wires),
        )(*unitary_args)
        return
    unitary_part(*unitary_args)


def nonlinear_layer(params, wires, num1_layers, num2_layers, ancilla_wire, count_only=False):
    # Input wires do not include the ancilla qubit.
    block_param_count = su16_block([], wires, count_only=True)
    total_layer_count = num1_layers + num2_layers
    if count_only:
        total_params_needed = total_layer_count * block_param_count
        return total_params_needed
    if total_layer_count == 0:
        return

    qml.Hadamard(wires=ancilla_wire)
    # First stage: repeat num1_layers times.
    for layer in range(num1_layers):
        start_idx = layer * block_param_count
        end_idx = start_idx + block_param_count
        
        # On the last layer, add the ancilla qubit as an extra control for the final multi-controlled gate.
        if layer == num1_layers - 1:
            su16_block(params[start_idx:end_idx], wires, extra_control_wires=[ancilla_wire])
        else:
            su16_block(params[start_idx:end_idx], wires)
    
    # Second stage: repeat num2_layers times.
    # Swap the roles of the last qubit and the ancilla qubit.
    new_wires = wires[:-1] + [ancilla_wire]
    # Switch the extra control qubit to the original last qubit.
    new_extra_control = [wires[-1]]
    
    for layer in range(num2_layers):
        start_idx = (num1_layers + layer) * block_param_count
        end_idx = start_idx + block_param_count
        
        # On the last layer, add the new extra control qubit.
        if layer == num2_layers - 1:
            su16_block(params[start_idx:end_idx], new_wires, extra_control_wires=new_extra_control)
        else:
            su16_block(params[start_idx:end_idx], new_wires)
    qml.measure(wires=ancilla_wire, postselect=1)
    # qml.Projector([0, 1], wires=ancilla_wire)

    # Use postselection measurement to enable ancilla reuse.
    # The ancilla only needs to be postselected into the target state at the end.
