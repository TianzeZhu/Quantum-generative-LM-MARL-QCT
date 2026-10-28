import pennylane as qml
from circuits import unitary_part

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

def SU8(params, wires):# Old version.
    if len(params)<63:
        raise ValueError('The number of parameters should be 63 to construct a SU(8).')
    qml.Rot(params[0], params[1], params[2], wires[0])
    qml.Rot(params[3], params[4], params[5], wires[1])
    qml.Rot(params[6], params[7], params[8], wires[2])
    qml.Toffoli(wires)
    SU4(params[9:24], (wires[1], wires[2]))
    qml.Toffoli([wires[1], wires[2], wires[0]])
    SU4(params[24:39], (wires[2], wires[0]))
    qml.Toffoli([wires[2], wires[0], wires[1]])
    SU4(params[39:54], (wires[0], wires[1]))
    qml.Toffoli(wires)
    qml.Rot(params[54], params[55], params[56], wires[0])
    qml.Rot(params[57], params[58], params[59], wires[1])
    qml.Rot(params[60], params[61], params[62], wires[2]) 

def SU16(params, wires):
    if len(params)<255:
        raise ValueError('The number of parameters should be 255 to construct a SU(16).')
    SU8(params[:63], wires[:3])
    qml.ctrl(qml.Toffoli(wires[1:]), wires[0], True)
    SU8(params[63:126], (wires[0], wires[1], wires[3]))
    qml.ctrl(qml.Toffoli([wires[1], wires[3], wires[2]]), wires[0], True)
    SU8(params[126:189], (wires[0], wires[2], wires[3]))
    qml.ctrl(qml.Toffoli([wires[2], wires[3], wires[1]]), wires[0], True)
    SU8(params[189:252], wires[1:])
    qml.ctrl(qml.Toffoli([wires[2], wires[3], wires[0]]), wires[1], True)
    qml.Rot(params[252], params[253], params[254], wires[3])

def su16_block(params, wires, repeat_time:int=1, count_only=False):
    if len(wires) == 0:
        raise ValueError("wires must be non-empty.")
    if len(wires)<4:
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
