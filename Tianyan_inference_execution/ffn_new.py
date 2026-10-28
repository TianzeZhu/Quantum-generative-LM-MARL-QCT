from attention import su16_block
import pennylane as qml
from pennylane import numpy as np


def diag_op_non(params, wires, ancilla):
    """Diagonal operator for the non-unitary circuit.

    Args:
        params (list): Thetas for the diagonal operator. The diagonal elements of the operator are cos(theta)s.
        wires (list): Wires on which to apply the operator.
        ancilla (int): Ancilla qubit wire.
    """
    if len(params)!=2**len(wires):
        raise ValueError("The number of parameters must match the number of wires.")
    qml.Hadamard(wires=ancilla)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, True)(qml.math.exp(params*(-1j)), wires)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, False)(qml.math.exp(params*1j), wires)
    qml.Hadamard(wires=ancilla)
    qml.RZ(-np.pi / 2, wires=ancilla)
    qml.measure(wires=ancilla, reset=True, postselect=0)
    # qml.Projector([1, 0], wires=ancilla)

def ffn_new(params, total_qubits, num_blocks, repeat_time=1, wires=None, ancilla=None, count_only=False):

    #control_bits = math.log2(num_blocks)
    #control_bits = int(control_bits)
    #target_qubits = total_qubits - control_bits


    #control_wires = wires[:control_bits]
    #target_wires = wires[control_bits:]
    ancilla1=ancilla[0]
    su_param_count = su16_block([], wires, repeat_time=repeat_time, count_only=True)
    diag_param_count= 2 ** len(wires)
    params_per_block = 2 * su_param_count + diag_param_count
    if count_only:
        return params_per_block
    su1_end = su_param_count
    diag_start = su1_end
    diag_end = diag_start + diag_param_count
    su2_start = diag_end
    su2_end = su2_start + su_param_count
    su1_params = params[:su1_end]
    diag_params = params[diag_start:diag_end]
    su2_params = params[su2_start:su2_end]
    su16_block(su1_params, wires, repeat_time)
    diag_op_non(diag_params,wires,ancilla1)
    #diag_op(pos=0,wires=target_wires,ancilla=ancilla2)
    su16_block(su2_params,wires,repeat_time)
    return wires



ffn = ffn_new
