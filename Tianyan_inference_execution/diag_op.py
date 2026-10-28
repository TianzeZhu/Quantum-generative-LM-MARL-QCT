import pennylane as qml
from pennylane import numpy as np

def diag_op(pos, wires, ancilla):
    n = len(wires)
    dim = 2 ** n
    
    if not (0 <= pos < dim):
        raise ValueError(f"pos must be in [0, {dim - 1}], got {pos}.")
    
    params = np.zeros(dim)
    params[pos] = np.pi / 4
    
    qml.Hadamard(wires=ancilla)
    qml.ctrl(qml.DiagonalQubitUnitary, control=ancilla, control_values=True)(
        np.exp(-1j * params), wires=wires
    )
    qml.ctrl(qml.DiagonalQubitUnitary, control=ancilla, control_values=False)(
        np.exp(1j * params), wires=wires
    )
    qml.Hadamard(wires=ancilla)
    qml.RZ(-np.pi / 2, wires=ancilla)
    qml.measure(wires=ancilla, reset=True, postselect=0)
    # qml.Projector([1, 0], wires=ancilla)
