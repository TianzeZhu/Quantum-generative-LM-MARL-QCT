import math
import pennylane as qml
from pennylane import numpy as np

def SU4(params, wires):
    if params.shape[0]<15:
        raise ValueError(f'2-bit general circuit needs 15 parameters, given {params.shape[0]}.')
    qml.Rot(*params[:3], wires[0])
    qml.Rot(*params[3:6], wires[1])
    qml.CNOT([wires[1], wires[0]])
    qml.RZ(params[6], wires[0])
    qml.RY(params[7], wires[1])
    qml.CNOT(wires)
    qml.RY(params[8], wires[1])
    qml.CNOT([wires[1], wires[0]])
    qml.Rot(*params[9:12], wires[0])
    qml.Rot(*params[12:15], wires[1])

def _su_init_layer(params, wires):
    """Initial single-qubit layer of the SU(2^n) brick wall ansatz (3n parameters).

    Applies RZ, then RY, then RZ on every qubit. The parameter index runs across the
    circuit width (qubits) first, then across depth, matching the ZYZ Euler box of the
    initial layer in the paper.
    """
    n = len(wires)
    for p, w in zip(params[:n], wires):
        qml.RZ(p, w)
    for p, w in zip(params[n:2 * n], wires):
        qml.RY(p, w)
    for p, w in zip(params[2 * n:3 * n], wires):
        qml.RZ(p, w)

def _su_block(params, w0, w1, num_params=4):
    """One brick of the SU(2^n) brick wall: a CZ followed by RY/RZ rotations.

    A full brick uses 4 parameters (RY, RY, RZ, RZ in width-then-depth order). The final,
    possibly incomplete, brick truncates the trailing rotations to use ``num_params``.
    """
    qml.CZ([w0, w1])
    qml.RY(params[0], w0)
    if num_params > 1:
        qml.RY(params[1], w1)
    if num_params > 2:
        qml.RZ(params[2], w0)
    if num_params > 3:
        qml.RZ(params[3], w1)

def _su_brickwall(params, wires):
    """Universal SU(2^n) brick wall circuit with optimal parameter and CZ count.

    Based on "Unitary synthesis with optimal brick wall circuits" (Wierichs, Kottmann,
    Killoran; arXiv:2511.16736), following the SU branch of the reference implementation
    ``XanaduAI/unicirc``. The circuit has d = 4^n - 1 parameters: an initial 3n-parameter
    single-qubit layer followed by ceil((d - 3n) / 4) bricks arranged in a brick wall.
    """
    n = len(wires)
    d = 4 ** n - 1
    p_init = 3 * n
    num_cz = math.ceil((d - p_init) / 4)
    # Relative starting wire index of each brick (densely packed brick wall layout).
    rel = [(2 * i) % (n - 1) for i in range(num_cz)]
    if n % 2:
        rel = [r + (i // (n // 2)) % 2 for i, r in enumerate(rel)]
    _su_init_layer(params[:p_init], wires)
    idx = p_init
    for i in range(num_cz):
        a, b = wires[rel[i]], wires[rel[i] + 1]
        npp = min(4, d - idx)
        _su_block(params[idx:idx + npp], a, b, npp)
        idx += npp

def SU8(params, wires):
    if params.shape[0]<63:
        raise ValueError(f'3-bit general circuit needs 63 parameters, given {params.shape[0]}.')
    _su_brickwall(params, wires)

def SU16(params, wires):
    if params.shape[0]<255:
        raise ValueError(f'4-bit general circuit needs 255 parameters, given {params.shape[0]}.')
    _su_brickwall(params, wires)


# --------------------------------------------------------------------------- #
# Large O(n)-depth circuits built from full SU16 blocks (linear nearest-       #
# neighbour) separated by all-to-all *parameterized* bridge gates.            #
#                                                                              #
# Design rationale (all numerically validated with effective_params.py):       #
#  * Two universal blocks sharing k qubits waste 4**k - 1 parameters.          #
#  * A fixed/absorbable separator (support inside one block) cannot recover     #
#    that waste; a separator helps only if its support lies inside *neither*    #
#    adjacent block -- the SU4 CNOT(0,1) principle.                            #
#  * A *parameterized*, non-absorbable bridge (e.g. CRY on an all-to-all pair   #
#    reaching outside both blocks) both breaks the redundancy AND contributes   #
#    an effective trainable parameter.                                         #
# So we interleave dense SU16 block layers with long-range parameterized        #
# controlled-rotation bridge layers.                                           #
# --------------------------------------------------------------------------- #

_BRIDGE_GATES = {"CRX": qml.CRX, "CRY": qml.CRY, "CRZ": qml.CRZ}


def _block_layers(n, n_layers, pattern):
    """Return ``n_layers`` block layers; each layer is a list of disjoint 4-qubit
    windows (tuples of wire indices) on which a full SU16 acts.

    * ``disjoint``: every layer uses windows [0-3],[4-7],...  (no inter-layer shift).
    * ``stride2`` : layers alternate offset 0 and 2 -> a dense brick wall whose
      consecutive blocks overlap by 2 qubits.
    * ``corner``  : 2-colouring of a stride-3 chain (consecutive blocks overlap by 1).
    """
    layers = []
    for t in range(n_layers):
        if pattern == "corner":
            off = 0 if t % 2 == 0 else 3
            step = 6
        elif pattern == "stride2":
            off = 0 if t % 2 == 0 else 2
            step = 4
        elif pattern == "disjoint":
            off, step = 0, 4
        else:
            raise ValueError(f"unknown block pattern {pattern!r}")
        layers.append([tuple(range(s, s + 4)) for s in range(off, n - 3, step)])
    return layers


def _bridge_specs(n, kind):
    """Return a list of ``(gate_name, (a, b))`` parameterized-bridge specs (indices
    into the wire list). ``long_range`` pairs qubit ``i`` with ``i + n//2`` so every
    qubit takes part in exactly one depth-1, all-to-all, non-absorbable bridge."""
    if not kind or kind == "none":
        return []
    if kind == "long_range":
        names = ("CRX", "CRY", "CRZ")
        return [(names[i % 3], (i, i + n // 2)) for i in range(n // 2)]
    raise ValueError(f"unknown bridge kind {kind!r}")


def unitary_part(params, wires, n_layers=4, block_pattern="stride2",
               bridge="long_range", count_only=False):
    """Assemble an O(n_layers)-depth circuit from full SU16 blocks and parameterized
    bridge layers.

    Structure (repeated implicitly through ``n_layers``):
        [SU16 block layer]  ([bridge layer]  [SU16 block layer]) x (n_layers-1)

    Args:
        params: flat parameter vector (length = ``large_su16(..., count_only=True)``).
        wires: list of qubit labels (length n). SU16 blocks act on consecutive wires.
        n_layers: number of SU16 block layers (depth is O(n_layers)).
        block_pattern: ``"stride2"`` | ``"disjoint"`` | ``"corner"``.
        bridge: ``"long_range"`` | ``"none"`` -- parameterized separator layer placed
            between consecutive block layers.
        count_only: if True, return the required parameter count without queuing gates.
    """
    n = len(wires)
    block_layers = _block_layers(n, n_layers, block_pattern)
    specs = _bridge_specs(n, bridge)

    if count_only:
        total = sum(len(layer) for layer in block_layers) * 255
        total += (n_layers - 1) * len(specs)
        return total

    idx = 0
    for t, layer in enumerate(block_layers):
        for win in layer:
            SU16(params[idx:idx + 255], [wires[w] for w in win])
            idx += 255
        if t < n_layers - 1:
            for gate_name, (a, b) in specs:
                _BRIDGE_GATES[gate_name](params[idx], wires=[wires[a], wires[b]])
                idx += 1

def non_unitary_diag_part(params, wires, ancilla):
    """Assemble a non-unitary diagonal circuit on ``wires`` with an ancilla qubit. The
    circuit applies diag(cos(params[0]), ..., cos(params[-1])) on ``wires``.
    
    Args:
        params: flat parameter vector (length = 2**len(wires)).
        wires: list of qubit labels (length n).
        ancilla: label of the ancilla qubit.
    """

    if params.shape[0] < 2**len(wires):
        raise ValueError(f"Diagonal non-unitary circuit needs {2**len(wires)} parameters, given {params.shape[0]}.")
    qml.H(ancilla)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, True)(np.exp(-1j * params), wires)
    qml.ctrl(qml.DiagonalQubitUnitary, ancilla, False)(np.exp(1j * params), wires)
    qml.H(ancilla)
    qml.RZ(-np.pi/2, ancilla)
    qml.measure(ancilla, reset=True, postselect=0)

# print(unitary_part(np.array([]), range(10), 8, count_only=True))  # Example usage to get parameter count
