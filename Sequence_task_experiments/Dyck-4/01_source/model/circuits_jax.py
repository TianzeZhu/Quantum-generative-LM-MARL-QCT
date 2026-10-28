"""PennyLane/JAX circuit primitives used by the serialized SCAN model.

This module is the readable PennyLane reference implementation. Formal GPU
training uses the algebraically equivalent pure-JAX kernels in
statevector_jax.py so XLA can fuse and batch independent pair simulations.
Only the successful branch of each internal SVD ancilla is retained. The
SwapTest ancilla is not selected here.
"""

from __future__ import annotations

import math
from functools import lru_cache

import jax.numpy as jnp
import pennylane as qml


def SU4(params, wires):
    """Queue the 15-parameter exact two-qubit ansatz used by the project."""
    wires = list(wires)
    if len(wires) != 2:
        raise ValueError("SU4 requires exactly two wires.")
    if len(params) < 15:
        raise ValueError(f"SU4 needs 15 parameters, given {len(params)}.")
    qml.Rot(*params[:3], wires=wires[0])
    qml.Rot(*params[3:6], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[6], wires=wires[0])
    qml.RY(params[7], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(params[8], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.Rot(*params[9:12], wires=wires[0])
    qml.Rot(*params[12:15], wires=wires[1])


def _su_init_layer(params, wires):
    n = len(wires)
    for parameter, wire in zip(params[:n], wires):
        qml.RZ(parameter, wires=wire)
    for parameter, wire in zip(params[n : 2 * n], wires):
        qml.RY(parameter, wires=wire)
    for parameter, wire in zip(params[2 * n : 3 * n], wires):
        qml.RZ(parameter, wires=wire)


def _su_block(params, wire0, wire1, num_params=4):
    qml.CZ(wires=[wire0, wire1])
    qml.RY(params[0], wires=wire0)
    if num_params > 1:
        qml.RY(params[1], wires=wire1)
    if num_params > 2:
        qml.RZ(params[2], wires=wire0)
    if num_params > 3:
        qml.RZ(params[3], wires=wire1)


def _su_brickwall(params, wires):
    """Queue the project's exact 4**n-1 parameter SU ansatz for n=3 or 4."""
    wires = list(wires)
    n = len(wires)
    if n not in (3, 4):
        raise ValueError("_su_brickwall is defined here for three or four wires.")
    required = 4**n - 1
    if len(params) < required:
        raise ValueError(f"SU(2**{n}) needs {required} parameters.")
    initial = 3 * n
    num_cz = math.ceil((required - initial) / 4)
    relative = [(2 * index) % (n - 1) for index in range(num_cz)]
    if n % 2:
        relative = [
            value + (index // (n // 2)) % 2
            for index, value in enumerate(relative)
        ]
    _su_init_layer(params[:initial], wires)
    parameter_index = initial
    for block_index in range(num_cz):
        start = relative[block_index]
        count = min(4, required - parameter_index)
        _su_block(
            params[parameter_index : parameter_index + count],
            wires[start],
            wires[start + 1],
            count,
        )
        parameter_index += count


def SU8(params, wires):
    wires = list(wires)
    if len(wires) != 3:
        raise ValueError("SU8 requires exactly three wires.")
    _su_brickwall(params, wires)


def SU16(params, wires):
    wires = list(wires)
    if len(wires) != 4:
        raise ValueError("SU16 requires exactly four wires.")
    _su_brickwall(params, wires)


def _small_unitary_part(params, wires):
    wires = list(wires)
    n = len(wires)
    required = 4**n - 1
    if len(params) < required:
        raise ValueError(
            f"A {n}-qubit unitary block needs {required} parameters, "
            f"given {len(params)}."
        )
    if n == 1:
        qml.Rot(params[0], params[1], params[2], wires=wires[0])
    elif n == 2:
        SU4(params[:required], wires)
    elif n == 3:
        SU8(params[:required], wires)
    elif n == 4:
        SU16(params[:required], wires)
    else:
        raise ValueError("_small_unitary_part supports one to four wires.")


def _block_layers(n, n_layers, pattern):
    layers = []
    for layer_index in range(n_layers):
        if pattern == "corner":
            offset = 0 if layer_index % 2 == 0 else 3
            step = 6
        elif pattern == "stride2":
            offset = 0 if layer_index % 2 == 0 else 2
            step = 4
        elif pattern == "disjoint":
            offset, step = 0, 4
        else:
            raise ValueError(f"Unknown block pattern {pattern!r}.")
        layers.append(
            [tuple(range(start, start + 4)) for start in range(offset, n - 3, step)]
        )
    return layers


def _bridge_specs(n, kind):
    """Return the paper-calibrated parameterized bridge layout."""
    if not kind or kind == "none":
        return []
    if kind != "long_range":
        raise ValueError(f"Unknown bridge kind {kind!r}.")
    names = ("CRX", "CRY", "CRZ")
    # These two layouts restore the paper/legacy nominal large-width design.
    # In particular, the n=7 case must touch wire 6 and gives four bridge
    # parameters per inter-layer gap, matching the paper's nominal P=1032
    # for four SU16 layers.
    if n == 6:
        pairs = [(0, 4), (0, 2), (3, 5), (1, 5)]
    elif n == 7:
        pairs = [(0, 4), (1, 5), (2, 6), (3, 6)]
    else:
        pairs = [
            (index, index + n // 2)
            for index in range(n // 2)
        ]
    return [
        (names[index % len(names)], pair)
        for index, pair in enumerate(pairs)
    ]


def _validate_wire_coverage(n, layers, bridges):
    """Reject large-width layouts that silently leave a wire untouched."""
    covered = set()
    for layer_index, layer in enumerate(layers):
        for window in layer:
            covered.update(window)
        if layer_index < len(layers) - 1:
            for _, pair in bridges:
                covered.update(pair)
    missing = sorted(set(range(n)) - covered)
    if missing:
        raise ValueError(
            "unitary_part configuration leaves target wire indices "
            f"{missing} untouched; increase n_layers or choose a covering "
            "block/bridge pattern."
        )


_BRIDGE_GATES = {"CRX": qml.CRX, "CRY": qml.CRY, "CRZ": qml.CRZ}


def unitary_part(
    params,
    wires,
    n_layers=4,
    block_pattern="stride2",
    bridge="long_range",
    count_only=False,
):
    """Queue the project unitary ansatz and return its exact parameter count."""
    wires = list(wires)
    n = len(wires)
    if n <= 0:
        raise ValueError("unitary_part requires at least one target wire.")
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative.")
    if n_layers == 0:
        return 0 if count_only else None

    # Critical for SCAN: an SU16-window construction has no window at n=3.
    if n <= 4:
        per_layer = 4**n - 1
        total = n_layers * per_layer
        if count_only:
            return total
        if len(params) < total:
            raise ValueError(
                f"unitary_part needs {total} parameters, given {len(params)}."
            )
        for layer_index in range(n_layers):
            start = layer_index * per_layer
            _small_unitary_part(params[start : start + per_layer], wires)
        return None

    layers = _block_layers(n, n_layers, block_pattern)
    bridges = _bridge_specs(n, bridge)
    _validate_wire_coverage(n, layers, bridges)
    total = sum(len(layer) for layer in layers) * 255
    total += max(0, n_layers - 1) * len(bridges)
    if count_only:
        return total
    if len(params) < total:
        raise ValueError(f"unitary_part needs {total} parameters, given {len(params)}.")

    parameter_index = 0
    for layer_index, layer in enumerate(layers):
        for window in layer:
            SU16(
                params[parameter_index : parameter_index + 255],
                [wires[index] for index in window],
            )
            parameter_index += 255
        if layer_index < n_layers - 1:
            for gate_name, (control, target) in bridges:
                _BRIDGE_GATES[gate_name](
                    params[parameter_index],
                    wires=[wires[control], wires[target]],
                )
                parameter_index += 1


def postselect_zero(wire, mode="mcm"):
    """Retain outcome zero with either MCM or analytic projector semantics."""
    if mode == "mcm":
        measurement = qml.measure(wires=wire, reset=False, postselect=0)
        qml.Identity(wires=wire)
        return measurement
    if mode == "projector":
        return qml.Projector([0], wires=wire)
    raise ValueError(f"postselection mode must be 'mcm' or 'projector', got {mode!r}.")


def non_unitary_diag_part(
    params,
    target_wires,
    ancilla_wire,
    count_only=False,
    postselection_mode="mcm",
):
    """Queue the paper diagonal block whose selected Kraus map is diag(cos(theta))."""
    target_wires = list(target_wires)
    if not target_wires:
        raise ValueError("non_unitary_diag_part requires target wires.")
    required = 2 ** len(target_wires)
    if count_only:
        return required
    if len(params) < required:
        raise ValueError(f"Diagonal circuit needs {required} parameters.")
    phases = jnp.exp(1j * params[:required])
    qml.Hadamard(wires=ancilla_wire)
    qml.ctrl(
        qml.DiagonalQubitUnitary,
        control=ancilla_wire,
        control_values=True,
    )(jnp.conj(phases), wires=target_wires)
    qml.ctrl(
        qml.DiagonalQubitUnitary,
        control=ancilla_wire,
        control_values=False,
    )(phases, wires=target_wires)
    qml.Hadamard(wires=ancilla_wire)
    postselect_zero(ancilla_wire, mode=postselection_mode)


def non_unitary_diag_mixed(
    params,
    density_matrix,
    target_wires,
    ancilla_wire=None,
    count_only=False,
    return_success_probability=False,
    epsilon=None,
):
    """Apply and normalize the selected diag(cos(theta)) Kraus branch."""
    del ancilla_wire
    target_wires = list(target_wires)
    required = 2 ** len(target_wires)
    if count_only:
        return required
    if len(params) < required:
        raise ValueError(f"Post-selected diagonal block needs {required} parameters.")
    expected = (required, required)
    if tuple(density_matrix.shape) != expected:
        raise ValueError(f"Expected density matrix shape {expected}.")
    weights = jnp.cos(params[:required])
    unnormalized = weights[:, None] * density_matrix * weights[None, :]
    success = jnp.real(jnp.trace(unnormalized))
    if epsilon is None:
        epsilon = jnp.finfo(unnormalized.real.dtype).tiny
    normalized = unnormalized / jnp.maximum(success, epsilon)
    if return_success_probability:
        return normalized, success
    return normalized


def SVD_circuit(
    params,
    target_wires,
    ancilla_wire,
    n_unitary_layer,
    block_pattern="stride2",
    bridge="long_range",
    count_only=False,
    postselection_mode="mcm",
):
    """Queue U diag(cos(theta)) V with one internal work ancilla."""
    target_wires = list(target_wires)
    unitary_count = unitary_part(
        [], range(len(target_wires)), n_unitary_layer,
        block_pattern, bridge, count_only=True
    )
    diagonal_count = 2 ** len(target_wires)
    total = 2 * unitary_count + diagonal_count
    if count_only:
        return total
    if len(params) < total:
        raise ValueError(f"SVD circuit needs {total} parameters, given {len(params)}.")
    diagonal_start = unitary_count
    diagonal_end = diagonal_start + diagonal_count
    unitary_part(
        params[:diagonal_start], target_wires, n_unitary_layer,
        block_pattern, bridge
    )
    non_unitary_diag_part(
        params[diagonal_start:diagonal_end],
        target_wires,
        ancilla_wire,
        postselection_mode=postselection_mode,
    )
    unitary_part(
        params[diagonal_end:total], target_wires, n_unitary_layer,
        block_pattern, bridge
    )


@lru_cache(maxsize=None)
def _unitary_state_qnode(n, n_unitary_layer, block_pattern, bridge):
    device = qml.device("default.qubit", wires=n)

    @qml.qnode(device, interface="jax", diff_method="backprop")
    def qnode(state, params):
        qml.StatePrep(state, wires=range(n), normalize=False)
        unitary_part(params, range(n), n_unitary_layer, block_pattern, bridge)
        return qml.state()

    return qnode


def SVD_circuit_state(
    params,
    input_state,
    target_wires,
    ancilla_wire,
    n_unitary_layer,
    block_pattern="stride2",
    bridge="long_range",
    count_only=False,
    return_success_probability=False,
    epsilon=None,
):
    """Ancilla-free selected state U diag(cos(theta)) V |psi>."""
    del ancilla_wire
    n = len(target_wires)
    unitary_count = unitary_part(
        [], range(n), n_unitary_layer, block_pattern, bridge, count_only=True
    )
    diagonal_count = 2**n
    total = 2 * unitary_count + diagonal_count
    if count_only:
        return total
    if len(params) < total:
        raise ValueError(f"SVD circuit needs {total} parameters, given {len(params)}.")
    if tuple(input_state.shape) != (2**n,):
        raise ValueError(f"input_state must have shape ({2**n},).")
    if epsilon is None:
        epsilon = jnp.finfo(jnp.asarray(input_state).real.dtype).tiny
    qnode = _unitary_state_qnode(n, n_unitary_layer, block_pattern, bridge)
    input_state = input_state / jnp.maximum(jnp.linalg.norm(input_state), epsilon)
    state = qnode(input_state, params[:unitary_count])
    diagonal_end = unitary_count + diagonal_count
    state = jnp.cos(params[unitary_count:diagonal_end]).astype(state.dtype) * state
    success = jnp.real(jnp.vdot(state, state))
    state = state / jnp.maximum(jnp.sqrt(success), epsilon)
    state = qnode(state, params[diagonal_end:total])
    if return_success_probability:
        return state, success
    return state


def SVD_circuit_mixed(
    params,
    density_matrix,
    target_wires,
    ancilla_wire,
    n_unitary_layer,
    block_pattern="stride2",
    bridge="long_range",
    count_only=False,
    return_success_probability=False,
):
    """Density-matrix reference for the same selected SVD branch."""
    n = len(target_wires)
    unitary_count = unitary_part(
        [], range(n), n_unitary_layer, block_pattern, bridge, count_only=True
    )
    diagonal_count = 2**n
    total = 2 * unitary_count + diagonal_count
    if count_only:
        return total
    if len(params) < total:
        raise ValueError(f"SVD circuit needs {total} parameters.")
    matrix_function = qml.matrix(unitary_part, wire_order=range(n))
    first = matrix_function(
        params[:unitary_count], range(n), n_unitary_layer,
        block_pattern, bridge
    )
    density_matrix = first @ density_matrix @ jnp.conj(first.T)
    diagonal_end = unitary_count + diagonal_count
    density_matrix, success = non_unitary_diag_mixed(
        params[unitary_count:diagonal_end],
        density_matrix,
        range(n),
        ancilla_wire,
        return_success_probability=True,
    )
    second = matrix_function(
        params[diagonal_end:total], range(n), n_unitary_layer,
        block_pattern, bridge
    )
    density_matrix = second @ density_matrix @ jnp.conj(second.T)
    if return_success_probability:
        return density_matrix, success
    return density_matrix


__all__ = [
    "SU4",
    "SU8",
    "SU16",
    "unitary_part",
    "postselect_zero",
    "non_unitary_diag_part",
    "non_unitary_diag_mixed",
    "SVD_circuit",
    "SVD_circuit_state",
    "SVD_circuit_mixed",
]




