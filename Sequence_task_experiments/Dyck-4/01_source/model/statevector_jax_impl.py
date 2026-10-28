"""Pure-JAX state-vector kernels for the serialized quantum Transformer.

The functions in this module reproduce the n=3 PennyLane reference topology,
but expose it as ordinary JAX array operations.  This lets XLA compile one
full-sequence loss and vectorize independent context pairs on a GPU.

Two analytic reductions are used only for simulation:

* every internal SVD work ancilla conditioned on zero is exactly the Kraus map
  diag(cos(theta)), followed by normalization;
* after a completed SwapTest the reduced swap ancilla is diagonal with weights
  (1+F)/2 and (1-F)/2.  K and Q never interact again, so the remaining
  seven-wire entangler can be evaluated as this exact two-branch mixture.

The logical paper circuit still uses 4*n+3 = 15 wires for n=3.  The largest
active state in this optimized simulator is seven wires (128 amplitudes).
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import jax

# Postselection normalization is sensitive in float32.  Keep x64 as the
# reproducibility default; a server run may opt out before import.
jax.config.update(
    "jax_enable_x64",
    os.environ.get("QT_JAX_ENABLE_X64", "1").lower() not in {"0", "false", "no"},
)

import jax.numpy as jnp
import numpy as np


TOKEN_QUBITS = 3
TOKEN_DIM = 2**TOKEN_QUBITS
PAIR_ACTIVE_QUBITS = 2 * TOKEN_QUBITS + 1
LOGICAL_PAIR_QUBITS = 4 * TOKEN_QUBITS + 3
UNITARY_PARAMETERS_PER_LAYER = 4**TOKEN_QUBITS - 1
SVD_PARAMETERS_PER_LAYER = 2 * UNITARY_PARAMETERS_PER_LAYER + TOKEN_DIM
NONLINEAR_PARAMETERS_PER_LAYER_PAIR = 2 * UNITARY_PARAMETERS_PER_LAYER

_CNOT = np.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
_CZ = np.diag(np.asarray([1, 1, 1, -1], dtype=np.complex128))
_H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)


def _complex_dtype(value):
    return jnp.complex128 if jnp.asarray(value).dtype == jnp.float64 else jnp.complex64


def _rx(theta):
    dtype = _complex_dtype(theta)
    cosine = jnp.cos(theta / 2).astype(dtype)
    sine = jnp.sin(theta / 2).astype(dtype)
    return jnp.stack(
        (
            jnp.stack((cosine, -1j * sine)),
            jnp.stack((-1j * sine, cosine)),
        )
    )


def _ry(theta):
    dtype = _complex_dtype(theta)
    cosine = jnp.cos(theta / 2).astype(dtype)
    sine = jnp.sin(theta / 2).astype(dtype)
    return jnp.stack(
        (
            jnp.stack((cosine, -sine)),
            jnp.stack((sine, cosine)),
        )
    )


def _rz(theta):
    diagonal = jnp.stack(
        (jnp.exp(-0.5j * theta), jnp.exp(0.5j * theta))
    )
    return jnp.diag(diagonal)


def _rot(phi, theta, omega):
    """PennyLane Rot(phi, theta, omega) matrix."""
    return _rz(omega) @ _ry(theta) @ _rz(phi)


def _mcx(num_controls, dtype):
    dimension = 2 ** (num_controls + 1)
    matrix = jnp.eye(dimension, dtype=dtype)
    index0 = dimension - 2
    index1 = dimension - 1
    matrix = matrix.at[index0, index0].set(0)
    matrix = matrix.at[index1, index1].set(0)
    matrix = matrix.at[index0, index1].set(1)
    matrix = matrix.at[index1, index0].set(1)
    return matrix


def apply_gate(state, gate, wires, num_wires):
    """Apply a dense k-wire gate to arbitrary ordered axes of a pure state."""
    wires = tuple(int(wire) for wire in wires)
    if len(set(wires)) != len(wires):
        raise ValueError("Gate wires must be distinct.")
    if any(wire < 0 or wire >= num_wires for wire in wires):
        raise ValueError("Gate wire is outside the state register.")
    remaining = tuple(wire for wire in range(num_wires) if wire not in wires)
    permutation = wires + remaining
    inverse = tuple(int(index) for index in np.argsort(permutation))
    tensor = jnp.reshape(state, (2,) * num_wires)
    front = jnp.transpose(tensor, permutation)
    front = jnp.reshape(front, (2 ** len(wires), -1))
    dtype = jnp.result_type(state.dtype, gate.dtype)
    updated = gate.astype(dtype) @ front.astype(dtype)
    updated = jnp.reshape(updated, (2,) * num_wires)
    return jnp.reshape(jnp.transpose(updated, inverse), (-1,))


def _apply_su4(params, state, wires, total_wires):
    state = apply_gate(state, _rot(*params[:3]), [wires[0]], total_wires)
    state = apply_gate(state, _rot(*params[3:6]), [wires[1]], total_wires)
    state = apply_gate(
        state, jnp.asarray(_CNOT), [wires[1], wires[0]], total_wires
    )
    state = apply_gate(state, _rz(params[6]), [wires[0]], total_wires)
    state = apply_gate(state, _ry(params[7]), [wires[1]], total_wires)
    state = apply_gate(state, jnp.asarray(_CNOT), wires, total_wires)
    state = apply_gate(state, _ry(params[8]), [wires[1]], total_wires)
    state = apply_gate(
        state, jnp.asarray(_CNOT), [wires[1], wires[0]], total_wires
    )
    state = apply_gate(state, _rot(*params[9:12]), [wires[0]], total_wires)
    return apply_gate(
        state, _rot(*params[12:15]), [wires[1]], total_wires
    )


def _apply_su_brickwall(params, state, wires, total_wires):
    """Apply SU brickwall with each CZ/RY/RY/RZ/RZ brick fused."""
    n = len(wires)
    required = 4**n - 1
    initial = 3 * n
    for wire_index, wire in enumerate(wires):
        state = apply_gate(
            state,
            _rot(
                params[wire_index],
                params[n + wire_index],
                params[2 * n + wire_index],
            ),
            [wire],
            total_wires,
        )
    num_cz = math.ceil((required - initial) / 4)
    relative = [(2 * index) % (n - 1) for index in range(num_cz)]
    if n % 2:
        relative = [
            value + (index // (n // 2)) % 2
            for index, value in enumerate(relative)
        ]
    identity = jnp.eye(2, dtype=state.dtype)
    parameter_index = initial
    for block_index in range(num_cz):
        start = relative[block_index]
        wire0, wire1 = wires[start], wires[start + 1]
        count = min(4, required - parameter_index)
        gate = jnp.asarray(_CZ, dtype=state.dtype)
        gate = jnp.kron(_ry(params[parameter_index]), identity) @ gate
        if count > 1:
            gate = jnp.kron(identity, _ry(params[parameter_index + 1])) @ gate
        if count > 2:
            gate = jnp.kron(_rz(params[parameter_index + 2]), identity) @ gate
        if count > 3:
            gate = jnp.kron(identity, _rz(params[parameter_index + 3])) @ gate
        state = apply_gate(state, gate, [wire0, wire1], total_wires)
        parameter_index += count
    return state


def apply_unitary_part(params, state, wires, total_wires, n_layers=1):
    """Apply the exact small-width unitary ansatz used by the SCAN model."""
    wires = tuple(int(wire) for wire in wires)
    n = len(wires)
    if n < 1 or n > 4:
        raise ValueError("The pure-JAX fast path supports one to four data wires.")
    per_layer = 4**n - 1
    for layer_index in range(n_layers):
        layer = params[layer_index * per_layer : (layer_index + 1) * per_layer]
        if n == 1:
            state = apply_gate(state, _rot(*layer[:3]), wires, total_wires)
        elif n == 2:
            state = _apply_su4(layer, state, wires, total_wires)
        else:
            state = _apply_su_brickwall(layer, state, wires, total_wires)
    return state


def normalize_state(state, epsilon=1.0e-12):
    """Normalize with a finite zero-branch gradient and sentinel state."""
    squared_norm = jnp.real(jnp.vdot(state, state))
    threshold = jnp.asarray(epsilon, dtype=squared_norm.dtype) ** 2
    denominator = jnp.sqrt(jnp.maximum(squared_norm, threshold))
    normalized = state / denominator
    sentinel = jnp.zeros_like(state).at[0].set(1)
    return jnp.where(
        squared_norm > threshold,
        normalized,
        jax.lax.stop_gradient(sentinel),
    )


def svd_parameter_count(n=TOKEN_QUBITS, n_layers=1):
    return 2 * n_layers * (4**n - 1) + 2**n


def nonlinear_parameter_count(n=TOKEN_QUBITS, layers1=1, layers2=1):
    return (layers1 + layers2) * (4**n - 1)


def apply_svd(params, input_state, n_layers=1, epsilon=1.0e-12):
    """Apply the normalized successful U diag(cos(theta)) V branch."""
    n = int(round(math.log2(input_state.shape[0])))
    unitary_count = n_layers * (4**n - 1)
    diagonal_count = 2**n
    state = normalize_state(input_state, epsilon)
    state = apply_unitary_part(
        params[:unitary_count], state, tuple(range(n)), n, n_layers
    )
    diagonal_end = unitary_count + diagonal_count
    state = jnp.cos(params[unitary_count:diagonal_end]).astype(state.dtype) * state
    success = jnp.real(jnp.vdot(state, state))
    state = normalize_state(state, epsilon)
    state = apply_unitary_part(
        params[diagonal_end:], state, tuple(range(n)), n, n_layers
    )
    return state, success


def embedding_table(embedding_logits):
    """Map each trainable seven-logit row to a normalized real 8-vector."""
    width = embedding_logits.shape[-1]
    bound = 0.8 / math.sqrt(width)
    prefix = bound * jnp.tanh(embedding_logits)
    remainder = jnp.maximum(
        1.0 - jnp.sum(prefix * prefix, axis=-1, keepdims=True), 0.0
    )
    return jnp.concatenate((prefix, jnp.sqrt(remainder)), axis=-1)


def apply_position_encoding(state, position, position_params):
    """Apply paper RX((position+1)*theta_k) rotations to one token state."""
    n = len(position_params)
    angle_scale = jnp.asarray(position + 1, dtype=position_params.dtype)
    for wire in range(n):
        state = apply_gate(
            state,
            _rx(angle_scale * position_params[wire]),
            [wire],
            n,
        )
    return state


def prepare_positioned_states(states, positions, position_params):
    return jax.vmap(apply_position_encoding, in_axes=(0, 0, None))(
        states, positions, position_params
    )


def _terminal_mcx(state, data_wires, extra_control, total_wires):
    target = data_wires[-1]
    controls = (extra_control, *data_wires[:-1])
    return apply_gate(
        state,
        _mcx(len(controls), state.dtype),
        (*controls, target),
        total_wires,
    )


def apply_nonlinear(
    params,
    input_state,
    layers1=1,
    layers2=1,
    epsilon=1.0e-12,
):
    """Apply the paper Softmax/Sigmoid topology and select ancilla outcome zero."""
    n = int(round(math.log2(input_state.shape[0])))
    data_wires = tuple(range(n))
    ancilla = n
    total_wires = n + 1
    state = jnp.kron(normalize_state(input_state, epsilon), jnp.asarray([1, 0]))
    state = apply_gate(state, jnp.asarray(_H), [ancilla], total_wires)

    first_count = layers1 * (4**n - 1)
    if layers1:
        state = apply_unitary_part(
            params[:first_count], state, data_wires, total_wires, layers1
        )
        state = _terminal_mcx(state, data_wires, ancilla, total_wires)

    second_wires = (*data_wires[:-1], ancilla)
    if layers2:
        state = apply_unitary_part(
            params[first_count:], state, second_wires, total_wires, layers2
        )
        state = _terminal_mcx(
            state, second_wires, data_wires[-1], total_wires
        )

    # Ancilla is the least-significant wire in this register ordering.
    selected = jnp.reshape(state, (2**n, 2))[:, 0]
    success = jnp.real(jnp.vdot(selected, selected))
    return normalize_state(selected, epsilon), success


def apply_strongly_entangling(state, weights):
    """Reproduce PennyLane StronglyEntanglingLayers with default CNOT ranges."""
    num_layers, num_wires, _ = weights.shape
    for layer in range(num_layers):
        for wire in range(num_wires):
            state = apply_gate(
                state, _rot(*weights[layer, wire]), [wire], num_wires
            )
        if num_wires > 1:
            gate_range = layer % (num_wires - 1) + 1
            for control in range(num_wires):
                target = (control + gate_range) % num_wires
                state = apply_gate(
                    state,
                    jnp.asarray(_CNOT),
                    [control, target],
                    num_wires,
                )
    return state


def pair_swap_zero_row(
    query_state,
    accumulator_state,
    key_state,
    value_state,
    entangling_weights,
):
    """Return P(final swap=0, accumulator=z) for one context pair.

    K and Q are traced only after their completed SwapTest.  The resulting
    two-component mixture is exact because neither register participates in
    the following swap/accumulator/value entangler.
    """
    fidelity = jnp.clip(
        jnp.real(jnp.vdot(key_state, query_state) * jnp.vdot(query_state, key_state)),
        0.0,
        1.0,
    )
    branch_weights = jnp.stack(((1.0 + fidelity) / 2, (1.0 - fidelity) / 2))
    swap_states = jnp.eye(2, dtype=accumulator_state.dtype)

    def evolve(swap_state):
        initial = jnp.kron(
            jnp.kron(swap_state, accumulator_state), value_state
        )
        final = apply_strongly_entangling(initial, entangling_weights)
        probabilities = jnp.abs(jnp.reshape(final, (2, TOKEN_DIM, TOKEN_DIM))) ** 2
        return jnp.sum(probabilities[0], axis=-1)

    rows = jax.vmap(evolve)(swap_states)
    return jnp.sum(branch_weights[:, None] * rows, axis=0)


def aggregate_swap_zero_rows(
    swap_zero_rows,
    context_mask,
    log_temperature,
    epsilon=1.0e-12,
):
    """Condition on final swap=0, softmax-weight pairs, and mix probabilities."""
    successes = jnp.sum(swap_zero_rows, axis=-1)
    safe_successes = jnp.maximum(successes, epsilon)
    conditioned = swap_zero_rows / safe_successes[:, None]
    temperature = jnp.exp(jnp.clip(jnp.ravel(log_temperature)[0], -5.0, 5.0))
    logits = jnp.log(safe_successes + epsilon) / temperature
    logits = jnp.where(context_mask, logits, -jnp.inf)
    weights = jax.nn.softmax(logits)
    aggregate = jnp.sum(weights[:, None] * conditioned, axis=0)
    aggregate = jnp.maximum(aggregate, 0.0) + epsilon
    return aggregate / jnp.sum(aggregate), weights, successes


def output_probabilities(
    context_probabilities,
    softmax_params,
    ffn_gw1,
    sigmoid_params,
    ffn_gw2,
    unitary_layers=1,
    softmax_layers1=1,
    softmax_layers2=1,
    sigmoid_layers1=1,
    sigmoid_layers2=1,
    epsilon=1.0e-12,
):
    """Apply Softmax -> GW1 -> Sigmoid -> GW2 and measure the token register."""
    probabilities = jnp.maximum(context_probabilities, 0.0) + epsilon
    state = jnp.sqrt(probabilities / jnp.sum(probabilities))
    state, _ = apply_nonlinear(
        softmax_params,
        state,
        softmax_layers1,
        softmax_layers2,
        epsilon,
    )
    state, _ = apply_svd(ffn_gw1, state, unitary_layers, epsilon)
    state, _ = apply_nonlinear(
        sigmoid_params,
        state,
        sigmoid_layers1,
        sigmoid_layers2,
        epsilon,
    )
    state, _ = apply_svd(ffn_gw2, state, unitary_layers, epsilon)
    result = jnp.abs(state) ** 2
    return result / jnp.sum(result)


def parameter_norm(parameters: Mapping[str, jax.Array]):
    leaves = jax.tree_util.tree_leaves(parameters)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


__all__ = [
    "TOKEN_QUBITS",
    "TOKEN_DIM",
    "PAIR_ACTIVE_QUBITS",
    "LOGICAL_PAIR_QUBITS",
    "UNITARY_PARAMETERS_PER_LAYER",
    "SVD_PARAMETERS_PER_LAYER",
    "NONLINEAR_PARAMETERS_PER_LAYER_PAIR",
    "apply_gate",
    "apply_unitary_part",
    "apply_svd",
    "apply_nonlinear",
    "apply_position_encoding",
    "prepare_positioned_states",
    "apply_strongly_entangling",
    "pair_swap_zero_row",
    "aggregate_swap_zero_rows",
    "output_probabilities",
    "embedding_table",
    "normalize_state",
    "svd_parameter_count",
    "nonlinear_parameter_count",
    "parameter_norm",
]




