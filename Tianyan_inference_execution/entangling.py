"""Quantum-mixing StronglyEntangling layers.

The register layout follows the mixing construction in the paper: for every
ordered pair ``(i, j)``, ``i != j``, one block acts on the SwapTest result
``ST(i, j)`` and one copy each of ``V_i`` and ``V_j``.

There are ``m`` copies of every value state. Copy ``i`` is reserved for output
``i``. Blocks belonging to different outputs therefore have disjoint value
wires and may be executed in parallel. The ``m - 1`` blocks that build the
same output share its accumulator ``V_i^(i)`` and are necessarily ordered.
"""

import pennylane as qml
from pennylane import numpy as np
from swaptest import ordered_partner_indices


def _value_copy_wires(value_wires, n_qubits, m_states, token_idx, copy_idx):
    """Return copy ``copy_idx`` of value state ``V_token_idx``."""
    start = (token_idx * m_states + copy_idx) * n_qubits
    return list(value_wires[start : start + n_qubits])


def quantum_mixing_layout(wires, n_qubits, m_states):
    """Describe ordered mixing blocks without queuing quantum operations.

    The outer list contains parallel rounds. Items within one round use
    disjoint wires. Each item records the output/partner indices, SwapTest
    ancilla, two value registers, and full StronglyEntangling wire list.
    """
    if n_qubits <= 0 or m_states <= 1:
        raise ValueError("m_states must be > 1 and n_qubits must be > 0")

    value_qubit_count = m_states * m_states * n_qubits
    ancilla_count = m_states * (m_states - 1)
    expected_wires = value_qubit_count + ancilla_count
    if len(wires) != expected_wires:
        raise ValueError(
            f"Expected {expected_wires} mixing wires, but got {len(wires)}."
        )

    value_wires = wires[:value_qubit_count]
    swaptest_ancillas = wires[value_qubit_count:]
    partners = [
        ordered_partner_indices(output_idx, m_states)
        for output_idx in range(m_states)
    ]

    rounds = []
    for pair_slot in range(m_states - 1):
        parallel_round = []
        for output_idx in range(m_states):
            partner_idx = partners[output_idx][pair_slot]
            ancilla = swaptest_ancillas[
                output_idx * (m_states - 1) + pair_slot
            ]
            accumulator = _value_copy_wires(
                value_wires, n_qubits, m_states, output_idx, output_idx
            )
            other_value = _value_copy_wires(
                value_wires, n_qubits, m_states, partner_idx, output_idx
            )
            block_wires = [ancilla, *accumulator, *other_value]
            parallel_round.append(
                {
                    "output": output_idx,
                    "partner": partner_idx,
                    "pair_slot": pair_slot,
                    "ancilla": ancilla,
                    "accumulator": accumulator,
                    "other_value": other_value,
                    "wires": block_wires,
                }
            )
        rounds.append(parallel_round)

    return rounds


def entangling_layer(params, wires, n_qubits, m_states):
    """Apply the pairwise StronglyEntangling part of quantum mixing.

    Args:
        params: Array shaped
            ``(m_states, m_states - 1, layers, 2*n_qubits + 1, 3)``.
            The first two axes select the output token and ordered pair slot.
        wires: Value wires followed by SwapTest ancillas. The value section has
            ``m_states`` token groups with ``m_states`` copies in each group.
        n_qubits: Qubits in one value state.
        m_states: Number of sequence states/tokens.

    Blocks for the same pair slot are queued together because their wires are
    disjoint and they form one parallel circuit round.
    """
    params = np.asarray(params)
    pair_width = 2 * n_qubits + 1

    if params.ndim != 5:
        raise ValueError(
            f"params must be a 5-D array, got {params.ndim} dimensions."
        )
    expected_prefix = (m_states, m_states - 1)
    if params.shape[:2] != expected_prefix:
        raise ValueError(
            "params first two dimensions must be "
            f"{expected_prefix}, got {params.shape[:2]}."
        )
    if params.shape[3] != pair_width:
        raise ValueError(
            f"Each mixing block needs {pair_width} wires, got {params.shape[3]}."
        )
    if params.shape[4] != 3:
        raise ValueError(
            f"The final params dimension must be 3, got {params.shape[4]}."
        )

    rounds = quantum_mixing_layout(wires, n_qubits, m_states)
    if params.shape[2] == 0:
        return

    for parallel_round in rounds:
        for block in parallel_round:
            qml.StronglyEntanglingLayers(
                params[block["output"], block["pair_slot"]],
                wires=block["wires"],
            )
