"""Structural regression tests for the corrected parallel circuit layout."""

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from attention import controlled_su16_array
from circuits import non_unitary_diag_part
from diag_op import diag_op
from entangling import entangling_layer, quantum_mixing_layout
from ffn_new import diag_op_non
from my_model import _entangling_param_count, _primary_v_token_wires
from swaptest import swaptest_layout


def test_swaptest_and_strong_entangling_pairs_are_aligned():
    m_states = 3
    n_qubits = 2
    q_count = m_states * (m_states - 1) * n_qubits
    k_count = q_count
    value_count = m_states * m_states * n_qubits
    ancilla_count = m_states * (m_states - 1)
    all_wires = list(range(q_count + k_count + value_count + ancilla_count))
    mixing_wires = all_wires[q_count + k_count :]

    swap_blocks = swaptest_layout(all_wires, n_qubits, m_states)
    mixing_rounds = quantum_mixing_layout(mixing_wires, n_qubits, m_states)
    swap_by_pair = {
        (block["output"], block["partner"]): block for block in swap_blocks
    }

    assert len(swap_blocks) == m_states * (m_states - 1)
    assert len(mixing_rounds) == m_states - 1

    for parallel_round in mixing_rounds:
        round_wires = []
        for block in parallel_round:
            pair = (block["output"], block["partner"])
            assert swap_by_pair[pair]["ancilla"] == block["ancilla"]
            assert len(block["wires"]) == 2 * n_qubits + 1
            round_wires.extend(block["wires"])

        # Outputs in one round use separate V copies and separate ancillas.
        assert len(round_wires) == len(set(round_wires))

    # The accumulator for output i is the diagonal V copy V_i^(i), and it is
    # reused sequentially for all partners of that output.
    for output_idx in range(m_states):
        accumulators = [
            tuple(round_[output_idx]["accumulator"])
            for round_ in mixing_rounds
        ]
        assert len(set(accumulators)) == 1


def test_entangling_template_uses_one_ordered_pair_per_block():
    m_states = 3
    n_qubits = 1
    layers = 1
    pair_width = 2 * n_qubits + 1
    wire_count = m_states * m_states * n_qubits + m_states * (m_states - 1)
    wires = list(range(wire_count))
    params = np.zeros(
        (m_states, m_states - 1, layers, pair_width, 3),
        requires_grad=True,
    )

    expected_wires = [
        block["wires"]
        for parallel_round in quantum_mixing_layout(wires, n_qubits, m_states)
        for block in parallel_round
    ]
    with qml.tape.QuantumTape() as tape:
        entangling_layer(params, wires, n_qubits, m_states)

    operations = [
        op for op in tape.operations if op.name == "StronglyEntanglingLayers"
    ]
    assert len(operations) == m_states * (m_states - 1)
    assert [list(op.wires) for op in operations] == expected_wires
    assert _entangling_param_count(n_qubits, m_states, layers) == params.size


def test_downstream_layers_use_the_mixed_output_accumulators():
    m_states = 3
    n_qubits = 2
    total_wires = (3 * m_states - 2) * m_states * n_qubits + m_states * (m_states - 1)
    wires = list(range(total_wires))
    output_wires = _primary_v_token_wires(wires, n_qubits, m_states)
    v_start = 2 * m_states * (m_states - 1) * n_qubits

    assert output_wires == [
        list(
            range(
                v_start + (token_idx * m_states + token_idx) * n_qubits,
                v_start + (token_idx * m_states + token_idx + 1) * n_qubits,
            )
        )
        for token_idx in range(m_states)
    ]


def _assert_final_rz(builder):
    with qml.tape.QuantumTape() as tape:
        builder()
    rz_indices = [
        idx
        for idx, op in enumerate(tape.operations)
        if op.name == "RZ" and list(op.wires) == [1]
    ]
    assert rz_indices
    rz_idx = rz_indices[-1]
    assert onp.allclose(float(tape.operations[rz_idx].data[0]), -onp.pi / 2)
    assert tape.operations[rz_idx + 1].name == "MidMeasureMP"
    assert list(tape.operations[rz_idx + 1].wires) == [1]


def test_non_unitary_diagonal_helpers_have_final_rz():
    _assert_final_rz(
        lambda: non_unitary_diag_part(np.asarray([0.1, 0.2]), [0], 1)
    )
    _assert_final_rz(lambda: diag_op(0, [0], 1))
    _assert_final_rz(lambda: diag_op_non(np.asarray([0.1, 0.2]), [0], 1))


def test_attention_non_unitary_blocks_have_final_rz():
    total_qubits = 2
    num_blocks = 2
    params_needed = controlled_su16_array(
        [],
        total_qubits=total_qubits,
        num_blocks=num_blocks,
        repeat_time=1,
        wires=[0, 1],
        ancilla=[2, 3],
        count_only=True,
    )
    with qml.tape.QuantumTape() as tape:
        controlled_su16_array(
            np.zeros(params_needed),
            total_qubits=total_qubits,
            num_blocks=num_blocks,
            repeat_time=1,
            wires=[0, 1],
            ancilla=[2, 3],
        )

    final_rz_indices = [
        idx
        for idx, op in enumerate(tape.operations)
        if op.name == "RZ"
        and list(op.wires) == [2]
        and onp.allclose(float(op.data[0]), -onp.pi / 2)
    ]
    assert len(final_rz_indices) == num_blocks
    for rz_idx in final_rz_indices:
        assert tape.operations[rz_idx + 1].name == "MidMeasureMP"
        assert list(tape.operations[rz_idx + 1].wires) == [2]
