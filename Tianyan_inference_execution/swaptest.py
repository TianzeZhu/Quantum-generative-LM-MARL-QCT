import pennylane as qml


def ordered_partner_indices(output_idx, m_states):
    """Partners for output ``i``, in query-major SwapTest ancilla order."""
    return [idx for idx in range(m_states) if idx != output_idx]


def swaptest_layout(wires, n_qubits, m_states):
    """Return the wire assignment for every ordered pair ``(i, j)``, ``i != j``.

    Ancillas are query/output-major: all pairs for output 0, then output 1,
    and so on. This is the same order consumed by ``quantum_mixing_layout``.
    """
    if n_qubits <= 0 or m_states <= 1:
        raise ValueError("m_states must be > 1 and n_qubits must be > 0")

    total_q_qubits = m_states * (m_states - 1) * n_qubits
    total_k_qubits = m_states * (m_states - 1) * n_qubits
    total_v_qubits = m_states * m_states * n_qubits
    ancilla_start = total_q_qubits + total_k_qubits + total_v_qubits
    required_len = ancilla_start + m_states * (m_states - 1)
    if len(wires) < required_len:
        raise ValueError(
            f"Need at least {required_len} wires, but got {len(wires)}."
        )

    layout = []
    for output_idx in range(m_states):
        for pair_slot, partner_idx in enumerate(
            ordered_partner_indices(output_idx, m_states)
        ):
            q_start = (
                output_idx * (m_states - 1) * n_qubits
                + pair_slot * n_qubits
            )
            key_copy_idx = output_idx if output_idx < partner_idx else output_idx - 1
            k_start = (
                total_q_qubits
                + partner_idx * (m_states - 1) * n_qubits
                + key_copy_idx * n_qubits
            )
            ancilla_idx = ancilla_start + output_idx * (m_states - 1) + pair_slot
            layout.append(
                {
                    "output": output_idx,
                    "partner": partner_idx,
                    "pair_slot": pair_slot,
                    "query_wires": list(wires[q_start : q_start + n_qubits]),
                    "key_wires": list(wires[k_start : k_start + n_qubits]),
                    "ancilla": wires[ancilla_idx],
                }
            )

    return layout

def swaptest_circuit(state1wires, state2wires, ancqubit):# (target state 1, target state 2, ancilla qubit)
    
    qml.Hadamard(wires=ancqubit)

    if len(state1wires) != len(state2wires):
        raise ValueError("交换测试中的两个目标态维度不一致！")
    
    for i in range(len(state1wires)):
        target_wire1 = state1wires[i]  
        target_wire2 = state2wires[i]  
        qml.CSWAP(wires=[ancqubit, target_wire1, target_wire2])
    qml.Hadamard(ancqubit)  

def swaptest_layer(wires, n_qubits, m_states): # (wire list, qubits per state n, number of quantum states m)
    # wires: List of all qubits assigned to the current model.
    # (m-1) Qi states, with i ranging from 1 to m.
    # (m-1) Ki states, with i ranging from 1 to m.
    # (m)   Vi states, with i ranging from 1 to m.
    
    for block in swaptest_layout(wires, n_qubits, m_states):
        swaptest_circuit(
            block["query_wires"],
            block["key_wires"],
            block["ancilla"],
        )

# #################################    swaptest_circuit test code    ####################################


# dev = qml.device("default.qubit", wires=5)  # Two 2-qubit quantum states + one ancilla qubit.
# def prepare_state1():
#     qml.StatePrep([1,0,0,0], [0,1])
# def prepare_state2():
#     qml.StatePrep([1,0,0,0], [2,3])
# def prepare_different_state2():
#     # Prepare a different quantum state |01>.
#     qml.StatePrep([1,0,0,0], [2,3])
#     qml.PauliX(wires=3) 
    
# @qml.qnode(dev, interface='autograd')
# def test_same_states():
#     # Test two identical quantum states.
#     prepare_state1()
#     prepare_state2()
#     swaptest_circuit([0,1], [2,3], 4)  
#     return qml.probs(wires=[4])

# @qml.qnode(dev, interface='autograd')
# def test_different_states():
#     # Test two different quantum states.
#     prepare_state1()
#     prepare_different_state2()
#     swaptest_circuit([0,1], [2,3], 4)  
#     return qml.probs(wires=[4])

# # Run tests.
# print("Testing two identical quantum states |00> and |00>:")
# result_same = test_same_states()
# print(f"Probability of measuring ancilla as 0: {result_same[0]:.4f}")
# print(f"Probability of measuring ancilla as 1: {result_same[1]:.4f}")

# print("\nTesting two different quantum states |00> and |01>:")
# result_diff = test_different_states()
# print(f"Probability of measuring ancilla as 0: {result_diff[0]:.4f}")
# print(f"Probability of measuring ancilla as 1: {result_diff[1]:.4f}")

# # Compute the fidelity estimate.
# fidelity_same = 2 * result_same[0] - 1
# fidelity_diff = 2 * result_diff[0] - 1

# print(f"\nEstimated fidelity (identical states): {fidelity_same:.4f}")
# print(f"Estimated fidelity (different states): {fidelity_diff:.4f}")

# fig, ax = qml.draw_mpl(test_different_states)()
# plt.show()



##################################    swap_test_layer test code    ####################################

# m = 3
# n = 1

# # Compute the total number of qubits.
# total_q_qubits = m * (m - 1) * n
# total_k_qubits = m * (m - 1) * n
# total_v_qubits = m * m * n
# total_ancilla = m * (m - 1)
# total_qubits = total_q_qubits + total_k_qubits + total_v_qubits + total_ancilla

# print(f"Total number of qubits: {total_qubits}")
# print(f"Number of Q-section qubits: {total_q_qubits} (0-{total_q_qubits-1})")
# print(f"Number of K-section qubits: {total_k_qubits} ({total_q_qubits}-{total_q_qubits+total_k_qubits-1})")
# print(f"Number of V-section qubits: {total_v_qubits} ({total_q_qubits+total_k_qubits}-{total_q_qubits+total_k_qubits+total_v_qubits-1})")
# print(f"Number of ancilla qubits: {total_ancilla} ({total_q_qubits+total_k_qubits+total_v_qubits}-{total_qubits-1})")


# dev = qml.device("default.qubit", wires=total_qubits)
# @qml.qnode(dev, interface='autograd')
# def circuit():
#     wires = list(range(total_qubits)) # Create the wires list.
#     swaptest_layer(wires, n, m) # Pass in wires.
#     return qml.state()

# fig, ax = qml.draw_mpl(circuit)()
# plt.show()
