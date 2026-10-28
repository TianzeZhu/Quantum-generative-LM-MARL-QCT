"""Small, executable smoke test for the parallel quantum-model code path.

This deliberately keeps the end-to-end model at m=2, n=1 (12 device wires).
The four-qubit unitary ansatz is checked separately so that ``unitary_part`` is
really executed without forcing the full parallel model to simulate 36+ wires.
"""

import argparse
from collections import Counter

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from attention import controlled_su16_array
from backend_config import (
    add_backend_arguments,
    backend_config_from_args,
    create_device,
)
from circuits import unitary_part
from embedding import random_embedding_logits
from my_model import (
    _primary_v_token_wires,
    my_model_with_attention_fnn_parameter_breakdown,
    my_model_with_attention_fnn_with_quantum_embedding,
)


SEED = 20260820


def _assert_probabilities(name, probabilities):
    values = onp.asarray(probabilities, dtype=float)
    if not onp.all(onp.isfinite(values)):
        raise AssertionError(f"{name}: probabilities contain NaN or infinity.")
    if not onp.isclose(values.sum(), 1.0, atol=1e-7):
        raise AssertionError(
            f"{name}: probabilities sum to {values.sum()}, expected 1."
        )


def smoke_unitary_part(rng, runtime_config):
    """Execute the new four-qubit SU16-backed unitary ansatz."""
    wires = [0, 1, 2, 3]
    param_count = unitary_part(
        [], wires, n_layers=1, block_pattern="stride2", count_only=True
    )
    params = np.asarray(rng.normal(scale=0.02, size=param_count))
    dev = create_device(4, runtime_config)

    @qml.qnode(
        dev,
        interface="autograd",
        diff_method=None if runtime_config.use_tianyan else "best",
    )
    def circuit(unitary_params):
        qml.Hadamard(wires=0)
        qml.RY(0.2, wires=2)
        unitary_part(
            unitary_params,
            wires,
            n_layers=1,
            block_pattern="stride2",
            bridge="long_range",
        )
        return qml.probs(wires=wires)

    probabilities = circuit(params)
    _assert_probabilities("unitary_part", probabilities)
    print(
        f"[PASS 1/3] unitary_part: 4 qubits, {param_count} parameters, "
        f"probability sum={float(onp.sum(probabilities)):.8f}"
    )


def smoke_controlled_attention(rng, runtime_config):
    """Execute a two-data-qubit controlled U-D-V attention block."""
    data_wires = [0, 1]
    ancilla_wires = [2, 3]
    num_blocks = 2
    param_count = controlled_su16_array(
        [],
        total_qubits=len(data_wires),
        num_blocks=num_blocks,
        repeat_time=1,
        wires=data_wires,
        ancilla=ancilla_wires,
        count_only=True,
    )
    params = np.asarray(rng.normal(scale=0.02, size=param_count))
    dev = create_device(4, runtime_config)

    @qml.qnode(
        dev,
        interface="autograd",
        diff_method=None if runtime_config.use_tianyan else "best",
        mcm_method="device",
    )
    def circuit(attention_params):
        qml.Hadamard(wires=0)
        qml.RY(0.3, wires=1)
        controlled_su16_array(
            attention_params,
            total_qubits=len(data_wires),
            num_blocks=num_blocks,
            repeat_time=1,
            wires=data_wires,
            ancilla=ancilla_wires,
        )
        return qml.probs(wires=data_wires)

    probabilities = circuit(params)
    _assert_probabilities("controlled attention", probabilities)
    print(
        f"[PASS 2/3] controlled attention: 2 data + 2 ancilla qubits, "
        f"{param_count} parameters, probability sum={float(onp.sum(probabilities)):.8f}"
    )


def smoke_full_parallel_chain(rng, runtime_config, with_gradient=False):
    """Run the complete small Embedding -> Attention -> Mixing -> FFN path."""
    m_states = 2
    n_qubits = 1
    entangling_layers = 1
    nonlinear_layers1 = 1
    nonlinear_layers2 = 1
    attention_num_blocks = 2
    attention_repeat_time = 1
    ffn_repeat_time = 1
    vocab_size = 4

    q_count = m_states * (m_states - 1) * n_qubits
    k_count = q_count
    value_count = m_states * m_states * n_qubits
    swap_ancilla_count = m_states * (m_states - 1)
    model_wire_count = q_count + k_count + value_count + swap_ancilla_count
    total_wire_count = model_wire_count + 2
    wires = list(range(total_wire_count))
    output_wires = [
        wire
        for token_wires in _primary_v_token_wires(
            wires[:model_wire_count], n_qubits, m_states
        )
        for wire in token_wires
    ]

    total_params = my_model_with_attention_fnn_with_quantum_embedding(
        wires=wires,
        n=n_qubits,
        m=m_states,
        params=None,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
        count_only=True,
    )
    breakdown = my_model_with_attention_fnn_parameter_breakdown(
        wires=wires,
        n=n_qubits,
        m=m_states,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
    )
    if breakdown["total"] != total_params:
        raise AssertionError(
            f"Parameter count mismatch: {breakdown['total']} != {total_params}."
        )

    params = np.array(
        rng.normal(scale=0.02, size=total_params), requires_grad=True
    )
    embedding_logits = random_embedding_logits(
        rng,
        vocab_size=vocab_size,
        num_qubits=n_qubits,
        requires_grad=False,
    )
    token_indices = onp.asarray([0, 1], dtype=onp.int32)
    dev = create_device(total_wire_count, runtime_config)

    @qml.qnode(
        dev,
        interface="autograd",
        diff_method=None if runtime_config.use_tianyan else "parameter-shift",
        mcm_method="device",
    )
    def circuit(model_params):
        my_model_with_attention_fnn_with_quantum_embedding(
            wires=wires,
            n=n_qubits,
            m=m_states,
            params=model_params,
            entangling_layers=entangling_layers,
            nonlinear_layers1=nonlinear_layers1,
            nonlinear_layers2=nonlinear_layers2,
            token_indices=token_indices,
            embedding_logits=embedding_logits,
            attention_num_blocks=attention_num_blocks,
            attention_repeat_time=attention_repeat_time,
            ffn_repeat_time=ffn_repeat_time,
            count_only=False,
        )
        return qml.probs(wires=output_wires)

    probabilities = circuit(params)
    _assert_probabilities("full parallel chain", probabilities)

    gradient_values = None
    if with_gradient:
        def loss_fn(model_params):
            return 1.0 - circuit(model_params)[0]

        gradient = qml.grad(loss_fn)(params)
        gradient_values = onp.asarray(gradient, dtype=float)
        if gradient_values.shape != (total_params,):
            raise AssertionError(
                f"Gradient shape is {gradient_values.shape}, expected {(total_params,)}."
            )
        if not onp.all(onp.isfinite(gradient_values)):
            raise AssertionError("Full-chain gradient contains NaN or infinity.")

    top_level_operations = Counter(op.name for op in circuit._tape.operations)
    print(
        f"[PASS 3/3] complete chain: m={m_states}, n={n_qubits}, "
        f"{total_wire_count} device wires, {total_params} parameters"
    )
    print(f"           output wires={output_wires}")
    print(f"           output probabilities={onp.asarray(probabilities)}")
    if gradient_values is None:
        print("           gradient check=skipped (use --with-gradient to enable)")
    else:
        print(f"           gradient L2 norm={onp.linalg.norm(gradient_values):.8f}")
    print(
        "           key top-level operations="
        f"{dict(sorted(top_level_operations.items()))}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a small smoke test for the parallel quantum-model chain."
    )
    parser.add_argument(
        "--with-gradient",
        action="store_true",
        help="also run one parameter-shift gradient check (slower)",
    )
    add_backend_arguments(parser)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually create the selected CQLib/TianYan device and submit jobs.",
    )
    args = parser.parse_args()
    runtime_config = backend_config_from_args(args)

    if args.with_gradient and runtime_config.use_tianyan:
        parser.error("--with-gradient is only supported by the local simulator.")
    if args.submit and not runtime_config.use_tianyan:
        parser.error("--submit requires --tianyan.")
    if runtime_config.use_tianyan and not args.submit:
        print(f"Selected runtime: {runtime_config.label}")
        print("Dry run only: add --submit to create the TianYan device and execute jobs.")
        return

    rng = onp.random.default_rng(SEED)
    print("Small parallel quantum-model smoke test")
    print("=" * 48)
    print(f"Runtime: {runtime_config.label}")
    smoke_unitary_part(rng, runtime_config)
    smoke_controlled_attention(rng, runtime_config)
    smoke_full_parallel_chain(
        rng,
        runtime_config,
        with_gradient=args.with_gradient,
    )
    print("=" * 48)
    print("ALL SMALL-SCALE CHECKS PASSED")


if __name__ == "__main__":
    main()
