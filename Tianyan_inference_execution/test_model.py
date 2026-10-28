import os
import sys

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from attention import controlled_su16_array
from embedding import random_embedding_logits
from my_model import (
    _primary_v_token_wires,
    my_model_with_attention_fnn,
    my_model_with_attention_fnn_parameter_breakdown,
    my_model_with_attention_fnn_with_quantum_embedding,
)


DEFAULT_COMPLEX_PRECISION_BITS = 128
PRECISION_ENV_VAR = "QT_COMPLEX_PRECISION"


def _resolve_complex_precision_bits():
    """Select NumPy complex precision: 64 -> complex64, 128 -> complex128."""
    raw_precision = os.environ.get(PRECISION_ENV_VAR, str(DEFAULT_COMPLEX_PRECISION_BITS))

    cli_args = sys.argv[1:]
    for idx, arg in enumerate(cli_args):
        if arg.startswith("--precision="):
            raw_precision = arg.split("=", 1)[1]
            break
        if arg == "--precision":
            if idx + 1 >= len(cli_args):
                raise ValueError("Missing value after --precision. Use 64 or 128.")
            raw_precision = cli_args[idx + 1]
            break

    normalized = str(raw_precision).strip().lower()
    if normalized in {"64", "complex64", "c64"}:
        return 64
    if normalized in {"128", "complex128", "c128"}:
        return 128
    raise ValueError(
        f"Unsupported precision '{raw_precision}'. Use 64/complex64 or 128/complex128."
    )


SELECTED_COMPLEX_PRECISION_BITS = _resolve_complex_precision_bits()
PARAM_DTYPE = np.float64 if SELECTED_COMPLEX_PRECISION_BITS == 128 else np.float32


def _print_shared_parameter_breakdown(breakdown, title="Parameter Breakdown"):
    total = breakdown["total"]
    attention = breakdown["attention"]
    model = breakdown["model"]
    ffn = breakdown["ffn"]

    def pct(count):
        return 100.0 * count / total if total else 0.0

    print(f"========== {title} ==========")
    print(
        f"Attention Q params: {attention['shared_per_q']} "
        f"(used {attention['q_call_count']} time(s), {pct(attention['shared_per_q']):.2f}% of total)"
    )
    print(
        f"Attention K params: {attention['shared_per_k']} "
        f"(used {attention['k_call_count']} time(s), {pct(attention['shared_per_k']):.2f}% of total)"
    )
    print(
        f"Attention V params: {attention['shared_per_v']} "
        f"(used {attention['v_call_count']} time(s), {pct(attention['shared_per_v']):.2f}% of total)"
    )
    print(
        f"Attention total params: {attention['total']} "
        f"({pct(attention['total']):.2f}% of total)"
    )
    print(
        f"Shared model entangling params: {model['entangling_shared']} "
        f"({pct(model['entangling_total']):.2f}% of total)"
    )
    print(
        f"Shared nonlinear params: {model['shared_nonlinear']} "
        f"(applied {model['nonlinear_call_count']} time(s), {pct(model['shared_nonlinear']):.2f}% of total)"
    )
    print(
        f"FFN params: {ffn['total']} "
        f"(input qubits per token: {ffn['input_qubits']}, reused {ffn['call_count']} time(s), {pct(ffn['total']):.2f}% of total)"
    )
    print(f"Total params: {total}\n")


def test_forward_pass():
    print(
        f"NumPy precision mode: complex{SELECTED_COMPLEX_PRECISION_BITS} / "
        f"float{SELECTED_COMPLEX_PRECISION_BITS // 2}"
    )

    m = 2
    n = 2
    entangling_layers = 1
    nonlinear_layers1 = 1
    nonlinear_layers2 = 1
    attention_num_blocks = 2
    attention_repeat_time = 1
    ffn_repeat_time = 1

    len_q = m * (m - 1) * n
    len_k = m * (m - 1) * n
    len_v = m * m * n
    swap_ancilla = m * (m - 1)
    wires_per_model = len_q + len_k + len_v + swap_ancilla
    total_wires = wires_per_model + 2
    wires = list(range(total_wires))
    output_wire = _primary_v_token_wires(wires[:wires_per_model], n, m)[0][0]

    print("========== Qubit Layout ==========")
    print(f"Model: m={m}, n={n}, attention_num_blocks={attention_num_blocks}")
    print(f"Q wires: {len_q}, K wires: {len_k}, V wires: {len_v}")
    print(f"Swap-test ancilla: {swap_ancilla}, shared attention ancilla: 2")
    print(f"Total wires: {total_wires}")

    total_params = my_model_with_attention_fnn(
        wires=wires,
        n=n,
        m=m,
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
        n=n,
        m=m,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
    )

    print(f"Total parameters needed: {total_params}")
    _print_shared_parameter_breakdown(breakdown)

    rng = onp.random.default_rng(42)
    params = np.asarray(rng.uniform(size=(total_params,)), dtype=PARAM_DTYPE)

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(
        dev,
        interface="autograd",
        mcm_method="device",
    )
    def circuit(model_params):
        my_model_with_attention_fnn(
            wires=wires,
            n=n,
            m=m,
            params=model_params,
            entangling_layers=entangling_layers,
            nonlinear_layers1=nonlinear_layers1,
            nonlinear_layers2=nonlinear_layers2,
            attention_num_blocks=attention_num_blocks,
            attention_repeat_time=attention_repeat_time,
            ffn_repeat_time=ffn_repeat_time,
            count_only=False,
        )
        return qml.probs(wires=[output_wire])

    print("Running forward pass...")
    result = circuit(params)
    print("Forward pass succeeded.")
    print("Output probabilities on wire 0:", result)


def test_forward_pass_with_quantum_embedding():
    print(
        f"NumPy precision mode: complex{SELECTED_COMPLEX_PRECISION_BITS} / "
        f"float{SELECTED_COMPLEX_PRECISION_BITS // 2}"
    )
    print("Running forward pass with shared token embeddings for Q/K/V registers")

    m = 2
    n = 1
    entangling_layers = 1
    nonlinear_layers1 = 1
    nonlinear_layers2 = 1
    attention_num_blocks = 2
    attention_repeat_time = 1
    ffn_repeat_time = 1
    vocab_size = 8

    len_q = m * (m - 1) * n
    len_k = m * (m - 1) * n
    len_v = m * m * n
    wires_per_model = len_q + len_k + len_v + m * (m - 1)
    total_wires = wires_per_model + 2
    wires = list(range(total_wires))
    output_wire = _primary_v_token_wires(wires[:wires_per_model], n, m)[0][0]

    total_params = my_model_with_attention_fnn_with_quantum_embedding(
        wires=wires,
        n=n,
        m=m,
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
        n=n,
        m=m,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
    )

    print(f"Embedding test total parameters: {total_params}")
    print("Trainable embedding logits are external inputs and are not included below.")
    _print_shared_parameter_breakdown(
        breakdown, title="Parameter Breakdown (Embeddings Excluded)"
    )

    rng = onp.random.default_rng(123)
    params = np.asarray(rng.uniform(size=(total_params,)), dtype=PARAM_DTYPE)
    embedding_logits = random_embedding_logits(
        rng, vocab_size, n, dtype=PARAM_DTYPE, requires_grad=True
    )
    token_indices = rng.integers(0, vocab_size, size=(m,), dtype=onp.int32)

    print("Selected token indices:", token_indices)

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(
        dev,
        interface="autograd",
        mcm_method="device",
    )
    def circuit(model_params, token_ids, embeddings):
        my_model_with_attention_fnn_with_quantum_embedding(
            wires=wires,
            n=n,
            m=m,
            params=model_params,
            entangling_layers=entangling_layers,
            nonlinear_layers1=nonlinear_layers1,
            nonlinear_layers2=nonlinear_layers2,
            token_indices=token_ids,
            embedding_logits=embeddings,
            attention_num_blocks=attention_num_blocks,
            attention_repeat_time=attention_repeat_time,
            ffn_repeat_time=ffn_repeat_time,
            count_only=False,
        )
        return qml.probs(wires=[output_wire])

    result = circuit(params, token_indices, embedding_logits)
    print("Embedding forward pass succeeded.")
    print("Output probabilities on wire 0:", result)


def test_attention_parameter_sharing_is_per_token_copy():
    m = 3
    n = 2
    entangling_layers = 1
    nonlinear_layers1 = 1
    nonlinear_layers2 = 1
    attention_num_blocks = 2
    attention_repeat_time = 1
    ffn_repeat_time = 1

    len_q = m * (m - 1) * n
    len_k = m * (m - 1) * n
    len_v = m * m * n
    total_wires = len_q + len_k + len_v + m * (m - 1) + 2
    wires = list(range(total_wires))

    total_params = my_model_with_attention_fnn(
        wires=wires,
        n=n,
        m=m,
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
        n=n,
        m=m,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
    )

    expected_shared_params = controlled_su16_array(
        [],
        total_qubits=n,
        num_blocks=attention_num_blocks,
        repeat_time=attention_repeat_time,
        wires=list(range(n)),
        ancilla=[-1, -2],
        count_only=True,
    )

    assert breakdown["attention"]["shared_per_q"] == expected_shared_params
    assert breakdown["attention"]["shared_per_k"] == expected_shared_params
    assert breakdown["attention"]["shared_per_v"] == expected_shared_params
    assert breakdown["attention"]["q_call_count"] == m * (m - 1)
    assert breakdown["attention"]["k_call_count"] == m * (m - 1)
    assert breakdown["attention"]["v_call_count"] == m * m
    assert breakdown["total"] == total_params


if __name__ == "__main__":
    test_forward_pass()
    print("\n" + "=" * 80 + "\n")
    test_forward_pass_with_quantum_embedding()
