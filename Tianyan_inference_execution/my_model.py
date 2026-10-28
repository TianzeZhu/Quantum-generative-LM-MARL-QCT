import pennylane as qml
from pennylane import numpy as np
from swaptest import swaptest_layer
from entangling import entangling_layer
from nonlinear import nonlinear_layer
from attention import controlled_su16_array
from ffn_new import ffn_new
from embedding import embedding_parameter_dimension, quantum_embedding
import math

def _primary_v_token_wires(wires, n, m):
    """Return the output accumulator ``V_i^(i)`` for every token ``i``."""
    v_start = 2 * m * (m - 1) * n
    token_wires = []
    for token_idx in range(m):
        token_base = v_start + token_idx * m * n
        output_copy_start = token_base + token_idx * n
        token_wires.append(wires[output_copy_start : output_copy_start + n])
    return token_wires



def _entangling_param_count(n, m, entangling_layers):
    pair_width = 2 * n + 1
    pair_count = m * (m - 1)
    return pair_count * entangling_layers * pair_width * 3


def _shared_nonlinear_param_count(n, nonlinear_layers1, nonlinear_layers2):
    if nonlinear_layers1 + nonlinear_layers2 == 0:
        return 0
    return nonlinear_layer(
        [],
        list(range(n)),
        nonlinear_layers1,
        nonlinear_layers2,
        ancilla_wire=-1,
        count_only=True,
    )

def my_model(wires, n, m, params, entangling_layers ,nonlinear_layers1, nonlinear_layers2, nonlinear_ancilla_wire=None, count_only=False):
    # Structure summary:
    # Q has m - 1 copies, K has m - 1 copies, and V has m copies.
    # The swap test, entangling layer, and nonlinear layer all use fixed wire slices.

    # Compute the entangling layer parameter count.
    pair_width = 2 * n + 1
    num_params_entlayer = _entangling_param_count(n, m, entangling_layers)

    # Compute the nonlinear layer parameter count.
    nonlinear_input_end = (3*m-2)*m*n + m*(m-1)
    tokenwise_nonlin_wires = _primary_v_token_wires(wires, n, m)
    num_params_nonlinlayer = _shared_nonlinear_param_count(
        n, nonlinear_layers1, nonlinear_layers2
    )
    ancilla_wire = nonlinear_ancilla_wire
    if num_params_nonlinlayer > 0:
        if ancilla_wire is None and len(wires) > nonlinear_input_end:
            ancilla_wire = wires[nonlinear_input_end]
        if ancilla_wire is None:
            ancilla_wire = 0
    
    total_params = num_params_entlayer + num_params_nonlinlayer
    
    if count_only:
        return total_params

    if (
        num_params_nonlinlayer > 0
        and nonlinear_ancilla_wire is None
        and len(wires) <= nonlinear_input_end
    ):
        raise ValueError("nonlinear_ancilla_wire must be provided when the model wires do not include a dedicated ancilla.")

    if params is None or len(params) < total_params:
        raise ValueError(f"参数不足，需要 {total_params} 个，实际只有 {len(params) if params is not None else 0} 个。")

    # Apply the swap test layer first.
    swaptest_layer(wires, n, m)

    params = np.asarray(params)

    wires_entlayers = wires[(2 * m - 2) * m * n : (3 * m - 2) * m * n + m * (m - 1)]
    params_entlayer = params[:num_params_entlayer].reshape(
        (m, m - 1, entangling_layers, pair_width, 3)
    )
    entangling_layer(params_entlayer, wires_entlayers, n, m)

    if num_params_nonlinlayer > 0:
        params_nonlinlayer = params[num_params_entlayer:]
        # Apply nonlinear to the primary V copy for each token, sharing parameters.
        for token_wires in tokenwise_nonlin_wires:
            nonlinear_layer(
                params_nonlinlayer,
                token_wires,
                nonlinear_layers1,
                nonlinear_layers2,
                ancilla_wire,
            )


def _group_token_replicas(register_wires, n, m, copies_per_token):
    groups = []
    for token_idx in range(m):
        token_groups = []
        token_base = token_idx * copies_per_token * n
        for copy_idx in range(copies_per_token):
            start = token_base + copy_idx * n
            token_groups.append(register_wires[start : start + n])
        groups.append(token_groups)
    return groups


def _collect_shared_attention_layout(wires, n, m, attention_num_blocks):
    len_q, len_k, len_v, wires_per_model, total_model_wires, total_needed = _qkv_register_layout(
        n, m, attention_num_blocks
    )
    if len(wires) < total_needed:
        raise ValueError(f"Need at least {total_needed} wires, got {len(wires)}.")

    model_wires = wires[:wires_per_model]

    # Reuse one work ancilla plus one extra diag_op ancilla across all Q/K/V calls.
    shared_attention_ancilla = wires[total_model_wires : total_model_wires + 2]

    q_wires = model_wires[0:len_q]
    k_wires = model_wires[len_q : len_q + len_k]
    v_wires = model_wires[len_q + len_k : len_q + len_k + len_v]

    return {
        "model_wires": model_wires,
        "q_wires": q_wires,
        "k_wires": k_wires,
        "v_wires": v_wires,
        "q_groups": _group_token_replicas(q_wires, n, m, max(m - 1, 0)),
        "k_groups": _group_token_replicas(k_wires, n, m, max(m - 1, 0)),
        "v_groups": _group_token_replicas(v_wires, n, m, m),
        "shared_attention_ancilla": shared_attention_ancilla,
        "wires_per_model": wires_per_model,
        "total_needed": total_needed,
    }


def _attention_param_count(target_qubits, attention_num_blocks, repeat_time):
    if target_qubits <= 0:
        return 0
    dummy_wires = list(range(target_qubits))
    return controlled_su16_array(
        [],
        target_qubits,
        attention_num_blocks,
        repeat_time=repeat_time,
        wires=dummy_wires,
        ancilla=[-1, -2],
        count_only=True,
    )


def _shared_attention_param_count(token_groups, attention_num_blocks, repeat_time):
    sample_qubits = None
    for token_group in token_groups:
        for copy_wires in token_group:
            current_qubits = len(copy_wires)
            if sample_qubits is None:
                sample_qubits = current_qubits
            elif current_qubits != sample_qubits:
                raise ValueError("All attention token copies must use the same number of qubits.")

    if sample_qubits is None:
        return 0

    return _attention_param_count(sample_qubits, attention_num_blocks, repeat_time)


def _attention_call_count(token_groups):
    return sum(len(token_group) for token_group in token_groups)


def _apply_attention_stage(
    params,
    token_groups,
    shared_attention_ancilla,
    attention_num_blocks,
    repeat_time,
):
    if _attention_call_count(token_groups) == 0:
        return

    for token_group in token_groups:
        for copy_wires in token_group:
            controlled_su16_array(
                params,
                len(copy_wires),
                attention_num_blocks,
                repeat_time=repeat_time,
                wires=copy_wires,
                ancilla=shared_attention_ancilla,
            )


def my_model_with_attention(wires, n, m, params, entangling_layers, nonlinear_layers1, nonlinear_layers2, attention_num_blocks=2, attention_repeat_time=1):
    """
    Attention stage with one shared Q/K/V register set.

    ``attention_num_blocks`` defines how many logical attention heads are encoded
    by the leading input qubits inside each shared Q/K/V register. No dedicated
    external control register is reserved.
    """

    params = np.asarray(params)

    attention_layout = _collect_shared_attention_layout(wires, n, m, attention_num_blocks)
    shared_attention_ancilla = attention_layout["shared_attention_ancilla"]

    num_params_attn_q = _shared_attention_param_count(
        attention_layout["q_groups"], attention_num_blocks, attention_repeat_time
    )
    num_params_attn_k = _shared_attention_param_count(
        attention_layout["k_groups"], attention_num_blocks, attention_repeat_time
    )
    num_params_attn_v = _shared_attention_param_count(
        attention_layout["v_groups"], attention_num_blocks, attention_repeat_time
    )

    num_params_entangling = _entangling_param_count(n, m, entangling_layers)
    num_params_shared_nonlin = _shared_nonlinear_param_count(
        n, nonlinear_layers1, nonlinear_layers2
    )

    total_params_needed = (
        num_params_attn_q
        + num_params_attn_k
        + num_params_attn_v
        + num_params_entangling
        + num_params_shared_nonlin
    )
    if len(params) < total_params_needed:
        raise ValueError(
            f"Parameter count is insufficient. Need {total_params_needed}, "
            f"got {len(params)}."
        )

    param_idx = 0

    params_q = params[param_idx : param_idx + num_params_attn_q]
    param_idx += num_params_attn_q
    _apply_attention_stage(
        params_q,
        attention_layout["q_groups"],
        shared_attention_ancilla,
        attention_num_blocks,
        attention_repeat_time,
    )

    params_k = params[param_idx : param_idx + num_params_attn_k]
    param_idx += num_params_attn_k
    _apply_attention_stage(
        params_k,
        attention_layout["k_groups"],
        shared_attention_ancilla,
        attention_num_blocks,
        attention_repeat_time,
    )

    params_v = params[param_idx : param_idx + num_params_attn_v]
    param_idx += num_params_attn_v
    _apply_attention_stage(
        params_v,
        attention_layout["v_groups"],
        shared_attention_ancilla,
        attention_num_blocks,
        attention_repeat_time,
    )

    entangling_params = params[param_idx : param_idx + num_params_entangling]
    param_idx += num_params_entangling
    shared_nonlin_params = params[param_idx : param_idx + num_params_shared_nonlin]
    params_curr_model = np.concatenate([entangling_params, shared_nonlin_params])

    my_model(
        attention_layout["model_wires"],
        n,
        m,
        params_curr_model,
        entangling_layers,
        nonlinear_layers1,
        nonlinear_layers2,
        nonlinear_ancilla_wire=shared_attention_ancilla[0],
    )


def my_model_with_attention_fnn(wires, n, m, params, entangling_layers, nonlinear_layers1, nonlinear_layers2, attention_num_blocks=2, attention_repeat_time=1, ffn_repeat_time=1, count_only=False):
    """
    Full model with shared Q/K/V registers across all attention heads.

    Only the attention stage depends on ``attention_num_blocks``. The downstream
    my_model stack and FFN remain single-register computations.
    """

    if params is not None:
        params = np.asarray(params)

    attention_layout = _collect_shared_attention_layout(wires, n, m, attention_num_blocks)
    shared_attention_ancilla = attention_layout["shared_attention_ancilla"]
    total_needed = attention_layout["total_needed"]
    ffn_token_wires = _primary_v_token_wires(attention_layout["model_wires"], n, m)

    num_params_attn_q = _shared_attention_param_count(
        attention_layout["q_groups"], attention_num_blocks, attention_repeat_time
    )
    num_params_attn_k = _shared_attention_param_count(
        attention_layout["k_groups"], attention_num_blocks, attention_repeat_time
    )
    num_params_attn_v = _shared_attention_param_count(
        attention_layout["v_groups"], attention_num_blocks, attention_repeat_time
    )

    num_params_entangling = _entangling_param_count(n, m, entangling_layers)
    num_params_shared_nonlin = _shared_nonlinear_param_count(
        n, nonlinear_layers1, nonlinear_layers2
    )

    attention_param_count = (
        num_params_attn_q
        + num_params_attn_k
        + num_params_attn_v
        + num_params_entangling
        + num_params_shared_nonlin
    )

    ffn_input_qubits = len(ffn_token_wires[0])
    ffn_dummy_wires = list(range(ffn_input_qubits))
    ffn_param_count = ffn_new(
        [],
        total_qubits=ffn_input_qubits,
        num_blocks=attention_num_blocks,
        repeat_time=ffn_repeat_time,
        wires=ffn_dummy_wires,
        ancilla=[-1],
        count_only=True,
    )

    total_params = attention_param_count + ffn_param_count
    if count_only:
        return total_params

    if params is None or len(params) < total_params:
        raise ValueError(
            f"Parameter count is insufficient. Need {total_params}, "
            f"got {len(params) if params is not None else 0}."
        )

    attention_params = params[:attention_param_count]
    ffn_params = params[attention_param_count : attention_param_count + ffn_param_count]

    my_model_with_attention(
        wires=wires[:total_needed],
        n=n,
        m=m,
        params=attention_params,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
    )

    for token_wires in ffn_token_wires:
        ffn_new(
            params=ffn_params,
            total_qubits=len(token_wires),
            num_blocks=attention_num_blocks,
            repeat_time=ffn_repeat_time,
            wires=token_wires,
            ancilla=shared_attention_ancilla,
            count_only=False,
        )


def my_model_with_attention_fnn_parameter_breakdown(
    wires,
    n,
    m,
    entangling_layers,
    nonlinear_layers1,
    nonlinear_layers2,
    attention_num_blocks=2,
    attention_repeat_time=1,
    ffn_repeat_time=1,
):
    """Return a structured parameter-count breakdown for the full model."""
    attention_layout = _collect_shared_attention_layout(wires, n, m, attention_num_blocks)
    ffn_token_wires = _primary_v_token_wires(attention_layout["model_wires"], n, m)

    q_attention_params = _shared_attention_param_count(
        attention_layout["q_groups"], attention_num_blocks, attention_repeat_time
    )
    k_attention_params = _shared_attention_param_count(
        attention_layout["k_groups"], attention_num_blocks, attention_repeat_time
    )
    v_attention_params = _shared_attention_param_count(
        attention_layout["v_groups"], attention_num_blocks, attention_repeat_time
    )
    entangling_params_shared = _entangling_param_count(n, m, entangling_layers)
    shared_nonlinear_params = _shared_nonlinear_param_count(
        n, nonlinear_layers1, nonlinear_layers2
    )

    ffn_input_qubits = len(ffn_token_wires[0])
    ffn_params = ffn_new(
        [],
        total_qubits=ffn_input_qubits,
        num_blocks=attention_num_blocks,
        repeat_time=ffn_repeat_time,
        wires=list(range(ffn_input_qubits)),
        ancilla=[-1],
        count_only=True,
    )

    q_call_count = _attention_call_count(attention_layout["q_groups"])
    k_call_count = _attention_call_count(attention_layout["k_groups"])
    v_call_count = _attention_call_count(attention_layout["v_groups"])
    attention_total = q_attention_params + k_attention_params + v_attention_params
    entangling_total = entangling_params_shared
    model_total = entangling_total + shared_nonlinear_params
    total_params = attention_total + model_total + ffn_params

    return {
        "attention": {
            "shared_per_q": q_attention_params,
            "shared_per_k": k_attention_params,
            "shared_per_v": v_attention_params,
            "q_call_count": q_call_count,
            "k_call_count": k_call_count,
            "v_call_count": v_call_count,
            "total": attention_total,
        },
        "model": {
            "entangling_per_pair": entangling_layers * (2 * n + 1) * 3,
            "entangling_pair_count": m * (m - 1),
            # Kept for compatibility with existing training/reporting scripts.
            "entangling_per_head": entangling_params_shared,
            "entangling_shared": entangling_params_shared,
            "num_heads": attention_num_blocks,
            "entangling_total": entangling_total,
            "shared_nonlinear": shared_nonlinear_params,
            "nonlinear_call_count": m if shared_nonlinear_params > 0 else 0,
            "total": model_total,
        },
        "ffn": {
            "total": ffn_params,
            "input_qubits": ffn_input_qubits,
            "call_count": m,
        },
        "total": total_params,
    }


def _qkv_register_layout(n, m, attention_num_blocks):
    len_q = m * (m - 1) * n
    len_k = m * (m - 1) * n
    len_v = m * m * n
    swap_ancilla_bits = m * (m - 1)
    wires_per_model = len_q + len_k + len_v + swap_ancilla_bits
    total_model_wires = wires_per_model
    total_needed = total_model_wires + 2
    return len_q, len_k, len_v, wires_per_model, total_model_wires, total_needed


def _validate_shared_token_embeddings(token_indices, embedding_logits, seq_len, token_qubits):
    token_indices = np.asarray(token_indices, dtype=np.int32)
    embedding_logits = np.asarray(embedding_logits)

    if token_indices.ndim != 1 or token_indices.shape[0] != seq_len:
        raise ValueError(f"token_indices must have shape ({seq_len},).")
    if embedding_logits.ndim != 2:
        raise ValueError("embedding_logits must be a rank-2 array.")
    expected_width = embedding_parameter_dimension(token_qubits)
    if embedding_logits.shape[1] != expected_width:
        raise ValueError(
            f"embedding_logits must have {expected_width} columns for "
            f"{token_qubits} token qubits."
        )

    return token_indices, embedding_logits


def _prepare_qkv_embeddings(
    wires,
    n,
    m,
    attention_num_blocks,
    token_indices,
    embedding_logits,
):
    """
    Prepare a shared token embedding sequence, then re-prepare the same source state
    across all Q/K/V copies within the shared register layout.
    """
    attention_layout = _collect_shared_attention_layout(wires, n, m, attention_num_blocks)
    token_indices, embedding_logits = _validate_shared_token_embeddings(
        token_indices, embedding_logits, m, n
    )

    for token_pos in range(m):
        token_index = token_indices[token_pos]
        for copy_wires in attention_layout["q_groups"][token_pos]:
            quantum_embedding(token_index, embedding_logits, copy_wires)
        for copy_wires in attention_layout["k_groups"][token_pos]:
            quantum_embedding(token_index, embedding_logits, copy_wires)
        for copy_wires in attention_layout["v_groups"][token_pos]:
            quantum_embedding(token_index, embedding_logits, copy_wires)


def my_model_with_attention_fnn_with_quantum_embedding(
    wires,
    n,
    m,
    params,
    entangling_layers,
    nonlinear_layers1,
    nonlinear_layers2,
    token_indices=None,
    embedding_logits=None,
    attention_num_blocks=2,
    attention_repeat_time=1,
    ffn_repeat_time=1,
    count_only=False,
):
    """
    Prepare Q/K/V registers with amplitude embeddings, then run the full model.

    Preferred mode:
    - provide one shared ``token_indices`` sequence of shape ``(m,)``
    - provide trainable ``embedding_logits`` with shape ``(vocab_size, 2**n - 1)``
    - the same source token state is re-prepared across all Q/K/V copies
    """
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
    if count_only:
        return total_params

    if params is None or len(params) < total_params:
        raise ValueError(
            f"Parameter count is insufficient. Need {total_params}, "
            f"got {len(params) if params is not None else 0}."
        )
    if token_indices is None or embedding_logits is None:
        raise ValueError("token_indices and embedding_logits must both be provided.")

    _, _, _, _, _, total_needed = _qkv_register_layout(n, m, attention_num_blocks)
    if len(wires) < total_needed:
        raise ValueError(f"Need at least {total_needed} wires, got {len(wires)}.")

    _prepare_qkv_embeddings(
        wires,
        n,
        m,
        attention_num_blocks,
        token_indices,
        embedding_logits,
    )

    my_model_with_attention_fnn(
        wires=wires,
        n=n,
        m=m,
        params=params,
        entangling_layers=entangling_layers,
        nonlinear_layers1=nonlinear_layers1,
        nonlinear_layers2=nonlinear_layers2,
        attention_num_blocks=attention_num_blocks,
        attention_repeat_time=attention_repeat_time,
        ffn_repeat_time=ffn_repeat_time,
        count_only=False,
    )


    



