"""Greedy autoregressive decoding with a serialized JAX scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    from .serial_model_jax import SCAN_EOS_BASIS
    from .statevector_jax import (
        TOKEN_DIM,
        aggregate_swap_zero_rows,
        apply_position_encoding,
        apply_svd,
        embedding_table,
        output_probabilities,
        pair_swap_zero_row,
        prepare_positioned_states,
    )
except ImportError:
    from serial_model_jax import SCAN_EOS_BASIS
    from statevector_jax import (
        TOKEN_DIM,
        aggregate_swap_zero_rows,
        apply_position_encoding,
        apply_svd,
        embedding_table,
        output_probabilities,
        pair_swap_zero_row,
        prepare_positioned_states,
    )


def greedy_decode(
    parameters,
    frozen,
    src_ids,
    src_mask,
    basis_to_target_ids,
    max_steps,
    config,
):
    """Generate token bases until EOS/INVALID or max_steps.

    Valid token bases are contiguous from zero; EOS remains 6. All other bases
    are INVALID (7 for SCAN; 4, 5, 7 for merged Dyck). Do not renormalize away
    INVALID probability mass during either training or inference.

    The source transforms are prepared once.  Each scan iteration emits one
    token, then prepares only that generated token's K/V state for the next
    iteration.  Independent context pairs inside an iteration are vectorized.
    """
    source_table = embedding_table(parameters["src_embedding_logits"])
    target_table = embedding_table(parameters["tgt_embedding_logits"])
    source_positions = jnp.arange(src_ids.shape[0], dtype=jnp.int32)
    source_length = jnp.sum(src_mask.astype(jnp.int32))
    source_states = prepare_positioned_states(
        source_table[src_ids], source_positions, parameters["position"]
    )
    transform_k = lambda state: apply_svd(
        parameters["k"], state, config.attention_layers, config.epsilon
    )[0]
    transform_v = lambda state: apply_svd(
        parameters["v"], state, config.attention_layers, config.epsilon
    )[0]
    source_keys = jax.vmap(transform_k)(source_states)
    source_values = jax.vmap(transform_v)(source_states)
    generated_ids = jnp.zeros((max_steps,), dtype=src_ids.dtype)
    generated_mask = jnp.zeros((max_steps,), dtype=jnp.bool_)
    generated_keys = jnp.zeros(
        (max_steps, TOKEN_DIM), dtype=source_keys.dtype
    )
    generated_values = jnp.zeros(
        (max_steps, TOKEN_DIM), dtype=source_values.dtype
    )
    positions = jnp.arange(max_steps, dtype=jnp.int32)

    def scan_step(carry, target_position):
        ids, mask, keys, values, finished = carry
        query_id = jnp.where(
            target_position == 0, config.bos_id, ids[target_position - 1]
        )
        query_state = apply_position_encoding(
            target_table[query_id], target_position, parameters["position"]
        )
        query_state = apply_svd(
            parameters["q"],
            query_state,
            config.attention_layers,
            config.epsilon,
        )[0]
        accumulator = apply_position_encoding(
            target_table[query_id], target_position, parameters["position"]
        )
        accumulator = apply_svd(
            parameters["v"],
            accumulator,
            config.attention_layers,
            config.epsilon,
        )[0]
        all_keys = jnp.concatenate((source_keys, keys), axis=0)
        all_values = jnp.concatenate((source_values, values), axis=0)
        context_mask = jnp.concatenate(
            (src_mask, mask & (positions < target_position)), axis=0
        )

        rows = jax.vmap(
            lambda key, value: pair_swap_zero_row(
                query_state,
                accumulator,
                key,
                value,
                parameters["entangling"],
            )
        )(all_keys, all_values)
        aggregate, _, _ = aggregate_swap_zero_rows(
            rows,
            context_mask,
            parameters["aggregation_log_temperature"],
            config.epsilon,
        )
        probabilities = output_probabilities(
            aggregate,
            frozen["softmax"],
            parameters["ffn_gw1"],
            frozen["sigmoid"],
            parameters["ffn_gw2"],
            unitary_layers=config.ffn_layers,
            softmax_layers1=config.softmax_layers1,
            softmax_layers2=config.softmax_layers2,
            sigmoid_layers1=config.sigmoid_layers1,
            sigmoid_layers2=config.sigmoid_layers2,
            epsilon=config.epsilon,
        )
        basis = jnp.argmax(probabilities).astype(jnp.int32)
        active = ~finished
        is_action = basis < len(basis_to_target_ids)
        target_id = jnp.where(
            is_action, basis_to_target_ids[jnp.minimum(basis, len(basis_to_target_ids) - 1)], 0
        )
        ids = ids.at[target_position].set(jnp.where(active, target_id, 0))
        store_action = active & is_action
        mask = mask.at[target_position].set(store_action)

        context_state = apply_position_encoding(
            target_table[target_id],
            source_length + target_position,
            parameters["position"],
        )
        new_key = transform_k(context_state)
        new_value = transform_v(context_state)
        keys = keys.at[target_position].set(
            jnp.where(store_action, new_key, keys[target_position])
        )
        values = values.at[target_position].set(
            jnp.where(store_action, new_value, values[target_position])
        )
        terminal = ~is_action
        next_finished = finished | (active & terminal)
        output = {
            "basis": jnp.where(active, basis, -1),
            "active": active,
            "probabilities": probabilities,
        }
        return (ids, mask, keys, values, next_finished), output

    initial = (
        generated_ids,
        generated_mask,
        generated_keys,
        generated_values,
        jnp.asarray(False),
    )
    final, outputs = jax.lax.scan(scan_step, initial, positions)
    emitted = outputs["basis"]
    active = outputs["active"]
    eos_seen = jnp.any((emitted == SCAN_EOS_BASIS) & active)
    invalid_seen = jnp.any(
        (emitted >= len(basis_to_target_ids)) & (emitted != SCAN_EOS_BASIS) & active
    )
    return {
        "basis_ids": emitted,
        "active_mask": active,
        "probabilities": outputs["probabilities"],
        "terminated": eos_seen,
        "invalid": invalid_seen,
        "hit_limit": (~eos_seen) & (~invalid_seen),
        "target_ids": final[0],
    }


def make_greedy_decoder(config, max_steps):
    config.validate()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    return jax.jit(
        lambda parameters, frozen, src_ids, src_mask, basis_to_target_ids: greedy_decode(
            parameters,
            frozen,
            src_ids,
            src_mask,
            basis_to_target_ids,
            max_steps,
            config,
        )
    )


__all__ = ["greedy_decode", "make_greedy_decoder"]
