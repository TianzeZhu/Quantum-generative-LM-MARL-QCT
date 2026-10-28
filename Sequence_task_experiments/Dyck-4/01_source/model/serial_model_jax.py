"""JAX-compiled serialized quantum Seq2Seq model for SCAN-length."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .statevector_jax import (
        LOGICAL_PAIR_QUBITS,
        PAIR_ACTIVE_QUBITS,
        TOKEN_DIM,
        aggregate_swap_zero_rows,
        apply_svd,
        embedding_table,
        nonlinear_parameter_count,
        output_probabilities,
        pair_swap_zero_row,
        prepare_positioned_states,
        svd_parameter_count,
    )
except ImportError:
    from statevector_jax import (
        LOGICAL_PAIR_QUBITS,
        PAIR_ACTIVE_QUBITS,
        TOKEN_DIM,
        aggregate_swap_zero_rows,
        apply_svd,
        embedding_table,
        nonlinear_parameter_count,
        output_probabilities,
        pair_swap_zero_row,
        prepare_positioned_states,
        svd_parameter_count,
    )


SCAN_ACTIONS = (
    "I_JUMP",
    "I_LOOK",
    "I_RUN",
    "I_TURN_LEFT",
    "I_TURN_RIGHT",
    "I_WALK",
)
SCAN_EOS_BASIS = 6
SCAN_INVALID_BASIS = 7


@dataclass(frozen=True)
class JaxSerialConfig:
    """Static topology values captured by JIT-compiled model functions."""

    n: int = 3
    num_heads: int = 1
    attention_layers: int = 1
    entangling_layers: int = 1
    softmax_layers1: int = 1
    softmax_layers2: int = 1
    ffn_layers: int = 1
    sigmoid_layers1: int = 1
    sigmoid_layers2: int = 1
    bos_id: int = 1
    epsilon: float = 1.0e-12

    def validate(self):
        if self.n != 3:
            raise ValueError("The direct SCAN readout requires exactly n=3.")
        if self.num_heads != 1:
            raise ValueError("The accelerated SCAN v1 supports one attention head.")
        if self.attention_layers <= 0 or self.ffn_layers <= 0:
            raise ValueError("SVD unitary layer counts must be positive.")
        if self.entangling_layers <= 0:
            raise ValueError("At least one seven-wire entangling layer is required.")
        if self.softmax_layers1 + self.softmax_layers2 <= 0:
            raise ValueError("Softmax circuit must contain a unitary segment.")
        if self.sigmoid_layers1 + self.sigmoid_layers2 <= 0:
            raise ValueError("Sigmoid circuit must contain a unitary segment.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")

    @property
    def logical_pair_qubits(self):
        return LOGICAL_PAIR_QUBITS

    @property
    def active_simulator_qubits(self):
        return PAIR_ACTIVE_QUBITS


def parameter_shapes(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: JaxSerialConfig = JaxSerialConfig(),
):
    config.validate()
    if src_vocab_size <= 0 or tgt_vocab_size <= 0:
        raise ValueError("Vocabulary sizes must be positive.")
    svd_count = svd_parameter_count(config.n, config.attention_layers)
    ffn_svd_count = svd_parameter_count(config.n, config.ffn_layers)
    return {
        "src_embedding_logits": (src_vocab_size, TOKEN_DIM - 1),
        "tgt_embedding_logits": (tgt_vocab_size, TOKEN_DIM - 1),
        "q": (svd_count,),
        "k": (svd_count,),
        "v": (svd_count,),
        "entangling": (
            config.entangling_layers,
            PAIR_ACTIVE_QUBITS,
            3,
        ),
        "position": (config.n,),
        "ffn_gw1": (ffn_svd_count,),
        "ffn_gw2": (ffn_svd_count,),
        "aggregation_log_temperature": (1,),
    }


def frozen_parameter_shapes(config: JaxSerialConfig = JaxSerialConfig()):
    config.validate()
    return {
        "softmax": (
            nonlinear_parameter_count(
                config.n, config.softmax_layers1, config.softmax_layers2
            ),
        ),
        "sigmoid": (
            nonlinear_parameter_count(
                config.n, config.sigmoid_layers1, config.sigmoid_layers2
            ),
        ),
    }


def initialize_parameters(
    key,
    src_vocab_size: int,
    tgt_vocab_size: int,
    config: JaxSerialConfig = JaxSerialConfig(),
):
    """Initialize the trainable PyTree; nonlinear parameters are separate/frozen."""
    shapes = parameter_shapes(src_vocab_size, tgt_vocab_size, config)
    names = tuple(shapes)
    keys = jax.random.split(key, len(names))
    result = {}
    for name, subkey in zip(names, keys):
        shape = shapes[name]
        if name == "aggregation_log_temperature":
            value = jnp.zeros(shape)
        else:
            scale = 0.25 if name.endswith("embedding_logits") else 0.08
            value = scale * jax.random.normal(subkey, shape)
        result[name] = value
    return result


def validate_frozen_parameters(frozen, config: JaxSerialConfig):
    expected = frozen_parameter_shapes(config)
    if set(frozen) != set(expected):
        raise ValueError(f"Frozen parameter keys must be {sorted(expected)}.")
    for name, shape in expected.items():
        if tuple(np.shape(frozen[name])) != shape:
            raise ValueError(
                f"Frozen {name} must have shape {shape}, got {np.shape(frozen[name])}."
            )


def count_parameters(parameters: Mapping[str, jax.Array]):
    return sum(int(np.prod(np.shape(value))) for value in parameters.values())


def _transform_many(states, params, layers, epsilon):
    transform = lambda state: apply_svd(params, state, layers, epsilon)[0]
    return jax.vmap(transform)(states)


def sequence_probabilities(
    parameters,
    frozen,
    src_ids,
    src_mask,
    target_ids,
    target_mask,
    config: JaxSerialConfig = JaxSerialConfig(),
):
    """Return every teacher-forced target distribution in serialized order.

    target_ids contains action-token IDs only and is padded to T.  target_mask
    marks those actions.  Output position len(actions) predicts EOS; callers
    select valid output positions with their label mask.
    """
    source_table = embedding_table(parameters["src_embedding_logits"])
    target_table = embedding_table(parameters["tgt_embedding_logits"])
    source_length = jnp.sum(src_mask.astype(jnp.int32))
    source_positions = jnp.arange(src_ids.shape[0], dtype=jnp.int32)
    target_positions = jnp.arange(target_ids.shape[0], dtype=jnp.int32)

    source_states = prepare_positioned_states(
        source_table[src_ids], source_positions, parameters["position"]
    )
    target_context_states = prepare_positioned_states(
        target_table[target_ids],
        source_length + target_positions,
        parameters["position"],
    )

    query_ids = jnp.concatenate(
        (
            jnp.asarray([config.bos_id], dtype=target_ids.dtype),
            target_ids[:-1],
        )
    )
    query_states = prepare_positioned_states(
        target_table[query_ids], target_positions, parameters["position"]
    )

    query_states = _transform_many(
        query_states, parameters["q"], config.attention_layers, config.epsilon
    )
    accumulator_states = _transform_many(
        prepare_positioned_states(
            target_table[query_ids], target_positions, parameters["position"]
        ),
        parameters["v"],
        config.attention_layers,
        config.epsilon,
    )
    source_keys = _transform_many(
        source_states, parameters["k"], config.attention_layers, config.epsilon
    )
    source_values = _transform_many(
        source_states, parameters["v"], config.attention_layers, config.epsilon
    )
    target_keys = _transform_many(
        target_context_states,
        parameters["k"],
        config.attention_layers,
        config.epsilon,
    )
    target_values = _transform_many(
        target_context_states,
        parameters["v"],
        config.attention_layers,
        config.epsilon,
    )
    all_keys = jnp.concatenate((source_keys, target_keys), axis=0)
    all_values = jnp.concatenate((source_values, target_values), axis=0)

    def token_output(target_position):
        prefix_mask = target_mask & (target_positions < target_position)
        context_mask = jnp.concatenate((src_mask, prefix_mask), axis=0)

        def one_pair(key_state, value_state):
            return pair_swap_zero_row(
                query_states[target_position],
                accumulator_states[target_position],
                key_state,
                value_state,
                parameters["entangling"],
            )

        rows = jax.vmap(one_pair)(all_keys, all_values)
        aggregate, _, _ = aggregate_swap_zero_rows(
            rows,
            context_mask,
            parameters["aggregation_log_temperature"],
            config.epsilon,
        )
        return output_probabilities(
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

    # lax.map is scan-backed: target tokens remain one-token-at-a-time even
    # though independent context pairs inside each token are vectorized.
    return jax.lax.map(token_output, target_positions)


def sequence_loss_and_metrics(
    parameters,
    frozen,
    src_ids,
    src_mask,
    target_ids,
    target_mask,
    labels,
    label_mask,
    config: JaxSerialConfig = JaxSerialConfig(),
):
    probabilities = sequence_probabilities(
        parameters,
        frozen,
        src_ids,
        src_mask,
        target_ids,
        target_mask,
        config,
    )
    selected = jnp.take_along_axis(
        probabilities, labels[:, None], axis=-1
    )[:, 0]
    token_nll = -jnp.log(jnp.maximum(selected, config.epsilon))
    weights = label_mask.astype(token_nll.dtype)
    count = jnp.sum(weights)
    sequence_nll = jnp.sum(token_nll * weights)
    predictions = jnp.argmax(probabilities, axis=-1)
    correct = jnp.sum((predictions == labels) * label_mask)
    return sequence_nll / jnp.maximum(count, 1), {
        "sequence_nll": sequence_nll,
        "target_tokens": count,
        "correct_tokens": correct,
        "probability_error": jnp.max(
            jnp.abs(jnp.sum(probabilities, axis=-1) - 1.0)
        ),
    }


def batch_loss_and_metrics(
    parameters,
    frozen,
    batch,
    config: JaxSerialConfig = JaxSerialConfig(),
):
    def one_example(src_ids, src_mask, target_ids, target_mask, labels, label_mask):
        return sequence_loss_and_metrics(
            parameters,
            frozen,
            src_ids,
            src_mask,
            target_ids,
            target_mask,
            labels,
            label_mask,
            config,
        )

    losses, metrics = jax.vmap(one_example)(
        batch["src_ids"],
        batch["src_mask"],
        batch["target_ids"],
        batch["target_mask"],
        batch["labels"],
        batch["label_mask"],
    )
    example_weights = batch["example_mask"].astype(losses.dtype)
    batch_size = jnp.maximum(jnp.sum(example_weights), 1)
    loss = jnp.sum(losses * example_weights) / batch_size
    target_tokens = jnp.sum(
        metrics["target_tokens"] * example_weights
    )
    correct_tokens = jnp.sum(
        metrics["correct_tokens"] * example_weights
    )
    return loss, {
        "token_nll": jnp.sum(
            metrics["sequence_nll"] * example_weights
        ) / jnp.maximum(target_tokens, 1),
        "token_accuracy": correct_tokens / jnp.maximum(target_tokens, 1),
        "target_tokens": target_tokens,
        "correct_tokens": correct_tokens,
        "max_probability_error": jnp.max(
            metrics["probability_error"] * example_weights
        ),
    }


def project_parameters(parameters):
    """Apply the same bounded domains as the earlier analytic trainer."""
    result = {}
    for name, value in parameters.items():
        if name.endswith("embedding_logits") or name == "aggregation_log_temperature":
            result[name] = jnp.clip(value, -5.0, 5.0)
        else:
            result[name] = (value + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return result


__all__ = [
    "SCAN_ACTIONS",
    "SCAN_EOS_BASIS",
    "SCAN_INVALID_BASIS",
    "JaxSerialConfig",
    "parameter_shapes",
    "frozen_parameter_shapes",
    "initialize_parameters",
    "validate_frozen_parameters",
    "count_parameters",
    "sequence_probabilities",
    "sequence_loss_and_metrics",
    "batch_loss_and_metrics",
    "project_parameters",
]


