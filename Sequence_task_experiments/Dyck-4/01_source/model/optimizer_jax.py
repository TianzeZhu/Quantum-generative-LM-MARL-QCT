"""Small dependency-free JAX Adam implementation for checkpointable training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

try:
    from .serial_model_jax import batch_loss_and_metrics, project_parameters
except ImportError:
    from serial_model_jax import batch_loss_and_metrics, project_parameters


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1.0e-8
    clip_norm: float = 5.0

    def validate(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1).")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Adam epsilon must be finite and positive.")
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0:
            raise ValueError("clip_norm must be finite and positive.")


def adam_init(parameters):
    return {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "m": jax.tree.map(jnp.zeros_like, parameters),
        "v": jax.tree.map(jnp.zeros_like, parameters),
    }


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(
        sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves)
    )


def tree_max_abs(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))


def tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def adam_update(parameters, gradients, state, config: AdamConfig):
    config.validate()
    gradient_norm = tree_l2_norm(gradients)
    clip_scale = jnp.minimum(
        1.0, config.clip_norm / jnp.maximum(gradient_norm, 1.0e-30)
    )
    clipped = jax.tree.map(lambda gradient: gradient * clip_scale, gradients)
    step = state["step"] + 1
    moment1 = jax.tree.map(
        lambda old, gradient: config.beta1 * old + (1 - config.beta1) * gradient,
        state["m"],
        clipped,
    )
    moment2 = jax.tree.map(
        lambda old, gradient: config.beta2 * old + (1 - config.beta2) * gradient**2,
        state["v"],
        clipped,
    )
    correction1 = 1 - config.beta1**step
    correction2 = 1 - config.beta2**step
    updates = jax.tree.map(
        lambda first, second: config.learning_rate
        * (first / correction1)
        / (jnp.sqrt(second / correction2) + config.epsilon),
        moment1,
        moment2,
    )
    updated = jax.tree.map(
        lambda parameter, update: parameter - update, parameters, updates
    )
    updated = project_parameters(updated)
    return updated, {"step": step, "m": moment1, "v": moment2}, {
        "gradient_l2": gradient_norm,
        "clipped_gradient_l2": tree_l2_norm(clipped),
        "gradient_max_abs": tree_max_abs(gradients),
        "clip_scale": clip_scale,
        "parameter_change_l2": tree_l2_norm(updates),
        "gradients_finite": tree_all_finite(gradients),
        "parameters_finite": tree_all_finite(updated),
    }


def make_train_step(model_config, adam_config: AdamConfig):
    """Return one JIT-compatible full-sequence loss and Adam update."""
    model_config.validate()
    adam_config.validate()

    def train_step(parameters, optimizer_state, frozen, batch):
        objective = lambda values: batch_loss_and_metrics(
            values, frozen, batch, model_config
        )
        (loss, metrics), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(parameters)
        updated, optimizer_state, optimizer_metrics = adam_update(
            parameters, gradients, optimizer_state, adam_config
        )
        return updated, optimizer_state, loss, metrics, optimizer_metrics

    return jax.jit(train_step)


def make_eval_step(model_config):
    model_config.validate()
    return jax.jit(
        lambda parameters, frozen, batch: batch_loss_and_metrics(
            parameters, frozen, batch, model_config
        )
    )


__all__ = [
    "AdamConfig",
    "adam_init",
    "adam_update",
    "tree_l2_norm",
    "tree_max_abs",
    "tree_all_finite",
    "make_train_step",
    "make_eval_step",
]
