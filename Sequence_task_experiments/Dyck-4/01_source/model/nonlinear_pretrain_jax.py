"""JAX pretraining for the frozen paper Softmax and Sigmoid circuits."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .optimizer_jax import AdamConfig, adam_init, adam_update
    from .statevector_jax import apply_nonlinear, nonlinear_parameter_count
except ImportError:
    from optimizer_jax import AdamConfig, adam_init, adam_update
    from statevector_jax import apply_nonlinear, nonlinear_parameter_count


@dataclass(frozen=True)
class NonlinearPretrainConfig:
    n: int = 3
    layers1: int = 1
    layers2: int = 1
    steps: int = 1600
    batch_size: int = 16
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1.0e-8
    softmax_beta: float = 1.0
    seed: int = 2026
    validation_size: int = 256

    def validate(self):
        if self.n != 3:
            raise ValueError("Current nonlinear pretraining requires n=3.")
        if self.layers1 < 0 or self.layers2 < 0:
            raise ValueError("Layer counts cannot be negative.")
        if self.layers1 + self.layers2 <= 0:
            raise ValueError("At least one nonlinear unitary segment is required.")
        if self.steps <= 0 or self.batch_size <= 0 or self.validation_size <= 0:
            raise ValueError("steps, batch_size and validation_size must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.softmax_beta) or self.softmax_beta <= 0:
            raise ValueError("softmax_beta must be finite and positive.")


def random_nonnegative_states(key, count, n=3):
    values = jax.random.uniform(key, (count, 2**n))
    return values / jnp.linalg.norm(values, axis=-1, keepdims=True)


def target_states(inputs, kind, softmax_beta=1.0):
    if kind == "softmax":
        values = jax.nn.softmax(softmax_beta * inputs, axis=-1)
    elif kind == "sigmoid":
        values = jax.nn.sigmoid(inputs)
    else:
        raise ValueError(f"Unknown nonlinearity {kind!r}.")
    return values / jnp.linalg.norm(values, axis=-1, keepdims=True)


def fidelity_loss(params, inputs, targets, layers1=1, layers2=1):
    outputs = jax.vmap(
        lambda state: apply_nonlinear(
            params, state, layers1=layers1, layers2=layers2
        )[0]
    )(inputs)
    overlaps = jnp.sum(jnp.conj(targets) * outputs, axis=-1)
    fidelities = jnp.abs(overlaps) ** 2
    return 1.0 - jnp.mean(fidelities)


def _make_step(config: NonlinearPretrainConfig):
    adam_config = AdamConfig(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        clip_norm=1.0e6,
    )

    def step(parameters, optimizer_state, inputs, targets):
        wrapped = {"nonlinear": parameters}
        objective = lambda values: fidelity_loss(
            values["nonlinear"],
            inputs,
            targets,
            config.layers1,
            config.layers2,
        )
        loss, gradients = jax.value_and_grad(objective)(wrapped)
        wrapped, optimizer_state, metrics = adam_update(
            wrapped, gradients, optimizer_state, adam_config
        )
        return wrapped["nonlinear"], optimizer_state, loss, metrics

    return jax.jit(step)


def train_one(kind, config: NonlinearPretrainConfig, progress=None):
    config.validate()
    key = jax.random.PRNGKey(config.seed + (0 if kind == "softmax" else 1))
    parameter_key, key = jax.random.split(key)
    count = nonlinear_parameter_count(config.n, config.layers1, config.layers2)
    parameters = 0.08 * jax.random.normal(parameter_key, (count,))
    optimizer_state = adam_init({"nonlinear": parameters})
    step_function = _make_step(config)
    history = []
    compile_seconds = None

    for step_index in range(1, config.steps + 1):
        key, sample_key = jax.random.split(key)
        inputs = random_nonnegative_states(sample_key, config.batch_size, config.n)
        targets = target_states(inputs, kind, config.softmax_beta)
        started = time.perf_counter()
        parameters, optimizer_state, loss, metrics = step_function(
            parameters, optimizer_state, inputs, targets
        )
        jax.block_until_ready(loss)
        elapsed = time.perf_counter() - started
        if compile_seconds is None:
            compile_seconds = elapsed
        scalar_loss = float(loss)
        if not math.isfinite(scalar_loss):
            raise FloatingPointError(f"{kind} pretraining produced non-finite loss.")
        if not bool(metrics["gradients_finite"]) or not bool(
            metrics["parameters_finite"]
        ):
            raise FloatingPointError(f"{kind} pretraining produced non-finite values.")
        history.append(scalar_loss)
        if progress is not None:
            progress(kind, step_index, config.steps, scalar_loss, elapsed)

    key, validation_key = jax.random.split(key)
    validation_inputs = random_nonnegative_states(
        validation_key, config.validation_size, config.n
    )
    validation_targets = target_states(
        validation_inputs, kind, config.softmax_beta
    )
    validation_loss = jax.jit(fidelity_loss, static_argnums=(3, 4))(
        parameters,
        validation_inputs,
        validation_targets,
        config.layers1,
        config.layers2,
    )
    jax.block_until_ready(validation_loss)
    return {
        "parameters": np.asarray(parameters),
        "history": np.asarray(history, dtype=np.float64),
        "validation_loss": float(validation_loss),
        "compile_seconds": float(compile_seconds),
    }


def pretrain_both(config: NonlinearPretrainConfig, progress=None):
    softmax = train_one("softmax", config, progress)
    sigmoid = train_one("sigmoid", config, progress)
    return {"softmax": softmax, "sigmoid": sigmoid}


__all__ = [
    "NonlinearPretrainConfig",
    "random_nonnegative_states",
    "target_states",
    "fidelity_loss",
    "train_one",
    "pretrain_both",
]
