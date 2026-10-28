#!/usr/bin/env python3
"""Train the serialized SCAN/Dyck quantum Seq2Seq model with JAX/XLA."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import pennylane as qml

try:
    from .data_jax import (
        dataset_sha256,
        encode_examples,
        iter_batches,
        load_split,
        load_vocabularies,
        make_batch,
        required_shapes,
        readout_metadata,
        split_statistics,
    )
    from .decoding_jax import make_greedy_decoder
    from .io_jax import (
        load_checkpoint,
        load_nonlinear_artifact,
        paired_paths,
        save_checkpoint,
        save_nonlinear_artifact,
    )
    from .nonlinear_pretrain_jax import (
        NonlinearPretrainConfig,
        pretrain_both,
    )
    from .optimizer_jax import (
        AdamConfig,
        adam_init,
        make_train_step,
    )
    from .serial_model_jax import (
        JaxSerialConfig,
        SCAN_ACTIONS,
        batch_loss_and_metrics,
        count_parameters,
        frozen_parameter_shapes,
        initialize_parameters,
        parameter_shapes,
        sequence_probabilities,
        validate_frozen_parameters,
    )
    from .statevector_jax import pair_swap_zero_row
except ImportError:
    from data_jax import (
        dataset_sha256,
        encode_examples,
        iter_batches,
        load_split,
        load_vocabularies,
        make_batch,
        required_shapes,
        readout_metadata,
        split_statistics,
    )
    from decoding_jax import make_greedy_decoder
    from io_jax import (
        load_checkpoint,
        load_nonlinear_artifact,
        paired_paths,
        save_checkpoint,
        save_nonlinear_artifact,
    )
    from nonlinear_pretrain_jax import (
        NonlinearPretrainConfig,
        pretrain_both,
    )
    from optimizer_jax import AdamConfig, adam_init, make_train_step
    from serial_model_jax import (
        JaxSerialConfig,
        SCAN_ACTIONS,
        batch_loss_and_metrics,
        count_parameters,
        frozen_parameter_shapes,
        initialize_parameters,
        parameter_shapes,
        sequence_probabilities,
        validate_frozen_parameters,
    )
    from statevector_jax import pair_swap_zero_row


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = MODULE_DIR.parent / "new_datasets"
STOP_REQUESTED = False


def _request_stop(signum, frame):
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


class JsonlLogger:
    def __init__(self, output_dir: Path | None = None):
        self.path = None
        self.handle = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.path = output_dir / "training.jsonl"
            self.handle = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: str, **fields):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(
            record, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()
        return record

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _model_config(args):
    config = JaxSerialConfig(
        n=args.token_qubits,
        num_heads=args.num_heads,
        attention_layers=args.attention_layers,
        entangling_layers=args.entangling_layers,
        softmax_layers1=args.softmax_layers1,
        softmax_layers2=args.softmax_layers2,
        ffn_layers=args.ffn_layers,
        sigmoid_layers1=args.sigmoid_layers1,
        sigmoid_layers2=args.sigmoid_layers2,
        epsilon=args.model_epsilon,
    )
    config.validate()
    if (
        config.softmax_layers1,
        config.softmax_layers2,
    ) != (
        config.sigmoid_layers1,
        config.sigmoid_layers2,
    ):
        raise ValueError(
            "One shared artifact currently requires identical Softmax and "
            "Sigmoid topology depths."
        )
    return config


def _adam_config(args):
    config = AdamConfig(
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        clip_norm=args.grad_clip_norm,
    )
    config.validate()
    return config


def _artifact_prefix(path: Path):
    path = path.expanduser().resolve()
    if path.is_dir() and (path / "paper.npz").is_file():
        return path / "paper"
    if not path.suffix and (path / "paper.npz").is_file():
        return path / "paper"
    return path


def _checkpoint_prefix(path: Path):
    path = path.expanduser().resolve()
    if path.is_dir():
        return path / "latest"
    return path


def _load_frozen(path, config):
    prefix = _artifact_prefix(Path(path))
    artifact = load_nonlinear_artifact(
        prefix,
        expected_n=config.n,
        expected_layers1=config.softmax_layers1,
        expected_layers2=config.softmax_layers2,
    )
    frozen = {
        name: jnp.asarray(value)
        for name, value in artifact.frozen_parameters.items()
    }
    validate_frozen_parameters(frozen, config)
    return artifact, frozen, prefix


def _random_frozen(config, seed):
    shapes = frozen_parameter_shapes(config)
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "softmax": 0.08 * jax.random.normal(key1, shapes["softmax"]),
        "sigmoid": 0.08 * jax.random.normal(key2, shapes["sigmoid"]),
    }


def _device_batch(batch):
    return {
        name: jnp.asarray(value)
        for name, value in batch.items()
        if name != "uids"
    }


def _device_record():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "pennylane": qml.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }


def _require_gpu(enabled):
    if enabled and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"GPU was required but JAX selected {jax.default_backend()!r}: "
            f"{jax.devices()}."
        )


def _load_encoded(data_root, split, source_vocab, target_vocab, maximum=None):
    examples = load_split(data_root, split)
    if maximum is not None:
        examples = examples[:maximum]
    return encode_examples(examples, source_vocab, target_vocab)


def _make_eval_step(config):
    def evaluate(parameters, frozen, batch):
        probabilities = jax.vmap(
            lambda src_ids, src_mask, target_ids, target_mask: sequence_probabilities(
                parameters,
                frozen,
                src_ids,
                src_mask,
                target_ids,
                target_mask,
                config,
            )
        )(
            batch["src_ids"],
            batch["src_mask"],
            batch["target_ids"],
            batch["target_mask"],
        )
        selected = jnp.take_along_axis(
            probabilities, batch["labels"][..., None], axis=-1
        )[..., 0]
        token_nll = -jnp.log(jnp.maximum(selected, config.epsilon))
        label_mask = batch["label_mask"]
        example_mask = batch["example_mask"]
        predictions = jnp.argmax(probabilities, axis=-1)
        token_correct = (predictions == batch["labels"]) & label_mask
        sequence_correct = jnp.all(
            jnp.where(label_mask, predictions == batch["labels"], True), axis=-1
        )
        target_tokens = jnp.sum(label_mask & example_mask[:, None])
        real_examples = jnp.sum(example_mask)
        per_sequence_count = jnp.maximum(jnp.sum(label_mask, axis=-1), 1)
        per_sequence_nll = (
            jnp.sum(token_nll * label_mask, axis=-1) / per_sequence_count
        )
        return {
            "sequence_mean_nll_sum": jnp.sum(
                per_sequence_nll * example_mask
            ),
            "token_nll_sum": jnp.sum(
                token_nll * label_mask * example_mask[:, None]
            ),
            "target_tokens": target_tokens,
            "correct_tokens": jnp.sum(
                token_correct * example_mask[:, None]
            ),
            "correct_sequences": jnp.sum(sequence_correct * example_mask),
            "examples": real_examples,
            "max_probability_error": jnp.max(
                jnp.abs(jnp.sum(probabilities, axis=-1) - 1.0)
                * example_mask[:, None]
            ),
        }

    return jax.jit(evaluate)


def _evaluate(
    parameters,
    frozen,
    examples,
    batch_size,
    max_source,
    max_target,
    config,
):
    evaluate_step = _make_eval_step(config)
    totals = {
        "sequence_mean_nll_sum": 0.0,
        "token_nll_sum": 0.0,
        "target_tokens": 0,
        "correct_tokens": 0,
        "correct_sequences": 0,
        "examples": 0,
        "max_probability_error": 0.0,
    }
    compile_seconds = None
    order = np.arange(len(examples))
    for _, batch in iter_batches(
        examples, order, batch_size, max_source, max_target
    ):
        started = time.perf_counter()
        values = evaluate_step(parameters, frozen, _device_batch(batch))
        jax.block_until_ready(values["token_nll_sum"])
        elapsed = time.perf_counter() - started
        if compile_seconds is None:
            compile_seconds = elapsed
        host = {name: np.asarray(value).item() for name, value in values.items()}
        for name in totals:
            if name == "max_probability_error":
                totals[name] = max(totals[name], float(host[name]))
            else:
                totals[name] += host[name]
    return {
        "mean_sequence_token_nll": totals["sequence_mean_nll_sum"]
        / max(totals["examples"], 1),
        "token_nll": totals["token_nll_sum"] / max(totals["target_tokens"], 1),
        "token_accuracy": totals["correct_tokens"]
        / max(totals["target_tokens"], 1),
        "teacher_forced_sequence_accuracy": totals["correct_sequences"]
        / max(totals["examples"], 1),
        "examples": int(totals["examples"]),
        "target_tokens": int(totals["target_tokens"]),
        "max_probability_error": totals["max_probability_error"],
        "compile_or_first_batch_seconds": compile_seconds,
    }


def _greedy_metrics(
    parameters,
    frozen,
    examples,
    source_vocab,
    target_vocab,
    max_source,
    max_steps,
    config,
):
    decoder = make_greedy_decoder(config, max_steps)
    action_ids = jnp.asarray(
        [target_vocab.token_to_id[action] for action in target_vocab.tokens[4:]],
        dtype=jnp.int32,
    )
    exact = 0
    terminated = 0
    invalid = 0
    compile_seconds = None
    for example in examples:
        src = np.zeros((max_source,), dtype=np.int32)
        mask = np.zeros((max_source,), dtype=np.bool_)
        src[: len(example.src_ids)] = example.src_ids
        mask[: len(example.src_ids)] = True
        started = time.perf_counter()
        result = decoder(
            parameters,
            frozen,
            jnp.asarray(src),
            jnp.asarray(mask),
            action_ids,
        )
        jax.block_until_ready(result["basis_ids"])
        elapsed = time.perf_counter() - started
        if compile_seconds is None:
            compile_seconds = elapsed
        bases = np.asarray(result["basis_ids"])
        active = np.asarray(result["active_mask"])
        emitted = [int(value) for value in bases[active]]
        eos = 6 in emitted
        actions = emitted[: emitted.index(6)] if eos else emitted
        expected = list(example.target_bases)
        exact += int(eos and actions == expected)
        terminated += int(np.asarray(result["terminated"]))
        invalid += int(np.asarray(result["invalid"]))
    count = len(examples)
    return {
        "greedy_examples": count,
        "greedy_exact_match": exact / max(count, 1),
        "greedy_termination_rate": terminated / max(count, 1),
        "greedy_invalid_rate": invalid / max(count, 1),
        "greedy_compile_or_first_seconds": compile_seconds,
    }


def _checkpoint_metadata(
    *,
    config,
    adam_config,
    data_hash,
    source_vocab,
    target_vocab,
    artifact_prefix,
    artifact_sha,
    batch_size,
    max_source,
    max_target,
    epoch,
    next_batch,
    global_step,
    best_dev_nll,
    seed,
):
    return {
        "run_kind": f"{readout_metadata(target_vocab)['task']}_serial_jax_full_sequence",
        "readout": readout_metadata(target_vocab),
        "model_config": asdict(config),
        "adam_config": asdict(adam_config),
        "dataset_sha256": data_hash,
        "source_vocab": list(source_vocab.tokens),
        "target_vocab": list(target_vocab.tokens),
        "nonlinear_artifact": str(artifact_prefix),
        "nonlinear_artifact_sha256": artifact_sha,
        "batch_size": batch_size,
        "max_source": max_source,
        "max_target_positions": max_target,
        "seed": seed,
        "progress": {
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "best_dev_token_nll": (
                None if not math.isfinite(best_dev_nll) else best_dev_nll
            ),
        },
    }


def command_preflight(args):
    _require_gpu(args.require_gpu)
    config = _model_config(args)
    source_vocab, target_vocab = load_vocabularies(args.data_root)
    splits = {
        name: split_statistics(load_split(args.data_root, name))
        for name in ("train", "dev", "test")
    }
    record = {
        "readout": readout_metadata(target_vocab),
        "environment": _device_record(),
        "logical_pair_qubits": config.logical_pair_qubits,
        "active_simulator_qubits": config.active_simulator_qubits,
        "parameter_shapes": {
            name: list(shape)
            for name, shape in parameter_shapes(
                len(source_vocab), len(target_vocab), config
            ).items()
        },
        "trainable_parameters": sum(
            int(np.prod(shape))
            for shape in parameter_shapes(
                len(source_vocab), len(target_vocab), config
            ).values()
        ),
        "frozen_parameters": sum(
            int(np.prod(shape))
            for shape in frozen_parameter_shapes(config).values()
        ),
        "dataset": splits,
    }
    if args.nonlinear_artifact is not None:
        artifact, _, prefix = _load_frozen(args.nonlinear_artifact, config)
        record["artifact"] = {
            "prefix": str(prefix),
            "format": artifact.source_format,
            "sha256": artifact.npz_sha256,
        }
    if not args.skip_kernel:
        keys = jax.random.split(jax.random.PRNGKey(args.seed), 5)
        states = []
        for key in keys[:4]:
            raw = jax.random.normal(key, (8,)) + 1j * jax.random.normal(
                jax.random.fold_in(key, 1), (8,)
            )
            states.append(raw / jnp.linalg.norm(raw))
        weights = 0.08 * jax.random.normal(keys[4], (1, 7, 3))
        probe = jax.jit(
            jax.value_and_grad(
                lambda values: jnp.sum(
                    pair_swap_zero_row(
                        states[0], states[1], states[2], states[3], values
                    )
                )
            )
        )
        started = time.perf_counter()
        value, gradient = probe(weights)
        jax.block_until_ready(gradient)
        record["kernel_probe"] = {
            "seconds_including_compile": time.perf_counter() - started,
            "value": float(value),
            "gradient_l2": float(jnp.linalg.norm(gradient)),
            "finite": bool(jnp.all(jnp.isfinite(gradient))),
        }
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_pretrain(args):
    _require_gpu(args.require_gpu)
    output_dir = args.output_dir.resolve()
    logger = JsonlLogger(output_dir)
    config = NonlinearPretrainConfig(
        n=args.token_qubits,
        layers1=args.layers1,
        layers2=args.layers2,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        softmax_beta=args.softmax_beta,
        seed=args.seed,
        validation_size=args.validation_size,
    )
    config.validate()
    logger.emit(
        "pretrain_start", config=asdict(config), environment=_device_record()
    )

    def progress(kind, step, total, loss, seconds):
        if step == 1 or step % args.log_every == 0 or step == total:
            logger.emit(
                "pretrain_step",
                kind=kind,
                step=step,
                total_steps=total,
                fidelity_loss=loss,
                fidelity=1.0 - loss,
                seconds=seconds,
            )

    result = pretrain_both(config, progress)
    io_config = {
        "n": config.n,
        "layers1": config.layers1,
        "layers2": config.layers2,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "lr": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.epsilon,
        "softmax_beta": config.softmax_beta,
        "seed": config.seed,
        "validation_size": config.validation_size,
    }
    artifact_result = {
        "config": io_config,
        "softmax_params": result["softmax"]["parameters"],
        "sigmoid_params": result["sigmoid"]["parameters"],
        "softmax_loss_history": result["softmax"]["history"],
        "sigmoid_loss_history": result["sigmoid"]["history"],
        "softmax_validation_loss": result["softmax"]["validation_loss"],
        "sigmoid_validation_loss": result["sigmoid"]["validation_loss"],
    }
    paths = save_nonlinear_artifact(
        output_dir / "paper",
        artifact_result,
        metadata={"environment": _device_record()},
    )
    logger.emit(
        "pretrain_complete",
        artifact_npz=str(paths[0]),
        artifact_json=str(paths[1]),
        softmax_validation_loss=result["softmax"]["validation_loss"],
        sigmoid_validation_loss=result["sigmoid"]["validation_loss"],
        softmax_validation_fidelity=1
        - result["softmax"]["validation_loss"],
        sigmoid_validation_fidelity=1
        - result["sigmoid"]["validation_loss"],
    )
    logger.close()
    return 0


def command_smoke(args):
    _require_gpu(args.require_gpu)
    output_dir = args.output_dir.resolve()
    logger = JsonlLogger(output_dir)
    config = _model_config(args)
    adam_config = _adam_config(args)
    source_vocab, target_vocab = load_vocabularies(args.data_root)
    examples = _load_encoded(
        args.data_root, "train", source_vocab, target_vocab
    )
    example = min(examples, key=lambda item: len(item.src_ids) + len(item.target_ids))
    max_source, max_target = required_shapes([example])
    batch = _device_batch(
        make_batch([example], max_source, max_target, batch_size=1)
    )
    parameters = initialize_parameters(
        jax.random.PRNGKey(args.seed),
        len(source_vocab),
        len(target_vocab),
        config,
    )
    if args.nonlinear_artifact is None:
        frozen = _random_frozen(config, args.seed + 1)
        artifact_prefix = "<random-smoke-only>"
        artifact_sha = None
    else:
        artifact, frozen, prefix = _load_frozen(
            args.nonlinear_artifact, config
        )
        artifact_prefix = str(prefix)
        artifact_sha = artifact.npz_sha256
    optimizer_state = adam_init(parameters)
    train_step = make_train_step(config, adam_config)
    logger.emit(
        "smoke_start",
        uid=example.uid,
        environment=_device_record(),
        logical_pair_qubits=config.logical_pair_qubits,
        active_simulator_qubits=config.active_simulator_qubits,
    )
    before = {name: np.asarray(value) for name, value in parameters.items()}
    started = time.perf_counter()
    parameters, optimizer_state, loss, metrics, optimizer_metrics = train_step(
        parameters, optimizer_state, frozen, batch
    )
    jax.block_until_ready(loss)
    elapsed = time.perf_counter() - started
    if not math.isfinite(float(loss)):
        raise FloatingPointError("Smoke loss is non-finite.")
    if not bool(optimizer_metrics["gradients_finite"]):
        raise FloatingPointError("Smoke gradient is non-finite.")
    change = math.sqrt(
        sum(
            float(np.sum((np.asarray(parameters[name]) - value) ** 2))
            for name, value in before.items()
        )
    )
    metadata = {
        "run_kind": f"{readout_metadata(target_vocab)['task']}_serial_jax_smoke",
        "readout": readout_metadata(target_vocab),
        "model_config": asdict(config),
        "adam_config": asdict(adam_config),
        "source_vocab": list(source_vocab.tokens),
        "target_vocab": list(target_vocab.tokens),
        "nonlinear_artifact": artifact_prefix,
        "nonlinear_artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha256(args.data_root),
        "max_source": max_source,
        "max_target_positions": max_target,
        "progress": {"global_step": 1},
    }
    save_checkpoint(
        output_dir / "latest", parameters, optimizer_state, metadata
    )
    loaded = load_checkpoint(
        output_dir / "latest",
        expected_parameter_shapes=parameter_shapes(
            len(source_vocab), len(target_vocab), config
        ),
    )
    for name in parameters:
        np.testing.assert_array_equal(
            np.asarray(parameters[name]), loaded.parameters[name]
        )
    logger.emit(
        "smoke_complete",
        loss=float(loss),
        token_nll=float(metrics["token_nll"]),
        token_accuracy=float(metrics["token_accuracy"]),
        target_tokens=int(metrics["target_tokens"]),
        gradient_l2=float(optimizer_metrics["gradient_l2"]),
        parameter_change_l2=change,
        seconds_including_compile=elapsed,
        checkpoint=str(output_dir / "latest"),
    )
    logger.close()
    return 0


def _validate_resume_adam(saved, current, allow_learning_rate_change=False):
    """An explicit opt-in may change LR only; Adam moments are restored intact."""
    if not isinstance(saved, dict):
        raise ValueError("Resume checkpoint has no valid Adam configuration.")
    previous, requested = dict(saved), dict(current)
    if allow_learning_rate_change:
        previous.pop("learning_rate", None)
        requested.pop("learning_rate", None)
    if previous != requested:
        raise ValueError(
            "Resume checkpoint Adam configuration differs. "
            "Use --allow-learning-rate-change only when changing learning rate; "
            "other Adam settings must remain unchanged."
        )


def _restore_training(
    args, config, adam_config, source_vocab, target_vocab, data_hash
):
    shapes = parameter_shapes(len(source_vocab), len(target_vocab), config)
    if args.resume is None:
        artifact, frozen, artifact_prefix = _load_frozen(
            args.nonlinear_artifact, config
        )
        parameters = initialize_parameters(
            jax.random.PRNGKey(args.seed),
            len(source_vocab),
            len(target_vocab),
            config,
        )
        optimizer_state = adam_init(parameters)
        return (
            parameters,
            optimizer_state,
            artifact,
            frozen,
            artifact_prefix,
            0,
            0,
            0,
            math.inf,
        )

    loaded = load_checkpoint(
        _checkpoint_prefix(args.resume),
        expected_parameter_shapes=shapes,
    )
    metadata = loaded.metadata
    if metadata.get("run_kind") != f"{readout_metadata(target_vocab)['task']}_serial_jax_full_sequence":
        raise ValueError("Resume checkpoint has the wrong run kind.")
    if metadata.get("readout", readout_metadata(target_vocab)) != readout_metadata(target_vocab):
        raise ValueError("Resume checkpoint readout differs.")
    if metadata.get("dataset_sha256") != data_hash:
        raise ValueError("Resume checkpoint dataset SHA256 differs.")
    if metadata.get("model_config") != asdict(config):
        raise ValueError("Resume checkpoint model configuration differs.")
    _validate_resume_adam(
        metadata.get("adam_config"), asdict(adam_config),
        getattr(args, "allow_learning_rate_change", False),
    )
    if int(metadata.get("seed", -1)) != args.seed:
        raise ValueError("Resume checkpoint seed differs from --seed.")
    progress = metadata["progress"]
    resumed_epoch = int(progress["epoch"])
    resumed_global_step = int(progress["global_step"])
    if args.epochs <= resumed_epoch:
        raise ValueError(
            "--epochs is the total bound and must exceed resumed epoch."
        )
    if args.max_steps is not None and args.max_steps <= resumed_global_step:
        raise ValueError(
            "--max-steps must exceed the checkpoint global step."
        )
    if metadata.get("source_vocab") != list(source_vocab.tokens) or metadata.get(
        "target_vocab"
    ) != list(target_vocab.tokens):
        raise ValueError("Resume checkpoint vocabularies differ.")
    artifact_path = (
        args.nonlinear_artifact
        if args.nonlinear_artifact is not None
        else Path(metadata["nonlinear_artifact"])
    )
    artifact, frozen, artifact_prefix = _load_frozen(artifact_path, config)
    if artifact.npz_sha256 != metadata.get("nonlinear_artifact_sha256"):
        raise ValueError("Resume checkpoint nonlinear artifact differs.")
    best = progress.get("best_dev_token_nll")
    return (
        {name: jnp.asarray(value) for name, value in loaded.parameters.items()},
        {
            "step": jnp.asarray(loaded.optimizer_state["step"], dtype=jnp.int32),
            "m": {
                name: jnp.asarray(value)
                for name, value in loaded.optimizer_state["m"].items()
            },
            "v": {
                name: jnp.asarray(value)
                for name, value in loaded.optimizer_state["v"].items()
            },
        },
        artifact,
        frozen,
        artifact_prefix,
        int(progress["epoch"]),
        int(progress["next_batch"]),
        int(progress["global_step"]),
        math.inf if best is None else float(best),
    )


def command_train(args):
    _require_gpu(args.require_gpu)
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    output_dir = args.output_dir.resolve()
    logger = JsonlLogger(output_dir)
    config = _model_config(args)
    adam_config = _adam_config(args)
    source_vocab, target_vocab = load_vocabularies(args.data_root)
    train_examples = _load_encoded(
        args.data_root,
        "train",
        source_vocab,
        target_vocab,
        args.max_train_examples,
    )
    dev_examples = _load_encoded(
        args.data_root,
        "dev",
        source_vocab,
        target_vocab,
        args.max_dev_examples,
    )
    max_source, max_target = required_shapes(train_examples, dev_examples)
    data_hash = dataset_sha256(args.data_root)
    (
        parameters,
        optimizer_state,
        artifact,
        frozen,
        artifact_prefix,
        start_epoch,
        start_batch,
        global_step,
        best_dev_nll,
    ) = _restore_training(
        args,
        config,
        adam_config,
        source_vocab,
        target_vocab,
        data_hash,
    )
    if args.resume is not None:
        loaded = load_checkpoint(_checkpoint_prefix(args.resume))
        if loaded.metadata["batch_size"] != args.batch_size:
            raise ValueError("Resume requires the original batch size.")
        if loaded.metadata["max_source"] != max_source or loaded.metadata[
            "max_target_positions"
        ] != max_target:
            raise ValueError("Resume fixed sequence shapes differ.")

    train_step = make_train_step(config, adam_config)
    batches_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    logger.emit(
        "train_start",
        readout=readout_metadata(target_vocab),
        environment=_device_record(),
        model_config=asdict(config),
        optimizer=asdict(adam_config),
        allow_learning_rate_change=getattr(args, "allow_learning_rate_change", False),
        objective="mean of per-sequence full teacher-forced token CE",
        update_unit="one Adam update per real mini-batch",
        train_examples=len(train_examples),
        dev_examples=len(dev_examples),
        batch_size=args.batch_size,
        batches_per_epoch=batches_per_epoch,
        max_source=max_source,
        max_target_positions=max_target,
        logical_pair_qubits=config.logical_pair_qubits,
        active_simulator_qubits=config.active_simulator_qubits,
        trainable_parameters=count_parameters(parameters),
        frozen_parameters=sum(value.size for value in artifact.frozen_parameters.values()),
        nonlinear_artifact=str(artifact_prefix),
        start_epoch=start_epoch,
        start_batch=start_batch,
        global_step=global_step,
        epoch_limit=args.epochs,
    )

    def save(prefix, epoch, next_batch):
        metadata = _checkpoint_metadata(
            config=config,
            adam_config=adam_config,
            data_hash=data_hash,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            artifact_prefix=artifact_prefix,
            artifact_sha=artifact.npz_sha256,
            batch_size=args.batch_size,
            max_source=max_source,
            max_target=max_target,
            epoch=epoch,
            next_batch=next_batch,
            global_step=global_step,
            best_dev_nll=best_dev_nll,
            seed=args.seed,
        )
        paths = save_checkpoint(
            output_dir / prefix, parameters, optimizer_state, metadata
        )
        logger.emit(
            "checkpoint_saved",
            checkpoint=prefix,
            npz=str(paths[0]),
            json=str(paths[1]),
            global_step=global_step,
        )

    stop_due_to_limit = False
    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(
            len(train_examples)
        )
        epoch_started = time.perf_counter()
        for batch_index, batch in iter_batches(
            train_examples,
            order,
            args.batch_size,
            max_source,
            max_target,
            start_batch if epoch == start_epoch else 0,
        ):
            started = time.perf_counter()
            (
                parameters,
                optimizer_state,
                loss,
                metrics,
                optimizer_metrics,
            ) = train_step(
                parameters, optimizer_state, frozen, _device_batch(batch)
            )
            jax.block_until_ready(loss)
            seconds = time.perf_counter() - started
            global_step += 1
            if not math.isfinite(float(loss)):
                raise FloatingPointError("Training loss became non-finite.")
            if not bool(optimizer_metrics["gradients_finite"]) or not bool(
                optimizer_metrics["parameters_finite"]
            ):
                raise FloatingPointError("Training gradient/parameters became non-finite.")
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= batches_per_epoch:
                next_epoch, next_batch = epoch + 1, 0
            if global_step == 1 or global_step % args.log_every == 0:
                logger.emit(
                    "train_step",
                    epoch=epoch,
                    batch=batch_index,
                    global_step=global_step,
                    loss=float(loss),
                    full_sequence_ce=float(loss),
                    token_nll=float(metrics["token_nll"]),
                    token_accuracy=float(metrics["token_accuracy"]),
                    target_tokens=int(metrics["target_tokens"]),
                    gradient_l2=float(optimizer_metrics["gradient_l2"]),
                    clipped_gradient_l2=float(
                        optimizer_metrics["clipped_gradient_l2"]
                    ),
                    gradient_max_abs=float(
                        optimizer_metrics["gradient_max_abs"]
                    ),
                    clip_scale=float(optimizer_metrics["clip_scale"]),
                    parameter_change_l2=float(
                        optimizer_metrics["parameter_change_l2"]
                    ),
                    learning_rate=adam_config.learning_rate,
                    seconds=seconds,
                    examples_per_second=sum(
                        uid != "<PAD-EXAMPLE>" for uid in batch["uids"]
                    )
                    / seconds,
                )
            if global_step % args.save_every == 0:
                save("latest", next_epoch, next_batch)
            if args.eval_every and global_step % args.eval_every == 0:
                started_eval = time.perf_counter()
                dev = _evaluate(
                    parameters,
                    frozen,
                    dev_examples,
                    args.eval_batch_size,
                    max_source,
                    max_target,
                    config,
                )
                logger.emit(
                    "dev_metrics",
                    global_step=global_step,
                    seconds=time.perf_counter() - started_eval,
                    **dev,
                )
                if dev["token_nll"] < best_dev_nll:
                    best_dev_nll = dev["token_nll"]
                    save("best", next_epoch, next_batch)
                save("latest", next_epoch, next_batch)
            if STOP_REQUESTED or (
                args.max_steps is not None and global_step >= args.max_steps
            ):
                stop_due_to_limit = True
                save("latest", next_epoch, next_batch)
                break
        start_batch = 0
        if stop_due_to_limit:
            break
        dev = _evaluate(
            parameters,
            frozen,
            dev_examples,
            args.eval_batch_size,
            max_source,
            max_target,
            config,
        )
        logger.emit(
            "epoch_complete",
            epoch=epoch,
            global_step=global_step,
            epoch_seconds=time.perf_counter() - epoch_started,
            dev=dev,
        )
        if dev["token_nll"] < best_dev_nll:
            best_dev_nll = dev["token_nll"]
            save("best", epoch + 1, 0)
        save("latest", epoch + 1, 0)

    logger.emit(
        "train_complete" if not STOP_REQUESTED else "train_interrupted",
        global_step=global_step,
        best_dev_token_nll=(
            None if not math.isfinite(best_dev_nll) else best_dev_nll
        ),
        stopped_by_max_steps=bool(
            args.max_steps is not None and global_step >= args.max_steps
        ),
        stopped_by_signal=STOP_REQUESTED,
        latest_checkpoint=str(output_dir / "latest"),
        best_checkpoint=str(output_dir / "best"),
    )
    logger.close()
    return 0


def command_evaluate(args):
    _require_gpu(args.require_gpu)
    checkpoint = load_checkpoint(_checkpoint_prefix(args.checkpoint))
    metadata = checkpoint.metadata
    config = JaxSerialConfig(**metadata["model_config"])
    config.validate()
    source_vocab, target_vocab = load_vocabularies(args.data_root)
    if metadata["dataset_sha256"] != dataset_sha256(args.data_root):
        raise ValueError("Checkpoint dataset SHA256 differs.")
    if metadata.get("source_vocab") != list(source_vocab.tokens) or metadata.get("target_vocab") != list(target_vocab.tokens):
        raise ValueError("Checkpoint vocabularies differ.")
    if metadata.get("readout", readout_metadata(target_vocab)) != readout_metadata(target_vocab):
        raise ValueError("Checkpoint readout differs.")
    artifact_path = (
        args.nonlinear_artifact
        if args.nonlinear_artifact is not None
        else Path(metadata["nonlinear_artifact"])
    )
    artifact, frozen, _ = _load_frozen(artifact_path, config)
    if artifact.npz_sha256 != metadata["nonlinear_artifact_sha256"]:
        raise ValueError("Checkpoint nonlinear artifact differs.")
    parameters = {
        name: jnp.asarray(value)
        for name, value in checkpoint.parameters.items()
    }
    examples = _load_encoded(
        args.data_root,
        args.split,
        source_vocab,
        target_vocab,
        args.max_examples,
    )
    max_source, max_target = required_shapes(examples)
    logger = JsonlLogger(args.output_dir.resolve() if args.output_dir else None)
    started = time.perf_counter()
    result = _evaluate(
        parameters,
        frozen,
        examples,
        args.batch_size,
        max_source,
        max_target,
        config,
    )
    if args.decode_examples:
        result.update(
            _greedy_metrics(
                parameters,
                frozen,
                examples[: args.decode_examples],
                source_vocab,
                target_vocab,
                max_source,
                args.max_decode_steps,
                config,
            )
        )
    logger.emit(
        "evaluation_complete",
        split=args.split,
        checkpoint=str(_checkpoint_prefix(args.checkpoint)),
        seconds=time.perf_counter() - started,
        **result,
    )
    logger.close()
    return 0


def command_benchmark(args):
    _require_gpu(args.require_gpu)
    output_dir = args.output_dir.resolve()
    logger = JsonlLogger(output_dir)
    config = _model_config(args)
    adam_config = _adam_config(args)
    source_vocab, target_vocab = load_vocabularies(args.data_root)
    examples = _load_encoded(
        args.data_root, "train", source_vocab, target_vocab
    )
    max_source, max_target = required_shapes(examples)
    longest = sorted(
        examples,
        key=lambda item: len(item.src_ids) + len(item.target_ids),
        reverse=True,
    )[: args.batch_size]
    batch = _device_batch(
        make_batch(
            longest,
            max_source,
            max_target,
            batch_size=args.batch_size,
        )
    )
    parameters = initialize_parameters(
        jax.random.PRNGKey(args.seed),
        len(source_vocab),
        len(target_vocab),
        config,
    )
    if args.nonlinear_artifact is None:
        frozen = _random_frozen(config, args.seed + 1)
    else:
        _, frozen, _ = _load_frozen(args.nonlinear_artifact, config)
    optimizer_state = adam_init(parameters)
    step = make_train_step(config, adam_config)
    logger.emit(
        "benchmark_start",
        environment=_device_record(),
        batch_size=args.batch_size,
        max_source=max_source,
        max_target_positions=max_target,
        repeats=args.repeats,
    )
    started = time.perf_counter()
    parameters, optimizer_state, loss, _, _ = step(
        parameters, optimizer_state, frozen, batch
    )
    jax.block_until_ready(loss)
    compile_seconds = time.perf_counter() - started
    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        parameters, optimizer_state, loss, _, _ = step(
            parameters, optimizer_state, frozen, batch
        )
        jax.block_until_ready(loss)
        times.append(time.perf_counter() - started)
    logger.emit(
        "benchmark_complete",
        compile_and_first_step_seconds=compile_seconds,
        steady_seconds=times,
        steady_median_seconds=float(np.median(times)),
        steady_p95_seconds=float(np.percentile(times, 95)),
        examples_per_second=args.batch_size / float(np.median(times)),
        final_loss=float(loss),
    )
    logger.close()
    return 0


def _add_data_root(parser):
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def _add_model(parser):
    parser.add_argument("--token-qubits", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--entangling-layers", type=int, default=1)
    parser.add_argument("--softmax-layers1", type=int, default=1)
    parser.add_argument("--softmax-layers2", type=int, default=1)
    parser.add_argument("--ffn-layers", type=int, default=1)
    parser.add_argument("--sigmoid-layers1", type=int, default=1)
    parser.add_argument("--sigmoid-layers2", type=int, default=1)
    parser.add_argument("--model-epsilon", type=float, default=1.0e-12)


def _add_adam(parser):
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)


def _add_gpu(parser):
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail instead of silently using CPU.",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_data_root(preflight)
    _add_model(preflight)
    _add_gpu(preflight)
    preflight.add_argument("--nonlinear-artifact", type=Path)
    preflight.add_argument("--skip-kernel", action="store_true")
    preflight.add_argument("--seed", type=int, default=2026)
    preflight.set_defaults(handler=command_preflight)

    pretrain = subparsers.add_parser("pretrain-nonlinear")
    _add_gpu(pretrain)
    pretrain.add_argument("--output-dir", type=Path, required=True)
    pretrain.add_argument("--token-qubits", type=int, default=3)
    pretrain.add_argument("--layers1", type=int, default=1)
    pretrain.add_argument("--layers2", type=int, default=1)
    pretrain.add_argument("--steps", type=int, default=1600)
    pretrain.add_argument("--batch-size", type=int, default=16)
    pretrain.add_argument("--learning-rate", type=float, default=0.01)
    pretrain.add_argument("--adam-beta1", type=float, default=0.9)
    pretrain.add_argument("--adam-beta2", type=float, default=0.99)
    pretrain.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    pretrain.add_argument("--softmax-beta", type=float, default=1.0)
    pretrain.add_argument("--validation-size", type=int, default=256)
    pretrain.add_argument("--seed", type=int, default=2026)
    pretrain.add_argument("--log-every", type=int, default=10)
    pretrain.set_defaults(handler=command_pretrain)

    smoke = subparsers.add_parser("smoke")
    _add_data_root(smoke)
    _add_model(smoke)
    _add_adam(smoke)
    _add_gpu(smoke)
    smoke.add_argument("--nonlinear-artifact", type=Path)
    smoke.add_argument(
        "--output-dir", type=Path, default=Path("runs/jax-smoke")
    )
    smoke.add_argument("--seed", type=int, default=2026)
    smoke.set_defaults(handler=command_smoke)

    train = subparsers.add_parser("train")
    _add_data_root(train)
    _add_model(train)
    _add_adam(train)
    _add_gpu(train)
    train.add_argument("--nonlinear-artifact", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--allow-learning-rate-change", action="store_true",
        help="Allow a new learning rate on resume while preserving Adam moments and step. Other optimizer settings must match.",
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--eval-batch-size", type=int, default=1)
    train.add_argument("--max-train-examples", type=int)
    train.add_argument("--max-dev-examples", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--eval-every", type=int, default=0)
    train.add_argument("--save-every", type=int, default=25)
    train.add_argument("--log-every", type=int, default=1)
    train.add_argument("--seed", type=int, default=2026)
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate")
    _add_data_root(evaluate)
    _add_gpu(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--nonlinear-artifact", type=Path)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument(
        "--split", choices=("train", "dev", "test", "gen_length", "gen_depth"), default="dev"
    )
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-examples", type=int)
    evaluate.add_argument("--decode-examples", type=int, default=0)
    evaluate.add_argument("--max-decode-steps", type=int, default=33)
    evaluate.set_defaults(handler=command_evaluate)

    benchmark = subparsers.add_parser("benchmark")
    _add_data_root(benchmark)
    _add_model(benchmark)
    _add_adam(benchmark)
    _add_gpu(benchmark)
    benchmark.add_argument("--nonlinear-artifact", type=Path)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.set_defaults(handler=command_benchmark)
    return parser


def _validate_args(args):
    if getattr(args, "allow_learning_rate_change", False) and getattr(args, "resume", None) is None:
        raise ValueError("--allow-learning-rate-change requires --resume.")
    positive = (
        "steps",
        "batch_size",
        "validation_size",
        "log_every",
        "epochs",
        "eval_batch_size",
        "max_train_examples",
        "max_dev_examples",
        "max_steps",
        "save_every",
        "max_examples",
        "max_decode_steps",
        "repeats",
    )
    for name in positive:
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if getattr(args, "decode_examples", 0) < 0:
        raise ValueError("--decode-examples cannot be negative.")
    if getattr(args, "eval_every", 0) < 0:
        raise ValueError("--eval-every cannot be negative.")
    if (
        getattr(args, "command", None) == "train"
        and args.resume is None
        and args.nonlinear_artifact is None
    ):
        raise ValueError(
            "--nonlinear-artifact is required for a new formal training run."
        )


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        return int(args.handler(args))
    except (
        AssertionError,
        FileNotFoundError,
        FloatingPointError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
