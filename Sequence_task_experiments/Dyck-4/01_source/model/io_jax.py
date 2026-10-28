"""Strict, dependency-light persistence for the JAX serial model.

This module deliberately imports only the Python standard library and NumPy.
Objects returned here therefore remain usable before JAX selects a platform or
allocates accelerator memory.  Callers can pass the returned arrays to
``jax.device_put`` when constructing the compiled training state.

Two unrelated formats are handled:

* frozen Softmax/Sigmoid artifacts, including the version-1 artifacts produced
  by ``concurrent_without_jax/nonlinear_pretraining.py``;
* JAX training checkpoints containing a named trainable parameter tree and the
  matching named Adam first/second moments.

Both formats commit the NPZ payload first and a SHA256-bound JSON record last.
Each individual file replacement is atomic.  If a process stops between the
two replacements, loading fails safely with a checksum/identifier mismatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np


LEGACY_ARTIFACT_FORMAT = "quantum_transformer.nonlinear_pretraining"
JAX_ARTIFACT_FORMAT = "quantum_transformer.nonlinear_pretraining.jax"
ARTIFACT_VERSION = 1
CHECKPOINT_SCHEMA = "quantum_transformer.serial_jax.checkpoint"
CHECKPOINT_VERSION = 1

_ARTIFACT_ID_KEY = "__artifact_id__"
_CHECKPOINT_ID_KEY = "__checkpoint_id__"
_ADAM_STEP_KEY = "__adam_step__"
_PARAMETER_ARRAYS = ("softmax_params", "sigmoid_params")
_HISTORY_ARRAYS = ("softmax_loss_history", "sigmoid_loss_history")
_LOSS_ARRAYS = ("softmax_validation_loss", "sigmoid_validation_loss")
_ARTIFACT_ARRAYS = {
    *_PARAMETER_ARRAYS,
    *_HISTORY_ARRAYS,
    *_LOSS_ARRAYS,
    _ARTIFACT_ID_KEY,
}
_LEGACY_CONFIG_KEYS = {
    "n",
    "layers1",
    "layers2",
    "steps",
    "batch_size",
    "lr",
    "beta1",
    "beta2",
    "eps",
    "softmax_beta",
    "seed",
    "validation_size",
}
_MANIFEST_KEYS = {"archive_key", "shape", "dtype", "sha256"}


@dataclass(frozen=True)
class NonlinearArtifact:
    """Validated frozen nonlinear parameters and their diagnostics."""

    config: dict[str, Any]
    softmax_params: np.ndarray
    sigmoid_params: np.ndarray
    softmax_loss_history: np.ndarray
    sigmoid_loss_history: np.ndarray
    softmax_validation_loss: float
    sigmoid_validation_loss: float
    metadata: dict[str, Any]
    source_format: str = JAX_ARTIFACT_FORMAT
    artifact_id: str | None = None
    npz_sha256: str | None = None

    @property
    def frozen_parameters(self) -> dict[str, np.ndarray]:
        """Return copies under the names expected by ``serial_model_jax``."""

        return {
            "softmax": np.array(self.softmax_params, copy=True),
            "sigmoid": np.array(self.sigmoid_params, copy=True),
        }


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Strictly validated NumPy view of a JAX training checkpoint."""

    parameters: dict[str, np.ndarray]
    optimizer_state: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint_id: str
    npz_sha256: str

    @property
    def adam_step(self) -> int:
        return int(self.optimizer_state["step"])


def paired_paths(prefix: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Resolve a prefix, ``.npz`` path, or ``.json`` path to its file pair."""

    base = Path(prefix)
    if base.suffix.lower() in {".npz", ".json"}:
        base = base.with_suffix("")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_array_sha256(value: np.ndarray) -> str:
    """Match nonlinear_pretraining.py's version-1 array digest exactly."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and bytes with unambiguous length prefixes."""

    array = np.ascontiguousarray(value)
    dtype = str(array.dtype).encode("ascii")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(dtype).to_bytes(8, "big"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(8, "big"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON metadata mapping keys must be strings.")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    # JAX scalars/arrays implement the NumPy array protocol but this module must
    # not import JAX merely to recognize them.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Metadata is not finite JSON-serializable data.") from exc


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer <= 0 or integer != value:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _unitary_parameter_count(n: int, layers: int) -> int:
    """Mirror the stride-2/long-range ansatz without importing PennyLane."""

    if layers == 0:
        return 0
    if n <= 4:
        return layers * (4**n - 1)
    window_count = 0
    for layer in range(layers):
        offset = 0 if layer % 2 == 0 else 2
        window_count += len(range(offset, n - 3, 4))
    return window_count * 255 + (layers - 1) * (n // 2)


def nonlinear_parameter_count(config: Mapping[str, Any]) -> int:
    """Return the expected parameter count of one nonlinear circuit copy."""

    normalized = _validate_nonlinear_config(config)
    return _unitary_parameter_count(normalized["n"], normalized["layers1"]) + (
        _unitary_parameter_count(normalized["n"], normalized["layers2"])
    )


def _config_mapping(value: Any, label: str = "config") -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping or dataclass.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return dict(_jsonable(value))


def _validate_nonlinear_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = _config_mapping(value)
    required = {"n", "layers1", "layers2"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Nonlinear config is missing keys: {sorted(missing)}.")
    config["n"] = _positive_integer(config["n"], "config.n")
    config["layers1"] = _nonnegative_integer(config["layers1"], "config.layers1")
    config["layers2"] = _nonnegative_integer(config["layers2"], "config.layers2")
    if config["layers1"] + config["layers2"] <= 0:
        raise ValueError("At least one nonlinear unitary segment is required.")

    for key in ("steps", "batch_size", "validation_size"):
        if key in config:
            config[key] = _positive_integer(config[key], f"config.{key}")
    if "seed" in config:
        config["seed"] = _nonnegative_integer(config["seed"], "config.seed")
    for key in ("lr", "eps", "softmax_beta"):
        if key in config:
            number = float(config[key])
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"config.{key} must be finite and positive.")
    for key in ("beta1", "beta2"):
        if key in config:
            number = float(config[key])
            if not math.isfinite(number) or not 0.0 <= number < 1.0:
                raise ValueError(f"config.{key} must be finite and in [0, 1).")
    return config


def _coerce_float_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating dtype, got {array.dtype}.")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return np.array(array, copy=True, order="C")


def _validated_artifact(
    *,
    config: Any,
    softmax_params: Any,
    sigmoid_params: Any,
    softmax_loss_history: Any,
    sigmoid_loss_history: Any,
    softmax_validation_loss: Any,
    sigmoid_validation_loss: Any,
    metadata: Mapping[str, Any] | None = None,
    source_format: str = JAX_ARTIFACT_FORMAT,
    artifact_id: str | None = None,
    npz_sha256: str | None = None,
) -> NonlinearArtifact:
    normalized_config = _validate_nonlinear_config(_config_mapping(config))
    expected = nonlinear_parameter_count(normalized_config)
    softmax = _coerce_float_array(softmax_params, "softmax_params", ndim=1)
    sigmoid = _coerce_float_array(sigmoid_params, "sigmoid_params", ndim=1)
    if softmax.shape != (expected,) or sigmoid.shape != (expected,):
        raise ValueError(
            "Nonlinear parameter shape mismatch: expected "
            f"{(expected,)}, got softmax={softmax.shape}, sigmoid={sigmoid.shape}."
        )
    softmax_history = _coerce_float_array(
        softmax_loss_history, "softmax_loss_history", ndim=1
    )
    sigmoid_history = _coerce_float_array(
        sigmoid_loss_history, "sigmoid_loss_history", ndim=1
    )
    if softmax_history.shape != sigmoid_history.shape:
        raise ValueError("Softmax and Sigmoid loss histories must have equal shapes.")
    if "steps" in normalized_config and softmax_history.shape != (
        normalized_config["steps"],
    ):
        raise ValueError("Loss history length does not match config.steps.")
    validation_losses: list[float] = []
    for name, value in (
        ("softmax_validation_loss", softmax_validation_loss),
        ("sigmoid_validation_loss", sigmoid_validation_loss),
    ):
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError(f"{name} must be scalar, got shape {array.shape}.")
        number = float(array.item())
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        validation_losses.append(number)
    if metadata is None:
        normalized_metadata: dict[str, Any] = {}
    else:
        if not isinstance(metadata, Mapping):
            raise TypeError("Artifact metadata must be a mapping.")
        normalized_metadata = dict(_jsonable(metadata))
        _json_bytes(normalized_metadata)
    return NonlinearArtifact(
        config=normalized_config,
        softmax_params=softmax,
        sigmoid_params=sigmoid,
        softmax_loss_history=softmax_history,
        sigmoid_loss_history=sigmoid_history,
        softmax_validation_loss=validation_losses[0],
        sigmoid_validation_loss=validation_losses[1],
        metadata=normalized_metadata,
        source_format=source_format,
        artifact_id=artifact_id,
        npz_sha256=npz_sha256,
    )


def _result_field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise ValueError(f"Nonlinear result is missing {name!r}.")
        return result[name]
    try:
        return getattr(result, name)
    except AttributeError as exc:
        raise ValueError(f"Nonlinear result is missing {name!r}.") from exc


def save_nonlinear_artifact(
    prefix: str | os.PathLike[str],
    result: NonlinearArtifact | Mapping[str, Any] | Any,
    *,
    config: Mapping[str, Any] | Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save a JAX artifact from a result object/mapping with named fields.

    ``result`` must expose the six conventional arrays/scalars.  Its ``config``
    is used unless explicitly supplied.  This accepts the existing
    ``NonlinearPretrainingResult`` without importing its PennyLane module.
    """

    source_config = config if config is not None else _result_field(result, "config")
    source_metadata = metadata
    if source_metadata is None and isinstance(result, NonlinearArtifact):
        source_metadata = result.metadata
    artifact = _validated_artifact(
        config=source_config,
        softmax_params=_result_field(result, "softmax_params"),
        sigmoid_params=_result_field(result, "sigmoid_params"),
        softmax_loss_history=_result_field(result, "softmax_loss_history"),
        sigmoid_loss_history=_result_field(result, "sigmoid_loss_history"),
        softmax_validation_loss=_result_field(result, "softmax_validation_loss"),
        sigmoid_validation_loss=_result_field(result, "sigmoid_validation_loss"),
        metadata=source_metadata,
    )
    artifact_id = uuid.uuid4().hex
    arrays = {
        "softmax_params": artifact.softmax_params,
        "sigmoid_params": artifact.sigmoid_params,
        "softmax_loss_history": artifact.softmax_loss_history,
        "sigmoid_loss_history": artifact.sigmoid_loss_history,
        "softmax_validation_loss": np.asarray(
            artifact.softmax_validation_loss, dtype=artifact.softmax_params.dtype
        ),
        "sigmoid_validation_loss": np.asarray(
            artifact.sigmoid_validation_loss, dtype=artifact.sigmoid_params.dtype
        ),
        _ARTIFACT_ID_KEY: np.asarray(artifact_id),
    }
    npz_path, json_path = paired_paths(prefix)

    def write_npz(handle: Any) -> None:
        np.savez_compressed(handle, **arrays)

    _atomic_write(npz_path, write_npz)
    npz_sha = _file_sha256(npz_path)
    manifests = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _legacy_array_sha256(value),
        }
        for name, value in arrays.items()
        if name != _ARTIFACT_ID_KEY
    }
    count = nonlinear_parameter_count(artifact.config)
    record = {
        "format": JAX_ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "artifact_id": artifact_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "npz_sha256": npz_sha,
        "config": artifact.config,
        "topology": {
            "name": "paper_nonlinear_layer",
            "n": artifact.config["n"],
            "layers1": artifact.config["layers1"],
            "layers2": artifact.config["layers2"],
            "parameter_count_per_function": count,
            "postselection": "zero_branch_normalization",
            "implementation": "pure_jax_statevector",
        },
        "producer": {"module": "io_jax", "numpy": np.__version__},
        "arrays": manifests,
        "metrics": {
            "softmax": {
                "last_train_loss": float(artifact.softmax_loss_history[-1]),
                "validation_loss": artifact.softmax_validation_loss,
                "validation_fidelity": 1.0 - artifact.softmax_validation_loss,
            },
            "sigmoid": {
                "last_train_loss": float(artifact.sigmoid_loss_history[-1]),
                "validation_loss": artifact.sigmoid_validation_loss,
                "validation_fidelity": 1.0 - artifact.sigmoid_validation_loss,
            },
        },
        "metadata": artifact.metadata,
    }
    encoded = _json_bytes(record)
    _atomic_write(json_path, lambda handle: handle.write(encoded))
    return npz_path, json_path


def _validate_artifact_manifest(
    manifests: Any, arrays: Mapping[str, np.ndarray]
) -> None:
    if not isinstance(manifests, Mapping):
        raise ValueError("Artifact arrays manifest must be a JSON object.")
    expected = _ARTIFACT_ARRAYS - {_ARTIFACT_ID_KEY}
    _require_exact_keys(manifests, expected, "artifact arrays manifest")
    for name in expected:
        manifest = manifests[name]
        if not isinstance(manifest, Mapping):
            raise ValueError(f"Artifact array manifest {name!r} must be an object.")
        _require_exact_keys(manifest, {"shape", "dtype", "sha256"}, name)
        array = arrays[name]
        if list(array.shape) != manifest["shape"]:
            raise ValueError(f"Artifact shape manifest mismatch for {name!r}.")
        if str(array.dtype) != manifest["dtype"]:
            raise ValueError(f"Artifact dtype manifest mismatch for {name!r}.")
        if _legacy_array_sha256(array) != manifest["sha256"]:
            raise ValueError(f"Artifact array SHA256 mismatch for {name!r}.")


def _validate_artifact_metrics(recorded: Any, artifact: NonlinearArtifact) -> None:
    if not isinstance(recorded, Mapping):
        raise ValueError("Artifact metrics must be a JSON object.")
    _require_exact_keys(recorded, {"softmax", "sigmoid"}, "artifact metrics")
    for kind in ("softmax", "sigmoid"):
        item = recorded[kind]
        if not isinstance(item, Mapping):
            raise ValueError(f"Artifact metric {kind!r} must be an object.")
        _require_exact_keys(
            item,
            {"last_train_loss", "validation_loss", "validation_fidelity"},
            f"artifact metric {kind}",
        )
        history = getattr(artifact, f"{kind}_loss_history")
        loss = getattr(artifact, f"{kind}_validation_loss")
        expected = (float(history[-1]), float(loss), 1.0 - float(loss))
        actual = (
            item["last_train_loss"],
            item["validation_loss"],
            item["validation_fidelity"],
        )
        for value, reference in zip(actual, expected):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), reference, rel_tol=0.0, abs_tol=1.0e-14
                )
            ):
                raise ValueError(f"Artifact metric validation failed for {kind!r}.")


def _validate_expected_artifact_config(
    config: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | Any | None,
    expected_n: int | None,
    expected_layers1: int | None,
    expected_layers2: int | None,
) -> None:
    expected: dict[str, Any] = {}
    if expected_config is not None:
        expected.update(_config_mapping(expected_config, "expected_config"))
    for name, value in (
        ("n", expected_n),
        ("layers1", expected_layers1),
        ("layers2", expected_layers2),
    ):
        if value is not None:
            if name in expected and expected[name] != value:
                raise ValueError(f"Conflicting expected nonlinear {name} values.")
            expected[name] = value
    for name, value in expected.items():
        if name not in config:
            raise ValueError(f"Artifact config has no expected field {name!r}.")
        if config[name] != value:
            raise ValueError(
                f"Artifact config mismatch for {name!r}: "
                f"expected {value!r}, got {config[name]!r}."
            )


def load_nonlinear_artifact(
    prefix: str | os.PathLike[str],
    *,
    expected_config: Mapping[str, Any] | Any | None = None,
    expected_n: int | None = None,
    expected_layers1: int | None = None,
    expected_layers2: int | None = None,
) -> NonlinearArtifact:
    """Load and validate either a legacy or pure-JAX nonlinear artifact."""

    npz_path, json_path = paired_paths(prefix)
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Artifact requires both {npz_path} and {json_path}.")
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse artifact metadata {json_path}.") from exc
    if not isinstance(record, Mapping):
        raise ValueError("Artifact metadata root must be a JSON object.")
    artifact_format = record.get("format")
    if artifact_format == LEGACY_ARTIFACT_FORMAT:
        expected_record_keys = {
            "format",
            "format_version",
            "artifact_id",
            "created_at_utc",
            "npz_sha256",
            "config",
            "topology",
            "software",
            "arrays",
            "metrics",
        }
    elif artifact_format == JAX_ARTIFACT_FORMAT:
        expected_record_keys = {
            "format",
            "format_version",
            "artifact_id",
            "created_at_utc",
            "npz_sha256",
            "config",
            "topology",
            "producer",
            "arrays",
            "metrics",
            "metadata",
        }
    else:
        raise ValueError(f"Unsupported nonlinear artifact format: {artifact_format!r}.")
    _require_exact_keys(record, expected_record_keys, "artifact metadata")
    if record["format_version"] != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported nonlinear artifact version: {record['format_version']!r}."
        )
    artifact_id = record["artifact_id"]
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("Artifact ID must be a non-empty string.")
    npz_sha = _file_sha256(npz_path)
    if record["npz_sha256"] != npz_sha:
        raise ValueError("Artifact NPZ SHA256 does not match its JSON record.")

    config = _validate_nonlinear_config(_config_mapping(record["config"]))
    if artifact_format == LEGACY_ARTIFACT_FORMAT:
        _require_exact_keys(config, _LEGACY_CONFIG_KEYS, "legacy artifact config")
        topology = {
            "name": "paper_nonlinear_layer",
            "n": config["n"],
            "layers1": config["layers1"],
            "layers2": config["layers2"],
            "parameter_count_per_function": nonlinear_parameter_count(config),
            "postselection": "projector_zero",
            "block_pattern": "stride2",
            "bridge": "long_range",
        }
        if record["topology"] != topology:
            raise ValueError("Legacy artifact topology does not match its config.")
        software = record["software"]
        if not isinstance(software, Mapping):
            raise ValueError("Legacy artifact software must be an object.")
        _require_exact_keys(
            software,
            {"pennylane", "numpy", "device", "interface", "diff_method"},
            "legacy artifact software",
        )
        if (
            software["device"] != "default.qubit"
            or software["interface"] != "autograd"
            or software["diff_method"] != "backprop"
        ):
            raise ValueError("Legacy artifact has an unsupported training pipeline.")
        metadata: dict[str, Any] = {}
    else:
        topology = record["topology"]
        expected_topology = {
            "name": "paper_nonlinear_layer",
            "n": config["n"],
            "layers1": config["layers1"],
            "layers2": config["layers2"],
            "parameter_count_per_function": nonlinear_parameter_count(config),
            "postselection": "zero_branch_normalization",
            "implementation": "pure_jax_statevector",
        }
        if topology != expected_topology:
            raise ValueError("JAX artifact topology does not match its config.")
        producer = record["producer"]
        if not isinstance(producer, Mapping):
            raise ValueError("JAX artifact producer must be an object.")
        _require_exact_keys(producer, {"module", "numpy"}, "artifact producer")
        if producer["module"] != "io_jax":
            raise ValueError("JAX artifact producer is unsupported.")
        raw_metadata = record["metadata"]
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("JAX artifact metadata must be an object.")
        metadata = dict(raw_metadata)

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != _ARTIFACT_ARRAYS:
                missing = sorted(_ARTIFACT_ARRAYS - set(archive.files))
                extra = sorted(set(archive.files) - _ARTIFACT_ARRAYS)
                raise ValueError(
                    f"Artifact NPZ keys differ; missing={missing}, extra={extra}."
                )
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Artifact NPZ"):
            raise
        raise ValueError(f"Cannot load artifact payload {npz_path}.") from exc
    if arrays[_ARTIFACT_ID_KEY].shape != ():
        raise ValueError("Artifact ID array must be scalar.")
    if str(arrays[_ARTIFACT_ID_KEY].item()) != artifact_id:
        raise ValueError("Artifact IDs differ between NPZ and JSON.")
    _validate_artifact_manifest(record["arrays"], arrays)

    artifact = _validated_artifact(
        config=config,
        softmax_params=arrays["softmax_params"],
        sigmoid_params=arrays["sigmoid_params"],
        softmax_loss_history=arrays["softmax_loss_history"],
        sigmoid_loss_history=arrays["sigmoid_loss_history"],
        softmax_validation_loss=arrays["softmax_validation_loss"],
        sigmoid_validation_loss=arrays["sigmoid_validation_loss"],
        metadata=metadata,
        source_format=artifact_format,
        artifact_id=artifact_id,
        npz_sha256=npz_sha,
    )
    _validate_artifact_metrics(record["metrics"], artifact)
    _validate_expected_artifact_config(
        artifact.config,
        expected_config=expected_config,
        expected_n=expected_n,
        expected_layers1=expected_layers1,
        expected_layers2=expected_layers2,
    )
    return artifact


def _parameter_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid parameter name: {name!r}.")
    if name.startswith("__") or "::" in name or any(char in name for char in "/\\"):
        raise ValueError(f"Invalid parameter name: {name!r}.")
    return name


def _real_training_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating dtype, got {array.dtype}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return np.array(array, copy=True, order="C")


def _copy_parameter_tree(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a non-empty named mapping.")
    copied: dict[str, np.ndarray] = {}
    for raw_name, value in parameters.items():
        name = _parameter_name(raw_name)
        copied[name] = _real_training_array(value, f"parameter {name!r}")
    return copied


def _adam_step(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Adam step must be a scalar integer.")
    step = int(array.item())
    if step < 0:
        raise ValueError("Adam step must be non-negative.")
    return step


def _copy_optimizer_state(
    state: Mapping[str, Any], parameters: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("optimizer_state must be a mapping.")
    _require_exact_keys(state, {"step", "m", "v"}, "optimizer_state")
    if not isinstance(state["m"], Mapping) or not isinstance(state["v"], Mapping):
        raise TypeError("Adam m and v must be named mappings.")
    names = set(parameters)
    _require_exact_keys(state["m"], names, "Adam m")
    _require_exact_keys(state["v"], names, "Adam v")
    moments1: dict[str, np.ndarray] = {}
    moments2: dict[str, np.ndarray] = {}
    for name, parameter in parameters.items():
        first = _real_training_array(state["m"][name], f"Adam m {name!r}")
        second = _real_training_array(state["v"][name], f"Adam v {name!r}")
        if first.shape != parameter.shape or second.shape != parameter.shape:
            raise ValueError(f"Adam moments do not match parameter {name!r} shape.")
        if first.dtype != parameter.dtype or second.dtype != parameter.dtype:
            raise ValueError(f"Adam moments do not match parameter {name!r} dtype.")
        if np.any(second < 0.0):
            raise ValueError(f"Adam v {name!r} contains a negative value.")
        moments1[name] = first
        moments2[name] = second
    return {"step": _adam_step(state["step"]), "m": moments1, "v": moments2}


def _manifest(array: np.ndarray, archive_key: str) -> dict[str, Any]:
    return {
        "archive_key": archive_key,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }


def save_checkpoint(
    prefix: str | os.PathLike[str],
    parameters: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically save trainable parameters, matching Adam state, and metadata."""

    copied_parameters = _copy_parameter_tree(parameters)
    copied_optimizer = _copy_optimizer_state(optimizer_state, copied_parameters)
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    normalized_metadata = dict(_jsonable(metadata))
    _json_bytes(normalized_metadata)

    checkpoint_id = uuid.uuid4().hex
    step_array = np.asarray(copied_optimizer["step"], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        _CHECKPOINT_ID_KEY: np.asarray(checkpoint_id),
        _ADAM_STEP_KEY: step_array,
    }
    parameter_manifest: dict[str, Any] = {}
    first_manifest: dict[str, Any] = {}
    second_manifest: dict[str, Any] = {}
    for name in sorted(copied_parameters):
        parameter_key = f"parameter::{name}"
        first_key = f"adam_m::{name}"
        second_key = f"adam_v::{name}"
        arrays[parameter_key] = copied_parameters[name]
        arrays[first_key] = copied_optimizer["m"][name]
        arrays[second_key] = copied_optimizer["v"][name]
        parameter_manifest[name] = _manifest(arrays[parameter_key], parameter_key)
        first_manifest[name] = _manifest(arrays[first_key], first_key)
        second_manifest[name] = _manifest(arrays[second_key], second_key)

    npz_path, json_path = paired_paths(prefix)
    _atomic_write(npz_path, lambda handle: np.savez(handle, **arrays))
    npz_sha = _file_sha256(npz_path)
    record = {
        "schema_name": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_VERSION,
        "checkpoint_id": checkpoint_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "npz_sha256": npz_sha,
        "metadata": normalized_metadata,
        "parameter_manifest": parameter_manifest,
        "adam_manifest": {
            "step": _manifest(step_array, _ADAM_STEP_KEY),
            "m": first_manifest,
            "v": second_manifest,
        },
    }
    encoded = _json_bytes(record)
    _atomic_write(json_path, lambda handle: handle.write(encoded))
    return npz_path, json_path


def _load_manifest_array(
    archive: Any, manifest: Any, logical_name: str
) -> np.ndarray:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Manifest for {logical_name!r} must be an object.")
    _require_exact_keys(manifest, _MANIFEST_KEYS, f"manifest {logical_name}")
    archive_key = manifest["archive_key"]
    if not isinstance(archive_key, str) or archive_key not in archive.files:
        raise ValueError(f"Manifest archive key is invalid for {logical_name!r}.")
    array = np.array(archive[archive_key], copy=True)
    if list(array.shape) != manifest["shape"]:
        raise ValueError(f"Checkpoint shape mismatch for {logical_name!r}.")
    if str(array.dtype) != manifest["dtype"]:
        raise ValueError(f"Checkpoint dtype mismatch for {logical_name!r}.")
    if _array_sha256(array) != manifest["sha256"]:
        raise ValueError(f"Checkpoint array SHA256 mismatch for {logical_name!r}.")
    return array


def load_checkpoint(
    prefix: str | os.PathLike[str],
    *,
    expected_parameter_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> LoadedCheckpoint:
    """Load a checkpoint after exact key, checksum, shape, and finite checks."""

    npz_path, json_path = paired_paths(prefix)
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Checkpoint requires both {npz_path} and {json_path}.")
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint JSON: {json_path}.") from exc
    if not isinstance(record, Mapping):
        raise ValueError("Checkpoint metadata root must be a JSON object.")
    _require_exact_keys(
        record,
        {
            "schema_name",
            "schema_version",
            "checkpoint_id",
            "created_at_utc",
            "npz_sha256",
            "metadata",
            "parameter_manifest",
            "adam_manifest",
        },
        "checkpoint metadata",
    )
    if record["schema_name"] != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint schema name is unsupported.")
    if record["schema_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version: {record['schema_version']!r}."
        )
    checkpoint_id = record["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("Checkpoint ID must be a non-empty string.")
    npz_sha = _file_sha256(npz_path)
    if record["npz_sha256"] != npz_sha:
        raise ValueError(f"Checkpoint checksum mismatch for {npz_path}.")
    if not isinstance(record["metadata"], Mapping):
        raise ValueError("Checkpoint user metadata must be a JSON object.")
    parameter_manifest = record["parameter_manifest"]
    adam_manifest = record["adam_manifest"]
    if not isinstance(parameter_manifest, Mapping) or not parameter_manifest:
        raise ValueError("Checkpoint parameter_manifest must be non-empty.")
    if not isinstance(adam_manifest, Mapping):
        raise ValueError("Checkpoint adam_manifest must be an object.")
    _require_exact_keys(adam_manifest, {"step", "m", "v"}, "Adam manifest")
    if not isinstance(adam_manifest["m"], Mapping) or not isinstance(
        adam_manifest["v"], Mapping
    ):
        raise ValueError("Adam moment manifests must be objects.")
    names = set(parameter_manifest)
    for name in names:
        _parameter_name(name)
    _require_exact_keys(adam_manifest["m"], names, "Adam m manifest")
    _require_exact_keys(adam_manifest["v"], names, "Adam v manifest")

    all_manifests = [adam_manifest["step"]]
    all_manifests.extend(parameter_manifest.values())
    all_manifests.extend(adam_manifest["m"].values())
    all_manifests.extend(adam_manifest["v"].values())
    archive_keys: list[str] = []
    for manifest in all_manifests:
        if not isinstance(manifest, Mapping):
            raise ValueError("Checkpoint contains a non-object array manifest.")
        _require_exact_keys(manifest, _MANIFEST_KEYS, "checkpoint array manifest")
        if not isinstance(manifest["archive_key"], str):
            raise ValueError("Checkpoint archive keys must be strings.")
        archive_keys.append(manifest["archive_key"])
    if len(archive_keys) != len(set(archive_keys)):
        raise ValueError("Checkpoint manifests reuse an archive key.")
    expected_archive_keys = set(archive_keys) | {_CHECKPOINT_ID_KEY}

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_archive_keys:
                missing = sorted(expected_archive_keys - set(archive.files))
                extra = sorted(set(archive.files) - expected_archive_keys)
                raise ValueError(
                    f"Checkpoint NPZ keys differ; missing={missing}, extra={extra}."
                )
            identifier = np.array(archive[_CHECKPOINT_ID_KEY], copy=True)
            if identifier.shape != () or str(identifier.item()) != checkpoint_id:
                raise ValueError("Checkpoint NPZ and JSON identifiers do not match.")
            step_array = _load_manifest_array(
                archive, adam_manifest["step"], "Adam step"
            )
            parameters = {
                name: _load_manifest_array(
                    archive, parameter_manifest[name], f"parameter {name}"
                )
                for name in sorted(names)
            }
            moment1 = {
                name: _load_manifest_array(
                    archive, adam_manifest["m"][name], f"Adam m {name}"
                )
                for name in sorted(names)
            }
            moment2 = {
                name: _load_manifest_array(
                    archive, adam_manifest["v"][name], f"Adam v {name}"
                )
                for name in sorted(names)
            }
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Checkpoint"):
            raise
        raise ValueError(f"Invalid checkpoint NPZ: {npz_path}.") from exc

    step = _adam_step(step_array)
    validated_parameters = _copy_parameter_tree(parameters)
    validated_optimizer = _copy_optimizer_state(
        {"step": step_array, "m": moment1, "v": moment2}, validated_parameters
    )
    if expected_parameter_shapes is not None:
        if not isinstance(expected_parameter_shapes, Mapping):
            raise TypeError("expected_parameter_shapes must be a mapping.")
        _require_exact_keys(
            expected_parameter_shapes, set(validated_parameters), "expected shapes"
        )
        for name, array in validated_parameters.items():
            try:
                shape = tuple(int(item) for item in expected_parameter_shapes[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid expected shape for {name!r}.") from exc
            if array.shape != shape:
                raise ValueError(
                    f"Checkpoint parameter {name!r} has shape {array.shape}; "
                    f"expected {shape}."
                )
    return LoadedCheckpoint(
        parameters=validated_parameters,
        optimizer_state=validated_optimizer,
        metadata=dict(record["metadata"]),
        checkpoint_id=checkpoint_id,
        npz_sha256=npz_sha,
    )


__all__ = [
    "ARTIFACT_VERSION",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_VERSION",
    "JAX_ARTIFACT_FORMAT",
    "LEGACY_ARTIFACT_FORMAT",
    "LoadedCheckpoint",
    "NonlinearArtifact",
    "load_checkpoint",
    "load_nonlinear_artifact",
    "nonlinear_parameter_count",
    "paired_paths",
    "save_checkpoint",
    "save_nonlinear_artifact",
]
