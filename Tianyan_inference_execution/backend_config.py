"""Runtime device selection for local simulation and TianYan execution.

The default is deliberately ``default.qubit``.  A CQLib/TianYan device is
created only when ``use_tianyan=True`` (or the corresponding CLI flag is
provided by a runner), so importing model modules never submits a cloud job.
"""

from dataclasses import dataclass
from copy import deepcopy
import os
from pathlib import Path
import re
from types import MethodType
import warnings

import pennylane as qml

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - cqlib-adapter installs python-dotenv
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


HARDWARE_BACKENDS = frozenset(
    {
        "tianyan24",
        "tianyan-287",
        "tianyan504",
        "tianyan176-2",
        "tianyan176",
    }
)
SUPPORTED_TIANYAN_BACKENDS = HARDWARE_BACKENDS | frozenset(
    {
        "default",
        "tianyan_sw",
        "tianyan_s",
        "tianyan_tn",
        "tianyan_tnn",
        "tianyan_sa",
        "tianyan_swn",
    }
)

# CQLib's full readout calibration constructs dense 2**n probability vectors
# and a 2**n-by-2**n correction matrix.  Keep it for the small validation
# stages, but use sparse empirical shot frequencies for the wide smoke stages.
MAX_DENSE_READOUT_CALIBRATION_QUBITS = 12


@dataclass(frozen=True)
class BackendConfig:
    """Execution backend settings shared by model runners."""

    use_tianyan: bool = False
    backend: str = "tianyan-287"
    shots: int = 10
    login_key_env: str = "TIANYAN_LOGIN_KEY"
    excluded_physical_qubits: tuple = ()

    def __post_init__(self):
        if self.shots <= 0:
            raise ValueError("shots must be positive.")
        if self.backend not in SUPPORTED_TIANYAN_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_TIANYAN_BACKENDS))
            raise ValueError(
                f"Unsupported TianYan backend {self.backend!r}. "
                f"Choose one of: {supported}."
            )
        normalized_exclusions = tuple(
            str(qubit).strip().upper()
            for qubit in self.excluded_physical_qubits
            if str(qubit).strip()
        )
        invalid = [
            qubit
            for qubit in normalized_exclusions
            if re.fullmatch(r"Q\d+", qubit) is None
        ]
        if invalid:
            raise ValueError(
                "Excluded physical qubits must look like Q24; got: "
                + ", ".join(invalid)
            )
        object.__setattr__(
            self,
            "excluded_physical_qubits",
            tuple(dict.fromkeys(normalized_exclusions)),
        )

    @property
    def is_hardware(self):
        return self.use_tianyan and self.backend in HARDWARE_BACKENDS

    @property
    def label(self):
        if not self.use_tianyan:
            return "local/default.qubit (analytic)"
        device_kind = "hardware" if self.is_hardware else "CQLib simulator"
        excluded = ""
        if self.excluded_physical_qubits:
            excluded = ", excluded=" + ",".join(self.excluded_physical_qubits)
        return (
            f"TianYan/{self.backend} "
            f"({device_kind}, shots={self.shots}{excluded})"
        )


def install_cqlib_openqasm_compatibility():
    """Restore the OpenQASM method expected by current CQLib adapters."""

    quantum_script = qml.tape.QuantumScript
    if hasattr(quantum_script, "to_openqasm"):
        return

    def to_openqasm(self, wires=None, rotations=True, measure_all=True, precision=None):
        return qml.to_openqasm(
            self,
            wires=wires,
            rotations=rotations,
            measure_all=measure_all,
            precision=precision,
        )

    quantum_script.to_openqasm = to_openqasm


def install_tianyan_287_metadata_compatibility():
    """Ignore unavailable calibration metadata when creating Tianyan-287.

    Some cqlib-adapter releases multiply the reported T1/T2 values by a unit
    which is ``None`` in the Tianyan-287 backend configuration.  Calibration
    properties are optional Qiskit Target metadata and are not needed for
    circuit transpilation or submission, so fall back to no qubit properties
    only for that specific adapter failure.
    """

    try:
        import cqlib_adapter.qiskit_ext.tianyan_backend as backend_module
    except ModuleNotFoundError:
        # Let qml.device produce its normal missing-plugin error below.
        return

    backend_class = backend_module.TianYanQuantumBackend
    original = backend_class._make_qubit_properties
    if getattr(original, "_tianyan_287_none_unit_compatibility", False):
        return

    def make_qubit_properties_or_none(self):
        try:
            return original(self)
        except TypeError as error:
            message = str(error)
            if "float" not in message or "NoneType" not in message:
                raise
            return None

    make_qubit_properties_or_none._tianyan_287_none_unit_compatibility = True
    backend_class._make_qubit_properties = make_qubit_properties_or_none


def install_physical_qubit_exclusions(device, excluded_qubits):
    """Make CQLib's transpiler omit physical qubits rejected by submission.

    TianYan's downloaded mapping configuration and its submit-time availability
    check can briefly disagree.  CQLib builds its coupling graph from
    ``download_config()['disabledQubits']``.  Extend that field on this backend
    instance so ``transpile_qcis`` selects a different valid physical layout.
    """

    if not excluded_qubits:
        return

    backend = getattr(device, "cqlib_backend", None)
    if backend is None:
        executor = getattr(device, "circuit_executor", None)
        backend = getattr(executor, "cqlib_backend", None)
    if backend is None or not hasattr(backend, "download_config"):
        raise RuntimeError(
            "The installed CQLib device does not expose download_config; "
            "physical-qubit exclusion cannot be applied."
        )

    original_download_config = backend.download_config
    excluded_qubits = tuple(excluded_qubits)

    def download_config_with_exclusions(self, *args, **kwargs):
        machine_config = deepcopy(original_download_config(*args, **kwargs))
        disabled_value = machine_config.get("disabledQubits", "")
        if isinstance(disabled_value, str):
            disabled = [item for item in disabled_value.split(",") if item]
        else:
            disabled = list(disabled_value or ())
        disabled.extend(excluded_qubits)
        machine_config["disabledQubits"] = ",".join(dict.fromkeys(disabled))
        return machine_config

    backend.download_config = MethodType(download_config_with_exclusions, backend)


def install_sparse_readout_fallback(
    device,
    wire_count,
    max_dense_qubits=MAX_DENSE_READOUT_CALIBRATION_QUBITS,
):
    """Avoid CQLib's exponentially large dense readout calibration.

    CQLib's PennyLane hardware executor always requests full readout
    calibration.  That path allocates objects with 2**n entries (and a dense
    correction matrix), which is impossible for the 34--92-wire smoke stages.
    For wide circuits, query the already-submitted job without readout
    calibration and construct only the probabilities observed in the finite
    shot sample.  Also fall back if CQLib unexpectedly overflows on a smaller
    circuit.
    """

    backend = getattr(device, "cqlib_backend", None)
    if backend is None:
        executor = getattr(device, "circuit_executor", None)
        backend = getattr(executor, "cqlib_backend", None)
    if backend is None or not hasattr(backend, "query_experiment"):
        raise RuntimeError(
            "The installed CQLib device does not expose query_experiment; "
            "sparse hardware readout cannot be installed."
        )

    original_query_experiment = backend.query_experiment
    force_sparse = wire_count > max_dense_qubits

    def sparse_probabilities(results):
        from cqlib.utils.laboratory_utils import LaboratoryUtils

        laboratory_utils = LaboratoryUtils()
        for item in results:
            basis_list = laboratory_utils.readout_data_to_state_probabilities(item)
            item["probability"] = laboratory_utils.original_onversion_part(
                basis_list
            )
        return results

    def query_experiment_with_sparse_fallback(self, *args, **kwargs):
        calibration_requested = bool(kwargs.get("readout_calibration", False))
        if not calibration_requested:
            return original_query_experiment(*args, **kwargs)

        if not force_sparse:
            try:
                return original_query_experiment(*args, **kwargs)
            except (MemoryError, OverflowError):
                pass

        raw_kwargs = dict(kwargs)
        raw_kwargs["readout_calibration"] = False
        results = original_query_experiment(*args, **raw_kwargs)
        warnings.warn(
            "Dense CQLib readout calibration was skipped; using sparse "
            "uncalibrated shot probabilities.",
            RuntimeWarning,
            stacklevel=2,
        )
        return sparse_probabilities(results)

    backend.query_experiment = MethodType(
        query_experiment_with_sparse_fallback,
        backend,
    )


def create_device(wires, config):
    """Create a local PennyLane device or an explicitly requested CQLib device."""

    if not isinstance(config, BackendConfig):
        raise TypeError("config must be a BackendConfig instance.")

    if not config.use_tianyan:
        return qml.device("default.qubit", wires=wires, shots=None)

    install_cqlib_openqasm_compatibility()
    if config.backend == "tianyan-287":
        install_tianyan_287_metadata_compatibility()
    login_key = os.environ.get(config.login_key_env, "").strip()
    if config.backend != "default" and not login_key:
        raise RuntimeError(
            f"Missing TianYan connection key. Set {config.login_key_env} "
            "before creating the CQLib device."
        )

    device_kwargs = {
        "wires": wires,
        "shots": config.shots,
        "cqlib_backend_name": config.backend,
    }
    if login_key:
        device_kwargs["login_key"] = login_key
    device = qml.device("cqlib.device", **device_kwargs)

    install_physical_qubit_exclusions(
        device,
        config.excluded_physical_qubits,
    )
    if config.is_hardware:
        install_sparse_readout_fallback(device, wires)

    # Older cqlib-adapter releases route hardware through the private
    # _is_tianyan_hardware() predicate and may not know about Tianyan-287.
    # Newer releases discover hardware dynamically from the TianYan API and no
    # longer expose that hook.  Patch only the legacy implementation; successful
    # construction of a newer device means its API-based routing already
    # accepted the backend.
    if config.backend == "tianyan-287":
        original_hardware_check = getattr(device, "_is_tianyan_hardware", None)
        if original_hardware_check is not None:

            def includes_tianyan_287(self):
                return (
                    self.machine_name == "tianyan-287"
                    or original_hardware_check()
                )

            device._is_tianyan_hardware = MethodType(
                includes_tianyan_287,
                device,
            )

    return device


def add_backend_arguments(parser):
    """Add the common local/TianYan command-line switches to a parser."""

    parser.add_argument(
        "--tianyan",
        action="store_true",
        help=(
            "Use cqlib.device. Without this flag the runner uses the local "
            "default.qubit simulator."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_TIANYAN_BACKENDS),
        default="tianyan-287",
        help="CQLib backend used with --tianyan.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10,
        help="Finite shots used by the CQLib/TianYan backend.",
    )
    parser.add_argument(
        "--exclude-qubits",
        default="",
        help=(
            "Comma-separated unavailable physical TianYan qubits to exclude "
            "during CQLib mapping, for example Q24 or Q24,Q37."
        ),
    )
    return parser


def backend_config_from_args(args):
    """Build :class:`BackendConfig` from arguments added above."""

    return BackendConfig(
        use_tianyan=bool(args.tianyan),
        backend=args.backend,
        shots=args.shots,
        excluded_physical_qubits=tuple(
            item.strip()
            for item in args.exclude_qubits.split(",")
            if item.strip()
        ),
    )
