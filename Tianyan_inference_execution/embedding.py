import pennylane as qml
import numpy as onp
from pennylane import numpy as np


def embedding_dimension(num_qubits):
    """Return the amplitude vector length for a register with num_qubits wires."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    return 2 ** int(num_qubits)


def embedding_parameter_dimension(num_qubits):
    """Return the number of unconstrained trainable values per embedding row."""
    return embedding_dimension(num_qubits) - 1


def _as_rng(seed_or_rng=None):
    if isinstance(seed_or_rng, onp.random.Generator):
        return seed_or_rng
    return onp.random.default_rng(seed_or_rng)


def random_embedding_logits(
    seed_or_rng=None,
    vocab_size=1,
    num_qubits=1,
    dtype=np.float64,
    requires_grad=True,
):
    """Initialize the unconstrained, trainable parameters of an embedding table."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    rng = _as_rng(seed_or_rng)
    parameter_dim = embedding_parameter_dimension(num_qubits)
    return np.array(
        rng.normal(size=(vocab_size, parameter_dim)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def embedding_table_from_logits(embedding_logits):
    """Map trainable logits to real, normalized amplitude vectors.

    Each row has ``2**n - 1`` trainable logits. The first amplitudes are
    bounded with ``tanh``; the final amplitude is computed as
    ``sqrt(1 - sum(prefix**2))``. Consequently every output row has unit norm.
    """
    embedding_logits = np.asarray(embedding_logits)
    if embedding_logits.ndim != 2:
        raise ValueError("embedding_logits must be a rank-2 array.")
    if embedding_logits.shape[1] <= 0:
        raise ValueError("embedding_logits must contain at least one column.")

    prefix_dim = embedding_logits.shape[1]
    feature_dim = prefix_dim + 1
    if feature_dim & (feature_dim - 1):
        raise ValueError("embedding_logits width must be 2**n - 1 for some n.")

    prefix_bound = 0.8 / np.sqrt(prefix_dim)
    prefix = prefix_bound * np.tanh(embedding_logits)
    final_amplitude = np.sqrt(1.0 - np.sum(prefix**2, axis=-1, keepdims=True))
    return np.concatenate((prefix, final_amplitude), axis=-1)


def random_embedding_table(
    seed_or_rng=None,
    vocab_size=1,
    num_qubits=1,
    dtype=np.float64,
    requires_grad=True,
):
    """Generate the normalized amplitude table corresponding to random logits."""
    logits = random_embedding_logits(
        seed_or_rng,
        vocab_size,
        num_qubits,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return embedding_table_from_logits(logits)


def quantum_embedding(index, embedding_logits, wires):
    """
    Prepare one embedding vector on the given wires using amplitude encoding.

    Args:
        index: Token/state index used to select one embedding row.
        embedding_logits: Trainable array with shape
            (vocab_size, 2**len(wires) - 1).
        wires: Target wires for the prepared quantum state.
    """
    if embedding_logits is None:
        raise ValueError("embedding_logits must be provided.")
    if wires is None or len(wires) == 0:
        raise ValueError("wires must be a non-empty sequence.")

    embedding_logits = np.asarray(embedding_logits)
    expected_width = embedding_parameter_dimension(len(wires))
    if embedding_logits.ndim != 2 or embedding_logits.shape[1] != expected_width:
        raise ValueError(
            f"embedding_logits must have shape (vocab_size, {expected_width}) "
            f"for {len(wires)} wires."
        )

    vector = embedding_table_from_logits(embedding_logits)[int(index)]
    # Use an explicit gate-based state preparation. Multiple AmplitudeEmbedding
    # (StatePrep) operations on disjoint Q/K/V registers interact badly with
    # mid-circuit measurements in PennyLane's wire-remapping transform. The
    # Mottonen template prepares the same normalized amplitudes and composes
    # safely across all independent register copies.
    qml.MottonenStatePreparation(state_vector=vector, wires=wires)
