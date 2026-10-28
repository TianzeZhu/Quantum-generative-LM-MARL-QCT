"""Public API for the corrected pure-JAX state-vector simulator.

All numerical logic lives in :mod:`statevector_jax_impl`.  Keeping this
module as a thin facade guarantees identical behavior for package imports and
direct script execution, without import-time monkeypatching.
"""

if __package__:
    from .statevector_jax_impl import *
    from .statevector_jax_impl import __all__ as _IMPLEMENTATION_EXPORTS
else:
    from statevector_jax_impl import *
    from statevector_jax_impl import __all__ as _IMPLEMENTATION_EXPORTS


__all__ = list(_IMPLEMENTATION_EXPORTS)
