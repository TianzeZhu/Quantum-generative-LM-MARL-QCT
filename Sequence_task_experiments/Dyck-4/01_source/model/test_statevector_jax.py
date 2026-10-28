"""Numerical equivalence and differentiability tests for the JAX fast path."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

MODULE_DIR = pathlib.Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import jax
import jax.numpy as jnp
import pennylane as qml

import circuits_jax
import statevector_jax as fast
import statevector_jax_impl as implementation


class StatevectorEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(104)
        cls.unitary_params = jnp.asarray(rng.normal(0, 0.2, 63))
        cls.svd_params = jnp.asarray(rng.normal(0, 0.2, 134))
        cls.nonlinear_params = jnp.asarray(rng.normal(0, 0.2, 126))
        cls.entangling = jnp.asarray(rng.normal(0, 0.2, (1, 7, 3)))

        def random_state():
            raw = rng.normal(size=8) + 1j * rng.normal(size=8)
            return jnp.asarray(raw / np.linalg.norm(raw))

        cls.state = random_state()
        cls.query = random_state()
        cls.key = random_state()
        cls.accumulator = random_state()
        cls.value = random_state()

    def test_parameter_counts_restore_scan_su8(self):
        self.assertEqual(
            circuits_jax.unitary_part([], range(3), 1, count_only=True), 63
        )
        self.assertEqual(
            circuits_jax.SVD_circuit(
                [], range(3), 3, 1, count_only=True
            ),
            134,
        )
        self.assertEqual(fast.nonlinear_parameter_count(), 126)

    def test_su8_matches_pennylane_reference(self):
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device, interface="jax", diff_method="backprop")
        def reference(state, params):
            qml.StatePrep(state, wires=range(3), normalize=False)
            circuits_jax.unitary_part(params, range(3), 1)
            return qml.state()

        expected = reference(self.state, self.unitary_params)
        actual = fast.apply_unitary_part(
            self.unitary_params, self.state, range(3), 3, 1
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_svd_state_returns_and_matches_fast_path(self):
        expected, expected_success = circuits_jax.SVD_circuit_state(
            self.svd_params,
            self.state,
            range(3),
            3,
            1,
            return_success_probability=True,
        )
        actual, actual_success = fast.apply_svd(self.svd_params, self.state)
        # Pure states may differ by a global phase.
        overlap = jnp.abs(jnp.vdot(expected, actual))
        self.assertAlmostEqual(float(overlap), 1.0, places=9)
        self.assertAlmostEqual(
            float(actual_success), float(expected_success), places=9
        )

    def test_nonlinear_selected_branch_matches_reference(self):
        device = qml.device("default.qubit", wires=4)

        @qml.qnode(device, interface="jax", diff_method="backprop")
        def reference(state, params):
            qml.StatePrep(state, wires=range(3), normalize=False)
            qml.Hadamard(wires=3)
            circuits_jax.unitary_part(params[:63], range(3), 1)
            qml.MultiControlledX(
                wires=[3, 0, 1, 2], control_values=[True, True, True]
            )
            circuits_jax.unitary_part(params[63:], [0, 1, 3], 1)
            qml.MultiControlledX(
                wires=[2, 0, 1, 3], control_values=[True, True, True]
            )
            qml.Projector([0], wires=3)
            return qml.state()

        full = reference(self.state, self.nonlinear_params)
        selected = jnp.reshape(full, (8, 2))[:, 0]
        expected = selected / jnp.linalg.norm(selected)
        actual, success = fast.apply_nonlinear(
            self.nonlinear_params, self.state
        )
        self.assertGreater(float(success), 0)
        overlap = jnp.abs(jnp.vdot(expected, actual))
        self.assertAlmostEqual(float(overlap), 1.0, places=9)

    def test_strongly_entangling_matches_pennylane(self):
        rng = np.random.default_rng(51)
        raw = rng.normal(size=128) + 1j * rng.normal(size=128)
        state = jnp.asarray(raw / np.linalg.norm(raw))
        device = qml.device("default.qubit", wires=7)

        @qml.qnode(device, interface="jax", diff_method="backprop")
        def reference(input_state, weights):
            qml.StatePrep(input_state, wires=range(7), normalize=False)
            qml.StronglyEntanglingLayers(weights, wires=range(7))
            return qml.state()

        expected = reference(state, self.entangling)
        actual = fast.apply_strongly_entangling(state, self.entangling)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_pair_reduction_matches_full_thirteen_wire_state(self):
        device = qml.device("default.qubit", wires=13)

        @qml.qnode(device, interface="jax", diff_method="backprop")
        def full_pair(query, key, accumulator, value, weights):
            qml.StatePrep(key, wires=[0, 1, 2], normalize=False)
            qml.StatePrep(query, wires=[3, 4, 5], normalize=False)
            qml.StatePrep(accumulator, wires=[7, 8, 9], normalize=False)
            qml.StatePrep(value, wires=[10, 11, 12], normalize=False)
            qml.Hadamard(wires=6)
            for left, right in zip([0, 1, 2], [3, 4, 5]):
                qml.CSWAP(wires=[6, left, right])
            qml.Hadamard(wires=6)
            qml.StronglyEntanglingLayers(
                weights, wires=[6, 7, 8, 9, 10, 11, 12]
            )
            return qml.probs(wires=[6, 7, 8, 9])

        expected = full_pair(
            self.query,
            self.key,
            self.accumulator,
            self.value,
            self.entangling,
        )[:8]
        actual = fast.pair_swap_zero_row(
            self.query,
            self.accumulator,
            self.key,
            self.value,
            self.entangling,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_jit_vmap_and_gradient_are_finite(self):
        keys = jnp.stack((self.key, self.query))
        values = jnp.stack((self.value, self.accumulator))

        def objective(weights):
            rows = jax.vmap(
                lambda key, value: fast.pair_swap_zero_row(
                    self.query,
                    self.accumulator,
                    key,
                    value,
                    weights,
                )
            )(keys, values)
            return jnp.sum(rows[:, 1])

        eager_value, eager_grad = jax.value_and_grad(objective)(self.entangling)
        jit_value, jit_grad = jax.jit(jax.value_and_grad(objective))(
            self.entangling
        )
        np.testing.assert_allclose(jit_value, eager_value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(jit_grad, eager_grad, rtol=1e-9, atol=1e-9)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jit_grad))))
        self.assertGreater(float(jnp.linalg.norm(jit_grad)), 0)


    def test_real_initial_state_preserves_rx_phase(self):
        """A real embedding must be promoted before an RX introduces phase."""
        state = jnp.zeros(8, dtype=jnp.float64).at[0].set(1.0)
        theta = jnp.asarray(0.37, dtype=jnp.float64)
        actual = implementation.apply_position_encoding(
            state, 0, jnp.asarray([theta, 0.0, 0.0])
        )
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = np.cos(float(theta) / 2.0)
        expected[4] = -1j * np.sin(float(theta) / 2.0)
        self.assertTrue(jnp.issubdtype(actual.dtype, jnp.complexfloating))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_zero_state_normalization_has_finite_gradient(self):
        zero = jnp.zeros(8, dtype=jnp.float64)

        def first_component(value):
            return jnp.real(implementation.normalize_state(value)[0])

        normalized = implementation.normalize_state(zero)
        gradient = jax.grad(first_component)(zero)
        np.testing.assert_allclose(normalized, jnp.eye(1, 8)[0])
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        np.testing.assert_allclose(gradient, jnp.zeros_like(gradient))


    def test_public_api_is_direct_implementation_export(self):
        for name in fast.__all__:
            self.assertTrue(hasattr(fast, name), name)
        self.assertIs(fast.apply_gate, implementation.apply_gate)
        self.assertIs(fast.pair_swap_zero_row, implementation.pair_swap_zero_row)

    def test_softmax_and_sigmoid_may_use_independent_depths(self):
        probabilities = fast.output_probabilities(
            jnp.full(8, 1.0 / 8.0),
            self.unitary_params,
            self.svd_params,
            self.nonlinear_params,
            self.svd_params,
            unitary_layers=1,
            softmax_layers1=1,
            softmax_layers2=0,
            sigmoid_layers1=1,
            sigmoid_layers2=1,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(probabilities))))
        self.assertAlmostEqual(float(jnp.sum(probabilities)), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()




