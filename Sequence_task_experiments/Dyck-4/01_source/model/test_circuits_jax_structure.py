"""Structural regression tests for the PennyLane reference ansatz."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_DIR = pathlib.Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import circuits_jax


class CircuitStructureRegressionTests(unittest.TestCase):
    def test_n7_paper_bridge_and_nominal_parameter_count(self):
        expected = [
            ("CRX", (0, 4)),
            ("CRY", (1, 5)),
            ("CRZ", (2, 6)),
            ("CRX", (3, 6)),
        ]
        self.assertEqual(circuits_jax._bridge_specs(7, "long_range"), expected)
        self.assertEqual(
            circuits_jax.unitary_part(
                [], range(7), 4, "stride2", "long_range", count_only=True
            ),
            1032,
        )

    def test_n7_queue_covers_every_wire_and_consumes_each_parameter_once(self):
        events = []

        def record_su16(parameters, wires):
            events.append(
                ("SU16", tuple(wires), tuple(int(x) for x in parameters))
            )

        def bridge_recorder(name):
            def record(parameter, wires):
                events.append(
                    (name, tuple(wires), (int(np.asarray(parameter).item()),))
                )

            return record

        bridge_gates = {
            name: bridge_recorder(name) for name in ("CRX", "CRY", "CRZ")
        }
        with mock.patch.object(circuits_jax, "SU16", new=record_su16), mock.patch.object(
            circuits_jax, "_BRIDGE_GATES", new=bridge_gates
        ):
            circuits_jax.unitary_part(
                np.arange(1032),
                range(7),
                4,
                "stride2",
                "long_range",
            )

        expected_kinds = []
        for layer_index in range(4):
            expected_kinds.append("SU16")
            if layer_index < 3:
                expected_kinds.extend(("CRX", "CRY", "CRZ", "CRX"))
        self.assertEqual([event[0] for event in events], expected_kinds)
        self.assertEqual(
            [event[1] for event in events if event[0] == "SU16"],
            [
                (0, 1, 2, 3),
                (2, 3, 4, 5),
                (0, 1, 2, 3),
                (2, 3, 4, 5),
            ],
        )

        used_wires = set()
        used_parameters = []
        for _, wires, parameter_indices in events:
            used_wires.update(wires)
            used_parameters.extend(parameter_indices)
        self.assertEqual(used_wires, set(range(7)))
        self.assertEqual(used_parameters, list(range(1032)))

        for gap_index in range(3):
            start = gap_index * 5 + 1
            gap_wires = set()
            for event in events[start : start + 4]:
                gap_wires.update(event[1])
            self.assertEqual(gap_wires, set(range(7)))

    def test_n6_legacy_bridge_layout_is_restored(self):
        expected = [
            ("CRX", (0, 4)),
            ("CRY", (0, 2)),
            ("CRZ", (3, 5)),
            ("CRX", (1, 5)),
        ]
        self.assertEqual(circuits_jax._bridge_specs(6, "long_range"), expected)
        self.assertEqual(
            set(wire for _, pair in expected for wire in pair), set(range(6))
        )

    def test_other_paper_width_parameter_counts_are_unchanged(self):
        self.assertEqual(
            circuits_jax.unitary_part(
                [], range(10), 8, "stride2", "long_range", count_only=True
            ),
            4115,
        )
        self.assertEqual(
            circuits_jax.unitary_part(
                [], range(14), 13, "stride2", "long_range", count_only=True
            ),
            10029,
        )

    def test_uncovered_large_width_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "untouched"):
            circuits_jax.unitary_part(
                [], range(7), 1, "stride2", "long_range", count_only=True
            )
        with self.assertRaisesRegex(ValueError, "untouched"):
            circuits_jax.unitary_part(
                [], range(5), 4, "stride2", "long_range", count_only=True
            )

    def test_n3_small_width_path_does_not_consult_bridges(self):
        with mock.patch.object(
            circuits_jax,
            "_bridge_specs",
            side_effect=AssertionError("small-width path consulted bridges"),
        ):
            self.assertEqual(
                circuits_jax.unitary_part([], range(3), 1, count_only=True), 63
            )
            self.assertEqual(
                circuits_jax.unitary_part([], range(3), 4, count_only=True), 252
            )
            self.assertEqual(
                circuits_jax.SVD_circuit(
                    [], range(3), 3, 1, count_only=True
                ),
                134,
            )


if __name__ == "__main__":
    unittest.main()
