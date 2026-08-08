"""Unit tests for target-independent domain value objects."""

from __future__ import annotations

import unittest

import numpy as np

from gate_experiment.models.parameters import ParameterSet
from gate_experiment.models.points import ExperimentPoint, PointSet, SweepAxis
from gate_experiment.targets.base import IdealOperation
from gate_experiment.targets.composite import UniversalCompositePulse, prx_unitary
from gate_experiment.targets.universal import universal_composite_pulses


class ParameterAndTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = universal_composite_pulses()["X5a"]

    def test_parameter_defaults_are_resolved_without_mutating_input(self) -> None:
        parameters = ParameterSet({"amplitude_error": 0.02})

        resolved = self.target.resolve_parameters(parameters)

        self.assertEqual(parameters, ParameterSet({"amplitude_error": 0.02}))
        self.assertEqual(resolved.require_float("amplitude_error"), 0.02)
        self.assertEqual(resolved.require_float("detuning_hz"), 0.0)

    def test_unknown_target_parameter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported parameters"):
            self.target.validate(ParameterSet({"amplitdue_error": 0.02}))

    def test_numpy_scalars_are_normalized_for_notebook_sweeps(self) -> None:
        parameters = ParameterSet(
            {
                "integer": np.int64(3),
                "floating": np.float64(0.25),
            }
        )

        self.assertIs(type(parameters["integer"]), int)
        self.assertIs(type(parameters["floating"]), float)

    def test_ideal_operation_rejects_nonunitary_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unitary"):
            IdealOperation(np.ones((2, 2)), arity=1, name="bad")

    def test_ideal_operation_copies_and_freezes_unitary(self) -> None:
        original = np.eye(2, dtype=complex)
        ideal = IdealOperation(original, arity=1, name="identity")
        original[0, 0] = 0

        self.assertEqual(ideal.unitary[0, 0], 1)
        with self.assertRaises(ValueError):
            ideal.unitary[0, 0] = 0

    def test_universal_registry_contains_expected_targets(self) -> None:
        registry = universal_composite_pulses()

        self.assertEqual(set(registry), {"X1", "X5a", "X5b", "X9a", "X9b", "H1"})
        self.assertEqual(registry["X9a"].metadata()["pulse_count"], 9)

    def test_custom_composite_sequence_uses_same_target_implementation(self) -> None:
        custom = UniversalCompositePulse.for_x_rotation(
            "custom",
            target_angle=np.pi,
            constituent_angle=np.pi,
            phases=(0.1, 0.2, 0.3),
        )

        self.assertEqual(custom.operation_id, "custom")
        self.assertEqual(len(custom.phases), 3)
        np.testing.assert_allclose(custom.ideal_operation.unitary, prx_unitary(np.pi, 0))


class PointSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = universal_composite_pulses()["X1"]

    def test_cartesian_sweep_has_deterministic_order(self) -> None:
        points = PointSet.cartesian(
            targets=(self.target,),
            axes=(
                SweepAxis("amplitude_error", (-0.1, 0.1)),
                SweepAxis("detuning_hz", (-1.0, 1.0)),
            ),
        )

        self.assertEqual(len(points), 4)
        self.assertEqual(
            [dict(point.parameters) for point in points],
            [
                {"amplitude_error": -0.1, "detuning_hz": -1.0},
                {"amplitude_error": -0.1, "detuning_hz": 1.0},
                {"amplitude_error": 0.1, "detuning_hz": -1.0},
                {"amplitude_error": 0.1, "detuning_hz": 1.0},
            ],
        )

    def test_zipped_sweep_pairs_axis_values(self) -> None:
        points = PointSet.zipped(
            targets=(self.target,),
            axes=(
                SweepAxis("amplitude_error", (-0.1, 0.1)),
                SweepAxis("detuning_hz", (-1.0, 1.0)),
            ),
        )

        self.assertEqual(len(points), 2)
        self.assertEqual(dict(points[1].parameters), {"amplitude_error": 0.1, "detuning_hz": 1.0})

    def test_fixed_parameter_cannot_overlap_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            PointSet.cartesian(
                targets=(self.target,),
                axes=(SweepAxis("detuning_hz", (0.0,)),),
                fixed={"detuning_hz": 1.0},
            )

    def test_duplicate_explicit_point_ids_are_rejected(self) -> None:
        point = ExperimentPoint("same", self.target, ParameterSet())

        with self.assertRaisesRegex(ValueError, "unique"):
            PointSet.explicit((point, point))


if __name__ == "__main__":
    unittest.main()
