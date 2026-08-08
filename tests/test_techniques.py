"""Pure numerical and planning tests for built-in techniques."""

from __future__ import annotations

import unittest

import numpy as np

from iqm_composite_experiment.models.acquisition import MeasurementData, ProbabilityData
from iqm_composite_experiment.models.parameters import ParameterSet
from iqm_composite_experiment.models.points import ExperimentPoint
from iqm_composite_experiment.targets.universal import universal_composite_pulses
from iqm_composite_experiment.techniques.base import TechniqueContext
from iqm_composite_experiment.techniques.ramsey import (
    TransitionPhaseRamsey,
    fit_fringe,
    ideal_ramsey_p1,
)
from iqm_composite_experiment.techniques.randomized_benchmarking import (
    InterleavedRandomizedBenchmarking,
    RBSettings,
    clifford_group,
    fit_decay,
)
from iqm_composite_experiment.techniques.tomography import (
    ProcessTomography,
    reconstruct_ptm,
    tomography_metrics,
)


class TomographyTests(unittest.TestCase):
    def test_identity_probabilities_reconstruct_identity_ptm(self) -> None:
        coordinates = tuple(
            (state, basis)
            for state in ("0", "1", "+x", "+y")
            for basis in ("x", "y", "z")
        )
        output_vectors = {
            "0": np.array([0.0, 0.0, 1.0]),
            "1": np.array([0.0, 0.0, -1.0]),
            "+x": np.array([1.0, 0.0, 0.0]),
            "+y": np.array([0.0, 1.0, 0.0]),
        }
        axis = {"x": 0, "y": 1, "z": 2}
        p1 = np.array(
            [(1 - output_vectors[state][axis[basis]]) / 2 for state, basis in coordinates]
        )

        ptm, bloch, translation = reconstruct_ptm(p1, coordinates)
        metrics = tomography_metrics(ptm, bloch, np.eye(2))

        np.testing.assert_allclose(ptm, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(metrics["process_infidelity"], 0.0, places=12)


class RamseyTests(unittest.TestCase):
    def test_fringe_fit_recovers_phase_and_contrast(self) -> None:
        phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        expected_phase = 0.37
        probabilities = 0.48 + 0.41 * np.cos(phases - expected_phase)

        fitted = fit_fringe(phases, probabilities)

        self.assertAlmostEqual(fitted["offset"], 0.48, places=12)
        self.assertAlmostEqual(fitted["contrast"], 0.41, places=12)
        self.assertAlmostEqual(fitted["phase"], expected_phase, places=12)

    def test_ideal_data_has_zero_transition_phase_error(self) -> None:
        target = universal_composite_pulses()["X1"]
        point = ExperimentPoint("point", target, ParameterSet())
        technique = TransitionPhaseRamsey()
        context = TechniqueContext((("QB1",),))
        plan = technique.build_plan(point, context)
        phases = tuple(float(circuit.coordinates["phase"]) for circuit in plan.circuits)
        p1 = ideal_ramsey_p1(target.ideal_operation.unitary, phases, prepare_half_pi=True)
        measurements = MeasurementData(plan, {("QB1",): ProbabilityData(p1)})

        result = technique.analyze(point, plan, measurements)[0]

        self.assertAlmostEqual(result.metrics["transition_phase_error"], 0.0, places=12)


class RandomizedBenchmarkingTests(unittest.TestCase):
    def test_clifford_group_contains_24_unique_elements(self) -> None:
        group = clifford_group()

        self.assertEqual(len(group), 24)

    def test_decay_fit_recovers_synthetic_parameter(self) -> None:
        lengths = np.array([1, 2, 4, 8, 16, 32])
        survival = 0.45 * 0.97**lengths + 0.5

        decay = fit_decay(lengths, survival)

        self.assertAlmostEqual(decay, 0.97, places=8)

    def test_rb_plan_is_deterministic_and_typed(self) -> None:
        target = universal_composite_pulses()["X1"]
        point = ExperimentPoint("point", target, ParameterSet())
        technique = InterleavedRandomizedBenchmarking(
            RBSettings(lengths=(1, 2, 4), samples=2, shots=8, seed=11)
        )
        context = TechniqueContext((("QB1",),))

        first = technique.build_plan(point, context)
        second = technique.build_plan(point, context)

        self.assertEqual(len(first.circuits), 12)
        self.assertEqual(
            [dict(circuit.coordinates) for circuit in first.circuits],
            [dict(circuit.coordinates) for circuit in second.circuits],
        )


if __name__ == "__main__":
    unittest.main()
