"""End-to-end framework tests using ideal local matrix propagation."""

from __future__ import annotations

import unittest

from gate_experiment.experiment import (
    CharacterizationExperiment,
    ExperimentRunner,
    ExperimentSettings,
    FailurePolicy,
)
from gate_experiment.models.acquisition import AcquisitionPlan, MeasurementData
from gate_experiment.models.points import PointSet
from gate_experiment.models.results import ResultStatus, TechniqueResult
from gate_experiment.readout import IndependentReadoutCorrection
from gate_experiment.targets.universal import universal_composite_pulses
from gate_experiment.techniques.base import CharacterizationTechnique, TechniqueContext
from gate_experiment.techniques import ProcessTomography, TransitionPhaseRamsey

from tests.fakes import MatrixBackend


class ExperimentIntegrationTests(unittest.TestCase):
    def test_selected_techniques_run_end_to_end_without_services(self) -> None:
        target = universal_composite_pulses()["X1"]
        experiment = CharacterizationExperiment(
            points=PointSet.cartesian(targets=(target,)),
            techniques=(ProcessTomography(), TransitionPhaseRamsey()),
            settings=ExperimentSettings(qubits=("QB1", "QB2")),
        )
        backend = MatrixBackend()

        result = ExperimentRunner(backend).run(experiment)

        self.assertIsNotNone(backend.session)
        self.assertTrue(backend.session.closed)
        self.assertEqual(backend.session.executed_techniques, ["tomography", "transition_phase_ramsey"])
        self.assertEqual(len(result.points), 1)
        self.assertEqual(len(result.points[0].techniques), 4)
        for technique_result in result.points[0].techniques:
            if technique_result.technique_id == "tomography":
                self.assertAlmostEqual(technique_result.metrics["process_infidelity"], 0.0, places=12)
            else:
                self.assertAlmostEqual(
                    technique_result.metrics["transition_phase_error"], 0.0, places=12
                )

    def test_omitted_techniques_are_never_planned_or_executed(self) -> None:
        target = universal_composite_pulses()["H1"]
        experiment = CharacterizationExperiment(
            points=PointSet.cartesian(targets=(target,)),
            techniques=(ProcessTomography(),),
        )
        backend = MatrixBackend()

        result = ExperimentRunner(backend).run(experiment)

        self.assertEqual(backend.session.executed_techniques, ["tomography"])
        self.assertEqual(
            set(result.metrics_dataframe()["technique_id"]),
            {"tomography"},
        )

    def test_result_views_include_target_and_point_parameters(self) -> None:
        target = universal_composite_pulses()["X1"]
        experiment = CharacterizationExperiment(
            points=PointSet.cartesian(
                targets=(target,),
                fixed={"amplitude_error": 0.01, "detuning_hz": 0.0},
            ),
            techniques=(ProcessTomography(),),
        )

        result = ExperimentRunner(MatrixBackend()).run(experiment)
        table = result.legacy_dataframe()

        self.assertEqual(table.loc[0, "sequence"], "X1")
        self.assertEqual(table.loc[0, "amplitude_error"], 0.01)
        self.assertEqual(table.loc[0, "pulse_count"], 1)

    def test_readout_calibration_runs_once_and_is_retained(self) -> None:
        target = universal_composite_pulses()["H1"]
        experiment = CharacterizationExperiment(
            points=PointSet.cartesian(targets=(target,)),
            techniques=(ProcessTomography(),),
            readout=IndependentReadoutCorrection(),
        )
        backend = MatrixBackend()

        result = ExperimentRunner(backend).run(experiment)

        self.assertEqual(
            backend.session.executed_techniques,
            ["readout_calibration", "tomography"],
        )
        self.assertEqual(len(result.calibrations), 1)
        calibration = result.calibrations[0].metrics_by_locus[("QB1",)]
        self.assertAlmostEqual(calibration["p1_given_0"], 0.0)
        self.assertAlmostEqual(calibration["p0_given_1"], 0.0)

    def test_record_and_continue_preserves_a_failed_technique_result(self) -> None:
        class FailingTechnique(CharacterizationTechnique):
            technique_id = "failing"

            def supports(self, target) -> bool:
                return True

            def validate(self, point, context: TechniqueContext) -> None:
                return None

            def build_plan(self, point, context: TechniqueContext) -> AcquisitionPlan:
                raise RuntimeError("synthetic failure")

            def analyze(
                self,
                point,
                plan: AcquisitionPlan,
                measurements: MeasurementData,
            ) -> tuple[TechniqueResult, ...]:
                raise AssertionError("analyze must not be reached")

        target = universal_composite_pulses()["X1"]
        experiment = CharacterizationExperiment(
            points=PointSet.cartesian(targets=(target,)),
            techniques=(FailingTechnique(), ProcessTomography()),
            settings=ExperimentSettings(
                failure_policy=FailurePolicy.RECORD_AND_CONTINUE
            ),
        )

        result = ExperimentRunner(MatrixBackend()).run(experiment)

        failed, successful = result.points[0].techniques
        self.assertIs(failed.status, ResultStatus.FAILED)
        self.assertIn("synthetic failure", failed.error)
        self.assertIs(successful.status, ResultStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
