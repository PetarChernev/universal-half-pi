"""Tests demonstrating target and backend compiler extension points."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from gate_experiment.backend.base import OperationCompiler, OperationCompilerRegistry
from gate_experiment.experiment import CharacterizationExperiment, ExperimentRunner
from gate_experiment.models.acquisition import Locus
from gate_experiment.models.parameters import ParameterSet, ParameterSpec
from gate_experiment.models.points import PointSet, SweepAxis
from gate_experiment.targets.base import IdealOperation, OperationUnderTest
from gate_experiment.targets.composite import prx_unitary
from gate_experiment.techniques import ProcessTomography

from tests.fakes import MatrixBackend


@dataclass(frozen=True, eq=False)
class ExampleShapedPulse(OperationUnderTest):
    """Minimal non-composite target resembling a pulse-shaping plugin."""

    operation_id: str
    ideal: IdealOperation

    @property
    def ideal_operation(self) -> IdealOperation:
        return self.ideal

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                "shape_beta",
                float,
                default=0.0,
                description="Example envelope-shape coefficient.",
            ),
        )


class ExampleShapedPulseCompiler(OperationCompiler[ExampleShapedPulse]):
    """Symbolic compiler used only to prove registry dispatch."""

    target_type = ExampleShapedPulse

    def compile(
        self,
        target: ExampleShapedPulse,
        parameters: ParameterSet,
        locus: Locus,
        context: object,
    ) -> object:
        resolved = target.resolve_parameters(parameters)
        return (target.operation_id, locus, resolved.require_float("shape_beta"), context)


class PluginExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = ExampleShapedPulse(
            operation_id="shaped_rx90",
            ideal=IdealOperation(prx_unitary(np.pi / 2, 0), arity=1, name="Rx(pi/2)"),
        )

    def test_new_target_class_works_with_existing_tomography(self) -> None:
        points = PointSet.cartesian(
            targets=(self.target,),
            axes=(SweepAxis("shape_beta", (-0.2, 0.2)),),
        )
        experiment = CharacterizationExperiment(
            points=points,
            techniques=(ProcessTomography(),),
        )

        result = ExperimentRunner(MatrixBackend()).run(experiment)

        self.assertEqual(len(result.points), 2)
        for point_result in result.points:
            self.assertAlmostEqual(
                point_result.techniques[0].metrics["process_infidelity"],
                0.0,
                places=12,
            )

    def test_backend_compiler_registry_dispatches_new_target_type(self) -> None:
        registry = OperationCompilerRegistry((ExampleShapedPulseCompiler(),))

        compiled = registry.compile(
            self.target,
            ParameterSet({"shape_beta": 0.4}),
            ("QB1",),
            "local-context",
        )

        self.assertTrue(registry.supports(self.target))
        self.assertEqual(
            compiled,
            ("shaped_rx90", ("QB1",), 0.4, "local-context"),
        )


if __name__ == "__main__":
    unittest.main()
