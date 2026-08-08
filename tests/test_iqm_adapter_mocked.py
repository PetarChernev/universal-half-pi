"""Mocked IQM adapter integration tests with no service-capable objects."""

from __future__ import annotations

import unittest

import numpy as np

from gate_experiment.backend.iqm import IQMBackend
from gate_experiment.models.acquisition import AcquisitionPlan, PlannedCircuit

from tests.test_readout_decoder import FakeDataArray, FakeDataset


class FakeTopology:
    qubits_sorted = ("QB1", "QB2")


class FakeSettings:
    def __init__(self) -> None:
        self.shots = None

    def set_shots(self, shots: int) -> None:
        self.shots = shots


class FakeCompiler:
    def __init__(self) -> None:
        self.settings = FakeSettings()
        self.generated = None

    def get_settings(self, *, timeboxes, qubits):
        self.timeboxes = timeboxes
        self.qubits = qubits
        return self.settings

    def compile(self, *, timeboxes, components, settings):
        self.generated = timeboxes(object())
        return "run-definition", {"mocked": True}


class FakeRunResult:
    def __init__(self, dataset) -> None:
        self.dataset = dataset


class FakeJob:
    job_id = "local-job"

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.waited = False

    def wait_for_completion(self) -> None:
        self.waited = True

    def result(self, compiler):
        return FakeRunResult(self.dataset)


class FakePulla:
    """A local object with the Pulla surface but no URL or network client."""

    def __init__(self) -> None:
        self.compiler = FakeCompiler()
        self.job = FakeJob(
            FakeDataset(
                {
                    "QB1__mock_excited_state_probability": FakeDataArray(
                        np.array([0.25]),
                        ("circuit_index",),
                    )
                }
            )
        )

    def get_chip_topology(self):
        return FakeTopology()

    def get_standard_compiler(self):
        return self.compiler

    def submit_playlist(self, run_definition, *, context):
        self.submission = (run_definition, context)
        return self.job


class IQMAdapterMockedTests(unittest.TestCase):
    def test_constructor_does_not_touch_pulla(self) -> None:
        class ExplodingPulla:
            def __getattr__(self, name):
                raise AssertionError(f"Unexpected access to {name}")

        IQMBackend(ExplodingPulla())

    def test_deferred_recipe_is_built_inside_compile_callback(self) -> None:
        pulla = FakePulla()
        backend = IQMBackend(pulla)
        session = backend.open_session(("QB1",))
        calls = []

        def build(factory):
            calls.append(factory)
            return "local-timebox"

        plan = AcquisitionPlan(
            technique_id="mock",
            measurement_key="mock",
            shots=17,
            circuits=(PlannedCircuit("mock:0", build),),
        )

        measurements = session.execute(plan)

        self.assertEqual(len(calls), 1)
        self.assertEqual(pulla.compiler.generated, ["local-timebox"])
        self.assertEqual(pulla.compiler.settings.shots, 17)
        self.assertTrue(pulla.job.waited)
        np.testing.assert_allclose(measurements.for_locus(("QB1",)).raw_p1, [0.25])


if __name__ == "__main__":
    unittest.main()
