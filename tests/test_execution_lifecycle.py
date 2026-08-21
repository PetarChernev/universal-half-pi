from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr


PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "iqm_composite_experiment"
sys.path.insert(0, str(PACKAGE_DIRECTORY))

import execution  # noqa: E402
import main as experiment_main  # noqa: E402
from config import Config  # noqa: E402
from execution import CompiledJob, PlannedBatch, PlannedJob  # noqa: E402


def planned_batch(
    index: int,
    *,
    technique: str = "process_tomography",
    shots: int = 10,
    measurement_key: str = "result",
    circuits: tuple[str, ...] | None = None,
) -> PlannedBatch:
    return PlannedBatch(
        batch_id=f"test_{index:04d}",
        technique=technique,
        acquisition_index=index,
        circuits=(circuits or (f"circuit-{index}",)),  # type: ignore[arg-type]
        qubits=("QB1",),
        shots=shots,
        measurement_key=measurement_key,
        metadata={"index": index},
    )


def planned_job(job_id: str, batches: tuple[PlannedBatch, ...]) -> PlannedJob:
    return PlannedJob(job_id, batches[0].technique, batches)


class TechniqueJobPlanningTests(unittest.TestCase):
    def test_calibration_flags_control_circuit_order_and_use_technique_settings(
        self,
    ) -> None:
        technique = planned_batch(
            0,
            shots=37,
            measurement_key="tomo",
            circuits=("technique-0", "technique-1"),
        )

        def calibration_batch(*args, **kwargs) -> PlannedBatch:
            del args
            position = kwargs["position"]
            return PlannedBatch(
                batch_id=kwargs["batch_id"],
                technique="readout_calibration",
                acquisition_index=0,
                circuits=(f"{position}-ground", f"{position}-excited"),  # type: ignore[arg-type]
                qubits=("QB1",),
                shots=kwargs["shots"],
                measurement_key=kwargs["measurement_key"],
                metadata={
                    "characterization_technique": kwargs[
                        "characterization_technique"
                    ],
                    "calibration_position": position,
                },
            )

        cases = (
            (True, False, ("pre-ground", "pre-excited", "technique-0", "technique-1")),
            (False, True, ("technique-0", "technique-1", "post-ground", "post-excited")),
            (
                True,
                True,
                (
                    "pre-ground",
                    "pre-excited",
                    "technique-0",
                    "technique-1",
                    "post-ground",
                    "post-excited",
                ),
            ),
            (False, False, ("technique-0", "technique-1")),
        )
        for pre, post, expected in cases:
            with self.subTest(pre=pre, post=post):
                config = Config(pre_calibration=pre, post_calibration=post)
                with patch.object(
                    experiment_main,
                    "build_readout_batch",
                    side_effect=calibration_batch,
                ):
                    job = experiment_main.build_technique_job(
                        object(),  # type: ignore[arg-type]
                        "tomography",
                        (technique,),
                        config,
                    )

                self.assertEqual(job.circuits, expected)
                self.assertTrue(all(batch.shots == 37 for batch in job.batches))
                self.assertTrue(
                    all(batch.measurement_key == "tomo" for batch in job.batches)
                )

    def test_job_rejects_incompatible_batch_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shots"):
            PlannedJob(
                "tomography",
                "process_tomography",
                (planned_batch(0, shots=10), planned_batch(1, shots=20)),
            )
        with self.assertRaisesRegex(ValueError, "same measurement key"):
            PlannedJob(
                "tomography",
                "process_tomography",
                (
                    planned_batch(0, measurement_key="first"),
                    planned_batch(1, measurement_key="second"),
                ),
            )

    def test_split_job_associates_pre_and_post_calibration_with_technique(self) -> None:
        def calibration(position: str) -> PlannedBatch:
            return PlannedBatch(
                batch_id=f"tomography_readout_{position}",
                technique="readout_calibration",
                acquisition_index=0,
                circuits=(f"{position}-ground", f"{position}-excited"),  # type: ignore[arg-type]
                qubits=("QB1",),
                shots=100,
                measurement_key="tomo",
                metadata={
                    "characterization_technique": "process_tomography",
                    "calibration_position": position,
                },
            )

        technique = planned_batch(
            0,
            shots=100,
            measurement_key="tomo",
            circuits=("tomography-0", "tomography-1"),
        )
        job = PlannedJob(
            "tomography",
            "process_tomography",
            (calibration("pre"), technique, calibration("post")),
        )
        variable = "QB1__tomo_excited_state_probability"
        dataset = xr.Dataset({
            variable: (
                "circuit_index",
                [0.1, 0.8, 0.4, 0.6, 0.2, 0.9],
            )
        })

        acquired = execution._split_dataset(job, dataset)
        readout = experiment_main._readout_for_technique(
            acquired,
            "process_tomography",
            ("QB1",),
            Config(pre_calibration=True, post_calibration=True),
        )

        np.testing.assert_array_equal(
            acquired[1].dataset[variable],
            [0.4, 0.6],
        )
        calibration_result = readout["QB1"]
        self.assertIsNotNone(calibration_result)
        assert calibration_result is not None
        self.assertAlmostEqual(calibration_result.p1_given_0, 0.15)
        self.assertAlmostEqual(calibration_result.p0_given_1, 0.15)

    def test_build_experiment_logs_job_progress(self) -> None:
        batch = planned_batch(0)
        with (
            patch.object(
                experiment_main.tomography,
                "build_batches",
                return_value=[batch],
            ),
            patch.object(
                experiment_main,
                "build_technique_job",
                return_value=planned_job("tomography", (batch,)),
            ),
            self.assertLogs(experiment_main.logger, level="INFO") as messages,
        ):
            jobs = experiment_main.build_experiment(
                object(),  # type: ignore[arg-type]
                ("QB1",),
                (),
                Config(),
            )

        self.assertEqual(len(jobs), 1)
        self.assertIn("Building process tomography job circuits", messages.output[0])
        self.assertIn("Built process tomography job with 1 circuits", messages.output[1])


class ExecutionLifecycleTests(unittest.TestCase):
    def test_one_compile_and_submission_per_job_and_results_are_split(self) -> None:
        events: list[str] = []

        class FakeSettings:
            def __init__(self, job_id: str) -> None:
                self.job_id = job_id

            def set_shots(self, shots: int) -> None:
                events.append(f"shots:{self.job_id}:{shots}")

        class FakeSweepDefinition:
            qpu_runtime = 0.1
            playlist = None

        class FakeRunDefinition:
            def __init__(self, job_id: str, circuit_count: int) -> None:
                self.job_id = job_id
                self.circuit_count = circuit_count
                self.sweep_definition = FakeSweepDefinition()

        class FakeCompiler:
            def get_settings(self, *, timeboxes: list[str]) -> FakeSettings:
                job_id = timeboxes[0].split("-")[0]
                events.append(f"settings:{job_id}:{','.join(timeboxes)}")
                return FakeSettings(job_id)

            def compile(self, *, timeboxes, components, settings):
                del components, settings
                job_id = timeboxes[0].split("-")[0]
                events.append(f"compile:{job_id}:{','.join(timeboxes)}")
                return (
                    FakeRunDefinition(job_id, len(timeboxes)),
                    {"id": job_id},
                )

        class FakeRunResult:
            def __init__(self, circuit_count: int) -> None:
                self.dataset = xr.Dataset({
                    "value": (
                        "circuit_index",
                        np.arange(circuit_count, dtype=float),
                    )
                })

        class FakeJob:
            def __init__(self, run_definition: FakeRunDefinition) -> None:
                self.run_definition = run_definition

            def wait_for_completion(self) -> None:
                events.append(f"wait:{self.run_definition.job_id}")

            def result(self, *, compiler):
                del compiler
                events.append(f"result:{self.run_definition.job_id}")
                return FakeRunResult(self.run_definition.circuit_count)

        class FakePulla:
            def submit_playlist(self, run_definition, *, context):
                if context != {"id": run_definition.job_id}:
                    raise AssertionError("Compiler context was paired with the wrong job.")
                events.append(f"submit:{run_definition.job_id}")
                return FakeJob(run_definition)

        tomography_batches = (
            planned_batch(0, circuits=("tomography-0", "tomography-1")),
            planned_batch(1, circuits=("tomography-2",)),
        )
        ramsey_batches = (
            planned_batch(
                2,
                technique="transition_phase_ramsey",
                circuits=("ramsey-0", "ramsey-1"),
            ),
        )
        plans = (
            planned_job("tomography", tomography_batches),
            planned_job("ramsey", ramsey_batches),
        )
        compiler = FakeCompiler()
        with (
            patch.object(execution, "RunDefinition", FakeRunDefinition),
            patch.object(execution, "RunResult", FakeRunResult),
        ):
            compiled = execution.compile_jobs(compiler, plans)
            acquired = execution.execute_jobs(FakePulla(), compiler, compiled)

        compile_events = [event for event in events if event.startswith("compile:")]
        submit_events = [event for event in events if event.startswith("submit:")]
        first_submit = next(i for i, event in enumerate(events) if event.startswith("submit:"))
        first_wait = next(i for i, event in enumerate(events) if event.startswith("wait:"))
        self.assertEqual(len(compile_events), 2)
        self.assertEqual(submit_events, ["submit:tomography", "submit:ramsey"])
        self.assertLess(max(i for i, event in enumerate(events) if event.startswith("compile:")), first_submit)
        self.assertLess(max(i for i, event in enumerate(events) if event.startswith("submit:")), first_wait)
        self.assertEqual([batch.plan for batch in acquired], list(tomography_batches + ramsey_batches))
        np.testing.assert_array_equal(acquired[0].dataset["value"], [0, 1])
        np.testing.assert_array_equal(acquired[1].dataset["value"], [2])
        np.testing.assert_array_equal(acquired[2].dataset["value"], [0, 1])

    def test_preflight_writes_job_summary_and_playlist_plots(self) -> None:
        class FakePlaylist:
            segments = [object(), object()]

        class FakeSweepDefinition:
            playlist = FakePlaylist()

            def __init__(self, runtime: float) -> None:
                self.qpu_runtime = runtime

        class FakeRunDefinition:
            def __init__(self, runtime: float) -> None:
                self.sweep_definition = FakeSweepDefinition(runtime)

        plans = (
            planned_job("tomography", (planned_batch(0),)),
            planned_job(
                "ramsey",
                (planned_batch(1, technique="transition_phase_ramsey"),),
            ),
        )
        jobs = tuple(
            CompiledJob(
                plan,
                FakeRunDefinition(runtime),  # type: ignore[arg-type]
                {},
            )
            for plan, runtime in zip(plans, (0.25, 0.75), strict=True)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(execution, "inspect_playlist", return_value="<p>plot</p>"):
                report = execution.write_preflight(jobs, output)

            self.assertEqual(report["job_count"], 2)
            self.assertEqual(report["batch_count"], 2)
            self.assertEqual(report["circuit_count"], 2)
            self.assertEqual(report["estimated_qpu_runtime_seconds"], 1.0)
            self.assertEqual(
                json.loads((output / "preflight.json").read_text()),
                report,
            )
            self.assertTrue((output / "playlists" / "tomography.html").exists())
            self.assertTrue((output / "playlists" / "ramsey.html").exists())

    def test_preflight_precedes_execution_and_prepare_only_skips_it(self) -> None:
        events: list[str] = []
        report = {
            "circuit_count": 3,
            "job_count": 2,
            "estimated_qpu_runtime_seconds": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(
                    experiment_main,
                    "write_preflight",
                    side_effect=lambda *args: events.append("preflight") or report,
                ),
                patch.object(
                    experiment_main,
                    "execute_jobs",
                    side_effect=lambda *args: events.append("execute") or tuple(),
                ),
            ):
                result = experiment_main.run_compiled_experiment(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    (),
                    output,
                    prepare_only=False,
                )
                self.assertEqual(result, ())
                self.assertEqual(events, ["preflight", "execute"])

                events.clear()
                result = experiment_main.run_compiled_experiment(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    (),
                    output,
                    prepare_only=True,
                )
                self.assertIsNone(result)
                self.assertEqual(events, ["preflight"])

    def test_submission_error_propagates_after_preflight(self) -> None:
        class ExpectedSubmissionError(Exception):
            pass

        events: list[str] = []
        report = {
            "circuit_count": 1,
            "job_count": 1,
            "estimated_qpu_runtime_seconds": 0.1,
        }

        def fail_submission(*args) -> None:
            events.append("submit")
            raise ExpectedSubmissionError

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    experiment_main,
                    "write_preflight",
                    side_effect=lambda *args: events.append("preflight") or report,
                ),
                patch.object(
                    experiment_main,
                    "execute_jobs",
                    side_effect=fail_submission,
                ),
            ):
                with self.assertRaises(ExpectedSubmissionError):
                    experiment_main.run_compiled_experiment(
                        object(),  # type: ignore[arg-type]
                        object(),  # type: ignore[arg-type]
                        (),
                        Path(directory),
                        prepare_only=False,
                    )
        self.assertEqual(events, ["preflight", "submit"])


if __name__ == "__main__":
    unittest.main()
