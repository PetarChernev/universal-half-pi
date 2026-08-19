from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1] / "iqm_composite_experiment"
sys.path.insert(0, str(PACKAGE_DIRECTORY))

import execution  # noqa: E402
import main as experiment_main  # noqa: E402
from execution import CompiledBatch, PlannedBatch  # noqa: E402


def planned_batch(index: int) -> PlannedBatch:
    return PlannedBatch(
        batch_id=f"test_{index:04d}",
        technique="test",
        acquisition_index=index,
        circuits=(f"circuit-{index}",),  # type: ignore[arg-type]
        qubits=("QB1",),
        shots=10 + index,
        measurement_key="test",
        metadata={"index": index},
    )


class ExecutionLifecycleTests(unittest.TestCase):
    def test_all_batches_are_compiled_then_submitted_before_first_wait(self) -> None:
        events: list[str] = []

        class FakeSettings:
            def set_shots(self, shots: int) -> None:
                events.append(f"shots:{shots}")

        class FakeSweepDefinition:
            qpu_runtime = 0.1
            playlist = None

        class FakeRunDefinition:
            def __init__(self, batch_id: str) -> None:
                self.batch_id = batch_id
                self.sweep_definition = FakeSweepDefinition()

        class FakeCompiler:
            def get_settings(self, *, timeboxes: list[str]) -> FakeSettings:
                events.append(f"settings:{timeboxes[0]}")
                return FakeSettings()

            def compile(self, *, timeboxes, components, settings):
                del components, settings
                events.append(f"compile:{timeboxes[0]}")
                return FakeRunDefinition(timeboxes[0]), {"id": timeboxes[0]}

        class FakeRunResult:
            def __init__(self, dataset: str) -> None:
                self.dataset = dataset

        class FakeJob:
            def __init__(self, batch_id: str) -> None:
                self.batch_id = batch_id

            def wait_for_completion(self) -> None:
                events.append(f"wait:{self.batch_id}")

            def result(self, *, compiler):
                del compiler
                events.append(f"result:{self.batch_id}")
                return FakeRunResult(self.batch_id)

        class FakePulla:
            def submit_playlist(self, run_definition, *, context):
                self.assert_context(run_definition.batch_id, context)
                events.append(f"submit:{run_definition.batch_id}")
                return FakeJob(run_definition.batch_id)

            @staticmethod
            def assert_context(batch_id: str, context: dict[str, str]) -> None:
                if context != {"id": batch_id}:
                    raise AssertionError("Compiler context was paired with the wrong batch.")

        plans = tuple(planned_batch(index) for index in range(3))
        compiler = FakeCompiler()
        with (
            patch.object(execution, "RunDefinition", FakeRunDefinition),
            patch.object(execution, "RunResult", FakeRunResult),
        ):
            compiled = execution.compile_batches(compiler, plans)
            acquired = execution.execute_batches(FakePulla(), compiler, compiled)

        first_submit = next(i for i, event in enumerate(events) if event.startswith("submit:"))
        first_wait = next(i for i, event in enumerate(events) if event.startswith("wait:"))
        compile_indices = [i for i, event in enumerate(events) if event.startswith("compile:")]
        submit_indices = [i for i, event in enumerate(events) if event.startswith("submit:")]
        self.assertLess(max(compile_indices), first_submit)
        self.assertLess(max(submit_indices), first_wait)
        self.assertEqual([batch.plan for batch in acquired], list(plans))

    def test_preflight_writes_runtime_summary_and_playlist_plots(self) -> None:
        class FakePlaylist:
            segments = [object(), object()]

        class FakeSweepDefinition:
            playlist = FakePlaylist()

            def __init__(self, runtime: float) -> None:
                self.qpu_runtime = runtime

        class FakeRunDefinition:
            def __init__(self, runtime: float) -> None:
                self.sweep_definition = FakeSweepDefinition(runtime)

        batches = tuple(
            CompiledBatch(
                planned_batch(index),
                FakeRunDefinition(runtime),  # type: ignore[arg-type]
                {},
            )
            for index, runtime in enumerate((0.25, 0.75))
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(execution, "inspect_playlist", return_value="<p>plot</p>"):
                report = execution.write_preflight(batches, output)

            self.assertEqual(report["batch_count"], 2)
            self.assertEqual(report["circuit_count"], 2)
            self.assertEqual(report["estimated_qpu_runtime_seconds"], 1.0)
            self.assertEqual(
                json.loads((output / "preflight.json").read_text()),
                report,
            )
            self.assertTrue((output / "playlists" / "test_0000.html").exists())
            self.assertTrue((output / "playlists" / "test_0001.html").exists())

    def test_preflight_precedes_execution_and_prepare_only_skips_it(self) -> None:
        events: list[str] = []
        report = {
            "circuit_count": 3,
            "batch_count": 2,
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
                    "execute_batches",
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
            "batch_count": 1,
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
                    "execute_batches",
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
