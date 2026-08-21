"""Planning, compilation, inspection, and execution of circuit batches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iqm.cpc.compiler.compiler import Compiler
from iqm.cpc.core.run_result import RunResult
from iqm.pulla.pulla import Pulla
from iqm.pulse.playlist.visualisation.base import inspect_playlist
from iqm.pulse.timebox import TimeBox
from iqm.station_control.interface.models import RunDefinition
from xarray import Dataset

from dataset_persistence import JsonValue
from sequences import SequenceSpec


@dataclass(frozen=True)
class PlannedBatch:
    """Circuits and result metadata for one logical acquisition batch."""

    batch_id: str
    technique: str
    acquisition_index: int
    circuits: tuple[TimeBox, ...]
    qubits: tuple[str, ...]
    shots: int
    measurement_key: str
    metadata: Mapping[str, JsonValue]
    manifest: Any = None
    sequence: SequenceSpec | None = None
    amplitude_error: float | None = None
    detuning_hz: float | None = None


@dataclass(frozen=True)
class PlannedJob:
    """Logical batches compiled and submitted as one technique-level job."""

    job_id: str
    technique: str
    batches: tuple[PlannedBatch, ...]

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("A job must contain at least one batch.")
        first = self.batches[0]
        for batch in self.batches:
            if batch.shots != first.shots:
                raise ValueError("Every batch in a job must use the same shots.")
            if batch.qubits != first.qubits:
                raise ValueError("Every batch in a job must use the same qubits.")
            if batch.measurement_key != first.measurement_key:
                raise ValueError(
                    "Every batch in a job must use the same measurement key."
                )
            if batch.technique not in (self.technique, "readout_calibration"):
                raise ValueError(
                    f"Batch {batch.batch_id!r} does not belong to {self.technique!r}."
                )

    @property
    def circuits(self) -> tuple[TimeBox, ...]:
        """Return all batch circuits in execution order."""
        return tuple(
            circuit
            for batch in self.batches
            for circuit in batch.circuits
        )

    @property
    def qubits(self) -> tuple[str, ...]:
        return self.batches[0].qubits

    @property
    def shots(self) -> int:
        return self.batches[0].shots


@dataclass(frozen=True)
class CompiledJob:
    """A technique job compiled locally and ready for submission."""

    plan: PlannedJob
    run_definition: RunDefinition
    context: Mapping[str, Any]

    @property
    def estimated_qpu_runtime_seconds(self) -> float:
        """Return IQM's rough QPU-runtime estimate for this playlist."""
        return float(self.run_definition.sweep_definition.qpu_runtime)


@dataclass(frozen=True)
class SubmittedJob:
    """A submitted job kept together with its compiler context and plan."""

    compiled: CompiledJob
    job: Any


@dataclass(frozen=True)
class AcquiredBatch:
    """A postprocessed IQM dataset paired with its original circuit plan."""

    plan: PlannedBatch
    dataset: Dataset


def compile_job(compiler: Compiler, plan: PlannedJob) -> CompiledJob:
    """Compile one complete technique job without submitting it."""
    circuits = list(plan.circuits)
    settings = compiler.get_settings(timeboxes=circuits)
    settings.set_shots(plan.shots)
    run_definition, context = compiler.compile(
        timeboxes=circuits,
        components=list(plan.qubits),
        settings=settings,
    )
    if not isinstance(run_definition, RunDefinition):
        raise TypeError("Compiler did not produce an IQM RunDefinition.")
    return CompiledJob(plan, run_definition, context)


def compile_jobs(
    compiler: Compiler,
    plans: Sequence[PlannedJob],
) -> tuple[CompiledJob, ...]:
    """Compile every planned job before returning any executable work."""
    return tuple(compile_job(compiler, plan) for plan in plans)


def submit_jobs(
    pulla: Pulla,
    jobs: Sequence[CompiledJob],
) -> tuple[SubmittedJob, ...]:
    """Submit all precompiled jobs without waiting for their results."""
    return tuple(
        SubmittedJob(
            compiled=job,
            job=pulla.submit_playlist(
                job.run_definition,
                context=dict(job.context),
            ),
        )
        for job in jobs
    )


def _split_dataset(plan: PlannedJob, dataset: Dataset) -> tuple[AcquiredBatch, ...]:
    """Split a technique-level result into its original logical batches."""
    circuit_count = len(plan.circuits)
    if "circuit_index" not in dataset.dims:
        if circuit_count != 1 or len(plan.batches) != 1:
            raise ValueError(
                f"Job {plan.job_id!r} returned no circuit_index dimension for "
                f"{circuit_count} circuits."
            )
        return (AcquiredBatch(plan.batches[0], dataset),)
    if dataset.sizes["circuit_index"] != circuit_count:
        raise ValueError(
            f"Job {plan.job_id!r} returned {dataset.sizes['circuit_index']} "
            f"circuit results for {circuit_count} circuits."
        )

    acquired: list[AcquiredBatch] = []
    start = 0
    for batch in plan.batches:
        stop = start + len(batch.circuits)
        acquired.append(
            AcquiredBatch(
                batch,
                dataset.isel(circuit_index=slice(start, stop)),
            )
        )
        start = stop
    return tuple(acquired)


def collect_jobs(
    compiler: Compiler,
    jobs: Sequence[SubmittedJob],
) -> tuple[AcquiredBatch, ...]:
    """Wait for submitted jobs and recover their logical batch datasets."""
    acquired: list[AcquiredBatch] = []
    for job in jobs:
        job.job.wait_for_completion()
        result = job.job.result(compiler=compiler)
        if not isinstance(result, RunResult):
            raise RuntimeError("IQM job completed without an EXA-style RunResult.")
        acquired.extend(_split_dataset(job.compiled.plan, result.dataset))
    return tuple(acquired)


def execute_jobs(
    pulla: Pulla,
    compiler: Compiler,
    jobs: Sequence[CompiledJob],
) -> tuple[AcquiredBatch, ...]:
    """Submit all compiled work, then wait for and collect every result."""
    return collect_jobs(compiler, submit_jobs(pulla, jobs))


def write_preflight(
    jobs: Sequence[CompiledJob],
    output_directory: Path,
) -> dict[str, JsonValue]:
    """Persist circuit plots and an IQM QPU-runtime estimate before submission."""
    plot_directory = output_directory / "playlists"
    plot_directory.mkdir(parents=True, exist_ok=True)
    job_rows: list[dict[str, JsonValue]] = []

    for job in jobs:
        playlist = job.run_definition.sweep_definition.playlist
        segment_count = 0 if playlist is None else len(playlist.segments)
        estimate = job.estimated_qpu_runtime_seconds
        batch_rows: list[dict[str, JsonValue]] = []
        start = 0
        for batch in job.plan.batches:
            stop = start + len(batch.circuits)
            batch_rows.append({
                "batch_id": batch.batch_id,
                "technique": batch.technique,
                "circuit_start": start,
                "circuit_stop": stop,
            })
            start = stop
        job_rows.append({
            "job_id": job.plan.job_id,
            "technique": job.plan.technique,
            "batch_count": len(job.plan.batches),
            "circuit_count": len(job.plan.circuits),
            "playlist_segment_count": segment_count,
            "shots": job.plan.shots,
            "estimated_qpu_runtime_seconds": estimate,
            "batches": batch_rows,
        })
        if playlist is not None and segment_count:
            inspection = inspect_playlist(playlist, segments=range(segment_count))
            (plot_directory / f"{job.plan.job_id}.html").write_text(
                "<!doctype html><html><body>" + inspection + "</body></html>\n",
                encoding="utf-8",
            )

    report: dict[str, JsonValue] = {
        "job_count": len(jobs),
        "batch_count": sum(len(job.plan.batches) for job in jobs),
        "circuit_count": sum(len(job.plan.circuits) for job in jobs),
        "estimated_qpu_runtime_seconds": sum(
            job.estimated_qpu_runtime_seconds for job in jobs
        ),
        "estimate_scope": "QPU runtime only; queue and network time are excluded.",
        "jobs": job_rows,
    }
    (output_directory / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
