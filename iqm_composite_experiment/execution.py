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
    """Circuits and result metadata for one independently executable batch."""

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
class CompiledBatch:
    """A planned batch compiled locally and ready for inspection or submission."""

    plan: PlannedBatch
    run_definition: RunDefinition
    context: Mapping[str, Any]

    @property
    def estimated_qpu_runtime_seconds(self) -> float:
        """Return IQM's rough QPU-runtime estimate for this playlist."""
        return float(self.run_definition.sweep_definition.qpu_runtime)


@dataclass(frozen=True)
class SubmittedBatch:
    """A submitted job kept together with its compiler context and manifest."""

    compiled: CompiledBatch
    job: Any


@dataclass(frozen=True)
class AcquiredBatch:
    """A postprocessed IQM dataset paired with its original circuit plan."""

    plan: PlannedBatch
    dataset: Dataset


def compile_batch(compiler: Compiler, plan: PlannedBatch) -> CompiledBatch:
    """Compile one complete batch without submitting it."""
    settings = compiler.get_settings(timeboxes=list(plan.circuits))
    settings.set_shots(plan.shots)
    run_definition, context = compiler.compile(
        timeboxes=list(plan.circuits),
        components=list(plan.qubits),
        settings=settings,
    )
    if not isinstance(run_definition, RunDefinition):
        raise TypeError("Compiler did not produce an IQM RunDefinition.")
    return CompiledBatch(plan, run_definition, context)


def compile_batches(
    compiler: Compiler,
    plans: Sequence[PlannedBatch],
) -> tuple[CompiledBatch, ...]:
    """Compile every planned batch before returning any executable work."""
    return tuple(compile_batch(compiler, plan) for plan in plans)


def submit_batches(
    pulla: Pulla,
    batches: Sequence[CompiledBatch],
) -> tuple[SubmittedBatch, ...]:
    """Submit all precompiled batches without waiting for their results."""
    return tuple(
        SubmittedBatch(
            compiled=batch,
            job=pulla.submit_playlist(
                batch.run_definition,
                context=dict(batch.context),
            ),
        )
        for batch in batches
    )


def collect_batches(
    compiler: Compiler,
    batches: Sequence[SubmittedBatch],
) -> tuple[AcquiredBatch, ...]:
    """Wait for submitted jobs and recover compiler-postprocessed datasets."""
    acquired: list[AcquiredBatch] = []
    for batch in batches:
        batch.job.wait_for_completion()
        result = batch.job.result(compiler=compiler)
        if not isinstance(result, RunResult):
            raise RuntimeError("IQM job completed without an EXA-style RunResult.")
        acquired.append(AcquiredBatch(batch.compiled.plan, result.dataset))
    return tuple(acquired)


def execute_batches(
    pulla: Pulla,
    compiler: Compiler,
    batches: Sequence[CompiledBatch],
) -> tuple[AcquiredBatch, ...]:
    """Submit all compiled work, then wait for and collect every result."""
    return collect_batches(compiler, submit_batches(pulla, batches))


def write_preflight(
    batches: Sequence[CompiledBatch],
    output_directory: Path,
) -> dict[str, JsonValue]:
    """Persist circuit plots and an IQM QPU-runtime estimate before submission."""
    plot_directory = output_directory / "playlists"
    plot_directory.mkdir(parents=True, exist_ok=True)
    batch_rows: list[dict[str, JsonValue]] = []

    for batch in batches:
        playlist = batch.run_definition.sweep_definition.playlist
        segment_count = 0 if playlist is None else len(playlist.segments)
        estimate = batch.estimated_qpu_runtime_seconds
        batch_rows.append({
            "batch_id": batch.plan.batch_id,
            "technique": batch.plan.technique,
            "circuit_count": len(batch.plan.circuits),
            "playlist_segment_count": segment_count,
            "shots": batch.plan.shots,
            "estimated_qpu_runtime_seconds": estimate,
        })
        if playlist is not None and segment_count:
            inspection = inspect_playlist(playlist, segments=range(segment_count))
            (plot_directory / f"{batch.plan.batch_id}.html").write_text(
                "<!doctype html><html><body>" + inspection + "</body></html>\n",
                encoding="utf-8",
            )

    report: dict[str, JsonValue] = {
        "batch_count": len(batches),
        "circuit_count": sum(len(batch.plan.circuits) for batch in batches),
        "estimated_qpu_runtime_seconds": sum(
            batch.estimated_qpu_runtime_seconds for batch in batches
        ),
        "estimate_scope": "QPU runtime only; queue and network time are excluded.",
        "batches": batch_rows,
    }
    (output_directory / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
