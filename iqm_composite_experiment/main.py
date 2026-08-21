"""Build, inspect, and optionally run the composite-gate experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
import logging
from pathlib import Path

from exa.common.errors.iqm_error import IQMError
from iqm.cpc.compiler.compiler import Compiler
from iqm.iqm_client import IQMClient
from iqm.pulla.pulla import Pulla
from iqm.pulse.builder import ScheduleBuilder
from requests import RequestException

from characterization import ramsey, rb, tomography
from common import (
    ReadoutMap,
    build_readout_batch,
    built_in_sequences,
    readout_calibration_from_datasets,
    readout_calibration_metadata,
)
from config import Config
from dataset_persistence import persist_dataset, persist_metadata
from execution import (
    AcquiredBatch,
    CompiledJob,
    PlannedBatch,
    PlannedJob,
    compile_job,
    execute_jobs,
    write_preflight,
)
from qubit_selection import select_qubits
from sequences import SequenceSpec


logger = logging.getLogger(__name__)


def build_technique_job(
    builder: ScheduleBuilder,
    job_id: str,
    batches: Sequence[PlannedBatch],
    config: Config,
) -> PlannedJob:
    """Wrap one technique's logical batches with optional readout calibration."""
    if not batches:
        raise ValueError(f"Cannot build an empty technique job: {job_id}")
    first = batches[0]
    planned: list[PlannedBatch] = []
    if config.pre_calibration:
        planned.append(build_readout_batch(
            builder,
            first.qubits,
            config,
            batch_id=f"{job_id}_readout_pre",
            characterization_technique=first.technique,
            position="pre",
            shots=first.shots,
            measurement_key=first.measurement_key,
        ))
    planned.extend(batches)
    if config.post_calibration:
        planned.append(build_readout_batch(
            builder,
            first.qubits,
            config,
            batch_id=f"{job_id}_readout_post",
            characterization_technique=first.technique,
            position="post",
            shots=first.shots,
            measurement_key=first.measurement_key,
        ))
    return PlannedJob(job_id, first.technique, tuple(planned))


def build_experiment(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> tuple[PlannedJob, ...]:
    """Materialize one job per enabled characterization technique."""
    jobs: list[PlannedJob] = []
    technique_builders = (
        ("tomography", "process tomography", tomography.build_batches),
        # ("ramsey", "transition-phase Ramsey", ramsey.build_batches),
        # ("rb", "interleaved randomized benchmarking", rb.build_batches),
    )
    for job_id, label, build_batches in technique_builders:
        logger.info("Building %s job circuits", label)
        batches = build_batches(builder, qubits, sequences, config)
        jobs.append(build_technique_job(builder, job_id, batches, config))
        logger.info("Built %s job with %d circuits", label, len(jobs[-1].circuits))
    return tuple(jobs)


def prepare_experiment(
    compiler: Compiler,
    qubits: Sequence[str],
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> tuple[CompiledJob, ...]:
    """Build and compile the full experiment before anything is submitted."""
    builder = compiler.get_schedule_builder()
    planned = build_experiment(builder, qubits, sequences, config)
    compiled: list[CompiledJob] = []
    for index, job in enumerate(planned, start=1):
        logger.info(
            "Compiling %s job (%d/%d, %d circuits)",
            job.technique,
            index,
            len(planned),
            len(job.circuits),
        )
        compiled.append(compile_job(compiler, job))
    return tuple(compiled)


def _readout_for_technique(
    acquisitions: Sequence[AcquiredBatch],
    technique: str,
    qubits: Sequence[str],
    config: Config,
) -> ReadoutMap:
    """Return the averaged pre/post calibration for one technique."""
    technique_acquisitions = [
        acquisition
        for acquisition in acquisitions
        if acquisition.plan.technique == technique
    ]
    if not technique_acquisitions:
        return readout_calibration_from_datasets((), qubits)

    calibration_acquisitions = [
        acquisition
        for acquisition in acquisitions
        if acquisition.plan.technique == "readout_calibration"
        and acquisition.plan.metadata.get("characterization_technique") == technique
    ]
    expected_positions = {
        position
        for position, enabled in (
            ("pre", config.pre_calibration),
            ("post", config.post_calibration),
        )
        if enabled
    }
    actual_positions = {
        acquisition.plan.metadata.get("calibration_position")
        for acquisition in calibration_acquisitions
    }
    if (
        actual_positions != expected_positions
        or len(calibration_acquisitions) != len(expected_positions)
    ):
        raise ValueError(
            f"Expected {sorted(expected_positions)} readout calibrations for "
            f"{technique}, got {sorted(str(value) for value in actual_positions)}."
        )
    return readout_calibration_from_datasets(
        [
            (acquisition.dataset, acquisition.plan.measurement_key)
            for acquisition in calibration_acquisitions
        ],
        qubits,
    )


def analyze_experiment(
    acquisitions: Sequence[AcquiredBatch],
    qubits: Sequence[str],
    config: Config,
    output_directory: Path,
) -> None:
    """Persist raw acquisitions, apply readout correction, and write summaries."""
    for acquisition in acquisitions:
        path = output_directory / f"raw_{acquisition.plan.batch_id}.nc"
        persist_dataset(
            acquisition.dataset,
            path,
            acquisition.plan.metadata,
        )

    technique_names = (
        "process_tomography",
        "transition_phase_ramsey",
        "interleaved_randomized_benchmarking",
    )
    readouts = {
        technique: _readout_for_technique(
            acquisitions,
            technique,
            qubits,
            config,
        )
        for technique in technique_names
    }

    for acquisition in acquisitions:
        if acquisition.plan.technique == "readout_calibration":
            continue
        metadata = dict(acquisition.plan.metadata)
        technique = acquisition.plan.technique
        metadata["readout_calibration"] = readout_calibration_metadata(
            readouts[technique]
        )
        metadata["readout_calibration_batches"] = [
            calibration.plan.batch_id
            for calibration in acquisitions
            if calibration.plan.technique == "readout_calibration"
            and calibration.plan.metadata.get("characterization_technique")
            == technique
        ]
        persist_metadata(
            output_directory / f"raw_{acquisition.plan.batch_id}.nc",
            metadata,
        )

    tomography_results = tomography.analyze(
        [
            acquisition
            for acquisition in acquisitions
            if acquisition.plan.technique == "process_tomography"
        ],
        readouts["process_tomography"],
    )
    tomography_results.to_csv(
        output_directory / "composite_tomography_results.csv",
        index=False,
    )

    ramsey_results = ramsey.analyze(
        [
            acquisition
            for acquisition in acquisitions
            if acquisition.plan.technique == "transition_phase_ramsey"
        ],
        readouts["transition_phase_ramsey"],
    )
    ramsey_results.to_csv(
        output_directory / "composite_ramsey_results.csv",
        index=False,
    )

    rb_results = rb.analyze(
        [
            acquisition
            for acquisition in acquisitions
            if acquisition.plan.technique == "interleaved_randomized_benchmarking"
        ],
        readouts["interleaved_randomized_benchmarking"],
    )
    rb_results.to_csv(
        output_directory / "composite_rb_results.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="build, compile, plot, and estimate the experiment without submitting",
    )
    return parser.parse_args()


def run_compiled_experiment(
    pulla: Pulla,
    compiler: Compiler,
    compiled: Sequence[CompiledJob],
    output_directory: Path,
    *,
    prepare_only: bool,
) -> tuple[AcquiredBatch, ...] | None:
    """Write preflight artifacts, then optionally submit compiled work."""
    report = write_preflight(compiled, output_directory)
    job_count = report["job_count"]
    job_label = "job" if job_count == 1 else "jobs"
    print(
        f"Prepared {report['circuit_count']} circuits in {job_count} "
        f"{job_label}; estimated QPU runtime: "
        f"{report['estimated_qpu_runtime_seconds']:.6f} s"
    )
    print(f"Preflight artifacts: {output_directory}")
    if prepare_only:
        return None
    return execute_jobs(pulla, compiler, compiled)


def main(*, prepare_only: bool = False) -> Path:
    registry = built_in_sequences()
    config = Config(qubits=None)
    sequences = [registry['H1'], registry["H21b"]]

    quantum_computer = "garnet"
    pulla = Pulla(quantum_computer=quantum_computer)
    compiler = pulla.get_standard_compiler()
    metric_client = None
    if type(config.qubits) is int:
        try:
            metric_client = IQMClient(quantum_computer=quantum_computer)
        except (IQMError, RequestException, ValueError) as error:
            logger.warning(
                "Could not initialize the IQM metric client; "
                "using graph separation: %s",
                error,
            )
    qubits = select_qubits(
        pulla,
        config.qubits,
        selection_metric=config.qubit_selection_metric,
        metric_client=metric_client,
        prx_implementation=config.prx_implementation,
    )

    # No authorization-dependent submission occurs until every batch has been
    # materialized, compiled, plotted, and included in the runtime estimate.
    compiled = prepare_experiment(compiler, qubits, sequences, config)
    output_directory = Path("results") / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    output_directory.mkdir(parents=True)
    acquisitions = run_compiled_experiment(
        pulla,
        compiler,
        compiled,
        output_directory,
        prepare_only=prepare_only,
    )
    if acquisitions is None:
        return output_directory

    analyze_experiment(acquisitions, qubits, config, output_directory)
    return output_directory


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s;%(levelname).1s] %(message)s",
    )
    logger.setLevel(logging.INFO)
    main(prepare_only=parse_args().prepare_only)
