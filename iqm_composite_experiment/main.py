"""Build, inspect, and optionally run the composite-gate experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from iqm.cpc.compiler.compiler import Compiler
from iqm.pulla.pulla import Pulla
from iqm.pulse.builder import ScheduleBuilder

from characterization import ramsey, rb, tomography
from common import (
    Config,
    build_readout_batch,
    built_in_sequences,
    readout_calibration_from_dataset,
    readout_calibration_metadata,
    select_qubits,
)
from dataset_persistence import persist_dataset, persist_metadata
from execution import (
    AcquiredBatch,
    CompiledBatch,
    PlannedBatch,
    compile_batches,
    execute_batches,
    write_preflight,
)
from sequences import SequenceSpec


def build_experiment(
    builder: ScheduleBuilder,
    qubits: Sequence[str],
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> tuple[PlannedBatch, ...]:
    """Materialize every circuit in the experiment without submitting work."""
    plans: list[PlannedBatch] = []
    readout = build_readout_batch(builder, qubits, config)
    if readout is not None:
        plans.append(readout)
    plans.extend(tomography.build_batches(builder, qubits, sequences, config))
    plans.extend(ramsey.build_batches(builder, qubits, sequences, config))
    plans.extend(rb.build_batches(builder, qubits, sequences, config))
    return tuple(plans)


def prepare_experiment(
    compiler: Compiler,
    qubits: Sequence[str],
    sequences: Sequence[SequenceSpec],
    config: Config,
) -> tuple[CompiledBatch, ...]:
    """Build and compile the full experiment before anything is submitted."""
    builder = compiler.get_schedule_builder()
    return compile_batches(
        compiler,
        build_experiment(builder, qubits, sequences, config),
    )


def analyze_experiment(
    acquisitions: Sequence[AcquiredBatch],
    qubits: Sequence[str],
    config: Config,
    output_directory: Path,
) -> None:
    """Persist raw acquisitions, apply readout correction, and write summaries."""
    readout_acquisition = next(
        (
            acquisition
            for acquisition in acquisitions
            if acquisition.plan.technique == "readout_calibration"
        ),
        None,
    )
    for acquisition in acquisitions:
        path = output_directory / f"raw_{acquisition.plan.batch_id}.nc"
        persist_dataset(
            acquisition.dataset,
            path,
            acquisition.plan.metadata,
        )

    readout = readout_calibration_from_dataset(
        None if readout_acquisition is None else readout_acquisition.dataset,
        qubits,
        config,
    )

    for acquisition in acquisitions:
        if acquisition.plan.technique == "readout_calibration":
            continue
        metadata = dict(acquisition.plan.metadata)
        metadata["readout_calibration"] = readout_calibration_metadata(readout)
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
        readout,
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
        readout,
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
        readout,
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
    compiled: Sequence[CompiledBatch],
    output_directory: Path,
    *,
    prepare_only: bool,
) -> tuple[AcquiredBatch, ...] | None:
    """Write preflight artifacts, then optionally submit compiled work."""
    report = write_preflight(compiled, output_directory)
    print(
        f"Prepared {report['circuit_count']} circuits in {report['batch_count']} "
        f"batches; estimated QPU runtime: "
        f"{report['estimated_qpu_runtime_seconds']:.6f} s"
    )
    print(f"Preflight artifacts: {output_directory}")
    if prepare_only:
        return None
    return execute_batches(pulla, compiler, compiled)


def main(*, prepare_only: bool = False) -> Path:
    registry = built_in_sequences()
    config = Config(qubits=None)
    sequences = [registry["H11a"]]

    pulla = Pulla(quantum_computer="garnet")
    compiler = pulla.get_standard_compiler()
    qubits = select_qubits(pulla, config.qubits)

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
    main(prepare_only=parse_args().prepare_only)
