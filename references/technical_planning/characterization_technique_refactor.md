# Characterization Execution Architecture

The experiment uses a two-phase acquisition model: prepare all work first, then
submit it. Circuit builders and scientific analysis never submit jobs directly.

## Lifecycle

1. `main.build_experiment()` materializes every readout, tomography, Ramsey, and
   randomized-benchmarking `TimeBox`.
2. `execution.compile_batches()` compiles every logical batch and retains its
   compiler context.
3. `execution.write_preflight()` writes playlist visualizations and IQM's QPU
   runtime estimate.
4. `execution.submit_batches()` submits every compiled batch without waiting.
5. `execution.collect_batches()` waits for jobs and pairs each dataset with its
   original manifest.
6. `main.analyze_experiment()` persists raw datasets and runs technique-specific
   analysis.

The first call to `Pulla.submit_playlist()` therefore occurs only after all
circuits have been built, all batches have been compiled, and preflight
artifacts have been written.

## Batch Contracts

`PlannedBatch` owns the ordered circuits, shots, qubits, measurement key,
scientific manifest, and persistence metadata for one acquisition.

`CompiledBatch` keeps the `RunDefinition` and compiler context attached to that
plan. The context must stay paired with the submitted job because IQM result
postprocessing uses it.

`AcquiredBatch` keeps the returned dataset attached to the plan. Tomography
coordinates, Ramsey analysis phases, and the randomized-benchmarking plan are
therefore not reconstructed from implicit loop ordering.

Logical acquisitions remain separate batches. This preserves their shot counts,
measurement keys, result shapes, and raw dataset boundaries. They can be
chunked or merged later only if these mappings remain explicit.

## Readout Calibration

Readout calibration is planned and compiled with all characterization work. Its
dataset is processed first after collection because corrected tomography,
Ramsey, and randomized-benchmarking analysis depends on it. It does not block
construction or submission of later circuits.

## Preflight

`preflight.json` reports circuit counts, batch counts, per-batch estimates, and
total estimated QPU runtime from `SweepDefinition.qpu_runtime`. This estimate
does not include queue or network time.

The `playlists/` directory contains an IQM playlist inspector HTML file for each
batch. `--prepare-only` stops after producing these artifacts and performs no
submission.

## Authorization Failure

Pulse-level access is required because the experiment mutates calibrated IQ
pulses to inject amplitude and detuning errors. The personal-account
`InvalidOperationError` is expected and is not caught, retried, or replaced by
a gate-level fallback. When it occurs during submission, all circuits and
preflight artifacts have already been produced.

## Verification Invariants

- Every compilation finishes before the first submission.
- Every submission occurs before the first job wait.
- Every compiler context remains paired with its run definition and job.
- Raw datasets are persisted before fitting or reconstruction.
- Preflight runtime is the sum of all compiled batch estimates.
- Prepare-only execution performs no submission.
