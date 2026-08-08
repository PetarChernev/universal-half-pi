# Extensible Quantum Operation Characterization Framework

## 1. Purpose

This document specifies the architecture for reusable experiments that evaluate
physical quantum operations. The first concrete operation family is the
universal composite PRX sequence, and the first characterization techniques are
process tomography, transition-phase Ramsey, and interleaved randomized
benchmarking.

The framework is intended for two audiences:

- Physicists assembling experiments interact with immutable Python objects,
  `config.yaml`, NumPy arrays, and pandas DataFrames.
- Developers add operations, techniques, and hardware adapters through explicit
  interfaces without modifying the central runner.

## 2. Design Goals

- Treat the operation under test and characterization technique as independent
  plugins.
- Represent one physical operating condition explicitly.
- Support Cartesian, zipped, one-dimensional, and irregular point collections.
- Keep scientific models independent of IQM and service APIs.
- Validate every target-technique combination before the first acquisition.
- Preserve raw observations alongside corrected values and derived metrics.
- Make optional techniques natural rather than filling mandatory wide tables.
- Support deterministic, entirely service-free unit and integration tests.
- Keep notebook usage concise and discoverable.

## 3. Non-Goals

- The initial IQM adapter does not implement multi-qubit characterization.
- The framework does not define a universal pulse intermediate representation.
- The runner does not currently optimize or merge batches across points.
- The current tomography implementation is linear inversion, not constrained
  maximum-likelihood reconstruction.
- `TransitionPhaseRamsey` is intentionally not presented as a general Ramsey
  protocol.

## 4. Vocabulary

### Operation under test

The scientific object whose physical implementation is evaluated. It defines an
ideal unitary and accepted operating-point parameters, but contains no hardware
client or IQM schedule.

### Experiment point

One operation under test at one set of physical parameters. For example, X5a at
2% amplitude error and 2 MHz detuning is one point.

### Technique coordinate

One circuit internal to a characterization method. Examples are a tomography
input/basis pair, a Ramsey analysis phase, or an RB trial. Technique coordinates
are not experiment points.

### Point set

An ordered immutable collection of experiment points, generated explicitly or
from sweep axes.

### Experiment definition

The selected point set, techniques, readout strategy, qubits, and failure policy.
It is declarative and performs no I/O.

### Backend session

The compiler and executor resources used for one experiment run. A production
IQM session may access services; fake sessions used by tests never do.

## 5. Architectural Overview

```text
config.yaml / notebook
          |
          v
CharacterizationExperiment
  |-- PointSet
  |     `-- ExperimentPoint
  |           |-- OperationUnderTest
  |           `-- ParameterSet
  |-- CharacterizationTechnique[]
  `-- ReadoutStrategy
          |
          v
ExperimentRunner ---> ExperimentBackend ---> BackendSession
       |                                        |
       |                                        v
       |                              OperationCompilerRegistry
       |                                        |
       v                                        v
ExperimentResult                     IQM target compilers
```

Dependency direction points inward. Targets and techniques do not import IQM.
The IQM adapter imports framework interfaces and implements them.

## 6. Operation Plugin Model

`OperationUnderTest` is the backend-neutral target interface. A target provides:

- A stable `operation_id`.
- An `IdealOperation` containing its unitary and logical arity.
- A tuple of `ParameterSpec` declarations.
- Validation and parameter-default resolution.
- Scalar result metadata.

`UniversalCompositePulse` is the supplied concrete implementation. It stores the
ideal operation, constituent angle, and phase vector. Its accepted point
parameters are `amplitude_error` and `detuning_hz`.

The target does not know how IQM realizes these fields. The
`IQMUniversalCompositePulseCompiler` performs that translation using calibrated
PRX operations and explicit IQ-pulse detuning.

This separation creates two independent extension points:

```text
new scientific target -> OperationUnderTest
new target on IQM      -> OperationCompiler[new target]
```

For example, a shaped-pulse target can contain an ideal unitary and envelope
parameters. Its IQM compiler can translate those values into an IQM instruction
without changing tomography, Ramsey, RB, point generation, results, or the
runner.

## 7. Parameter Model

`ParameterSet` is an immutable mapping from names to scalar values. A target's
`ParameterSpec` entries define type, default, unit, description, and optional
bounds. Unknown parameters are rejected to catch misspelled configuration.

Parameters belong to the operation implementation, not to global experiment
configuration. This avoids changing unrelated classes when a new pulse strategy
introduces `drag_beta`, `duration`, `alpha`, or `epsilon`.

## 8. Point And Sweep Model

`ExperimentPoint` contains `point_id`, `target`, `parameters`, and descriptive
metadata. It validates itself against the target on construction.

`PointSet` provides these constructors:

- `explicit` for arbitrary points.
- `cartesian` for a full product of axes.
- `zipped` for correlated axis values.
- `concatenate` for combining independently generated collections.

A one-dimensional sweep is a Cartesian sweep with one axis. Constants are
placed in the `fixed` parameter mapping. Point order and generated IDs are
deterministic.

## 9. Technique Plugin Model

Every technique implements `CharacterizationTechnique`:

```python
class CharacterizationTechnique(ABC):
    technique_id: str

    def supports(self, target: OperationUnderTest) -> bool: ...
    def validate(self, point, context) -> None: ...
    def build_plan(self, point, context) -> AcquisitionPlan: ...
    def analyze(self, point, plan, measurements) -> tuple[TechniqueResult, ...]: ...
```

`build_plan` creates deferred circuits with explicit coordinates. `analyze`
receives backend-normalized probabilities and produces scalar metrics plus array
artifacts. No technique submits hardware jobs or decodes IQM datasets.

### ProcessTomography

- Supports arbitrary one-qubit ideal unitaries.
- Acquires four input states in three measurement bases.
- Reconstructs an affine Pauli-transfer matrix.
- Computes fidelity against the complete ideal unitary.
- Retains PTM, Bloch matrix, translation, and measured probabilities.

### TransitionPhaseRamsey

- Supports ideal +X π and +X π/2 rotations.
- Makes the manuscript-specific preparation and normalization assumptions
  explicit.
- Fits a linear sinusoidal model.
- Retains phases, ideal fringe, fitted fringe, and measured probabilities.

### InterleavedRandomizedBenchmarking

- Supports one-qubit targets whose ideal unitary is a Clifford.
- Uses typed RB trial coordinates rather than unvalidated dictionaries.
- Can pair random Clifford sequences across points or derive per-point seeds.
- Retains survival curves and fitted decay metrics.

## 10. Deferred Circuit Construction

IQM Pulla 14 creates `ScheduleBuilder` during compilation and supplies it to a
callable whose parameter is named `builder`. Therefore `AcquisitionPlan` stores a
recipe for each circuit rather than an already-built `TimeBox`.

At execution time the IQM backend creates `IQMCircuitFactory` inside that
callback and invokes each recipe. Fake backends invoke the same recipes with a
local matrix-based or symbolic factory.

This is both an IQM compatibility requirement and the principal service-free
integration-test seam.

## 11. IQM Adapter

`IQMBackend` is constructed around an already initialized `Pulla` object but
does not access it until `open_session` is called. The session owns topology,
compiler, selected loci, compilation, submission, result retrieval, and dataset
decoding.

The adapter targets the installed Pulla/Pulse 14.x APIs:

- The standard compiler receives a deferred `timeboxes(builder)` callback.
- Explicit PRX selection uses `impl_name`.
- Detuning is converted from hertz to fractions of drive-channel sample rate.
- Detuning rejects non-atomic or multi-IQ-pulse PRX implementations.
- Missing or ambiguous dataset variables fail explicitly.
- Circuit ordering is obtained from the `circuit_index` dimension.

`ProbabilitySource.EXA_PROBABILITY` consumes IQM post-processed probabilities.
These may already be assignment-corrected. `THRESHOLDED_READOUT` averages binary
single-shot readout and must be used with local independent readout correction.

## 12. Readout Lifecycle

Readout correction is an independent strategy rather than a technique setting.

- `NoReadoutCorrection` uses backend values unchanged.
- `IndependentReadoutCorrection` acquires parallel |0> and |1> circuits once per
  run and applies one binary confusion model per locus.

Readout has its own shot setting. Raw and corrected probabilities remain
available to each technique. The YAML loader rejects local correction paired
with EXA probabilities to prevent accidental double correction.

## 13. Experiment Execution

`CharacterizationExperiment` is a pure definition. `ExperimentRunner` performs this
lifecycle:

1. Open one backend session for the selected qubits.
2. Validate every target, backend compiler, and technique combination.
3. Prepare the configured readout strategy.
4. Iterate through points in deterministic order.
5. Build and execute each selected technique plan.
6. Apply readout correction.
7. Analyze observations into structured results.
8. Abort or record failures according to `FailurePolicy`.
9. Close the session and return `ExperimentResult`.

There are no technique-specific branches in the runner.

## 14. Result Model

The canonical hierarchy is:

```text
ExperimentResult
  `-- PointResult[]
        `-- TechniqueResult[]
```

`TechniqueResult` identifies point, target, technique, and physical locus. It
separates scalar metrics from NumPy artifacts and can represent success, failure,
or skipped work.

Notebook views are available through:

- `metrics_dataframe`: one row per point, technique, and locus.
- `long_dataframe`: one row per scalar metric.
- `legacy_dataframe`: one merged row per point and locus.

The structured result remains authoritative because a DataFrame cannot naturally
represent matrices, fit traces, or optional techniques.

## 15. YAML Composition

`config.yaml` contains all settings used by `run_experiment.py`:

- IQM URL, computer, qubits, PRX implementation, and probability source.
- Selected target IDs.
- Sweep kind, axes, and fixed parameters.
- Technique selection and technique-owned settings.
- Readout strategy settings.
- Failure policy and output view/path.

There is no CLI or argument parser. `run_experiment.py` loads the adjacent YAML,
constructs Pulla only after confirming that a URL is configured, runs the
experiment, and writes the selected table.

The configuration loader accepts custom target and technique registries. Plugin
projects can therefore extend YAML composition without editing framework code.

## 16. Extending The Framework

### Adding an operation under test

1. Implement `OperationUnderTest` with an ideal operation and parameter specs.
2. Implement an IQM `OperationCompiler` for that target type.
3. Register the compiler with `OperationCompilerRegistry`.
4. Add target instances to the notebook or YAML target registry.
5. Test parameter validation and compiled operation behavior using fakes or local
   IQM value objects.

### Adding a characterization technique

1. Implement `CharacterizationTechnique`.
2. Keep technique-specific settings beside the implementation.
3. Represent every circuit's coordinates explicitly.
4. Keep numerical analysis in pure functions where possible.
5. Add a YAML factory only if the technique should be configurable by the plain
   script.

### Adding a backend

1. Implement `ExperimentBackend` and `BackendSession`.
2. Provide a circuit factory matching the framework protocol.
3. Register target compilers supported by that backend.
4. Normalize observations into `MeasurementData`.

## 17. Testing And Service Safety

The test suite uses Python's standard `unittest` package because the qti
environment does not include pytest. Tests are divided into:

- Unit tests for parameters, sweeps, target validation, Clifford generation,
  fitting, tomography reconstruction, Ramsey analysis, decoder behavior, and
  readout inversion.
- Integration tests that run real technique classes through `ExperimentRunner`
  against an in-memory backend and fake circuit factory.
- Configuration tests that load temporary YAML files without creating Pulla.

Tests must never instantiate real `Pulla`, call `open_session` on `IQMBackend`,
submit a playlist, or access an IQM URL. Production service access exists only in
the explicitly executed `run_experiment.py` composition root.

## 18. Compatibility And Current Constraints

`legacy.py` retains the original `Config`, `SequenceSpec`, `built_in_sequences`,
and `run_experiment` entry points by translating them into the new model.

Current built-in techniques and the IQM adapter operate on one-qubit loci. The
core result and target models use `Locus` and operation arity so multi-qubit
extensions can be added without changing stored result identity.

The qti environment currently provides `iqm-pulla 14.0.0` and `iqm-pulse 14.0.0`.
The package is constrained to 14.x, and the concrete adapter must be revalidated
against IQM API changes before upgrading production environments.
